#include "Intent/Target/Triton/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>

using namespace mlir;

namespace intent::triton {
namespace {

constexpr llvm::StringLiteral legalizedAttr = "intent_gpu.triton.legalized";
constexpr llvm::StringLiteral reduceFormAttr = "intent_gpu.triton.reduce_form";
constexpr llvm::StringLiteral contractFormAttr =
    "intent_gpu.triton.contract_form";
constexpr llvm::StringLiteral tensorDescriptorChoice =
    "USE_TENSOR_DESCRIPTOR";
constexpr llvm::StringLiteral tensorDescriptorEligibility =
    "TENSOR_DESCRIPTOR_ELIGIBLE";
constexpr int64_t maxTritonTensorElements = 1048576;

struct TritonConfig {
  std::map<std::string, int64_t> kernelParameters;
  int64_t warps = 0;
  int64_t stages = 0;
  int64_t ctas = 0;
};

struct TritonLocalOptions {
  int64_t warps;
  int64_t stages;
  int64_t ctas;
};

StringRef localOptionsFamily(ArrayRef<gpu::ParameterCategory> categories,
                             bool twoAxisPointwise,
                             bool blackwellRecurrentContraction,
                             bool fp32Contractions) {
  if (llvm::is_contained(categories,
                         gpu::ParameterCategory::RegionReduction))
    return "region_reduction";
  if (llvm::is_contained(categories,
                         gpu::ParameterCategory::RegionContraction))
    return "region_contraction";
  if (llvm::is_contained(categories,
                         gpu::ParameterCategory::PersistentContraction)) {
    if (blackwellRecurrentContraction)
      return "blackwell_recurrent_contraction";
    return "persistent_contraction";
  }
  if (llvm::is_contained(categories, gpu::ParameterCategory::Contraction))
    return fp32Contractions ? "contraction_f32" : "contraction";
  if (llvm::is_contained(categories, gpu::ParameterCategory::Histogram))
    return "histogram";
  if (llvm::is_contained(categories, gpu::ParameterCategory::Reduction) ||
      llvm::is_contained(categories, gpu::ParameterCategory::Scan))
    return "reduction_scan";
  if (twoAxisPointwise)
    return "pointwise_two_axis";
  return "pointwise";
}

bool valueDependsOn(Value value, Value root, scf::ForOp owner,
                    llvm::SmallDenseSet<Value, 32> &visited) {
  if (value == root)
    return true;
  if (!visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition || !owner->isProperAncestor(definition))
    return false;
  if (auto nested = dyn_cast<scf::ForOp>(definition)) {
    auto result = dyn_cast<OpResult>(value);
    auto yield = dyn_cast<scf::YieldOp>(nested.getBody()->getTerminator());
    if (result && yield && result.getResultNumber() < nested.getInitArgs().size()) {
      unsigned index = result.getResultNumber();
      if (valueDependsOn(nested.getInitArgs()[index], root, owner, visited) ||
          valueDependsOn(yield.getOperand(index), root, owner, visited))
        return true;
    }
  }
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueDependsOn(operand, root, owner, visited);
  });
}

bool valueDependsOnNestedContract(Value value, scf::ForOp owner,
                                  llvm::SmallDenseSet<Value, 32> &visited) {
  if (!visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition || !owner->isProperAncestor(definition))
    return false;
  if (isa<gpu::ContractOp>(definition))
    return true;
  if (auto nested = dyn_cast<scf::ForOp>(definition)) {
    auto result = dyn_cast<OpResult>(value);
    auto yield = dyn_cast<scf::YieldOp>(nested.getBody()->getTerminator());
    if (result && yield && result.getResultNumber() < nested.getInitArgs().size()) {
      unsigned index = result.getResultNumber();
      if (valueDependsOnNestedContract(nested.getInitArgs()[index], owner,
                                       visited) ||
          valueDependsOnNestedContract(yield.getOperand(index), owner, visited))
        return true;
    }
  }
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueDependsOnNestedContract(operand, owner, visited);
  });
}

bool isDirectContractionAccumulator(Value value,
                                    llvm::SmallDenseSet<Value, 8> &visited) {
  if (!visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (isa_and_nonnull<gpu::ContractOp>(definition))
    return true;
  auto nested = dyn_cast_or_null<scf::ForOp>(definition);
  auto result = dyn_cast<OpResult>(value);
  auto yield = nested
                   ? dyn_cast<scf::YieldOp>(nested.getBody()->getTerminator())
                   : scf::YieldOp();
  return nested && result && yield &&
         result.getResultNumber() < nested.getInitArgs().size() &&
         isDirectContractionAccumulator(
             yield.getOperand(result.getResultNumber()), visited);
}

bool hasRecurrentContraction(func::FuncOp kernel) {
  bool found = false;
  kernel.walk([&](scf::ForOp loop) {
    if (found || loop.getInitArgs().empty())
      return;
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != loop.getRegionIterArgs().size())
      return;
    for (auto [index, next] : llvm::enumerate(yield.getOperands())) {
      llvm::SmallDenseSet<Value, 8> directVisited;
      if (isDirectContractionAccumulator(next, directVisited))
        continue;
      llvm::SmallDenseSet<Value, 32> contractionVisited;
      if (!valueDependsOnNestedContract(next, loop, contractionVisited))
        continue;
      llvm::SmallDenseSet<Value, 32> carryVisited;
      if (valueDependsOn(next, loop.getRegionIterArgs()[index], loop,
                         carryVisited)) {
        found = true;
        return;
      }
    }
  });
  return found;
}

std::optional<int64_t>
evaluateCompileTimeExpression(gpu::PhysicalExprAttr expression,
                              const TritonConfig &config) {
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  if (kind == gpu::PhysicalExprKind::Constant)
    return expression.getValue();
  if (kind == gpu::PhysicalExprKind::Parameter) {
    auto found = config.kernelParameters.find(expression.getSymbol().getValue().str());
    return found == config.kernelParameters.end()
               ? std::nullopt
               : std::optional<int64_t>(found->second);
  }
  if (kind == gpu::PhysicalExprKind::Dimension ||
      kind == gpu::PhysicalExprKind::ScalarABI)
    return std::nullopt;

  SmallVector<int64_t> operands;
  for (Attribute operand : expression.getOperands()) {
    std::optional<int64_t> value = evaluateCompileTimeExpression(
        cast<gpu::PhysicalExprAttr>(operand), config);
    if (!value)
      return std::nullopt;
    operands.push_back(*value);
  }
  auto checked = [](const __int128 value) -> std::optional<int64_t> {
    if (value < std::numeric_limits<int64_t>::min() ||
        value > std::numeric_limits<int64_t>::max())
      return std::nullopt;
    return static_cast<int64_t>(value);
  };
  if (kind == gpu::PhysicalExprKind::Add)
    return checked(static_cast<__int128>(operands[0]) + operands[1]);
  if (kind == gpu::PhysicalExprKind::Subtract)
    return checked(static_cast<__int128>(operands[0]) - operands[1]);
  if (kind == gpu::PhysicalExprKind::Multiply)
    return checked(static_cast<__int128>(operands[0]) * operands[1]);
  if (kind == gpu::PhysicalExprKind::Minimum)
    return std::min(operands[0], operands[1]);
  if (kind == gpu::PhysicalExprKind::Maximum)
    return std::max(operands[0], operands[1]);
  if (kind == gpu::PhysicalExprKind::Select)
    return operands[0] ? operands[1] : operands[2];
  if (kind == gpu::PhysicalExprKind::FloorDiv) {
    if (operands[1] <= 0 || operands[0] < 0)
      return std::nullopt;
    return operands[0] / operands[1];
  }
  if (kind == gpu::PhysicalExprKind::CeilDiv) {
    if (operands[1] <= 0 || operands[0] < 0)
      return std::nullopt;
    return checked((static_cast<__int128>(operands[0]) + operands[1] - 1) /
                   operands[1]);
  }
  if (kind == gpu::PhysicalExprKind::NextPowerOfTwo) {
    if (operands[0] <= 0)
      return std::nullopt;
    uint64_t value = static_cast<uint64_t>(operands[0] - 1);
    for (unsigned shift = 1; shift < 64; shift <<= 1)
      value |= value >> shift;
    if (value == std::numeric_limits<uint64_t>::max() ||
        value + 1 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return std::nullopt;
    return static_cast<int64_t>(value + 1);
  }
  return std::nullopt;
}

bool isCompileTimeExpression(gpu::PhysicalExprAttr expression) {
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  if (kind == gpu::PhysicalExprKind::Constant ||
      kind == gpu::PhysicalExprKind::Parameter)
    return true;
  if (kind == gpu::PhysicalExprKind::Dimension ||
      kind == gpu::PhysicalExprKind::ScalarABI)
    return false;
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isCompileTimeExpression(cast<gpu::PhysicalExprAttr>(operand));
  });
}

struct BlockAccessPlan {
  SmallVector<Value> offsets;
  SmallVector<int64_t> blockAxes;
  SmallVector<int64_t> order;
  SmallVector<int64_t> boundaryAxes;
};

bool isUnitStep(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
  return constant && constant.value() == 1;
}

Value stripShapeOnly(Value value) {
  while (true) {
    if (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>()) {
      value = broadcast.getValue();
      continue;
    }
    if (auto reshape = value.getDefiningOp<gpu::ReshapeOp>()) {
      value = reshape.getValue();
      continue;
    }
    if (auto splat = value.getDefiningOp<gpu::SplatOp>()) {
      value = splat.getValue();
      continue;
    }
    return value;
  }
}

bool isTrueValue(Value value) {
  auto constant = stripShapeOnly(value).getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  return integer && integer.getType().isInteger(1) && integer.getValue().isOne();
}

bool isZeroValue(Value value) {
  value = stripShapeOnly(value);
  while (auto cast = value.getDefiningOp<gpu::CastOp>())
    value = cast.getValue();
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
    return floating.getValue().isZero();
  return false;
}

std::optional<BlockAccessPlan>
planBlockAccess(Value resource, ValueRange coordinates,
                ArrayRef<int64_t> sourceAxes, Type valueType, Value valid,
                Value fill,
                const gpu::PhysicalAccessBoundaryFact &boundaryFact) {
  auto view = dyn_cast<gpu::ViewType>(resource.getType());
  auto fragment = dyn_cast<gpu::FragmentType>(valueType);
  if (!view || !fragment || fragment.getShape().empty() || !isa<BlockArgument>(resource) ||
      coordinates.size() != view.getRank() ||
      sourceAxes.size() != view.getRank() ||
      !view.getLayout().getHasStrides() ||
      view.getLayout().getStrides().size() != view.getRank())
    return std::nullopt;

  BlockAccessPlan plan;
  plan.offsets.resize(view.getRank());
  plan.blockAxes.assign(fragment.getShape().size(), -1);
  llvm::SmallBitVector seenViewAxes(view.getRank());
  llvm::SmallBitVector seenFragmentAxes(fragment.getShape().size());
  for (auto [coordinateIndex, coordinate] : llvm::enumerate(coordinates)) {
    int64_t viewAxis = sourceAxes[coordinateIndex];
    if (viewAxis < 0 || viewAxis >= static_cast<int64_t>(view.getRank()) ||
        seenViewAxes.test(viewAxis))
      return std::nullopt;
    seenViewAxes.set(viewAxis);
    if (coordinate.getType().isIndex()) {
      plan.offsets[viewAxis] = coordinate;
      continue;
    }
    auto coordinateType = dyn_cast<gpu::FragmentType>(coordinate.getType());
    auto range = coordinate.getDefiningOp<gpu::MakeRangeOp>();
    if (!coordinateType || coordinateType.getShape().size() != 1 ||
        !coordinateType.getElementType().isIndex() || !range ||
        !isUnitStep(range.getStep()))
      return std::nullopt;
    auto coordinateMap =
        cast<gpu::AxisMapAttr>(coordinateType.getAxisMaps()[0]);
    std::optional<unsigned> selected;
    for (auto [fragmentAxis, mapping] :
         llvm::enumerate(fragment.getAxisMaps())) {
      auto resultMap = cast<gpu::AxisMapAttr>(mapping);
      if (resultMap.getSourceId() == coordinateMap.getSourceId() &&
          resultMap.getSourceAxis() == coordinateMap.getSourceAxis() &&
          resultMap.getDimensionId() == coordinateMap.getDimensionId() &&
          resultMap.getDerived() == coordinateMap.getDerived()) {
        if (selected)
          return std::nullopt;
        selected = fragmentAxis;
      }
    }
    if (!selected || seenFragmentAxes.test(*selected) ||
        coordinateType.getShape()[0] != fragment.getShape()[*selected])
      return std::nullopt;
    auto extent = cast<gpu::PhysicalExprAttr>(fragment.getShape()[*selected]);
    if (!isCompileTimeExpression(extent))
      return std::nullopt;
    seenFragmentAxes.set(*selected);
    plan.blockAxes[*selected] = viewAxis;
    plan.offsets[viewAxis] = range.getStart();
  }
  if (seenViewAxes.count() != view.getRank() ||
      seenFragmentAxes.count() != fragment.getShape().size() ||
      llvm::any_of(plan.offsets, [](Value value) { return !value; }))
    return std::nullopt;
  for (Attribute stride : view.getLayout().getStrides())
    if (!isa<IntegerAttr, StringAttr>(stride))
      return std::nullopt;

  if (!boundaryFact.isExact())
    return std::nullopt;
  if (valid) {
    if (fill && !isZeroValue(fill))
      return std::nullopt;
  } else if (fill) {
    return std::nullopt;
  }
  for (int64_t viewAxis : boundaryFact.boundaryAxes) {
    auto found = llvm::find(plan.blockAxes, viewAxis);
    if (found == plan.blockAxes.end())
      return std::nullopt;
    plan.boundaryAxes.push_back(
        static_cast<int64_t>(std::distance(plan.blockAxes.begin(), found)));
  }
  llvm::sort(plan.boundaryAxes);
  for (int64_t axis = 0;
       axis < static_cast<int64_t>(fragment.getShape().size()); ++axis)
    plan.order.push_back(axis);
  llvm::sort(plan.order, [&](int64_t lhs, int64_t rhs) {
    return plan.blockAxes[lhs] > plan.blockAxes[rhs];
  });
  return plan;
}

LogicalResult materializeBlockPointerForms(func::FuncOp kernel) {
  SmallVector<std::pair<gpu::LoadOp, BlockAccessPlan>, 4> loads;
  SmallVector<std::pair<gpu::StoreOp, BlockAccessPlan>, 4> stores;
  gpu::PhysicalProgramAnalysis analysis(kernel);
  kernel.walk([&](gpu::LoadOp load) {
    gpu::PhysicalAccessBoundaryFact boundary =
        analysis.boundaryValidity(load);
    if (std::optional<BlockAccessPlan> plan = planBlockAccess(
            load.getResource(), load.getCoordinates(), load.getSourceAxes(),
            load.getResult().getType(), load.getValid(), load.getFill(),
            boundary))
      loads.emplace_back(load, std::move(*plan));
  });
  kernel.walk([&](gpu::StoreOp store) {
    gpu::PhysicalAccessBoundaryFact boundary =
        analysis.boundaryValidity(store);
    if (std::optional<BlockAccessPlan> plan = planBlockAccess(
            store.getResource(), store.getCoordinates(), store.getSourceAxes(),
            store.getValue().getType(), store.getValid(), Value(), boundary))
      stores.emplace_back(store, std::move(*plan));
  });
  if (loads.empty() && stores.empty())
    return success();

  for (auto &[load, plan] : loads) {
    OpBuilder builder(load);
    auto block = builder.create<BlockLoadOp>(
        load.getLoc(), load.getResult().getType(), load.getResource(),
        plan.offsets,
        DenseI64ArrayAttr::get(kernel.getContext(), plan.blockAxes),
        DenseI64ArrayAttr::get(kernel.getContext(), plan.order),
        DenseI64ArrayAttr::get(kernel.getContext(), plan.boundaryAxes),
        builder.getStringAttr("zero"));
    if (Attribute origin = load->getAttr(gpu::originAttr))
      block->setAttr(gpu::originAttr, origin);
    load.getResult().replaceAllUsesWith(block.getResult());
    load.erase();
  }
  for (auto &[store, plan] : stores) {
    OpBuilder builder(store);
    auto block = builder.create<BlockStoreOp>(
        store.getLoc(), store.getResource(), plan.offsets, store.getValue(),
        DenseI64ArrayAttr::get(kernel.getContext(), plan.blockAxes),
        DenseI64ArrayAttr::get(kernel.getContext(), plan.order),
        DenseI64ArrayAttr::get(kernel.getContext(), plan.boundaryAxes));
    if (Attribute origin = store->getAttr(gpu::originAttr))
      block->setAttr(gpu::originAttr, origin);
    store.erase();
  }
  return success();
}

std::optional<int64_t> descriptorElementBytes(Type type) {
  unsigned bitWidth = type.getIntOrFloatBitWidth();
  if (bitWidth < 8 || bitWidth % 8 != 0)
    return std::nullopt;
  return bitWidth / 8;
}

bool descriptorStrideAvailable(func::FuncOp kernel, gpu::ViewType view,
                               unsigned axis) {
  Attribute stride = view.getLayout().getStrides()[axis];
  if (auto constant = dyn_cast<IntegerAttr>(stride))
    return constant.getInt() > 0;
  auto symbol = dyn_cast<StringAttr>(stride);
  if (!symbol)
    return false;
  unsigned matches = 0;
  for (auto [index, argument] : llvm::enumerate(kernel.getArguments())) {
    auto name = kernel.getArgAttrOfType<StringAttr>(index, gpu::abiNameAttr);
    auto kind = kernel.getArgAttrOfType<StringAttr>(index, gpu::abiKindAttr);
    auto source =
        kernel.getArgAttrOfType<IntegerAttr>(index, gpu::sourceABIAttr);
    auto sourceAxis =
        kernel.getArgAttrOfType<IntegerAttr>(index, gpu::sourceAxisAttr);
    if (name == symbol && kind && kind.getValue() == "stride" && source &&
        source.getInt() == view.getAbiIndex() && sourceAxis &&
        sourceAxis.getInt() == axis && argument.getType().isIndex())
      ++matches;
  }
  return matches == 1;
}

bool descriptorAccessEligible(func::FuncOp kernel, Value viewValue,
                              ValueRange offsets,
                              ArrayRef<int64_t> blockAxes,
                              gpu::FragmentType fragment) {
  auto view = cast<gpu::ViewType>(viewValue.getType());
  unsigned blockRank = fragment.getShape().size();
  if (view.getRank() < 2 || view.getRank() > 5 || blockRank != 2 ||
      blockAxes != ArrayRef<int64_t>{
                       static_cast<int64_t>(view.getRank()) - 2,
                       static_cast<int64_t>(view.getRank()) - 1})
    return false;
  std::optional<int64_t> elementBytes =
      descriptorElementBytes(view.getElementType());
  auto layout = view.getLayout();
  if (!elementBytes || !layout.getHasStrides() ||
      layout.getStrides().size() != view.getRank())
    return false;
  auto strides = layout.getStrides();
  for (unsigned axis = 0; axis < view.getRank(); ++axis)
    if (!descriptorStrideAvailable(kernel, view, axis))
      return false;
  if (auto last = dyn_cast<IntegerAttr>(strides[strides.size() - 1]);
      last && last.getInt() != 1)
    return false;
  // The final two axes become descriptor rows and columns.  A padded row
  // stride is directly representable, while earlier source axes must flatten
  // contiguously into that row coordinate.
  for (unsigned axis = 0; axis + 2 < view.getRank(); ++axis) {
    auto stride = dyn_cast<IntegerAttr>(strides[axis]);
    auto nextStride = dyn_cast<IntegerAttr>(strides[axis + 1]);
    auto nextExtent =
        dyn_cast<gpu::PhysicalExprAttr>(layout.getExtents()[axis + 1]);
    if (!stride || !nextStride || !nextExtent ||
        nextExtent.getKind() !=
            static_cast<uint32_t>(gpu::PhysicalExprKind::Constant))
      continue;
    __int128 expected = static_cast<__int128>(nextStride.getInt()) *
                        nextExtent.getValue();
    if (expected != stride.getInt())
      return false;
  }
  auto rowStride = dyn_cast<IntegerAttr>(strides[view.getRank() - 2]);
  if (rowStride && (rowStride.getInt() * *elementBytes) % 16 != 0)
    return false;
  auto lastOffset =
      offsets[blockAxes.back()].getDefiningOp<arith::ConstantIndexOp>();
  return lastOffset && (lastOffset.value() * *elementBytes) % 16 == 0;
}

void copyOrigin(Operation *source, Operation *target) {
  if (Attribute origin = source->getAttr(gpu::originAttr))
    target->setAttr(gpu::originAttr, origin);
}

FailureOr<Value> descriptorStrideValue(OpBuilder &builder, func::FuncOp kernel,
                                       Location location, gpu::ViewType view,
                                       unsigned axis) {
  Attribute stride = view.getLayout().getStrides()[axis];
  if (auto constant = dyn_cast<IntegerAttr>(stride))
    return Value(builder.create<arith::ConstantIndexOp>(location,
                                                        constant.getInt()));
  auto symbol = dyn_cast<StringAttr>(stride);
  if (!symbol)
    return failure();
  for (auto [index, argument] : llvm::enumerate(kernel.getArguments())) {
    auto name = kernel.getArgAttrOfType<StringAttr>(index, gpu::abiNameAttr);
    auto kind = kernel.getArgAttrOfType<StringAttr>(index, gpu::abiKindAttr);
    auto source =
        kernel.getArgAttrOfType<IntegerAttr>(index, gpu::sourceABIAttr);
    auto sourceAxis =
        kernel.getArgAttrOfType<IntegerAttr>(index, gpu::sourceAxisAttr);
    if (name == symbol && kind && kind.getValue() == "stride" && source &&
        source.getInt() == view.getAbiIndex() && sourceAxis &&
        sourceAxis.getInt() == axis && argument.getType().isIndex())
      return argument;
  }
  return failure();
}

FailureOr<SmallVector<Value>> materializeDescriptorOffsets(
    OpBuilder &builder, func::FuncOp kernel, Location location, Value viewValue,
    ValueRange offsets) {
  auto view = cast<gpu::ViewType>(viewValue.getType());
  Value rowElements;
  for (unsigned axis = 0; axis + 1 < view.getRank(); ++axis) {
    FailureOr<Value> stride =
        descriptorStrideValue(builder, kernel, location, view, axis);
    if (failed(stride))
      return failure();
    Value term = builder.create<gpu::BinaryOp>(
        location, builder.getIndexType(), offsets[axis], *stride,
        BinaryOperator::Multiply);
    rowElements = rowElements
                      ? Value(builder.create<gpu::BinaryOp>(
                            location, builder.getIndexType(), rowElements, term,
                            BinaryOperator::Add))
                      : term;
  }
  FailureOr<Value> rowStride = descriptorStrideValue(
      builder, kernel, location, view, view.getRank() - 2);
  if (!rowElements || failed(rowStride))
    return failure();
  Value row = builder.create<gpu::BinaryOp>(
      location, builder.getIndexType(), rowElements, *rowStride,
      BinaryOperator::FloorDivide);
  return SmallVector<Value>{row, offsets.back()};
}

FailureOr<TensorDescriptorChoiceOp>
materializeTensorDescriptorForms(
    func::FuncOp kernel, ArrayRef<TritonLocalOptions> localOptions) {
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities || capabilities.getComputeCapabilityMajor() < 9)
    return TensorDescriptorChoiceOp();
  bool hasContraction = false;
  kernel.walk([&](Operation *operation) {
    hasContraction |= isa<gpu::ContractOp, gpu::ScaledContractOp>(operation);
  });
  if (!hasContraction)
    return TensorDescriptorChoiceOp();

  SmallVector<BlockLoadOp> loads;
  SmallVector<BlockStoreOp> stores;
  kernel.walk([&](BlockLoadOp load) {
    if (descriptorAccessEligible(kernel, load.getView(), load.getOffsets(),
                                 load.getBlockAxes(), load.getResult().getType()))
      loads.push_back(load);
  });
  kernel.walk([&](BlockStoreOp store) {
    if (descriptorAccessEligible(kernel, store.getView(), store.getOffsets(),
                                 store.getBlockAxes(), store.getValue().getType()))
      stores.push_back(store);
  });
  if (loads.empty() && stores.empty())
    return TensorDescriptorChoiceOp();

  OpBuilder entry(&kernel.getBody().front(), kernel.getBody().front().begin());
  entry.create<TensorDescriptorAllocatorOp>(
      kernel.getLoc(), entry.getI64IntegerAttr(0), entry.getI64IntegerAttr(1),
      entry.getI64IntegerAttr(2), entry.getStringAttr("launch"),
      entry.getStringAttr("torch_cuda_current_device"));
  struct DescriptorPlan {
    Value view;
    gpu::FragmentType fragment;
    SmallVector<int64_t> blockAxes;
    TensorDescriptorOp descriptor;
  };
  SmallVector<DescriptorPlan> descriptors;
  auto descriptorFor = [&](Value viewValue, gpu::FragmentType fragment,
                           ArrayRef<int64_t> blockAxes)
      -> FailureOr<TensorDescriptorOp> {
    for (const DescriptorPlan &plan : descriptors)
      if (plan.view == viewValue && plan.fragment == fragment &&
          ArrayRef<int64_t>(plan.blockAxes) == blockAxes)
        return plan.descriptor;
    auto view = cast<gpu::ViewType>(viewValue.getType());
    SmallVector<Value> dimensions;
    for (unsigned axis = 0; axis < view.getRank(); ++axis)
      dimensions.push_back(entry.create<gpu::DimOp>(
          kernel.getLoc(), entry.getIndexType(), viewValue, axis));
    Value rows = dimensions.front();
    for (unsigned axis = 1; axis + 1 < dimensions.size(); ++axis)
      rows = entry.create<gpu::BinaryOp>(kernel.getLoc(), entry.getIndexType(),
                                        rows, dimensions[axis],
                                        BinaryOperator::Multiply);
    Value one = entry.create<arith::ConstantIndexOp>(kernel.getLoc(), 1);
    SmallVector<Value> shape{rows, dimensions.back()};
    FailureOr<Value> rowStride = descriptorStrideValue(
        entry, kernel, kernel.getLoc(), view, view.getRank() - 2);
    if (failed(rowStride))
      return failure();
    SmallVector<Value> strides{*rowStride, one};
    SmallVector<Value> descriptorBlockShape;
    for (Attribute extent : fragment.getShape())
      descriptorBlockShape.push_back(entry.create<gpu::PhysicalExprOp>(
          kernel.getLoc(), entry.getIndexType(),
          cast<gpu::PhysicalExprAttr>(extent)));
    std::optional<int64_t> elementBytes =
        descriptorElementBytes(view.getElementType());
    if (!elementBytes || 16 % *elementBytes != 0)
      return failure();
    int64_t maximumBlockElements = maxTritonTensorElements;
    if (llvm::all_of(localOptions, [](const TritonLocalOptions &options) {
          return options.ctas == 1;
        })) {
      // Single-CTA TMA materializes the entire descriptor block in shared
      // memory. Other buffers and barriers remain the provider's responsibility.
      maximumBlockElements = std::min(
          maximumBlockElements,
          capabilities.getMaxDynamicSharedMemoryPerBlock() / *elementBytes);
    }
    SmallVector<int64_t> flattenedContiguousAxes;
    for (unsigned axis = 0; axis + 2 < view.getRank(); ++axis)
      flattenedContiguousAxes.push_back(axis);
    auto descriptor = entry.create<TensorDescriptorOp>(
        kernel.getLoc(), viewValue.getType(), viewValue, shape, strides,
        descriptorBlockShape, blockAxes,
        ArrayRef<int64_t>{1, 16 / *elementBytes}, "flattened_row_major",
        "zero", flattenedContiguousAxes,
        ArrayRef<int64_t>{static_cast<int64_t>(view.getRank()) - 2},
        ArrayRef<int64_t>{static_cast<int64_t>(view.getRank()) - 1},
        /*requirePositiveShape=*/true,
        /*requirePositiveStrides=*/true,
        /*requirePowerOfTwoBlockShape=*/true, /*alignment=*/16,
        /*minimumContiguousBytes=*/16,
        /*maximumShapeExtent=*/std::numeric_limits<int32_t>::max(),
        maximumBlockElements);
    descriptors.push_back(
        {viewValue, fragment,
         SmallVector<int64_t>(blockAxes.begin(), blockAxes.end()), descriptor});
    return descriptor;
  };
  for (BlockLoadOp load : loads)
    if (failed(descriptorFor(load.getView(), load.getResult().getType(),
                             load.getBlockAxes())))
      return load.emitOpError(
          "could not declare its tensor-descriptor runtime contract");
  for (BlockStoreOp store : stores)
    if (failed(descriptorFor(store.getView(), store.getValue().getType(),
                             store.getBlockAxes())))
      return store.emitOpError(
          "could not declare its tensor-descriptor runtime contract");
  SmallVector<Value> descriptorValues;
  for (DescriptorPlan &plan : descriptors)
    descriptorValues.push_back(plan.descriptor.getResult());
  auto choice = entry.create<TensorDescriptorChoiceOp>(
      kernel.getLoc(), entry.getI1Type(), descriptorValues,
      entry.getStringAttr("host"), entry.getStringAttr("all_eligible"),
      entry.getStringAttr(tensorDescriptorChoice),
      entry.getStringAttr(tensorDescriptorEligibility));
  auto prepareBranch = [](Region &region) {
    Block &block = region.front();
    if (!block.empty() && isa<scf::YieldOp>(block.back()))
      block.back().erase();
    return OpBuilder(&block, block.end());
  };

  for (BlockLoadOp load : loads) {
    OpBuilder builder(load);
    FailureOr<TensorDescriptorOp> descriptorDeclaration = descriptorFor(
        load.getView(), load.getResult().getType(), load.getBlockAxes());
    FailureOr<SmallVector<Value>> descriptorOffsets =
        materializeDescriptorOffsets(builder, kernel, load.getLoc(),
                                     load.getView(), load.getOffsets());
    if (failed(descriptorDeclaration) || failed(descriptorOffsets))
      return load.emitOpError(
          "could not materialize the declared tensor-descriptor ABI");
    auto conditional = builder.create<scf::IfOp>(
        load.getLoc(), TypeRange{load.getResult().getType()}, choice.getResult(),
        /*withElseRegion=*/true);
    OpBuilder descriptorBuilder = prepareBranch(conditional.getThenRegion());
    auto descriptorLoad = descriptorBuilder.create<DescriptorLoadOp>(
        load.getLoc(), load.getResult().getType(),
        descriptorDeclaration->getResult(),
        *descriptorOffsets, load.getBoundaryAxesAttr());
    copyOrigin(load, descriptorLoad);
    descriptorBuilder.create<scf::YieldOp>(load.getLoc(),
                                           descriptorLoad.getResult());

    OpBuilder blockBuilder = prepareBranch(conditional.getElseRegion());
    auto block = blockBuilder.create<BlockLoadOp>(
        load.getLoc(), load.getResult().getType(), load.getView(),
        load.getOffsets(), load.getBlockAxesAttr(), load.getOrderAttr(),
        load.getBoundaryAxesAttr(), load.getPaddingAttr());
    copyOrigin(load, block);
    blockBuilder.create<scf::YieldOp>(load.getLoc(), block.getResult());
    load.getResult().replaceAllUsesWith(conditional.getResult(0));
    load.erase();
  }

  for (BlockStoreOp store : stores) {
    OpBuilder builder(store);
    FailureOr<TensorDescriptorOp> descriptorDeclaration = descriptorFor(
        store.getView(), store.getValue().getType(), store.getBlockAxes());
    FailureOr<SmallVector<Value>> descriptorOffsets =
        materializeDescriptorOffsets(builder, kernel, store.getLoc(),
                                     store.getView(), store.getOffsets());
    if (failed(descriptorDeclaration) || failed(descriptorOffsets))
      return store.emitOpError(
          "could not materialize the declared tensor-descriptor ABI");
    auto conditional = builder.create<scf::IfOp>(
        store.getLoc(), TypeRange{}, choice.getResult(),
        /*withElseRegion=*/true);
    OpBuilder descriptorBuilder = prepareBranch(conditional.getThenRegion());
    auto descriptorStore = descriptorBuilder.create<DescriptorStoreOp>(
        store.getLoc(), descriptorDeclaration->getResult(), *descriptorOffsets,
        store.getValue(), store.getBoundaryAxesAttr());
    copyOrigin(store, descriptorStore);
    descriptorBuilder.create<scf::YieldOp>(store.getLoc());

    OpBuilder blockBuilder = prepareBranch(conditional.getElseRegion());
    auto block = blockBuilder.create<BlockStoreOp>(
        store.getLoc(), store.getView(), store.getOffsets(), store.getValue(),
        store.getBlockAxesAttr(), store.getOrderAttr(),
        store.getBoundaryAxesAttr());
    copyOrigin(store, block);
    blockBuilder.create<scf::YieldOp>(store.getLoc());
    store.erase();
  }
  return choice;
}

bool fragmentFitsTritonTensor(gpu::FragmentType fragment,
                              const TritonConfig &config) {
  __int128 elements = 1;
  for (Attribute extent : fragment.getShape()) {
    std::optional<int64_t> value = evaluateCompileTimeExpression(
        cast<gpu::PhysicalExprAttr>(extent), config);
    // Runtime dimensions and full-coverage heuristics are checked by Triton at
    // specialization time.  This pass rejects only candidates whose typed
    // compile-time shape is already known to be illegal.
    if (!value)
      return true;
    if (*value <= 0)
      return false;
    elements *= *value;
    if (elements > maxTritonTensorElements)
      return false;
  }
  return true;
}

bool typeFitsTritonTensor(Type type, const TritonConfig &config) {
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return fragmentFitsTritonTensor(fragment, config);
  if (auto record = dyn_cast<gpu::RecordType>(type))
    return llvm::all_of(record.getFieldTypes(), [&](Attribute field) {
      return typeFitsTritonTensor(cast<TypeAttr>(field).getValue(), config);
    });
  return true;
}

bool descriptorFragmentFits(gpu::FragmentType fragment,
                            const TritonConfig &config,
                            const llvm::StringMap<SmallVector<int64_t>>
                                &parameterDomains,
                            const llvm::StringSet<> &coverageParameters) {
  std::optional<int64_t> elementBytes =
      descriptorElementBytes(fragment.getElementType());
  if (!elementBytes)
    return false;
  __int128 elements = 1;
  int64_t minimumLastExtent = std::numeric_limits<int64_t>::max();
  bool runtimeGuardsLastExtent = false;
  bool runtimeGuardsElementCount = false;
  for (auto [axis, extent] : llvm::enumerate(fragment.getShape())) {
    auto expression = cast<gpu::PhysicalExprAttr>(extent);
    SmallVector<int64_t> values;
    if (std::optional<int64_t> value =
            evaluateCompileTimeExpression(expression, config)) {
      values.push_back(*value);
    } else if (static_cast<gpu::PhysicalExprKind>(expression.getKind()) ==
               gpu::PhysicalExprKind::Parameter) {
      auto domain = parameterDomains.find(expression.getSymbol().getValue());
      if (domain == parameterDomains.end())
        return false;
      values.append(domain->second.begin(), domain->second.end());
      if (axis + 1 == fragment.getShape().size())
        runtimeGuardsLastExtent =
            coverageParameters.contains(expression.getSymbol().getValue());
      runtimeGuardsElementCount |=
          coverageParameters.contains(expression.getSymbol().getValue());
    } else {
      return false;
    }
    int64_t maximum = 0;
    int64_t minimum = std::numeric_limits<int64_t>::max();
    for (int64_t value : values) {
      if (value <= 0 || !llvm::isPowerOf2_64(value))
        return false;
      maximum = std::max(maximum, value);
      minimum = std::min(minimum, value);
    }
    elements *= maximum;
    if (axis + 1 == fragment.getShape().size())
      minimumLastExtent = minimum;
    if (elements > maxTritonTensorElements && !runtimeGuardsElementCount)
      return false;
  }
  return minimumLastExtent != std::numeric_limits<int64_t>::max() &&
         (runtimeGuardsLastExtent ||
          minimumLastExtent * *elementBytes >= 16);
}

LogicalResult materializeLegalConfigs(func::FuncOp kernel,
                                      TensorDescriptorChoiceOp descriptorChoice,
                                      ArrayRef<TritonLocalOptions> localOptions) {
  struct Domain {
    StringRef name;
    gpu::ParameterRole role;
    ArrayRef<int64_t> candidates;
    bool coverage;
  };
  SmallVector<Domain> domains;
  llvm::StringSet<> names;
  WalkResult schema = kernel.walk([&](gpu::ParameterOp parameter) {
    auto definition = parameter.getParameter();
    StringRef name = definition.getName().getValue();
    if (!names.insert(name).second) {
      parameter.emitOpError("duplicates a Triton parameter name");
      return WalkResult::interrupt();
    }
    domains.push_back({name,
                       static_cast<gpu::ParameterRole>(definition.getRole()),
                       definition.getCandidates().asArrayRef(),
                       parameter->hasAttr(gpu::coverageDimensionAttr)});
    return WalkResult::advance();
  });
  if (schema.wasInterrupted())
    return failure();
  llvm::StringMap<SmallVector<int64_t>> parameterDomains;
  llvm::StringSet<> coverageParameters;
  for (const Domain &domain : domains)
    parameterDomains[domain.name] =
        SmallVector<int64_t>(domain.candidates.begin(), domain.candidates.end());
  for (const Domain &domain : domains)
    if (domain.coverage)
      coverageParameters.insert(domain.name);

  auto shared =
      kernel->getAttrOfType<ArrayAttr>(gpu::sharedConfigTuplesAttr);
  if (!shared || shared.empty())
    return kernel.emitError(
        "Triton legalization requires shared config tuples");
  SmallVector<TritonConfig> configs;
  const Domain *warps = nullptr;
  const Domain *stages = nullptr;
  const Domain *ctas = nullptr;
  for (const Domain &domain : domains) {
    if (domain.role == gpu::ParameterRole::ProviderWarps)
      warps = &domain;
    else if (domain.role == gpu::ParameterRole::ProviderStages)
      stages = &domain;
    else if (domain.role == gpu::ParameterRole::ProviderCTAs)
      ctas = &domain;
    else if (domain.role == gpu::ParameterRole::ProviderThreads ||
             domain.role == gpu::ParameterRole::ProviderAccessForm)
      return kernel.emitError(
          "Triton program contains a foreign provider parameter");
  }
  if (!warps || !stages || !ctas)
    return kernel.emitError("Triton provider parameter domains are incomplete");
  for (Attribute attribute : shared) {
    auto tuple = dyn_cast<DictionaryAttr>(attribute);
    if (!tuple)
      return kernel.emitError("shared config tuple is malformed");
    TritonConfig sharedConfig;
    for (const Domain &domain : domains) {
      if (domain.coverage)
        continue;
      if (domain.role == gpu::ParameterRole::ProviderWarps ||
          domain.role == gpu::ParameterRole::ProviderStages ||
          domain.role == gpu::ParameterRole::ProviderCTAs)
        continue;
      auto value = tuple.getAs<IntegerAttr>(domain.name);
      if (!value || !llvm::is_contained(domain.candidates, value.getInt()))
        return kernel.emitError(
                   "shared config tuple does not bind a Triton kernel parameter: ")
               << domain.name;
      sharedConfig.kernelParameters[domain.name.str()] = value.getInt();
    }
    if (tuple.size() != sharedConfig.kernelParameters.size())
      return kernel.emitError(
          "shared config tuple contains a non-kernel binding");
    for (const TritonLocalOptions &options : localOptions) {
      int64_t formCount = descriptorChoice ? 2 : 1;
      for (int64_t form = 0; form < formCount; ++form) {
        TritonConfig config = sharedConfig;
        if (descriptorChoice)
          config.kernelParameters[descriptorChoice.getConfigParameter().str()] =
              form;
        config.warps = options.warps;
        config.stages = options.stages;
        config.ctas = options.ctas;
        if (llvm::none_of(configs, [&](const TritonConfig &existing) {
              return existing.kernelParameters == config.kernelParameters &&
                     existing.warps == config.warps &&
                     existing.stages == config.stages &&
                     existing.ctas == config.ctas;
            }))
          configs.push_back(std::move(config));
      }
    }
  }

  SmallVector<Attribute> encoded;
  Builder builder(kernel.getContext());
  for (const TritonConfig &config : configs) {
    if (config.warps <= 0 || config.stages <= 0 || config.ctas <= 0)
      return kernel.emitError("Triton provider parameter domains are incomplete");
    bool legal = true;
    bool descriptorConfig =
        descriptorChoice &&
        config.kernelParameters.at(
            descriptorChoice.getConfigParameter().str()) != 0;
    kernel.walk([&](Operation *operation) {
      if (!legal)
        return WalkResult::interrupt();
      if (auto range = dyn_cast<gpu::MakeRangeOp>(operation)) {
        auto fragment = dyn_cast<gpu::FragmentType>(range.getResult().getType());
        if (!fragment || fragment.getShape().size() != 1) {
          legal = false;
          return WalkResult::interrupt();
        }
        std::optional<int64_t> extent = evaluateCompileTimeExpression(
            cast<gpu::PhysicalExprAttr>(fragment.getShape()[0]), config);
        if (extent && (*extent <= 0 ||
                       (static_cast<uint64_t>(*extent) &
                        (static_cast<uint64_t>(*extent) - 1)) != 0)) {
          legal = false;
          return WalkResult::interrupt();
        }
      }
      if (descriptorConfig) {
        if (auto load = dyn_cast<DescriptorLoadOp>(operation))
          legal &= descriptorFragmentFits(load.getResult().getType(), config,
                                          parameterDomains,
                                          coverageParameters);
        else if (auto store = dyn_cast<DescriptorStoreOp>(operation))
          legal &= descriptorFragmentFits(store.getValue().getType(), config,
                                          parameterDomains,
                                          coverageParameters);
        if (!legal)
          return WalkResult::interrupt();
      }
      for (Type type : operation->getResultTypes())
        if (!typeFitsTritonTensor(type, config)) {
          legal = false;
          return WalkResult::interrupt();
        }
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            if (!typeFitsTritonTensor(argument.getType(), config)) {
              legal = false;
              return WalkResult::interrupt();
            }
      return WalkResult::advance();
    });
    if (!legal)
      continue;
    SmallVector<NamedAttribute> parameters;
    for (const auto &[name, value] : config.kernelParameters)
      parameters.push_back(builder.getNamedAttr(name, builder.getI64IntegerAttr(value)));
    encoded.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("parameters", builder.getDictionaryAttr(parameters)),
        builder.getNamedAttr("num_warps", builder.getI64IntegerAttr(config.warps)),
        builder.getNamedAttr("num_stages", builder.getI64IntegerAttr(config.stages)),
        builder.getNamedAttr("num_ctas", builder.getI64IntegerAttr(config.ctas)),
    }));
  }
  if (encoded.empty())
    return kernel.emitError(
        "all Triton parameter candidates violate typed fragment legality");
  kernel->setAttr(gpu::tritonConfigsAttr, builder.getArrayAttr(encoded));
  SmallVector<NamedAttribute> reductionBounds;
  kernel.walk([&](gpu::ParameterOp parameter) {
    if (parameter.getParameter().getRole() !=
            static_cast<uint32_t>(gpu::ParameterRole::Reduction) ||
        parameter->hasAttr(gpu::coverageDimensionAttr))
      return;
    auto dimension = parameter->getAttrOfType<IntegerAttr>(gpu::dimensionAttr);
    if (!dimension)
      return;
    for (BlockArgument argument : kernel.getArguments()) {
      DictionaryAttr attrs = kernel.getArgAttrDict(argument.getArgNumber());
      auto kind = attrs.getAs<StringAttr>(gpu::abiKindAttr);
      if (!kind || kind.getValue() != "dimension" ||
          attrs.getAs<IntegerAttr>(gpu::dimensionAttr) != dimension)
        continue;
      auto logicalExtent = gpu::PhysicalExprAttr::get(
          kernel.getContext(),
          static_cast<uint32_t>(gpu::PhysicalExprKind::Dimension),
          dimension.getInt(), attrs.getAs<StringAttr>(gpu::abiNameAttr),
          builder.getArrayAttr({}));
      auto bound = gpu::PhysicalExprAttr::get(
          kernel.getContext(),
          static_cast<uint32_t>(gpu::PhysicalExprKind::NextPowerOfTwo), 0,
          builder.getStringAttr(""), builder.getArrayAttr({logicalExtent}));
      reductionBounds.push_back(builder.getNamedAttr(
          parameter.getParameter().getName().getValue(), bound));
      break;
    }
  });
  kernel->setAttr(gpu::tritonReductionBoundsAttr,
                  builder.getDictionaryAttr(reductionBounds));
  return success();
}

bool isTritonScalarType(Type type) {
  if (type.isIndex() || isa<Float16Type, BFloat16Type, Float32Type,
                            Float64Type, Float8E4M3FNType,
                            Float8E5M2Type>(type))
    return true;
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() == 1 || integer.getWidth() == 8 ||
           integer.getWidth() == 16 || integer.getWidth() == 32 ||
           integer.getWidth() == 64;
  return false;
}

bool isTritonDataType(Type type) {
  if (isTritonScalarType(type))
    return true;
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return isTritonScalarType(fragment.getElementType());
  if (auto record = dyn_cast<gpu::RecordType>(type)) {
    for (Attribute field : record.getFieldTypes())
      if (!isTritonDataType(cast<TypeAttr>(field).getValue()))
        return false;
    return true;
  }
  return false;
}

bool isTritonExpression(gpu::PhysicalExprAttr expression);

Type elementType(Type type) {
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return fragment.getElementType();
  return type;
}

bool isTritonAtomicAddType(Type type) {
  type = elementType(type);
  if (isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(type))
    return true;
  auto integer = dyn_cast<IntegerType>(type);
  return integer && (integer.getWidth() == 32 || integer.getWidth() == 64);
}

bool samePhysicalShape(Type lhs, Type rhs) {
  auto left = dyn_cast<gpu::FragmentType>(lhs);
  auto right = dyn_cast<gpu::FragmentType>(rhs);
  if (static_cast<bool>(left) != static_cast<bool>(right))
    return false;
  return !left ||
         (left.getShape() == right.getShape() &&
          left.getAxisMaps() == right.getAxisMaps() &&
          left.getValidity() == right.getValidity() &&
          left.getOwner() == right.getOwner());
}

std::optional<unsigned>
expandedGatherAxis(gpu::FragmentType source, gpu::FragmentType result,
                   unsigned selectedSourceAxis) {
  if (source.getOwner() != result.getOwner() ||
      source.getShape().size() >= result.getShape().size() ||
      selectedSourceAxis >= source.getShape().size())
    return std::nullopt;

  gpu::BroadcastProjection projection = gpu::queryBroadcastProjection(source, result);
  if (!projection.isExact())
    return std::nullopt;
  std::optional<unsigned> selected;
  for (auto [resultIndex, sourceIndex] :
       llvm::enumerate(projection.targetToSource)) {
    if (sourceIndex) {
      if (*sourceIndex == selectedSourceAxis)
        selected = resultIndex;
      continue;
    }
    auto extent = dyn_cast<gpu::PhysicalExprAttr>(result.getShape()[resultIndex]);
    if (!extent ||
        extent.getKind() !=
            static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) ||
        extent.getValue() != 1)
      return std::nullopt;
  }
  return selected;
}

FailureOr<Value> zeroLike(OpBuilder &builder, Location location, Type type) {
  Type scalarType = elementType(type);
  Value zero;
  if (scalarType.isIndex())
    zero = builder.create<arith::ConstantIndexOp>(location, 0);
  else if (auto integer = dyn_cast<IntegerType>(scalarType))
    zero = builder.create<arith::ConstantOp>(
        location, integer, builder.getIntegerAttr(integer, 0));
  else
    return failure();
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return Value(builder.create<gpu::SplatOp>(location, fragment, zero));
  return zero;
}

std::optional<int64_t> staticGatherCoordinate(gpu::GatherOp gather) {
  if (gather.getCoordinates().size() != 1 ||
      gather.getSourceAxes().size() != 1 || !gather.getValid() ||
      !isTrueValue(gather.getValid()))
    return std::nullopt;
  Value coordinate = stripShapeOnly(gather.getCoordinates().front());
  while (auto cast = coordinate.getDefiningOp<gpu::CastOp>())
    coordinate = cast.getValue();
  auto constant = coordinate.getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  if (!integer || (integer.getInt() != 0 && integer.getInt() != 1))
    return std::nullopt;

  auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
  if (!source || !result || source.getShape().size() != result.getShape().size() + 1 ||
      gather.getSourceAxes().front() !=
          static_cast<int64_t>(source.getShape().size() - 1) ||
      source.getElementType() != result.getElementType() ||
      source.getValidity() != result.getValidity() ||
      source.getOwner() != result.getOwner())
    return std::nullopt;
  auto trailing = dyn_cast<gpu::PhysicalExprAttr>(
      source.getShape()[source.getShape().size() - 1]);
  if (!trailing ||
      trailing.getKind() !=
          static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) ||
      trailing.getValue() != 2 ||
      !std::equal(result.getShape().begin(), result.getShape().end(),
                  source.getShape().begin()) ||
      !std::equal(result.getAxisMaps().begin(), result.getAxisMaps().end(),
                  source.getAxisMaps().begin()))
    return std::nullopt;
  return integer.getInt();
}

bool belongsToSplitGatherPair(gpu::GatherOp gather) {
  std::optional<int64_t> coordinate = staticGatherCoordinate(gather);
  if (!coordinate || !gather->getBlock())
    return false;
  for (Operation *user : gather.getSource().getUsers()) {
    auto complement = dyn_cast<gpu::GatherOp>(user);
    std::optional<int64_t> other = complement
                                       ? staticGatherCoordinate(complement)
                                       : std::optional<int64_t>();
    if (complement && complement != gather &&
        complement->getBlock() == gather->getBlock() && other &&
        *other == 1 - *coordinate &&
        complement.getResult().getType() == gather.getResult().getType())
      return true;
  }
  return false;
}

LogicalResult legalizeSplitGatherPairs(func::FuncOp kernel) {
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  llvm::SmallPtrSet<Operation *, 8> rewritten;
  bool changed = false;
  for (gpu::GatherOp low : gathers) {
    if (rewritten.contains(low.getOperation()) || !low->getBlock() ||
        staticGatherCoordinate(low) != std::optional<int64_t>(0))
      continue;
    for (gpu::GatherOp high : gathers) {
      if (rewritten.contains(high.getOperation()) || !high->getBlock() ||
          high->getBlock() != low->getBlock() ||
          high.getSource() != low.getSource() ||
          high.getResult().getType() != low.getResult().getType() ||
          staticGatherCoordinate(high) != std::optional<int64_t>(1))
        continue;
      Operation *anchor = low->isBeforeInBlock(high) ? low.getOperation()
                                                    : high.getOperation();
      OpBuilder builder(anchor);
      auto split = builder.create<SplitOp>(
          anchor->getLoc(), low.getResult().getType(), high.getResult().getType(),
          low.getSource());
      low.getResult().replaceAllUsesWith(split.getLow());
      high.getResult().replaceAllUsesWith(split.getHigh());
      rewritten.insert(low.getOperation());
      rewritten.insert(high.getOperation());
      changed = true;
      break;
    }
  }
  if (changed) {
    for (gpu::GatherOp gather : gathers)
      if (rewritten.contains(gather.getOperation()))
        gather.erase();
    gpu::eraseDeadPhysicalValues(kernel);
  }
  return success();
}

LogicalResult legalizeMaskedGather(func::FuncOp kernel) {
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  for (gpu::GatherOp gather : gathers) {
    if (!gather.getValid() || belongsToSplitGatherPair(gather))
      continue;
    auto scalarConstant = [](Value value) -> arith::ConstantOp {
      while (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>())
        value = broadcast.getValue();
      while (auto splat = value.getDefiningOp<gpu::SplatOp>())
        value = splat.getValue();
      while (auto cast = value.getDefiningOp<gpu::CastOp>())
        value = cast.getValue();
      return value.getDefiningOp<arith::ConstantOp>();
    };
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    if (source && !result && source.getShape().size() == 1 &&
        gather.getCoordinates().size() == 1 &&
        gather.getSourceAxes() == ArrayRef<int64_t>{0} &&
        isa<IndexType, IntegerType>(gather.getCoordinates().front().getType()) &&
        gather.getValid().getType().isInteger(1)) {
      OpBuilder builder(gather);
      Value coordinate = gather.getCoordinates().front();
      FailureOr<Value> zero = zeroLike(builder, gather.getLoc(), coordinate.getType());
      if (failed(zero))
        return gather.emitOpError("scalar gather coordinate has no integral zero");
      Value safeIndex = builder.create<gpu::SelectOp>(
          gather.getLoc(), coordinate.getType(), gather.getValid(), coordinate, *zero);
      auto loaded = builder.create<gpu::GatherOp>(
          gather.getLoc(), gather.getResult().getType(), gather.getSource(),
          ValueRange{safeIndex}, Value(), Value(), gather.getSourceAxes());
      auto selected = builder.create<gpu::SelectOp>(
          gather.getLoc(), gather.getResult().getType(), gather.getValid(),
          loaded.getResult(), gather.getFill());
      if (Attribute origin = gather->getAttr(gpu::originAttr))
        selected->setAttr(gpu::originAttr, origin);
      gather.getResult().replaceAllUsesWith(selected.getResult());
      gather.erase();
      continue;
    }
    if (gather.getCoordinates().size() == 1 &&
        gather.getSourceAxes().size() == 1 && source && result &&
        source.getShape().size() < result.getShape().size()) {
      std::optional<unsigned> selectedAxis = expandedGatherAxis(
          source, result, gather.getSourceAxes().front());
      if (selectedAxis) {
        OpBuilder builder(gather);
        auto expandedSourceType = gpu::FragmentType::get(
            gather.getContext(), source.getElementType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        Value expandedSource = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), expandedSourceType, gather.getSource());
        Value coordinate = gather.getCoordinates().front();
        auto coordinateType = gpu::FragmentType::get(
            gather.getContext(), coordinate.getType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        if (auto coordinateFragment =
                dyn_cast<gpu::FragmentType>(coordinate.getType())) {
          coordinateType = gpu::FragmentType::get(
              gather.getContext(), coordinateFragment.getElementType(),
              result.getShape(), result.getAxisMaps(), result.getValidity(),
              result.getOwner());
        }
        coordinate = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), coordinateType, coordinate);
        FailureOr<Value> zero =
            zeroLike(builder, gather.getLoc(), coordinateType);
        if (succeeded(zero)) {
          Value safeIndex = builder.create<gpu::SelectOp>(
              gather.getLoc(), coordinateType, gather.getValid(), coordinate,
              *zero);
          auto safeGather = builder.create<gpu::GatherOp>(
              gather.getLoc(), result, expandedSource, ValueRange{safeIndex},
              Value(), Value(), ArrayRef<int64_t>{static_cast<int64_t>(*selectedAxis)});
          auto selected = builder.create<gpu::SelectOp>(
              gather.getLoc(), result, gather.getValid(), safeGather.getResult(),
              gather.getFill());
          if (Attribute origin = gather->getAttr(gpu::originAttr))
            selected->setAttr(gpu::originAttr, origin);
          gather.getResult().replaceAllUsesWith(selected.getResult());
          gather.erase();
          continue;
        }
      }
    }
    if (gather.getCoordinates().size() == 1 &&
        gather.getSourceAxes().size() == 1 && source && result &&
        source.getShape().size() == result.getShape().size() + 1) {
      unsigned selectedAxis = gather.getSourceAxes().front();
      auto selectedExtent =
          selectedAxis < source.getShape().size()
              ? dyn_cast<gpu::PhysicalExprAttr>(source.getShape()[selectedAxis])
              : gpu::PhysicalExprAttr();
      bool compatible = selectedExtent &&
                        selectedExtent.getKind() == static_cast<uint32_t>(
                                                        gpu::PhysicalExprKind::Constant) &&
                        selectedExtent.getValue() > 0;
      unsigned resultAxis = 0;
      for (unsigned sourceAxis = 0;
           compatible && sourceAxis < source.getShape().size(); ++sourceAxis) {
        if (sourceAxis == selectedAxis)
          continue;
        compatible = resultAxis < result.getShape().size() &&
                     source.getShape()[sourceAxis] == result.getShape()[resultAxis];
        ++resultAxis;
      }
      if (compatible) {
        OpBuilder builder(gather);
        SmallVector<Attribute> selectedShape(source.getShape().begin(),
                                             source.getShape().end());
        selectedShape[selectedAxis] = gpu::PhysicalExprAttr::get(
            gather.getContext(),
            static_cast<uint32_t>(gpu::PhysicalExprKind::Constant), 1,
            builder.getStringAttr(""), builder.getArrayAttr({}));
        auto coordinateType = gpu::FragmentType::get(
            gather.getContext(), gather.getCoordinates().front().getType(),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            source.getValidity(), source.getOwner());
        auto predicateType = gpu::FragmentType::get(
            gather.getContext(), builder.getI1Type(),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            source.getValidity(), source.getOwner());
        auto selectedResultType = gpu::FragmentType::get(
            gather.getContext(), result.getElementType(),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            result.getValidity(), result.getOwner());
        Value coordinate = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), coordinateType, gather.getCoordinates().front());
        Value valid = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), predicateType, gather.getValid());
        Value fill = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), selectedResultType, gather.getFill());
        FailureOr<Value> zero = zeroLike(builder, gather.getLoc(), coordinateType);
        if (succeeded(zero)) {
          Value safeIndex = builder.create<gpu::SelectOp>(
              gather.getLoc(), coordinateType, valid, coordinate, *zero);
          auto safeGather = builder.create<gpu::GatherOp>(
              gather.getLoc(), selectedResultType, gather.getSource(),
              ValueRange{safeIndex}, Value(), Value(), gather.getSourceAxes());
          Value selected = builder.create<gpu::SelectOp>(
              gather.getLoc(), selectedResultType, valid, safeGather.getResult(),
              fill);
          FailureOr<ArrayAttr> reassociation =
              gpu::inferReshapeReassociation(selectedResultType, result);
          if (failed(reassociation))
            return gather.emitOpError(
                "gather fallback has no exact row-major reassociation");
          auto reshaped = builder.create<gpu::ReshapeOp>(
              gather.getLoc(), result, selected, *reassociation);
          if (Attribute origin = gather->getAttr(gpu::originAttr))
            reshaped->setAttr(gpu::originAttr, origin);
          gather.getResult().replaceAllUsesWith(reshaped.getResult());
          gather.erase();
          continue;
        }
      }
    }
    if (gather.getCoordinates().size() == 1 &&
        isa<gpu::FragmentType>(gather.getCoordinates().front().getType()) &&
        gather.getSourceAxes().size() == 1 && source) {
      unsigned axis = gather.getSourceAxes().front();
      auto coordinate = scalarConstant(gather.getCoordinates().front());
      auto predicate = scalarConstant(gather.getValid());
      auto extent = axis < source.getShape().size()
                        ? dyn_cast<gpu::PhysicalExprAttr>(source.getShape()[axis])
                        : gpu::PhysicalExprAttr();
      auto coordinateValue =
          coordinate ? dyn_cast<IntegerAttr>(coordinate.getValue()) : IntegerAttr();
      auto predicateValue = predicate ? dyn_cast<IntegerAttr>(predicate.getValue())
                                      : IntegerAttr();
      if (extent &&
          extent.getKind() == static_cast<uint32_t>(
                                  gpu::PhysicalExprKind::Constant) &&
          coordinateValue && coordinateValue.getInt() >= 0 &&
          coordinateValue.getInt() < extent.getValue() && predicateValue &&
          predicateValue.getValue().isOne()) {
        OpBuilder builder(gather);
        auto replacement = builder.create<gpu::GatherOp>(
            gather.getLoc(), gather.getResult().getType(), gather.getSource(),
            gather.getCoordinates(), Value(), Value(), gather.getSourceAxes());
        if (Attribute origin = gather->getAttr(gpu::originAttr))
          replacement->setAttr(gpu::originAttr, origin);
        gather.getResult().replaceAllUsesWith(replacement.getResult());
        gather.erase();
        continue;
      }
    }
    if (gather.getCoordinates().size() != 1)
      continue;
    OpBuilder builder(gather);
    Value coordinate = gather.getCoordinates().front();
    if (!isa<gpu::FragmentType>(coordinate.getType())) {
      auto predicateType = dyn_cast<gpu::FragmentType>(gather.getValid().getType());
      if (!predicateType ||
          !isa<IntegerType, IndexType>(coordinate.getType()))
        continue;
      auto coordinateType = gpu::FragmentType::get(
          gather.getContext(), coordinate.getType(), predicateType.getShape(),
          predicateType.getAxisMaps(), predicateType.getValidity(),
          predicateType.getOwner());
      coordinate = builder.create<gpu::BroadcastOp>(gather.getLoc(),
                                                     coordinateType, coordinate);
    }
    if (!samePhysicalShape(gather.getValid().getType(), coordinate.getType()) ||
        !samePhysicalShape(gather.getValid().getType(),
                           gather.getResult().getType()))
      continue;
    FailureOr<Value> zero =
        zeroLike(builder, gather.getLoc(), coordinate.getType());
    if (failed(zero))
      continue;
    Value safeIndex = builder.create<gpu::SelectOp>(
        gather.getLoc(), coordinate.getType(), gather.getValid(), coordinate,
        *zero);
    auto safeGather = builder.create<gpu::GatherOp>(
        gather.getLoc(), gather.getResult().getType(), gather.getSource(),
        ValueRange{safeIndex}, Value(), Value(), gather.getSourceAxes());
    auto selected = builder.create<gpu::SelectOp>(
        gather.getLoc(), gather.getResult().getType(), gather.getValid(),
        safeGather.getResult(), gather.getFill());
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      selected->setAttr(gpu::originAttr, origin);
    gather.getResult().replaceAllUsesWith(selected.getResult());
    gather.erase();
  }
  return success();
}

LogicalResult legalizeExpandingGathers(func::FuncOp kernel) {
  auto [nextSource, nextDimension] = gpu::nextPhysicalAxisIdentities(kernel);
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  for (gpu::GatherOp gather : gathers) {
    if (gather.getValid() || gather.getCoordinates().size() != 1 ||
        gather.getSourceAxes().size() != 1)
      continue;
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    auto indices = dyn_cast<gpu::FragmentType>(
        gather.getCoordinates().front().getType());
    if (!source || !result || !indices || source.getShape().size() < 2 ||
        source.getShape().size() != result.getShape().size() ||
        indices.getShape() != result.getShape() ||
        indices.getAxisMaps() != result.getAxisMaps())
      continue;
    unsigned axis = gather.getSourceAxes().front();
    bool compatible = true;
    for (unsigned position = 0; position < source.getShape().size(); ++position)
      compatible &= position == axis ||
                    (source.getShape()[position] == result.getShape()[position] &&
                     source.getAxisMaps()[position] == result.getAxisMaps()[position]);
    auto sourceExtent = cast<gpu::PhysicalExprAttr>(source.getShape()[axis]);
    auto resultExtent = cast<gpu::PhysicalExprAttr>(result.getShape()[axis]);
    if (!compatible || sourceExtent == resultExtent ||
        !isCompileTimeExpression(sourceExtent) ||
        !isCompileTimeExpression(resultExtent))
      continue;
    auto constantExtent = [](gpu::PhysicalExprAttr extent) {
      return extent.getKind() ==
             static_cast<uint32_t>(gpu::PhysicalExprKind::Constant);
    };
    bool constantShape = constantExtent(sourceExtent) &&
                         constantExtent(resultExtent);
    if (constantShape && resultExtent.getValue() <= sourceExtent.getValue())
      continue;

    OpBuilder builder(gather);
    Location location = gather.getLoc();
    auto product = [&](ArrayRef<Attribute> shape) {
      auto extent = gpu::PhysicalExprAttr::get(
          kernel.getContext(),
          static_cast<uint32_t>(gpu::PhysicalExprKind::Constant), 1,
          builder.getStringAttr(""), builder.getArrayAttr({}));
      if (shape.empty())
        return extent;
      extent = cast<gpu::PhysicalExprAttr>(shape.front());
      for (Attribute dimension : shape.drop_front())
        extent = gpu::PhysicalExprAttr::get(
            kernel.getContext(),
            static_cast<uint32_t>(gpu::PhysicalExprKind::Multiply), 0,
            builder.getStringAttr(""), builder.getArrayAttr({extent, dimension}));
      return extent;
    };
    auto flatSourceAxis = gpu::AxisMapAttr::get(
        kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
    auto flatResultAxis = gpu::AxisMapAttr::get(
        kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
    auto sourceElements = product(source.getShape().getValue());
    auto resultElements = product(result.getShape().getValue());
    auto flatType = [&](gpu::FragmentType original, Type element,
                        gpu::PhysicalExprAttr extent, gpu::AxisMapAttr mapping) {
      return gpu::FragmentType::get(
          kernel.getContext(), element, builder.getArrayAttr({extent}),
          builder.getArrayAttr({mapping}), original.getValidity(),
          original.getOwner());
    };
    auto flatSourceType = flatType(source, source.getElementType(),
                                   sourceElements, flatSourceAxis);
    auto flatResultType = flatType(result, result.getElementType(),
                                   resultElements, flatResultAxis);
    auto flatIndexType = flatType(indices, builder.getIndexType(),
                                  resultElements, flatResultAxis);
    auto linearize = [&](OpBuilder &nested) -> FailureOr<Value> {
      auto reshape = [&](Value value,
                         gpu::FragmentType target) -> FailureOr<Value> {
        auto relation = gpu::inferReshapeReassociation(
            cast<gpu::FragmentType>(value.getType()), target);
        if (failed(relation))
          return failure();
        return Value(nested.create<gpu::ReshapeOp>(location, target, value,
                                                  *relation));
      };
      auto flatOriginalIndexType = flatType(
          indices, indices.getElementType(), resultElements, flatResultAxis);
      FailureOr<Value> flatSource = reshape(gather.getSource(), flatSourceType);
      FailureOr<Value> flatIndices =
          reshape(gather.getCoordinates().front(), flatOriginalIndexType);
      if (failed(flatSource) || failed(flatIndices))
        return failure();
      Value index = *flatIndices;
      if (index.getType() != flatIndexType)
        index = nested.create<gpu::CastOp>(location, flatIndexType, index);
      auto extentValue = [&](gpu::PhysicalExprAttr extent) -> Value {
        return nested.create<gpu::PhysicalExprOp>(
            location, nested.getIndexType(), extent);
      };
      auto broadcastExtent = [&](gpu::PhysicalExprAttr extent) -> Value {
        return nested.create<gpu::BroadcastOp>(location, flatIndexType,
                                               extentValue(extent));
      };
      auto binary = [&](Value lhs, Value rhs, BinaryOperator kind) -> Value {
        return nested.create<gpu::BinaryOp>(location, flatIndexType, lhs, rhs,
                                            kind);
      };
      Value zero = nested.create<arith::ConstantIndexOp>(location, 0);
      Value one = nested.create<arith::ConstantIndexOp>(location, 1);
      Value count = extentValue(resultElements);
      Value ordinal = nested.create<gpu::MakeRangeOp>(
          location, flatIndexType, zero, count, one, zero, count,
          flatResultAxis.getSourceId(), flatResultAxis.getSourceAxis(), true);
      Value inner = broadcastExtent(
          product(result.getShape().getValue().drop_front(axis + 1)));
      Value sourceStride = binary(broadcastExtent(sourceExtent), inner,
                                   BinaryOperator::Multiply);
      Value resultStride = binary(broadcastExtent(resultExtent), inner,
                                   BinaryOperator::Multiply);
      Value outer = binary(ordinal, resultStride, BinaryOperator::FloorDivide);
      Value tail = binary(ordinal, inner, BinaryOperator::Remainder);
      Value offset = binary(outer, sourceStride, BinaryOperator::Multiply);
      offset = binary(offset, binary(index, inner, BinaryOperator::Multiply),
                      BinaryOperator::Add);
      offset = binary(offset, tail, BinaryOperator::Add);
      Value selected = nested.create<gpu::GatherOp>(
          location, flatResultType, *flatSource, ValueRange{offset}, Value(),
          Value(), ArrayRef<int64_t>{0});
      return reshape(selected, result);
    };

    // Expanding gathers can require cross-warp exchange.  A linear gather
    // retains that meaning without the rank-dependent warp-local rewrite.
    Value replacement;
    if (constantShape) {
      FailureOr<Value> linear = linearize(builder);
      if (failed(linear))
        return gather.emitOpError("expanding gather has no exact linear reshape");
      replacement = *linear;
    } else {
      Value sourceSize = builder.create<gpu::PhysicalExprOp>(
          location, builder.getIndexType(), sourceExtent);
      Value resultSize = builder.create<gpu::PhysicalExprOp>(
          location, builder.getIndexType(), resultExtent);
      Value expanding = builder.create<gpu::CompareOp>(
          location, builder.getI1Type(), resultSize, sourceSize,
          ComparePredicate::Gt);
      auto conditional = builder.create<scf::IfOp>(
          location, TypeRange{result}, expanding, /*withElseRegion=*/true);
      OpBuilder linearBuilder = conditional.getThenBodyBuilder();
      FailureOr<Value> linear = linearize(linearBuilder);
      if (failed(linear))
        return gather.emitOpError("expanding gather has no exact linear reshape");
      linearBuilder.create<scf::YieldOp>(location, *linear);
      OpBuilder originalBuilder = conditional.getElseBodyBuilder();
      Operation *original = originalBuilder.clone(*gather.getOperation());
      originalBuilder.create<scf::YieldOp>(location, original->getResults());
      replacement = conditional.getResult(0);
    }
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      replacement.getDefiningOp()->setAttr(gpu::originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement);
    gather.erase();
  }
  return success();
}

bool isAddCombine(gpu::ScatterReduceOp scatter) {
  return gpu::queryBinaryCombineKind(scatter.getCombine()) ==
         BinaryOperator::Add;
}

std::optional<StringRef> nativeReduceForm(gpu::ReduceOp reduce) {
  if (reduce.getSourceCount() != 1 || reduce.getIdentityCount() != 1 ||
      reduce.getCaptureCount() != 0 || reduce.getResults().size() != 1 ||
      reduce.getCombine().empty() || reduce.getCombine().getBlocks().size() != 1)
    return std::nullopt;
  std::optional<BinaryOperator> combine =
      gpu::queryBinaryCombineKind(reduce.getCombine());
  if (!combine)
    return std::nullopt;
  if (*combine == BinaryOperator::Add)
    return "sum";
  auto source = dyn_cast<gpu::FragmentType>(reduce.getInputs().front().getType());
  if (!source || !isa<IntegerType, IndexType>(source.getElementType()))
    return std::nullopt;
  if (*combine == BinaryOperator::Maximum)
    return "max";
  if (*combine == BinaryOperator::Minimum)
    return "min";
  return std::nullopt;
}

void selectNativeReduceForms(func::FuncOp kernel) {
  kernel.walk([&](gpu::ReduceOp reduce) {
    if (std::optional<StringRef> form = nativeReduceForm(reduce))
      reduce->setAttr(reduceFormAttr,
                      StringAttr::get(kernel.getContext(), *form));
  });
}

LogicalResult legalizeContractShapes(func::FuncOp kernel) {
  auto [nextSource, nextDimension] = gpu::nextPhysicalAxisIdentities(kernel);
  SmallVector<gpu::ContractOp> contracts;
  kernel.walk([&](gpu::ContractOp contract) { contracts.push_back(contract); });
  for (gpu::ContractOp contract : contracts) {
    auto lhs = contract.getLhs().getType();
    auto rhs = contract.getRhs().getType();
    if (contract.getLhsReductionAxes().size() != 1 ||
        contract.getRhsReductionAxes().size() != 1 ||
        (lhs.getShape().size() <= 2 && rhs.getShape().size() <= 2))
      continue;
    ArrayRef<int64_t> lhsBatch = contract.getLhsBatchAxes();
    ArrayRef<int64_t> rhsBatch = contract.getRhsBatchAxes();
    unsigned batchRank = lhsBatch.size();
    if (batchRank > 1)
      continue;
    auto canonicalBatch = [&](ArrayRef<int64_t> axes) {
      return llvm::all_of(llvm::enumerate(axes), [](auto entry) {
        return static_cast<int64_t>(entry.index()) == entry.value();
      });
    };
    if (batchRank && lhs.getShape().size() == batchRank + 2 &&
        rhs.getShape().size() == batchRank + 2 &&
        canonicalBatch(lhsBatch) && canonicalBatch(rhsBatch) &&
        contract.getLhsReductionAxes().front() == batchRank + 1 &&
        contract.getRhsReductionAxes().front() == batchRank)
      continue;
    auto freeAxes = [](gpu::FragmentType type, int64_t reduction,
                       ArrayRef<int64_t> batch) {
      SmallVector<int64_t> axes;
      for (int64_t axis = 0; axis < static_cast<int64_t>(type.getShape().size()); ++axis)
        if (axis != reduction && !llvm::is_contained(batch, axis))
          axes.push_back(axis);
      return axes;
    };
    auto lhsFree = freeAxes(lhs, contract.getLhsReductionAxes().front(), lhsBatch);
    auto rhsFree = freeAxes(rhs, contract.getRhsReductionAxes().front(), rhsBatch);
    if (lhsFree.empty() || rhsFree.empty())
      return contract.emitOpError("Triton matrix normalization requires free axes on both operands");
    OpBuilder builder(contract);
    Location location = contract.getLoc();
    auto remap = [&](gpu::AxisMapAttr axis, unsigned position) {
      return gpu::AxisMapAttr::get(kernel.getContext(), axis.getSourceId(),
          axis.getSourceAxis(), axis.getDimensionId(), position, axis.getDerived());
    };
    auto collapsedAxis = [&](gpu::FragmentType type, ArrayRef<int64_t> axes,
                             unsigned position) {
      if (axes.size() == 1)
        return remap(cast<gpu::AxisMapAttr>(type.getAxisMaps()[axes.front()]), position);
      return gpu::AxisMapAttr::get(kernel.getContext(), nextSource++, 0,
                                   nextDimension++, position, true);
    };
    auto product = [&](gpu::FragmentType type, ArrayRef<int64_t> axes) {
      auto extent = cast<gpu::PhysicalExprAttr>(type.getShape()[axes.front()]);
      for (int64_t axis : axes.drop_front())
        extent = gpu::PhysicalExprAttr::get(kernel.getContext(),
            static_cast<uint32_t>(gpu::PhysicalExprKind::Multiply), 0,
            builder.getStringAttr(""),
            builder.getArrayAttr({extent, type.getShape()[axis]}));
      return extent;
    };
    auto makeType = [&](gpu::FragmentType source, ArrayRef<Attribute> shape,
                        ArrayRef<Attribute> mappings) {
      return gpu::FragmentType::get(kernel.getContext(), source.getElementType(),
          builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
          source.getValidity(), source.getOwner());
    };
    auto transpose = [&](Value value, ArrayRef<int64_t> permutation) {
      if (llvm::all_of(llvm::enumerate(permutation), [](auto entry) {
            return static_cast<int64_t>(entry.index()) == entry.value();
          }))
        return value;
      auto type = cast<gpu::FragmentType>(value.getType());
      SmallVector<Attribute> shape, mappings;
      for (auto [position, axis] : llvm::enumerate(permutation)) {
        shape.push_back(type.getShape()[axis]);
        mappings.push_back(remap(cast<gpu::AxisMapAttr>(type.getAxisMaps()[axis]), position));
      }
      return Value(builder.create<gpu::TransposeOp>(location,
          makeType(type, shape, mappings), value, permutation));
    };
    auto reshape = [&](Value value, gpu::FragmentType target) -> FailureOr<Value> {
      if (value.getType() == target) return value;
      auto relation = gpu::inferReshapeReassociation(
          cast<gpu::FragmentType>(value.getType()), target);
      if (failed(relation)) return failure();
      return Value(builder.create<gpu::ReshapeOp>(location, target, value, *relation));
    };
    auto originalResult = contract.getResult().getType();
    SmallVector<int64_t> resultPermutation;
    auto appendResultAxes = [&](gpu::FragmentType operand, ArrayRef<int64_t> axes) {
      for (int64_t axis : axes) {
        auto mapping = cast<gpu::AxisMapAttr>(operand.getAxisMaps()[axis]);
        auto source = gpu::queryFragmentAxis(originalResult, gpu::sourceAxisIdentity(mapping));
        auto dimension = gpu::queryFragmentDimension(originalResult, mapping.getDimensionId());
        std::optional<int64_t> position;
        if (source.isExact()) position = source.fragmentAxis;
        else if (dimension.isExact()) position = dimension.fragmentAxis;
        if (!position || llvm::is_contained(resultPermutation, *position))
          return failure();
        resultPermutation.push_back(*position);
      }
      return success();
    };
    if (failed(appendResultAxes(lhs, lhsBatch)) ||
        failed(appendResultAxes(lhs, lhsFree)) ||
        failed(appendResultAxes(rhs, rhsFree)) ||
        resultPermutation.size() != originalResult.getShape().size())
      return contract.emitOpError("Triton matrix free axes have no bijective result projection");
    auto m = product(lhs, lhsFree);
    auto n = product(rhs, rhsFree);
    auto mAxis = collapsedAxis(lhs, lhsFree, batchRank);
    auto nAxis = collapsedAxis(rhs, rhsFree, batchRank + 1);
    int64_t lhsK = contract.getLhsReductionAxes().front();
    int64_t rhsK = contract.getRhsReductionAxes().front();
    SmallVector<int64_t> lhsPermutation(lhsBatch);
    llvm::append_range(lhsPermutation, lhsFree);
    lhsPermutation.push_back(lhsK);
    SmallVector<int64_t> rhsPermutation(rhsBatch);
    rhsPermutation.push_back(rhsK);
    llvm::append_range(rhsPermutation, rhsFree);
    SmallVector<Attribute> lhsShape, lhsMappings, rhsShape, rhsMappings;
    SmallVector<Attribute> resultShape, resultMappings;
    SmallVector<int64_t> matrixBatch;
    for (auto [position, pair] : llvm::enumerate(llvm::zip(lhsBatch, rhsBatch))) {
      auto [left, right] = pair;
      lhsShape.push_back(lhs.getShape()[left]);
      lhsMappings.push_back(remap(
          cast<gpu::AxisMapAttr>(lhs.getAxisMaps()[left]), position));
      rhsShape.push_back(rhs.getShape()[right]);
      rhsMappings.push_back(remap(
          cast<gpu::AxisMapAttr>(rhs.getAxisMaps()[right]), position));
      int64_t resultAxis = resultPermutation[position];
      resultShape.push_back(originalResult.getShape()[resultAxis]);
      resultMappings.push_back(remap(
          cast<gpu::AxisMapAttr>(originalResult.getAxisMaps()[resultAxis]), position));
      matrixBatch.push_back(position);
    }
    llvm::append_range(lhsShape, ArrayRef<Attribute>{m, lhs.getShape()[lhsK]});
    llvm::append_range(lhsMappings, ArrayRef<Attribute>{mAxis,
        remap(cast<gpu::AxisMapAttr>(lhs.getAxisMaps()[lhsK]), batchRank + 1)});
    llvm::append_range(rhsShape, ArrayRef<Attribute>{rhs.getShape()[rhsK], n});
    llvm::append_range(rhsMappings, ArrayRef<Attribute>{
        remap(cast<gpu::AxisMapAttr>(rhs.getAxisMaps()[rhsK]), batchRank), nAxis});
    llvm::append_range(resultShape, ArrayRef<Attribute>{m, n});
    llvm::append_range(resultMappings, ArrayRef<Attribute>{mAxis, nAxis});
    auto matrixResult = makeType(originalResult, resultShape, resultMappings);
    FailureOr<Value> matrixLhs = reshape(transpose(contract.getLhs(), lhsPermutation),
        makeType(lhs, lhsShape, lhsMappings));
    FailureOr<Value> matrixRhs = reshape(transpose(contract.getRhs(), rhsPermutation),
        makeType(rhs, rhsShape, rhsMappings));
    Value orderedAccumulator = transpose(contract.getAccumulator(), resultPermutation);
    auto orderedResult = cast<gpu::FragmentType>(orderedAccumulator.getType());
    FailureOr<Value> matrixAccumulator = reshape(orderedAccumulator, matrixResult);
    if (failed(matrixLhs) || failed(matrixRhs) || failed(matrixAccumulator))
      return contract.emitOpError("Triton matrix form has no exact row-major reshape");
    contract->setOperands(ValueRange{*matrixLhs, *matrixRhs, *matrixAccumulator});
    contract->setAttr("lhs_reduction_axes", builder.getDenseI64ArrayAttr({batchRank + 1}));
    contract->setAttr("rhs_reduction_axes", builder.getDenseI64ArrayAttr({batchRank}));
    contract->setAttr("lhs_batch_axes", builder.getDenseI64ArrayAttr(matrixBatch));
    contract->setAttr("rhs_batch_axes", builder.getDenseI64ArrayAttr(matrixBatch));
    contract.getResult().setType(matrixResult);
    builder.setInsertionPointAfter(contract);
    FailureOr<Value> restored = reshape(contract.getResult(), orderedResult);
    if (failed(restored))
      return contract.emitOpError("Triton matrix result has no inverse row-major reshape");
    Operation *firstRestore = (*restored).getDefiningOp();
    SmallVector<int64_t> inverse(resultPermutation.size());
    for (auto [position, axis] : llvm::enumerate(resultPermutation))
      inverse[axis] = position;
    Value result = transpose(*restored, inverse);
    contract.getResult().replaceUsesWithIf(result, [&](OpOperand &use) {
      return use.getOwner() != firstRestore &&
             use.getOwner() != result.getDefiningOp();
    });
  }
  return success();
}

void selectContractForms(func::FuncOp kernel) {
  SmallVector<gpu::ContractOp> contracts;
  kernel.walk([&](gpu::ContractOp contract) { contracts.push_back(contract); });
  for (gpu::ContractOp contract : contracts) {
    auto lhs = contract.getLhs().getType();
    if (lhs.getShape().empty())
      continue;
    auto reductionExtent =
        dyn_cast<gpu::PhysicalExprAttr>(
            lhs.getShape()[lhs.getShape().size() - 1]);
    if (reductionExtent &&
        reductionExtent.getKind() ==
            static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) &&
        reductionExtent.getValue() < 16) {
      contract->setAttr(contractFormAttr,
                        StringAttr::get(kernel.getContext(), "multiply_sum"));
      continue;
    }
    if (lhs.getShape().size() != 2 || !lhs.getElementType().isF32() ||
        !contract.getRhs().getType().getElementType().isF32() ||
        !contract.getAccumulator().getType().getElementType().isF32())
      continue;
    auto rows = cast<gpu::PhysicalExprAttr>(lhs.getShape()[0]);
    if (rows.getKind() == static_cast<uint32_t>(gpu::PhysicalExprKind::Constant)) {
      if (rows.getValue() <= 4)
        contract->setAttr(contractFormAttr,
                          StringAttr::get(kernel.getContext(), "multiply_sum"));
      continue;
    }
    if (rows.getKind() != static_cast<uint32_t>(gpu::PhysicalExprKind::Parameter))
      continue;
    FailureOr<gpu::ParameterOp> parameter =
        gpu::queryParameterBySymbol(kernel, rows.getSymbol());
    if (failed(parameter))
      continue;
    auto candidates = parameter->getParameter().getCandidates().asArrayRef();
    if (llvm::none_of(candidates, [](int64_t extent) { return extent <= 4; }))
      continue;
    OpBuilder builder(contract);
    Value limit = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 4);
    Value small = builder.create<gpu::CompareOp>(
        contract.getLoc(), builder.getI1Type(), parameter->getResult(), limit,
        ComparePredicate::Le);
    auto choice = builder.create<scf::IfOp>(
        contract.getLoc(), TypeRange{contract.getResult().getType()}, small, true);
    builder.setInsertionPointToStart(&choice.getThenRegion().front());
    auto vector = cast<gpu::ContractOp>(builder.clone(*contract));
    vector->setAttr(contractFormAttr,
                    StringAttr::get(kernel.getContext(), "multiply_sum"));
    builder.create<scf::YieldOp>(contract.getLoc(), vector.getResult());
    builder.setInsertionPointToStart(&choice.getElseRegion().front());
    auto matrix = cast<gpu::ContractOp>(builder.clone(*contract));
    builder.create<scf::YieldOp>(contract.getLoc(), matrix.getResult());
    contract.getResult().replaceAllUsesWith(choice.getResult(0));
    contract.erase();
  }
}

LogicalResult legalizeScatterAdd(func::FuncOp kernel) {
  SmallVector<gpu::ScatterReduceOp> scatters;
  kernel.walk([&](gpu::ScatterReduceOp scatter) { scatters.push_back(scatter); });
  for (gpu::ScatterReduceOp scatter : scatters) {
    if (!isAddCombine(scatter) || !isTritonAtomicAddType(scatter.getValue().getType()))
      continue;
    OpBuilder builder(scatter);
    auto atomic = builder.create<gpu::AtomicRMWOp>(
        scatter.getLoc(), scatter.getValue().getType(), scatter.getResource(),
        scatter.getCoordinates(), scatter.getValue(), scatter.getValid(),
        AtomicRMWKind::Add, AtomicOrdering::Relaxed, scatter.getSharing(),
        scatter.getSourceAxes());
    if (Attribute origin = scatter->getAttr(gpu::originAttr))
      atomic->setAttr(gpu::originAttr, origin);
    scatter.erase();
  }
  return success();
}

bool isTritonFragmentExtent(Attribute attribute) {
  auto expression = dyn_cast<gpu::PhysicalExprAttr>(attribute);
  if (!expression)
    return false;
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  return kind != gpu::PhysicalExprKind::ScalarABI &&
         isTritonExpression(expression);
}

bool isTritonExpression(gpu::PhysicalExprAttr expression) {
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  switch (kind) {
  case gpu::PhysicalExprKind::Constant:
  case gpu::PhysicalExprKind::Parameter:
  case gpu::PhysicalExprKind::Dimension:
  case gpu::PhysicalExprKind::ScalarABI:
    return expression.getOperands().empty();
  case gpu::PhysicalExprKind::Add:
  case gpu::PhysicalExprKind::Multiply:
  case gpu::PhysicalExprKind::CeilDiv:
  case gpu::PhysicalExprKind::Minimum:
  case gpu::PhysicalExprKind::Subtract:
  case gpu::PhysicalExprKind::FloorDiv:
  case gpu::PhysicalExprKind::Maximum:
    if (expression.getOperands().size() != 2)
      return false;
    break;
  case gpu::PhysicalExprKind::Select:
    if (expression.getOperands().size() != 3)
      return false;
    break;
  case gpu::PhysicalExprKind::NextPowerOfTwo:
    if (expression.getOperands().size() != 1)
      return false;
    break;
  default:
    return false;
  }
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isTritonExpression(cast<gpu::PhysicalExprAttr>(operand));
  });
}

LogicalResult verifyAccess(Operation *operation, Value resource,
                           ValueRange coordinates,
                           ArrayRef<int64_t> sourceAxes) {
  auto view = dyn_cast<gpu::ViewType>(resource.getType());
  if (!view)
    return operation->emitOpError(
        "Triton pointer access requires a legalized external-view resource");
  if (coordinates.size() != sourceAxes.size())
    return operation->emitOpError(
        "Triton access coordinates and source axes are not bijective");
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t axis : sourceAxes)
    if (axis < 0 || axis >= view.getRank() || !seen.insert(axis).second)
      return operation->emitOpError(
          "Triton access contains an invalid or duplicate source axis");
  return success();
}

Type scalarCallbackType(Type type) {
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return fragment.getElementType();
  if (auto record = dyn_cast<gpu::RecordType>(type)) {
    SmallVector<Attribute> fields;
    for (Attribute field : record.getFieldTypes())
      fields.push_back(TypeAttr::get(scalarCallbackType(cast<TypeAttr>(field).getValue())));
    return gpu::RecordType::get(type.getContext(), record.getFieldNames(),
                                ArrayAttr::get(type.getContext(), fields), record.getOwner());
  }
  return type;
}

LogicalResult legalizeCollectiveCallbacks(func::FuncOp kernel) {
  SmallVector<Operation *> collectives;
  kernel.walk([&](Operation *operation) {
    if (auto reduce = dyn_cast<gpu::ReduceOp>(operation)) {
      if (!reduce->hasAttr(reduceFormAttr)) collectives.push_back(operation);
    } else if (isa<gpu::ScanOp>(operation)) {
      collectives.push_back(operation);
    }
  });
  for (Operation *operation : collectives) {
    auto reduce = dyn_cast<gpu::ReduceOp>(operation);
    auto scan = dyn_cast<gpu::ScanOp>(operation);
    unsigned count = reduce ? reduce.getSourceCount() : scan.getSourceCount();
    if ((reduce && (reduce.getAxes().size() != 1 || reduce.getCaptureCount())) ||
        (scan && (!scan.getInclusive() || scan.getCaptureCount())))
      return operation->emitOpError("native collective requires one axis and an inclusive, capture-free callback");
    Region &region = reduce ? reduce.getCombine() : scan.getCombine();
    Block &body = region.front();
    for (Operation &nested : body) {
      for (Value operand : nested.getOperands())
        if (operand.getParentBlock() != &body)
          return nested.emitOpError("native collective callback cannot capture enclosing values");
      if (!isa<arith::ConstantOp, gpu::SplatOp, gpu::BroadcastOp,
               gpu::UnaryOp, gpu::BinaryOp, gpu::CompareOp, gpu::SelectOp,
               gpu::CastOp, gpu::BitcastOp, gpu::MakeRecordOp, gpu::ExtractOp,
               gpu::YieldOp>(nested))
        return nested.emitOpError("Triton native collective requires an elementwise scalarizable combine");
      if (auto broadcast = dyn_cast<gpu::BroadcastOp>(nested)) {
        auto source = dyn_cast<gpu::FragmentType>(broadcast.getValue().getType());
        if (source) {
          auto target = broadcast.getResult().getType();
          auto projection = gpu::queryAxisProjection(source, target);
          if (source.getShape() != target.getShape() || !projection.isExact() ||
              llvm::any_of(llvm::enumerate(projection.targetToSource), [](auto pair) {
                return !pair.value() || *pair.value() != pair.index();
              }))
            return broadcast.emitOpError("non-identity fragment broadcast in a collective requires prior lane-wise legalization");
        }
      }
    }
    OpBuilder builder(operation);
    OperationState state(operation->getLoc(), reduce ? ReduceOp::getOperationName() : ScanOp::getOperationName());
    state.addOperands(operation->getOperands());
    state.addTypes(operation->getResultTypes());
    state.addAttribute("source_count", builder.getI64IntegerAttr(count));
    state.addAttribute("axis", builder.getI64IntegerAttr(reduce ? reduce.getAxes().front() : scan.getAxis()));
    state.addAttribute("reverse", builder.getBoolAttr(scan && scan.getReverse()));
    if (Attribute origin = operation->getAttr(gpu::originAttr)) state.addAttribute(gpu::originAttr, origin);
    state.addRegion();
    Operation *native = builder.create(state);
    Block *scalarBody = new Block();
    native->getRegion(0).push_back(scalarBody);
    IRMapping mapping;
    for (BlockArgument argument : body.getArguments())
      mapping.map(argument, scalarBody->addArgument(scalarCallbackType(argument.getType()), argument.getLoc()));
    builder.setInsertionPointToEnd(scalarBody);
    for (Operation &nested : body) {
      if (isa<gpu::SplatOp, gpu::BroadcastOp>(nested)) {
        mapping.map(nested.getResult(0), mapping.lookup(nested.getOperand(0)));
        continue;
      }
      Operation *cloned = builder.clone(nested, mapping);
      for (Value result : cloned->getResults())
        result.setType(scalarCallbackType(result.getType()));
    }
    operation->replaceAllUsesWith(native->getResults());
    operation->erase();
  }
  return success();
}

void foldIntegerScanTails(func::FuncOp kernel) {
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  bool changed = false;
  for (gpu::GatherOp gather : gathers) {
    auto type = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    if (!type || type.getShape().size() != 1 ||
        (!type.getElementType().isInteger(32) &&
         !type.getElementType().isInteger(64)) ||
        gather.getType() != type.getElementType() ||
        gather.getSourceAxes() != ArrayRef<int64_t>{0} ||
        gather.getCoordinates().size() != 1 ||
        (gather.getValid() &&
         (!gather.getValid().getType().isInteger(1) || !gather.getFill())))
      continue;
    auto extent = cast<gpu::PhysicalExprAttr>(type.getShape()[0]);
    bool positive = extent.getKind() ==
                        static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) &&
                    extent.getValue() > 0;
    if (extent.getKind() ==
        static_cast<uint32_t>(gpu::PhysicalExprKind::Parameter)) {
      auto parameter = gpu::queryParameterBySymbol(kernel, extent.getSymbol());
      positive = succeeded(parameter) && llvm::all_of(
          (*parameter).getParameter().getCandidates().asArrayRef(),
          [](int64_t candidate) { return candidate > 0; });
    }
    if (!positive)
      continue;
    Value index = gather.getCoordinates().front();
    auto coordinate = gpu::queryLaunchExpression(index);
    bool last = coordinate &&
                coordinate.getKind() ==
                    static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) &&
                extent.getKind() == coordinate.getKind() &&
                coordinate.getValue() == extent.getValue() - 1;
    if (auto subtract = index.getDefiningOp<gpu::BinaryOp>();
        subtract && subtract.getOperatorKind() == BinaryOperator::Subtract) {
      auto one = gpu::queryLaunchExpression(subtract.getRhs());
      last |= gpu::queryLaunchExpression(subtract.getLhs()) == extent && one &&
              one.getKind() ==
                  static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) &&
              one.getValue() == 1;
    }
    if (!last)
      continue;

    Value source = gather.getSource();
    Value base;
    auto addition = source.getDefiningOp<gpu::BinaryOp>();
    if (addition && addition.getOperatorKind() == BinaryOperator::Add) {
      auto scalar = [&](Value value) -> Value {
        if (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>())
          value = broadcast.getValue();
        else if (auto splat = value.getDefiningOp<gpu::SplatOp>())
          value = splat.getValue();
        return value.getType() == type.getElementType() ? value : Value();
      };
      if ((base = scalar(addition.getLhs())))
        source = addition.getRhs();
      else if ((base = scalar(addition.getRhs())))
        source = addition.getLhs();
    }
    auto scan = source.getDefiningOp<gpu::ScanOp>();
    if (!scan || scan.getSourceCount() != 1 || scan.getIdentityCount() != 1 ||
        scan.getCaptureCount() || scan.getAxis() != 0 || !scan.getInclusive() ||
        scan.getReverse() || source.getType() != type ||
        !gpu::isLiteralZeroProjection(scan.getInputs()[1]))
      continue;
    Block &body = scan.getCombine().front();
    auto combine = dyn_cast<gpu::BinaryOp>(body.front());
    auto yield = cast<gpu::YieldOp>(body.getTerminator());
    if (!llvm::hasSingleElement(body.without_terminator()) || !combine ||
        combine.getOperatorKind() != BinaryOperator::Add ||
        combine.getLhs() != body.getArgument(0) ||
        combine.getRhs() != body.getArgument(1) ||
        yield.getValues().front() != combine.getResult())
      continue;

    // A scalar cross-warp gather materializes the whole prefix in shared memory.
    // Integer addition gives the same terminal value through a scalar reduction.
    OpBuilder builder(gather);
    Value zero = builder.create<arith::ConstantOp>(
        gather.getLoc(), builder.getZeroAttr(type.getElementType()));
    OperationState state(gather.getLoc(), gpu::ReduceOp::getOperationName());
    state.addOperands({scan.getInputs().front(), zero});
    state.addTypes(type.getElementType());
    state.addAttribute("source_count", builder.getI64IntegerAttr(1));
    state.addAttribute("identity_count", builder.getI64IntegerAttr(1));
    state.addAttribute("capture_count", builder.getI64IntegerAttr(0));
    state.addAttribute("axes", builder.getDenseI64ArrayAttr({0}));
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      state.addAttribute(gpu::originAttr, origin);
    state.addRegion();
    auto reduced = cast<gpu::ReduceOp>(builder.create(state));
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *scalarBody = new Block();
      reduced.getCombine().push_back(scalarBody);
      Value lhs = scalarBody->addArgument(type.getElementType(), gather.getLoc());
      Value rhs = scalarBody->addArgument(type.getElementType(), gather.getLoc());
      builder.setInsertionPointToEnd(scalarBody);
      auto sum = builder.create<gpu::BinaryOp>(
          gather.getLoc(), type.getElementType(), lhs, rhs, BinaryOperator::Add);
      sum->setAttrs(combine->getAttrs());
      builder.create<gpu::YieldOp>(gather.getLoc(), sum.getResult());
    }
    Value result = reduced.getResult(0);
    if (base) {
      auto sum = builder.create<gpu::BinaryOp>(
          gather.getLoc(), type.getElementType(), base, result,
          BinaryOperator::Add);
      sum->setAttrs(addition->getAttrs());
      result = sum;
    }
    if (gather.getValid())
      result = builder.create<gpu::SelectOp>(
          gather.getLoc(), type.getElementType(), gather.getValid(), result,
          gather.getFill());
    gather.getResult().replaceAllUsesWith(result);
    gather.erase();
    changed = true;
  }
  if (changed)
    gpu::eraseDeadPhysicalValues(kernel);
}

LogicalResult verifyKernel(func::FuncOp kernel) {
  auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
  if (!space || space.empty() || space.size() > 3)
    return kernel.emitError(
        "Triton source surface requires a one-to-three dimensional grid");
  for (Attribute extent : space)
    if (!isTritonExpression(cast<gpu::PhysicalExprAttr>(extent)))
      return kernel.emitError(
          "Triton launch expression has no deterministic terminal spelling");

  for (Type type : kernel.getArgumentTypes()) {
    if (auto view = dyn_cast<gpu::ViewType>(type)) {
      if (!isTritonScalarType(view.getElementType()))
        return kernel.emitError("Triton view element type is unsupported");
      continue;
    }
    if (!isTritonScalarType(type) ||
        isa<Float8E4M3FNType, Float8E5M2Type>(type))
      return kernel.emitError("Triton physical ABI type is unsupported");
  }

  llvm::SmallDenseSet<uint32_t> providerRoles;
  LogicalResult parameterSchema = success();
  kernel.walk([&](gpu::ParameterOp parameter) {
    uint32_t role = parameter.getParameter().getRole();
    auto category = static_cast<gpu::ParameterCategory>(
        parameter.getParameter().getCategory());
    bool providerRole =
        role == static_cast<uint32_t>(gpu::ParameterRole::ProviderWarps) ||
        role == static_cast<uint32_t>(gpu::ParameterRole::ProviderStages) ||
        role == static_cast<uint32_t>(gpu::ParameterRole::ProviderCTAs);
    if ((category == gpu::ParameterCategory::Provider) != providerRole) {
      parameter.emitOpError(
          "Triton program contains a foreign provider parameter");
      parameterSchema = failure();
      return;
    }
    if (!providerRole)
      return;
    if (!providerRoles.insert(role).second) {
      parameter.emitOpError("duplicates a Triton provider-parameter role");
      parameterSchema = failure();
      return;
    }
    if (role == static_cast<uint32_t>(gpu::ParameterRole::ProviderWarps))
      for (int64_t candidate :
           parameter.getParameter().getCandidates().asArrayRef())
        if (!llvm::isPowerOf2_64(candidate)) {
          parameter.emitOpError(
              "declares a non-power-of-two Triton num_warps candidate");
          parameterSchema = failure();
          return;
        }
  });
  if (failed(parameterSchema))
    return failure();

  WalkResult result = kernel.walk([&](Operation *operation) {
    if (auto unary = dyn_cast<gpu::UnaryOp>(operation);
        unary && unary.getApproximate() &&
        unary.getOperatorKind() == UnaryOperator::Tanh) {
      auto capabilities =
          kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
      if (!capabilities ||
          10 * capabilities.getComputeCapabilityMajor() +
                  capabilities.getComputeCapabilityMinor() < 75) {
        unary.emitOpError(
            "native approximate tanh requires compute capability 7.5 or newer");
        return WalkResult::interrupt();
      }
    }
    if (isa<gpu::AtomicLoadOp>(operation)) {
      operation->emitOpError(
          "has no Triton atomic-load primitive with preserved memory-order semantics");
      return WalkResult::interrupt();
    }
    if (auto scatter = dyn_cast<gpu::ScatterReduceOp>(operation)) {
      scatter.emitOpError(
          "requires an add combine and Triton-native atomic-add dtype legalization");
      return WalkResult::interrupt();
    }
    if (isa<gpu::BufferOp>(operation)) {
      operation->emitOpError(
          "requires provider-local mutable-buffer realization before Triton serialization");
      return WalkResult::interrupt();
    }
    if (isa<gpu::SparseContractOp>(operation)) {
      operation->emitOpError(
          "has no native Triton structured-sparse contraction surface");
      return WalkResult::interrupt();
    }
    for (Type type : operation->getResultTypes()) {
      if (auto fragment = dyn_cast<gpu::FragmentType>(type)) {
        if (!isTritonScalarType(fragment.getElementType()) ||
            !llvm::all_of(fragment.getShape(), isTritonFragmentExtent)) {
          operation->emitOpError(
              "has no statically blocked Triton fragment representation");
          return WalkResult::interrupt();
        }
      } else if (isa<gpu::RecordType>(type)) {
        if (!isTritonDataType(type)) {
          operation->emitOpError("contains a record field outside Triton data types");
          return WalkResult::interrupt();
        }
      } else if (!isa<gpu::ViewType, gpu::RangeType>(type) &&
                 !isTritonScalarType(type)) {
        operation->emitOpError("has a result type outside the Triton surface");
        return WalkResult::interrupt();
      }
    }
    if (auto load = dyn_cast<gpu::LoadOp>(operation)) {
      if (failed(verifyAccess(operation, load.getResource(),
                              load.getCoordinates(), load.getSourceAxes())))
        return WalkResult::interrupt();
    } else if (auto store = dyn_cast<gpu::StoreOp>(operation)) {
      if (failed(verifyAccess(operation, store.getResource(),
                              store.getCoordinates(), store.getSourceAxes()))) {
        return WalkResult::interrupt();
      }
    } else if (auto atomic = dyn_cast<gpu::AtomicStoreOp>(operation)) {
      if (failed(verifyAccess(operation, atomic.getResource(),
                              atomic.getCoordinates(), atomic.getSourceAxes())))
        return WalkResult::interrupt();
    } else if (auto atomic = dyn_cast<gpu::AtomicRMWOp>(operation)) {
      if (failed(verifyAccess(operation, atomic.getResource(),
                              atomic.getCoordinates(), atomic.getSourceAxes())))
        return WalkResult::interrupt();
    } else if (auto atomic =
                   dyn_cast<gpu::AtomicCompareExchangeOp>(operation)) {
      if (atomic.getValid()) {
        atomic.emitOpError(
            "Triton tl.atomic_cas has no mask and cannot preserve physical validity");
        return WalkResult::interrupt();
      }
      if (failed(verifyAccess(operation, atomic.getResource(),
                              atomic.getCoordinates(), atomic.getSourceAxes())))
        return WalkResult::interrupt();
    }
    if (auto contract = dyn_cast<gpu::ContractOp>(operation)) {
      unsigned lhsRank = contract.getLhs().getType().getShape().size();
      unsigned rhsRank = contract.getRhs().getType().getShape().size();
      SmallVector<int64_t> lhsBatch;
      SmallVector<int64_t> rhsBatch;
      for (unsigned axis = 0; axis + 2 < lhsRank; ++axis) {
        lhsBatch.push_back(axis);
        rhsBatch.push_back(axis);
      }
      if (lhsRank < 2 || rhsRank != lhsRank ||
          contract.getLhsReductionAxes() !=
              ArrayRef<int64_t>{static_cast<int64_t>(lhsRank - 1)} ||
          contract.getRhsReductionAxes() !=
              ArrayRef<int64_t>{static_cast<int64_t>(rhsRank - 2)} ||
          contract.getLhsBatchAxes() != ArrayRef<int64_t>(lhsBatch) ||
          contract.getRhsBatchAxes() != ArrayRef<int64_t>(rhsBatch)) {
        contract.emitOpError(
            "requires provider legalization to [...,M,K] x [...,K,N] tl.dot form");
        return WalkResult::interrupt();
      }
      auto form = contract->getAttrOfType<StringAttr>(contractFormAttr);
      if ((!form || form.getValue() != "multiply_sum") &&
          isa<BFloat16Type>(contract.getResult().getType().getElementType())) {
        contract.emitOpError(
            "Triton tl.dot does not support a bfloat16 accumulator; requires precision-preserving provider legalization");
        return WalkResult::interrupt();
      }
    } else if (auto range = dyn_cast<gpu::MakeRangeOp>(operation)) {
      auto fragment = dyn_cast<gpu::FragmentType>(range.getResult().getType());
      auto physicalExtent =
          fragment && fragment.getShape().size() == 1
              ? dyn_cast<gpu::PhysicalExprAttr>(fragment.getShape()[0])
              : gpu::PhysicalExprAttr();
      if (!physicalExtent || !isCompileTimeExpression(physicalExtent)) {
        InFlightDiagnostic diagnostic = range.emitOpError(
            "Triton tl.arange physical extent must be a compile-time physical expression");
        diagnostic << "; source_id=" << range.getSourceId();
        if (fragment && fragment.getAxisMaps().size() == 1)
          diagnostic << ", source_dimension="
                     << cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[0])
                            .getDimensionId();
        unsigned loopDepth = 0;
        for (Operation *parent = range->getParentOp(); parent;
             parent = parent->getParentOp())
          loopDepth += isa<scf::ForOp>(parent);
        diagnostic << ", loop_depth=" << loopDepth
                   << ", physical_extent=" << physicalExtent;
        diagnostic << (range.getResult().use_empty() ? " and is dead"
                                                      : " and still has live uses");
        for (Operation *user : range.getResult().getUsers()) {
          diagnostic << " " << user->getName();
          if (auto load = dyn_cast<gpu::LoadOp>(user)) {
            if (auto view = dyn_cast<gpu::ViewType>(load.getResource().getType()))
              diagnostic << "(abi=" << view.getAbiIndex() << ",source_axes=["
                         << load.getSourceAxes() << "])";
            for (Operation *loadUser : load.getResult().getUsers()) {
              diagnostic << "->" << loadUser->getName();
              for (Value result : loadUser->getResults())
                for (Operation *resultUser : result.getUsers()) {
                  diagnostic << "->" << resultUser->getName();
                  for (Value nextResult : resultUser->getResults())
                    for (Operation *nextUser : nextResult.getUsers()) {
                      diagnostic << "->" << nextUser->getName();
                      if (auto reduction = dyn_cast<gpu::ReduceOp>(nextUser))
                        diagnostic << "(axes=[" << reduction.getAxes()
                                   << "],sources="
                                   << reduction.getSourceCount() << ")";
                    }
                }
            }
          }
        }
        return WalkResult::interrupt();
      }
    } else if (auto gather = dyn_cast<gpu::GatherOp>(operation)) {
      if (gather.getValid()) {
        gather.emitOpError(
            "requires safe-index legalization before Triton tl.gather: source=")
            << gather.getSource().getType()
            << ", coordinate=" << gather.getCoordinates().front().getType()
            << ", valid=" << gather.getValid().getType()
            << ", fill=" << gather.getFill().getType()
            << ", result=" << gather.getResult().getType();
        return WalkResult::interrupt();
      }
      if (gather.getCoordinates().size() != 1 ||
          gather.getSourceAxes().size() != 1) {
        gather.emitOpError(
            "requires provider legalization to one-axis tl.gather");
        return WalkResult::interrupt();
      }
      if (!isa<gpu::FragmentType>(gather.getResult().getType()) &&
          (cast<gpu::FragmentType>(gather.getSource().getType()).getShape().size() != 1 ||
           gather.getSourceAxes().front() != 0 ||
           !isa<IndexType, IntegerType>(gather.getCoordinates().front().getType()))) {
        gather.emitOpError("Triton scalar gather requires one scalar index into a vector");
        return WalkResult::interrupt();
      }
    } else if (auto reduce = dyn_cast<gpu::ReduceOp>(operation)) {
      if (reduce.getAxes().size() != 1 || reduce.getCaptureCount() != 0 ||
          !reduce->hasAttr(reduceFormAttr)) {
        reduce.emitOpError(
            "Triton reduce requires a native form or an explicitly scalarized callback");
        return WalkResult::interrupt();
      }
    } else if (auto scan = dyn_cast<gpu::ScanOp>(operation)) {
      scan.emitOpError("Triton associative_scan requires an explicitly scalarized callback");
      return WalkResult::interrupt();
    } else if (auto contract = dyn_cast<gpu::ScaledContractOp>(operation)) {
      unsigned lhsRank = contract.getLhs().getType().getShape().size();
      unsigned rhsRank = contract.getRhs().getType().getShape().size();
      Type lhsScaleElement =
          contract.getLhsScale().getType().getElementType();
      Type rhsScaleElement =
          contract.getRhsScale().getType().getElementType();
      if ((lhsRank != 2 && lhsRank != 3) || rhsRank != lhsRank ||
          contract.getLhsFormat() == ScaledFormat::E8M0 ||
          contract.getRhsFormat() == ScaledFormat::E8M0 ||
          contract.getLhsGroupSize() != 32 ||
          contract.getRhsGroupSize() != 32 ||
          !lhsScaleElement.isUnsignedInteger(8) ||
          !rhsScaleElement.isUnsignedInteger(8) ||
          !contract.getResult().getType().getElementType().isF32() ||
          contract.getLhsReductionAxes() !=
              ArrayRef<int64_t>{static_cast<int64_t>(lhsRank - 1)} ||
          contract.getRhsReductionAxes() !=
              ArrayRef<int64_t>{static_cast<int64_t>(rhsRank - 2)}) {
        contract.emitOpError(
            "is outside Triton tl.dot_scaled rank/format/e8m0-scale/group/axis legality");
        return WalkResult::interrupt();
      }
    } else if (auto histogram = dyn_cast<gpu::HistogramOp>(operation)) {
      if (!histogram.getBins().getDefiningOp<arith::ConstantOp>() &&
          !histogram.getBins().getDefiningOp<gpu::ParameterOp>()) {
        histogram.emitOpError(
            "Triton histogram bin count must be compile-time bound");
        return WalkResult::interrupt();
      }
    }

    if (isa<CtaBarrierOp, TensorDescriptorChoiceOp, TensorDescriptorAllocatorOp,
            TensorDescriptorOp, BlockLoadOp, BlockStoreOp, DescriptorLoadOp,
            DescriptorStoreOp, SplitOp, ReduceOp, ScanOp, gpu::ParameterOp,
            gpu::PhysicalExprOp,
            gpu::ProgramIdOp,
            gpu::WorksetCoordinateOp, gpu::DelinearizeOp,
            gpu::DimOp, gpu::RangeOp, gpu::RangeBoundOp, gpu::MakeRangeOp,
            gpu::SplatOp, gpu::BroadcastOp, gpu::UnaryOp, gpu::BinaryOp,
            gpu::CompareOp, gpu::SelectOp, gpu::CastOp, gpu::BitcastOp,
            gpu::ReshapeOp, gpu::TransposeOp, gpu::JoinOp, gpu::MakeRecordOp,
            gpu::ExtractOp, gpu::LoadOp, gpu::GatherOp, gpu::StoreOp,
            gpu::ContractOp, gpu::ReduceOp, gpu::ScanOp,
            gpu::ScaledContractOp, gpu::HistogramOp, gpu::AtomicStoreOp,
            gpu::AtomicRMWOp, gpu::AtomicCompareExchangeOp,
            gpu::RandomBitsOp, gpu::YieldOp, arith::ConstantOp, scf::ForOp,
            scf::IfOp, scf::WhileOp, scf::ConditionOp, scf::YieldOp,
            func::FuncOp, func::ReturnOp>(operation))
      return WalkResult::advance();
    operation->emitOpError("is outside the closed Triton provider surface");
    return WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

} // namespace

LogicalResult verifyTritonProgram(ModuleOp module) {
  if (failed(mlir::verify(module.getOperation())))
    return failure();
  SmallVector<func::FuncOp> kernels;
  module.walk([&](func::FuncOp function) {
    if (function->hasAttr(gpu::kernelAttr))
      kernels.push_back(function);
  });
  if (kernels.size() != 1)
    return module.emitError("Triton provider program requires one physical kernel");
  return verifyKernel(kernels.front());
}

LogicalResult legalizeGPUProgram(ModuleOp module,
                                const gpu::TuningProfiles &profiles) {
  if (failed(gpu::verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> physicalKernel = gpu::getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  if (failed(legalizeProgramGrid(module)) ||
      failed(gpu::verifyGPUProgram(module)))
    return failure();
  foldIntegerScanTails(kernel);
  if (failed(legalizeMaskedGather(kernel)) ||
      failed(legalizeExpandingGathers(kernel)) ||
      failed(legalizeScatterAdd(kernel)) ||
      failed(legalizeContractShapes(kernel)) ||
      failed(gpu::verifyGPUProgram(module)))
    return failure();
  selectNativeReduceForms(kernel);
  bool hasWorkspace = llvm::any_of(kernel.getArgumentTypes(), [](Type type) {
    return isa<gpu::BufferType>(type);
  });
  SmallVector<gpu::ParameterCategory> categories;
  bool twoAxisPointwise = false;
  kernel.walk([&](gpu::ParameterOp parameter) {
    auto schema = parameter.getParameter();
    auto category = static_cast<gpu::ParameterCategory>(schema.getCategory());
    twoAxisPointwise |= category == gpu::ParameterCategory::Pointwise &&
                       schema.getRole() == static_cast<uint32_t>(gpu::ParameterRole::OwnershipM);
    if (category != gpu::ParameterCategory::Coverage &&
        category != gpu::ParameterCategory::Provider &&
        !llvm::is_contained(categories, category))
      categories.push_back(category);
  });
  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  bool blackwell = capabilities.getComputeCapabilityMajor() == 10 ||
                   capabilities.getComputeCapabilityMajor() == 12;
  bool hasContraction = false;
  bool allFp32 = true;
  kernel.walk([&](gpu::ContractOp contract) {
    hasContraction = true;
    allFp32 &= contract.getLhs().getType().getElementType().isF32() &&
               contract.getRhs().getType().getElementType().isF32() &&
               contract.getResult().getType().getElementType().isF32();
  });
  auto rows = profiles.get("triton", localOptionsFamily(
      categories, twoAxisPointwise, blackwell && hasRecurrentContraction(kernel),
      hasContraction && allFp32), kernel.getLoc());
  if (failed(rows))
    return failure();
  SmallVector<TritonLocalOptions> localOptions;
  SmallVector<int64_t> warpDomain, stageDomain, ctaDomain;
  for (const auto &row : *rows) {
    int64_t warps = row[0], stages = row[1], ctas = row[2];
    if (hasWorkspace && ctas != 1)
      continue;
    if ((warps & (warps - 1)) != 0 ||
        warps > capabilities.getMaxThreadsPerBlock() / 32 ||
        stages > std::numeric_limits<int32_t>::max() ||
        (ctas & (ctas - 1)) != 0 || ctas > 16 ||
        (ctas > 1 && capabilities.getComputeCapabilityMajor() < 9))
      continue;
    localOptions.push_back({warps, stages, ctas});
    if (!llvm::is_contained(warpDomain, warps)) warpDomain.push_back(warps);
    if (!llvm::is_contained(stageDomain, stages)) stageDomain.push_back(stages);
    if (!llvm::is_contained(ctaDomain, ctas)) ctaDomain.push_back(ctas);
  }
  if (localOptions.empty())
    return kernel.emitError("Triton tuning profile has no legal provider options");
  OpBuilder builder(&kernel.getBody().front(), kernel.getBody().front().begin());
  llvm::StringSet<> names;
  kernel.walk([&](gpu::ParameterOp parameter) {
    names.insert(parameter.getParameter().getName().getValue());
  });
  auto declareProviderParameter = [&](StringRef name, gpu::ParameterRole role,
                                      ArrayRef<int64_t> candidates) {
    if (names.contains(name))
      return;
    auto schema = gpu::ParameterAttr::get(
        module.getContext(), builder.getStringAttr(name),
        static_cast<uint32_t>(role),
        static_cast<uint32_t>(gpu::ParameterCategory::Provider),
        /*elementBitWidth=*/0,
        DenseI64ArrayAttr::get(module.getContext(), candidates));
    builder.create<gpu::ParameterOp>(kernel.getLoc(), builder.getIndexType(),
                                     schema);
    names.insert(name);
  };
  declareProviderParameter("NUM_WARPS", gpu::ParameterRole::ProviderWarps,
                           warpDomain);
  declareProviderParameter("NUM_STAGES", gpu::ParameterRole::ProviderStages,
                           stageDomain);
  declareProviderParameter("NUM_CTAS", gpu::ParameterRole::ProviderCTAs,
                           ctaDomain);
  if (failed(gpu::verifyGPUProgram(module)) ||
      failed(lowerInvocationWorkspaces(module)) ||
      failed(legalizeSplitGatherPairs(kernel)) ||
      failed(materializeBlockPointerForms(kernel)))
    return failure();
  FailureOr<TensorDescriptorChoiceOp> tensorDescriptorForms =
      materializeTensorDescriptorForms(kernel, localOptions);
  if (failed(tensorDescriptorForms) ||
      failed(materializeLegalConfigs(kernel, *tensorDescriptorForms, localOptions)))
    return failure();
  selectContractForms(kernel);
  SmallVector<gpu::AssumeInBoundsOp> boundsAssumptions;
  kernel.walk([&](gpu::AssumeInBoundsOp assumption) {
    boundsAssumptions.push_back(assumption);
  });
  for (gpu::AssumeInBoundsOp assumption : boundsAssumptions)
    assumption.erase();
  if (failed(gpu::eliminateCommonValues(module)) ||
      failed(legalizeCollectiveCallbacks(kernel)) ||
      failed(verifyTritonProgram(module)))
    return failure();
  kernel->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::triton
