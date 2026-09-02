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
constexpr int64_t maxTritonTensorElements = 1048576;

struct TritonConfig {
  std::map<std::string, int64_t> kernelParameters;
  int64_t warps = 0;
  int64_t stages = 0;
  int64_t ctas = 0;
};

int64_t selectProviderCandidate(int64_t requested,
                                ArrayRef<int64_t> candidates) {
  int64_t selected = *std::min_element(candidates.begin(), candidates.end());
  for (int64_t candidate : candidates) {
    if (candidate == requested)
      return candidate;
    if (candidate <= requested && candidate > selected)
      selected = candidate;
  }
  return selected;
}

struct TritonLocalOptions {
  int64_t warps;
  int64_t stages;
  int64_t ctas;
};

SmallVector<TritonLocalOptions, 6>
localOptionsFor(ArrayRef<gpu::ParameterCategory> categories,
                bool twoAxisPointwise,
                bool blackwellRecurrentContraction) {
  if (llvm::is_contained(categories,
                         gpu::ParameterCategory::RegionReduction))
    return {{32, 2, 1}, {16, 2, 1}, {8, 2, 1}};
  if (llvm::is_contained(categories,
                         gpu::ParameterCategory::RegionContraction))
    return {{4, 2, 1}, {8, 2, 1}, {4, 3, 1},
            {8, 3, 1}, {4, 4, 1}, {8, 4, 1}};
  if (llvm::is_contained(categories,
                         gpu::ParameterCategory::PersistentContraction)) {
    if (blackwellRecurrentContraction)
      return {{2, 1, 1}, {4, 1, 1}, {8, 1, 1}};
    return {{4, 3, 1}, {8, 3, 1}, {4, 4, 1}};
  }
  if (llvm::is_contained(categories, gpu::ParameterCategory::Contraction))
    return {{4, 2, 1}, {8, 3, 1}, {4, 4, 1}};
  if (llvm::is_contained(categories, gpu::ParameterCategory::Histogram))
    return {{16, 2, 1}, {32, 2, 1}, {8, 2, 1}};
  if (llvm::is_contained(categories, gpu::ParameterCategory::Reduction) ||
      llvm::is_contained(categories, gpu::ParameterCategory::Scan))
    return {{8, 2, 1}, {4, 2, 1}, {16, 2, 1}};
  if (twoAxisPointwise)
    return {{4, 2, 1}, {8, 2, 1}, {4, 1, 1}, {2, 5, 1}};
  return {{1, 3, 1}, {2, 1, 1}, {2, 2, 1}, {2, 3, 1},
          {4, 2, 1}, {8, 2, 1}, {4, 1, 1}, {4, 3, 1}};
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

bool collectBoundaryAxes(Value predicate, ValueRange coordinates,
                         ArrayRef<int64_t> sourceAxes,
                         llvm::SmallDenseSet<int64_t, 4> &axes) {
  if (isTrueValue(predicate))
    return true;
  if (auto broadcast = predicate.getDefiningOp<gpu::BroadcastOp>())
    return collectBoundaryAxes(broadcast.getValue(), coordinates, sourceAxes,
                               axes);
  if (auto reshape = predicate.getDefiningOp<gpu::ReshapeOp>())
    return collectBoundaryAxes(reshape.getValue(), coordinates, sourceAxes,
                               axes);
  if (auto binary = predicate.getDefiningOp<gpu::BinaryOp>()) {
    if (binary.getOperatorKind() != BinaryOperator::LogicalAnd &&
        binary.getOperatorKind() != BinaryOperator::BitwiseAnd)
      return false;
    return collectBoundaryAxes(binary.getLhs(), coordinates, sourceAxes, axes) &&
           collectBoundaryAxes(binary.getRhs(), coordinates, sourceAxes, axes);
  }
  auto compare = predicate.getDefiningOp<gpu::CompareOp>();
  if (!compare || !compare->hasAttr(gpu::physicalTailAttr) ||
      compare.getPredicate() != ComparePredicate::Lt)
    return false;
  for (auto [index, coordinate] : llvm::enumerate(coordinates))
    if (compare.getLhs() == coordinate &&
        isa<gpu::FragmentType>(coordinate.getType())) {
      axes.insert(sourceAxes[index]);
      return true;
    }
  return false;
}

std::optional<BlockAccessPlan>
planBlockAccess(Value resource, ValueRange coordinates,
                ArrayRef<int64_t> sourceAxes, Type valueType, Value valid,
                Value fill) {
  auto view = dyn_cast<gpu::ViewType>(resource.getType());
  auto fragment = dyn_cast<gpu::FragmentType>(valueType);
  if (!view || !fragment || !isa<BlockArgument>(resource) ||
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

  llvm::SmallDenseSet<int64_t, 4> boundary;
  if (valid) {
    if (fill && !isZeroValue(fill))
      return std::nullopt;
    if (!collectBoundaryAxes(valid, coordinates, sourceAxes, boundary))
      return std::nullopt;
  } else if (fill) {
    return std::nullopt;
  }
  for (int64_t viewAxis : boundary) {
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
  kernel.walk([&](gpu::LoadOp load) {
    if (std::optional<BlockAccessPlan> plan = planBlockAccess(
            load.getResource(), load.getCoordinates(), load.getSourceAxes(),
            load.getResult().getType(), load.getValid(), load.getFill()))
      loads.emplace_back(load, std::move(*plan));
  });
  kernel.walk([&](gpu::StoreOp store) {
    if (std::optional<BlockAccessPlan> plan = planBlockAccess(
            store.getResource(), store.getCoordinates(), store.getSourceAxes(),
            store.getValue().getType(), store.getValid(), Value()))
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
  for (unsigned axis = 0; axis + 1 < view.getRank(); ++axis) {
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
  for (unsigned axis = 0; axis + 1 < view.getRank(); ++axis) {
    auto stride = dyn_cast<IntegerAttr>(strides[axis]);
    if (stride && (stride.getInt() * *elementBytes) % 16 != 0)
      return false;
  }
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
    if (name && name.getValue() == symbol.getValue() && argument.getType().isIndex())
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

FailureOr<bool> materializeTensorDescriptorForms(func::FuncOp kernel) {
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities || capabilities.getComputeCapabilityMajor() < 9)
    return false;
  bool hasContraction = false;
  kernel.walk([&](Operation *operation) {
    hasContraction |= isa<gpu::ContractOp, gpu::ScaledContractOp>(operation);
  });
  if (!hasContraction)
    return false;

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
    return false;

  OpBuilder entry(&kernel.getBody().front(), kernel.getBody().front().begin());
  entry.create<TensorDescriptorAllocatorOp>(
      kernel.getLoc(), entry.getI64IntegerAttr(0), entry.getI64IntegerAttr(1),
      entry.getI64IntegerAttr(2), entry.getStringAttr("launch"));
  auto choice = entry.create<TensorDescriptorChoiceOp>(
      kernel.getLoc(), entry.getI1Type(), entry.getStringAttr("host"));
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
    SmallVector<Value> strides{dimensions.back(), one};
    SmallVector<Value> descriptorBlockShape;
    for (Attribute extent : fragment.getShape())
      descriptorBlockShape.push_back(entry.create<gpu::PhysicalExprOp>(
          kernel.getLoc(), entry.getIndexType(),
          cast<gpu::PhysicalExprAttr>(extent)));
    std::optional<int64_t> elementBytes =
        descriptorElementBytes(view.getElementType());
    if (!elementBytes || 16 % *elementBytes != 0)
      return failure();
    auto descriptor = entry.create<TensorDescriptorOp>(
        kernel.getLoc(), viewValue.getType(), viewValue, shape, strides,
        descriptorBlockShape,
        DenseI64ArrayAttr::get(kernel.getContext(), blockAxes),
        DenseI64ArrayAttr::get(kernel.getContext(),
                               ArrayRef<int64_t>{1, 16 / *elementBytes}),
        entry.getStringAttr("contiguous"), entry.getStringAttr("zero"),
        entry.getI64IntegerAttr(16), entry.getI64IntegerAttr(16));
    descriptors.push_back(
        {viewValue, fragment,
         SmallVector<int64_t>(blockAxes.begin(), blockAxes.end()), descriptor});
    return descriptor;
  };
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
  return true;
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
                                      bool hasTensorDescriptorForms) {
  struct Domain {
    StringRef name;
    gpu::ParameterRole role;
    ArrayRef<int64_t> candidates;
    bool coverage;
  };
  SmallVector<Domain> domains;
  SmallVector<gpu::ParameterCategory> categories;
  bool twoAxisPointwise = false;
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
    auto category =
        static_cast<gpu::ParameterCategory>(definition.getCategory());
    twoAxisPointwise |=
        category == gpu::ParameterCategory::Pointwise &&
        static_cast<gpu::ParameterRole>(definition.getRole()) ==
            gpu::ParameterRole::OwnershipM;
    if (category != gpu::ParameterCategory::Coverage &&
        category != gpu::ParameterCategory::Provider &&
        !llvm::is_contained(categories, category))
      categories.push_back(category);
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
    else if (domain.role == gpu::ParameterRole::ProviderThreads)
      return kernel.emitError(
          "Triton program contains a foreign provider parameter");
  }
  if (!warps || !stages || !ctas)
    return kernel.emitError("Triton provider parameter domains are incomplete");
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  bool blackwell = capabilities &&
                   (capabilities.getComputeCapabilityMajor() == 10 ||
                    capabilities.getComputeCapabilityMajor() == 12);
  SmallVector<TritonLocalOptions, 4> localOptions =
      localOptionsFor(categories, twoAxisPointwise,
                      blackwell && hasRecurrentContraction(kernel));
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
      int64_t formCount = hasTensorDescriptorForms ? 2 : 1;
      for (int64_t form = 0; form < formCount; ++form) {
        TritonConfig config = sharedConfig;
        if (hasTensorDescriptorForms)
          config.kernelParameters[tensorDescriptorChoice.str()] = form;
        config.warps = selectProviderCandidate(options.warps, warps->candidates);
        config.stages =
            selectProviderCandidate(options.stages, stages->candidates);
        config.ctas = selectProviderCandidate(options.ctas, ctas->candidates);
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
        hasTensorDescriptorForms &&
        config.kernelParameters.at(tensorDescriptorChoice.str()) != 0;
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

bool isAddCombine(gpu::ScatterReduceOp scatter) {
  return gpu::queryBinaryCombineKind(scatter.getCombine()) ==
         BinaryOperator::Add;
}

bool isNativeAddReduce(gpu::ReduceOp reduce) {
  if (reduce.getSourceCount() != 1 || reduce.getIdentityCount() != 1 ||
      reduce.getCaptureCount() != 0 || reduce.getResults().size() != 1 ||
      reduce.getCombine().empty() || reduce.getCombine().getBlocks().size() != 1)
    return false;
  return gpu::queryBinaryCombineKind(reduce.getCombine()) ==
         BinaryOperator::Add;
}

void selectNativeReduceForms(func::FuncOp kernel) {
  kernel.walk([&](gpu::ReduceOp reduce) {
    if (isNativeAddReduce(reduce))
      reduce->setAttr(reduceFormAttr,
                      StringAttr::get(kernel.getContext(), "sum"));
  });
}

void selectContractForms(func::FuncOp kernel) {
  kernel.walk([&](gpu::ContractOp contract) {
    auto lhs = contract.getLhs().getType();
    if (lhs.getShape().empty())
      return;
    auto reductionExtent =
        dyn_cast<gpu::PhysicalExprAttr>(
            lhs.getShape()[lhs.getShape().size() - 1]);
    if (!reductionExtent ||
        reductionExtent.getKind() !=
            static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) ||
        reductionExtent.getValue() >= 16)
      return;
    contract->setAttr(contractFormAttr,
                      StringAttr::get(kernel.getContext(), "multiply_sum"));
  });
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
    if (!isTritonScalarType(type))
      return kernel.emitError("Triton physical ABI type is unsupported");
  }

  llvm::SmallDenseSet<uint32_t> providerRoles;
  LogicalResult parameterSchema = success();
  kernel.walk([&](gpu::ParameterOp parameter) {
    uint32_t role = parameter.getParameter().getRole();
    bool providerRole =
        role == static_cast<uint32_t>(gpu::ParameterRole::ProviderWarps) ||
        role == static_cast<uint32_t>(gpu::ParameterRole::ProviderStages) ||
        role == static_cast<uint32_t>(gpu::ParameterRole::ProviderCTAs);
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
    } else if (auto reduce = dyn_cast<gpu::ReduceOp>(operation)) {
      if (reduce.getAxes().size() != 1 || reduce.getCaptureCount() != 0) {
        reduce.emitOpError(
            "Triton reduce requires one physical axis and capture-free helper");
        return WalkResult::interrupt();
      }
    } else if (auto scan = dyn_cast<gpu::ScanOp>(operation)) {
      if (!scan.getInclusive() || scan.getCaptureCount() != 0) {
        scan.emitOpError(
            "Triton associative_scan requires inclusive capture-free physical form");
        return WalkResult::interrupt();
      }
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

    if (isa<TensorDescriptorChoiceOp, TensorDescriptorAllocatorOp,
            TensorDescriptorOp, BlockLoadOp, BlockStoreOp, DescriptorLoadOp,
            DescriptorStoreOp, SplitOp, gpu::ParameterOp,
            gpu::PhysicalExprOp,
            gpu::ProgramIdOp,
            gpu::WorksetCoordinateOp, gpu::DelinearizeOp,
            gpu::DimOp, gpu::RangeOp, gpu::RangeBoundOp, gpu::MakeRangeOp,
            gpu::SplatOp, gpu::BroadcastOp, gpu::UnaryOp, gpu::BinaryOp,
            gpu::CompareOp, gpu::SelectOp, gpu::CastOp, gpu::BitcastOp,
            gpu::ReshapeOp, gpu::TransposeOp, gpu::JoinOp, gpu::MakeRecordOp,
            gpu::ExtractOp, gpu::LoadOp, gpu::GatherOp,
            gpu::AssumeInBoundsOp, gpu::StoreOp,
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

LogicalResult legalizeGPUProgram(ModuleOp module) {
  if (failed(gpu::verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> physicalKernel = gpu::getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  if (failed(legalizeProgramGrid(module)) ||
      failed(gpu::verifyGPUProgram(module)))
    return failure();
  if (failed(legalizeMaskedGather(kernel)) ||
      failed(legalizeScatterAdd(kernel)) ||
      failed(gpu::verifyGPUProgram(module)))
    return failure();
  selectNativeReduceForms(kernel);
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
                           ArrayRef<int64_t>{1, 2, 4, 8, 16, 32});
  declareProviderParameter("NUM_STAGES", gpu::ParameterRole::ProviderStages,
                           ArrayRef<int64_t>{1, 2, 3, 4, 5, 6});
  declareProviderParameter("NUM_CTAS", gpu::ParameterRole::ProviderCTAs,
                           ArrayRef<int64_t>{1});
  if (failed(gpu::verifyGPUProgram(module)) ||
      failed(legalizeSplitGatherPairs(kernel)) ||
      failed(materializeBlockPointerForms(kernel)))
    return failure();
  FailureOr<bool> tensorDescriptorForms =
      materializeTensorDescriptorForms(kernel);
  if (failed(tensorDescriptorForms) ||
      failed(materializeLegalConfigs(kernel, *tensorDescriptorForms)))
    return failure();
  selectContractForms(kernel);
  if (failed(verifyTritonProgram(module)))
    return failure();
  kernel->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::triton
