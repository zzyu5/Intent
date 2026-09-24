#include "Intent/Target/Triton/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>

using namespace mlir;

namespace intent::triton {
namespace {

constexpr llvm::StringLiteral legalizedAttr = "intent_gpu.triton.legalized";
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
      kind == gpu::PhysicalExprKind::Parameter ||
      kind == gpu::PhysicalExprKind::Dimension)
    return true;
  if (kind == gpu::PhysicalExprKind::ScalarABI)
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
  Attribute constant =
      UniformValueAnalysis(gpu::describeUniformValue).evaluate(value);
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant))
    return integer.getValue().isZero();
  // Native zero padding produces positive floating zero.
  if (auto floating = dyn_cast<FloatAttr>(constant))
    return floating.getValue().isZero() && !floating.getValue().isNegative();
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
    if (!coordinateType)
      return std::nullopt;
    auto projection = gpu::queryBroadcastProjection(coordinateType, fragment);
    if (!projection.isExact())
      return std::nullopt;
    // Broadcasting a Cartesian range does not make the access indirect.
    // Compose the actual projections rather than matching equal extents or
    // dropping a reshape that might change the coordinate's varying axis.
    while (auto broadcast = coordinate.getDefiningOp<gpu::BroadcastOp>()) {
      auto source = dyn_cast<gpu::FragmentType>(broadcast.getValue().getType());
      if (!source)
        return std::nullopt;
      auto step = gpu::queryBroadcastProjection(source, coordinateType);
      if (!step.isExact())
        return std::nullopt;
      for (std::optional<unsigned> &axis : projection.targetToSource)
        if (axis)
          axis = step.targetToSource[*axis];
      coordinate = broadcast.getValue();
      coordinateType = source;
    }
    auto range = coordinate.getDefiningOp<gpu::MakeRangeOp>();
    if (coordinateType.getShape().size() != 1 ||
        !coordinateType.getElementType().isIndex() || !range ||
        !isUnitStep(range.getStep()))
      return std::nullopt;
    std::optional<unsigned> selected;
    for (auto [fragmentAxis, sourceAxis] :
         llvm::enumerate(projection.targetToSource)) {
      if (sourceAxis) {
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
  llvm::append_range(plan.boundaryAxes, boundaryFact.boundaryAxes);
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
  auto projectBoundaries = [](BlockAccessPlan &plan) {
    for (int64_t &axis : plan.boundaryAxes) {
      auto found = llvm::find(plan.blockAxes, axis);
      if (found == plan.blockAxes.end())
        return false;
      axis = std::distance(plan.blockAxes.begin(), found);
    }
    llvm::sort(plan.boundaryAxes);
    return true;
  };
  kernel.walk([&](gpu::LoadOp load) {
    gpu::PhysicalAccessBoundaryFact boundary =
        analysis.boundaryValidity(load);
    if (std::optional<BlockAccessPlan> plan = planBlockAccess(
            load.getResource(), load.getCoordinates(), load.getSourceAxes(),
            load.getResult().getType(), load.getValid(), load.getFill(),
            boundary);
        plan && projectBoundaries(*plan))
      loads.emplace_back(load, std::move(*plan));
  });
  kernel.walk([&](gpu::StoreOp store) {
    gpu::PhysicalAccessBoundaryFact boundary =
        analysis.boundaryValidity(store);
    if (std::optional<BlockAccessPlan> plan = planBlockAccess(
            store.getResource(), store.getCoordinates(), store.getSourceAxes(),
            store.getValue().getType(), store.getValid(), Value(), boundary);
        plan && projectBoundaries(*plan))
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

void orientPointerLoads(func::FuncOp kernel) {
  llvm::DenseMap<Operation *, SmallVector<gpu::ContractOp>> simtRhsConsumers;
  kernel.walk([&](gpu::ContractOp contract) {
    if (!gpu::uniformElementType(contract.getLhs().getType()).isF32() ||
        !gpu::uniformElementType(contract.getRhs().getType()).isF32())
      return;
    SmallVector<Value> pending{contract.getRhs()};
    llvm::DenseSet<Operation *> visited;
    while (!pending.empty()) {
      Operation *producer = pending.pop_back_val().getDefiningOp();
      if (!producer || !visited.insert(producer).second)
        continue;
      if (auto load = dyn_cast<gpu::LoadOp>(producer)) {
        simtRhsConsumers[load].push_back(contract);
        continue;
      }
      if (isa<gpu::BroadcastOp, gpu::ReshapeOp, gpu::TransposeOp, gpu::CastOp,
              gpu::UnaryOp, gpu::BinaryOp, gpu::SelectOp>(producer))
        llvm::append_range(pending, producer->getOperands());
    }
  });
  SmallVector<gpu::LoadOp> loads;
  kernel.walk([&](gpu::LoadOp load) {
    if (isa<gpu::ViewType>(load.getResource().getType()) &&
        isa<gpu::FragmentType>(load.getResult().getType()))
      loads.push_back(load);
  });
  for (gpu::LoadOp load : loads) {
    auto result = cast<gpu::FragmentType>(load.getResult().getType());
    unsigned rank = result.getShape().size();
    if (rank < 2)
      continue;
    gpu::PhysicalProgramAnalysis analysis(kernel);
    SmallVector<int64_t> viewAxes(rank, -1);
    Value innermostCoordinate;
    int64_t innermostViewAxis = -1;
    bool exact = true;
    for (auto [position, coordinate] : llvm::enumerate(load.getCoordinates())) {
      auto type = dyn_cast<gpu::FragmentType>(coordinate.getType());
      if (!type)
        continue;
      auto projection = gpu::queryAxisProjection(type, result);
      if (!projection.isExact()) {
        exact = false;
        break;
      }
      unsigned varying = 0;
      for (auto [axis, source] : llvm::enumerate(projection.targetToSource)) {
        if (!source)
          continue;
        auto extent = cast<gpu::PhysicalExprAttr>(type.getShape()[*source]);
        if (extent.getKind() == static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) &&
            extent.getValue() == 1)
          continue;
        auto ranges = analysis.axisRanges(coordinate, *source);
        if (!ranges.isExact() || !ranges.blockers.empty() ||
            !ranges.accesses.empty()) {
          exact = false;
          break;
        }
        if (ranges.roots.empty())
          continue;
        if (++varying != 1) {
          exact = false;
          break;
        }
        // A broadcast axis does not vary. Reshape-derived coordinates may
        // share one varying fragment axis across several resource axes.
        viewAxes[axis] = std::max(viewAxes[axis], load.getSourceAxes()[position]);
        if (viewAxes[axis] > innermostViewAxis) {
          innermostViewAxis = viewAxes[axis];
          innermostCoordinate = coordinate;
        }
      }
      if (!exact)
        break;
    }
    // Leave a unit-step innermost view coordinate to native coalescing. A
    // unit-step coordinate on an outer view axis can still be strided.
    auto view = cast<gpu::ViewType>(load.getResource().getType());
    if (innermostCoordinate && innermostViewAxis + 1 == view.getRank()) {
      auto range = gpu::stripAdditiveProjection(innermostCoordinate)
                       .getDefiningOp<gpu::MakeRangeOp>();
      if (range && gpu::isUnitStepRange(range))
        continue;
    }
    auto sameSchema = [&](Type element) {
      return gpu::FragmentType::get(kernel.getContext(), element,
          result.getShape(), result.getAxisMaps(), result.getValidity(),
          result.getOwner());
    };
    for (Value operand : load->getOperands())
      if (auto fragment = dyn_cast<gpu::FragmentType>(operand.getType()))
        exact &= gpu::queryBroadcastProjection(
            fragment, sameSchema(fragment.getElementType())).isExact();
    if (!exact)
      continue;
    SmallVector<int64_t> permutation;
    for (unsigned axis = 0; axis < rank; ++axis)
      permutation.push_back(axis);
    llvm::stable_sort(permutation, [&](int64_t lhs, int64_t rhs) {
      return viewAxes[lhs] > viewAxes[rhs];
    });
    if (llvm::all_of(llvm::enumerate(permutation), [](auto entry) {
          return static_cast<int64_t>(entry.index()) == entry.value();
        }))
      continue;
    // IEEE f32 dot uses SIMT FMA. Its RHS shared tile has no native swizzle;
    // making K the leading load axis can serialize the free-axis reads.
    auto leadingRanges = analysis.axisRanges(load.getResult(), permutation.front());
    bool reductionLeading = llvm::any_of(simtRhsConsumers.lookup(load),
        [&](gpu::ContractOp contract) {
          if (!leadingRanges.isExact() || leadingRanges.roots.empty())
            return false;
          auto axes = analysis.rangeAxes(contract.getRhs(), leadingRanges.roots);
          return axes.isExact() &&
                 llvm::any_of(axes.fragmentAxes, [&](unsigned axis) {
                   return llvm::is_contained(contract.getRhsReductionAxes(), axis);
                 });
        });
    if (reductionLeading)
      continue;
    OpBuilder builder(load);
    auto permutedType = [&](gpu::FragmentType type, ArrayRef<int64_t> order) {
      SmallVector<Attribute> shape, mappings;
      for (auto [position, axis] : llvm::enumerate(order)) {
        shape.push_back(type.getShape()[axis]);
        auto mapping = cast<gpu::AxisMapAttr>(type.getAxisMaps()[axis]);
        mappings.push_back(gpu::AxisMapAttr::get(kernel.getContext(),
            mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), position, mapping.getDerived()));
      }
      return gpu::FragmentType::get(kernel.getContext(),
          type.getElementType(), builder.getArrayAttr(shape),
          builder.getArrayAttr(mappings), type.getValidity(), type.getOwner());
    };
    auto transpose = [&](Value value, ArrayRef<int64_t> order) -> Value {
      auto target = permutedType(cast<gpu::FragmentType>(value.getType()), order);
      return builder.create<gpu::TransposeOp>(load.getLoc(), target, value, order);
    };
    IRMapping mapping;
    for (Value operand : load->getOperands()) {
      auto type = dyn_cast<gpu::FragmentType>(operand.getType());
      if (!type || mapping.contains(operand))
        continue;
      auto target = sameSchema(type.getElementType());
      Value expanded = operand;
      if (type != target)
        expanded = builder.create<gpu::BroadcastOp>(load.getLoc(), target, operand);
      mapping.map(operand, transpose(expanded, permutation));
    }
    auto oriented = cast<gpu::LoadOp>(builder.clone(*load, mapping));
    oriented.getResult().setType(permutedType(result, permutation));
    SmallVector<int64_t> inverse(rank);
    for (auto [position, axis] : llvm::enumerate(permutation))
      inverse[axis] = position;
    Value restored = transpose(oriented.getResult(), inverse);
    load.getResult().replaceAllUsesWith(restored);
    load.erase();
  }
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

bool descriptorOffsetAligned(Value value, int64_t alignment,
                             StringAttr alignedBlockParameter,
                             unsigned depth = 0) {
  if (!value || !value.getType().isIndex() || depth >= 32)
    return false;
  if (alignment == 1)
    return true;
  auto constant = dyn_cast_or_null<IntegerAttr>(
      UniformValueAnalysis(gpu::describeUniformValue).evaluate(value));
  if (constant)
    return constant.getInt() % alignment == 0;
  gpu::ParameterOp parameter = value.getDefiningOp<gpu::ParameterOp>();
  if (auto physical = value.getDefiningOp<gpu::PhysicalExprOp>();
      physical && physical.getExpression().getKind() ==
          static_cast<uint32_t>(gpu::PhysicalExprKind::Parameter)) {
    auto resolved = gpu::queryParameterBySymbol(
        physical->getParentOfType<func::FuncOp>(),
        physical.getExpression().getSymbol());
    if (succeeded(resolved))
      parameter = *resolved;
  }
  if (parameter) {
    if (parameter.getParameter().getName() == alignedBlockParameter)
      return true;
    auto candidates = parameter.getParameter().getCandidates().asArrayRef();
    return !candidates.empty() && llvm::all_of(candidates, [&](int64_t candidate) {
      return candidate % alignment == 0;
    });
  }
  auto aligned = [&](Value operand) {
    return descriptorOffsetAligned(operand, alignment, alignedBlockParameter,
                                   depth + 1);
  };
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    return loop && argument == loop.getInductionVar() &&
           aligned(loop.getLowerBound()) && aligned(loop.getStep());
  }
  if (auto select = value.getDefiningOp<gpu::SelectOp>())
    return aligned(select.getTrueValue()) && aligned(select.getFalseValue());
  if (auto binary = value.getDefiningOp<gpu::BinaryOp>()) {
    if (binary.getOperatorKind() == BinaryOperator::Multiply)
      return aligned(binary.getLhs()) || aligned(binary.getRhs());
    if (binary.getOperatorKind() == BinaryOperator::Add ||
        binary.getOperatorKind() == BinaryOperator::Subtract)
      return aligned(binary.getLhs()) && aligned(binary.getRhs());
  }
  return false;
}

bool descriptorAccessEligible(func::FuncOp kernel, Value viewValue,
                              ValueRange offsets,
                              ArrayRef<int64_t> blockAxes,
                              gpu::FragmentType fragment) {
  auto view = cast<gpu::ViewType>(viewValue.getType());
  unsigned blockRank = fragment.getShape().size();
  if (view.getRank() < 1 || view.getRank() > 5 || blockRank == 0 ||
      !llvm::is_contained(blockAxes, static_cast<int64_t>(view.getRank()) - 1))
    return false;
  std::optional<int64_t> elementBytes =
      descriptorElementBytes(view.getElementType());
  auto layout = view.getLayout();
  if (!elementBytes || 16 % *elementBytes != 0 || !layout.getHasStrides() ||
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
    if (stride && (stride.getInt() * *elementBytes) % 16 != 0)
      return false;
  }
  // TMA requires aligned coordinates, not constant coordinates. Tile starts
  // and reduction-loop induction values preserve power-of-two divisibility,
  // including under fixed-width index arithmetic.
  // Descriptor configs already require a power-of-two contiguous block of
  // at least 16 bytes. Its extent parameter is aligned in this branch even
  // when the shared parameter domain also includes smaller pointer tiles.
  unsigned contiguousAxis = std::distance(
      blockAxes.begin(), llvm::find(blockAxes, view.getRank() - 1));
  auto extent =
      cast<gpu::PhysicalExprAttr>(fragment.getShape()[contiguousAxis]);
  StringAttr alignedBlockParameter;
  if (extent.getKind() ==
      static_cast<uint32_t>(gpu::PhysicalExprKind::Parameter))
    alignedBlockParameter = extent.getSymbol();
  return descriptorOffsetAligned(offsets.back(), 16 / *elementBytes,
                                  alignedBlockParameter);
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
    OpBuilder &builder, Location location, ValueRange offsets) {
  auto nativeOffsets = [&](ValueRange coordinates) {
    // Descriptor shapes are at most INT32_MAX and blocks at most 2^20
    // elements. A start outside signed i32 therefore denotes an entirely
    // padded block; preserve that fact instead of wrapping it into the view.
    Value minimum = builder.create<arith::ConstantIndexOp>(
        location, std::numeric_limits<int32_t>::min());
    Value maximum = builder.create<arith::ConstantIndexOp>(
        location, std::numeric_limits<int32_t>::max());
    SmallVector<Value> result;
    for (Value offset : coordinates) {
      Value lower = builder.create<gpu::CompareOp>(
          location, builder.getI1Type(), offset, minimum, ComparePredicate::Ge);
      Value upper = builder.create<gpu::CompareOp>(
          location, builder.getI1Type(), offset, maximum, ComparePredicate::Le);
      Value inside = builder.create<gpu::BinaryOp>(
          location, builder.getI1Type(), lower, upper, BinaryOperator::LogicalAnd);
      result.push_back(builder.create<gpu::SelectOp>(
          location, builder.getIndexType(), inside, offset, minimum));
    }
    return result;
  };
  return nativeOffsets(offsets);
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

  SmallVector<std::pair<gpu::LoadOp, BlockAccessPlan>, 4> loads;
  SmallVector<std::pair<gpu::StoreOp, BlockAccessPlan>, 4> stores;
  gpu::PhysicalProgramAnalysis analysis(kernel);
  kernel.walk([&](gpu::LoadOp load) {
    auto plan = planBlockAccess(
        load.getResource(), load.getCoordinates(), load.getSourceAxes(),
        load.getResult().getType(), load.getValid(), load.getFill(),
        analysis.boundaryValidity(load));
    if (plan && descriptorAccessEligible(
                    kernel, load.getResource(), plan->offsets, plan->blockAxes,
                    cast<gpu::FragmentType>(load.getResult().getType())))
      loads.emplace_back(load, std::move(*plan));
  });
  kernel.walk([&](gpu::StoreOp store) {
    auto plan = planBlockAccess(
        store.getResource(), store.getCoordinates(), store.getSourceAxes(),
        store.getValue().getType(), store.getValid(), Value(),
        analysis.boundaryValidity(store));
    if (plan && descriptorAccessEligible(
                    kernel, store.getResource(), plan->offsets, plan->blockAxes,
                    cast<gpu::FragmentType>(store.getValue().getType())))
      stores.emplace_back(store, std::move(*plan));
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
    gpu::FragmentType orderedFragment;
    gpu::FragmentType nativeFragment;
    SmallVector<int64_t> blockAxes;
    SmallVector<int64_t> permutation;
    TensorDescriptorOp descriptor;
  };
  SmallVector<DescriptorPlan> descriptors;
  auto [nextSource, nextDimension] = gpu::nextPhysicalAxisIdentities(kernel);
  auto descriptorFor = [&](Value viewValue, gpu::FragmentType fragment,
                           ArrayRef<int64_t> blockAxes)
      -> FailureOr<DescriptorPlan> {
    for (const DescriptorPlan &plan : descriptors)
      if (plan.view == viewValue && plan.fragment == fragment &&
          ArrayRef<int64_t>(plan.blockAxes) == blockAxes)
        return plan;
    auto view = cast<gpu::ViewType>(viewValue.getType());
    SmallVector<Value> dimensions;
    for (unsigned axis = 0; axis < view.getRank(); ++axis)
      dimensions.push_back(entry.create<gpu::DimOp>(
          kernel.getLoc(), entry.getIndexType(), viewValue, axis));
    SmallVector<Value> strides;
    SmallVector<int64_t> alignedAxes;
    for (unsigned axis = 0; axis + 1 < view.getRank(); ++axis) {
      auto stride = descriptorStrideValue(entry, kernel, kernel.getLoc(), view,
                                          axis);
      if (failed(stride))
        return failure();
      strides.push_back(*stride);
      alignedAxes.push_back(axis);
    }
    strides.push_back(entry.create<arith::ConstantIndexOp>(kernel.getLoc(), 1));
    SmallVector<Attribute> shape, mappings, orderedShape, orderedMappings;
    SmallVector<int64_t> permutation;
    for (unsigned axis = 0; axis < view.getRank(); ++axis) {
      auto found = llvm::find(blockAxes, axis);
      if (found != blockAxes.end()) {
        unsigned fragmentAxis = std::distance(blockAxes.begin(), found);
        shape.push_back(fragment.getShape()[fragmentAxis]);
        auto mapping = cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
        mappings.push_back(gpu::AxisMapAttr::get(
            kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), axis, mapping.getDerived()));
        orderedShape.push_back(fragment.getShape()[fragmentAxis]);
        orderedMappings.push_back(gpu::AxisMapAttr::get(
            kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), permutation.size(), mapping.getDerived()));
        permutation.push_back(fragmentAxis);
      } else {
        shape.push_back(gpu::PhysicalExprAttr::get(
            kernel.getContext(),
            static_cast<uint32_t>(gpu::PhysicalExprKind::Constant), 1,
            entry.getStringAttr(""), entry.getArrayAttr({})));
        mappings.push_back(gpu::AxisMapAttr::get(
            kernel.getContext(), nextSource++, 0, nextDimension++, axis, true));
      }
    }
    auto nativeFragment = gpu::FragmentType::get(
        kernel.getContext(), fragment.getElementType(), entry.getArrayAttr(shape),
        entry.getArrayAttr(mappings), fragment.getValidity(), fragment.getOwner());
    auto orderedFragment = gpu::FragmentType::get(
        kernel.getContext(), fragment.getElementType(),
        entry.getArrayAttr(orderedShape), entry.getArrayAttr(orderedMappings),
        fragment.getValidity(), fragment.getOwner());
    SmallVector<Value> descriptorBlockShape;
    for (Attribute extent : nativeFragment.getShape())
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
    SmallVector<int64_t> initialBlockShape(view.getRank(), 1);
    initialBlockShape.back() = 16 / *elementBytes;
    auto descriptor = entry.create<TensorDescriptorOp>(
        kernel.getLoc(), viewValue.getType(), viewValue, dimensions, strides,
        descriptorBlockShape, blockAxes,
        initialBlockShape, "strided", "zero", alignedAxes,
        ArrayRef<int64_t>{static_cast<int64_t>(view.getRank()) - 1},
        /*requirePositiveShape=*/true,
        /*requirePositiveStrides=*/true,
        /*requirePowerOfTwoBlockShape=*/true, /*alignment=*/16,
        /*minimumContiguousBytes=*/16,
        /*pipelineBlockAlignment=*/1,
        /*maximumShapeExtent=*/std::numeric_limits<int32_t>::max(),
        maximumBlockElements);
    descriptors.push_back(
        {viewValue, fragment, orderedFragment, nativeFragment,
         SmallVector<int64_t>(blockAxes.begin(), blockAxes.end()),
         permutation, descriptor});
    return descriptors.back();
  };
  for (auto &[load, plan] : loads)
    if (failed(descriptorFor(load.getResource(),
                            cast<gpu::FragmentType>(load.getResult().getType()),
                            plan.blockAxes)))
      return load.emitOpError(
          "could not declare its tensor-descriptor runtime contract");
  for (auto &[store, plan] : stores)
    if (failed(descriptorFor(store.getResource(),
                            cast<gpu::FragmentType>(store.getValue().getType()),
                            plan.blockAxes)))
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

  auto reshape = [&](OpBuilder &builder, Location location, Value value,
                     gpu::FragmentType target) -> FailureOr<Value> {
    if (value.getType() == target)
      return value;
    auto reassociation = gpu::inferReshapeReassociation(
        cast<gpu::FragmentType>(value.getType()), target);
    if (failed(reassociation))
      return failure();
    return Value(builder.create<gpu::ReshapeOp>(location, target, value,
                                               *reassociation));
  };
  for (auto &[load, plan] : loads) {
    OpBuilder builder(load);
    auto fragment = cast<gpu::FragmentType>(load.getResult().getType());
    auto descriptor = descriptorFor(load.getResource(), fragment, plan.blockAxes);
    FailureOr<SmallVector<Value>> descriptorOffsets =
        materializeDescriptorOffsets(builder, load.getLoc(), plan.offsets);
    if (failed(descriptor) || failed(descriptorOffsets))
      return load.emitOpError(
          "could not materialize the declared tensor-descriptor ABI");
    if (load->getParentOfType<scf::ForOp>())
      descriptor->descriptor.setPipelineBlockAlignmentAttr(
          builder.getI64IntegerAttr(128));
    auto conditional = builder.create<scf::IfOp>(
        load.getLoc(), TypeRange{load.getResult().getType()}, choice.getResult(),
        /*withElseRegion=*/true);
    OpBuilder descriptorBuilder = prepareBranch(conditional.getThenRegion());
    auto descriptorLoad = descriptorBuilder.create<DescriptorLoadOp>(
        load.getLoc(), descriptor->nativeFragment,
        descriptor->descriptor.getResult(), *descriptorOffsets,
        descriptorBuilder.getDenseI64ArrayAttr(plan.boundaryAxes));
    copyOrigin(load, descriptorLoad);
    auto result = reshape(descriptorBuilder, load.getLoc(),
                          descriptorLoad.getResult(), descriptor->orderedFragment);
    if (failed(result))
      return load.emitOpError("cannot remove descriptor unit axes");
    if (!llvm::is_sorted(descriptor->permutation)) {
      SmallVector<int64_t> inverse(descriptor->permutation.size());
      for (auto [axis, original] : llvm::enumerate(descriptor->permutation))
        inverse[original] = axis;
      result = Value(descriptorBuilder.create<gpu::TransposeOp>(
          load.getLoc(), fragment, *result,
          descriptorBuilder.getDenseI64ArrayAttr(inverse)));
    }
    descriptorBuilder.create<scf::YieldOp>(load.getLoc(), *result);

    OpBuilder blockBuilder = prepareBranch(conditional.getElseRegion());
    auto block = cast<gpu::LoadOp>(blockBuilder.clone(*load));
    blockBuilder.create<scf::YieldOp>(load.getLoc(), block.getResult());
    load.getResult().replaceAllUsesWith(conditional.getResult(0));
    load.erase();
  }

  for (auto &[store, plan] : stores) {
    OpBuilder builder(store);
    auto descriptor = descriptorFor(
        store.getResource(), cast<gpu::FragmentType>(store.getValue().getType()),
        plan.blockAxes);
    FailureOr<SmallVector<Value>> descriptorOffsets =
        materializeDescriptorOffsets(builder, store.getLoc(), plan.offsets);
    if (failed(descriptor) || failed(descriptorOffsets))
      return store.emitOpError(
          "could not materialize the declared tensor-descriptor ABI");
    auto conditional = builder.create<scf::IfOp>(
        store.getLoc(), TypeRange{}, choice.getResult(),
        /*withElseRegion=*/true);
    OpBuilder descriptorBuilder = prepareBranch(conditional.getThenRegion());
    Value ordered = store.getValue();
    if (!llvm::is_sorted(descriptor->permutation))
      ordered = descriptorBuilder.create<gpu::TransposeOp>(
          store.getLoc(), descriptor->orderedFragment, ordered,
          descriptorBuilder.getDenseI64ArrayAttr(descriptor->permutation));
    auto value = reshape(descriptorBuilder, store.getLoc(), ordered,
                         descriptor->nativeFragment);
    if (failed(value))
      return store.emitOpError("cannot insert descriptor unit axes");
    auto descriptorStore = descriptorBuilder.create<DescriptorStoreOp>(
        store.getLoc(), descriptor->descriptor.getResult(), *descriptorOffsets,
        *value, descriptorBuilder.getDenseI64ArrayAttr(plan.boundaryAxes));
    copyOrigin(store, descriptorStore);
    descriptorBuilder.create<scf::YieldOp>(store.getLoc());

    OpBuilder blockBuilder = prepareBranch(conditional.getElseRegion());
    blockBuilder.clone(*store);
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

bool isTritonFragmentExtent(Attribute attribute);

SmallVector<gpu::FragmentType> collectiveFragments(func::FuncOp kernel) {
  SmallVector<gpu::FragmentType> fragments;
  kernel.walk([&](Operation *operation) {
    auto store = dyn_cast<gpu::StoreOp>(operation);
    bool workspaceStore = store && isa<gpu::BufferType>(store.getResource().getType());
    if (store)
      if (auto argument = dyn_cast<BlockArgument>(store.getResource());
          argument && argument.getOwner() == &kernel.front()) {
        auto kind = kernel.getArgAttrOfType<StringAttr>(
            argument.getArgNumber(), gpu::abiKindAttr);
        workspaceStore |= kind && kind.getValue() == "workspace";
      }
    if (!isa<gpu::ReduceOp, gpu::ScanOp, ReduceOp, ScanOp>(operation) &&
        !workspaceStore)
      return;
    auto collect = [&](Value operand) {
      if (auto fragment = dyn_cast<gpu::FragmentType>(operand.getType());
          fragment && !llvm::is_contained(fragments, fragment))
        fragments.push_back(fragment);
    };
    if (workspaceStore)
      collect(store.getValue());
    else
      for (Value operand : operation->getOperands())
        collect(operand);
  });
  return fragments;
}

LogicalResult materializeDeferredResourceBounds(func::FuncOp kernel) {
  TritonConfig known;
  gpu::ParameterAttr warpParameter;
  kernel.walk([&](gpu::ParameterOp parameter) {
    auto schema = parameter.getParameter();
    if (schema.getRole() ==
        static_cast<uint32_t>(gpu::ParameterRole::ProviderWarps))
      warpParameter = schema;
    if (schema.getCategory() !=
            static_cast<uint32_t>(gpu::ParameterCategory::Coverage) &&
        !parameter->hasAttr(gpu::coverageDimensionAttr))
      known.kernelParameters[schema.getName().getValue().str()] =
          schema.getCandidates().asArrayRef().front();
  });
  SmallVector<gpu::FragmentType> fragments;
  std::function<void(Type)> collect = [&](Type type) {
    if (auto fragment = dyn_cast<gpu::FragmentType>(type)) {
      if (!llvm::is_contained(fragments, fragment))
        fragments.push_back(fragment);
    } else if (auto record = dyn_cast<gpu::RecordType>(type)) {
      for (Attribute field : record.getFieldTypes())
        collect(cast<TypeAttr>(field).getValue());
    }
  };
  kernel.walk([&](Operation *operation) {
    for (Type type : operation->getOperandTypes())
      collect(type);
    for (Type type : operation->getResultTypes())
      collect(type);
  });
  SmallVector<gpu::PhysicalExprAttr> bounds;
  for (gpu::FragmentType fragment : fragments) {
    gpu::PhysicalExprAttr elements;
    for (Attribute attribute : fragment.getShape()) {
      auto extent = cast<gpu::PhysicalExprAttr>(attribute);
      if (extent.getKind() ==
              static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) &&
          extent.getValue() == 1)
        continue;
      elements = !elements ? extent : gpu::PhysicalExprAttr::get(
          kernel.getContext(),
          static_cast<uint32_t>(gpu::PhysicalExprKind::Multiply), 0,
          StringAttr::get(kernel.getContext(), ""),
          ArrayAttr::get(kernel.getContext(), {elements, extent}));
    }
    if (!elements || evaluateCompileTimeExpression(elements, known))
      continue;
    if (!isTritonFragmentExtent(elements))
      return kernel.emitError("Triton tensor bounds require constexpr fragment extents");
    if (!llvm::is_contained(bounds, elements))
      bounds.push_back(elements);
  }
  kernel.getContext()->getOrLoadDialect<cf::ControlFlowDialect>();
  OpBuilder builder = OpBuilder::atBlockBegin(&kernel.front());
  auto assertBound = [&](gpu::PhysicalExprAttr expression, int64_t limit,
                         StringRef message) {
    Value maximum = builder.create<arith::ConstantIndexOp>(kernel.getLoc(), limit);
    Value count = builder.create<gpu::PhysicalExprOp>(
        kernel.getLoc(), builder.getIndexType(), expression);
    Value valid = builder.create<gpu::CompareOp>(
        kernel.getLoc(), builder.getI1Type(), count, maximum, ComparePredicate::Le);
    builder.create<cf::AssertOp>(kernel.getLoc(), valid, message);
  };
  for (gpu::PhysicalExprAttr elements : bounds)
    assertBound(elements, maxTritonTensorElements,
                "Triton block tensor exceeds the maximum element count");
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  SmallVector<ValueRange> reductionSources;
  kernel.walk([&](ReduceOp reduce) {
    reductionSources.push_back(reduce.getInputs().take_front(reduce.getSourceCount()));
  });
  gpu::materializeDeferredReductionBounds(kernel, reductionSources);
  if (capabilities && capabilities.getRegistersPerUnit() > 0) {
    if (warpParameter && warpParameter.getCandidates().size() > 1) {
      auto expression = [&](gpu::PhysicalExprKind kind, int64_t value,
                            ArrayRef<Attribute> operands) {
        return gpu::PhysicalExprAttr::get(
            kernel.getContext(), static_cast<uint32_t>(kind), value,
            builder.getStringAttr(""), builder.getArrayAttr(operands));
      };
      auto constant = [&](int64_t value) {
        return expression(gpu::PhysicalExprKind::Constant, value, {});
      };
      auto warps = gpu::PhysicalExprAttr::get(
          kernel.getContext(),
          static_cast<uint32_t>(gpu::PhysicalExprKind::Parameter), 0,
          warpParameter.getName(), builder.getArrayAttr({}));
      int64_t maximumWarps =
          *llvm::max_element(warpParameter.getCandidates().asArrayRef());
      auto belowMaximum = expression(gpu::PhysicalExprKind::Subtract, 0,
                                     {warps, constant(maximumWarps)});
      auto nominalBudget = expression(gpu::PhysicalExprKind::Multiply, 0,
                                      {warps, constant(32 * 255)});
      // Apply the same per-fragment policy after specialization binds the
      // physical extents. The widest supplied option may spill.
      auto budget = expression(gpu::PhysicalExprKind::Select, 0,
                               {belowMaximum, nominalBudget,
                                constant(std::numeric_limits<int64_t>::max())});
      for (gpu::FragmentType fragment : collectiveFragments(kernel)) {
        auto footprint = gpu::fragmentRegisterFootprint(fragment);
        if (evaluateCompileTimeExpression(footprint, TritonConfig{}))
          continue;
        Value count = builder.create<gpu::PhysicalExprOp>(
            kernel.getLoc(), builder.getIndexType(), footprint);
        Value maximum = builder.create<gpu::PhysicalExprOp>(
            kernel.getLoc(), builder.getIndexType(), budget);
        Value valid = builder.create<gpu::CompareOp>(
            kernel.getLoc(), builder.getI1Type(), count, maximum,
            ComparePredicate::Le);
        builder.create<cf::AssertOp>(
            kernel.getLoc(), valid,
            "Triton collective fragment exceeds the nominal per-thread register budget");
      }
    }
  }
  return success();
}

bool fitsReductionRegisterBudget(gpu::ReduceOp reduce,
                                 const TritonConfig &config,
                                 func::FuncOp kernel) {
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities || capabilities.getRegistersPerUnit() <= 0)
    return true;
  __int128 registers = 0;
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto footprint = gpu::reductionRegisterFootprint(ValueRange{source}, kernel);
    if (!footprint)
      continue;
    auto size = evaluateCompileTimeExpression(footprint, config);
    if (!size)
      return true;
    registers += *size;
    if (registers > capabilities.getRegistersPerUnit())
      return false;
  }
  return true;
}

bool descriptorFragmentFits(gpu::FragmentType fragment,
                            const TritonConfig &config,
                            int64_t pipelineBlockAlignment,
                            const llvm::StringMap<SmallVector<int64_t>>
                                &parameterDomains,
                            const llvm::StringSet<> &coverageParameters) {
  std::optional<int64_t> elementBytes =
      descriptorElementBytes(fragment.getElementType());
  if (!elementBytes)
    return false;
  __int128 elements = 1;
  __int128 concreteElements = 1;
  bool concrete = true;
  int64_t minimumLastExtent = std::numeric_limits<int64_t>::max();
  bool runtimeGuardsLastExtent = false;
  bool runtimeGuardsElementCount = false;
  for (auto [axis, extent] : llvm::enumerate(fragment.getShape())) {
    auto expression = cast<gpu::PhysicalExprAttr>(extent);
    SmallVector<int64_t> values;
    if (std::optional<int64_t> value =
            evaluateCompileTimeExpression(expression, config)) {
      values.push_back(*value);
      concreteElements *= *value;
    } else if (static_cast<gpu::PhysicalExprKind>(expression.getKind()) ==
               gpu::PhysicalExprKind::Parameter) {
      auto domain = parameterDomains.find(expression.getSymbol().getValue());
      if (domain == parameterDomains.end())
        return false;
      concrete = false;
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
  // Triton's canPipelineTMALoad requires each shared stage to begin at a
  // 128-byte boundary. Small legal descriptor tiles can otherwise produce a
  // misaligned second buffer in a multistage pipeline.
  if (config.stages > 1 && concrete &&
      (concreteElements * *elementBytes) % pipelineBlockAlignment != 0)
    return false;
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
  int64_t maximumWarps = *llvm::max_element(warps->candidates);
  SmallVector<gpu::PhysicalExprAttr> collectiveFootprints;
  for (gpu::FragmentType fragment : collectiveFragments(kernel)) {
    auto footprint = gpu::fragmentRegisterFootprint(fragment);
    if (!llvm::is_contained(collectiveFootprints, footprint))
      collectiveFootprints.push_back(footprint);
  }
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
    // Apply the existing per-fragment policy after binding the shared tuple.
    // ABI-dependent extents retain their deferred specialization assertion.
    if (config.warps < maximumWarps &&
        llvm::any_of(collectiveFootprints, [&](gpu::PhysicalExprAttr footprint) {
          auto words = evaluateCompileTimeExpression(footprint, config);
          return words && *words > config.warps * 32 * 255;
        }))
      continue;
    bool legal = true;
    bool descriptorConfig =
        descriptorChoice &&
        config.kernelParameters.at(
            descriptorChoice.getConfigParameter().str()) != 0;
    kernel.walk([&](Operation *operation) {
      if (!legal)
        return WalkResult::interrupt();
      if (auto reduce = dyn_cast<gpu::ReduceOp>(operation))
        if (!fitsReductionRegisterBudget(reduce, config, kernel)) {
          legal = false;
          return WalkResult::interrupt();
        }
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
                                          load.getDescriptor()
                                              .getDefiningOp<TensorDescriptorOp>()
                                              .getPipelineBlockAlignment(),
                                          parameterDomains,
                                          coverageParameters);
        else if (auto store = dyn_cast<DescriptorStoreOp>(operation))
          legal &= descriptorFragmentFits(store.getValue().getType(), config,
                                          store.getDescriptor()
                                              .getDefiningOp<TensorDescriptorOp>()
                                              .getPipelineBlockAlignment(),
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
        "all Triton parameter candidates violate typed fragment legality or the reduction register budget");
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
  auto [nextSource, nextDimension] = gpu::nextPhysicalAxisIdentities(kernel);
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
    if (source && !source.getShape().empty() &&
        gather.getCoordinates().size() == source.getShape().size() &&
        llvm::all_of(gather.getCoordinates(), [](Value coordinate) {
          Type element = coordinate.getType();
          if (auto fragment = dyn_cast<gpu::FragmentType>(element))
            element = fragment.getElementType();
          return isa<IndexType, IntegerType>(element);
        }) &&
        (result || gather.getValid().getType().isInteger(1))) {
      OpBuilder builder(gather);
      Location location = gather.getLoc();
      Type indexType = builder.getIndexType();
      Type predicateType = builder.getI1Type();
      if (result) {
        indexType = gpu::FragmentType::get(
            kernel.getContext(), builder.getIndexType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        predicateType = gpu::FragmentType::get(
            kernel.getContext(), builder.getI1Type(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
      }
      auto projectIndex = [&](Value value) -> FailureOr<Value> {
        if (auto fragment = dyn_cast<gpu::FragmentType>(value.getType())) {
          if (!result)
            return failure();
          if (!fragment.getElementType().isIndex()) {
            auto converted = gpu::FragmentType::get(
                kernel.getContext(), builder.getIndexType(), fragment.getShape(),
                fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
            value = builder.create<gpu::CastOp>(location, converted, value);
          }
        } else if (!value.getType().isIndex()) {
          value = builder.create<gpu::CastOp>(location, builder.getIndexType(), value);
        }
        if (result)
          return gpu::projectPhysicalValueToSchema(
              builder, location, value, cast<gpu::FragmentType>(indexType));
        return value;
      };
      SmallVector<Value> coordinates(source.getShape().size());
      for (auto [axis, value] :
           llvm::zip(gather.getSourceAxes(), gather.getCoordinates())) {
        FailureOr<Value> projected = projectIndex(value);
        if (failed(projected))
          return gather.emitOpError("gather coordinate has no exact result-axis projection");
        coordinates[axis] = *projected;
      }
      Value sourceValue = gather.getSource();
      Value coordinate = coordinates.front();
      Value valid = gather.getValid();
      if (result && valid.getType() != predicateType) {
        auto projected = gpu::projectPhysicalValueToSchema(
            builder, location, valid, cast<gpu::FragmentType>(predicateType));
        if (failed(projected))
          return gather.emitOpError("gather validity has no exact result-axis projection");
        valid = *projected;
      }
      if (source.getShape().size() > 1) {
        auto elements = cast<gpu::PhysicalExprAttr>(source.getShape()[0]);
        auto first = projectIndex(builder.create<arith::ConstantIndexOp>(location, 0));
        if (failed(first))
          return failure();
        coordinate = *first;
        for (auto [axis, value] : llvm::enumerate(coordinates)) {
          auto extent = cast<gpu::PhysicalExprAttr>(source.getShape()[axis]);
          if (axis)
            elements = gpu::PhysicalExprAttr::get(
                kernel.getContext(),
                static_cast<uint32_t>(gpu::PhysicalExprKind::Multiply), 0,
                builder.getStringAttr(""), builder.getArrayAttr({elements, extent}));
          Value size = builder.create<gpu::PhysicalExprOp>(
              location, builder.getIndexType(), extent);
          auto projectedSize = projectIndex(size);
          if (failed(projectedSize))
            return failure();
          coordinate = builder.create<gpu::BinaryOp>(
              location, indexType, coordinate, *projectedSize, BinaryOperator::Multiply);
          coordinate = builder.create<gpu::BinaryOp>(
              location, indexType, coordinate, value, BinaryOperator::Add);
        }
        if (auto count = evaluateCompileTimeExpression(elements, TritonConfig{}))
          elements = gpu::PhysicalExprAttr::get(
              kernel.getContext(),
              static_cast<uint32_t>(gpu::PhysicalExprKind::Constant), *count,
              builder.getStringAttr(""), builder.getArrayAttr({}));
        auto mapping = gpu::AxisMapAttr::get(
            kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
        auto flatType = gpu::FragmentType::get(
            kernel.getContext(), source.getElementType(),
            builder.getArrayAttr({elements}), builder.getArrayAttr({mapping}),
            source.getValidity(), source.getOwner());
        auto reassociation = gpu::inferReshapeReassociation(source, flatType);
        if (failed(reassociation))
          return gather.emitOpError("scalar gather has no exact row-major linearization");
        sourceValue = builder.create<gpu::ReshapeOp>(
            location, flatType, sourceValue, *reassociation);
        Value size = builder.create<gpu::PhysicalExprOp>(
            location, builder.getIndexType(), elements);
        auto projectedSize = projectIndex(size);
        if (failed(projectedSize))
          return failure();
        Value lower = builder.create<gpu::CompareOp>(
            location, predicateType, coordinate, *first, ComparePredicate::Ge);
        Value upper = builder.create<gpu::CompareOp>(
            location, predicateType, coordinate, *projectedSize, ComparePredicate::Lt);
        valid = builder.create<gpu::BinaryOp>(
            location, predicateType, valid, lower, BinaryOperator::LogicalAnd);
        valid = builder.create<gpu::BinaryOp>(
            location, predicateType, valid, upper, BinaryOperator::LogicalAnd);
      }
      FailureOr<Value> zero = zeroLike(builder, gather.getLoc(), coordinate.getType());
      if (failed(zero))
        return gather.emitOpError("scalar gather coordinate has no integral zero");
      Value safeIndex = builder.create<gpu::SelectOp>(
          gather.getLoc(), coordinate.getType(), valid, coordinate, *zero);
      Type loadedType = gather.getResult().getType();
      if (result && result.getShape().size() > 1) {
        auto elements = cast<gpu::PhysicalExprAttr>(result.getShape()[0]);
        for (Attribute extent : result.getShape().getValue().drop_front())
          elements = gpu::PhysicalExprAttr::get(
              kernel.getContext(),
              static_cast<uint32_t>(gpu::PhysicalExprKind::Multiply), 0,
              builder.getStringAttr(""), builder.getArrayAttr({elements, extent}));
        auto mapping = gpu::AxisMapAttr::get(
            kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
        auto flatIndex = gpu::FragmentType::get(
            kernel.getContext(), builder.getIndexType(), builder.getArrayAttr({elements}),
            builder.getArrayAttr({mapping}), result.getValidity(), result.getOwner());
        auto relation = gpu::inferReshapeReassociation(
            cast<gpu::FragmentType>(safeIndex.getType()), flatIndex);
        if (failed(relation))
          return gather.emitOpError("gather result has no exact row-major linearization");
        safeIndex = builder.create<gpu::ReshapeOp>(location, flatIndex, safeIndex, *relation);
        loadedType = gpu::FragmentType::get(
            kernel.getContext(), result.getElementType(), flatIndex.getShape(),
            flatIndex.getAxisMaps(), result.getValidity(), result.getOwner());
      }
      auto loaded = builder.create<gpu::GatherOp>(
          gather.getLoc(), loadedType, sourceValue,
          ValueRange{safeIndex}, Value(), Value(), ArrayRef<int64_t>{0});
      Value loadedValue = loaded.getResult();
      if (loadedType != gather.getResult().getType()) {
        auto relation = gpu::inferReshapeReassociation(
            cast<gpu::FragmentType>(loadedType), result);
        if (failed(relation))
          return gather.emitOpError("linear gather has no exact result reassociation");
        loadedValue = builder.create<gpu::ReshapeOp>(location, result, loadedValue, *relation);
      }
      auto selected = builder.create<gpu::SelectOp>(
          gather.getLoc(), gather.getResult().getType(), valid,
          loadedValue, gather.getFill());
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
          ((selectedExtent.getKind() == static_cast<uint32_t>(
                                           gpu::PhysicalExprKind::Constant) &&
            selectedExtent.getValue() > 0) ||
           selectedExtent.getKind() == static_cast<uint32_t>(
                                          gpu::PhysicalExprKind::Parameter));
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
            gather.getContext(), elementType(gather.getCoordinates().front().getType()),
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
    if (!samePhysicalShape(gather.getValid().getType(), coordinate.getType())) {
      auto predicateType = dyn_cast<gpu::FragmentType>(gather.getValid().getType());
      Type indexElement = elementType(coordinate.getType());
      if (!predicateType || !isa<IntegerType, IndexType>(indexElement))
        continue;
      if (isa<gpu::FragmentType>(coordinate.getType()) &&
          (!source || !result || gather.getSourceAxes().size() != 1 ||
           source.getShape().size() != result.getShape().size()))
        continue;
      auto coordinateType = gpu::FragmentType::get(
          gather.getContext(), indexElement, predicateType.getShape(),
          predicateType.getAxisMaps(), predicateType.getValidity(),
          predicateType.getOwner());
      auto projected = gpu::projectPhysicalValueToSchema(
          builder, gather.getLoc(), coordinate, coordinateType);
      if (failed(projected))
        continue;
      coordinate = *projected;
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

void canonicalizeBroadcastProjections(func::FuncOp kernel) {
  SmallVector<Value> pending;
  kernel.walk([&](gpu::ContractOp contract) {
    pending.push_back(contract.getLhs());
    pending.push_back(contract.getRhs());
  });
  llvm::DenseSet<Operation *> visited;
  SmallVector<gpu::ReshapeOp> candidates;
  while (!pending.empty()) {
    Operation *producer = pending.pop_back_val().getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    llvm::append_range(pending, producer->getOperands());
    for (Region &region : producer->getRegions())
      for (Block &block : region)
        llvm::append_range(pending, block.getTerminator()->getOperands());
    if (auto reshape = dyn_cast<gpu::ReshapeOp>(producer))
      candidates.push_back(reshape);
  }
  // A load's address and predicate producers are outside this additional scope.
  kernel.walk([&](gpu::ReshapeOp reshape) {
    auto source = cast<gpu::FragmentType>(reshape.getValue().getType());
    Value resource;
    if (auto load = reshape.getValue().getDefiningOp<gpu::LoadOp>())
      resource = load.getResource();
    else if (auto load = reshape.getValue().getDefiningOp<BlockLoadOp>())
      resource = load.getView();
    else
      return;
    // A vector resource may be loaded across several execution axes.
    auto view = dyn_cast<gpu::ViewType>(resource.getType());
    if ((source.getShape().size() == 1 || (view && view.getRank() == 1)) &&
        visited.insert(reshape).second)
      candidates.push_back(reshape);
  });
  SmallVector<gpu::ReshapeOp> projections;
  for (gpu::ReshapeOp reshape : candidates) {
    auto source = cast<gpu::FragmentType>(reshape.getValue().getType());
    auto target = cast<gpu::FragmentType>(reshape.getResult().getType());
    if (source.getShape().size() >= target.getShape().size() ||
        !llvm::all_of(reshape.getReassociation(), [](Attribute attribute) {
          auto group = cast<gpu::ReshapeGroupAttr>(attribute);
          return !group.getResultAxes().empty() &&
                 group.getSourceAxes().size() <= 1 &&
                 (group.getSourceAxes().empty() ||
                  group.getResultAxes().size() == 1);
        }))
      continue;
    auto projection = gpu::queryBroadcastProjection(source, target);
    if (!projection.isExact())
      continue;
    unsigned nextSource = 0;
    bool insertsUnits = llvm::all_of(
        llvm::enumerate(projection.targetToSource), [&](const auto &entry) {
          auto [axis, mapped] = entry;
          if (mapped)
            return *mapped == nextSource++ &&
                   source.getShape()[*mapped] == target.getShape()[axis];
          auto extent = cast<gpu::PhysicalExprAttr>(target.getShape()[axis]);
          return extent.getKind() ==
                     static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) &&
                 extent.getValue() == 1;
        });
    if (insertsUnits && nextSource == source.getShape().size())
      projections.push_back(reshape);
  }
  // Expand-dims preserves dot input alignment and avoids redundant register
  // copies (and loads) when broadcasting a loaded vector. Leave unrelated
  // high-rank and reduction result layout choices free.
  for (gpu::ReshapeOp reshape : projections) {
    OpBuilder builder(reshape);
    auto broadcast = builder.create<gpu::BroadcastOp>(
        reshape.getLoc(), reshape.getResult().getType(), reshape.getValue());
    broadcast->setDiscardableAttrs(llvm::to_vector(reshape->getDiscardableAttrs()));
    reshape.getResult().replaceAllUsesWith(broadcast.getResult());
    reshape.erase();
  }
}

void selectContractForms(func::FuncOp kernel) {
  kernel.getContext()->getOrLoadDialect<cf::ControlFlowDialect>();
  SmallVector<gpu::ContractOp> contracts;
  kernel.walk([&](gpu::ContractOp contract) { contracts.push_back(contract); });
  for (gpu::ContractOp contract : contracts) {
    auto lhs = contract.getLhs().getType();
    auto rhs = contract.getRhs().getType();
    bool ieeeFp32 = lhs.getElementType().isF32() &&
                    rhs.getElementType().isF32() &&
                    contract.getResult().getType().getElementType().isF32();
    if (lhs.getShape().empty())
      continue;
    auto reductionExtent =
        dyn_cast<gpu::PhysicalExprAttr>(
            lhs.getShape()[lhs.getShape().size() - 1]);
    if (!reductionExtent)
      continue;
    OpBuilder builder(contract);
    auto expandedFits = [&]() -> Value {
      auto elements = reductionExtent;
      for (Attribute dimension : contract.getAccumulator().getType().getShape())
        elements = gpu::PhysicalExprAttr::get(
            kernel.getContext(),
            static_cast<uint32_t>(gpu::PhysicalExprKind::Multiply), 0,
            builder.getStringAttr(""), builder.getArrayAttr({elements, dimension}));
      Value count = builder.create<gpu::PhysicalExprOp>(
          contract.getLoc(), builder.getIndexType(), elements);
      Value maximum = builder.create<arith::ConstantIndexOp>(
          contract.getLoc(), maxTritonTensorElements);
      return builder.create<gpu::CompareOp>(contract.getLoc(), builder.getI1Type(),
                                           count, maximum, ComparePredicate::Le);
    };
    if (reductionExtent.getKind() ==
        static_cast<uint32_t>(gpu::PhysicalExprKind::Constant)) {
      if (reductionExtent.getValue() < 16) {
        builder.create<cf::AssertOp>(contract.getLoc(), expandedFits(),
            "Triton expanded contraction exceeds the maximum element count");
        contract->setAttr(contractFormAttr,
                          StringAttr::get(kernel.getContext(), "multiply_sum"));
        continue;
      }
      if (!ieeeFp32)
        continue;
    }
    // IEEE dot has no matrix reuse along a unit free axis. Prefer its parallel
    // reduction expansion there, as well as when estimated staging is too large.
    // The provider still validates the native form's actual resource usage.
    Value canExpand = expandedFits();
    Value extent = builder.create<gpu::PhysicalExprOp>(
        contract.getLoc(), builder.getIndexType(), reductionExtent);
    Value minimum = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 16);
    Value small = builder.create<gpu::CompareOp>(
        contract.getLoc(), builder.getI1Type(), extent, minimum,
        ComparePredicate::Lt);
    if (ieeeFp32) {
      Value footprint = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 0);
      for (auto operand : {lhs, rhs}) {
        Value bytes = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 4);
        for (Attribute dimension : operand.getShape()) {
          Value width = builder.create<gpu::PhysicalExprOp>(
              contract.getLoc(), builder.getIndexType(),
              cast<gpu::PhysicalExprAttr>(dimension));
          bytes = builder.create<gpu::BinaryOp>(contract.getLoc(),
              builder.getIndexType(), bytes, width, BinaryOperator::Multiply);
        }
        footprint = builder.create<gpu::BinaryOp>(contract.getLoc(),
            builder.getIndexType(), footprint, bytes, BinaryOperator::Add);
      }
      auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
      Value capacity = builder.create<arith::ConstantIndexOp>(
          contract.getLoc(), capabilities.getMaxDynamicSharedMemoryPerBlock());
      Value preferExpansion = builder.create<gpu::CompareOp>(contract.getLoc(),
          builder.getI1Type(), footprint, capacity, ComparePredicate::Gt);
      auto shape = contract.getAccumulator().getType().getShape().getValue();
      for (Attribute dimension : shape.take_back(std::min<size_t>(2, shape.size()))) {
        Value width = builder.create<gpu::PhysicalExprOp>(
            contract.getLoc(), builder.getIndexType(),
            cast<gpu::PhysicalExprAttr>(dimension));
        Value one = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 1);
        Value unit = builder.create<gpu::CompareOp>(contract.getLoc(),
            builder.getI1Type(), width, one, ComparePredicate::Eq);
        preferExpansion = builder.create<gpu::BinaryOp>(contract.getLoc(),
            builder.getI1Type(), preferExpansion, unit, BinaryOperator::LogicalOr);
      }
      preferExpansion = builder.create<gpu::BinaryOp>(contract.getLoc(),
          builder.getI1Type(), preferExpansion, canExpand, BinaryOperator::LogicalAnd);
      small = builder.create<gpu::BinaryOp>(contract.getLoc(), builder.getI1Type(),
          small, preferExpansion, BinaryOperator::LogicalOr);
    }
    auto choice = builder.create<scf::IfOp>(
        contract.getLoc(), TypeRange{contract.getResult().getType()}, small, true);
    builder.setInsertionPointToStart(&choice.getThenRegion().front());
    builder.create<cf::AssertOp>(contract.getLoc(), canExpand,
        "Triton expanded contraction exceeds the maximum element count");
    auto expanded = cast<gpu::ContractOp>(builder.clone(*contract));
    expanded->setAttr(contractFormAttr,
                      StringAttr::get(kernel.getContext(), "multiply_sum"));
    builder.create<scf::YieldOp>(contract.getLoc(), expanded.getResult());
    builder.setInsertionPointToStart(&choice.getElseRegion().front());
    auto native = cast<gpu::ContractOp>(builder.clone(*contract));
    builder.create<scf::YieldOp>(contract.getLoc(), native.getResult());
    contract.getResult().replaceAllUsesWith(choice.getResult(0));
    contract.erase();
  }

  contracts.clear();
  kernel.walk([&](gpu::ContractOp contract) {
    auto form = contract->getAttrOfType<StringAttr>(contractFormAttr);
    if (form && form.getValue() == "multiply_sum" &&
        contract.getAccumulator().getType().getElementType().isF32())
      contracts.push_back(contract);
  });
  for (gpu::ContractOp contract : contracts) {
    OpBuilder builder(contract);
    auto shape = contract.getAccumulator().getType().getShape().getValue();
    // A unit matrix axis offers no second free axis to amortize serial K work.
    // Keep its legal multiply/reduce expansion parallel along K.
    ArrayRef<Attribute> matrixAxes =
        shape.take_back(std::min<size_t>(2, shape.size()));
    if (llvm::any_of(matrixAxes, [&](Attribute dimension) {
          auto extent = evaluateCompileTimeExpression(
              cast<gpu::PhysicalExprAttr>(dimension), TritonConfig{});
          return extent && *extent == 1;
        }))
      continue;
    auto elements = cast<gpu::PhysicalExprAttr>(shape.front());
    for (Attribute extent : shape.drop_front())
      elements = gpu::PhysicalExprAttr::get(
          kernel.getContext(),
          static_cast<uint32_t>(gpu::PhysicalExprKind::Multiply), 0,
          builder.getStringAttr(""), builder.getArrayAttr({elements, extent}));
    auto fmaForm = builder.getStringAttr("fma");
    // Serial K accumulation needs enough independent output elements.
    if (auto count = evaluateCompileTimeExpression(elements, TritonConfig{})) {
      if (*count >= 256)
        contract->setAttr(contractFormAttr, fmaForm);
      continue;
    }
    Value count = builder.create<gpu::PhysicalExprOp>(
        contract.getLoc(), builder.getIndexType(), elements);
    Value limit = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 256);
    Value wide = builder.create<gpu::CompareOp>(
        contract.getLoc(), builder.getI1Type(), count, limit,
        ComparePredicate::Ge);
    for (Attribute dimension : matrixAxes) {
      auto extent = cast<gpu::PhysicalExprAttr>(dimension);
      if (evaluateCompileTimeExpression(extent, TritonConfig{}))
        continue;
      Value width = builder.create<gpu::PhysicalExprOp>(
          contract.getLoc(), builder.getIndexType(), extent);
      Value one = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 1);
      Value multiple = builder.create<gpu::CompareOp>(
          contract.getLoc(), builder.getI1Type(), width, one, ComparePredicate::Gt);
      wide = builder.create<gpu::BinaryOp>(
          contract.getLoc(), builder.getI1Type(), wide, multiple,
          BinaryOperator::LogicalAnd);
    }
    auto choice = builder.create<scf::IfOp>(
        contract.getLoc(), TypeRange{contract.getResult().getType()}, wide, true);
    builder.setInsertionPointToStart(&choice.getThenRegion().front());
    auto fma = cast<gpu::ContractOp>(builder.clone(*contract));
    fma->setAttr(contractFormAttr, fmaForm);
    builder.create<scf::YieldOp>(contract.getLoc(), fma.getResult());
    builder.setInsertionPointToStart(&choice.getElseRegion().front());
    auto reduction = cast<gpu::ContractOp>(builder.clone(*contract));
    builder.create<scf::YieldOp>(contract.getLoc(), reduction.getResult());
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
         isTritonExpression(expression) &&
         llvm::all_of(expression.getOperands(), isTritonFragmentExtent);
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

bool hasMapRelation(Type type, gpu::FragmentType result) {
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return fragment.getShape() == result.getShape() &&
           fragment.getAxisMaps() == result.getAxisMaps() &&
           fragment.getValidity() == result.getValidity() &&
           fragment.getOwner() == result.getOwner();
  return isa<IntegerType, IndexType, FloatType>(type);
}

bool isScalarizableMapProducer(Operation *operation,
                              gpu::FragmentType result) {
  if (!llvm::all_of(operation->getOperandTypes(), [&](Type type) {
        return hasMapRelation(type, result);
      }) || !llvm::all_of(operation->getResultTypes(), [&](Type type) {
        return hasMapRelation(type, result);
      }))
    return false;
  if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    APInt lower, upper, step;
    if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
        !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
        !step.isStrictlyPositive())
      return false;
    if (lower.slt(upper)) {
      bool overflow = false;
      (void)(upper - 1).sadd_ov(step, overflow);
      if (overflow)
        return false;
    }
    return llvm::all_of(loop.getBody()->without_terminator(),
                       [&](Operation &nested) {
                         return isScalarizableMapProducer(&nested, result);
                       });
  }
  return isa<arith::ConstantOp, gpu::SplatOp, gpu::BroadcastOp,
             gpu::UnaryOp, gpu::BinaryOp, gpu::CompareOp, gpu::SelectOp,
             gpu::CastOp, gpu::BitcastOp>(operation);
}

SmallVector<Value> mapProducerInputs(Operation *operation) {
  llvm::SetVector<Value> inputs;
  inputs.insert(operation->operand_begin(), operation->operand_end());
  for (Region &region : operation->getRegions())
    getUsedValuesDefinedAbove(region, inputs);
  return SmallVector<Value>(inputs.begin(), inputs.end());
}

bool hasExpensiveMapProducer(Operation *operation) {
  if (isa<scf::ForOp>(operation))
    return true;
  if (auto binary = dyn_cast<gpu::BinaryOp>(operation))
    return binary.getOperatorKind() == BinaryOperator::TrueDivide ||
           binary.getOperatorKind() == BinaryOperator::Power;
  auto unary = dyn_cast<gpu::UnaryOp>(operation);
  if (!unary)
    return false;
  switch (unary.getOperatorKind()) {
  case UnaryOperator::Exp:
  case UnaryOperator::Exp2:
  case UnaryOperator::Log:
  case UnaryOperator::Log1p:
  case UnaryOperator::Lgamma:
  case UnaryOperator::Sin:
  case UnaryOperator::Cos:
  case UnaryOperator::Erf:
  case UnaryOperator::Erfc:
  case UnaryOperator::I0:
  case UnaryOperator::Tanh:
    return !unary.getApproximate();
  default:
    return false;
  }
}

void sinkSelectProducers(func::FuncOp kernel) {
  SmallVector<gpu::SelectOp> selects;
  bool changed = false;
  kernel.walk([&](gpu::SelectOp select) { selects.push_back(select); });
  // Start at consumers so nested selects stay inside one scalar callback.
  for (gpu::SelectOp select : llvm::reverse(selects)) {
    auto resultType = dyn_cast<gpu::FragmentType>(select.getType());
    if (!resultType || select->use_empty())
      continue;
    Block *block = select->getBlock();
    llvm::SmallPtrSet<Operation *, 32> slices[2];
    for (unsigned arm = 0; arm < 2; ++arm) {
      SmallVector<Value> pending{select->getOperand(arm + 1)};
      while (!pending.empty()) {
        Operation *producer = pending.pop_back_val().getDefiningOp();
        if (!producer || producer->getBlock() != block ||
            (!isa<arith::ConstantOp>(producer) &&
             !llvm::any_of(producer->getResultTypes(), [](Type type) {
               return isa<gpu::FragmentType>(type);
             })) ||
            !isScalarizableMapProducer(producer, resultType) ||
            !slices[arm].insert(producer).second)
          continue;
        llvm::append_range(pending, mapProducerInputs(producer));
      }
    }
    // Shared producers and values used outside the selected arm stay eager.
    // Only a closed, finite, effect-free slice can move under the condition.
    llvm::SmallPtrSet<Operation *, 32> exclusive[2];
    for (unsigned arm = 0; arm < 2; ++arm)
      for (Operation *producer : slices[arm])
        if (!slices[1 - arm].contains(producer))
          exclusive[arm].insert(producer);
    for (unsigned arm = 0; arm < 2; ++arm) {
      bool changed;
      do {
        SmallVector<Operation *> retained;
        for (Operation *producer : exclusive[arm])
          if (llvm::any_of(producer->getResults(), [&](Value value) {
                return llvm::any_of(value.getUses(), [&](OpOperand &use) {
                  if (use.getOwner() == select)
                    return use.getOperandNumber() != arm + 1;
                  Operation *owner = use.getOwner();
                  while (owner->getBlock() != block && owner->getParentOp())
                    owner = owner->getParentOp();
                  return !exclusive[arm].contains(owner);
                });
              }))
            retained.push_back(producer);
        for (Operation *producer : retained)
          exclusive[arm].erase(producer);
        changed = !retained.empty();
      } while (changed);
    }
    // Without branch frequencies, keep short expressions predicated. A scalar
    // callback must avoid a loop or multiple library calls to pay for control.
    auto profitable = [&](const auto &slice) {
      return llvm::any_of(slice, [](Operation *operation) {
               return isa<scf::ForOp>(operation);
             }) || llvm::count_if(slice, hasExpensiveMapProducer) >= 2;
    };
    if (!profitable(exclusive[0]) && !profitable(exclusive[1]))
      continue;

    llvm::SetVector<Value> captures;
    captures.insert(select.getCondition());
    for (unsigned arm = 0; arm < 2; ++arm) {
      auto capture = [&](Value value) {
        if (!exclusive[arm].contains(value.getDefiningOp()))
          captures.insert(value);
      };
      capture(select->getOperand(arm + 1));
      for (Operation &operation : *block)
        if (exclusive[arm].contains(&operation))
          for (Value input : mapProducerInputs(&operation))
            capture(input);
    }
    if (!llvm::all_of(captures, [&](Value value) {
          return hasMapRelation(value.getType(), resultType);
        }) || !llvm::any_of(captures, [](Value value) {
          return isa<gpu::FragmentType>(value.getType());
        }))
      continue;
    OpBuilder builder(select);
    auto map = builder.create<MapElementwiseOp>(select.getLoc(), resultType,
                                                captures.getArrayRef());
    if (Attribute origin = select->getAttr(gpu::originAttr))
      map->setAttr(gpu::originAttr, origin);
    SmallVector<Type> types;
    for (Value value : captures)
      types.push_back(gpu::scalarCallbackType(value.getType()));
    Block *body = builder.createBlock(&map.getBody(), {}, types,
        SmallVector<Location>(types.size(), select.getLoc()));
    IRMapping mapping;
    mapping.map(captures.getArrayRef(), body->getArguments());
    auto conditional = builder.create<scf::IfOp>(select.getLoc(),
        TypeRange{resultType.getElementType()},
        mapping.lookup(select.getCondition()), /*withElseRegion=*/true);
    for (unsigned arm = 0; arm < 2; ++arm) {
      Block &branch = conditional->getRegion(arm).front();
      builder.setInsertionPointToStart(&branch);
      IRMapping branchMapping(mapping);
      for (Operation &operation : *block) {
        if (!exclusive[arm].contains(&operation))
          continue;
        Operation *clone = builder.clone(operation, branchMapping);
        clone->walk([&](Operation *nested) {
          for (Value result : nested->getResults())
            result.setType(gpu::scalarCallbackType(result.getType()));
          for (Region &region : nested->getRegions())
            for (Block &block : region)
              for (BlockArgument argument : block.getArguments())
              argument.setType(gpu::scalarCallbackType(argument.getType()));
        });
      }
      builder.create<scf::YieldOp>(select.getLoc(),
          branchMapping.lookup(select->getOperand(arm + 1)));
    }
    builder.setInsertionPointToEnd(body);
    builder.create<gpu::YieldOp>(select.getLoc(), conditional.getResults());
    SmallVector<Operation *> broadcasts;
    map.walk([&](Operation *operation) {
      if (isa<gpu::SplatOp, gpu::BroadcastOp>(operation))
        broadcasts.push_back(operation);
    });
    for (Operation *broadcast : llvm::reverse(broadcasts)) {
      broadcast->getResult(0).replaceAllUsesWith(broadcast->getOperand(0));
      broadcast->erase();
    }
    select.getResult().replaceAllUsesWith(map.getResult());
    select.erase();
    changed = true;
  }
  if (changed)
    gpu::eraseDeadPhysicalValues(kernel);
}

LogicalResult legalizeCollectiveCallbacks(func::FuncOp kernel) {
  SmallVector<Operation *> collectives;
  kernel.walk([&](Operation *operation) {
    if (isa<gpu::ReduceOp, gpu::ScanOp>(operation))
      collectives.push_back(operation);
  });
  for (Operation *operation : collectives) {
    auto reduce = dyn_cast<gpu::ReduceOp>(operation);
    auto scan = dyn_cast<gpu::ScanOp>(operation);
    unsigned count = reduce ? reduce.getSourceCount() : scan.getSourceCount();
    if ((reduce && (reduce.getAxes().size() != 1 || reduce.getCaptureCount())) ||
        (scan && scan.getCaptureCount()))
      return operation->emitOpError("native collective requires one axis and a capture-free callback");
    Region &region = reduce ? reduce.getCombine() : scan.getCombine();
    OpBuilder builder(operation);
    OperationState state(operation->getLoc(), reduce ? ReduceOp::getOperationName()
                                                    : ScanOp::getOperationName());
    state.addOperands(operation->getOperands());
    state.addTypes(operation->getResultTypes());
    state.addAttribute("source_count", builder.getI64IntegerAttr(count));
    state.addAttribute("axis", builder.getI64IntegerAttr(reduce ? reduce.getAxes().front() : scan.getAxis()));
    state.addAttribute("reverse", builder.getBoolAttr(scan && scan.getReverse()));
    if (Attribute origin = operation->getAttr(gpu::originAttr)) state.addAttribute(gpu::originAttr, origin);
    state.addRegion();
    Operation *native = builder.create(state);
    if (failed(gpu::scalarizeElementwiseCallback(region, native->getRegion(0))))
      return failure();
    SmallVector<Value> results(native->getResults());
    if (scan && !scan.getInclusive()) {
      builder.setInsertionPointAfter(native);
      Location location = scan.getLoc();
      for (unsigned component = 0; component < count; ++component) {
        auto type = dyn_cast<gpu::FragmentType>(results[component].getType());
        if (!type || scan.getAxis() >= type.getShape().size())
          return scan.emitOpError("exclusive scan requires a ranked physical result");
        auto axis = cast<gpu::AxisMapAttr>(type.getAxisMaps()[scan.getAxis()]);
        auto extent = cast<gpu::PhysicalExprAttr>(type.getShape()[scan.getAxis()]);
        auto ordinalAxis = gpu::AxisMapAttr::get(
            kernel.getContext(), axis.getSourceId(), axis.getSourceAxis(),
            axis.getDimensionId(), 0, axis.getDerived());
        auto rangeType = gpu::FragmentType::get(
            kernel.getContext(), builder.getIndexType(),
            builder.getArrayAttr({extent}), builder.getArrayAttr({ordinalAxis}),
            type.getValidity(), type.getOwner());
        auto indexType = gpu::FragmentType::get(
            kernel.getContext(), builder.getIndexType(), type.getShape(),
            type.getAxisMaps(), type.getValidity(), type.getOwner());
        auto predicateType = gpu::FragmentType::get(
            kernel.getContext(), builder.getI1Type(), type.getShape(),
            type.getAxisMaps(), type.getValidity(), type.getOwner());
        Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
        Value one = builder.create<arith::ConstantIndexOp>(location, 1);
        Value size = builder.create<gpu::PhysicalExprOp>(
            location, builder.getIndexType(), extent);
        Value ordinal = builder.create<gpu::MakeRangeOp>(
            location, rangeType, zero, size, one, zero, size,
            axis.getSourceId(), axis.getSourceAxis(), axis.getDerived());
        ordinal = builder.create<gpu::BroadcastOp>(location, indexType, ordinal);
        Value step = builder.create<gpu::BroadcastOp>(location, indexType, one);
        Value first = builder.create<gpu::BroadcastOp>(location, indexType, zero);
        Value shifted = builder.create<gpu::BinaryOp>(
            location, indexType, ordinal, step,
            scan.getReverse() ? BinaryOperator::Add : BinaryOperator::Subtract);
        Value bound = scan.getReverse()
                          ? Value(builder.create<gpu::BroadcastOp>(location, indexType, size))
                          : first;
        Value valid = builder.create<gpu::CompareOp>(
            location, predicateType, shifted, bound,
            scan.getReverse() ? ComparePredicate::Lt : ComparePredicate::Ge);
        Value safeIndex = builder.create<gpu::SelectOp>(
            location, indexType, valid, shifted, first);
        Value prefix = builder.create<gpu::GatherOp>(
            location, type, results[component], ValueRange{safeIndex},
            Value(), Value(), ArrayRef<int64_t>{static_cast<int64_t>(scan.getAxis())});
        Value identity = builder.create<gpu::BroadcastOp>(
            location, type, scan.getInputs()[count + component]);
        results[component] = builder.create<gpu::SelectOp>(
            location, type, valid, prefix, identity);
      }
    }
    operation->replaceAllUsesWith(results);
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

LogicalResult legalizeLargeScalarGathers(func::FuncOp kernel) {
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  for (gpu::GatherOp gather : gathers) {
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    if (!source || !isa<IntegerType, FloatType>(source.getElementType()) ||
        gather.getCoordinates().empty() ||
        gather.getCoordinates().size() != gather.getSourceAxes().size())
      continue;
    uint64_t bytes = (source.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    uint64_t limit = capabilities.getMaxDynamicSharedMemoryPerBlock();
    bool oversized = bytes > limit;
    bool fixed = true;
    for (Attribute attribute : source.getShape()) {
      auto extent = cast<gpu::PhysicalExprAttr>(attribute);
      if (extent.getKind() != static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) ||
          extent.getValue() <= 0) {
        fixed = false;
        break;
      }
      if (!oversized) {
        oversized = bytes > limit / extent.getValue();
        if (!oversized)
          bytes *= extent.getValue();
      }
    }
    if (!fixed || !oversized || (result && result.getShape().empty()))
      continue;
    SmallVector<std::pair<unsigned, Value>> selected;
    SmallVector<std::pair<unsigned, Value>> indexed;
    llvm::SmallDenseSet<unsigned> axes;
    llvm::SmallDenseSet<unsigned> seenAxes;
    bool compatible = true;
    for (auto [coordinate, axis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
      Value originalCoordinate = coordinate;
      coordinate = stripShapeOnly(coordinate);
      if (axis < 0 || static_cast<unsigned>(axis) >= source.getShape().size() ||
          !seenAxes.insert(axis).second) {
        compatible = false;
        break;
      }
      if (auto fragment = dyn_cast<gpu::FragmentType>(coordinate.getType())) {
        if (!isa<IntegerType, IndexType>(fragment.getElementType())) {
          compatible = false;
          break;
        }
        indexed.emplace_back(axis, originalCoordinate);
        continue;
      }
      if (!isa<IntegerType, IndexType>(coordinate.getType())) {
        compatible = false;
        break;
      }
      axes.insert(axis);
      selected.emplace_back(axis, coordinate);
    }
    if (!compatible || selected.empty())
      continue;
    unsigned resultAxis = 0;
    for (unsigned axis = 0; indexed.empty() && compatible && axis < source.getShape().size(); ++axis)
      if (!axes.contains(axis)) {
        compatible = result && resultAxis < result.getShape().size() &&
                     source.getShape()[axis] == result.getShape()[resultAxis];
        ++resultAxis;
      }
    if (!compatible || (indexed.empty() &&
                        (result ? resultAxis != result.getShape().size() : resultAxis != 0)))
      continue;

    // Exactly one source element contributes to each result lane. Combining
    // its integer representation with zero preserves NaNs and signed zero,
    // while native reductions avoid the whole-source scratch of a gather.
    OpBuilder builder(gather);
    Location location = gather.getLoc();
    auto bits = builder.getIntegerType(source.getElementType().getIntOrFloatBitWidth());
    auto bitsType = gpu::FragmentType::get(
        kernel.getContext(), bits, source.getShape(), source.getAxisMaps(),
        source.getValidity(), source.getOwner());
    Value value = builder.create<gpu::BitcastOp>(location, bitsType, gather.getSource());
    llvm::sort(selected, [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
    for (auto [axis, coordinate] : selected) {
      // Select one axis at a time so later selections consume the reduced
      // fragment rather than constructing another full-source predicate.
      auto input = cast<gpu::FragmentType>(value.getType());
      auto mapping = cast<gpu::AxisMapAttr>(input.getAxisMaps()[axis]);
      auto ordinalAxis = gpu::AxisMapAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), 0, mapping.getDerived());
      auto rangeType = gpu::FragmentType::get(
          kernel.getContext(), builder.getIndexType(),
          builder.getArrayAttr({input.getShape()[axis]}),
          builder.getArrayAttr({ordinalAxis}), input.getValidity(), input.getOwner());
      Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
      Value one = builder.create<arith::ConstantIndexOp>(location, 1);
      Value size = builder.create<gpu::PhysicalExprOp>(
          location, builder.getIndexType(), cast<gpu::PhysicalExprAttr>(input.getShape()[axis]));
      Value ordinal = builder.create<gpu::MakeRangeOp>(
          location, rangeType, zero, size, one, zero, size,
          mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived());
      if (!coordinate.getType().isIndex())
        coordinate = builder.create<gpu::CastOp>(location, builder.getIndexType(), coordinate);
      Value index = builder.create<gpu::BroadcastOp>(location, rangeType, coordinate);
      auto predicate = gpu::FragmentType::get(
          kernel.getContext(), builder.getI1Type(), rangeType.getShape(),
          rangeType.getAxisMaps(), rangeType.getValidity(), rangeType.getOwner());
      Value equal = builder.create<gpu::CompareOp>(
          location, predicate, ordinal, index, ComparePredicate::Eq);
      auto expanded = gpu::projectPredicateToFragmentAxis(
          builder, location, equal, input, axis);
      if (failed(expanded))
        return gather.emitOpError("scalar selection lost its source-axis projection");
      auto emptyBits = zeroLike(builder, location, input);
      if (failed(emptyBits))
        return failure();
      value = builder.create<gpu::SelectOp>(location, input, *expanded, value, *emptyBits);
      SmallVector<Attribute> shape, mappings;
      for (unsigned sourceAxis = 0; sourceAxis < input.getShape().size(); ++sourceAxis) {
        if (sourceAxis == axis)
          continue;
        auto mapping = cast<gpu::AxisMapAttr>(input.getAxisMaps()[sourceAxis]);
        mappings.push_back(gpu::AxisMapAttr::get(
            kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), shape.size(), mapping.getDerived()));
        shape.push_back(input.getShape()[sourceAxis]);
      }
      Type reducedType = bits;
      if (!shape.empty())
        reducedType = gpu::FragmentType::get(
            kernel.getContext(), bits, builder.getArrayAttr(shape),
            builder.getArrayAttr(mappings), input.getValidity(), input.getOwner());
      auto identity = zeroLike(builder, location, reducedType);
      if (failed(identity))
        return failure();
      OperationState state(location, gpu::ReduceOp::getOperationName());
      state.addOperands({value, *identity});
      state.addTypes(reducedType);
      state.addAttribute("axes", builder.getDenseI64ArrayAttr({axis}));
      state.addAttribute("source_count", builder.getI64IntegerAttr(1));
      state.addAttribute("identity_count", builder.getI64IntegerAttr(1));
      state.addAttribute("capture_count", builder.getI64IntegerAttr(0));
      state.addRegion();
      auto reduction = cast<gpu::ReduceOp>(builder.create(state));
      {
        OpBuilder::InsertionGuard guard(builder);
        Block *body = builder.createBlock(&reduction.getCombine(), {},
            {reducedType, reducedType}, {location, location});
        Value combined = builder.create<gpu::BinaryOp>(
            location, reducedType, body->getArgument(0), body->getArgument(1),
            BinaryOperator::BitwiseOr);
        builder.create<gpu::YieldOp>(location, combined);
      }
      value = reduction.getResult(0);
    }
    Type decodedType = source.getElementType();
    if (auto fragment = dyn_cast<gpu::FragmentType>(value.getType()))
      decodedType = gpu::FragmentType::get(
          kernel.getContext(), source.getElementType(), fragment.getShape(),
          fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
    value = builder.create<gpu::BitcastOp>(location, decodedType, value);
    if (!indexed.empty()) {
      SmallVector<Value> coordinates;
      SmallVector<int64_t> sourceAxes;
      for (auto [axis, coordinate] : indexed) {
        unsigned removed = llvm::count_if(selected, [&](const auto &item) {
          return item.first < axis;
        });
        coordinates.push_back(coordinate);
        sourceAxes.push_back(axis - removed);
      }
      value = builder.create<gpu::GatherOp>(location, gather.getResult().getType(),
          value, coordinates, gather.getValid(), gather.getFill(), sourceAxes);
      if (Attribute origin = gather->getAttr(gpu::originAttr))
        value.getDefiningOp()->setAttr(gpu::originAttr, origin);
      gather.getResult().replaceAllUsesWith(value);
      gather.erase();
      continue;
    }
    if (value.getType() != gather.getResult().getType()) {
      auto reassociation = gpu::inferReshapeReassociation(
          cast<gpu::FragmentType>(value.getType()), result);
      if (failed(reassociation))
        return gather.emitOpError("scalar selection lost its result relation");
      value = builder.create<gpu::ReshapeOp>(location, result, value, *reassociation);
    }
    if (gather.getValid())
      value = builder.create<gpu::SelectOp>(
          location, gather.getResult().getType(), gather.getValid(), value, gather.getFill());
    gather.getResult().replaceAllUsesWith(value);
    gather.erase();
  }
  gpu::eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult materializeOversizedGathers(func::FuncOp kernel) {
  auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
  auto tuples = kernel->getAttrOfType<ArrayAttr>(gpu::sharedConfigTuplesAttr);
  if (!space || space.size() != 1 || !tuples || tuples.empty())
    return success();
  int64_t maximumPrograms = 0;
  for (Attribute attribute : tuples) {
    auto tuple = cast<DictionaryAttr>(attribute);
    TritonConfig config;
    kernel.walk([&](gpu::ParameterOp parameter) {
      auto schema = parameter.getParameter();
      if (schema.getCandidates().size() == 1)
        config.kernelParameters[schema.getName().getValue().str()] =
            schema.getCandidates()[0];
    });
    for (NamedAttribute entry : tuple)
      config.kernelParameters[entry.getName().getValue().str()] =
          cast<IntegerAttr>(entry.getValue()).getInt();
    auto count = evaluateCompileTimeExpression(
        cast<gpu::PhysicalExprAttr>(space[0]), config);
    if (!count || *count <= 0)
      return success();
    maximumPrograms = std::max(maximumPrograms, *count);
  }
  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  llvm::MapVector<Value, SmallVector<gpu::GatherOp>> readers;
  kernel.walk([&](gpu::GatherOp gather) {
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    if (!source || gather.getCoordinates().empty() ||
        belongsToSplitGatherPair(gather) ||
        !isa<FloatType, IntegerType>(source.getElementType()))
      return;
    int64_t bytes = (source.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    for (Attribute attribute : source.getShape()) {
      auto extent = cast<gpu::PhysicalExprAttr>(attribute);
      if (extent.getKind() != static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) ||
          extent.getValue() <= 0 ||
          bytes > std::numeric_limits<int64_t>::max() / extent.getValue())
        return;
      bytes *= extent.getValue();
    }
    if (bytes <= capabilities.getMaxDynamicSharedMemoryPerBlock())
      return;
    SmallVector<Value> selected(source.getShape().size());
    for (auto [coordinate, axis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes()))
      selected[axis] = coordinate;
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis) {
      Type coordinateType;
      if (selected[axis]) {
        coordinateType = selected[axis].getType();
      } else {
        if (!result)
          return;
        auto mapping = cast<gpu::AxisMapAttr>(source.getAxisMaps()[axis]);
        auto projection = gpu::queryFragmentAxis(result, gpu::sourceAxisIdentity(mapping));
        if (!projection.isExact() ||
            projection.dimensionId != mapping.getDimensionId() ||
            result.getShape()[projection.fragmentAxis] != source.getShape()[axis])
          return;
        auto ordinal = gpu::AxisMapAttr::get(kernel.getContext(),
            mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDimensionId(),
            0, mapping.getDerived());
        coordinateType = gpu::FragmentType::get(kernel.getContext(),
            IndexType::get(kernel.getContext()), ArrayAttr::get(kernel.getContext(),
                {source.getShape()[axis]}), ArrayAttr::get(kernel.getContext(), {ordinal}),
            source.getValidity(), source.getOwner());
      }
      if (auto coordinate = dyn_cast<gpu::FragmentType>(coordinateType)) {
        if (!result)
          return;
        auto projected = gpu::FragmentType::get(kernel.getContext(),
            coordinate.getElementType(), result.getShape(), result.getAxisMaps(),
            result.getValidity(), result.getOwner());
        if (!gpu::queryBroadcastProjection(coordinate, projected).isExact())
          return;
      }
    }
    readers[gather.getSource()].push_back(gather);
  });
  if (readers.empty())
    return success();
  OpBuilder entry(&kernel.front(), kernel.front().begin());
  gpu::ProgramIdOp programId;
  kernel.walk([&](gpu::ProgramIdOp operation) {
    if (operation.getAxis() == 0)
      programId = operation;
  });
  if (programId) {
    if (programId->getBlock() != &kernel.front() ||
        programId.getOperation() != &kernel.front().front())
      programId->moveBefore(&kernel.front(), kernel.front().begin());
  } else {
    programId = entry.create<gpu::ProgramIdOp>(kernel.getLoc(), entry.getIndexType(), 0);
  }
  Value program = programId.getResult();
  entry.setInsertionPointAfter(programId);
  auto prefix = gpu::PhysicalExprAttr::get(kernel.getContext(),
      static_cast<uint32_t>(gpu::PhysicalExprKind::Constant), maximumPrograms,
      entry.getStringAttr(""), entry.getArrayAttr({}));
  for (auto &readerGroup : readers) {
    auto &gathers = readerGroup.second;
    Value source = gathers.front().getSource();
    auto payload = cast<gpu::FragmentType>(source.getType());
    SmallVector<Attribute> shape{prefix};
    llvm::append_range(shape, payload.getShape());
    Value workspace = gpu::createInvocationWorkspace(
        kernel, source.getLoc(), payload.getElementType(),
        entry.getArrayAttr(shape), payload.getOwner());
    // The maximum is evaluated over every shared tuple; the private prefix
    // remains valid while Triton chooses its local configuration.
    entry.create<gpu::AssumeInBoundsOp>(source.getLoc(), program, workspace, 0);
    OpBuilder builder(kernel.getContext());
    if (Operation *definition = source.getDefiningOp())
      builder.setInsertionPointAfter(definition);
    else
      builder.setInsertionPointToStart(cast<BlockArgument>(source).getOwner());
    SmallVector<Value> coordinates{program};
    SmallVector<int64_t> sourceAxes{0};
    SmallVector<Value> ordinals;
    for (auto [axis, attribute] : llvm::enumerate(payload.getShape())) {
      auto mapping = cast<gpu::AxisMapAttr>(payload.getAxisMaps()[axis]);
      auto ordinalMap = gpu::AxisMapAttr::get(kernel.getContext(),
          mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDimensionId(),
          0, mapping.getDerived());
      auto type = gpu::FragmentType::get(kernel.getContext(), builder.getIndexType(),
          builder.getArrayAttr({attribute}), builder.getArrayAttr({ordinalMap}),
          payload.getValidity(), payload.getOwner());
      Value zero = builder.create<arith::ConstantIndexOp>(source.getLoc(), 0);
      Value one = builder.create<arith::ConstantIndexOp>(source.getLoc(), 1);
      Value size = builder.create<gpu::PhysicalExprOp>(source.getLoc(),
          builder.getIndexType(), cast<gpu::PhysicalExprAttr>(attribute));
      Value ordinal = builder.create<gpu::MakeRangeOp>(source.getLoc(), type,
          zero, size, one, zero, size, mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDerived());
      ordinals.push_back(ordinal);
      coordinates.push_back(ordinal);
      sourceAxes.push_back(axis + 1);
    }
    builder.create<gpu::StoreOp>(source.getLoc(), workspace, coordinates, source,
                                 Value(), sourceAxes);
    for (gpu::GatherOp gather : gathers) {
      builder.setInsertionPoint(gather);
      auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
      SmallVector<Value> selected(payload.getShape().size());
      for (auto [coordinate, axis] :
           llvm::zip(gather.getCoordinates(), gather.getSourceAxes()))
        selected[axis] = coordinate;
      SmallVector<Value> access{program};
      for (unsigned axis = 0; axis < payload.getShape().size(); ++axis) {
        Value coordinate = selected[axis] ? selected[axis] : ordinals[axis];
        if (result) {
          auto indices = gpu::FragmentType::get(kernel.getContext(),
              elementType(coordinate.getType()), result.getShape(), result.getAxisMaps(),
              result.getValidity(), result.getOwner());
          if (coordinate.getType() != indices)
            coordinate = builder.create<gpu::BroadcastOp>(gather.getLoc(), indices, coordinate);
        }
        access.push_back(coordinate);
      }
      auto load = builder.create<gpu::LoadOp>(gather.getLoc(), gather.getResult().getType(), workspace,
          access, gather.getValid(), gather.getFill(), sourceAxes);
      if (Attribute origin = gather->getAttr(gpu::originAttr))
        load->setAttr(gpu::originAttr, origin);
      gather.getResult().replaceAllUsesWith(load.getResult());
      gather.erase();
    }
  }
  gpu::eraseDeadPhysicalValues(kernel);
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
    if (auto assertion = dyn_cast<cf::AssertOp>(operation)) {
      auto compare = assertion.getArg().getDefiningOp<gpu::CompareOp>();
      if (!compare || !llvm::all_of(compare->getOperands(), [&](Value operand) {
            if (auto expression = operand.getDefiningOp<gpu::PhysicalExprOp>())
              return isTritonFragmentExtent(expression.getExpression());
            return operand.getDefiningOp<arith::ConstantOp>() != nullptr;
          })) {
        assertion.emitOpError("Triton assertions require a constexpr condition");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
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
      if (form && form.getValue() != "multiply_sum" &&
          form.getValue() != "fma") {
        contract.emitOpError("has an unknown Triton contraction form");
        return WalkResult::interrupt();
      }
      if (form && form.getValue() == "fma" &&
          !contract.getAccumulator().getType().getElementType().isF32()) {
        contract.emitOpError(
            "Triton explicit FMA contraction requires f32 accumulation");
        return WalkResult::interrupt();
      }
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
    } else if (isa<gpu::ReduceOp, gpu::ScanOp>(operation)) {
      operation->emitOpError(
          "Triton collective requires an explicitly scalarized callback");
      return WalkResult::interrupt();
    } else if (auto contract = dyn_cast<gpu::ScaledContractOp>(operation)) {
      unsigned lhsRank = contract.getLhs().getType().getShape().size();
      unsigned rhsRank = contract.getRhs().getType().getShape().size();
      Type lhsScaleElement =
          contract.getLhsScale().getType().getElementType();
      Type rhsScaleElement =
          contract.getRhsScale().getType().getElementType();
      if (lhsRank != 3 || rhsRank != 3 ||
          contract.getLhsFormat() == ScaledFormat::E8M0 ||
          contract.getRhsFormat() == ScaledFormat::E8M0 ||
          contract.getLhsGroupSize() != 32 ||
          contract.getRhsGroupSize() != 32 ||
          !lhsScaleElement.isUnsignedInteger(8) ||
          !rhsScaleElement.isUnsignedInteger(8) ||
          !contract.getResult().getType().getElementType().isF32() ||
          contract.getLhsReductionAxes() !=
              ArrayRef<int64_t>{1, 2} ||
          contract.getRhsReductionAxes() !=
              ArrayRef<int64_t>{0, 1}) {
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

    if (isa<CtaBarrierOp, gpu::ViewOverlapOp, TensorDescriptorChoiceOp, TensorDescriptorAllocatorOp,
            TensorDescriptorOp, BlockLoadOp, BlockStoreOp, DescriptorLoadOp,
            DescriptorStoreOp, SplitOp, ReduceOp, ScanOp, MapElementwiseOp, gpu::ParameterOp,
            gpu::PhysicalExprOp,
            gpu::ProgramIdOp,
            gpu::WorksetCoordinateOp, gpu::DelinearizeOp,
            gpu::DimOp, gpu::RangeOp, gpu::RangeBoundOp, gpu::MakeRangeOp,
            gpu::SplatOp, gpu::BroadcastOp, gpu::UnaryOp, gpu::BinaryOp,
            gpu::CompareOp, gpu::SelectOp, gpu::CastOp, gpu::BitcastOp,
            gpu::ReshapeOp, gpu::TransposeOp, gpu::JoinOp, gpu::MakeRecordOp,
            gpu::ExtractOp, gpu::LoadOp, gpu::GatherOp, gpu::StoreOp,
            gpu::ContractOp,
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

bool viewsMayAlias(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  auto left = cast<gpu::ViewType>(lhs.getType());
  auto right = cast<gpu::ViewType>(rhs.getType());
  return !left.getLayout().getNoalias() && !right.getLayout().getNoalias();
}

bool hasOrderedViewDependencies(func::FuncOp kernel) {
  llvm::DenseSet<Value> reads, writes;
  kernel.walk([&](gpu::LoadOp load) {
    if (isa<gpu::ViewType>(load.getResource().getType()))
      reads.insert(load.getResource());
  });
  kernel.walk([&](gpu::StoreOp store) {
    if (isa<gpu::ViewType>(store.getResource().getType()))
      writes.insert(store.getResource());
  });
  return llvm::any_of(reads, [&](Value resource) {
    return llvm::any_of(writes, [&](Value other) { return viewsMayAlias(resource, other); });
  });
}

LogicalResult legalizeOrderedViewDependencies(func::FuncOp kernel) {
  if (!hasOrderedViewDependencies(kernel))
    return success();
  using Accesses = llvm::DenseMap<Value, unsigned>;
  auto accesses = [](Operation *operation) {
    Accesses modes;
    operation->walk([&](Operation *nested) {
      if (auto load = dyn_cast<gpu::LoadOp>(nested)) {
        if (isa<gpu::ViewType>(load.getResource().getType()))
          modes[load.getResource()] |= 1;
      } else if (auto store = dyn_cast<gpu::StoreOp>(nested)) {
        if (isa<gpu::ViewType>(store.getResource().getType()))
          modes[store.getResource()] |= 2;
      }
    });
    return modes;
  };
  auto conflicts = [](const Accesses &pending, const Accesses &current) {
    for (auto [resource, mode] : pending)
      for (auto [other, otherMode] : current)
        if (((mode | otherMode) & 2) && viewsMayAlias(resource, other))
          return true;
    return false;
  };
  llvm::DenseMap<Value, Value> nonOverlappingViews;
  auto disjointLoopAccesses = [&](Block &body, const Accesses &bodyAccesses) {
    llvm::DenseMap<Value, Value> conditions;
    auto loop = dyn_cast<scf::ForOp>(body.getParentOp());
    if (!loop || !loop.getUpperBound().getType().isIndex())
      return conditions;
    auto stepBounds = gpu::queryPositiveExtentBounds(
        gpu::queryLaunchExpression(loop.getStep()), kernel);
    if (!stepBounds)
      return conditions;
    int64_t maximumStep = stepBounds->second;
    bool modeledEffects = true;
    body.walk([&](Operation *operation) {
      if (!isa<gpu::LoadOp, gpu::StoreOp, scf::ForOp, scf::IfOp,
               scf::WhileOp>(operation) && !isMemoryEffectFree(operation))
        modeledEffects = false;
    });
    if (!modeledEffects)
      return conditions;
    llvm::DenseMap<Value, bool> varying;
    Value safeRange;
    for (auto [resource, mode] : bodyAccesses) {
      auto argument = dyn_cast<BlockArgument>(resource);
      auto view = dyn_cast<gpu::ViewType>(resource.getType());
      if (!(mode & 2) || !argument || argument.getOwner() != &kernel.front() ||
          !view || !view.getLayout().getHasStrides() ||
          view.getLayout().getStrides().size() != view.getRank())
        continue;
      gpu::StoreOp selected;
      unsigned stores = 0;
      body.walk([&](gpu::StoreOp store) {
        if (store.getResource() == resource) {
          selected = store;
          ++stores;
        }
      });
      if (stores != 1 || selected->getBlock() != &body)
        continue;
      auto chunkAxis = [&](ValueRange coordinates,
                           ArrayRef<int64_t> sourceAxes) -> std::optional<int64_t> {
        std::optional<int64_t> axis;
        for (auto [coordinate, sourceAxis] : llvm::zip(coordinates, sourceAxes)) {
          Value root = coordinate;
          while (true) {
            if (auto broadcast = root.getDefiningOp<gpu::BroadcastOp>())
              root = broadcast.getValue();
            else if (auto reshape = root.getDefiningOp<gpu::ReshapeOp>())
              root = reshape.getValue();
            else
              break;
          }
          auto range = root.getDefiningOp<gpu::MakeRangeOp>();
          if (root == loop.getInductionVar() ||
              (range && range.getStart() == loop.getInductionVar() &&
               gpu::isUnitStepRange(range) &&
               gpu::samePhysicalScalarExpression(range.getExtent(),
                                                  loop.getStep()))) {
            if (axis)
              return std::nullopt;
            axis = sourceAxis;
          } else if (gpu::variesWithIteration(root, loop, varying)) {
            return std::nullopt;
          }
        }
        return axis;
      };
      auto storeAxis = chunkAxis(selected.getCoordinates(), selected.getSourceAxes());
      if (!storeAxis)
        continue;
      bool disjoint = true;
      body.walk([&](gpu::LoadOp load) {
        if (load.getResource() == resource &&
            chunkAxis(load.getCoordinates(), load.getSourceAxes()) != storeAxis)
          disjoint = false;
      });
      if (!disjoint)
        continue;
      Value &layout = nonOverlappingViews[resource];
      if (!layout) {
        auto proof = gpu::materializeNonOverlappingView(kernel, resource);
        if (failed(proof))
          continue;
        layout = *proof;
      }
      OpBuilder builder(loop);
      if (!safeRange) {
        // Include the final padded chunk and the terminating IV update in
        // the no-wrap proof; access masks can only narrow each chunk.
        Value limit = builder.create<arith::ConstantIndexOp>(
            loop.getLoc(), std::numeric_limits<int64_t>::max() - maximumStep);
        safeRange = builder.create<gpu::CompareOp>(
            loop.getLoc(), builder.getI1Type(), loop.getUpperBound(), limit,
            ComparePredicate::Le);
      }
      conditions[resource] = builder.create<gpu::BinaryOp>(
          loop.getLoc(), builder.getI1Type(), layout, safeRange,
          BinaryOperator::LogicalAnd);
    }
    return conditions;
  };
  const llvm::DenseMap<Value, Value> noDisjointIterations;
  std::map<std::pair<unsigned, unsigned>, Value> overlapFacts;
  auto insertBarrier = [&](OpBuilder &builder, Location location,
                           Accesses &pending, const Accesses &current,
                           const llvm::DenseMap<Value, Value> &disjointIterations) {
    SmallVector<std::pair<unsigned, unsigned>> pairs;
    Value condition;
    auto externalArgument = [&](Value value) -> BlockArgument {
      auto argument = dyn_cast<BlockArgument>(value);
      if (!argument || argument.getOwner() != &kernel.front())
        return {};
      auto kind = kernel.getArgAttrOfType<StringAttr>(argument.getArgNumber(),
                                                    gpu::abiKindAttr);
      return kind && kind.getValue() == "view" ? argument : BlockArgument();
    };
    for (auto [resource, mode] : pending)
      for (auto [other, otherMode] : current) {
        if (!((mode | otherMode) & 2) || !viewsMayAlias(resource, other))
          continue;
        if (resource == other)
          if (auto proof = disjointIterations.find(resource);
              proof != disjointIterations.end()) {
            Value zero = builder.create<arith::ConstantIntOp>(location, 0, 1);
            Value needed = builder.create<gpu::CompareOp>(
                location, builder.getI1Type(), proof->second, zero,
                ComparePredicate::Eq);
            condition = condition
                            ? Value(builder.create<gpu::BinaryOp>(
                                  location, builder.getI1Type(), condition,
                                  needed, BinaryOperator::LogicalOr))
                            : needed;
            continue;
          }
        auto left = externalArgument(resource);
        auto right = externalArgument(other);
        if (resource == other || !left || !right) {
          builder.create<CtaBarrierOp>(location);
          pending.clear();
          return;
        }
        unsigned lhs = left.getArgNumber(), rhs = right.getArgNumber();
        pairs.emplace_back(std::min(lhs, rhs), std::max(lhs, rhs));
      }
    llvm::sort(pairs);
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    for (auto pair : pairs) {
      Value &overlap = overlapFacts[pair];
      if (!overlap) {
        OpBuilder entry(&kernel.front(), kernel.front().begin());
        overlap = entry.create<gpu::ViewOverlapOp>(
            location, entry.getI1Type(), kernel.getArgument(pair.first),
            kernel.getArgument(pair.second));
      }
      condition = condition
                      ? Value(builder.create<gpu::BinaryOp>(
                            location, builder.getI1Type(), condition, overlap,
                            BinaryOperator::LogicalOr))
                      : overlap;
    }
    auto guard = builder.create<scf::IfOp>(location, condition, false);
    OpBuilder nested = guard.getThenBodyBuilder();
    nested.create<CtaBarrierOp>(location);
    // A false guard performs no synchronization: keep all outstanding accesses
    // so a later operation cannot lose an unrelated dependency.
  };
  llvm::DenseSet<Value> visiting;
  std::function<bool(Value)> uniform = [&](Value value) {
    if (isa<gpu::FragmentType, gpu::RecordType>(value.getType()))
      return false;
    if (!visiting.insert(value).second)
      return true;
    bool result = false;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Operation *parent = argument.getOwner()->getParentOp();
      if (isa<func::FuncOp>(parent)) {
        result = true;
      } else if (auto loop = dyn_cast<scf::ForOp>(parent)) {
        result = uniform(loop.getLowerBound()) && uniform(loop.getUpperBound()) &&
                 uniform(loop.getStep());
        if (result && argument.getArgNumber()) {
          unsigned index = argument.getArgNumber() - 1;
          result = uniform(loop.getInitArgs()[index]) &&
                   uniform(loop.getBody()->getTerminator()->getOperand(index));
        }
      } else if (auto loop = dyn_cast<scf::WhileOp>(parent)) {
        auto condition = cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
        unsigned index = argument.getArgNumber();
        if (argument.getOwner() == &loop.getBefore().front())
          result = uniform(loop.getInits()[index]) &&
                   uniform(loop.getAfter().front().getTerminator()->getOperand(index));
        else
          result = uniform(condition.getCondition()) && uniform(condition.getArgs()[index]);
      }
    } else if (Operation *producer = value.getDefiningOp()) {
      if (auto loop = dyn_cast<scf::ForOp>(producer)) {
        auto index = cast<OpResult>(value).getResultNumber();
        result = uniform(loop.getRegionIterArgs()[index]);
      } else if (auto loop = dyn_cast<scf::WhileOp>(producer)) {
        auto condition = cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
        unsigned index = cast<OpResult>(value).getResultNumber();
        result = uniform(condition.getCondition()) && uniform(condition.getArgs()[index]);
      } else if (auto branch = dyn_cast<scf::IfOp>(producer)) {
        unsigned index = cast<OpResult>(value).getResultNumber();
        result = uniform(branch.getCondition()) &&
                 uniform(branch.thenBlock()->getTerminator()->getOperand(index)) &&
                 uniform(branch.elseBlock()->getTerminator()->getOperand(index));
      } else if (isa<gpu::ReduceOp, gpu::GatherOp, gpu::DimOp, gpu::ParameterOp,
              gpu::PhysicalExprOp, gpu::ProgramIdOp, gpu::ViewOverlapOp,
              arith::ConstantOp>(producer)) {
        result = true;
      } else if (isa<gpu::LoadOp, gpu::UnaryOp, gpu::BinaryOp, gpu::CompareOp,
                     gpu::SelectOp, gpu::CastOp, gpu::BitcastOp,
                     gpu::WorksetCoordinateOp, gpu::DelinearizeOp>(producer)) {
        result = llvm::all_of(producer->getOperands(), uniform);
      }
    }
    visiting.erase(value);
    return result;
  };
  std::function<LogicalResult(Block &, bool, bool, Accesses &)> synchronize =
      [&](Block &block, bool loopBody, bool uniformControl,
          Accesses &pending) -> LogicalResult {
    Accesses bodyAccesses;
    for (Operation &operation : block)
      for (auto [resource, mode] : accesses(&operation))
        bodyAccesses[resource] |= mode;
    auto disjointIterations = loopBody ? disjointLoopAccesses(block, bodyAccesses)
                                      : llvm::DenseMap<Value, Value>();
    for (Operation &operation : llvm::make_early_inc_range(block.without_terminator())) {
      if (isa<CtaBarrierOp>(operation)) {
        pending.clear();
        continue;
      }
      Accesses current = accesses(&operation);
      if (conflicts(pending, current)) {
        if (!uniformControl)
          return operation.emitOpError("ordered view dependency requires uniform CTA control before synchronization");
        OpBuilder builder(&operation);
        insertBarrier(builder, operation.getLoc(), pending, current,
                      noDisjointIterations);
      }
      if (operation.getNumRegions()) {
        bool nestedUniform = uniformControl;
        if (auto branch = dyn_cast<scf::IfOp>(operation))
          nestedUniform &= uniform(branch.getCondition());
        else if (auto loop = dyn_cast<scf::ForOp>(operation))
          nestedUniform &= uniform(loop.getInductionVar());
        else if (auto loop = dyn_cast<scf::WhileOp>(operation))
          nestedUniform &= uniform(
              cast<scf::ConditionOp>(loop.getBefore().front().getTerminator()).getCondition());
        for (Region &region : operation.getRegions())
          for (Block &nested : region) {
            Accesses outstanding;
            if (failed(synchronize(nested, isa<scf::ForOp, scf::WhileOp>(operation),
                                   nestedUniform, outstanding)))
              return failure();
            for (auto [resource, mode] : outstanding)
              pending[resource] |= mode;
          }
      } else {
        for (auto [resource, mode] : current)
          pending[resource] |= mode;
      }
    }
    if (loopBody && conflicts(pending, bodyAccesses)) {
      if (!uniformControl)
        return block.getTerminator()->emitOpError("loop-carried view dependency requires uniform CTA control before synchronization");
      OpBuilder builder(block.getTerminator());
      insertBarrier(builder, block.getTerminator()->getLoc(), pending,
                    bodyAccesses, disjointIterations);
    }
    return success();
  };
  Accesses pending;
  return synchronize(kernel.front(), false, true, pending);
}

void selectOrderedLoadUnrolling(func::FuncOp kernel) {
  UniformValueAnalysis constants(gpu::describeUniformValue);
  auto integer = [&](Value value) {
    return dyn_cast_or_null<IntegerAttr>(constants.evaluate(value));
  };
  kernel.walk([&](scf::ForOp loop) {
    if (loop.getNumResults() != 1 ||
        !isa<FloatType>(gpu::uniformElementType(loop.getResult(0).getType())))
      return;
    auto lower = integer(loop.getLowerBound());
    auto step = integer(loop.getStep());
    if (!lower || !lower.getValue().isZero() || !step ||
        !step.getValue().isOne() ||
        !gpu::queryNonNegativeIndexUpperBound(loop.getUpperBound()))
      return;
    int64_t factor = 4;
    if (auto upper = integer(loop.getUpperBound())) {
      if (upper.getInt() <= 1)
        return;
      factor = std::min<int64_t>(factor, upper.getInt());
    }

    unsigned loads = 0;
    bool product = false;
    Value carry = loop.getRegionIterArgs().front();
    for (Operation &operation : loop.getBody()->without_terminator()) {
      if (operation.getNumRegions() ||
          isa<gpu::ContractOp, gpu::ScaledContractOp, gpu::SparseContractOp,
              gpu::HistogramOp>(operation))
        return;
      if (auto load = dyn_cast<gpu::LoadOp>(operation)) {
        for (Value operand : load->getOperands()) {
          llvm::SmallDenseSet<Value, 32> visited;
          if (valueDependsOn(operand, carry, loop, visited))
            return;
        }
        ++loads;
      } else if (!isMemoryEffectFree(&operation)) {
        return;
      }
      if (auto binary = dyn_cast<gpu::BinaryOp>(operation))
        product |= binary.getOperatorKind() == BinaryOperator::Multiply &&
                   isa<FloatType>(
                       gpu::uniformElementType(binary.getResult().getType()));
    }
    if (loads == 0 || loads > 2 || !product)
      return;
    // Native unrolling preserves the accumulator chain and handles the tail;
    // independent reads from later iterations can overlap the current update.
    loop->setAttr("intent_gpu.triton.loop_unroll_factor",
                  IntegerAttr::get(IntegerType::get(kernel.getContext(), 32),
                                   factor));
  });
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
  if (failed(gpu::materializeProgramBuffers(module)) ||
      failed(materializeOversizedGathers(kernel)) ||
      failed(legalizeLargeScalarGathers(kernel)) ||
      failed(legalizeMaskedGather(kernel)) ||
      failed(legalizeExpandingGathers(kernel)) ||
      failed(legalizeScatterAdd(kernel)) ||
      failed(gpu::normalizeMatrixContractShapes(kernel)) ||
      failed(gpu::verifyGPUProgram(module)))
    return failure();
  bool requiresCtaSynchronization = llvm::any_of(kernel.getArgumentTypes(), [](Type type) {
    return isa<gpu::BufferType>(type);
  }) || hasOrderedViewDependencies(kernel);
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
  bool mayFormDot = false;
  kernel.walk([&](Operation *operation) {
    // Triton can combine a broadcast-multiply-reduce into a dot later.
    mayFormDot |= isa<gpu::ReduceOp, ReduceOp>(operation);
    if (isa<gpu::ReduceOp, gpu::ScanOp>(operation) &&
        !llvm::is_contained(categories, gpu::ParameterCategory::Reduction))
      categories.push_back(gpu::ParameterCategory::Reduction);
  });
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
  if (hasContraction &&
      !llvm::is_contained(categories, gpu::ParameterCategory::Contraction))
    categories.push_back(gpu::ParameterCategory::Contraction);
  auto rows = profiles.get("triton", localOptionsFamily(
      categories, twoAxisPointwise, blackwell && hasRecurrentContraction(kernel),
      hasContraction && allFp32), kernel.getLoc());
  if (failed(rows))
    return failure();
  bool straightLinePointwise =
      llvm::is_contained(categories, gpu::ParameterCategory::Pointwise) &&
      llvm::all_of(categories, [](gpu::ParameterCategory category) {
        return category == gpu::ParameterCategory::Pointwise;
      }) && llvm::all_of(kernel.front(), [](Operation &operation) {
        return operation.getNumRegions() == 0;
      });
  bool pipelineStagesAffectProgram = hasContraction && !allFp32;
  kernel.walk([&](Operation *operation) {
    pipelineStagesAffectProgram |=
        isa<gpu::ScaledContractOp, gpu::SparseContractOp>(operation) ||
        ((hasContraction || mayFormDot) &&
         operation->getParentOfType<scf::ForOp>() &&
         !isMemoryEffectFree(operation));
  });
  SmallVector<TritonLocalOptions> localOptions;
  SmallVector<int64_t> warpDomain, stageDomain, ctaDomain;
  auto isDeviceOption = [&](int64_t warps, int64_t stages, int64_t ctas) {
    return (!requiresCtaSynchronization || ctas == 1) &&
           (warps & (warps - 1)) == 0 &&
           warps <= capabilities.getMaxThreadsPerBlock() / 32 &&
           stages <= std::numeric_limits<int32_t>::max() &&
           (ctas & (ctas - 1)) == 0 && ctas <= 16 &&
           (ctas == 1 || capabilities.getComputeCapabilityMajor() >= 9);
  };
  for (const auto &row : *rows) {
    int64_t warps = row[0], stages = row[1], ctas = row[2];
    if (straightLinePointwise)
      stages = 1;
    if (!isDeviceOption(warps, stages, ctas))
      continue;
    if (llvm::any_of(localOptions, [&](const TritonLocalOptions &option) {
          // Kernel-level stages pipeline dot producers. Ordinary loads need
          // an explicit tl.range stage binding; keep one supplied setting
          // when the current program has no such pipeline producer.
          return option.warps == warps &&
                 (!pipelineStagesAffectProgram || option.stages == stages) &&
                 option.ctas == ctas;
        }))
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
      failed(gpu::lowerInvocationWorkspaces(module)))
    return failure();
  if (failed(legalizeOrderedViewDependencies(kernel)) ||
      failed(legalizeSplitGatherPairs(kernel)))
    return failure();
  selectOrderedLoadUnrolling(kernel);
  FailureOr<TensorDescriptorChoiceOp> tensorDescriptorForms =
      materializeTensorDescriptorForms(kernel, localOptions);
  if (failed(tensorDescriptorForms))
    return failure();
  if (failed(materializeBlockPointerForms(kernel)))
    return failure();
  orientPointerLoads(kernel);
  if (failed(materializeLegalConfigs(kernel, *tensorDescriptorForms, localOptions)))
    return failure();
  selectContractForms(kernel);
  gpu::foldExactConstantDivisions(kernel);
  canonicalizeBroadcastProjections(kernel);
  SmallVector<gpu::AssumeInBoundsOp> boundsAssumptions;
  kernel.walk([&](gpu::AssumeInBoundsOp assumption) {
    boundsAssumptions.push_back(assumption);
  });
  for (gpu::AssumeInBoundsOp assumption : boundsAssumptions)
    assumption.erase();
  if (failed(gpu::eliminateCommonValues(module)) ||
      failed(legalizeCollectiveCallbacks(kernel)) ||
      failed(materializeDeferredResourceBounds(kernel)))
    return failure();
  sinkSelectProducers(kernel);
  if (failed(verifyTritonProgram(module)))
    return failure();
  kernel->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::triton
