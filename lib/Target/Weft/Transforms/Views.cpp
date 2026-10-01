#include "Views.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace intent::weft_provider {
namespace {

std::optional<int64_t> constant(OpFoldResult value) {
  if (auto attribute = dyn_cast<Attribute>(value))
    if (auto integer = dyn_cast<IntegerAttr>(attribute)) return integer.getInt();
  if (auto operand = dyn_cast<Value>(value)) {
    llvm::APInt integer;
    if (matchPattern(operand, m_ConstantInt(&integer))) return integer.getSExtValue();
  }
  return std::nullopt;
}

bool same(OpFoldResult actual, Value metadata, int64_t fixed) {
  if (auto value = dyn_cast<Value>(actual); value && value == metadata) return true;
  auto integer = constant(actual);
  return integer && !ShapedType::isDynamic(fixed) && *integer == fixed;
}

// An extent is either a known integer, one scalar SSA value, or one dimension
// of one current descriptor. A dynamic type marker is never an identity.
struct SizeIdentity {
  std::optional<int64_t> fixed;
  Value value;
  std::optional<unsigned> axis;

  bool operator==(const SizeIdentity &other) const {
    return fixed == other.fixed && value == other.value && axis == other.axis;
  }
};

SizeIdentity sizeIdentity(OpFoldResult size);

SizeIdentity dimensionIdentity(Value memory, unsigned axis) {
  auto type = cast<MemRefType>(memory.getType());
  if (!type.isDynamicDim(axis)) return {type.getDimSize(axis), {}, {}};
  if (auto cast = memory.getDefiningOp<memref::CastOp>())
    return dimensionIdentity(cast.getSource(), axis);
  if (auto subview = memory.getDefiningOp<memref::SubViewOp>()) {
    unsigned retained = 0;
    for (unsigned original = 0; original < subview.getSourceType().getRank(); ++original)
      if (!subview.getDroppedDims().test(original) && retained++ == axis)
        return sizeIdentity(subview.getMixedSizes()[original]);
  }
  if (auto allocation = memory.getDefiningOp<memref::AllocOp>())
    return sizeIdentity(allocation.getDynamicSizes()[type.getDynamicDimIndex(axis)]);
  if (auto allocation = memory.getDefiningOp<memref::AllocaOp>())
    return sizeIdentity(allocation.getDynamicSizes()[type.getDynamicDimIndex(axis)]);
  if (auto view = memory.getDefiningOp<memref::ReinterpretCastOp>())
    return sizeIdentity(view.getMixedSizes()[axis]);
  if (auto expand = memory.getDefiningOp<memref::ExpandShapeOp>()) {
    for (auto [source, group] : llvm::enumerate(expand.getReassociationIndices())) {
      if (!llvm::is_contained(group, axis)) continue;
      if (llvm::count_if(group, [&](int64_t result) { return type.getDimSize(result) != 1; }) == 1)
        return dimensionIdentity(expand.getSrc(), source);
      break;
    }
  }
  if (auto collapse = memory.getDefiningOp<memref::CollapseShapeOp>()) {
    auto groups = collapse.getReassociationIndices();
    std::optional<unsigned> active;
    bool unitProjection = true;
    for (int64_t source : groups[axis]) {
      if (collapse.getSrcType().getDimSize(source) == 1) continue;
      if (active) { unitProjection = false; break; }
      active = source;
    }
    if (unitProjection)
      return active ? dimensionIdentity(collapse.getSrc(), *active) : SizeIdentity{1, {}, {}};
  }
  if (auto argument = dyn_cast<BlockArgument>(memory))
    if (auto task = dyn_cast<cpu::TasksOp>(argument.getOwner()->getParentOp());
        task && argument.getArgNumber())
      return dimensionIdentity(task.getCaptures()[argument.getArgNumber() - 1], axis);
  return {{}, memory, axis};
}

SizeIdentity sizeIdentity(OpFoldResult size) {
  if (auto fixed = constant(size)) return {fixed, {}, {}};
  Value value = cast<Value>(size);
  if (auto dimension = value.getDefiningOp<memref::DimOp>())
    if (auto axis = dimension.getConstantIndex())
      return dimensionIdentity(dimension.getSource(), *axis);
  if (auto metadata = value.getDefiningOp<memref::ExtractStridedMetadataOp>())
    for (auto [axis, extent] : llvm::enumerate(metadata.getSizes()))
      if (value == extent) return dimensionIdentity(metadata.getSource(), axis);
  if (auto argument = dyn_cast<BlockArgument>(value))
    if (auto task = dyn_cast<cpu::TasksOp>(argument.getOwner()->getParentOp());
        task && argument.getArgNumber())
      return sizeIdentity(task.getCaptures()[argument.getArgNumber() - 1]);
  return {{}, value, {}};
}

bool sameSize(OpFoldResult actual, memref::ExtractStridedMetadataOp metadata, unsigned axis) {
  return sizeIdentity(actual) == dimensionIdentity(metadata.getSource(), axis);
}

bool isViewDefinition(Operation *operation) {
  return operation && (isAxisView(operation) ||
      isa<memref::SubViewOp, memref::CastOp, memref::ExtractStridedMetadataOp>(operation));
}

class CaptureReifier {
public:
  CaptureReifier(cpu::TasksOp task) : task(task), builder(&task.getBody().front(), task.getBody().front().begin()) {}

  LogicalResult run() {
    SmallVector<Value> captures(task.getCaptures());
    auto &body = task.getBody().front();
    for (auto [number, capture] : llvm::enumerate(captures)) {
      auto argument = body.getArgument(number + 1);
      if (argument.use_empty() || !isa<MemRefType>(capture.getType()) ||
          !isViewDefinition(capture.getDefiningOp())) continue;
      auto replacement = materialize(capture);
      if (failed(replacement)) return failure();
      argument.replaceAllUsesWith(*replacement);
    }
    return success();
  }

private:
  Value capture(Value value) {
    for (auto [number, existing] : llvm::enumerate(task.getCaptures()))
      if (existing == value) {
        Value argument = task.getBody().front().getArgument(number + 1);
        mapping.map(value, argument);
        return argument;
      }
    task.getCapturesMutable().append(value);
    auto argument = task.getBody().front().addArgument(value.getType(), value.getLoc());
    mapping.map(value, argument);
    return argument;
  }

  FailureOr<Value> materialize(Value value) {
    if (auto found = mapping.lookupOrNull(value)) return found;
    Operation *definition = value.getDefiningOp();
    // These values belong to the host descriptor being reified. Capturing an
    // already computed size/stride keeps metadata-only storage out of the task
    // ABI and does not replay its scalar dependencies inside each task.
    if (!isa<MemRefType>(value.getType()) &&
        !isa_and_nonnull<arith::ConstantOp>(definition)) return capture(value);
    bool clone = isViewDefinition(definition) ||
        isa_and_nonnull<arith::ConstantOp>(definition);
    if (!clone) {
      if (isa<MemRefType>(value.getType())) {
        auto argument = dyn_cast<BlockArgument>(value);
        bool entry = argument && isa<func::FuncOp>(argument.getOwner()->getParentOp());
        if (!entry && !isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(definition))
          return emitError(value.getLoc(), "Weft task view must trace to explicit entry storage or an allocation; loop-carried and opaque descriptors are unsupported"), failure();
      }
      return capture(value);
    }
    for (Value operand : definition->getOperands())
      if (failed(materialize(operand))) return failure();
    builder.clone(*definition, mapping);
    return mapping.lookup(value);
  }

  cpu::TasksOp task;
  OpBuilder builder;
  IRMapping mapping;
};

} // namespace

bool isAxisView(Operation *operation) {
  return isa_and_nonnull<memref::ReinterpretCastOp, memref::ExpandShapeOp, memref::CollapseShapeOp>(operation);
}

FailureOr<AxisView> queryAxisView(Value value) {
  auto reject = [&]() -> FailureOr<AxisView> {
    return emitError(value.getLoc(), "Weft view requires an offset-preserving axis permutation or unit-axis insertion/removal; general strided reinterpretation and axis flattening are unsupported"), failure();
  };
  auto resultType = cast<MemRefType>(value.getType());
  if (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto metadata = view.getSource().getDefiningOp<memref::ExtractStridedMetadataOp>();
    if (!metadata || view.getSource() != metadata.getBaseBuffer()) return reject();
    auto sourceType = cast<MemRefType>(metadata.getSource().getType());
    SmallVector<int64_t> sourceStrides;
    int64_t offset;
    if (sourceType.getElementType() != resultType.getElementType() ||
        sourceType.getMemorySpace() != resultType.getMemorySpace() ||
        failed(sourceType.getStridesAndOffset(sourceStrides, offset)) ||
        !same(view.getMixedOffsets().front(), metadata.getOffset(), offset)) return reject();
    AxisView result{metadata.getSource(), {}};
    SmallVector<bool> used(sourceType.getRank(), false);
    auto sizes = view.getMixedSizes(), strides = view.getMixedStrides();
    for (unsigned axis = 0; axis < static_cast<unsigned>(resultType.getRank()); ++axis) {
      if (constant(sizes[axis]) == 1) { result.sourceAxes.push_back(std::nullopt); continue; }
      std::optional<unsigned> match;
      for (unsigned source = 0; source < static_cast<unsigned>(sourceType.getRank()); ++source) {
        if (used[source] || sourceType.getDimSize(source) == 1 ||
            !sameSize(sizes[axis], metadata, source) ||
            !same(strides[axis], metadata.getStrides()[source], sourceStrides[source])) continue;
        if (match) return reject();
        match = source;
      }
      if (!match) return reject();
      used[*match] = true;
      result.sourceAxes.push_back(match);
    }
    for (unsigned source = 0; source < used.size(); ++source)
      if (!used[source] && sourceType.getDimSize(source) != 1) return reject();
    return result;
  }
  if (auto expand = value.getDefiningOp<memref::ExpandShapeOp>()) {
    auto sourceType = expand.getSrcType();
    AxisView result{expand.getSrc(), SmallVector<std::optional<unsigned>>(resultType.getRank())};
    for (auto [source, group] : llvm::enumerate(expand.getReassociationIndices())) {
      std::optional<unsigned> active;
      for (int64_t axis : group) if (resultType.getDimSize(axis) != 1) {
        if (active) return reject();
        active = axis;
      }
      if (active) result.sourceAxes[*active] = source;
      else if (sourceType.getDimSize(source) != 1) return reject();
    }
    return result;
  }
  if (auto collapse = value.getDefiningOp<memref::CollapseShapeOp>()) {
    auto sourceType = collapse.getSrcType();
    AxisView result{collapse.getSrc(), {}};
    for (auto [axis, group] : llvm::enumerate(collapse.getReassociationIndices())) {
      std::optional<unsigned> active;
      for (int64_t source : group) if (sourceType.getDimSize(source) != 1) {
        if (active) return reject();
        active = source;
      }
      if (!active && resultType.getDimSize(axis) != 1) return reject();
      result.sourceAxes.push_back(active);
    }
    return result;
  }
  return reject();
}

LogicalResult reifyTaskViewCaptures(func::FuncOp function) {
  SmallVector<cpu::TasksOp> tasks;
  function.walk([&](cpu::TasksOp task) { tasks.push_back(task); });
  for (auto task : tasks) if (failed(CaptureReifier(task).run())) return failure();
  return success();
}

} // namespace intent::weft_provider
