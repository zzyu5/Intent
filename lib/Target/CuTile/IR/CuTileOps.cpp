#include "Intent/Target/CuTile/IR/CuTileOps.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

namespace intent::cutile {

namespace {

Type elementType(Type type) {
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return fragment.getElementType();
  return type;
}

bool isIntegerCoordinate(Type type) {
  Type element = elementType(type);
  return element.isIntOrIndex();
}

bool samePhysicalDomain(gpu::FragmentType lhs, gpu::FragmentType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getAxisMaps() == rhs.getAxisMaps() &&
         lhs.getValidity() == rhs.getValidity() &&
         lhs.getOwner() == rhs.getOwner();
}

bool sameLogicalAxis(gpu::FragmentType lhs, unsigned lhsAxis,
                     gpu::FragmentType rhs, unsigned rhsAxis) {
  auto left = dyn_cast<gpu::AxisMapAttr>(lhs.getAxisMaps()[lhsAxis]);
  auto right = dyn_cast<gpu::AxisMapAttr>(rhs.getAxisMaps()[rhsAxis]);
  return left && right && left.getSourceId() == right.getSourceId() &&
         left.getSourceAxis() == right.getSourceAxis();
}

LogicalResult verifyCoordinateDomains(Operation *owner, ValueRange coordinates,
                                      Type targetType) {
  auto target = dyn_cast<gpu::FragmentType>(targetType);
  for (Value coordinate : coordinates) {
    if (!isIntegerCoordinate(coordinate.getType()))
      return owner->emitOpError("coordinates must have integer element type");
    auto source = dyn_cast<gpu::FragmentType>(coordinate.getType());
    if (!source)
      continue;
    if (!target || !samePhysicalDomain(source, target))
      return owner->emitOpError(
          "coordinate tile was not materialized on the selected physical domain");
  }
  return success();
}

LogicalResult verifySourceAxes(Operation *owner, ArrayRef<int64_t> sourceAxes,
                               unsigned rank) {
  if (sourceAxes.size() != rank)
    return owner->emitOpError("requires one source-axis mapping per view axis");
  llvm::SmallBitVector seen(rank);
  for (int64_t axis : sourceAxes) {
    if (axis < 0 || axis >= static_cast<int64_t>(rank) || seen.test(axis))
      return owner->emitOpError(
          "source-axis mapping must be a permutation of the view axes");
    seen.set(axis);
  }
  return success();
}

} // namespace

LogicalResult TileLoadOp::verify() {
  auto view = getResource().getType();
  auto result = getResult().getType();
  if (getTileIndices().size() != view.getRank())
    return emitOpError("requires one tile-space index per source axis");
  if (view.getElementType() != result.getElementType())
    return emitOpError("view and tile element types disagree");
  return success();
}

void TileLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

LogicalResult TileStoreOp::verify() {
  auto view = getResource().getType();
  if (getTileIndices().size() != view.getRank())
    return emitOpError("requires one tile-space index per destination axis");
  return view.getElementType() == getValue().getType().getElementType()
             ? success()
             : emitOpError("view and tile element types disagree");
}

void TileStoreOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult ScalarLoadOp::verify() {
  auto view = getResource().getType();
  if (getIndices().size() != view.getRank() ||
      getResult().getType() != view.getElementType())
    return emitOpError("requires one scalar index per source axis");
  if (static_cast<bool>(getValid()) != static_cast<bool>(getFill()))
    return emitOpError("requires validity and padding together");
  if (getValid() &&
      (!getValid().getType().isInteger(1) ||
       getFill().getType() != view.getElementType()))
    return emitOpError("validity/padding do not match the scalar load");
  return success();
}

void ScalarLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

LogicalResult ScalarStoreOp::verify() {
  auto view = getResource().getType();
  if (getIndices().size() != view.getRank() ||
      getValue().getType() != view.getElementType())
    return emitOpError("requires one scalar index per destination axis");
  if (getValid() && !getValid().getType().isInteger(1))
    return emitOpError("scalar store validity must be i1");
  return success();
}

void ScalarStoreOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult GatherLoadOp::verify() {
  auto view = getResource().getType();
  if (getCoordinates().size() != view.getRank() ||
      view.getElementType() != getResult().getType().getElementType())
    return emitOpError("requires one advanced coordinate per source axis");
  if (failed(verifySourceAxes(*this, getSourceAxes(), view.getRank())))
    return failure();
  if (failed(verifyCoordinateDomains(*this, getCoordinates(), getResult().getType())))
    return failure();
  if (static_cast<bool>(getValid()) != static_cast<bool>(getFill()))
    return emitOpError("requires validity and padding together");
  if (getValid()) {
    if (!elementType(getValid().getType()).isInteger(1))
      return emitOpError("mask must be a boolean fragment");
    auto valid = dyn_cast<gpu::FragmentType>(getValid().getType());
    if (valid && !samePhysicalDomain(valid, getResult().getType()))
      return emitOpError("mask does not match the gathered physical domain");
  }
  if (getFill() && getFill().getType() != view.getElementType())
    return emitOpError("padding scalar must match the view element type");
  return success();
}

void GatherLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

LogicalResult ScatterStoreOp::verify() {
  auto view = getResource().getType();
  if (getCoordinates().size() != view.getRank() ||
      view.getElementType() != getValue().getType().getElementType())
    return emitOpError("requires one advanced coordinate per destination axis");
  if (failed(verifySourceAxes(*this, getSourceAxes(), view.getRank())))
    return failure();
  if (failed(verifyCoordinateDomains(*this, getCoordinates(), getValue().getType())))
    return failure();
  if (getValid()) {
    if (!elementType(getValid().getType()).isInteger(1))
      return emitOpError("mask must be a boolean fragment");
    auto valid = dyn_cast<gpu::FragmentType>(getValid().getType());
    if (valid && !samePhysicalDomain(valid, getValue().getType()))
      return emitOpError("mask does not match the scattered physical domain");
  }
  return success();
}

void ScatterStoreOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult AtomicRMWOp::verify() {
  auto view = getResource().getType();
  if (getCoordinates().size() != view.getRank() ||
      getValue().getType() != getResult().getType() ||
      elementType(getValue().getType()) != view.getElementType())
    return emitOpError(
        "requires a view-ranked cuTile atomic address and matching value schema");
  if (failed(verifyCoordinateDomains(*this, getCoordinates(), getValue().getType())))
    return failure();
  return success();
}

void AtomicRMWOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult ExtractScalarOp::verify() {
  auto source = getSource().getType();
  if (getCoordinates().size() != source.getShape().size() ||
      getResult().getType() != source.getElementType())
    return emitOpError(
        "requires one scalar element coordinate per source tile axis");
  for (Value coordinate : getCoordinates())
    if (!coordinate.getType().isIntOrIndex())
      return emitOpError("requires integer scalar coordinates");
  if (static_cast<bool>(getValid()) != static_cast<bool>(getFill()) ||
      (getValid() && (!getValid().getType().isInteger(1) ||
                      getFill().getType() != getResult().getType())))
    return emitOpError("validity/fill do not match the extracted scalar");
  return success();
}

LogicalResult MMAOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto accumulator = getAccumulator().getType();
  auto result = getResult().getType();
  if (lhs.getShape().size() != 2 || rhs.getShape().size() != 2 ||
      accumulator.getShape().size() != 2 || result != accumulator ||
      lhs.getOwner() != rhs.getOwner() ||
      lhs.getOwner() != accumulator.getOwner() ||
      lhs.getShape()[0] != result.getShape()[0] ||
      lhs.getShape()[1] != rhs.getShape()[0] ||
      rhs.getShape()[1] != result.getShape()[1] ||
      !sameLogicalAxis(lhs, 0, result, 0) ||
      !sameLogicalAxis(lhs, 1, rhs, 0) ||
      !sameLogicalAxis(rhs, 1, result, 1))
    return emitOpError("requires a canonical [M,K] x [K,N] cuTile MMA form");
  return success();
}

LogicalResult ScaledMMAOp::verify() {
  auto lhs = getLhs().getType();
  auto lhsScale = getLhsScale().getType();
  auto rhs = getRhs().getType();
  auto rhsScale = getRhsScale().getType();
  auto accumulator = getAccumulator().getType();
  if (lhs.getShape().size() != 3 || lhsScale.getShape().size() != 2 ||
      rhs.getShape().size() != 3 || rhsScale.getShape().size() != 2 ||
      accumulator.getShape().size() != 2 || getResult().getType() != accumulator ||
      lhs.getOwner() != lhsScale.getOwner() ||
      lhs.getOwner() != rhs.getOwner() ||
      lhs.getOwner() != rhsScale.getOwner() ||
      lhs.getOwner() != accumulator.getOwner() ||
      lhs.getShape()[0] != accumulator.getShape()[0] ||
      lhs.getShape()[0] != lhsScale.getShape()[0] ||
      lhs.getShape()[1] != lhsScale.getShape()[1] ||
      lhs.getShape()[1] != rhs.getShape()[0] ||
      lhs.getShape()[1] != rhsScale.getShape()[0] ||
      lhs.getShape()[2] != rhs.getShape()[1] ||
      rhs.getShape()[2] != accumulator.getShape()[1] ||
      rhs.getShape()[2] != rhsScale.getShape()[1] ||
      !sameLogicalAxis(lhs, 0, accumulator, 0) ||
      !sameLogicalAxis(lhs, 0, lhsScale, 0) ||
      !sameLogicalAxis(lhs, 1, lhsScale, 1) ||
      !sameLogicalAxis(lhs, 1, rhs, 0) ||
      !sameLogicalAxis(lhs, 1, rhsScale, 0) ||
      !sameLogicalAxis(lhs, 2, rhs, 1) ||
      !sameLogicalAxis(rhs, 2, accumulator, 1) ||
      !sameLogicalAxis(rhs, 2, rhsScale, 1))
    return emitOpError("requires [M,G,S]/[M,G] x [G,S,N]/[G,N] scaled MMA operands");
  auto inner = dyn_cast<gpu::PhysicalExprAttr>(lhs.getShape()[2]);
  if (!inner ||
      inner.getKind() !=
          static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) ||
      inner.getValue() != static_cast<int64_t>(getLhsGroupSize()) ||
      getLhsGroupSize() != getRhsGroupSize() ||
      getLhsFormat() != ScaledFormat::E4M3 ||
      getRhsFormat() != ScaledFormat::E4M3 ||
      !lhsScale.getElementType().isUnsignedInteger(8) ||
      !rhsScale.getElementType().isUnsignedInteger(8) ||
      !accumulator.getElementType().isF32())
    return emitOpError("requires constant group extent, E8M0 scales and f32 accumulator");
  return success();
}

LogicalResult ReduceOp::verify() {
  if (static_cast<size_t>(getAxis()) >=
      getSource().getType().getShape().size())
    return emitOpError("has an invalid cuTile native reduction axis/kind");
  switch (getKind()) {
  case BinaryOperator::Add:
  case BinaryOperator::MaximumNum:
  case BinaryOperator::MinimumNum:
    return success();
  case BinaryOperator::LogicalOr:
  case BinaryOperator::LogicalAnd:
    return elementType(getResult().getType()).isInteger(1)
               ? success()
               : emitOpError("logical reduction kind requires an i1 result");
  default:
    return emitOpError("has no semantics-preserving cuTile native reduction");
  }
}

LogicalResult ScanOp::verify() {
  if (static_cast<size_t>(getAxis()) >=
          getSource().getType().getShape().size() ||
      getKind() != BinaryOperator::Add)
    return emitOpError("cuTile native scan currently requires additive cumsum");
  return getResult().getType() == getSource().getType()
             ? success()
             : emitOpError("scan must preserve the physical tile schema");
}

} // namespace intent::cutile

#define GET_OP_CLASSES
#include "Intent/Target/CuTile/IR/CuTileOps.cpp.inc"
