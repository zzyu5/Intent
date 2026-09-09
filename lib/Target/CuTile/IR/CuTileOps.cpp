#include "Intent/Target/CuTile/IR/CuTileOps.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
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
  return left && right && left.getDimensionId() > 0 &&
         left.getDimensionId() == right.getDimensionId();
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

LogicalResult verifyResourceOrderedTile(Operation *owner, unsigned rank,
                                        gpu::FragmentType tile,
                                        ValueRange tileIndices) {
  if (tileIndices.size() != rank || tile.getShape().size() != rank)
    return owner->emitOpError(
        "requires one resource-ordered tile axis and index per view axis");
  for (unsigned axis = 0; axis < rank; ++axis) {
    auto mapping = dyn_cast<gpu::AxisMapAttr>(tile.getAxisMaps()[axis]);
    if (!mapping || mapping.getFragmentAxis() != axis)
      return owner->emitOpError(
          "tile relation is not indexed in resource-axis order");
    Type indexType = tileIndices[axis].getType();
    if (!indexType.isIndex() && !indexType.isSignlessInteger(32))
      return owner->emitOpError("tile-space indices must have index or i32 type");
  }
  return success();
}

LogicalResult verifySerializedCoordinateOrder(
    Operation *owner, ArrayRef<int64_t> sourceAxes) {
  for (auto [position, sourceAxis] : llvm::enumerate(sourceAxes))
    if (position != static_cast<size_t>(sourceAxis))
      return owner->emitOpError(
          "serialized coordinates must be in resource-axis order");
  return success();
}

LogicalResult verifyLoadLatency(Operation *operation, Value latency) {
  if (!latency)
    return success();
  auto parameter = latency.getDefiningOp<gpu::ParameterOp>();
  if (!parameter || parameter.getParameter().getRole() !=
                        static_cast<uint32_t>(
                            gpu::ParameterRole::ProviderLoadPolicy) ||
      !llvm::all_of(parameter.getParameter().getCandidates().asArrayRef(),
                    isLegalLoadPolicy))
    return operation->emitOpError(
        "load policy requires an inferred or explicit latency domain");
  return success();
}

LogicalResult verifyFullTileCondition(TileLoadOp load) {
  if (!load.getFullTiles())
    return success();
  auto view = load.getResource().getType();
  auto shape = load.getResult().getType().getShape();
  llvm::SmallBitVector covered(view.getRank());
  for (auto [axis, attribute] : llvm::enumerate(shape)) {
    auto width = cast<gpu::PhysicalExprAttr>(attribute);
    auto size = cast<gpu::PhysicalExprAttr>(view.getLayout().getExtents()[axis]);
    auto constant = static_cast<uint32_t>(gpu::PhysicalExprKind::Constant);
    covered[axis] = width.getKind() == constant && width.getValue() > 0 &&
                    (width.getValue() == 1 ||
                     (size.getKind() == constant &&
                      size.getValue() % width.getValue() == 0));
  }
  SmallVector<Value> pending{load.getFullTiles()};
  while (!pending.empty()) {
    Value predicate = pending.pop_back_val();
    if (matchPattern(predicate, m_Zero()))
      return success();
    if (matchPattern(predicate, m_One()))
      continue;
    if (auto conjunction = predicate.getDefiningOp<gpu::BinaryOp>();
        conjunction &&
        conjunction.getOperatorKind() == BinaryOperator::LogicalAnd) {
      pending.push_back(conjunction.getLhs());
      pending.push_back(conjunction.getRhs());
      continue;
    }
    auto equal = predicate.getDefiningOp<gpu::CompareOp>();
    if (!equal || equal.getPredicate() != ComparePredicate::Eq ||
        !matchPattern(equal.getRhs(), m_Zero()))
      return load.emitOpError("full tiles require a view-size divisibility proof");
    auto remainder = equal.getLhs().getDefiningOp<gpu::BinaryOp>();
    auto size = remainder ? remainder.getLhs().getDefiningOp<gpu::DimOp>()
                          : gpu::DimOp();
    auto width = remainder ? remainder.getRhs().getDefiningOp<gpu::PhysicalExprOp>()
                           : gpu::PhysicalExprOp();
    if (!remainder || remainder.getOperatorKind() != BinaryOperator::Remainder ||
        !size || !width || size.getView() != load.getResource() ||
        size.getAxis() >= view.getRank() ||
        width.getExpression() != shape[size.getAxis()])
      return load.emitOpError("full-tile proof must describe this native load");
    covered[size.getAxis()] = true;
  }
  return covered.all()
             ? success()
             : load.emitOpError("full-tile proof does not cover every view axis");
}

} // namespace

LogicalResult ArrayViewOp::verify() {
  auto base = dyn_cast<BlockArgument>(getBase());
  auto kernel = (*this)->getParentOfType<func::FuncOp>();
  auto view = getBase().getType();
  if (!base || !kernel || base.getOwner() != &kernel.getBody().front() ||
      (*this)->getBlock() != base.getOwner() || view.getAccess() != 0 ||
      getResult().getType() != view)
    return emitOpError(
        "requires a read-only kernel view and preserves its logical ABI");
  auto ends = getGroupEnds();
  if (ends.size() < 2 || ends.size() >= view.getRank() ||
      ends.back() != view.getRank())
    return emitOpError("requires an ordered partition that reduces the array rank");
  unsigned begin = 0;
  for (int64_t end : ends) {
    if (end <= begin || end > view.getRank())
      return emitOpError("array collapse groups must partition every source axis");
    for (unsigned axis = begin + 1; axis < static_cast<unsigned>(end); ++axis) {
      auto extent =
          dyn_cast<gpu::PhysicalExprAttr>(view.getLayout().getExtents()[axis]);
      if (!extent ||
          extent.getKind() !=
              static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) ||
          extent.getValue() <= 1)
        return emitOpError(
            "collapsed inner axes require positive non-unit static extents");
    }
    begin = end;
  }
  for (Operation *user : getResult().getUsers())
    if (!isa<TileLoadOp>(user))
      return emitOpError(
          "conditional array aliases are only consumed by native tile loads");
  return success();
}

FailureOr<TileLoadOp> unfoldedArrayLoad(TileLoadOp load) {
  auto array = load.getResource().getDefiningOp<ArrayViewOp>();
  auto choice = load->getParentOfType<scf::IfOp>();
  if (!array || !choice || choice.getCondition() != array.getEligible() ||
      load->getBlock() != choice.thenBlock() ||
      choice.thenBlock()->getOperations().size() != 3 ||
      choice.getElseRegion().empty() ||
      choice.elseBlock()->getOperations().size() != 2)
    return failure();
  auto original = dyn_cast<TileLoadOp>(&choice.elseBlock()->front());
  auto restore = dyn_cast<gpu::ReshapeOp>(load->getNextNode());
  auto thenYield = dyn_cast<scf::YieldOp>(choice.thenBlock()->getTerminator());
  auto elseYield = dyn_cast<scf::YieldOp>(choice.elseBlock()->getTerminator());
  if (!original || !restore || !thenYield || !elseYield ||
      thenYield.getNumOperands() != 1 || elseYield.getNumOperands() != 1 ||
      original.getResource() != array.getBase() ||
      original.getAllowTma() != load.getAllowTma() ||
      original.getLatencyPolicy() != load.getLatencyPolicy() ||
      original.getFullTiles() != load.getFullTiles() ||
      restore.getValue() != load.getResult() ||
      restore.getResult().getType() != original.getResult().getType() ||
      thenYield.getOperand(0) != restore.getResult() ||
      elseYield.getOperand(0) != original.getResult())
    return failure();
  return original;
}

LogicalResult TileLoadOp::verify() {
  auto view = getResource().getType();
  auto result = getResult().getType();
  if (!getAllowTma().getType().isInteger(1))
    return emitOpError("allow_tma must be a compile-time i1 access decision");
  if (failed(verifyLoadLatency(*this, getLatencyPolicy())))
    return failure();
  auto array = getResource().getDefiningOp<ArrayViewOp>();
  unsigned rank = array ? array.getGroupEnds().size() : view.getRank();
  if (failed(verifyResourceOrderedTile(*this, rank, result,
                                       getTileIndices())))
    return failure();
  if (view.getElementType() != result.getElementType())
    return emitOpError("view and tile element types disagree");
  if (!array)
    return verifyFullTileCondition(*this);
  if (failed(array.verify()))
    return failure();
  auto original = unfoldedArrayLoad(*this);
  if (failed(original))
    return emitOpError(
        "collapsed load must have a guarded reshape and unchanged alternative");
  auto source = original->getResult().getType();
  if (source.getShape().size() != view.getRank() ||
      original->getTileIndices().size() != view.getRank() ||
      source.getOwner() != result.getOwner() ||
      source.getValidity() != result.getValidity())
    return emitOpError("collapsed load must preserve the original physical domain");
  unsigned begin = 0;
  for (auto [group, end] : llvm::enumerate(array.getGroupEnds())) {
    Attribute extent = source.getShape()[begin];
    if (getTileIndices()[group] != original->getTileIndices()[begin])
      return emitOpError(
          "collapsed tile origin must be the original outer tile origin");
    for (unsigned axis = begin + 1; axis < static_cast<unsigned>(end); ++axis) {
      if (source.getShape()[axis] != view.getLayout().getExtents()[axis] ||
          !matchPattern(original->getTileIndices()[axis], m_Zero()))
        return emitOpError(
            "collapsed inner tile axes must cover the full source axis from zero");
      extent = gpu::PhysicalExprAttr::get(
          getContext(), static_cast<uint32_t>(gpu::PhysicalExprKind::Multiply), 0,
          StringAttr::get(getContext()),
          ArrayAttr::get(getContext(), {extent, source.getShape()[axis]}));
    }
    if (result.getShape()[group] != extent)
      return emitOpError(
          "collapsed tile extent must equal the original contiguous product");
    begin = end;
  }
  return success();
}

void TileLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

LogicalResult TileStoreOp::verify() {
  auto view = getResource().getType();
  if (!getAllowTma().getType().isInteger(1))
    return emitOpError("allow_tma must be a compile-time i1 access decision");
  if (getResource().getDefiningOp<ArrayViewOp>())
    return emitOpError("conditional array aliases are read-only");
  if (failed(verifyResourceOrderedTile(*this, view.getRank(), getValue().getType(),
                                       getTileIndices())))
    return failure();
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
  if (failed(verifyLoadLatency(*this, getLatencyPolicy())))
    return failure();
  auto view = getResource().getType();
  if (getCoordinates().size() != view.getRank() ||
      view.getElementType() != getResult().getType().getElementType())
    return emitOpError("requires one advanced coordinate per source axis");
  if (failed(verifySourceAxes(*this, getSourceAxes(), view.getRank())))
    return failure();
  if (failed(verifySerializedCoordinateOrder(*this, getSourceAxes())))
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
  if (failed(verifySerializedCoordinateOrder(*this, getSourceAxes())))
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

LogicalResult ExtractOp::verify() {
  auto source = getSource().getType();
  unsigned rank = source.getShape().size();
  if (getCoordinates().size() != rank || getExtractionShape().size() != rank ||
      elementType(getResult().getType()) != source.getElementType())
    return emitOpError(
        "requires source-ranked tile coordinates/shape and matching element type");
  llvm::SmallBitVector retained(rank);
  int64_t previous = -1;
  for (int64_t axis : getRetainedAxes()) {
    if (axis <= previous || axis >= static_cast<int64_t>(rank))
      return emitOpError("retained axes must be an ordered subset of source axes");
    retained.set(axis);
    previous = axis;
  }
  for (unsigned axis = 0; axis < rank; ++axis) {
    Value coordinate = getCoordinates()[axis];
    if (!coordinate.getType().isInteger(32))
      return emitOpError("requires i32 scalar tile coordinates");
    auto extent = dyn_cast<gpu::PhysicalExprAttr>(getExtractionShape()[axis]);
    if (!extent)
      return emitOpError("extraction shape must contain physical expressions");
    if (retained.test(axis)) {
      if (extent != source.getShape()[axis] || !matchPattern(coordinate, m_Zero()))
        return emitOpError("retained axes require full source extents and zero tile indices");
    } else if (extent.getKind() !=
                   static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) ||
               extent.getValue() != 1) {
      return emitOpError("selected axes require unit extraction extents");
    }
  }
  auto result = dyn_cast<gpu::FragmentType>(getResult().getType());
  if (!result)
    return getRetainedAxes().empty()
               ? success()
               : emitOpError("scalar extraction cannot retain source axes");
  if (result.getShape().size() != getRetainedAxes().size() ||
      result.getOwner() != source.getOwner())
    return emitOpError("extracted fragment must preserve retained rank and owner");
  for (auto [resultAxis, sourceAxis] : llvm::enumerate(getRetainedAxes()))
    if (result.getShape()[resultAxis] != source.getShape()[sourceAxis] ||
        !sameLogicalAxis(source, sourceAxis, result, resultAxis))
      return emitOpError("extracted fragment must preserve retained axis relations");
  return success();
}

LogicalResult MMAOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto accumulator = getAccumulator().getType();
  auto result = getResult().getType();
  const unsigned rank = lhs.getShape().size();
  bool valid = rank >= 2 && rank <= 3 && rhs.getShape().size() == rank &&
               accumulator.getShape().size() == rank &&
               result == accumulator && lhs.getOwner() == rhs.getOwner() &&
               lhs.getOwner() == accumulator.getOwner();
  for (unsigned axis = 0; valid && axis + 2 < rank; ++axis)
    valid = lhs.getShape()[axis] == rhs.getShape()[axis] &&
            lhs.getShape()[axis] == result.getShape()[axis] &&
            sameLogicalAxis(lhs, axis, rhs, axis) &&
            sameLogicalAxis(lhs, axis, result, axis);
  if (valid) {
    const unsigned matrixAxis = rank - 2;
    valid = lhs.getShape()[matrixAxis] == result.getShape()[matrixAxis] &&
            lhs.getShape()[matrixAxis + 1] ==
                rhs.getShape()[matrixAxis] &&
            rhs.getShape()[matrixAxis + 1] ==
                result.getShape()[matrixAxis + 1] &&
            sameLogicalAxis(lhs, matrixAxis, result, matrixAxis) &&
            sameLogicalAxis(lhs, matrixAxis + 1, rhs, matrixAxis) &&
            sameLogicalAxis(rhs, matrixAxis + 1, result, matrixAxis + 1);
  }
  if (!valid) {
    InFlightDiagnostic diagnostic = emitOpError(
        "requires a canonical [M,K] x [K,N] or [B,M,K] x [B,K,N] "
        "cuTile MMA form");
    diagnostic << "; lhs=" << lhs << "; rhs=" << rhs
               << "; accumulator=" << accumulator << "; result=" << result;
    return failure();
  }
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
