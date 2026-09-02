#include "Intent/Target/TileLang/IR/TileLangOps.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

namespace intent::tilelang {

namespace {

LogicalResult verifyCopyAxes(Operation *owner, ArrayRef<int64_t> axes,
                             unsigned viewRank, unsigned bufferRank) {
  if (axes.size() != bufferRank)
    return owner->emitOpError(
        "requires one external-view axis per physical buffer axis");
  llvm::SmallBitVector seen(viewRank);
  for (int64_t axis : axes) {
    if (axis < 0 || axis >= static_cast<int64_t>(viewRank) || seen.test(axis))
      return owner->emitOpError(
          "copy axes must be distinct axes of the external view");
    seen.set(axis);
  }
  return success();
}

bool isConstantExtent(Attribute attribute, int64_t expected) {
  auto expression = dyn_cast<gpu::PhysicalExprAttr>(attribute);
  return expression &&
         expression.getKind() ==
             static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) &&
         expression.getValue() == expected;
}

bool hasNoStaticQuotientConflict(Attribute quotientAttribute,
                                 Attribute dividendAttribute,
                                 int64_t divisor) {
  auto quotient = dyn_cast<gpu::PhysicalExprAttr>(quotientAttribute);
  auto dividend = dyn_cast<gpu::PhysicalExprAttr>(dividendAttribute);
  if (!quotient || !dividend || divisor <= 0)
    return false;
  if (quotient.getKind() ==
          static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) &&
      dividend.getKind() ==
          static_cast<uint32_t>(gpu::PhysicalExprKind::Constant))
    return dividend.getValue() > 0 && dividend.getValue() % divisor == 0 &&
           quotient.getValue() == dividend.getValue() / divisor;
  if (quotient.getKind() ==
          static_cast<uint32_t>(gpu::PhysicalExprKind::FloorDiv) &&
      quotient.getOperands().size() == 2 &&
      quotient.getOperands()[0] == dividendAttribute &&
      isConstantExtent(quotient.getOperands()[1], divisor))
    return true;
  // Symbolic extents are checked for every materialized provider config before
  // serialization.  The local op verifier rejects only an already-known
  // contradiction so a parameterized physical program remains well-formed.
  return true;
}

} // namespace

LogicalResult LaunchConfigOp::verify() {
  auto parameter = getThreads().getDefiningOp<gpu::ParameterOp>();
  if (!parameter ||
      parameter.getParameter().getRole() !=
          static_cast<uint32_t>(gpu::ParameterRole::ProviderThreads))
    return emitOpError("threads must be an explicit TileLang provider parameter");
  return success();
}

LogicalResult PipelineOp::verify() {
  auto parameter = getStages().getDefiningOp<gpu::ParameterOp>();
  if (!parameter ||
      parameter.getParameter().getRole() !=
          static_cast<uint32_t>(gpu::ParameterRole::ProviderStages))
    return emitOpError("stages must be an explicit provider-stage parameter");
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("requires one explicit body block");
  Block &block = getBody().front();
  if (std::distance(block.begin(), block.end()) != 2 ||
      !isa<scf::ForOp>(block.front()) || !isa<YieldOp>(block.back()))
    return emitOpError("must contain exactly one bufferized physical loop");
  auto loop = cast<scf::ForOp>(block.front());
  return loop.getNumResults() == 0
             ? success()
             : emitOpError("cannot pipeline an SSA-carried TileLang loop");
}

LogicalResult AllocOp::verify() { return success(); }

void AllocOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get());
}

LogicalResult ClearOp::verify() {
  return getBuffer().getType().getSpace().getValue() == BufferSpace::Fragment
             ? success()
             : emitOpError("clear requires a fragment accumulator allocation");
}

void ClearOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult FillOp::verify() {
  return getBuffer().getType().getElementType() == getValue().getType()
             ? success()
             : emitOpError("fill value must match the selected buffer element type");
}

void FillOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult SyncOp::verify() { return success(); }

LogicalResult CopyInOp::verify() {
  auto view = getSource().getType();
  auto buffer = getDestination().getType();
  if (getOffsets().size() != view.getRank() ||
      view.getElementType() != buffer.getElementType())
    return emitOpError(
        "requires one source-ordered offset per external view axis");
  return verifyCopyAxes(*this, getSourceAxes(), view.getRank(),
                        buffer.getShape().size());
}

void CopyInOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult CopyOutOp::verify() {
  auto buffer = getSource().getType();
  auto view = getDestination().getType();
  if (getOffsets().size() != view.getRank() ||
      view.getElementType() != buffer.getElementType())
    return emitOpError(
        "requires one destination-ordered offset per external view axis");
  return verifyCopyAxes(*this, getDestinationAxes(), view.getRank(),
                        buffer.getShape().size());
}

void CopyOutOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult CastCopyOutOp::verify() {
  auto buffer = getSource().getType();
  auto view = getDestination().getType();
  Type source = buffer.getElementType();
  Type destination = view.getElementType();
  bool numeric = isa<IntegerType, FloatType>(source) &&
                 isa<IntegerType, FloatType>(destination);
  if (getOffsets().size() != view.getRank() || !numeric ||
      source == destination)
    return emitOpError(
        "requires a numeric dtype-changing copy with one destination offset per view axis");
  return verifyCopyAxes(*this, getDestinationAxes(), view.getRank(),
                        buffer.getShape().size());
}

void CastCopyOutOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult ParallelOp::verify() {
  if (getShape().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("requires a non-empty physical shape and one body block");
  for (Attribute extent : getShape())
    if (!isa<gpu::PhysicalExprAttr>(extent))
      return emitOpError("shape extents must be typed physical expressions");
  Block &body = getBody().front();
  if (body.getNumArguments() != getShape().size() ||
      llvm::any_of(body.getArgumentTypes(),
                   [](Type type) { return !type.isIndex(); }) ||
      !isa<YieldOp>(body.getTerminator()))
    return emitOpError(
        "body must receive one index per physical axis and end in tilelang.yield");
  return success();
}

LogicalResult YieldOp::verify() {
  return isa<ParallelOp, PipelineOp>(getOperation()->getParentOp())
             ? success()
             : emitOpError("must terminate a TileLang parallel or pipeline region");
}

LogicalResult BufferLoadOp::verify() {
  auto buffer = getBuffer().getType();
  return getIndices().size() == buffer.getShape().size() &&
                 getResult().getType() == buffer.getElementType()
             ? success()
             : emitOpError("indices/result do not match the TileLang buffer");
}

LogicalResult BufferStoreOp::verify() {
  auto buffer = getBuffer().getType();
  return getIndices().size() == buffer.getShape().size() &&
                 getValue().getType() == buffer.getElementType()
             ? success()
             : emitOpError("indices/value do not match the TileLang buffer");
}

void BufferStoreOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult ViewLoadOp::verify() {
  auto view = getView().getType();
  if (getIndices().size() != view.getRank() ||
      getResult().getType() != view.getElementType())
    return emitOpError("indices/result do not match the external view");
  if (static_cast<bool>(getValid()) != static_cast<bool>(getFill()))
    return emitOpError("validity and fill must be present together");
  if (getValid() &&
      (!getValid().getType().isInteger(1) ||
       getFill().getType() != view.getElementType()))
    return emitOpError("validity/fill types do not match the external load");
  return success();
}

void ViewLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

LogicalResult ViewStoreOp::verify() {
  auto view = getView().getType();
  if (getIndices().size() != view.getRank() ||
      getValue().getType() != view.getElementType())
    return emitOpError("indices/value do not match the external view");
  if (getValid() && !getValid().getType().isInteger(1))
    return emitOpError("store validity must be scalar i1");
  return success();
}

void ViewStoreOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult ReduceOp::verify() {
  auto source = getSource().getType();
  auto destination = getDestination().getType();
  if (static_cast<size_t>(getAxis()) >= source.getShape().size() ||
      source.getElementType() != destination.getElementType() ||
      source.getSpace().getValue() != BufferSpace::Fragment ||
      destination.getSpace().getValue() != BufferSpace::Fragment)
    return emitOpError("has an invalid native reduction schema");
  switch (getKind()) {
  case BinaryOperator::Add:
  case BinaryOperator::MaximumNum:
  case BinaryOperator::MinimumNum:
    break;
  case BinaryOperator::LogicalAnd:
  case BinaryOperator::LogicalOr:
    if (!source.getElementType().isInteger(1))
      return emitOpError("logical native reduction requires i1 elements");
    break;
  case BinaryOperator::BitwiseAnd:
  case BinaryOperator::BitwiseOr:
  case BinaryOperator::BitwiseXor:
    if (!isa<IntegerType>(source.getElementType()))
      return emitOpError("bitwise native reduction requires integer elements");
    break;
  default:
    return emitOpError("has no semantics-preserving TileLang native reduction");
  }
  SmallVector<Attribute> expected;
  for (auto [axis, extent] : llvm::enumerate(source.getShape()))
    if (axis != static_cast<size_t>(getAxis()))
      expected.push_back(extent);
  if (expected.empty())
    expected.push_back(gpu::PhysicalExprAttr::get(
        getContext(), 0, 1, StringAttr::get(getContext()),
        ArrayAttr::get(getContext(), {})));
  return destination.getShape() == ArrayAttr::get(getContext(), expected)
             ? success()
             : emitOpError("destination shape is not the reduced source shape");
}

void ReduceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult ScanOp::verify() {
  auto source = getSource().getType();
  auto destination = getDestination().getType();
  return static_cast<size_t>(getAxis()) < source.getShape().size() &&
                 (getKind() == BinaryOperator::Add ||
                  getKind() == BinaryOperator::MaximumNum) &&
                 source == destination &&
                 source.getSpace().getValue() == BufferSpace::Fragment
             ? success()
             : emitOpError("has an invalid native scan schema");
}

void ScanOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult GemmOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto accumulator = getAccumulator().getType();
  if (lhs.getSpace().getValue() != BufferSpace::Shared ||
      rhs.getSpace().getValue() != BufferSpace::Shared ||
      accumulator.getSpace().getValue() != BufferSpace::Fragment ||
      lhs.getShape().size() != 2 ||
      rhs.getShape().size() != 2 || accumulator.getShape().size() != 2)
    return emitOpError("requires two shared operands and one fragment accumulator");
  Attribute lhsM = lhs.getShape()[getTransposeLhs() ? 1 : 0];
  Attribute lhsK = lhs.getShape()[getTransposeLhs() ? 0 : 1];
  Attribute rhsK = rhs.getShape()[getTransposeRhs() ? 1 : 0];
  Attribute rhsN = rhs.getShape()[getTransposeRhs() ? 0 : 1];
  if (lhsK != rhsK || accumulator.getShape()[0] != lhsM ||
      accumulator.getShape()[1] != rhsN)
    return emitOpError("TileLang GEMM operand and accumulator extents disagree");
  return success();
}

void GemmOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult SparseGemmOp::verify() {
  auto compressed = getCompressed().getType();
  auto metadata = getMetadata().getType();
  auto rhs = getRhs().getType();
  auto accumulator = getAccumulator().getType();
  if (compressed.getSpace().getValue() != BufferSpace::Shared ||
      metadata.getSpace().getValue() != BufferSpace::Shared ||
      rhs.getSpace().getValue() != BufferSpace::Shared ||
      accumulator.getSpace().getValue() != BufferSpace::Fragment ||
      compressed.getShape().size() != 2 || metadata.getShape().size() != 2 ||
      rhs.getShape().size() != 2 || accumulator.getShape().size() != 2)
    return emitOpError(
        "requires compressed/metadata/dense shared tiles and one fragment accumulator");
  Type dataType = compressed.getElementType();
  if ((!dataType.isF16() && !dataType.isBF16()) ||
      rhs.getElementType() != dataType ||
      !metadata.getElementType().isInteger(16) ||
      (accumulator.getElementType() != dataType &&
       !accumulator.getElementType().isF32()))
    return emitOpError(
        "supports two-of-four f16/bf16 data, packed i16 metadata, and f16/bf16/f32 accumulation");

  Attribute compressedM =
      compressed.getShape()[getTransposeCompressed() ? 1 : 0];
  Attribute compressedK =
      compressed.getShape()[getTransposeCompressed() ? 0 : 1];
  Attribute metadataM = metadata.getShape()[getTransposeMetadata() ? 1 : 0];
  Attribute metadataK = metadata.getShape()[getTransposeMetadata() ? 0 : 1];
  Attribute rhsK = rhs.getShape()[getTransposeRhs() ? 1 : 0];
  Attribute rhsN = rhs.getShape()[getTransposeRhs() ? 0 : 1];
  if (compressedM != metadataM || accumulator.getShape()[0] != compressedM ||
      accumulator.getShape()[1] != rhsN ||
      !hasNoStaticQuotientConflict(compressedK, rhsK, 2) ||
      !hasNoStaticQuotientConflict(metadataK, rhsK, 16))
    return emitOpError(
        "two-of-four sparse GEMM tile extents disagree with the dense logical K");
  return success();
}

void SparseGemmOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

} // namespace intent::tilelang

#define GET_OP_CLASSES
#include "Intent/Target/TileLang/IR/TileLangOps.cpp.inc"
