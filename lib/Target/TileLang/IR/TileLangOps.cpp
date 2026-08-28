#include "Intent/Target/TileLang/IR/TileLangOps.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::tilelang {

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
  return getBuffer().getType().getSpace() == 1
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

LogicalResult CopyInOp::verify() {
  auto view = getSource().getType();
  auto buffer = getDestination().getType();
  return getOffsets().size() == view.getRank() &&
                 view.getElementType() == buffer.getElementType()
             ? success()
             : emitOpError(
                   "requires one source-ordered offset per external view axis");
}

void CopyInOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult CopyOutOp::verify() {
  auto buffer = getSource().getType();
  auto view = getDestination().getType();
  return getOffsets().size() == view.getRank() &&
                 view.getElementType() == buffer.getElementType()
             ? success()
             : emitOpError(
                   "requires one destination-ordered offset per external view axis");
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
  return getOffsets().size() == view.getRank() && numeric &&
                 source != destination
             ? success()
             : emitOpError(
                   "requires a numeric dtype-changing copy with one destination offset per view axis");
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
      getKind() > 5 ||
      source.getElementType() != destination.getElementType() ||
      source.getSpace() != 1 || destination.getSpace() != 1)
    return emitOpError("has an invalid native reduction schema");
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
                 (getKind() == 0 || getKind() == 1) && source == destination &&
                 source.getSpace() == 1
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
  if (lhs.getSpace() != 0 || rhs.getSpace() != 0 ||
      accumulator.getSpace() != 1 || lhs.getShape().size() != 2 ||
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

} // namespace intent::tilelang

#define GET_OP_CLASSES
#include "Intent/Target/TileLang/IR/TileLangOps.cpp.inc"
