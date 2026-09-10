#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace intent::cpu;

#define GET_OP_CLASSES
#include "Intent/Dialect/CPU/IR/CPUOps.cpp.inc"

LogicalResult TasksOp::verify() {
  Block &body = getBody().front();
  if (body.empty() || body.getNumArguments() != getCaptures().size() + 1 ||
      !body.getArgument(0).getType().isIndex() ||
      !isa<TaskYieldOp>(body.getTerminator()))
    return emitOpError("tasks require an index coordinate, explicit captures and task_yield");
  for (auto [argument, capture] : llvm::zip(body.getArguments().drop_front(), getCaptures()))
    if (argument.getType() != capture.getType())
      return emitOpError("task capture and body argument types must agree");
  llvm::APInt count;
  if (matchPattern(getCount(), m_ConstantInt(&count)) && count.isNegative())
    return emitOpError("task count cannot be negative");
  if (getOperation()->getParentOfType<TasksOp>())
    return emitOpError("nested task scheduling has not been realized by CPU partitioning");
  return success();
}

LogicalResult ReduceOp::verify() {
  if (getInitial().getType() != getResult().getType() || getInputs().empty() ||
      getIndexingMaps().size() != getInputs().size() ||
      !llvm::hasSingleElement(getCombine()))
    return emitOpError("reduction requires one result/identity dtype, input maps and a single combine block");
  Block &block = getCombine().front();
  if (block.empty() || block.getNumArguments() != getInputs().size() + 1 ||
      block.getArgument(0).getType() != getResult().getType())
    return emitOpError("combine arguments must be accumulator followed by input elements");
  for (auto [number, input] : llvm::enumerate(getInputs())) {
    auto memory = dyn_cast<MemRefType>(input.getType());
    Type element = memory ? memory.getElementType() : input.getType();
    auto map = dyn_cast<AffineMapAttr>(getIndexingMaps()[number]);
    if (!map || map.getValue().getNumDims() != 1 || map.getValue().getNumSymbols() ||
        map.getValue().getNumResults() != (memory ? memory.getRank() : 0) ||
        block.getArgument(number + 1).getType() != element)
      return emitOpError("reduction input map or scalar combine type is incomplete");
    for (AffineExpr index : map.getValue().getResults())
      if (!isa<AffineDimExpr>(index) &&
          (!isa<AffineConstantExpr>(index) || cast<AffineConstantExpr>(index).getValue() != 0))
        return emitOpError("reduction maps require the reduction coordinate or a broadcast zero");
  }
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield || yield.getValue().getType() != getResult().getType())
    return emitOpError("combine must yield its accumulator dtype");
  for (Operation &operation : block.without_terminator())
    if (operation.getNumRegions() || !isMemoryEffectFree(&operation))
      return emitOpError("combine must be a closed pure scalar expression");
  if (getOrder().getAdjacentReassociation()) {
    Operation *combine = yield.getValue().getDefiningOp();
    Value accumulator = block.getArgument(0);
    if (!combine || !isa<arith::AddFOp, arith::MaxNumFOp>(combine) || !accumulator.hasOneUse() ||
        !llvm::is_contained(combine->getOperands(), accumulator))
      return emitOpError("adjacent reassociation requires one floating-add/maximumNumber accumulator consumer");
  }
  return success();
}

namespace {
bool recordType(Type type, int64_t bytes) {
  auto memory = dyn_cast<MemRefType>(type);
  return memory && memory.getRank() == 2 &&
      (memory.isDynamicDim(1) || memory.getDimSize(1) == bytes) &&
      memory.getElementType().isUnsignedInteger(8);
}
}

LogicalResult QuantizeOp::verify() {
  auto input = cast<MemRefType>(getInput().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  if (getFormat() != intent::QuantFormat::Q8K || input.getRank() != 2 ||
      (!input.isDynamicDim(1) && input.getDimSize(1) != 256) ||
      !input.getElementType().isF32() || !recordType(output, 292) ||
      (!input.isDynamicDim(0) && !output.isDynamicDim(0) &&
       input.getDimSize(0) != output.getDimSize(0)))
    return emitOpError("Q8_K preparation requires f32[G,256] and u8[G,292] storage");
  return success();
}

LogicalResult QuantizedDotOp::verify() {
  if (getLhsFormat() != intent::QuantFormat::Q4K || getRhsFormat() != intent::QuantFormat::Q8K ||
      !recordType(getLhs().getType(), 144) || !recordType(getRhs().getType(), 292))
    return emitOpError("quantized dot requires Q4_K and Q8_K record storage");
  auto lhs = cast<MemRefType>(getLhs().getType());
  auto rhs = cast<MemRefType>(getRhs().getType());
  if (!lhs.isDynamicDim(0) && !rhs.isDynamicDim(0) && lhs.getDimSize(0) != rhs.getDimSize(0))
    return emitOpError("quantized dot record extents must agree");
  return success();
}
