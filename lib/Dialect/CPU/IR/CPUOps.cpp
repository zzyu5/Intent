#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace intent::cpu;

#define GET_OP_CLASSES
#include "Intent/Dialect/CPU/IR/CPUOps.cpp.inc"

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
    auto add = yield.getValue().getDefiningOp<arith::AddFOp>();
    Value accumulator = block.getArgument(0);
    if (!add || !accumulator.hasOneUse() ||
        (add.getLhs() != accumulator && add.getRhs() != accumulator))
      return emitOpError("adjacent reassociation requires one floating-add accumulator consumer");
  }
  return success();
}
