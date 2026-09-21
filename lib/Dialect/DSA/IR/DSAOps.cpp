#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
using namespace mlir;
using namespace intent::dsa;
#define GET_OP_CLASSES
#include "Intent/Dialect/DSA/IR/DSAOps.cpp.inc"
namespace {
bool tile(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 2 && type.hasStaticShape() &&
      type.getNumElements() > 0 && (type.getElementType().isF16() || type.getElementType().isF32() ||
                                  type.getElementType().isInteger(1) || type.getElementType().isInteger(32) || type.getElementType().isInteger(64)) &&
      type.getMemorySpaceAsInt() == nramSpace;
}
bool same(Value a, Value b) { return a.getType() == b.getType() && tile(a); }
}
LogicalResult TaskIdOp::verify() { return success(); }
LogicalResult SynchronizeOp::verify() { return success(); }
LogicalResult TaskCountOp::verify() { return success(); }
LogicalResult StrideOp::verify() {
  auto type = cast<MemRefType>(getSource().getType());
  return getAxis() < static_cast<uint64_t>(type.getRank()) ? success() : emitOpError("invalid resource axis");
}
LogicalResult LoadScalarOp::verify() {
  return cast<MemRefType>(getSource().getType()).getElementType() == getResult().getType()
      ? success() : emitOpError("scalar load changes the storage dtype");
}
LogicalResult StoreScalarOp::verify() {
  return cast<MemRefType>(getDestination().getType()).getElementType() == getValue().getType()
      ? success() : emitOpError("scalar store changes the storage dtype");
}
LogicalResult LoadTileOp::verify() {
  return tile(getOutput()) && cast<MemRefType>(getSource().getType()).getElementType() ==
      cast<MemRefType>(getOutput().getType()).getElementType()
      ? success() : emitOpError("load must preserve dtype into a static local tile");
}
LogicalResult StoreTileOp::verify() {
  return tile(getInput()) && cast<MemRefType>(getDestination().getType()).getElementType() ==
      cast<MemRefType>(getInput().getType()).getElementType()
      ? success() : emitOpError("store must preserve dtype from a static local tile");
}
LogicalResult FillOp::verify() {
  return tile(getOutput()) && cast<MemRefType>(getOutput().getType()).getElementType() == getValue().getType()
      ? success() : emitOpError("fill value and local tile must have the same dtype");
}
LogicalResult SelectOp::verify() {
  auto predicate = dyn_cast<MemRefType>(getCondition().getType());
  bool condition = getCondition().getType().isInteger(1) || (tile(getCondition()) &&
      predicate.getElementType().isInteger(1) && predicate.getShape() == cast<MemRefType>(getOutput().getType()).getShape());
  return condition && same(getTrueValue(), getFalseValue()) && same(getTrueValue(), getOutput())
      ? success() : emitOpError("selection requires matching data tiles and a boolean scalar or tile");
}
LogicalResult UnaryOp::verify() {
  return same(getInput(), getOutput()) ? success() : emitOpError("unary tiles must match");
}
LogicalResult BinaryOp::verify() {
  return same(getLhs(), getRhs()) && same(getLhs(), getOutput())
      ? success() : emitOpError("binary tiles must match");
}
LogicalResult CastOp::verify() {
  return tile(getInput()) && tile(getOutput()) &&
      cast<MemRefType>(getInput().getType()).getShape() == cast<MemRefType>(getOutput().getType()).getShape()
      ? success() : emitOpError("cast must preserve local tile shape");
}
LogicalResult ReduceOp::verify() {
  auto input = cast<MemRefType>(getInput().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  return same(getInput(), getScratch()) && tile(getOutput()) && input.getDimSize(0) == 1 &&
      output.getNumElements() == 1 && input.getElementType() == output.getElementType() &&
      getIdentity().getType() == output.getElementType()
      ? success() : emitOpError("reduce requires one local row, matching scratch and scalar destination");
}
LogicalResult PrepareMatrixOp::verify() {
  auto input = cast<MemRefType>(getInput().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  auto scratch = cast<MemRefType>(getScratch().getType());
  return tile(getInput()) && tile(getScratch()) && output.hasStaticShape() && output.getRank() == 2 &&
      output.getMemorySpaceAsInt() == matrixSpace && input.getElementType() == output.getElementType() &&
      scratch.getElementType() == input.getElementType() &&
      output.getShape() == input.getShape() && scratch.getDimSize(0) == input.getDimSize(1) &&
      scratch.getDimSize(1) == input.getDimSize(0)
      ? success() : emitOpError("matrix preparation requires explicit transposed scratch and matrix-local storage");
}
LogicalResult MatMulOp::verify() {
  auto a = cast<MemRefType>(getLhs().getType()), b = cast<MemRefType>(getRhs().getType());
  auto c = cast<MemRefType>(getAccumulator().getType());
  return tile(getLhs()) && tile(getAccumulator()) && b.hasStaticShape() && b.getRank() == 2 &&
      a.getElementType() == b.getElementType() && c.getElementType().isF32() &&
      a.getDimSize(1) == b.getDimSize(0) && a.getDimSize(0) == c.getDimSize(0) &&
      b.getDimSize(1) == c.getDimSize(1) && (!getScratch() || same(getScratch(), getAccumulator()))
      ? success() : emitOpError("matmul requires matching M/K/N tiles and an f32 accumulator");
}
LogicalResult intent::dsa::verifyProgram(ModuleOp module, bool bound) {
  if (failed(mlir::verify(module))) return failure();
  auto functions = llvm::to_vector(module.getOps<func::FuncOp>());
  if (functions.size() != 1) return module.emitError("DSA artifact requires one physical kernel");
  auto function = functions.front();
  auto interface = function->getAttrOfType<InterfaceAttr>("intent_dsa.interface");
  auto config = function->getAttrOfType<ConfigurationAttr>("intent_dsa.configuration");
  if (!interface || !config || interface.getArguments().size() != function.getNumArguments())
    return function.emitError("DSA kernel requires a complete interface and configuration");
  auto walk = function.walk([&](Operation *op) {
    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect != "intent_dsa" && dialect != "arith" && dialect != "math" &&
        dialect != "scf" && dialect != "memref" && dialect != "func") {
      op->emitError("operation is outside the DSA execution program");
      return WalkResult::interrupt();
    }
    if (auto allocation = dyn_cast<memref::AllocaOp>(op)) {
      auto type = allocation.getType();
      if (!type.hasStaticShape() || type.getRank() != 2 ||
          (type.getMemorySpaceAsInt() != nramSpace && type.getMemorySpaceAsInt() != matrixSpace) ||
          (bound && !op->hasAttr("bangc.offset"))) {
        op->emitError("local allocation requires bounded shape, storage ownership and a bound offset");
        return WalkResult::interrupt();
      }
    }
    for (Type type : op->getResultTypes()) {
      if (isa<MemRefType>(type) && !isa<memref::AllocaOp>(op)) {
        op->emitError("DSA local buffers require explicit storage; alias results and escaping control state are not supported");
        return WalkResult::interrupt();
      }
    }
    if (auto yield = dyn_cast<scf::YieldOp>(op)) {
      if (yield->getNumOperands()) {
        op->emitError("DSA control results must be copied into explicit state storage before yielding");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}
