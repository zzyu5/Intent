#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool isScalar(Type type) {
  return isa<IntegerType, IndexType, FloatType>(type);
}

bool canPredicate(Block &block) {
  for (Operation &operation : block.without_terminator()) {
    if (operation.getNumRegions() || !operation.getNumResults() ||
        !llvm::all_of(operation.getResultTypes(), isScalar))
      return false;
    if (isa<LoadOp>(operation))
      continue;
    if (auto binary = dyn_cast<BinaryOp>(operation)) {
      auto kind = binary.getOperatorKind();
      if (kind == BinaryOperator::FloorDivide ||
          kind == BinaryOperator::Remainder ||
          kind == BinaryOperator::LeftShift ||
          kind == BinaryOperator::RightShift)
        return false;
    }
    if (auto cast = dyn_cast<CastOp>(operation)) {
      if ((isa<FloatType>(cast.getValue().getType()) &&
           !isa<FloatType>(cast.getType())) ||
          isa<Float8E4M3FNType>(cast.getType()))
        return false;
    }
    if (!isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
             DimOp, PhysicalExprOp, arith::ConstantOp>(operation) ||
        !isSpeculatable(&operation) || !isMemoryEffectFree(&operation))
      return false;
  }
  return true;
}

SmallVector<Value> predicateBlock(OpBuilder &builder, Block &block,
                                  Value predicate) {
  IRMapping mapping;
  for (Operation &operation : block.without_terminator()) {
    auto load = dyn_cast<LoadOp>(operation);
    if (!load) {
      builder.clone(operation, mapping);
      continue;
    }
    SmallVector<Value> coordinates;
    for (Value coordinate : load.getCoordinates())
      coordinates.push_back(mapping.lookupOrDefault(coordinate));
    Value valid = predicate;
    if (load.getValid())
      valid = builder.create<BinaryOp>(
          load.getLoc(), builder.getI1Type(), predicate,
          mapping.lookupOrDefault(load.getValid()), BinaryOperator::LogicalAnd);
    Value fill = load.getFill()
                     ? mapping.lookupOrDefault(load.getFill())
                     : builder.create<arith::ConstantOp>(
                           load.getLoc(), builder.getZeroAttr(load.getType()));
    // Untaken branches must not issue memory accesses, even when the original
    // read relied on its enclosing condition to establish valid coordinates.
    auto replacement = builder.create<LoadOp>(
        load.getLoc(), load.getType(), mapping.lookupOrDefault(load.getResource()),
        coordinates, valid, fill, load.getSourceAxes());
    if (Attribute origin = load->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    mapping.map(load.getResult(), replacement.getResult());
  }
  SmallVector<Value> results;
  for (Value value : block.getTerminator()->getOperands())
    results.push_back(mapping.lookupOrDefault(value));
  return results;
}

} // namespace

LogicalResult predicateScalarControl(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  SmallVector<scf::IfOp> conditionals;
  kernel->walk<WalkOrder::PostOrder>(
      [&](scf::IfOp conditional) { conditionals.push_back(conditional); });
  for (scf::IfOp conditional : conditionals) {
    if (!conditional.getNumResults() || conditional.getElseRegion().empty() ||
        !llvm::all_of(conditional.getResultTypes(), isScalar) ||
        !canPredicate(conditional.getThenRegion().front()) ||
        !canPredicate(conditional.getElseRegion().front()))
      continue;
    OpBuilder builder(conditional);
    Value otherwise = builder.create<UnaryOp>(
        conditional.getLoc(), builder.getI1Type(), conditional.getCondition(),
        UnaryOperator::Not);
    SmallVector<Value> thenValues = predicateBlock(
        builder, conditional.getThenRegion().front(), conditional.getCondition());
    SmallVector<Value> elseValues = predicateBlock(
        builder, conditional.getElseRegion().front(), otherwise);
    for (auto [result, thenValue, elseValue] :
         llvm::zip(conditional.getResults(), thenValues, elseValues)) {
      auto replacement = builder.create<SelectOp>(
          conditional.getLoc(), result.getType(), conditional.getCondition(),
          thenValue, elseValue);
      if (Attribute origin = conditional->getAttr(originAttr))
        replacement->setAttr(originAttr, origin);
      result.replaceAllUsesWith(replacement.getResult());
    }
    conditional.erase();
  }
  eraseDeadPhysicalValues(*kernel);
  return success();
}

} // namespace intent::gpu
