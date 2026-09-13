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

bool canPredicate(Block &block, bool allowStores = false) {
  for (Operation &operation : block.without_terminator()) {
    if (auto store = dyn_cast<StoreOp>(operation)) {
      if (!allowStores || !isScalar(store.getValue().getType()) ||
          !llvm::all_of(store.getCoordinates(), [](Value coordinate) {
            return isScalar(coordinate.getType());
          }))
        return false;
      continue;
    }
    if (operation.getNumRegions() || !operation.getNumResults() ||
        !llvm::all_of(operation.getResultTypes(), isScalar))
      return false;
    if (isa<LoadOp, GatherOp>(operation))
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
    auto maskedValidity = [&](Value valid) -> Value {
      if (!valid)
        return predicate;
      return builder.create<BinaryOp>(
          operation.getLoc(), builder.getI1Type(), predicate,
          mapping.lookupOrDefault(valid), BinaryOperator::LogicalAnd);
    };
    auto mappedCoordinates = [&](ValueRange coordinates) {
      SmallVector<Value> result;
      for (Value coordinate : coordinates)
        result.push_back(mapping.lookupOrDefault(coordinate));
      return result;
    };
    auto mappedFill = [&](Value fill, Type type) -> Value {
      return fill ? mapping.lookupOrDefault(fill)
                  : builder.create<arith::ConstantOp>(
                        operation.getLoc(), builder.getZeroAttr(type));
    };
    if (auto store = dyn_cast<StoreOp>(operation)) {
      auto replacement = builder.create<StoreOp>(
          store.getLoc(), mapping.lookupOrDefault(store.getResource()),
          mappedCoordinates(store.getCoordinates()),
          mapping.lookupOrDefault(store.getValue()),
          maskedValidity(store.getValid()), store.getSourceAxes());
      if (Attribute origin = store->getAttr(originAttr))
        replacement->setAttr(originAttr, origin);
      continue;
    }
    if (auto gather = dyn_cast<GatherOp>(operation)) {
      auto replacement = builder.create<GatherOp>(
          gather.getLoc(), gather.getType(),
          mapping.lookupOrDefault(gather.getSource()),
          mappedCoordinates(gather.getCoordinates()),
          maskedValidity(gather.getValid()),
          mappedFill(gather.getFill(), gather.getType()), gather.getSourceAxes());
      if (Attribute origin = gather->getAttr(originAttr))
        replacement->setAttr(originAttr, origin);
      mapping.map(gather.getResult(), replacement.getResult());
      continue;
    }
    auto load = dyn_cast<LoadOp>(operation);
    if (!load) {
      builder.clone(operation, mapping);
      continue;
    }
    // Untaken branches must not issue memory accesses, even when the original
    // read relied on its enclosing condition to establish valid coordinates.
    auto replacement = builder.create<LoadOp>(
        load.getLoc(), load.getType(), mapping.lookupOrDefault(load.getResource()),
        mappedCoordinates(load.getCoordinates()), maskedValidity(load.getValid()),
        mappedFill(load.getFill(), load.getType()), load.getSourceAxes());
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
    if (conditional->hasAttr(executionGroupAttr))
      continue;
    if (!conditional.getNumResults()) {
      bool emptyElse = conditional.getElseRegion().empty() ||
          conditional.getElseRegion().front().without_terminator().empty();
      if (!emptyElse ||
          !canPredicate(conditional.getThenRegion().front(),
                        /*allowStores=*/true))
        continue;
      OpBuilder builder(conditional);
      predicateBlock(builder, conditional.getThenRegion().front(),
                     conditional.getCondition());
      conditional.erase();
      continue;
    }
    if (conditional.getElseRegion().empty() ||
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
