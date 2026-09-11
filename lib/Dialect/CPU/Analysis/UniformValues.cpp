#include "Intent/Dialect/CPU/Analysis/UniformValues.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
namespace intent::cpu {

Attribute foldUniformComputation(linalg::GenericOp operation,
    const UniformBindings &operands, const UniformBindings &scalarFacts) {
  if (operation.getOutputs().size() != 1 || operation.getNumResults()) return {};
  Block &body = operation.getRegion().front();
  if (llvm::any_of(body.without_terminator(), [](Operation &nested) {
        return nested.getNumRegions() || !isMemoryEffectFree(&nested);
      })) return {};
  UniformValueAnalysis values(describeScalarValue);
  auto maps = operation.getIndexingMapsArray();
  Value output = operation.getOutputs()[0];
  UniformBindings bindings = scalarFacts;
  for (auto &operand : operands) bindings[operand.first] = operand.second;
  UniformExpression expression;
  expression.type = cast<MemRefType>(output.getType()).getElementType();
  if (isMatrixContraction(operation)) {
    expression.kind = UniformKind::Contract;
    expression.operands.assign(operation.getInputs().begin(), operation.getInputs().end());
    expression.operands.push_back(output);
    return values.fold(expression, bindings);
  }
  if (!operation.getNumReductionLoops() && maps.back().isPermutation()) {
    for (auto [argument, operand] : llvm::zip(body.getArguments(), operation->getOperands()))
      bindings[argument] = operands.lookup(operand);
    return values.evaluate(body.getTerminator()->getOperand(0), bindings);
  }
  if (!operation.getNumReductionLoops()) return {};
  expression.kind = UniformKind::Fold;
  expression.stateCount = 1;
  expression.operands.push_back(output);
  llvm::append_range(expression.operands, operation.getInputs());
  expression.parameters.push_back(body.getArguments().back());
  llvm::append_range(expression.parameters, body.getArguments().drop_back());
  expression.yields.push_back(body.getTerminator()->getOperand(0));
  auto iterators = operation.getIteratorTypesArray();
  SmallVector<bool> positive(iterators.size(), false);
  for (auto [operand, map] : llvm::zip(operation->getOperands(), maps)) {
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type) continue;
    for (auto [axis, coordinate] : llvm::enumerate(map.getResults()))
      if (auto dimension = dyn_cast<AffineDimExpr>(coordinate))
        positive[dimension.getPosition()] = positive[dimension.getPosition()] || type.getDimSize(axis) > 0;
  }
  expression.nonempty = llvm::all_of(llvm::enumerate(iterators), [&](auto iterator) {
    return iterator.value() != utils::IteratorType::reduction || positive[iterator.index()];
  });
  return values.fold(expression, bindings);
}

}
