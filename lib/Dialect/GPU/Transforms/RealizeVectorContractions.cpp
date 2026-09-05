#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

#include <numeric>

using namespace mlir;

namespace intent::gpu {
namespace {

AxisMapAttr onAxis(AxisMapAttr source, unsigned axis) {
  return AxisMapAttr::get(source.getContext(), source.getSourceId(),
                          source.getSourceAxis(), source.getDimensionId(), axis,
                          source.getDerived());
}

FragmentType withElement(FragmentType source, Type element) {
  return FragmentType::get(source.getContext(), element, source.getShape(),
                           source.getAxisMaps(), source.getValidity(),
                           source.getOwner());
}

FailureOr<Value> projectOperand(OpBuilder &builder, ContractOp contract,
                                Value operand, ArrayRef<unsigned> targetAxes,
                                FragmentType productType) {
  auto source = cast<FragmentType>(operand.getType());
  if (source.getElementType() != productType.getElementType()) {
    source = withElement(source, productType.getElementType());
    operand = builder.create<CastOp>(contract.getLoc(), source, operand);
  }
  SmallVector<int64_t> permutation(source.getShape().size());
  std::iota(permutation.begin(), permutation.end(), 0);
  llvm::sort(permutation, [&](int64_t left, int64_t right) {
    return targetAxes[left] < targetAxes[right];
  });
  if (!llvm::is_sorted(targetAxes)) {
    SmallVector<Attribute> shape;
    SmallVector<Attribute> mappings;
    for (auto [axis, oldAxis] : llvm::enumerate(permutation)) {
      shape.push_back(source.getShape()[oldAxis]);
      mappings.push_back(onAxis(cast<AxisMapAttr>(source.getAxisMaps()[oldAxis]), axis));
    }
    auto transposed = FragmentType::get(
        contract.getContext(), source.getElementType(), builder.getArrayAttr(shape),
        builder.getArrayAttr(mappings), source.getValidity(), source.getOwner());
    operand = builder.create<TransposeOp>(contract.getLoc(), transposed, operand, permutation);
  }
  return materializeBroadcastToFragment(builder, contract.getLoc(), operand, productType);
}

LogicalResult realizeVectorContract(ContractOp contract) {
  auto lhs = contract.getLhs().getType();
  auto rhs = contract.getRhs().getType();
  unsigned lhsFree = lhs.getShape().size() - contract.getLhsReductionAxes().size() -
                     contract.getLhsBatchAxes().size();
  unsigned rhsFree = rhs.getShape().size() - contract.getRhsReductionAxes().size() -
                     contract.getRhsBatchAxes().size();
  if (lhsFree != 0 && rhsFree != 0)
    return success();

  OpBuilder builder(contract);
  Location location = contract.getLoc();
  auto resultType = contract.getResult().getType();
  SmallVector<Attribute> shape(resultType.getShape().begin(), resultType.getShape().end());
  SmallVector<Attribute> mappings(resultType.getAxisMaps().begin(), resultType.getAxisMaps().end());
  SmallVector<unsigned> lhsAxes(lhs.getShape().size());
  SmallVector<unsigned> rhsAxes(rhs.getShape().size());
  unsigned resultAxis = 0;
  for (unsigned axis = 0; axis < lhsAxes.size(); ++axis)
    if (!llvm::is_contained(contract.getLhsReductionAxes(), axis))
      lhsAxes[axis] = resultAxis++;
  for (auto [left, right] : llvm::zip(contract.getLhsBatchAxes(), contract.getRhsBatchAxes()))
    rhsAxes[right] = lhsAxes[left];
  for (unsigned axis = 0; axis < rhsAxes.size(); ++axis)
    if (!llvm::is_contained(contract.getRhsReductionAxes(), axis) &&
        !llvm::is_contained(contract.getRhsBatchAxes(), axis))
      rhsAxes[axis] = resultAxis++;
  SmallVector<int64_t> reductionAxes;
  for (auto [left, right] : llvm::zip(contract.getLhsReductionAxes(), contract.getRhsReductionAxes())) {
    lhsAxes[left] = rhsAxes[right] = resultAxis;
    reductionAxes.push_back(resultAxis);
    shape.push_back(lhs.getShape()[left]);
    mappings.push_back(onAxis(cast<AxisMapAttr>(lhs.getAxisMaps()[left]), resultAxis++));
  }
  auto productType = FragmentType::get(
      contract.getContext(), resultType.getElementType(), builder.getArrayAttr(shape),
      builder.getArrayAttr(mappings), resultType.getValidity(), resultType.getOwner());
  FailureOr<Value> left = projectOperand(builder, contract, contract.getLhs(), lhsAxes, productType);
  FailureOr<Value> right = projectOperand(builder, contract, contract.getRhs(), rhsAxes, productType);
  if (failed(left) || failed(right))
    return contract.emitOpError("cannot project paired vector-contraction axes to a reduction fragment");
  auto product = builder.create<BinaryOp>(location, productType, *left, *right, BinaryOperator::Multiply);
  FailureOr<Value> zero = materializeZeroFragment(builder, location, resultType);
  if (failed(zero))
    return contract.emitOpError("cannot form vector-contraction additive identity");
  Type reducedType = resultType.getShape().empty() ? resultType.getElementType() : Type(resultType);
  Value identity = *zero;
  if (resultType.getShape().empty())
    identity = identity.getDefiningOp<BroadcastOp>().getValue();
  OperationState state(location, ReduceOp::getOperationName());
  state.addOperands({product.getResult(), identity});
  state.addTypes(reducedType);
  state.addAttribute("axes", builder.getDenseI64ArrayAttr(reductionAxes));
  state.addAttribute("source_count", builder.getI64IntegerAttr(1));
  state.addAttribute("identity_count", builder.getI64IntegerAttr(1));
  state.addAttribute("capture_count", builder.getI64IntegerAttr(0));
  state.addRegion();
  auto reduction = cast<ReduceOp>(builder.create(state));
  Block *combine = builder.createBlock(&reduction.getCombine(), {},
                                      {reducedType, reducedType}, {location, location});
  Value combined = builder.create<BinaryOp>(location, reducedType,
      combine->getArgument(0), combine->getArgument(1), BinaryOperator::Add);
  builder.create<YieldOp>(location, combined);
  builder.setInsertionPoint(contract);
  Value result = reduction.getResult(0);
  if (resultType.getShape().empty())
    result = builder.create<SplatOp>(location, resultType, result);
  auto accumulated = builder.create<BinaryOp>(location, resultType, result,
                                               contract.getAccumulator(), BinaryOperator::Add);
  if (Attribute origin = contract->getAttr(originAttr)) {
    product->setAttr(originAttr, origin);
    reduction->setAttr(originAttr, origin);
    accumulated->setAttr(originAttr, origin);
  }
  contract.getResult().replaceAllUsesWith(accumulated.getResult());
  contract.erase();
  return success();
}

} // namespace

LogicalResult realizeVectorContractions(ModuleOp module) {
  SmallVector<ContractOp> contracts;
  module.walk([&](ContractOp contract) { contracts.push_back(contract); });
  for (ContractOp contract : contracts)
    if (failed(realizeVectorContract(contract)))
      return failure();
  return success();
}

} // namespace intent::gpu
