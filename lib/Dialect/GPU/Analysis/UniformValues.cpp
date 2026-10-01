#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"

using namespace mlir;
namespace intent::gpu {

Type uniformElementType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type)) return fragment.getElementType();
  return type;
}

UniformExpression describeUniformValue(Value value) {
  UniformExpression result = describeScalarValue(value);
  result.type = uniformElementType(value.getType());
  Operation *op = value.getDefiningOp();
  if (!op) return result;
  using K = UniformKind;
  if (auto physical = dyn_cast<PhysicalExprOp>(op)) {
    if (auto literal = constantPhysicalExpression(physical.getExpression())) {
      result.kind = K::Constant;
      result.literal = IntegerAttr::get(value.getType(), *literal);
    }
    return result;
  }
  if (isa<BroadcastOp, SplatOp, ReshapeOp, TransposeOp>(op)) result.kind = K::Forward;
  else if (isa<JoinOp>(op)) result.kind = K::Join;
  else if (isa<MakeRecordOp>(op)) result.kind = K::Aggregate;
  else if (auto extract = dyn_cast<ExtractOp>(op)) {
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
      result.kind = K::Forward;
      result.operands = {record.getFields()[extract.getField()]};
      return result;
    }
    result.kind = K::Extract;
    result.result = extract.getField();
  } else if (isa<SelectOp>(op)) result.kind = K::Select;
  else if (isa<CastOp>(op)) result.kind = K::Cast;
  else if (isa<BitcastOp>(op)) result.kind = K::Bitcast;
  else if (auto unary = dyn_cast<UnaryOp>(op)) {
    if (unary.getApproximate() || unary.getFlushToZero()) return result;
    switch (unary.getOperatorKind()) {
    case UnaryOperator::Negate: result.kind = K::Negate; break;
    case UnaryOperator::Not: result.kind = K::Not; break;
    case UnaryOperator::Exp: case UnaryOperator::Exp2: result.kind = K::Exp; break;
    default: break;
    }
  } else if (auto binary = dyn_cast<BinaryOp>(op)) {
    if (binary.getApproximate() || binary.getFlushToZero()) return result;
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add: result.kind = K::Add; break;
    case BinaryOperator::Subtract: result.kind = K::Subtract; break;
    case BinaryOperator::Multiply: result.kind = K::Multiply; break;
    case BinaryOperator::LogicalAnd: case BinaryOperator::BitwiseAnd: result.kind = K::And; break;
    case BinaryOperator::LogicalOr: case BinaryOperator::BitwiseOr: result.kind = K::Or; break;
    case BinaryOperator::BitwiseXor: result.kind = K::Xor; break;
    case BinaryOperator::Maximum: result.kind = K::Maximum; break;
    case BinaryOperator::Minimum: result.kind = K::Minimum; break;
    case BinaryOperator::MaximumNum: result.kind = K::MaximumNum; break;
    case BinaryOperator::MinimumNum: result.kind = K::MinimumNum; break;
    default: break;
    }
  } else if (auto compare = dyn_cast<CompareOp>(op)) {
    result.kind = K::Compare;
    switch (compare.getPredicate()) {
    case ComparePredicate::Eq: result.predicate = UniformPredicate::Equal; break;
    case ComparePredicate::Ne: result.predicate = UniformPredicate::NotEqual; result.unorderedTrue = true; break;
    case ComparePredicate::Lt: result.predicate = UniformPredicate::Less; break;
    case ComparePredicate::Le: result.predicate = UniformPredicate::LessEqual; break;
    case ComparePredicate::Gt: result.predicate = UniformPredicate::Greater; break;
    case ComparePredicate::Ge: result.predicate = UniformPredicate::GreaterEqual; break;
    default: result.kind = K::Unknown; break;
    }
  } else if (auto reduce = dyn_cast<ReduceOp>(op)) {
    bool nonempty = false;
    if (auto source = dyn_cast<FragmentType>(reduce.getSources().front().getType())) {
      auto kernel = reduce->getParentOfType<func::FuncOp>();
      nonempty = kernel && llvm::all_of(reduce.getAxes(), [&](int64_t axis) {
        return axis >= 0 && static_cast<unsigned>(axis) < source.getShape().size() &&
            isKnownPositiveExtent(cast<PhysicalExprAttr>(source.getShape()[axis]), kernel);
      });
    }
    return describeStructuredReduction(cast<OpResult>(value), result.type, nonempty);
  } else if (isa<ContractOp>(op)) result.kind = K::Contract;
  result.operands.assign(op->operand_begin(), op->operand_end());
  return result;
}

}
