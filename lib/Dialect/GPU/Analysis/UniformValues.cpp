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

Value uniformScalarSource(Value value) {
  while (value && isa<FragmentType>(value.getType())) {
    UniformExpression expression = describeUniformValue(value);
    if (expression.kind != UniformKind::Forward ||
        expression.operands.size() != 1)
      return {};
    value = expression.operands.front();
  }
  return value && value.getType().isIntOrIndexOrFloat() ? value : Value();
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
    return describeUniformUnary(result.type, unary.getOperatorKind(), unary.getInput(),
                                 unary.getApproximate(), unary.getFlushToZero());
  } else if (auto binary = dyn_cast<BinaryOp>(op)) {
    return describeUniformBinary(result.type, binary.getOperatorKind(), binary.getLhs(), binary.getRhs(),
                                  binary.getApproximate(), binary.getFlushToZero());
  } else if (auto compare = dyn_cast<CompareOp>(op)) {
    return describeUniformCompare(result.type, uniformElementType(compare.getLhs().getType()),
                                   compare.getPredicate(), compare.getLhs(), compare.getRhs());
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
