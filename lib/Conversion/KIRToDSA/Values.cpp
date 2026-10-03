#include "Construction.h"
#include "Intent/Conversion/ScalarLowering.h"

namespace intent::kir_to_dsa {

Value Construction::index(Location loc, int64_t n) { return b.create<arith::ConstantIndexOp>(loc, n); }

Value Construction::add(Location loc, Value a, Value c) { return b.createOrFold<arith::AddIOp>(loc, a, c); }

Value Construction::mul(Location loc, Value a, Value c) { return b.createOrFold<arith::MulIOp>(loc, a, c); }

Value Construction::sub(Location loc, Value a, Value c) { return b.createOrFold<arith::SubIOp>(loc, a, c); }

Value Construction::get(Value source) {
  Value value = values.lookupOrNull(source);
  if (!value && source.getDefiningOp()) {
    if (failed(materialize(source.getDefiningOp()))) return {};
    value = values.lookupOrNull(source);
  }
  if (value) if (auto tensor = dyn_cast<RankedTensorType>(source.getType())) {
    value = projectTensor(source.getLoc(), tensor, value, source);
    if (value) values.map(source, value);
  }
  return value;
}

LogicalResult Construction::materialize(Operation *op) {
  if (!materializing.insert(op).second) return op->emitError("cyclic DSA value materialization");
  LogicalResult result = lowerOperation(op);
  materializing.erase(op);
  return result;
}

Value Construction::asIndex(Value value, Location loc) {
  if (!value || !value.getType().isIntOrIndex())
    return {};
  auto result = castScalarValue(b, loc, value, b.getIndexType());
  return succeeded(result) ? *result : Value();
}

Value Construction::extent(RankedTensorType type, unsigned axis, Location loc) {
  if (!type.isDynamicDim(axis)) return index(loc, type.getDimSize(axis));
  auto shape = dyn_cast_or_null<TensorShapeAttr>(type.getEncoding());
  return shape ? dimensions.lookup(shape.getDimensions()[axis]) : Value();
}

bool Construction::bindLogicalExtent(Value source, int64_t dimension, DenseSet<Value> &visited) {
  if (dimensions.count(dimension)) return true;
  if (!visited.insert(source).second) return false;
  Operation *op = source.getDefiningOp();
  if (!op) return false;
  if (isa<DomainOp, SubregionOp>(op)) {
    auto ids = op->getAttrOfType<ArrayAttr>("extent_dimensions");
    if (ids && llvm::any_of(ids, [&](Attribute id) { return cast<IntegerAttr>(id).getInt() == dimension; }))
      return bindDomain(source) && dimensions.count(dimension);
  }
  if (auto relation = op->getAttrOfType<ShapeRelationAttr>("shape")) {
    for (Attribute attribute : relation.getAxes()) {
      auto expression = cast<ShapeExprAttr>(attribute);
      if (expression.getDimension() != dimension) continue;
      Value size;
      if (expression.getKind() == 0) size = index(op->getLoc(), expression.getPayload());
      if (expression.getKind() == 1) size = asIndex(get(op->getOperand(expression.getPayload())), op->getLoc());
      if (size) { dimensions[dimension] = size; return true; }
    }
  }
  // Follow the actual value's definition without materializing its tensor.
  // A logical extent may be established in an ancestor workset's domain.
  for (Value operand : op->getOperands())
    if (bindLogicalExtent(operand, dimension, visited)) return true;
  return false;
}

Value Construction::logicalExtent(Value value, unsigned axis, Location loc) {
  auto type = cast<RankedTensorType>(value.getType());
  if (Value size = extent(type, axis, loc)) return size;
  auto identities = cast<TensorShapeAttr>(type.getEncoding()).getDimensions();
  DenseSet<Value> visited;
  bindLogicalExtent(value, identities[axis], visited);
  return extent(type, axis, loc);
}

bool Construction::sameIndex(Value a, Value c) {
  if (a == c) return true;
  APInt lhs, rhs;
  return matchPattern(a, m_ConstantInt(&lhs)) && matchPattern(c, m_ConstantInt(&rhs)) && lhs == rhs;
}

LogicalResult Construction::lowerScalarOperation(Operation *operation) {
  SmallVector<Value> operands;
  for (Value input : operation->getOperands()) {
    Value value = get(input);
    if (!value || isa<MemRefType>(value.getType()))
      return operation->emitError("DSA scalar operand is unavailable");
    operands.push_back(value);
  }
  auto result = scalarOperation(operation, operands);
  if (failed(result)) return failure();
  values.map(operation->getResult(0), *result);
  return success();
}

FailureOr<Value> Construction::scalarOperation(Operation *operation,
                                              ValueRange operands) {
  Location loc = operation->getLoc();
  SmallVector<Value> scalars;
  for (auto [input, value] : llvm::zip(operation->getOperands(), operands)) {
    auto scalar = castScalarValue(b, loc, value, getElementTypeOrSelf(input.getType()));
    if (failed(scalar))
      return operation->emitError("DSA element has incompatible scalar dtype"), failure();
    scalars.push_back(*scalar);
  }
  if (auto binary = dyn_cast<BinaryOp>(operation)) {
    Value lhs = scalars[0], rhs = scalars[1];
    if (binary.getOperatorKind() == BinaryOperator::TrueDivide && binary.getApproximate() && lhs.getType().isF32()) {
      Value left = allocate(loc, b.getF32Type(), 1, 1), right = allocateLike(loc, left), output = allocateLike(loc, left);
      b.create<dsa::FillOp>(loc, left, lhs); b.create<dsa::FillOp>(loc, right, rhs);
      b.create<dsa::BinaryOp>(loc, left, right, output, binary.getOperatorKindAttr(), binary.getApproximateAttr(), binary.getFlushToZeroAttr(), Value());
      return loadLocal(loc, output, {});
    }
    if (binary.getApproximate() || binary.getFlushToZero())
      return binary.emitError("DSA scalar approximate arithmetic is not implemented"), failure();
  }
  if (auto unary = dyn_cast<UnaryOp>(operation)) {
    Value input = scalars[0];
    if (unary.getOperatorKind() == UnaryOperator::Exp2 && unary.getApproximate() && input.getType().isF32()) {
      Value source = allocate(loc, b.getF32Type(), 1, 1), output = allocateLike(loc, source);
      b.create<dsa::FillOp>(loc, source, input);
      b.create<dsa::UnaryOp>(loc, source, output, unary.getOperatorKindAttr(), unary.getApproximateAttr(), unary.getFlushToZeroAttr(), Value());
      return loadLocal(loc, output, {});
    }
    if (unary.getApproximate() || unary.getFlushToZero())
      return unary.emitError("DSA scalar numerical mode is not implemented"), failure();
    if (unary.getOperatorKind() == UnaryOperator::Sigmoid) {
      Type declared = input.getType();
      if (declared.isF16() || declared.isBF16()) input = scalarCast(loc, input, b.getF32Type());
      Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(input.getType(), 1.0));
      Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(input.getType()));
      Value absolute = b.create<math::AbsFOp>(loc, input);
      Value negative = b.create<arith::NegFOp>(loc, absolute);
      Value exponential = b.create<math::ExpOp>(loc, negative);
      Value denominator = b.create<arith::AddFOp>(loc, one, exponential);
      Value belowZero = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, input, zero);
      Value numerator = b.create<arith::SelectOp>(loc, belowZero, exponential, one);
      return scalarCast(loc, b.create<arith::DivFOp>(loc, numerator, denominator), declared);
    }
  }
  return intent::lowerScalarOperation(operation, scalars, b);
}

bool Construction::constantTrue(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op) return false;
  if (isa<FullOp, BroadcastOp>(op)) return constantTrue(op->getOperand(0));
  auto constant = dyn_cast<ConstantOp>(op);
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
  return integer && integer.getType().isInteger(1) && integer.getValue().isOne();
}

Value Construction::scalarCast(Location loc, Value value, Type type) {
  auto result = castScalarValue(b, loc, value, type);
  return succeeded(result) ? *result : Value();
}

SmallVector<Value> Construction::flatten(ValueRange inputs) {
  SmallVector<Value> result;
  for (Value input : inputs) {
    if (getProductComponents(input.getType())) {
      if (!products.count(input) && input.getDefiningOp() && failed(materialize(input.getDefiningOp()))) return {};
      auto fields = products.lookup(input);
      {
        SmallVector<Type> schema; appendProductLeafTypes(input.getType(), schema);
        if (fields.size() != schema.size()) return {};
        for (unsigned i = 0; i < schema.size(); ++i) if (auto tensor = dyn_cast<RankedTensorType>(schema[i])) {
          fields[i] = projectTensor(input.getLoc(), tensor, fields[i]);
          if (!fields[i]) return {};
        }
        products[input] = fields;
      }
      llvm::append_range(result, fields);
    }
    else result.push_back(get(input));
  }
  return result;
}

void Construction::bindProduct(Value original, ValueRange fields) {
  if (getProductComponents(original.getType())) products[original] = llvm::to_vector(fields);
  else values.map(original, fields.front());
}

FailureOr<SmallVector<Value>> Construction::makeSlots(TypeRange types, Location loc) {
  SmallVector<Value> slots;
  for (Type type : types) {
    SmallVector<Type> flat;
    appendProductLeafTypes(type, flat);
    for (Type field : flat) {
      if (auto tensor = dyn_cast<RankedTensorType>(field)) {
        auto shape = localShape(tensor, loc);
        if (failed(shape)) return failure();
        slots.push_back(allocateTensor(loc, tensor.getElementType(), *shape));
      } else {
        Value slot = allocate(loc,
            field.isIndex() || isa<LogicalIndexType>(field) ? b.getI64Type() : field,
            1, 1);
        localShapes[slot] = {};
        slots.push_back(slot);
      }
    }
  }
  return slots;
}

LogicalResult Construction::copyTo(Value value, Value slot, Location loc) {
  if (!value) return emitError(loc, "DSA state field has no physical value");
  if (isa<MemRefType>(value.getType())) {
    if (value.getType() != slot.getType()) return emitError(loc, "DSA state update changes its local shape or dtype");
    b.create<memref::CopyOp>(loc, value, slot);
  } else {
    value = scalarCast(loc, value, cast<MemRefType>(slot.getType()).getElementType());
    if (!value) return emitError(loc, "DSA state scalar has no compatible storage type");
    b.create<memref::StoreOp>(loc, value, slot, ValueRange{index(loc, 0), index(loc, 0)});
  }
  return success();
}

void Construction::bindSlots(ValueRange originals, ValueRange slots, Location loc) {
  for (auto [original, range] : llvm::zip_equal(originals, getProductLeafRanges(originals.getTypes()))) {
    SmallVector<Type> flat;
    appendProductLeafTypes(original.getType(), flat);
    SmallVector<Value> fields;
    for (auto [type, slot] : llvm::zip_equal(flat, slots.slice(range.offset, range.size))) {
      Value value = slot;
      if (!isa<RankedTensorType>(type)) {
        value = b.create<memref::LoadOp>(loc, slot, ValueRange{index(loc, 0), index(loc, 0)});
        if (type.isIndex() || isa<LogicalIndexType>(type)) value = asIndex(value, loc);
      }
      fields.push_back(value);
    }
    bindProduct(original, fields);
  }
}

FailureOr<Value> Construction::lowerResults(Block &block, ValueRange slots) {
  if (failed(lowerOperations(block))) return failure();
  auto condition = dyn_cast<ConditionOp>(block.getTerminator());
  Value predicate = condition ? get(condition.getCondition()) : Value();
  auto fields = flatten(condition ? condition.getArgs()
                                 : cast<YieldOp>(block.getTerminator()).getInputs());
  if (fields.size() != slots.size()) return block.getParentOp()->emitError("DSA control state schema mismatch"), failure();
  for (auto [field, slot] : llvm::zip(fields, slots))
    if (failed(copyTo(field, slot, block.getParentOp()->getLoc()))) return failure();
  return predicate;
}

} // namespace intent::kir_to_dsa
