#include "Construction.h"

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
  if (value && value.getType().isIndex()) return value;
  if (value && value.getType().isInteger(1)) value = b.create<arith::ExtUIOp>(loc, b.getI64Type(), value);
  if (value && isa<IntegerType>(value.getType())) return b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
  return {};
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
  Location loc = operation->getLoc();
  if (auto constant = dyn_cast<ConstantOp>(operation)) {
    Value value;
    if (constant.getResult().getType().isIndex()) value = index(loc, cast<IntegerAttr>(constant.getValue()).getInt());
    else if (auto floating = dyn_cast<FloatType>(constant.getResult().getType())) {
      auto number = cast<FloatAttr>(constant.getValue()).getValue();
      bool losesInformation;
      number.convert(floating.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInformation);
      value = b.create<arith::ConstantOp>(loc, floating, FloatAttr::get(floating, number));
    } else value = b.create<arith::ConstantOp>(loc, constant.getResult().getType(),
        b.getIntegerAttr(constant.getResult().getType(), cast<IntegerAttr>(constant.getValue()).getValue()));
    values.map(constant.getResult(), value); return success();
  }
  if (auto castOp = dyn_cast<CastOp>(operation)) {
    if (castOp.getRounding()) return castOp.emitError("DSA explicit rounding is not implemented");
    Value input = get(castOp.getInput());
    if (!input) return castOp.emitError("DSA cast input is unavailable");
    Value output = scalarCast(loc, input, castOp.getResult().getType());
    if (!output) return castOp.emitError("unsupported DSA scalar cast");
    values.map(castOp.getResult(), output);
    return success();
  }
  if (auto binary = dyn_cast<BinaryOp>(operation)) {
    Value lhs = get(binary.getLhs()), rhs = get(binary.getRhs());
    if (!lhs || !rhs) return binary.emitError("DSA binary operand is unavailable");
    if (binary.getOperatorKind() == BinaryOperator::TrueDivide && binary.getApproximate() && lhs.getType().isF32()) {
      Value left = allocate(loc, b.getF32Type(), 1, 1), right = allocateLike(loc, left), output = allocateLike(loc, left);
      b.create<dsa::FillOp>(loc, left, lhs); b.create<dsa::FillOp>(loc, right, rhs);
      b.create<dsa::BinaryOp>(loc, left, right, output, binary.getOperatorKindAttr(), binary.getApproximateAttr(), binary.getFlushToZeroAttr(), Value());
      values.map(binary.getResult(), loadLocal(loc, output, {})); return success();
    }
    if (binary.getApproximate() || binary.getFlushToZero()) return binary.emitError("DSA scalar approximate arithmetic is not implemented");
    Value output;
    bool floating = isa<FloatType>(lhs.getType());
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add: output = floating ? Value(b.create<arith::AddFOp>(loc, lhs, rhs)) : add(loc, lhs, rhs); break;
    case BinaryOperator::Subtract: output = floating ? Value(b.create<arith::SubFOp>(loc, lhs, rhs)) : sub(loc, lhs, rhs); break;
    case BinaryOperator::Multiply: output = floating ? Value(b.create<arith::MulFOp>(loc, lhs, rhs)) : mul(loc, lhs, rhs); break;
    case BinaryOperator::TrueDivide: if (floating) output = b.create<arith::DivFOp>(loc, lhs, rhs); break;
    case BinaryOperator::FloorDivide: if (!floating) output = b.create<arith::FloorDivSIOp>(loc, lhs, rhs); break;
    case BinaryOperator::Remainder:
      if (!floating) output = sub(loc, lhs, mul(loc, b.create<arith::FloorDivSIOp>(loc, lhs, rhs), rhs));
      break;
    case BinaryOperator::LogicalAnd: case BinaryOperator::BitwiseAnd: output = b.create<arith::AndIOp>(loc, lhs, rhs); break;
    case BinaryOperator::LogicalOr: case BinaryOperator::BitwiseOr: output = b.create<arith::OrIOp>(loc, lhs, rhs); break;
    case BinaryOperator::BitwiseXor: output = b.create<arith::XOrIOp>(loc, lhs, rhs); break;
    case BinaryOperator::LeftShift: output = b.create<arith::ShLIOp>(loc, lhs, rhs); break;
    case BinaryOperator::RightShift: output = b.create<arith::ShRSIOp>(loc, lhs, rhs); break;
    case BinaryOperator::Maximum: output = floating ? Value(b.create<arith::MaximumFOp>(loc, lhs, rhs)) : Value(b.create<arith::MaxSIOp>(loc, lhs, rhs)); break;
    case BinaryOperator::Minimum: output = floating ? Value(b.create<arith::MinimumFOp>(loc, lhs, rhs)) : Value(b.create<arith::MinSIOp>(loc, lhs, rhs)); break;
    case BinaryOperator::MaximumNum: output = floating ? Value(b.create<arith::MaxNumFOp>(loc, lhs, rhs)) : Value(b.create<arith::MaxSIOp>(loc, lhs, rhs)); break;
    case BinaryOperator::MinimumNum: output = floating ? Value(b.create<arith::MinNumFOp>(loc, lhs, rhs)) : Value(b.create<arith::MinSIOp>(loc, lhs, rhs)); break;
    default: break;
    }
    if (!output) return binary.emitError("unsupported DSA scalar operation");
    values.map(binary.getResult(), output); return success();
  }
  if (auto compare = dyn_cast<CompareOp>(operation)) {
    Value lhs = get(compare.getLhs()), rhs = get(compare.getRhs());
    if (!lhs || !rhs || isa<MemRefType>(lhs.getType()) || isa<MemRefType>(rhs.getType()))
      return compare.emitError("DSA comparison requires scalar operands");
    static constexpr arith::CmpIPredicate integer[] = {arith::CmpIPredicate::eq, arith::CmpIPredicate::ne,
        arith::CmpIPredicate::slt, arith::CmpIPredicate::sle, arith::CmpIPredicate::sgt, arith::CmpIPredicate::sge};
    static constexpr arith::CmpFPredicate floating[] = {arith::CmpFPredicate::OEQ, arith::CmpFPredicate::UNE,
        arith::CmpFPredicate::OLT, arith::CmpFPredicate::OLE, arith::CmpFPredicate::OGT, arith::CmpFPredicate::OGE};
    unsigned predicate = static_cast<unsigned>(compare.getPredicate());
    Value output = isa<FloatType>(lhs.getType()) ? Value(b.create<arith::CmpFOp>(loc, floating[predicate], lhs, rhs))
        : Value(b.create<arith::CmpIOp>(loc, integer[predicate], lhs, rhs));
    values.map(compare.getResult(), output); return success();
  }
  if (auto select = dyn_cast<SelectOp>(operation)) {
    Value condition = get(select.getCondition()), lhs = get(select.getTrueValue()), rhs = get(select.getFalseValue());
    if (!condition || !lhs || !rhs) return select.emitError("DSA selection operand is unavailable");
    if (!condition.getType().isInteger(1)) return select.emitError("DSA scalar selection needs a scalar predicate");
    values.map(select.getResult(), b.create<arith::SelectOp>(loc, condition, lhs, rhs));
    return success();
  }
  if (auto mask = dyn_cast<MaskOp>(operation)) {
    Value condition = get(mask.getPredicate()), value = get(mask.getValue()), fill = get(mask.getFill());
    if (!condition || !value || !fill || !condition.getType().isInteger(1)) return mask.emitError("DSA mask needs a scalar predicate");
    values.map(mask.getResult(), b.create<arith::SelectOp>(loc, condition, value, fill));
    return success();
  }
  if (auto unary = dyn_cast<UnaryOp>(operation)) {
    Value input = get(unary.getInput());
    if (!input) return unary.emitError("DSA unary operand is unavailable");
    if (unary.getOperatorKind() == UnaryOperator::Exp2 && unary.getApproximate() && input.getType().isF32()) {
      Value source = allocate(loc, b.getF32Type(), 1, 1), output = allocateLike(loc, source);
      b.create<dsa::FillOp>(loc, source, input);
      b.create<dsa::UnaryOp>(loc, source, output, unary.getOperatorKindAttr(), unary.getApproximateAttr(), unary.getFlushToZeroAttr(), Value());
      values.map(unary.getResult(), loadLocal(loc, output, {})); return success();
    }
    if (unary.getApproximate() || unary.getFlushToZero()) return unary.emitError("DSA scalar numerical mode is not implemented");
    Value output;
    switch (unary.getOperatorKind()) {
    case UnaryOperator::Sigmoid: {
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
      output = scalarCast(loc, b.create<arith::DivFOp>(loc, numerator, denominator), declared);
      break;
    }
    case UnaryOperator::Exp: output = b.create<math::ExpOp>(loc, input); break;
    case UnaryOperator::Exp2: output = b.create<math::Exp2Op>(loc, input); break;
    case UnaryOperator::Log: output = b.create<math::LogOp>(loc, input); break;
    case UnaryOperator::Sqrt: output = b.create<math::SqrtOp>(loc, input); break;
    case UnaryOperator::Rsqrt: output = b.create<math::RsqrtOp>(loc, input); break;
    case UnaryOperator::Tanh: output = b.create<math::TanhOp>(loc, input); break;
    case UnaryOperator::Sin: output = b.create<math::SinOp>(loc, input); break;
    case UnaryOperator::Cos: output = b.create<math::CosOp>(loc, input); break;
    case UnaryOperator::Floor: output = b.create<math::FloorOp>(loc, input); break;
    case UnaryOperator::Abs: output = b.create<math::AbsFOp>(loc, input); break;
    case UnaryOperator::Negate: output = b.create<arith::NegFOp>(loc, input); break;
    default: return unary.emitError("DSA scalar unary operation is not implemented");
    }
    values.map(unary.getResult(), output); return success();
  }
  return operation->emitError("operation has no DSA construction implementation");
}

// Reuse the scalar numerical lowering for elementwise tensor operations.
FailureOr<Value> Construction::scalarOperation(Operation *source, ValueRange operands) {
  OperationState state(source->getLoc(), source->getName());
  state.addAttributes(source->getAttrs());
  SmallVector<Value> scalars;
  for (auto [operand, value] : llvm::zip(source->getOperands(), operands)) {
    Type type = operand.getType();
    if (auto tensor = dyn_cast<RankedTensorType>(type)) type = tensor.getElementType();
    value = scalarCast(source->getLoc(), value, type);
    if (!value) return source->emitError("DSA element has incompatible scalar dtype"), failure();
    scalars.push_back(value);
  }
  state.addOperands(scalars);
  state.addTypes(getElementTypeOrSelf(source->getResult(0).getType()));
  Operation *scalar = Operation::create(state);
  auto saved = values;
  for (Value value : scalars) values.map(value, value);
  LogicalResult status = lowerScalarOperation(scalar);
  Value result = values.lookupOrNull(scalar->getResult(0));
  values = std::move(saved);
  scalar->destroy();
  if (failed(status) || !result) return failure();
  return result;
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
  if (!value) return {};
  if (value.getType() == type) return value;
  if (value.getType().isIndex() && isa<FloatType>(type))
    return scalarCast(loc, b.create<arith::IndexCastOp>(loc, b.getI64Type(), value), type);
  if (isa<FloatType>(value.getType()) && type.isIndex())
    return b.create<arith::IndexCastOp>(loc, type, b.create<arith::FPToSIOp>(loc, b.getI64Type(), value));
  if (value.getType().isInteger(1) && type.isIndex()) return asIndex(value, loc);
  if (type.isInteger(1)) {
    if (isa<FloatType>(value.getType())) {
      Value zero = b.create<arith::ConstantOp>(loc, b.getFloatAttr(value.getType(), 0.0));
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, value, zero);
    }
    Value zero = value.getType().isIndex() ? index(loc, 0)
        : Value(b.create<arith::ConstantOp>(loc, b.getIntegerAttr(value.getType(), 0)));
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, value, zero);
  }
  if (isa<FloatType>(value.getType()) && isa<FloatType>(type)) {
    if (value.getType().getIntOrFloatBitWidth() == type.getIntOrFloatBitWidth()) {
      Value wider = b.create<arith::ExtFOp>(loc, b.getF32Type(), value);
      return b.create<arith::TruncFOp>(loc, type, wider);
    }
    if (value.getType().getIntOrFloatBitWidth() < type.getIntOrFloatBitWidth())
      return b.create<arith::ExtFOp>(loc, type, value);
    return b.create<arith::TruncFOp>(loc, type, value);
  }
  if (value.getType().isIndex() && isa<IntegerType>(type)) return b.create<arith::IndexCastOp>(loc, type, value);
  if (isa<IntegerType>(value.getType()) && type.isIndex()) return b.create<arith::IndexCastOp>(loc, type, value);
  if (isa<IntegerType>(value.getType()) && isa<IntegerType>(type)) {
    if (value.getType().getIntOrFloatBitWidth() > type.getIntOrFloatBitWidth()) return b.create<arith::TruncIOp>(loc, type, value);
    if (value.getType().isInteger(1)) return b.create<arith::ExtUIOp>(loc, type, value);
    return b.create<arith::ExtSIOp>(loc, type, value);
  }
  if (value.getType().isInteger(1) && isa<FloatType>(type)) return b.create<arith::UIToFPOp>(loc, type, value);
  if (isa<IntegerType>(value.getType()) && isa<FloatType>(type)) return b.create<arith::SIToFPOp>(loc, type, value);
  if (isa<FloatType>(value.getType()) && isa<IntegerType>(type)) return b.create<arith::FPToSIOp>(loc, type, value);
  return {};
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
      } else slots.push_back(allocate(loc, field.isIndex() || isa<LogicalIndexType>(field) ? b.getI64Type() : field, 1, 1));
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
