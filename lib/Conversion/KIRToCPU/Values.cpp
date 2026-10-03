#include "Construction.h"
#include "Intent/Analysis/ProductSchema.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace intent::kir_to_cpu {

Value Construction::constant(Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

FailureOr<Value> Construction::indexValue(Value value, Type logicalType, Location loc) {
  if (value.getType().isIndex()) return value;
  if (isa<IntegerType>(value.getType()) && !value.getType().isInteger(1)) {
    auto integer = dyn_cast<IntegerType>(logicalType);
    if (!integer && !logicalType.isIndex() && !isa<LogicalIndexType>(logicalType))
      return emitError(loc, "CPU coordinate has no integer interpretation"), failure();
    return integer && integer.isUnsigned()
        ? Value(builder.create<arith::IndexCastUIOp>(loc, builder.getIndexType(), value))
        : Value(builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), value));
  }
  return emitError(loc, "CPU coordinates require index or integer values"), failure();
}

Value Construction::domainExtent(Value begin, Value end, Value step, Location loc) {
  if (matchPattern(begin, m_Zero()) && matchPattern(step, m_One()) && end.getDefiningOp<memref::DimOp>())
    return end;
  Value distance = builder.createOrFold<arith::SubIOp>(loc, end, begin);
  Value nonnegative = builder.createOrFold<arith::MaxSIOp>(loc, distance, constant(loc, 0));
  return builder.createOrFold<arith::CeilDivSIOp>(loc, nonnegative, step);
}

FailureOr<SmallVector<Value>> Construction::extents(RankedTensorType tensor, Location loc) {
  SmallVector<Value> result;
  auto shape = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
  for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
    if (!tensor.isDynamicDim(axis)) {
      result.push_back(constant(loc, tensor.getDimSize(axis)));
      continue;
    }
    if (!shape || !dimensions.count(shape.getDimensions()[axis])) {
      emitError(loc, "CPU construction cannot resolve the canonical dynamic extent");
      return failure();
    }
    result.push_back(dimensions.lookup(shape.getDimensions()[axis]));
  }
  return result;
}

LogicalResult Construction::bindShape(Operation *operation) {
  auto shape = operation->getAttrOfType<ShapeRelationAttr>("shape");
  if (!shape) return success();
  Location loc = operation->getLoc();
  SmallVector<Value> sizes;
  std::optional<int64_t> inferred;
  for (Attribute axis : shape.getAxes()) {
    auto expression = cast<ShapeExprAttr>(axis);
    if (expression.getKind() == 2) {
      inferred = expression.getDimension();
      continue;
    }
    Value size;
    if (expression.getKind() == 0) {
      size = constant(loc, expression.getPayload());
    } else {
      Value operand = isa<BufferOp>(operation)
          ? cast<BufferOp>(operation).getExtents()[expression.getPayload()]
          : operation->getOperand(expression.getPayload());
      auto extent = indexValue(values.lookup(operand), operand.getType(), loc);
      if (failed(extent)) return failure();
      size = *extent;
    }
    dimensions[expression.getDimension()] = size;
    sizes.push_back(size);
  }
  if (inferred) {
    Value knownElements = constant(loc, 1);
    for (Value size : sizes)
      knownElements = builder.createOrFold<arith::MulIOp>(loc, knownElements, size);
    if (matchPattern(knownElements, m_Zero()))
      return operation->emitError("CPU inferred reshape requires a uniquely determined extent");
    Value source = values.lookup(operation->getOperand(0));
    Value sourceElements = constant(loc, 1);
    for (int64_t axis = 0; axis < cast<ShapedType>(source.getType()).getRank(); ++axis)
      sourceElements = builder.createOrFold<arith::MulIOp>(loc, sourceElements,
          dimension(builder, loc, source, axis));
    // A legal inferred reshape has a nonzero known product and exact quotient.
    dimensions[*inferred] = builder.createOrFold<arith::DivSIOp>(loc, sourceElements, knownElements);
  }
  return success();
}

RankedTensorType Construction::tensorType(Type type) {
  auto tensor = cast<RankedTensorType>(type);
  return RankedTensorType::get(tensor.getShape(), tensor.getElementType());
}

Type Construction::valueType(Type type) {
  return isa<RankedTensorType>(type) ? Type(tensorType(type)) : type;
}

Value Construction::dimension(OpBuilder &b, Location loc, Value value, int64_t axis) {
  return isa<RankedTensorType>(value.getType())
      ? Value(b.createOrFold<tensor::DimOp>(loc, value, axis))
      : Value(b.createOrFold<memref::DimOp>(loc, value, axis));
}

Value Construction::extractElement(OpBuilder &b, Location loc, Value value,
                            ValueRange indices) {
  return isa<RankedTensorType>(value.getType())
      ? Value(b.create<tensor::ExtractOp>(loc, value, indices))
      : Value(b.create<memref::LoadOp>(loc, value, indices));
}

Value Construction::emptyTensor(RankedTensorType tensor, ArrayRef<Value> sizes, Location loc) {
  SmallVector<Value> dynamic;
  for (auto [axis, size] : llvm::enumerate(sizes))
    if (tensor.isDynamicDim(axis))
      dynamic.push_back(size);
  return builder.create<tensor::EmptyOp>(loc, tensor.getShape(),
                                         tensor.getElementType(), dynamic);
}

FailureOr<SmallVector<Value>> Construction::emptyResults(TypeRange types, Location loc) {
  SmallVector<Type> leaves;
  appendProductLeafTypes(types, leaves);
  SmallVector<Value> result;
  for (Type leaf : leaves) {
    auto tensor = dyn_cast<RankedTensorType>(leaf);
    if (!tensor) tensor = RankedTensorType::get({}, leaf);
    auto sizes = extents(tensor, loc);
    if (failed(sizes)) return failure();
    result.push_back(emptyTensor(tensor, *sizes, loc));
  }
  return result;
}

SmallVector<Value> Construction::flattened(Value value) {
  if (getProductComponents(value.getType())) return products.at(value);
  return {values.lookup(value)};
}

SmallVector<Value> Construction::flattened(ValueRange inputs) {
  SmallVector<Value> result;
  for (Value value : inputs) llvm::append_range(result, flattened(value));
  return result;
}

void Construction::bindProduct(Value original, ValueRange components) {
  if (getProductComponents(original.getType())) products[original] = llvm::to_vector(components);
  else values.map(original, components.front());
}

void Construction::bindDimensions(Type original, Value value, Location loc) {
  auto tensor = dyn_cast<RankedTensorType>(original);
  if (!tensor || !isa<ShapedType>(value.getType())) return;
  auto identities = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions();
  for (auto [axis, identity] : llvm::enumerate(identities.asArrayRef()))
    dimensions[identity] = dimension(builder, loc, value, axis);
}

void Construction::bindValues(ValueRange originals, ValueRange components, Location loc) {
  auto ranges = getProductLeafRanges(originals.getTypes());
  for (auto [original, range] : llvm::zip(originals, ranges)) {
    SmallVector<Type> leaves;
    appendProductLeafTypes(original.getType(), leaves);
    SmallVector<Value> parts;
    for (auto [leaf, value] :
         llvm::zip(leaves, components.slice(range.offset, range.size))) {
      bindDimensions(leaf, value, loc);
      parts.push_back(value);
    }
    bindProduct(original, parts);
  }
}

ArrayAttr Construction::fieldPaths(TypeRange types) {
  SmallVector<Attribute> fields;
  for (auto [index, type] : llvm::enumerate(types))
    walkProductLeaves(type, [&](Type, ArrayRef<unsigned> path) {
      std::string name = std::to_string(index);
      if (!path.empty()) name += "." + getProductPathName(type, path);
      fields.push_back(builder.getStringAttr(name));
    });
  return builder.getArrayAttr(fields);
}

SmallVector<AffineMap> Construction::pointwiseMaps(ValueRange inputs, int64_t rank) {
  SmallVector<AffineMap> maps;
  for (Value input : inputs) {
    SmallVector<AffineExpr> axes;
    if (auto memory = dyn_cast<RankedTensorType>(input.getType()))
      for (int64_t axis = 0; axis < memory.getRank(); ++axis)
        axes.push_back(memory.getDimSize(axis) == 1
            ? builder.getAffineConstantExpr(0)
            : builder.getAffineDimExpr(rank - memory.getRank() + axis));
    maps.push_back(AffineMap::get(rank, 0, axes, builder.getContext()));
  }
  maps.push_back(builder.getMultiDimIdentityMap(rank));
  return maps;
}

Value Construction::elementAt(Value input, ValueRange members, OpBuilder &nested, Location loc) {
  auto type = dyn_cast<ShapedType>(input.getType());
  if (!type) return input;
  SmallVector<Value> indices;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    indices.push_back(type.getDimSize(axis) == 1
        ? Value(nested.create<arith::ConstantIndexOp>(loc, 0))
        : members[members.size() - type.getRank() + axis]);
  return extractElement(nested, loc, input, indices);
}

LogicalResult Construction::lower(DimOp op) {
  if (!dimensions.count(op.getDimension()))
    return op.emitError("CPU dimension has no runtime ABI binding");
  values.map(op.getResult(), dimensions.lookup(op.getDimension()));
  return success();
}

LogicalResult Construction::lower(DomainOp op) {
  Location loc = op.getLoc();
  Value begin = values.lookup(op.getBounds()[0]);
  Value end = values.lookup(op.getBounds()[1]);
  Value step = op.getBounds().size() == 3 ? values.lookup(op.getBounds()[2]) : constant(loc, 1);
  auto physicalBegin = indexValue(begin, op.getBounds()[0].getType(), loc);
  auto physicalEnd = indexValue(end, op.getBounds()[1].getType(), loc);
  auto physicalStep = indexValue(step, op.getBounds().size() == 3 ? op.getBounds()[2].getType() : builder.getIndexType(), loc);
  if (failed(physicalBegin) || failed(physicalEnd) || failed(physicalStep)) return failure();
  Value extent = domainExtent(*physicalBegin, *physicalEnd, *physicalStep, loc);
  Domain domain{*physicalBegin, *physicalEnd, *physicalStep, extent};
  domains[op.getResult()] = domain;
  dimensions[cast<IntegerAttr>(op.getExtentDimensions()[0]).getInt()] = domain.extent;
  return success();
}

LogicalResult Construction::lower(SubregionOp op) {
  Location loc = op.getLoc();
  auto source = domains.find(op.getInputs()[0]);
  if (source == domains.end()) return op.emitError("CPU subregion requires a realized source domain");
  Domain domain = source->second;
  unsigned position = 1;
  if (op.getHasStart()) {
    Value input = op.getInputs()[position++];
    auto start = indexValue(values.lookup(input), input.getType(), loc);
    if (failed(start)) return failure();
    domain.begin = *start;
  }
  if (op.getHasStop()) {
    Value input = op.getInputs()[position];
    auto stop = indexValue(values.lookup(input), input.getType(), loc);
    if (failed(stop)) return failure();
    domain.end = *stop;
  }
  domain.extent = domainExtent(domain.begin, domain.end, domain.step, loc);
  domains[op.getResult()] = domain;
  dimensions[cast<IntegerAttr>(op.getExtentDimensions()[0]).getInt()] = domain.extent;
  return success();
}

LogicalResult Construction::lower(RegionEndOp op) {
  auto source = domains.find(op.getSource());
  if (source == domains.end()) return op.emitError("CPU region end requires a realized source domain");
  values.map(op.getResult(), source->second.end);
  return success();
}

LogicalResult Construction::lower(MakeRecordOp op) {
  bindProduct(op.getResult(), flattened(op.getFields()));
  return success();
}

LogicalResult Construction::lower(MakeTupleOp op) {
  bindProduct(op.getResult(), flattened(op.getComponents()));
  return success();
}

LogicalResult Construction::lower(ExtractOp op) {
  auto range = getProductLeafRange(op.getProduct().getType(),
                                  {static_cast<unsigned>(op.getField())});
  if (failed(range)) return op.emitError("CPU extract has no canonical field schema");
  bindProduct(op.getResult(), ValueRange(products.at(op.getProduct()))
                                  .slice(range->offset, range->size));
  return success();
}

} // namespace intent::kir_to_cpu
