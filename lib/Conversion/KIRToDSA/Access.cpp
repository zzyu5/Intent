#include "Construction.h"

namespace intent::kir_to_dsa {

FailureOr<AffineIndices> Construction::affineIndex(Value original) {
  Location loc = original.getLoc();
  auto type = dyn_cast<RankedTensorType>(original.getType());
  if (!type) {
    Value scalar = asIndex(get(original), loc);
    if (!scalar) return failure();
    return AffineIndices{scalar, {}};
  }
  if (!type.getElementType().isIndex() && !type.getElementType().isInteger(64)) return failure();
  if (auto indices = original.getDefiningOp<IndicesOp>()) {
    if (!bindDomain(indices.getSource())) return failure();
    Domain domain = domains.lookup(indices.getSource());
    return AffineIndices{domain.begin, {domain.step}};
  }
  Operation *op = original.getDefiningOp();
  if (!op) return failure();
  if (isa<BroadcastOp, FullOp>(op)) {
    auto source = affineIndex(op->getOperand(0));
    if (failed(source)) return failure();
    AffineIndices result{source->base, SmallVector<Value>(type.getRank(), index(loc, 0))};
    if (auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType())) {
      if (input.getRank() > type.getRank()) return failure();
      unsigned leading = type.getRank() - input.getRank();
      for (unsigned axis = 0; axis < input.getRank(); ++axis) {
        if (singletonAxis(op->getOperand(0), axis)) continue;
        if (!equalAxisExtent(op->getOperand(0), axis, original, leading + axis)) return failure();
        result.steps[leading + axis] = source->steps[axis];
      }
    }
    return result;
  }
  if (auto cast = dyn_cast<CastOp>(op)) {
    auto input = dyn_cast<RankedTensorType>(cast.getInput().getType());
    if (!input || (!input.getElementType().isIndex() && !input.getElementType().isInteger(64))) return failure();
    return affineIndex(cast.getInput());
  }
  if (auto transpose = dyn_cast<TransposeOp>(op)) {
    auto source = affineIndex(transpose.getInput());
    if (failed(source)) return failure();
    AffineIndices result{source->base, {}};
    for (Attribute axis : transpose.getPermutation()) result.steps.push_back(source->steps[cast<IntegerAttr>(axis).getInt()]);
    return result;
  }
  if (auto gather = dyn_cast<GatherOp>(op)) {
    if (gather.getValid() && !constantTrue(gather.getValid())) return failure();
    auto relation = analysis.indexRelation(gather);
    if (failed(relation)) return failure();
    auto source = affineIndex(relation->source);
    if (failed(source)) return failure();
    AffineIndices result{source->base, SmallVector<Value>(type.getRank(), index(loc, 0))};
    for (const auto &term : relation->terms) {
      if (term.kind == 1) continue;
      if (!term.sourceAxis) return failure();
      Value coefficient = source->steps[*term.sourceAxis];
      if (term.kind == 0) result.steps[term.resultAxes.front()] = coefficient;
      else if (term.kind == 4) {
        if (!bindDomain(term.operands.front())) return failure();
        Domain domain = domains.lookup(term.operands.front());
        result.base = add(loc, result.base, mul(loc, coefficient, domain.begin));
        result.steps[term.resultAxes.front()] = mul(loc, coefficient, domain.step);
      } else if (term.kind == 2) {
        int64_t literal = *term.staticValues.front();
        Value coordinate = index(loc, literal);
        if (literal < 0) {
          Value size = logicalExtent(relation->source, *term.sourceAxis, loc);
          if (!size) return failure();
          coordinate = add(loc, size, coordinate);
        }
        result.base = add(loc, result.base, mul(loc, coefficient, coordinate));
      } else return failure();
    }
    return result;
  }
  auto binary = dyn_cast<BinaryOp>(op);
  if (!binary) return failure();
  auto lhs = affineIndex(binary.getLhs()), rhs = affineIndex(binary.getRhs());
  if (failed(lhs) || failed(rhs)) return failure();
  if (lhs->steps.empty()) lhs->steps.assign(type.getRank(), index(loc, 0));
  if (rhs->steps.empty()) rhs->steps.assign(type.getRank(), index(loc, 0));
  if (lhs->steps.size() != type.getRank() || rhs->steps.size() != type.getRank()) return failure();
  AffineIndices result{Value(), SmallVector<Value>(type.getRank())};
  if (binary.getOperatorKind() == BinaryOperator::Add || binary.getOperatorKind() == BinaryOperator::Subtract) {
    bool plus = binary.getOperatorKind() == BinaryOperator::Add;
    result.base = plus ? add(loc, lhs->base, rhs->base) : sub(loc, lhs->base, rhs->base);
    for (unsigned axis = 0; axis < type.getRank(); ++axis)
      result.steps[axis] = plus ? add(loc, lhs->steps[axis], rhs->steps[axis]) : sub(loc, lhs->steps[axis], rhs->steps[axis]);
    return result;
  }
  if (binary.getOperatorKind() == BinaryOperator::Multiply) {
    auto uniform = [](const AffineIndices &map) { return llvm::all_of(map.steps, [](Value step) { return matchPattern(step, m_Zero()); }); };
    const AffineIndices *variable = nullptr; Value factor;
    if (uniform(*lhs)) { variable = &*rhs; factor = lhs->base; }
    else if (uniform(*rhs)) { variable = &*lhs; factor = rhs->base; }
    if (variable) {
      result.base = mul(loc, lhs->base, rhs->base);
      for (unsigned axis = 0; axis < type.getRank(); ++axis) result.steps[axis] = mul(loc, factor, variable->steps[axis]);
      return result;
    }
  }
  return failure();
}

LogicalResult Construction::tensorAccess(Operation *op) {
  auto relation = analysis.indexRelation(op);
  if (failed(relation)) return failure();
  Location loc = op->getLoc();
  bool store = isa<ViewStoreOp, ScatterUniqueOp>(op);
  bool external = isa<ViewType>(relation->source.getType());
  Value source = get(relation->source);
  if (!source) return op->emitError("DSA indexed source is unavailable");
  auto access = cast<IndexedAccessOpInterface>(op);
  Value original = store ? access.getStoredValue() : op->getResult(0);
  auto tensor = dyn_cast<RankedTensorType>(original.getType());
  LocalShape shape;
  if (tensor) {
    auto selected = localShape(original, loc);
    if (failed(selected)) return failure();
    shape = *selected;
  }
  Value data = store ? get(original) : tensor ? allocateTensor(loc, tensor.getElementType(), shape) : Value();
  if (store && !data) return op->emitError("DSA store data is unavailable");
  Value validity = access.getAccessValidity(), fallback = access.getAccessFill();
  bool allValid = !validity || constantTrue(validity);
  if (relation->resultDimensionIdentities.size() != shape.size())
    return op->emitError("DSA index relation and local result ranks disagree");
  // Inserting unit axes changes the logical shape without permuting elements.
  // Retain the independent snapshot and explicitly zero inactive padding.
  if (!external && !store && tensor && localShapes.count(source) &&
      allValid) {
    const auto &inputShape = localShapes.find(source)->second;
    auto inputType = cast<MemRefType>(source.getType()), outputType = cast<MemRefType>(data.getType());
    bool contiguous = inputType.getLayout().isIdentity() && inputType.getElementType() == outputType.getElementType() &&
        inputType.getNumElements() == outputType.getNumElements();
    bool complete = true;
    unsigned sourceAxis = 0;
    for (const auto &term : relation->terms) {
      if (term.kind != 0 && term.kind != 1) { contiguous = false; break; }
      const auto &target = shape[term.resultAxes.front()];
      if (term.kind == 1) {
        contiguous &= target.capacity == 1 && matchPattern(target.extent, m_One()) &&
            matchPattern(target.count, m_One()) && matchPattern(target.begin, m_Zero());
      } else {
        if (!term.sourceAxis || *term.sourceAxis != sourceAxis || sourceAxis >= inputShape.size()) {
          contiguous = false; break;
        }
        const auto &input = inputShape[sourceAxis++];
        complete &= sameIndex(input.count, index(loc, input.capacity));
        contiguous &= input.capacity == target.capacity &&
            sameIndex(input.begin, target.begin) && sameIndex(input.count, target.count) &&
            sameIndex(input.extent, target.extent);
      }
    }
    if (contiguous && sourceAxis == inputShape.size() && (complete || shape.size() <= 2)) {
      Value reshaped = contiguousView(loc, source, shape);
      if (complete) b.create<memref::CopyOp>(loc, reshaped, data);
      else b.create<dsa::LoadTileOp>(loc, reshaped, data, index(loc, 0), index(loc, outputType.getDimSize(1)), index(loc, 1),
          shape.size() == 2 ? shape[0].count : index(loc, 1), shape.empty() ? index(loc, 1) : shape.back().count);
      values.map(original, data);
      return success();
    }
  }
  Value valid = !allValid && validity ? get(validity) : Value();
  Value fill = !allValid && fallback ? get(fallback) : Value();
  if (!allValid && ((validity && !valid) || (fallback && !fill)))
    return op->emitError("DSA access predicate or fill is unavailable");
  auto sourceExtent = [&](unsigned axis) -> Value {
    if (!external) return localShapes.lookup(source)[axis].extent;
    auto buffer = cast<MemRefType>(source.getType());
    return buffer.isDynamicDim(axis) ? Value(b.create<memref::DimOp>(loc, source, axis)) : index(loc, buffer.getDimSize(axis));
  };
  auto staticCoordinate = [&](const IndexTermFact &term) {
    int64_t literal = *term.staticValues.front();
    return literal < 0 ? add(loc, sourceExtent(*term.sourceAxis), index(loc, literal)) : index(loc, literal);
  };

  // Rectangular external accesses preserve arbitrary view strides. Their
  // selected origins and tails are explicit before BANG C serialization.
  bool rectangle = external && allValid && (tensor || !store) && shape.size() <= 2;
  DenseMap<Value, AffineIndices> affine;
  for (const auto &term : relation->terms) {
    rectangle &= term.kind >= 0 && term.kind <= 4;
    if (term.kind == 3 && isa<RankedTensorType>(term.operands.front().getType())) {
      auto coordinate = affineIndex(term.operands.front());
      if (failed(coordinate)) rectangle = false;
      else affine[term.operands.front()] = *coordinate;
    }
  }
  if (rectangle) {
    Value offset = index(loc, 0);
    SmallVector<Value> strides(shape.size(), index(loc, 0));
    for (const auto &term : relation->terms) {
      if (term.kind == 1) continue;
      if (!term.sourceAxis) return op->emitError("DSA access has no source axis");
      Value step = stride(loc, source, *term.sourceAxis), coordinate;
      if (term.kind == 2) coordinate = staticCoordinate(term);
      else if (term.kind == 3) {
        if (auto found = affine.find(term.operands.front()); found != affine.end()) {
          coordinate = found->second.base;
          auto type = cast<RankedTensorType>(term.operands.front().getType());
          for (unsigned axis = 0; axis < type.getRank(); ++axis) {
            if (singletonAxis(term.operands.front(), axis)) continue;
            unsigned mapped = term.indexAxes[axis];
            Value coefficient = found->second.steps[axis];
            coordinate = add(loc, coordinate, mul(loc, shape[mapped].begin, coefficient));
            strides[mapped] = add(loc, strides[mapped], mul(loc, step, coefficient));
          }
        } else coordinate = asIndex(get(term.operands.front()), loc);
      }
      else {
        unsigned outputAxis = term.resultAxes.front();
        coordinate = shape[outputAxis].begin;
        if (term.kind == 4) {
          if (term.operands.size() != 1 || !bindDomain(term.operands[0])) return op->emitError("DSA access interval is unavailable");
          Domain domain = domains.lookup(term.operands[0]);
          coordinate = add(loc, domain.begin, mul(loc, coordinate, domain.step));
          step = mul(loc, step, domain.step);
        }
        strides[outputAxis] = add(loc, strides[outputAxis], step);
      }
      if (!coordinate) return op->emitError("DSA scalar address is unavailable");
      offset = add(loc, offset, mul(loc, coordinate, stride(loc, source, *term.sourceAxis)));
    }
    if (!tensor) {
      Value value = b.create<dsa::LoadScalarOp>(loc, cast<MemRefType>(source.getType()).getElementType(), source, offset);
      values.map(original, scalarCast(loc, value, original.getType()));
      return success();
    }
    Value rows = shape.size() == 2 ? shape[0].count : index(loc, 1);
    Value cols = shape.empty() ? index(loc, 1) : shape.back().count;
    Value rowStride = strides.size() == 2 ? strides[0] : index(loc, 0);
    Value colStride = strides.empty() ? index(loc, 0) : strides.back();
    if (store) b.create<dsa::StoreTileOp>(loc, data, source, offset, rowStride, colStride, rows, cols);
    else { b.create<dsa::LoadTileOp>(loc, source, data, offset, rowStride, colStride, rows, cols); values.map(original, data); }
    return success();
  }
  if (!store && !data) data = allocateTensor(loc, original.getType(), shape);
  auto readAt = [&](Value value, ValueRange coordinates) {
    return isa<MemRefType>(value.getType()) ? loadLocal(loc, value, coordinates) : value;
  };
  DenseMap<Value, Value> indexValues;
  for (const auto &term : relation->terms) if (term.kind == 3) {
    Value value = get(term.operands.front());
    if (!value) return op->emitError("DSA index operand is unavailable");
    indexValues[term.operands.front()] = value;
  }
  auto locate = [&](ValueRange coordinates, Value &offset, SmallVectorImpl<Value> &sourceCoordinates) -> LogicalResult {
      offset = index(loc, 0);
      for (const auto &term : relation->terms) {
        if (term.kind == 1) continue;
        if (!term.sourceAxis) return op->emitError("DSA indexing requires explicit source axes");
        Value coordinate;
        if (term.kind == 0 || term.kind == 4) {
          unsigned outputAxis = term.resultAxes.front();
          coordinate = add(loc, shape[outputAxis].begin, coordinates[outputAxis]);
          if (term.kind == 4) {
            if (!bindDomain(term.operands.front())) return op->emitError("DSA index interval is unavailable");
            Domain domain = domains.lookup(term.operands.front());
            coordinate = add(loc, domain.begin, mul(loc, coordinate, domain.step));
          }
        } else if (term.kind == 2) coordinate = staticCoordinate(term);
        else if (term.kind == 3 && term.operands.size() == 1) {
          Value value = indexValues.lookup(term.operands.front());
          if (!value) return op->emitError("DSA index operand is unavailable");
          if (auto indexType = dyn_cast<RankedTensorType>(term.operands.front().getType())) {
            SmallVector<Value> indexCoordinates;
            const auto &indexShape = localShapes.lookup(value);
            for (int64_t axis = 0; axis < indexType.getRank(); ++axis) {
              unsigned mapped = term.indexAxes[axis];
              bool singleton = indexType.getDimSize(axis) == 1 || matchPattern(indexShape[axis].extent, m_One());
              indexCoordinates.push_back(singleton ? index(loc, 0) : sub(loc, add(loc, shape[mapped].begin, coordinates[mapped]), indexShape[axis].begin));
            }
            coordinate = asIndex(loadLocal(loc, value, indexCoordinates), loc);
          } else coordinate = asIndex(value, loc);
        }
        if (!coordinate) return op->emitError("DSA access index form is not implemented");
        if (external) offset = add(loc, offset, mul(loc, coordinate, stride(loc, source, *term.sourceAxis)));
        else {
          if (!localShapes.count(source)) return op->emitError("DSA gather source has no local axis binding");
          sourceCoordinates.push_back(sub(loc, coordinate, localShapes.lookup(source)[*term.sourceAxis].begin));
        }
      }
      return success();
  };
  // Preserve irregular row addresses together with their regular trailing
  // intervals, allowing the target to coalesce rows during local supply.
  if (external && !store && allValid && tensor && shape.size() == 1 &&
      cast<MemRefType>(source.getType()).getElementType() == cast<MemRefType>(data.getType()).getElementType()) {
    int64_t capacity = shape[0].capacity;
    Value offsets = allocate(loc, b.getI64Type(), 1, capacity);
    if (failed(loop(loc, index(loc, 0), shape[0].count, index(loc, 1), [&](Value i) {
      Value offset;
      SmallVector<Value> ignored;
      if (failed(locate(ValueRange{i}, offset, ignored))) return failure();
      Value stored = b.create<arith::IndexCastOp>(loc, b.getI64Type(), offset);
      b.create<memref::StoreOp>(loc, stored, offsets, ValueRange{index(loc, 0), i});
      return success();
    }))) return failure();
    auto storage = cast<MemRefType>(data.getType());
    auto transposed = MemRefType::get({capacity, 1}, storage.getElementType(), MemRefLayoutAttrInterface{}, storage.getMemorySpace());
    Value rows = b.create<memref::ReinterpretCastOp>(loc, transposed, data, int64_t(0),
        ArrayRef<int64_t>{capacity, 1}, ArrayRef<int64_t>{1, 1});
    b.create<dsa::GatherRowsOp>(loc, source, offsets, rows, index(loc, 1), shape[0].count, index(loc, 1), Value(), false);
    values.map(original, data);
    return success();
  }
  bool rowTiles = external && !store && allValid && tensor && shape.size() == 2 &&
      cast<MemRefType>(source.getType()).getElementType() == cast<MemRefType>(data.getType()).getElementType();
  int columnTerm = -1;
  for (auto [i, term] : llvm::enumerate(relation->terms)) {
    rowTiles &= term.kind >= 0 && term.kind <= 4;
    rowTiles &= term.kind == 1 || bool(term.sourceAxis);
    if (term.kind == 4) rowTiles &= term.operands.size() == 1;
    if ((term.kind == 0 || term.kind == 4) && term.resultAxes.front() == 1) {
      rowTiles &= columnTerm < 0;
      columnTerm = i;
    }
    if (term.kind != 3) continue;
    auto indexType = dyn_cast<RankedTensorType>(term.operands.front().getType());
    if (!indexType) continue;
    auto local = localShapes.find(indexValues.lookup(term.operands.front()));
    if (local == localShapes.end()) { rowTiles = false; continue; }
    for (int64_t axis = 0; axis < indexType.getRank(); ++axis) {
      unsigned mapped = term.indexAxes[axis];
      if (mapped == 1 && indexType.getDimSize(axis) != 1 && !matchPattern(local->second[axis].extent, m_One()))
        rowTiles = false;
    }
  }
  if (rowTiles && columnTerm >= 0) {
    const auto &term = relation->terms[columnTerm];
    Value columnStride = stride(loc, source, *term.sourceAxis);
    if (term.kind == 4) {
      if (!bindDomain(term.operands.front())) return op->emitError("DSA row interval is unavailable");
      columnStride = mul(loc, columnStride, domains.lookup(term.operands.front()).step);
    }
    Value offsets = allocate(loc, b.getI64Type(), 1, shape[0].capacity);
    Value nonempty = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, shape[1].count, index(loc, 0));
    auto branch = b.create<scf::IfOp>(loc, nonempty, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(branch.thenBlock());
      if (failed(loop(loc, index(loc, 0), shape[0].count, index(loc, 1), [&](Value i) {
        Value offset;
        SmallVector<Value> ignored;
        if (failed(locate(ValueRange{i, index(loc, 0)}, offset, ignored))) return failure();
        Value stored = b.create<arith::IndexCastOp>(loc, b.getI64Type(), offset);
        b.create<memref::StoreOp>(loc, stored, offsets, ValueRange{index(loc, 0), i});
        return success();
      }))) return failure();
      b.create<dsa::GatherRowsOp>(loc, source, offsets, data, columnStride, shape[0].count, shape[1].count, Value(), false);
    }
    values.map(original, data); return success();
  }
  if (failed(eachElement(loc, shape, [&](ValueRange coordinates) -> LogicalResult {
    if (fill) storeLocal(loc, readAt(fill, coordinates), data, coordinates);
    auto access = [&]() -> LogicalResult {
      Value offset;
      SmallVector<Value> sourceCoordinates;
      if (failed(locate(coordinates, offset, sourceCoordinates))) return failure();
      if (store) {
        Value value = isa<MemRefType>(data.getType()) ? loadLocal(loc, data, coordinates) : data;
        b.create<dsa::StoreScalarOp>(loc, value, source, offset);
      } else {
        Value value = external ? Value(b.create<dsa::LoadScalarOp>(loc, cast<MemRefType>(source.getType()).getElementType(), source, offset))
                               : loadLocal(loc, source, sourceCoordinates);
        storeLocal(loc, value, data, coordinates);
      }
      return success();
    };
    if (!valid) return access();
    auto branch = b.create<scf::IfOp>(loc, readAt(valid, coordinates), false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(branch.thenBlock());
    return access();
  }))) return failure();
  if (!store) values.map(original, tensor ? data : scalarCast(loc, loadLocal(loc, data, {}), original.getType()));
  return success();
}

Value Construction::stride(Location loc, Value view, int64_t axis) {
  if (auto argument = dyn_cast<BlockArgument>(view); argument && argument.getOwner() == &function.front()) {
    auto interface = intent::getPublicInterface(function);
    auto parameter = intent::getPublicView(interface, argument.getArgNumber());
    if (parameter.getConstraints().getHasStrides())
      if (auto fixed = dyn_cast<IntegerAttr>(parameter.getConstraints().getStrides()[axis]))
        return index(loc, fixed.getInt());
  }
  return b.create<dsa::StrideOp>(loc, b.getIndexType(), view, b.getI64IntegerAttr(axis));
}

} // namespace intent::kir_to_dsa
