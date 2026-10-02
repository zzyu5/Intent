#include "Construction.h"

namespace intent::kir_to_dsa {

void Construction::bindHelperValue(Value formal, ArrayRef<Value> fields, bool rebase, int64_t sourceAxis) {
  SmallVector<Type> types; appendProductLeafTypes(formal.getType(), types);
  SmallVector<Value> bound;
  for (auto [type, field] : llvm::zip(types, fields)) {
    Value value = field;
    if (!isa<RankedTensorType>(type) && isa<MemRefType>(field.getType()))
      value = scalarCast(formal.getLoc(), loadLocal(formal.getLoc(), field, {}), type);
    bound.push_back(value);
  }
  bindProduct(formal, bound);
  for (auto [type, field] : llvm::zip(types, fields)) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !localShapes.count(field)) continue;
    auto shape = localShapes.lookup(field);
    auto ids = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions();
    for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
      LocalAxis binding = shape[axis];
      if (rebase && axis == sourceAxis) {
        binding.extent = binding.count;
        binding.begin = index(formal.getLoc(), 0);
        shape[axis] = binding;
      }
      dimensions[ids[axis]] = binding.extent;
      axisBindings[ids[axis]] = binding;
    }
    if (rebase) localShapes[field] = shape;
  }
}

FailureOr<SmallVector<Value>> Construction::helper(Block &block, ArrayRef<SmallVector<Value>> arguments,
                                    unsigned sourceCount, int64_t sourceAxis) {
  if (arguments.size() != block.getNumArguments()) return block.getParentOp()->emitError("DSA helper argument schema mismatch"), failure();
  auto savedValues = values; auto savedProducts = products;
  auto savedDimensions = dimensions; auto savedAxes = axisBindings;
  auto savedShapes = localShapes;
  auto savedStreamed = streamedOperations;
  for (auto [i, formal] : llvm::enumerate(block.getArguments()))
    bindHelperValue(formal, arguments[i], i < sourceCount, sourceAxis);
  LogicalResult status = lowerOperations(block);
  SmallVector<Value> result;
  if (succeeded(status)) result = flatten(block.getTerminator()->getOperands());
  values = std::move(savedValues); products = std::move(savedProducts);
  dimensions = std::move(savedDimensions); axisBindings = std::move(savedAxes);
  streamedOperations = std::move(savedStreamed);
  for (auto &entry : savedShapes) localShapes[entry.first] = entry.second;
  if (failed(status) || result.empty() || llvm::is_contained(result, Value())) return failure();
  return result;
}

SmallVector<SmallVector<Value>> Construction::splitFields(TypeRange types, ValueRange fields) {
  SmallVector<SmallVector<Value>> result;
  for (ProductLeafRange range : getProductLeafRanges(types))
    result.push_back(llvm::to_vector(fields.slice(range.offset, range.size)));
  return result;
}

LogicalResult Construction::streamReduction(ReduceOp reduce, unsigned axis, int64_t dimension, const LocalShape &shape) {
  Location loc = reduce.getLoc();
  ValueRange sources = reduce.getSources();
  ValueRange identities = reduce.getIdentities();
  ValueRange captures = reduce.getCaptures();
  SmallVector<Value> initial = flatten(identities), outputs;
  SmallVector<Type> elementTypes, resultTypes;
  for (Type type : identities.getTypes()) appendProductLeafTypes(type, elementTypes);
  for (Type type : reduce->getResultTypes()) appendProductLeafTypes(type, resultTypes);
  if (initial.size() != elementTypes.size() || initial.size() != resultTypes.size() ||
      llvm::any_of(elementTypes, [](Type type) { return isa<RankedTensorType>(type); }))
    return reduce.emitError("DSA streamed reduction requires scalar summary fields for each output position");
  LocalShape resultShape = shape;
  resultShape.erase(resultShape.begin() + axis);
  for (auto [value, type] : llvm::zip(initial, elementTypes)) {
    Value output = allocateTensor(loc, type, resultShape);
    Value identity = scalarCast(loc, value, cast<MemRefType>(output.getType()).getElementType());
    if (!identity) return reduce.emitError("DSA streamed reduction identity dtype is unavailable");
    b.create<dsa::FillOp>(loc, output, identity);
    outputs.push_back(output);
  }
  // Captures are immutable whole values evaluated before source slicing;
  // their storage and logical shape remain live across all chunks.
  SmallVector<SmallVector<Value>> captured;
  for (Value value : captures) captured.push_back(flatten(ValueRange{value}));
  auto savedValues = values; auto savedProducts = products; auto savedBindings = axisBindings;
  int64_t capacity = std::min(config.getRegionTile(), shape[axis].capacity);
  LogicalResult status = loop(loc, index(loc, 0), shape[axis].count, index(loc, capacity), [&](Value begin) -> LogicalResult {
    values = savedValues; products = savedProducts; axisBindings = savedBindings;
    Value count = b.create<arith::MinSIOp>(loc, sub(loc, shape[axis].count, begin), index(loc, capacity));
    axisBindings[dimension] = {shape[axis].extent, begin, count, capacity};
    DenseSet<Value> visited;
    for (Value source : sources) forgetReplayedTensors(source, visited);
    auto inputs = flatten(sources);
    if (inputs.size() != outputs.size() || llvm::is_contained(inputs, Value()))
      return reduce.emitError("DSA streamed reduction source fields are unavailable");
    // Keep the author's element-combination order while staging each source
    // chunk once for all output positions.
    return loop(loc, index(loc, 0), count, index(loc, 1), [&](Value member) {
      return eachElement(loc, resultShape, [&](ValueRange coordinates) -> LogicalResult {
        SmallVector<Value> sourceCoordinates(coordinates), old, elements;
        sourceCoordinates.insert(sourceCoordinates.begin() + axis, member);
        for (auto [output, input, type] : llvm::zip(outputs, inputs, elementTypes)) {
          old.push_back(scalarCast(loc, loadLocal(loc, output, coordinates), type));
          elements.push_back(scalarCast(loc, loadLocal(loc, input, sourceCoordinates), type));
        }
        auto arguments = splitFields(identities.getTypes(), old);
        llvm::append_range(arguments, splitFields(identities.getTypes(), elements));
        llvm::append_range(arguments, captured);
        auto updated = helper(reduce.getCombine().front(), arguments);
        if (failed(updated) || updated->size() != outputs.size()) return failure();
        for (auto [value, output] : llvm::zip(*updated, outputs)) storeLocal(loc, value, output, coordinates);
        return success();
      });
    });
  });
  values = std::move(savedValues); products = std::move(savedProducts); axisBindings = std::move(savedBindings);
  if (failed(status)) return failure();
  SmallVector<Value> results;
  for (auto [type, output] : llvm::zip(resultTypes, outputs))
    results.push_back(isa<RankedTensorType>(type) ? output : scalarCast(loc, loadLocal(loc, output, {}), type));
  auto grouped = splitFields(reduce->getResultTypes(), results);
  for (auto [result, fields] : llvm::zip(reduce->getResults(), grouped)) bindProduct(result, fields);
  return success();
}

FailureOr<SmallVector<Value>> Construction::mappedCombine(Block &block, ArrayRef<SmallVector<Value>> arguments) {
  DenseMap<Value, SmallVector<Value>> mapped;
  for (auto [formal, fields] : llvm::zip(block.getArguments(), arguments)) mapped[formal] = fields;
  auto lookup = [&](Value value) -> SmallVector<Value> {
    auto found = mapped.find(value);
    return found == mapped.end() ? flatten(ValueRange{value}) : found->second;
  };
  for (Operation &op : block.without_terminator()) {
    if (isa<MakeRecordOp, MakeTupleOp>(op)) {
      SmallVector<Value> fields;
      for (Value operand : op.getOperands()) llvm::append_range(fields, lookup(operand));
      mapped[op.getResult(0)] = std::move(fields);
      continue;
    }
    if (auto extract = dyn_cast<ExtractOp>(op)) {
      auto fields = lookup(extract.getProduct());
      auto range = getProductLeafRange(extract.getProduct().getType(),
          {static_cast<unsigned>(extract.getField())});
      if (failed(range) || range->offset + range->size > fields.size()) return failure();
      mapped[extract.getResult()] = llvm::to_vector(ArrayRef(fields).slice(range->offset, range->size));
      continue;
    }
    SmallVector<Value> operands;
    Value tile;
    for (Value operand : op.getOperands()) {
      auto fields = lookup(operand);
      if (fields.size() != 1 || !fields.front()) return failure();
      operands.push_back(fields.front());
      if (isa<MemRefType>(fields.front().getType())) tile = fields.front();
    }
    if (!tile) {
      auto result = scalarOperation(&op, operands);
      if (failed(result)) return failure();
      mapped[op.getResult(0)] = {*result};
      continue;
    }
    Location loc = op.getLoc();
    for (Value &operand : operands) if (!isa<MemRefType>(operand.getType())) {
      Value broadcast = allocateLike(loc, tile, operand.getType());
      b.create<dsa::FillOp>(loc, broadcast, operand);
      operand = broadcast;
    }
    Value output = allocateLike(loc, tile, op.getResult(0).getType());
    if (auto binary = dyn_cast<BinaryOp>(op); binary && binary.getType().isF32())
      b.create<dsa::BinaryOp>(loc, operands[0], operands[1], output,
          binary.getOperatorKindAttr(), binary.getApproximateAttr(), binary.getFlushToZeroAttr(), Value());
    else if (auto cast = dyn_cast<CastOp>(op); cast && !cast.getRounding())
      b.create<dsa::CastOp>(loc, operands[0], output);
    else return op.emitError("DSA mapped product combine has no typed tile implementation"), failure();
    mapped[op.getResult(0)] = {output};
  }
  SmallVector<Value> result;
  for (Value value : block.getTerminator()->getOperands()) llvm::append_range(result, lookup(value));
  return result;
}

std::optional<LogicalResult> Construction::productReduction(ReduceOp reduce) {
  auto sources = reduce.getSources();
  auto identities = reduce.getIdentities();
  auto captures = reduce.getCaptures();
  SmallVector<Value> sourceFields;
  std::function<bool(Value)> collect = [&](Value source) {
    while (auto extract = source.getDefiningOp<ExtractOp>()) {
      Operation *aggregate = extract.getProduct().getDefiningOp();
      if (!isa_and_nonnull<MakeRecordOp, MakeTupleOp>(aggregate)) break;
      source = aggregate->getOperand(extract.getField());
    }
    if (!getProductComponents(source.getType())) { sourceFields.push_back(source); return true; }
    Operation *definition = source.getDefiningOp();
    if (!isa_and_nonnull<MakeRecordOp, MakeTupleOp>(definition)) return false;
    return llvm::all_of(definition->getOperands(), collect);
  };
  if (!llvm::all_of(sources, collect) || sourceFields.size() < 2 ||
      reduce.getAxes().size() != 2 || cast<IntegerAttr>(reduce.getAxes()[0]).getInt() != 0 ||
      cast<IntegerAttr>(reduce.getAxes()[1]).getInt() != 1) return std::nullopt;
  auto first = dyn_cast<RankedTensorType>(sourceFields.front().getType());
  if (!first || first.getRank() != 2) return std::nullopt;
  auto shape = localShape(sourceFields.front(), reduce.getLoc());
  if (failed(shape)) return failure();
  auto ids = cast<TensorShapeAttr>(first.getEncoding()).getDimensions();
  if (!completeShape(*shape) || (*shape)[0].capacity <= 0 || (*shape)[1].capacity <= 0) return std::nullopt;
  int64_t width = 1;
  while (width * 2 <= std::min(config.getRegionTile(), (*shape)[1].capacity)) width *= 2;
  if (width < 2 || (*shape)[1].capacity % width) return std::nullopt;
  SmallVector<Attribute> uniform;
  UniformValueAnalysis uniformValues(describeCanonicalUniformValue);
  for (Value field : sourceFields) {
    auto type = dyn_cast<RankedTensorType>(field.getType());
    if (!type || type.getRank() != 2 ||
        cast<TensorShapeAttr>(type.getEncoding()).getDimensions() != ids ||
        !equalAxisExtent(first, 0, type, 0) || !equalAxisExtent(first, 1, type, 1)) return std::nullopt;
    auto constant = uniformValues.evaluate(field);
    if (!type.getElementType().isF32() && !constant) return std::nullopt;
    uniform.push_back(constant);
    for (int64_t dimension : ids.asArrayRef()) {
      DenseSet<Value> visited;
      if (!replayableSlice(field, dimension, visited)) return std::nullopt;
    }
  }
  for (Operation &op : reduce.getCombine().front().without_terminator())
    if (!isa<ConstantOp, BinaryOp, CastOp, ExtractOp, MakeRecordOp, MakeTupleOp>(op)) return std::nullopt;
  Location loc = reduce.getLoc();
  auto initial = flatten(identities);
  if (initial.size() != sourceFields.size()) return failure();
  SmallVector<SmallVector<Value>> captured;
  for (Value capture : captures) captured.push_back(flatten(ValueRange{capture}));
  auto combine = [&](ValueRange left, ValueRange right) -> FailureOr<SmallVector<Value>> {
    auto arguments = splitFields(identities.getTypes(), left);
    llvm::append_range(arguments, splitFields(identities.getTypes(), right));
    llvm::append_range(arguments, captured);
    return mappedCombine(reduce.getCombine().front(), arguments);
  };
  auto savedValues = values;
  auto savedProducts = products;
  auto savedBindings = axisBindings;
  auto load = [&](Value linear) -> FailureOr<SmallVector<Value>> {
    values = savedValues; products = savedProducts; axisBindings = savedBindings;
    Value spatial = index(loc, (*shape)[1].capacity);
    Value batch = b.create<arith::DivSIOp>(loc, linear, spatial);
    Value column = b.create<arith::RemSIOp>(loc, linear, spatial);
    axisBindings[ids[0]] = {(*shape)[0].extent, batch, index(loc, 1), 1};
    axisBindings[ids[1]] = {(*shape)[1].extent, column, index(loc, width), width};
    DenseSet<Value> visited;
    for (Value field : sourceFields) forgetReplayedTensors(field, visited);
    SmallVector<Value> result;
    for (auto [field, constant] : llvm::zip(sourceFields, uniform)) {
      Value value;
      if (constant && !cast<RankedTensorType>(field.getType()).getElementType().isF32())
        value = b.create<arith::ConstantOp>(loc, cast<TypedAttr>(constant));
      else value = get(field);
      if (!value) return failure();
      result.push_back(value);
    }
    return result;
  };
  auto horizontal = [&](SmallVector<Value> partial) -> FailureOr<SmallVector<Value>> {
    for (int64_t count = width; count > 1; count /= 2) {
      SmallVector<Value> left, right;
      for (Value value : partial) {
        if (!isa<MemRefType>(value.getType())) { left.push_back(value); right.push_back(value); continue; }
        auto element = cast<MemRefType>(value.getType()).getElementType();
        Value a = allocate(loc, element, 1, count / 2), c = allocate(loc, element, 1, count / 2);
        b.create<dsa::LoadTileOp>(loc, value, a, index(loc, 0), index(loc, 0), index(loc, 1), index(loc, 1), index(loc, count / 2));
        b.create<dsa::LoadTileOp>(loc, value, c, index(loc, count / 2), index(loc, 0), index(loc, 1), index(loc, 1), index(loc, count / 2));
        left.push_back(a); right.push_back(c);
      }
      auto next = combine(left, right);
      if (failed(next)) return failure();
      partial = *next;
    }
    for (Value &value : partial) if (isa<MemRefType>(value.getType())) value = loadLocal(loc, value, {});
    return partial;
  };
  Value total = index(loc, (*shape)[0].capacity * (*shape)[1].capacity);
  SmallVector<Value> slots;
  SmallVector<bool> shaped;
  auto store = [&](ValueRange fields) -> LogicalResult {
    for (auto [value, slot] : llvm::zip(fields, slots)) if (failed(copyTo(value, slot, loc))) return failure();
    return success();
  };
  auto state = [&]() {
    SmallVector<Value> result;
    for (auto [slot, tile] : llvm::zip(slots, shaped)) result.push_back(tile ? slot : loadLocal(loc, slot, {}));
    return result;
  };
  auto seeds = load(index(loc, 0));
  if (failed(seeds)) return failure();
  for (Value value : *seeds) {
    bool tile = isa<MemRefType>(value.getType());
    shaped.push_back(tile);
    slots.push_back(tile ? allocateLike(loc, value) : allocate(loc, value.getType(), 1, 1));
  }
  if (failed(store(*seeds))) return failure();
  if (failed(loop(loc, index(loc, width), total, index(loc, width), [&](Value position) -> LogicalResult {
    auto part = load(position);
    if (failed(part)) return failure();
    auto updated = combine(state(), *part);
    if (failed(updated)) return failure();
    return store(*updated);
  }))) return failure();
  auto partial = horizontal(state());
  if (failed(partial)) return failure();
  auto combined = combine(initial, *partial);
  if (failed(combined)) return failure();
  values = std::move(savedValues); products = std::move(savedProducts); axisBindings = std::move(savedBindings);
  auto grouped = splitFields(reduce->getResultTypes(), *combined);
  for (auto [output, fields] : llvm::zip(reduce->getResults(), grouped)) bindProduct(output, fields);
  return success();
}

LogicalResult Construction::reduceTensor(ReduceOp reduce) {
  Location loc = reduce.getLoc();
  if (auto product = productReduction(reduce)) return *product;
  if (reduce.getAxes().size() != 1) return reduce.emitError("DSA local reduction currently requires one selected axis");
  unsigned axis = cast<IntegerAttr>(reduce.getAxes()[0]).getInt();
  ValueRange sources = reduce.getSources();
  ValueRange identities = reduce.getIdentities();
  ValueRange captures = reduce.getCaptures();
  SmallVector<Type> sourceTypes;
  for (Type type : sources.getTypes()) appendProductLeafTypes(type, sourceTypes);
  if (!sourceTypes.empty() && valueSlices.empty()) {
    auto first = dyn_cast<RankedTensorType>(sourceTypes.front());
    if (first && axis < first.getRank()) {
      auto shape = localShape(sources.front(), loc);
      if (failed(shape)) return failure();
      int64_t elements = 1;
      for (const auto &local : *shape) elements *= local.capacity;
      int64_t bytes = std::max<int64_t>(1, (storageElement(first.getElementType()).getIntOrFloatBitWidth() + 7) / 8);
      int64_t dimension = cast<TensorShapeAttr>(first.getEncoding()).getDimensions()[axis];
      bool stream = (*shape)[axis].capacity > config.getRegionTile() &&
          elements > config.getLocalBytes() / bytes / 2 && completeShape(*shape);
      for (Type field : sourceTypes) {
        auto type = dyn_cast<RankedTensorType>(field);
        if (!type || type.getRank() != first.getRank()) { stream = false; break; }
        auto ids = cast<TensorShapeAttr>(type.getEncoding()).getDimensions();
        stream &= ids[axis] == dimension && llvm::count(ids.asArrayRef(), dimension) == 1;
        for (unsigned a = 0; a < type.getRank(); ++a) stream &= equalAxisExtent(first, a, type, a);
      }
      DenseSet<Value> visited;
      for (Value source : sources) stream &= replayableSlice(source, dimension, visited);
      if (stream) return streamReduction(reduce, axis, dimension, *shape);
    }
  }
  auto inputs = flatten(sources), initial = flatten(identities);
  if (inputs.empty() || inputs.size() != initial.size()) return reduce.emitError("DSA reduction has unbound source fields");
  for (Value input : inputs)
    if (!input || !localShapes.count(input) || axis >= localShapes.lookup(input).size())
      return reduce.emitError("DSA reduction source has no bounded tensor shape");
  LocalShape inputShape = localShapes.lookup(inputs.front());
  for (Value input : llvm::drop_begin(inputs)) {
    LocalShape other = localShapes.lookup(input);
    if (other.size() != inputShape.size())
      return reduce.emitError("DSA jointly reduced source fields require matching local ranks");
    for (unsigned a = 0; a < other.size(); ++a)
      if (other[a].capacity != inputShape[a].capacity || !sameIndex(other[a].count, inputShape[a].count))
        return reduce.emitError("DSA jointly reduced source fields require matching local extents");
  }
  LocalShape resultShape = inputShape;
  resultShape.erase(resultShape.begin() + axis);
  SmallVector<Type> resultTypes;
  for (Type type : reduce->getResultTypes()) appendProductLeafTypes(type, resultTypes);
  if (resultTypes.size() != initial.size()) return reduce.emitError("DSA reduction result schema mismatch");
  SmallVector<Value> outputs;
  for (Type type : resultTypes) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    outputs.push_back(allocateTensor(loc, tensor ? tensor.getElementType() : type, resultShape));
  }
  auto combineOps = reduce.getCombine().front().without_terminator();
  auto binary = llvm::hasSingleElement(combineOps) ? dyn_cast<BinaryOp>(&*combineOps.begin()) : BinaryOp();
  bool simple = inputs.size() == 1 && reduce.getIdentities().size() == 1 && captures.empty() && binary &&
      !binary.getApproximate() && !binary.getFlushToZero() &&
      binary.getLhs() == cast<StructuredOpInterface>(reduce.getOperation()).getCombineLhs().front() &&
      binary.getRhs() == cast<StructuredOpInterface>(reduce.getOperation()).getCombineRhs().front() &&
      reduce.getCombine().front().getTerminator()->getOperand(0) == binary.getResult();
  Type inputElement = cast<MemRefType>(inputs.front().getType()).getElementType();
  BinaryOperator kind = binary ? binary.getOperatorKind() : BinaryOperator::Add;
  bool boolean = inputElement.isInteger(1) && (kind == BinaryOperator::LogicalOr || kind == BinaryOperator::LogicalAnd);
  bool numeric = inputElement.isF32() && (kind == BinaryOperator::Add || kind == BinaryOperator::Maximum ||
      kind == BinaryOperator::Minimum || kind == BinaryOperator::MaximumNum || kind == BinaryOperator::MinimumNum);
  bool boundedRows = inputShape.size() == 2 && axis == 1 && inputShape[0].capacity > 1 &&
      inputShape[1].capacity < 65536;
  bool booleanRows = boundedRows && boolean && initial.front().getType().isInteger(1);
  bool floatSum = inputElement.isF32() && kind == BinaryOperator::Add &&
      initial.front().getType().isF32() && cast<MemRefType>(outputs.front().getType()).getElementType().isF32();
  bool floatRows = boundedRows && numeric && initial.front().getType().isF32() &&
      cast<MemRefType>(outputs.front().getType()).getElementType().isF32() &&
      (kind == BinaryOperator::Add ||
       inputShape[0].capacity * (inputShape[1].capacity + 1) <= config.getLocalBytes() / 16);
  if (simple && inputShape.size() == 2 && ((floatSum && axis == 0) || floatRows || booleanRows)) {
    // Retain the independent rows or columns so the target can select a complete
    // local reduction implementation rather than reconstruct scalar loops.
    Value input = inputs.front(), identity = initial.front();
    if (booleanRows) {
      input = allocateTensor(loc, b.getF32Type(), inputShape);
      b.create<dsa::CastOp>(loc, inputs.front(), input);
      identity = b.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), b.getF32Type());
    }
    int64_t columns = inputShape[1 - axis].capacity;
    Value accumulator = allocate(loc, b.getF32Type(), 1, columns);
    Value scratch = floatRows && kind != BinaryOperator::Add
        ? allocate(loc, b.getF32Type(), 1, columns * (inputShape[1].capacity + 1))
        : allocateLike(loc, accumulator);
    b.create<dsa::ReduceOp>(loc, input, accumulator, scratch, inputShape[axis].count, identity,
        BinaryOperatorAttr::get(b.getContext(), booleanRows ? BinaryOperator::Add : kind), b.getI64IntegerAttr(axis));
    if (booleanRows) {
      // Every input is exactly 0 or 1. With fewer than 65536 members all
      // partial sums are exactly representable in f32, including the tail.
      Value count = b.create<arith::IndexCastOp>(loc, b.getI64Type(), inputShape[axis].count);
      Value threshold = kind == BinaryOperator::LogicalAnd
          ? scalarCast(loc, count, b.getF32Type()) : identity;
      if (failed(eachElement(loc, resultShape, [&](ValueRange coordinates) {
        Value sum = b.create<memref::LoadOp>(loc, accumulator, ValueRange{index(loc, 0), coordinates.front()});
        Value predicate = b.create<arith::CmpFOp>(loc, kind == BinaryOperator::LogicalAnd
            ? arith::CmpFPredicate::OEQ : arith::CmpFPredicate::UNE, sum, threshold);
        Value value = kind == BinaryOperator::LogicalAnd
            ? Value(b.create<arith::AndIOp>(loc, initial.front(), predicate))
            : Value(b.create<arith::OrIOp>(loc, initial.front(), predicate));
        storeLocal(loc, value, outputs.front(), coordinates);
        return success();
      }))) return failure();
    } else {
      b.create<dsa::LoadTileOp>(loc, accumulator, outputs.front(), index(loc, 0),
          index(loc, 0), index(loc, 1), index(loc, 1), inputShape[1 - axis].count);
    }
    bindProduct(reduce->getResult(0), ValueRange{outputs.front()});
    return success();
  }
  if (simple && axis + 1 == inputShape.size() && (boolean || numeric)) {
    Value input = inputs.front();
    if (boolean) {
      Value promoted = allocateTensor(loc, b.getF32Type(), inputShape);
      b.create<dsa::CastOp>(loc, input, promoted); input = promoted;
      kind = kind == BinaryOperator::LogicalOr ? BinaryOperator::Maximum : BinaryOperator::Minimum;
    }
    int64_t capacity = (inputShape[axis].capacity + 31) / 32 * 32;
    bool directRow = inputShape.size() == 1 && inputShape[axis].capacity == capacity;
    Value row = directRow ? input : allocate(loc, b.getF32Type(), 1, capacity);
    Value scratch = allocateLike(loc, row);
    Value reduced = allocate(loc, b.getF32Type(), 1, 1);
    Value identity = scalarCast(loc, initial.front(), b.getF32Type());
    if (failed(eachElement(loc, resultShape, [&](ValueRange coordinates) {
      SmallVector<Value> sourceCoordinates(coordinates);
      sourceCoordinates.push_back(index(loc, 0));
      auto physical = physicalCoordinates(loc, input, sourceCoordinates);
      Value offset = mul(loc, physical[0], index(loc, inputShape[axis].capacity));
      if (!directRow)
        b.create<dsa::LoadTileOp>(loc, input, row, offset, index(loc, 0), index(loc, 1), index(loc, 1), inputShape[axis].count);
      b.create<dsa::ReduceOp>(loc, row, reduced, scratch, inputShape[axis].count, identity,
          BinaryOperatorAttr::get(b.getContext(), kind));
      storeLocal(loc, loadLocal(loc, reduced, {}), outputs.front(), coordinates);
      return success();
    }))) return failure();
    Value output = outputs.front();
    if (!isa<RankedTensorType>(resultTypes.front())) output = scalarCast(loc, loadLocal(loc, output, {}), resultTypes.front());
    bindProduct(reduce->getResult(0), ValueRange{output});
    return success();
  }
  auto slots = makeSlots(identities.getTypes(), loc), next = makeSlots(identities.getTypes(), loc);
  if (failed(slots) || failed(next)) return failure();
  SmallVector<SmallVector<Value>> captureFields;
  for (Value capture : captures) captureFields.push_back(flatten(ValueRange{capture}));
  if (failed(eachElement(loc, resultShape, [&](ValueRange coordinates) -> LogicalResult {
    for (auto [value, slot] : llvm::zip(initial, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
    if (failed(loop(loc, index(loc, 0), inputShape[axis].count, index(loc, 1), [&](Value i) -> LogicalResult {
      SmallVector<Value> sourceCoordinates(coordinates);
      sourceCoordinates.insert(sourceCoordinates.begin() + axis, i);
      SmallVector<Value> old, elements;
      SmallVector<Type> scalarTypes;
      for (Type type : identities.getTypes()) appendProductLeafTypes(type, scalarTypes);
      for (auto [slot, type] : llvm::zip(*slots, scalarTypes)) old.push_back(scalarCast(loc, loadLocal(loc, slot, {}), type));
      for (auto [input, type] : llvm::zip(inputs, scalarTypes)) elements.push_back(scalarCast(loc, loadLocal(loc, input, sourceCoordinates), type));
      auto arguments = splitFields(identities.getTypes(), old);
      llvm::append_range(arguments, splitFields(identities.getTypes(), elements));
      llvm::append_range(arguments, captureFields);
      auto updated = helper(reduce.getCombine().front(), arguments);
      if (failed(updated) || updated->size() != slots->size()) return failure();
      for (auto [value, slot] : llvm::zip(*updated, *next)) if (failed(copyTo(value, slot, loc))) return failure();
      for (auto [value, slot] : llvm::zip(*next, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
      return success();
    }))) return failure();
    for (auto [slot, output] : llvm::zip(*slots, outputs)) storeLocal(loc, loadLocal(loc, slot, {}), output, coordinates);
    return success();
  }))) return failure();
  SmallVector<Value> results;
  for (auto [type, output] : llvm::zip(resultTypes, outputs))
    results.push_back(isa<RankedTensorType>(type) ? output : scalarCast(loc, loadLocal(loc, output, {}), type));
  auto grouped = splitFields(reduce->getResultTypes(), results);
  for (auto [result, fields] : llvm::zip(reduce->getResults(), grouped)) bindProduct(result, fields);
  return success();
}

LogicalResult Construction::advanceScan(ScanOp scan, ValueRange slots, ValueRange next,
                         ValueRange elements, ArrayRef<SmallVector<Value>> captures) {
  ValueRange identities = scan.getIdentities();
  auto arguments = splitFields(identities.getTypes(), slots);
  llvm::append_range(arguments, splitFields(identities.getTypes(), elements));
  llvm::append_range(arguments, captures);
  auto updated = helper(scan.getCombine().front(), arguments);
  if (failed(updated) || updated->size() != slots.size()) return failure();
  for (auto [value, slot] : llvm::zip(*updated, next))
    if (failed(copyTo(value, slot, scan.getLoc()))) return failure();
  for (auto [value, slot] : llvm::zip(next, slots))
    if (failed(copyTo(value, slot, scan.getLoc()))) return failure();
  return success();
}

LogicalResult Construction::scanTensor(ScanOp scan) {
  Location loc = scan.getLoc();
  unsigned axis = scan.getAxis();
  ValueRange sources = scan.getSources();
  ValueRange identities = scan.getIdentities();
  auto inputs = flatten(sources), initial = flatten(identities);
  SmallVector<Type> schema;
  for (Type type : identities.getTypes()) appendProductLeafTypes(type, schema);
  if (inputs.empty() || inputs.size() != initial.size() || inputs.size() != schema.size())
    return scan.emitError("DSA scan has unbound source or identity fields");
  for (Value input : inputs)
    if (!input || !localShapes.count(input) || axis >= localShapes.lookup(input).size())
      return scan.emitError("DSA scan source requires a bounded local shape");
  LocalShape firstShape = localShapes.lookup(inputs.front());
  bool elementwise = llvm::none_of(schema, [](Type type) { return isa<RankedTensorType>(type); });
  LocalShape independentShape;
  if (elementwise) {
    independentShape = firstShape;
    independentShape.erase(independentShape.begin() + axis);
  }
  for (auto [input, type] : llvm::zip(inputs, schema)) {
    LocalShape slice = localShapes.lookup(input);
    if (!sameIndex(slice[axis].count, firstShape[axis].count))
      return scan.emitError("DSA scan source fields require a common traversal extent");
    slice.erase(slice.begin() + axis);
    auto tensor = dyn_cast<RankedTensorType>(type);
    LocalShape stateShape;
    if (tensor) {
      auto shape = localShape(tensor, loc);
      if (failed(shape)) return failure();
      stateShape = *shape;
    }
    const auto &required = elementwise ? independentShape : stateShape;
    if (slice.size() != required.size()) return scan.emitError("DSA scan state must match a source slice");
    for (unsigned i = 0; i < slice.size(); ++i)
      if (slice[i].capacity != required[i].capacity || !sameIndex(slice[i].count, required[i].count))
        return scan.emitError("DSA scan state and source slices have different local extents");
  }
  auto slots = makeSlots(identities.getTypes(), loc), next = makeSlots(identities.getTypes(), loc);
  auto items = makeSlots(identities.getTypes(), loc);
  if (failed(slots) || failed(next) || failed(items)) return failure();
  SmallVector<Value> outputs;
  for (Value input : inputs)
    outputs.push_back(allocateTensor(loc, cast<MemRefType>(input.getType()).getElementType(), localShapes.lookup(input)));
  SmallVector<SmallVector<Value>> captures;
  for (Value capture : scan.getCaptures()) {
    auto fields = flatten(ValueRange{capture});
    if (fields.empty() || llvm::is_contained(fields, Value())) return failure();
    captures.push_back(std::move(fields));
  }
  if (failed(eachElement(loc, independentShape, [&](ValueRange independent) -> LogicalResult {
    for (auto [value, slot] : llvm::zip(initial, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
    return loop(loc, index(loc, 0), firstShape[axis].count, index(loc, 1), [&](Value position) -> LogicalResult {
      Value i = scan.getReverse() ? sub(loc, sub(loc, firstShape[axis].count, index(loc, 1)), position) : position;
      auto transfer = [&](ValueRange state, bool read) -> LogicalResult {
        for (auto [field, slot] : llvm::enumerate(state)) {
          LocalShape slice = elementwise ? LocalShape{} : localShapes.lookup(slot);
          if (failed(eachElement(loc, slice, [&](ValueRange coordinates) {
            SmallVector<Value> sourceCoordinates(elementwise ? independent : coordinates);
            sourceCoordinates.insert(sourceCoordinates.begin() + axis, i);
            if (read) storeLocal(loc, loadLocal(loc, inputs[field], sourceCoordinates), slot, coordinates);
            else storeLocal(loc, loadLocal(loc, slot, coordinates), outputs[field], sourceCoordinates);
            return success();
          }))) return failure();
        }
        return success();
      };
      if (!scan.getInclusive() && failed(transfer(*slots, false))) return failure();
      if (failed(transfer(*items, true)) || failed(advanceScan(scan, *slots, *next, *items, captures))) return failure();
      if (scan.getInclusive() && failed(transfer(*slots, false))) return failure();
      return success();
    });
  }))) return failure();
  auto grouped = splitFields(scan->getResultTypes(), outputs);
  for (auto [result, fields] : llvm::zip(scan->getResults(), grouped)) bindProduct(result, fields);
  return success();
}

bool Construction::canStreamScan(ScanOp scan, Block &block) {
  unsigned axis = scan.getAxis();
  auto sources = scan.getSources();
  auto identities = scan.getIdentities();
  auto first = cast<RankedTensorType>(sources.front().getType());
  int64_t dimension = cast<TensorShapeAttr>(first.getEncoding()).getDimensions()[axis];
  if (dimension <= 0 || axisBindings.count(dimension) ||
      llvm::none_of(identities, [](Value value) { return isa<RankedTensorType>(value.getType()); })) return false;
  for (auto [source, identity] : llvm::zip(sources, identities)) {
    auto type = cast<RankedTensorType>(source.getType());
    auto ids = cast<TensorShapeAttr>(type.getEncoding()).getDimensions();
    auto state = dyn_cast<RankedTensorType>(identity.getType());
    if (axis >= type.getRank() || ids[axis] != dimension || llvm::count(ids.asArrayRef(), dimension) != 1 ||
        type.getRank() != (state ? state.getRank() : 0) + 1) return false;
    DenseSet<Value> visited;
    if (!replayableSlice(source, dimension, visited)) return false;
  }
  // All observable consumers must keep the prefix position free. A pair of
  // prefix axes, a reduction over it, or a nonlocal lookup needs materialization.
  DenseSet<Operation *> dependent;
  SmallVector<Value> pending(scan->getResults());
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (user->getBlock() != &block || user == block.getTerminator()) return false;
      if (dependent.insert(user).second) llvm::append_range(pending, user->getResults());
    }
  }
  auto ids = [](Value value) -> ArrayRef<int64_t> {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type ? cast<TensorShapeAttr>(type.getEncoding()).getDimensions().asArrayRef() : ArrayRef<int64_t>();
  };
  bool after = false, hasOutput = false;
  for (Operation &op : block.without_terminator()) {
    if (&op == scan) { after = true; continue; }
    if (auto store = dyn_cast<ViewStoreOp>(op)) {
      if (!after || !dependent.contains(&op) || llvm::count(ids(store.getValue()), dimension) != 1)
        return false;
      hasOutput = true;
    } else if (auto load = dyn_cast<ViewLoadOp>(op)) {
      if (cast<ViewType>(load.getSource().getType()).getAccess() != 0) return false;
    } else if (!isMemoryEffectFree(&op) && !isa<AssumeInBoundsOp>(op)) return false;
    if (isa<ScanOp, RegionFoldOp, RegionScanOp, ForOp, WhileOp, IfOp, ParallelOp, ReshapeOp, JoinOp>(op)) return false;
    for (Type type : op.getResultTypes()) {
      SmallVector<Type> fields; appendProductLeafTypes(type, fields);
      for (Type field : fields) if (auto tensor = dyn_cast<RankedTensorType>(field))
        if (llvm::count(cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions().asArrayRef(), dimension) > 1) return false;
    }
    if (auto reduce = dyn_cast<ReduceOp>(op))
      for (Value source : reduce.getSources()) {
        SmallVector<Type> fields; appendProductLeafTypes(source.getType(), fields);
        for (Type field : fields) if (auto tensor = dyn_cast<RankedTensorType>(field))
          for (Attribute reduced : reduce.getAxes())
            if (cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions()[cast<IntegerAttr>(reduced).getInt()] == dimension) return false;
      }
    if (auto matrix = dyn_cast<ContractOp>(op)) {
      if (!matrix.getBatch().empty()) return false;
      for (Attribute entry : matrix.getReduce()) {
        auto pair = cast<ArrayAttr>(entry);
        if (ids(matrix.getLhs())[cast<IntegerAttr>(pair[0]).getInt()] == dimension ||
            ids(matrix.getRhs())[cast<IntegerAttr>(pair[1]).getInt()] == dimension) return false;
      }
    }
    if (auto gather = dyn_cast<GatherOp>(op)) {
      auto relation = analysis.indexRelation(gather);
      if (failed(relation)) return false;
      for (const auto &term : relation->terms)
        if (term.sourceAxis && ids(relation->source)[*term.sourceAxis] == dimension && term.kind != 0) return false;
    }
  }
  return hasOutput;
}

LogicalResult Construction::streamScan(ScanOp scan, Block &block) {
  Location loc = scan.getLoc();
  unsigned axis = scan.getAxis();
  auto sources = scan.getSources();
  auto identities = scan.getIdentities();
  auto first = cast<RankedTensorType>(sources.front().getType());
  int64_t dimension = cast<TensorShapeAttr>(first.getEncoding()).getDimensions()[axis];
  for (Operation &op : block.without_terminator()) {
    if (&op == scan) break;
    if (!canDefer(&op) && failed(lowerOperation(&op))) return failure();
  }
  Value size = extent(first, axis, loc);
  if (!size) return scan.emitError("DSA scan traversal extent is unavailable");
  auto initial = flatten(identities);
  auto slots = makeSlots(identities.getTypes(), loc), next = makeSlots(identities.getTypes(), loc);
  auto items = makeSlots(identities.getTypes(), loc);
  if (failed(slots) || failed(next) || failed(items) || initial.size() != slots->size()) return failure();
  for (auto [value, slot] : llvm::zip(initial, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
  SmallVector<SmallVector<Value>> captures;
  for (Value capture : scan.getCaptures()) {
    auto fields = flatten(ValueRange{capture});
    if (fields.empty() || llvm::is_contained(fields, Value())) return failure();
    captures.push_back(std::move(fields));
  }
  auto savedValues = values; auto savedProducts = products; auto savedAxes = axisBindings;
  auto savedDimensions = dimensions;
  LogicalResult status = loop(loc, index(loc, 0), size, index(loc, 1), [&](Value position) -> LogicalResult {
    Value i = scan.getReverse() ? sub(loc, sub(loc, size, index(loc, 1)), position) : position;
    LocalAxis selected{size, i, index(loc, 1), 1};
    axisBindings[dimension] = selected;
    DenseSet<Value> replayed;
    for (Value source : sources) forgetReplayedTensors(source, replayed);
    auto inputs = flatten(sources);
    if (inputs.size() != items->size() || llvm::is_contained(inputs, Value())) return failure();
    for (auto [input, item] : llvm::zip(inputs, *items)) {
      LocalShape slice = localShapes.lookup(item);
      if (failed(eachElement(loc, slice, [&](ValueRange coordinates) {
        SmallVector<Value> from(coordinates); from.insert(from.begin() + axis, index(loc, 0));
        storeLocal(loc, loadLocal(loc, input, from), item, coordinates);
        return success();
      }))) return failure();
    }
    auto emit = [&]() -> LogicalResult {
      auto savedShapes = localShapes;
      for (auto [result, slot] : llvm::zip(scan->getResults(), *slots)) {
        auto type = cast<RankedTensorType>(result.getType());
        auto shape = localShape(type, loc);
        if (failed(shape)) return failure();
        int64_t rows = 1;
        for (unsigned a = 0; a + 1 < shape->size(); ++a) rows *= (*shape)[a].capacity;
        Value output = slot;
        auto physical = cast<MemRefType>(slot.getType());
        if (physical.getDimSize(0) != rows || physical.getDimSize(1) != shape->back().capacity) {
          output = allocateTensor(loc, type.getElementType(), *shape);
          if (failed(eachElement(loc, localShapes.lookup(slot), [&](ValueRange coordinates) {
            SmallVector<Value> to(coordinates); to.insert(to.begin() + axis, index(loc, 0));
            storeLocal(loc, loadLocal(loc, slot, coordinates), output, to);
            return success();
          }))) return failure();
        } else localShapes[output] = *shape;
        values.map(result, output);
      }
      for (Operation *op = scan->getNextNode(); op && op != block.getTerminator(); op = op->getNextNode())
        if (!canDefer(op) && failed(lowerOperation(op))) return failure();
      for (auto &entry : savedShapes) localShapes[entry.first] = entry.second;
      return success();
    };
    if (!scan.getInclusive() && failed(emit())) return failure();
    if (failed(advanceScan(scan, *slots, *next, *items, captures))) return failure();
    return scan.getInclusive() ? emit() : success();
  });
  values = std::move(savedValues); products = std::move(savedProducts);
  axisBindings = std::move(savedAxes); dimensions = std::move(savedDimensions);
  return status;
}

} // namespace intent::kir_to_dsa
