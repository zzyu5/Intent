#include "Intent/Conversion/KIRToDSA/KIRToDSA.h"
#include "Intent/Analysis/CanonicalKernel.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallSet.h"
#include <functional>
using namespace mlir;
namespace intent {
namespace {
struct Domain { Value begin, end, step; int64_t dimension; };
class Construction {
public:
  Construction(ModuleOp original, ModuleOp target, dsa::ConfigurationAttr configuration)
      : analysis(original), target(target), b(target.getContext()), config(configuration) {}

  LogicalResult lower(func::FuncOp source) {
    auto parameters = source->getAttrOfType<ArrayAttr>("intent.parameters");
    if (!parameters || parameters.size() != source.getNumArguments())
      return source.emitError("DSA construction requires canonical parameter metadata");
    SmallVector<Type> arguments;
    SmallVector<Attribute> interface;
    SmallVector<Value> sourceArguments;
    for (auto [argument, parameter] : llvm::zip(source.getArguments(), parameters)) {
      auto name = cast<ParameterAttr>(parameter).getName();
      if (isa<ConstexprType>(argument.getType()) && argument.use_empty()) continue;
      if (auto view = dyn_cast<ViewType>(argument.getType())) {
        auto tensor = cast<RankedTensorType>(view.getTensor());
        Type element = tensor.getElementType();
        if (!element.isF16() && !element.isF32() && !element.isInteger(32) && !element.isInteger(64) && !element.isInteger(1))
          return source.emitError("DSA construction does not implement this view storage type");
        auto shape = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
        if (!shape) return source.emitError("DSA view has no dimension identities");
        arguments.push_back(MemRefType::get(tensor.getShape(), tensor.getElementType()));
        interface.push_back(dsa::ViewArgumentAttr::get(b.getContext(), name, tensor.getElementType(),
            b.getDenseI64ArrayAttr(tensor.getShape()), shape.getDimensions(), view.getAccess(), view.getConstraints()));
      } else if (argument.getType().isF32() || argument.getType().isIndex() || argument.getType().isInteger(64) || argument.getType().isInteger(32) || argument.getType().isInteger(1)) {
        arguments.push_back(argument.getType());
        interface.push_back(dsa::ScalarArgumentAttr::get(b.getContext(), name, argument.getType()));
      } else return source.emitError("unsupported DSA scalar ABI type");
      sourceArguments.push_back(argument);
    }
    b.setInsertionPointToEnd(target.getBody());
    function = b.create<func::FuncOp>(source.getLoc(), source.getName(), b.getFunctionType(arguments, {}));
    function->setAttr("intent_dsa.interface", dsa::InterfaceAttr::get(b.getContext(), b.getArrayAttr(interface)));
    function->setAttr("intent_dsa.configuration", config);
    function.addEntryBlock();
    b.setInsertionPointToStart(&function.front());
    for (auto [old, value] : llvm::zip(sourceArguments, function.getArguments())) {
      values.map(old, value);
      if (auto view = dyn_cast<ViewType>(old.getType())) {
        auto tensor = cast<RankedTensorType>(view.getTensor());
        auto ids = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions();
        for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
          Value size = tensor.isDynamicDim(axis) ? Value(b.create<memref::DimOp>(source.getLoc(), value, axis))
                                                : index(source.getLoc(), tensor.getDimSize(axis));
          if (ids[axis] > 0) dimensions.try_emplace(ids[axis], size);
        }
      }
    }
    taskId = b.create<dsa::TaskIdOp>(source.getLoc(), b.getIndexType());
    taskCount = b.create<dsa::TaskCountOp>(source.getLoc(), b.getIndexType());
    SmallVector<ContractOp> matrices;
    source.walk([&](ContractOp matrix) { matrices.push_back(matrix); });
    if (!matrices.empty()) {
      if (matrices.size() != 1 || matrices.front()->getBlock() != &source.front())
        return source.emitError("DSA matrix construction currently requires one top-level MatMul");
      if (failed(lowerMatrix(source, matrices.front()))) return failure();
    } else {
      SmallVector<ParallelOp> roots;
      source.walk([&](ParallelOp op) { if (!op->getParentOfType<ParallelOp>()) roots.push_back(op); });
      if (!roots.empty()) {
        if (roots.size() != 1 || roots.front()->getBlock() != &source.front())
          return source.emitError("DSA block tasks currently require one outer parallel workset; multiple task phases need a target-wide join realization");
        auto worksets = analysis.logicalWorksets(source);
        if (failed(worksets)) return failure();
        for (Operation &op : source.front().without_terminator()) {
          if (isa<ViewStoreOp>(op)) return op.emitError("DSA outer writes need an explicit task owner outside the parallel workset");
          if (auto load = dyn_cast<ViewLoadOp>(op)) {
            auto relation = analysis.indexRelation(load);
            if (failed(relation)) return failure();
            if (cast<ViewType>(relation->source.getType()).getAccess() != 0)
              return load.emitError("DSA distributed workset cannot duplicate a mutable input snapshot");
          }
        }
        if (failed(lowerBlock(source.front()))) return failure();
      } else {
        auto owner = b.create<arith::CmpIOp>(source.getLoc(), arith::CmpIPredicate::eq, taskId, index(source.getLoc(), 0));
        auto single = b.create<scf::IfOp>(source.getLoc(), owner, false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(single.thenBlock());
        if (failed(lowerBlock(source.front()))) return failure();
      }
    }
    b.create<func::ReturnOp>(source.getLoc());
    function->setAttr("intent_dsa.full_extent_dimensions", b.getDenseI64ArrayAttr(fullExtents));
    return success();
  }

private:
  Value index(Location loc, int64_t n) { return b.create<arith::ConstantIndexOp>(loc, n); }
  Value add(Location loc, Value a, Value c) { return b.create<arith::AddIOp>(loc, a, c); }
  Value mul(Location loc, Value a, Value c) { return b.create<arith::MulIOp>(loc, a, c); }
  Value sub(Location loc, Value a, Value c) { return b.create<arith::SubIOp>(loc, a, c); }
  Value get(Value source) { return values.lookupOrNull(source); }
  Value asIndex(Value value, Location loc) {
    if (value && value.getType().isIndex()) return value;
    if (value && value.getType().isInteger(64)) return b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
    return {};
  }
  Value extent(RankedTensorType type, unsigned axis, Location loc) {
    if (!type.isDynamicDim(axis)) return index(loc, type.getDimSize(axis));
    auto shape = dyn_cast_or_null<TensorShapeAttr>(type.getEncoding());
    return shape ? dimensions.lookup(shape.getDimensions()[axis]) : Value();
  }
  Value allocate(Location loc, Type element, int64_t rows, int64_t columns, int64_t space = dsa::nramSpace) {
    auto type = MemRefType::get({rows, columns}, element, MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(space));
    auto allocation = b.create<memref::AllocaOp>(loc, type);
    allocation.setAlignment(128);
    return allocation;
  }
  Value allocateLike(Location loc, Value input, Type element = {}) {
    auto type = cast<MemRefType>(input.getType());
    return allocate(loc, element ? element : type.getElementType(), type.getDimSize(0), type.getDimSize(1));
  }
  LogicalResult loop(Location loc, Value begin, Value end, Value step,
                     const std::function<LogicalResult(Value)> &body) {
    auto op = b.create<scf::ForOp>(loc, begin, end, step);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(op.getBody());
    return body(op.getInductionVar());
  }
  Value stride(Location loc, Value view, int64_t axis) {
    return b.create<dsa::StrideOp>(loc, b.getIndexType(), view, b.getI64IntegerAttr(axis));
  }
  Value scalarCast(Location loc, Value value, Type type) {
    if (!value) return {};
    if (value.getType() == type) return value;
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
    if (isa<IntegerType>(value.getType()) && isa<FloatType>(type)) return b.create<arith::SIToFPOp>(loc, type, value);
    if (isa<FloatType>(value.getType()) && isa<IntegerType>(type)) return b.create<arith::FPToSIOp>(loc, type, value);
    return {};
  }
  Value tensorOperand(Value original, Value shape, Location loc) {
    Value value = get(original);
    if (!value) return {};
    if (isa<MemRefType>(value.getType())) return value;
    Value output = allocateLike(loc, shape);
    value = scalarCast(loc, value, cast<MemRefType>(output.getType()).getElementType());
    if (!value) return {};
    b.create<dsa::FillOp>(loc, output, value);
    return output;
  }
  ArrayAttr components(Type type) {
    if (auto record = dyn_cast<RecordType>(type)) return record.getFieldTypes();
    if (auto tuple = dyn_cast<intent::TupleType>(type)) return tuple.getComponentTypes();
    return {};
  }
  void leaves(Type type, SmallVectorImpl<Type> &types) {
    if (auto fields = components(type)) {
      for (Attribute field : fields) leaves(cast<TypeAttr>(field).getValue(), types);
    } else types.push_back(type);
  }
  SmallVector<Value> flatten(ValueRange inputs) {
    SmallVector<Value> result;
    for (Value input : inputs) {
      if (components(input.getType())) llvm::append_range(result, products.lookup(input));
      else result.push_back(get(input));
    }
    return result;
  }
  void bindProduct(Value original, ValueRange fields) {
    if (components(original.getType())) products[original] = llvm::to_vector(fields);
    else values.map(original, fields.front());
  }
  FailureOr<SmallVector<Value>> makeSlots(TypeRange types, Location loc) {
    SmallVector<Value> slots;
    for (Type type : types) {
      SmallVector<Type> flat;
      leaves(type, flat);
      for (Type field : flat) {
        if (auto tensor = dyn_cast<RankedTensorType>(field)) {
          if (tensor.getRank() == 1 && activeCount) slots.push_back(allocate(loc, tensor.getElementType(), 1, config.getTile()));
          else if (tensor.hasStaticShape() && tensor.getRank() <= 2) {
            int64_t rows = tensor.getRank() == 2 ? tensor.getDimSize(0) : 1;
            int64_t columns = tensor.getRank() ? tensor.getDimSize(tensor.getRank() - 1) : 1;
            slots.push_back(allocate(loc, tensor.getElementType(), rows, columns));
          } else return emitError(loc, "DSA state tensor requires a bound local shape"), failure();
        } else slots.push_back(allocate(loc, field.isIndex() || isa<LogicalIndexType>(field) ? b.getI64Type() : field, 1, 1));
      }
    }
    return slots;
  }
  LogicalResult copyTo(Value value, Value slot, Location loc) {
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
  void bindSlots(ValueRange originals, ValueRange slots, Location loc) {
    unsigned offset = 0;
    for (Value original : originals) {
      SmallVector<Type> flat;
      leaves(original.getType(), flat);
      SmallVector<Value> fields;
      for (Type type : flat) {
        Value slot = slots[offset++];
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
  FailureOr<Value> lowerResults(Block &block, ValueRange slots, bool condition = false) {
    if (failed(lowerOperations(block))) return failure();
    ValueRange yielded = block.getTerminator()->getOperands();
    Value predicate = condition ? get(yielded.front()) : Value();
    auto fields = flatten(condition ? yielded.drop_front() : yielded);
    if (fields.size() != slots.size()) return block.getParentOp()->emitError("DSA control state schema mismatch"), failure();
    for (auto [field, slot] : llvm::zip(fields, slots))
      if (failed(copyTo(field, slot, block.getParentOp()->getLoc()))) return failure();
    return predicate;
  }
  LogicalResult orderedControl(Operation *operation) {
    Location loc = operation->getLoc();
    auto savedDimensions = dimensions;
    auto slots = makeSlots(operation->getResultTypes(), loc);
    if (failed(slots)) return failure();
    if (auto conditional = dyn_cast<IfOp>(operation)) {
      auto target = b.create<scf::IfOp>(loc, get(conditional.getCondition()), true);
      for (auto [source, destination] : llvm::zip(operation->getRegions(), target->getRegions())) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&destination.front());
        if (failed(lowerResults(source.front(), *slots))) return failure();
        dimensions = savedDimensions;
      }
    } else {
      auto forLoop = dyn_cast<ForOp>(operation);
      auto initial = flatten(forLoop ? operation->getOperands().drop_front() : operation->getOperands());
      if (initial.size() != slots->size()) return operation->emitError("DSA initial and result state schemas differ");
      auto next = makeSlots(operation->getResultTypes(), loc);
      if (failed(next)) return failure();
      for (auto [value, slot] : llvm::zip(initial, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
      auto advance = [&]() -> LogicalResult {
        for (auto [value, slot] : llvm::zip(*next, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
        return success();
      };
      if (forLoop) {
        if (!domains.count(forLoop.getInputs().front())) return operation->emitError("DSA ordered for requires a bound interval");
        Domain domain = domains.lookup(forLoop.getInputs().front());
        if (failed(loop(loc, domain.begin, domain.end, domain.step, [&](Value iv) {
          Block &body = forLoop.getBody().front();
          values.map(body.getArgument(0), iv);
          bindSlots(body.getArguments().drop_front(), *slots, loc);
          if (failed(lowerResults(body, *next))) return failure();
          return advance();
        }))) return failure();
      } else {
        auto target = b.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
        target.getBefore().emplaceBlock(); target.getAfter().emplaceBlock();
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(&target.getBefore().front());
          Block &before = operation->getRegion(0).front();
          bindSlots(before.getArguments(), *slots, loc);
          auto predicate = lowerResults(before, *next, true);
          if (failed(predicate) || !*predicate || failed(advance())) return failure();
          b.create<scf::ConditionOp>(loc, *predicate, ValueRange{});
        }
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(&target.getAfter().front());
          Block &after = operation->getRegion(1).front();
          bindSlots(after.getArguments(), *slots, loc);
          if (failed(lowerResults(after, *next)) || failed(advance())) return failure();
          b.create<scf::YieldOp>(loc);
        }
      }
    }
    dimensions = std::move(savedDimensions);
    bindSlots(operation->getResults(), *slots, loc);
    return success();
  }
  LogicalResult reduceProduct(ReduceOp reduce) {
    Location loc = reduce.getLoc();
    if (!activeCount || reduce.getAxes().size() != 1 || cast<IntegerAttr>(reduce.getAxes()[0]).getInt() != 0)
      return reduce.emitError("DSA product reduction requires a bound row axis");
    ValueRange inputs = reduce.getInputs();
    ValueRange sources = inputs.take_front(reduce.getSourceCount());
    ValueRange identities = inputs.drop_front(reduce.getSourceCount()).take_front(reduce.getIdentityCount());
    ValueRange captures = inputs.drop_front(reduce.getSourceCount() + reduce.getIdentityCount());
    auto sourceFields = flatten(sources), initial = flatten(identities);
    auto slots = makeSlots(reduce->getResultTypes(), loc), next = makeSlots(reduce->getResultTypes(), loc);
    if (failed(slots) || failed(next)) return failure();
    if (sourceFields.size() != initial.size() || slots->size() != initial.size())
      return reduce.emitError("DSA reduction source and identity schemas differ");
    for (auto [source, value, slot] : llvm::zip(sourceFields, initial, *slots)) {
      auto tensor = dyn_cast_or_null<MemRefType>(source ? source.getType() : Type());
      if (!tensor || tensor.getDimSize(0) != 1 || tensor.getDimSize(1) != config.getTile())
        return reduce.emitError("DSA product reduction requires row-shaped source fields");
      if (failed(copyTo(value, slot, loc))) return failure();
    }
    if (failed(loop(loc, index(loc, 0), activeCount, index(loc, 1), [&](Value i) {
      Block &combine = reduce.getCombine().front();
      bindSlots(combine.getArguments().take_front(identities.size()), *slots, loc);
      unsigned offset = 0;
      for (Value argument : combine.getArguments().drop_front(identities.size()).take_front(sources.size())) {
        SmallVector<Type> types; leaves(argument.getType(), types);
        SmallVector<Value> fields;
        for (unsigned field = 0; field < types.size(); ++field)
          fields.push_back(b.create<memref::LoadOp>(loc, sourceFields[offset++], ValueRange{index(loc, 0), i}));
        bindProduct(argument, fields);
      }
      for (auto [capture, argument] : llvm::zip(captures, combine.getArguments().take_back(captures.size())))
        bindProduct(argument, flatten(ValueRange{capture}));
      if (failed(lowerResults(combine, *next))) return failure();
      for (auto [value, slot] : llvm::zip(*next, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
      return success();
    }))) return failure();
    bindSlots(reduce->getResults(), *slots, loc);
    return success();
  }
  LogicalResult scanRow(ScanOp scan) {
    Location loc = scan.getLoc();
    if (!activeCount || scan.getAxis() != 0) return scan.emitError("DSA scan requires a bound row axis");
    ValueRange inputs = scan.getInputs();
    ValueRange sources = inputs.take_front(scan.getSourceCount());
    ValueRange identities = inputs.drop_front(scan.getSourceCount()).take_front(scan.getIdentityCount());
    ValueRange captures = inputs.drop_front(scan.getSourceCount() + scan.getIdentityCount());
    auto sourceFields = flatten(sources), initial = flatten(identities);
    auto slots = makeSlots(identities.getTypes(), loc), next = makeSlots(identities.getTypes(), loc);
    if (failed(slots) || failed(next)) return failure();
    SmallVector<Value> outputs;
    if (sourceFields.size() != initial.size() || slots->size() != initial.size())
      return scan.emitError("DSA scan source and identity schemas differ");
    for (auto [source, value, slot] : llvm::zip(sourceFields, initial, *slots)) {
      auto tensor = dyn_cast_or_null<MemRefType>(source ? source.getType() : Type());
      if (!tensor || tensor.getDimSize(0) != 1 || tensor.getDimSize(1) != config.getTile())
        return scan.emitError("DSA scan requires row-shaped source fields");
      if (failed(copyTo(value, slot, loc))) return failure();
      outputs.push_back(allocateLike(loc, source));
    }
    if (failed(loop(loc, index(loc, 0), activeCount, index(loc, 1), [&](Value position) {
      Value i = scan.getReverse() ? sub(loc, sub(loc, activeCount, index(loc, 1)), position) : position;
      Block &combine = scan.getCombine().front();
      auto write = [&](ValueRange state) {
        for (auto [slot, output] : llvm::zip(state, outputs)) {
          Value value = b.create<memref::LoadOp>(loc, slot, ValueRange{index(loc, 0), index(loc, 0)});
          b.create<memref::StoreOp>(loc, value, output, ValueRange{index(loc, 0), i});
        }
      };
      if (!scan.getInclusive()) write(*slots);
      // Reverse scans traverse the reversed sequence while keeping the same
      // combine(accumulator, current) argument order.
      bindSlots(combine.getArguments().take_front(identities.size()), *slots, loc);
      unsigned offset = 0;
      for (Value argument : combine.getArguments().slice(identities.size(), sources.size())) {
        SmallVector<Type> types; leaves(argument.getType(), types);
        SmallVector<Value> fields;
        for (unsigned field = 0; field < types.size(); ++field)
          fields.push_back(b.create<memref::LoadOp>(loc, sourceFields[offset++], ValueRange{index(loc, 0), i}));
        bindProduct(argument, fields);
      }
      for (auto [capture, argument] : llvm::zip(captures, combine.getArguments().take_back(captures.size())))
        bindProduct(argument, flatten(ValueRange{capture}));
      if (failed(lowerResults(combine, *next))) return failure();
      for (auto [value, slot] : llvm::zip(*next, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
      if (scan.getInclusive()) write(*slots);
      return success();
    }))) return failure();
    unsigned offset = 0;
    for (Value result : scan->getResults()) {
      SmallVector<Type> types; leaves(result.getType(), types);
      if (offset + types.size() > outputs.size()) return scan.emitError("DSA scan result schema mismatch");
      bindProduct(result, ArrayRef<Value>(outputs).slice(offset, types.size())); offset += types.size();
    }
    if (offset != outputs.size()) return scan.emitError("DSA scan has unbound result fields");
    return success();
  }
  LogicalResult lowerBlock(Block &block) {
    if (!activeCount) {
      RankedTensorType tensor;
      bool full = false;
      for (Operation &op : block.without_terminator()) {
        full |= isa<ReduceOp, ScanOp>(op);
        for (Type type : op.getResultTypes()) {
          auto ranked = dyn_cast<RankedTensorType>(type);
          if (!ranked) continue;
          if (ranked.getRank() != 1) return op.emitError("DSA pointwise construction requires one tensor axis per work item");
          if (!tensor) tensor = ranked;
          else if (ranked.getShape() != tensor.getShape() || ranked.getEncoding() != tensor.getEncoding())
            return op.emitError("DSA row tensors must share an exact logical extent");
        }
      }
      if (tensor) {
        Value size = extent(tensor, 0, block.getParentOp()->getLoc());
        if (!size) return block.getParentOp()->emitError("DSA cannot resolve the row's logical extent");
        auto shape = cast<TensorShapeAttr>(tensor.getEncoding());
        int64_t identity = shape.getDimensions()[0];
        if (full && identity <= 0 && (tensor.isDynamicDim(0) || tensor.getDimSize(0) > config.getTile()))
          return block.getParentOp()->emitError("DSA row reduction exceeds its configured local extent");
        if (full && identity > 0 && !llvm::is_contained(fullExtents, identity)) fullExtents.push_back(identity);
        auto saved = values;
        auto body = [&](Value begin) {
          activeBegin = begin;
          activeCount = full ? size : Value(b.create<arith::MinSIOp>(block.getParentOp()->getLoc(),
              sub(block.getParentOp()->getLoc(), size, begin), index(block.getParentOp()->getLoc(), config.getTile())));
          activeDimension = identity;
          LogicalResult result = lowerOperations(block);
          activeBegin = {}; activeCount = {}; activeDimension = 0;
          values = saved;
          return result;
        };
        if (full) return body(index(block.getParentOp()->getLoc(), 0));
        return loop(block.getParentOp()->getLoc(), index(block.getParentOp()->getLoc(), 0), size,
                    index(block.getParentOp()->getLoc(), config.getTile()), body);
      }
    }
    return lowerOperations(block);
  }
  LogicalResult lowerOperations(Block &block) {
    for (Operation &op : block.without_terminator()) if (failed(lowerOperation(&op))) return failure();
    return success();
  }
  LogicalResult lowerAccess(Operation *operation, bool store) {
    auto fact = analysis.indexRelation(operation);
    if (failed(fact)) return failure();
    Value resource = get(fact->source);
    if (!resource || !isa<MemRefType>(resource.getType())) return operation->emitError("DSA access needs an external view");
    if (!store && cast<ViewLoadOp>(operation).getValidOperandIndex())
      return operation->emitError("DSA predicated source access is not implemented");
    Location loc = operation->getLoc();
    Value offset = index(loc, 0), columnStride = index(loc, 0);
    bool row = false;
    for (auto &term : fact->terms) {
      if (!term.sourceAxis) return operation->emitError("DSA access does not implement newaxis or ellipsis");
      Value coordinate;
      if (term.kind == 3 && term.operands.size() == 1) coordinate = asIndex(get(term.operands.front()), loc);
      else if (term.kind == 4 && term.operands.size() == 1 && domains.count(term.operands.front())) {
        Domain domain = domains.lookup(term.operands.front());
        if (!activeCount || row || domain.dimension != activeDimension || !matchPattern(domain.step, m_One()))
          return operation->emitError("DSA tile access requires the same unit-step row domain");
        coordinate = add(loc, domain.begin, activeBegin);
        columnStride = stride(loc, resource, *term.sourceAxis);
        row = true;
      }
      if (!coordinate) return operation->emitError("DSA access requires scalar coordinates and one explicit row domain");
      offset = add(loc, offset, mul(loc, coordinate, stride(loc, resource, *term.sourceAxis)));
    }
    if (store) {
      auto write = cast<ViewStoreOp>(operation);
      Value value = get(write.getInputs()[write.getValueOperandIndex()]);
      if (value && !row && !isa<MemRefType>(value.getType())) {
        b.create<dsa::StoreScalarOp>(loc, value, resource, offset);
        return success();
      }
      if (!value || !row || !isa<MemRefType>(value.getType()))
        return operation->emitError("DSA store requires one tensor row");
      b.create<dsa::StoreTileOp>(loc, value, resource, offset, index(loc, 0), columnStride,
                                index(loc, 1), activeCount);
    } else if (!row) {
      if (isa<RankedTensorType>(operation->getResult(0).getType()))
        return operation->emitError("DSA scalar source requires a scalar result");
      Value loaded = b.create<dsa::LoadScalarOp>(loc, operation->getResult(0).getType(), resource, offset);
      values.map(operation->getResult(0), loaded);
    } else {
      auto type = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
      if (!type || type.getRank() != 1) return operation->emitError("DSA tile load requires one tensor row");
      Value output = allocate(loc, type.getElementType(), 1, config.getTile());
      b.create<dsa::LoadTileOp>(loc, resource, output, offset, index(loc, 0), columnStride,
                               index(loc, 1), activeCount);
      values.map(operation->getResult(0), output);
    }
    return success();
  }
  LogicalResult lowerOperation(Operation *operation) {
    Location loc = operation->getLoc();
    if (isa<IfOp, ForOp, WhileOp>(operation)) return orderedControl(operation);
    if (auto scan = dyn_cast<ScanOp>(operation)) return scanRow(scan);
    if (isa<MakeTupleOp, MakeRecordOp>(operation)) {
      products[operation->getResult(0)] = flatten(operation->getOperands());
      return success();
    }
    if (auto extract = dyn_cast<ExtractOp>(operation)) {
      auto types = components(extract.getProduct().getType());
      unsigned offset = 0;
      for (unsigned i = 0; i < extract.getField(); ++i) {
        SmallVector<Type> fields; leaves(cast<TypeAttr>(types[i]).getValue(), fields); offset += fields.size();
      }
      SmallVector<Type> fields; leaves(extract.getResult().getType(), fields);
      auto values = products.lookup(extract.getProduct());
      if (offset + fields.size() > values.size()) return extract.emitError("DSA record has unavailable components");
      bindProduct(extract.getResult(), ArrayRef<Value>(values).slice(offset, fields.size()));
      return success();
    }
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
    if (auto dim = dyn_cast<DimOp>(operation)) {
      Value value = dimensions.lookup(dim.getDimension());
      if (!value && isa<ViewType>(dim.getSource().getType())) value = b.create<memref::DimOp>(loc, get(dim.getSource()), dim.getAxis());
      if (!value) return dim.emitError("DSA dimension has no runtime binding");
      values.map(dim.getResult(), value); return success();
    }
    if (auto domain = dyn_cast<DomainOp>(operation)) {
      if (domain.getBounds().size() < 2 || domain.getBounds().size() > 3 || domain.getExtentDimensions().size() != 1)
        return domain.emitError("DSA construction requires rank-one interval domains");
      Value begin = asIndex(get(domain.getBounds()[0]), loc), end = asIndex(get(domain.getBounds()[1]), loc);
      Value step = domain.getBounds().size() == 3 ? asIndex(get(domain.getBounds()[2]), loc) : index(loc, 1);
      if (!begin || !end || !step || !matchPattern(step, m_One())) return domain.emitError("DSA construction requires a unit-step interval");
      int64_t identity = cast<IntegerAttr>(domain.getExtentDimensions()[0]).getInt();
      domains[domain.getResult()] = {begin, end, step, identity};
      if (identity > 0) dimensions[identity] = sub(loc, end, begin);
      return success();
    }
    if (auto parallel = dyn_cast<ParallelOp>(operation)) {
      if (!domains.count(parallel.getSource()) || parallel.getBody().front().getNumArguments() != 1)
        return parallel.emitError("DSA parallel work requires a single interval");
      auto domain = domains.lookup(parallel.getSource());
      auto savedDimensions = dimensions;
      bool distribute = parallelDepth++ == 0;
      Value begin = distribute ? add(loc, domain.begin, mul(loc, taskId, domain.step)) : domain.begin;
      Value step = distribute ? mul(loc, taskCount, domain.step) : domain.step;
      LogicalResult result = loop(loc, begin, domain.end, step, [&](Value i) {
        values.map(parallel.getBody().front().getArgument(0), i);
        return lowerBlock(parallel.getBody().front());
      });
      --parallelDepth; dimensions = std::move(savedDimensions); return result;
    }
    if (isa<ViewLoadOp>(operation)) return lowerAccess(operation, false);
    if (isa<ViewStoreOp>(operation)) return lowerAccess(operation, true);
    if (isa<BroadcastOp, FullOp>(operation)) {
      Value input = get(operation->getOperand(0));
      if (!input) return operation->emitError("DSA broadcast input is unavailable");
      if (isa<MemRefType>(input.getType())) { values.map(operation->getResult(0), input); return success(); }
      auto type = cast<RankedTensorType>(operation->getResult(0).getType());
      Value output = allocate(loc, type.getElementType(), 1, config.getTile());
      input = scalarCast(loc, input, type.getElementType());
      if (!input) return operation->emitError("DSA broadcast requires a compatible float scalar");
      b.create<dsa::FillOp>(loc, output, input);
      values.map(operation->getResult(0), output); return success();
    }
    if (auto castOp = dyn_cast<CastOp>(operation)) {
      if (castOp.getRounding()) return castOp.emitError("DSA explicit rounding is not implemented");
      Value input = get(castOp.getInput());
      if (!input) return castOp.emitError("DSA cast input is unavailable");
      if (auto tensor = dyn_cast<RankedTensorType>(castOp.getResult().getType())) {
        Value output = allocateLike(loc, input, tensor.getElementType());
        b.create<dsa::CastOp>(loc, input, output); values.map(castOp.getResult(), output);
      } else {
        Value output = scalarCast(loc, input, castOp.getResult().getType());
        if (!output) return castOp.emitError("unsupported DSA scalar cast");
        values.map(castOp.getResult(), output);
      }
      return success();
    }
    if (auto binary = dyn_cast<BinaryOp>(operation)) {
      Value lhs = get(binary.getLhs()), rhs = get(binary.getRhs());
      if (!lhs || !rhs) return binary.emitError("DSA binary operand is unavailable");
      if (isa<RankedTensorType>(binary.getResult().getType())) {
        Value shape = isa<MemRefType>(lhs.getType()) ? lhs : rhs;
        lhs = tensorOperand(binary.getLhs(), shape, loc); rhs = tensorOperand(binary.getRhs(), shape, loc);
        if (!lhs || !rhs) return binary.emitError("DSA binary broadcast is not implemented");
        Value output = allocateLike(loc, lhs);
        b.create<dsa::BinaryOp>(loc, lhs, rhs, output, binary.getOperatorKindAttr(), binary.getApproximateAttr(), binary.getFlushToZeroAttr());
        values.map(binary.getResult(), output); return success();
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
      if (isa<RankedTensorType>(select.getResult().getType())) {
        Value shape = isa<MemRefType>(lhs.getType()) ? lhs : rhs;
        if (!isa<MemRefType>(shape.getType())) return select.emitError("DSA tensor selection requires a bound data tile");
        lhs = tensorOperand(select.getTrueValue(), shape, loc); rhs = tensorOperand(select.getFalseValue(), shape, loc);
        if (!lhs || !rhs) return select.emitError("DSA selection broadcast is unavailable");
        Value output = allocateLike(loc, shape);
        b.create<dsa::SelectOp>(loc, condition, lhs, rhs, output);
        values.map(select.getResult(), output);
      } else {
        if (!condition.getType().isInteger(1)) return select.emitError("DSA scalar selection needs a scalar predicate");
        values.map(select.getResult(), b.create<arith::SelectOp>(loc, condition, lhs, rhs));
      }
      return success();
    }
    if (auto unary = dyn_cast<UnaryOp>(operation)) {
      Value input = get(unary.getInput());
      if (!input) return unary.emitError("DSA unary operand is unavailable");
      if (!isa<MemRefType>(input.getType())) {
        if (unary.getApproximate() || unary.getFlushToZero()) return unary.emitError("DSA scalar numerical mode is not implemented");
        Value output;
        switch (unary.getOperatorKind()) {
        case UnaryOperator::Exp: output = b.create<math::ExpOp>(loc, input); break;
        case UnaryOperator::Exp2: output = b.create<math::Exp2Op>(loc, input); break;
        case UnaryOperator::Log: output = b.create<math::LogOp>(loc, input); break;
        case UnaryOperator::Sqrt: output = b.create<math::SqrtOp>(loc, input); break;
        case UnaryOperator::Rsqrt: output = b.create<math::RsqrtOp>(loc, input); break;
        case UnaryOperator::Tanh: output = b.create<math::TanhOp>(loc, input); break;
        case UnaryOperator::Abs: output = b.create<math::AbsFOp>(loc, input); break;
        case UnaryOperator::Negate: output = b.create<arith::NegFOp>(loc, input); break;
        default: return unary.emitError("DSA scalar unary operation is not implemented");
        }
        values.map(unary.getResult(), output); return success();
      }
      Value output = allocateLike(loc, input);
      b.create<dsa::UnaryOp>(loc, input, output, unary.getOperatorKindAttr(), unary.getApproximateAttr(), unary.getFlushToZeroAttr());
      values.map(unary.getResult(), output); return success();
    }
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      if (reduce.getSourceCount() != 1 || reduce.getIdentityCount() != 1 || reduce.getCaptureCount() ||
          reduce.getAxes().size() != 1 || cast<IntegerAttr>(reduce.getAxes()[0]).getInt() != 0 ||
          reduce->getNumResults() != 1 || !isa<FloatType>(reduce->getResult(0).getType()))
        return reduceProduct(reduce);
      auto operations = reduce.getCombine().front().without_terminator();
      if (!llvm::hasSingleElement(operations)) return reduceProduct(reduce);
      auto combine = dyn_cast<BinaryOp>(&*operations.begin());
      if (!combine || combine.getApproximate() || combine.getFlushToZero())
        return reduceProduct(reduce);
      auto kind = combine.getOperatorKind();
      if (kind != BinaryOperator::Add && kind != BinaryOperator::Maximum && kind != BinaryOperator::Minimum &&
          kind != BinaryOperator::MaximumNum && kind != BinaryOperator::MinimumNum)
        return reduceProduct(reduce);
      Value input = get(reduce.getInputs()[0]), identity = get(reduce.getInputs()[1]);
      if (!input || !identity || !activeCount) return reduce.emitError("DSA reduction input is unavailable");
      Value output = allocate(loc, reduce->getResult(0).getType(), 1, 1);
      Value scratch = allocateLike(loc, input);
      b.create<dsa::ReduceOp>(loc, input, output, scratch, activeCount, identity, combine.getOperatorKindAttr());
      Value scalar = b.create<memref::LoadOp>(loc, output, ValueRange{index(loc, 0), index(loc, 0)});
      values.map(reduce->getResult(0), scalar); return success();
    }
    if (isa<AssumeInBoundsOp>(operation)) return success();
    return operation->emitError("operation has no DSA construction implementation");
  }

  FailureOr<Value> matrixView(Value tensor) {
    auto load = tensor.getDefiningOp<ViewLoadOp>();
    if (!load || load.getValidOperandIndex()) return failure();
    auto fact = analysis.indexRelation(load);
    if (failed(fact) || fact->sourceRank != 2 || fact->terms.size() != 2) return failure();
    for (auto &term : fact->terms) {
      if (term.kind != 4 || term.operands.size() != 1) return failure();
      auto domain = term.operands[0].getDefiningOp<DomainOp>();
      if (!domain || domain.getBounds().size() != 2) return failure();
      auto begin = domain.getBounds()[0].getDefiningOp<ConstantOp>();
      auto end = domain.getBounds()[1].getDefiningOp<DimOp>();
      if (!begin || !cast<IntegerAttr>(begin.getValue()).getValue().isZero() || !end) return failure();
      auto sourceType = cast<RankedTensorType>(cast<ViewType>(fact->source.getType()).getTensor());
      auto sourceIds = cast<TensorShapeAttr>(sourceType.getEncoding()).getDimensions();
      if (!term.sourceAxis || sourceIds[*term.sourceAxis] != static_cast<int64_t>(end.getDimension())) return failure();
    }
    return get(fact->source);
  }
  LogicalResult lowerMatrix(func::FuncOp source, ContractOp matrix) {
    auto a = matrixView(matrix.getLhs()), c = matrixView(matrix.getRhs());
    auto pairs = matrix.getReduce();
    if (failed(a) || failed(c) || !matrix.getBatch().empty() || pairs.size() != 1)
      return matrix.emitError("DSA MatMul requires full rank-two views and one reduction pair");
    auto pair = dyn_cast<ArrayAttr>(pairs[0]);
    if (!pair || pair.size() != 2 || cast<IntegerAttr>(pair[0]).getInt() != 1 || cast<IntegerAttr>(pair[1]).getInt() != 0)
      return matrix.emitError("DSA MatMul currently requires A[M,K] and B[K,N]");
    if (!cast<RankedTensorType>(matrix.getResult().getType()).getElementType().isF32())
      return matrix.emitError("DSA MatMul requires f32 accumulation");
    SmallVector<Operation *> epilogue;
    bool after = false;
    for (Operation &op : source.front().without_terminator()) {
      if (&op == matrix) { after = true; continue; }
      if (after) epilogue.push_back(&op);
      else if (!isa<ConstantOp, DimOp, DomainOp, ViewLoadOp>(op))
        return op.emitError("DSA matrix inputs must be direct views");
    }
    ViewStoreOp output;
    for (Operation *op : epilogue) {
      if (auto store = dyn_cast<ViewStoreOp>(op)) {
        if (output) return store.emitError("DSA matrix epilogue currently requires one output");
        output = store;
      } else if (!isa<CastOp>(op)) return op->emitError("DSA matrix epilogue currently supports casts and a full output store");
    }
    if (!output) return matrix.emitError("DSA MatMul has no output store");
    auto outputFact = analysis.indexRelation(output);
    if (failed(outputFact) || outputFact->sourceRank != 2 || outputFact->terms.size() != 2)
      return output.emitError("DSA matrix output requires a rank-two view");
    auto outputType = cast<RankedTensorType>(cast<ViewType>(outputFact->source.getType()).getTensor());
    auto outputDimensions = cast<TensorShapeAttr>(outputType.getEncoding()).getDimensions();
    auto resultDimensions = cast<TensorShapeAttr>(cast<RankedTensorType>(matrix.getResult().getType()).getEncoding()).getDimensions();
    if (outputDimensions != resultDimensions) return output.emitError("DSA matrix output must preserve the MatMul logical axes");
    for (auto &term : outputFact->terms) {
      if (term.kind != 4 || term.operands.size() != 1 || !term.sourceAxis)
        return output.emitError("DSA matrix output requires full output domains");
      auto domain = term.operands[0].getDefiningOp<DomainOp>();
      if (!domain || domain.getBounds().size() != 2) return output.emitError("DSA matrix output requires full output intervals");
      auto begin = domain.getBounds()[0].getDefiningOp<ConstantOp>();
      auto end = domain.getBounds()[1].getDefiningOp<DimOp>();
      if (!begin || !cast<IntegerAttr>(begin.getValue()).getValue().isZero() || !end ||
          static_cast<int64_t>(end.getDimension()) != outputDimensions[*term.sourceAxis])
        return output.emitError("DSA matrix output interval does not cover its logical result axis");
    }
    Value destination = get(outputFact->source);
    Location loc = matrix.getLoc();
    Value M = b.create<memref::DimOp>(loc, *a, 0), K = b.create<memref::DimOp>(loc, *a, 1);
    Value N = b.create<memref::DimOp>(loc, *c, 1);
    Value tm = index(loc, config.getTileM()), tn = index(loc, config.getTileN()), tk = index(loc, config.getTileK());
    Value gridM = b.create<arith::CeilDivSIOp>(loc, M, tm), gridN = b.create<arith::CeilDivSIOp>(loc, N, tn);
    return loop(loc, taskId, mul(loc, gridM, gridN), taskCount, [&](Value task) -> LogicalResult {
      Value mi = b.create<arith::DivSIOp>(loc, task, gridN), ni = b.create<arith::RemSIOp>(loc, task, gridN);
      Value m0 = mul(loc, mi, tm), n0 = mul(loc, ni, tn);
      Value rows = b.create<arith::MinSIOp>(loc, sub(loc, M, m0), tm);
      Value cols = b.create<arith::MinSIOp>(loc, sub(loc, N, n0), tn);
      auto inputType = cast<MemRefType>((*a).getType()).getElementType();
      Value lhs = allocate(loc, inputType, config.getTileM(), config.getTileK());
      Value rhs = allocate(loc, inputType, config.getTileK(), config.getTileN());
      Value accumulator = allocate(loc, b.getF32Type(), config.getTileM(), config.getTileN());
      Value zero = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0));
      b.create<dsa::FillOp>(loc, accumulator, zero);
      if (failed(loop(loc, index(loc, 0), K, tk, [&](Value k0) {
        Value depth = b.create<arith::MinSIOp>(loc, sub(loc, K, k0), tk);
        Value ar = stride(loc, *a, 0), ac = stride(loc, *a, 1);
        Value br = stride(loc, *c, 0), bc = stride(loc, *c, 1);
        b.create<dsa::LoadTileOp>(loc, *a, lhs, add(loc, mul(loc, m0, ar), mul(loc, k0, ac)), ar, ac, rows, depth);
        b.create<dsa::LoadTileOp>(loc, *c, rhs, add(loc, mul(loc, k0, br), mul(loc, n0, bc)), br, bc, depth, cols);
        b.create<dsa::MatMulOp>(loc, lhs, rhs, accumulator, Value());
        return success();
      }))) return failure();
      values.map(matrix.getResult(), accumulator);
      for (Operation *op : epilogue) {
        if (isa<CastOp>(op)) { if (failed(lowerOperation(op))) return failure(); }
        else {
          Value result = get(output.getInputs()[output.getValueOperandIndex()]);
          if (!result) return output.emitError("DSA matrix result is unavailable");
          Value rowStride = stride(loc, destination, 0), colStride = stride(loc, destination, 1);
          b.create<dsa::StoreTileOp>(loc, result, destination,
              add(loc, mul(loc, m0, rowStride), mul(loc, n0, colStride)), rowStride, colStride, rows, cols);
        }
      }
      return success();
    });
  }

  CanonicalKernelAnalysis analysis;
  ModuleOp target;
  OpBuilder b;
  dsa::ConfigurationAttr config;
  func::FuncOp function;
  IRMapping values;
  DenseMap<Value, SmallVector<Value>> products;
  DenseMap<int64_t, Value> dimensions;
  DenseMap<Value, Domain> domains;
  Value taskId, taskCount, activeBegin, activeCount;
  int64_t activeDimension = 0;
  unsigned parallelDepth = 0;
  SmallVector<int64_t> fullExtents;
};
}
LogicalResult lowerCanonicalKIRToDSA(ModuleOp module, dsa::ConfigurationAttr configuration) {
  CanonicalKernelAnalysis analysis(module);
  if (failed(analysis.verify())) return failure();
  OwningOpRef<ModuleOp> physical = ModuleOp::create(module.getLoc());
  Construction construction(module, *physical, configuration);
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (failed(construction.lower(function))) return failure();
  if (failed(dsa::verifyProgram(*physical))) return failure();
  module.getBodyRegion().takeBody(physical->getBodyRegion());
  module->setAttrs((*physical)->getAttrs());
  return success();
}
}
