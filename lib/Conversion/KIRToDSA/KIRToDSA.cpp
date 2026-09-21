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
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include <functional>
#include <limits>
using namespace mlir;
namespace intent {
namespace {
struct Domain { Value begin, end, step; int64_t dimension; std::optional<int64_t> capacity; };
// An axis keeps its source extent even when only one execution slice is local.
struct LocalAxis { Value extent, begin, count; int64_t capacity; };
using LocalShape = SmallVector<LocalAxis, 2>;
struct AccessAxes {
  unsigned rank = 0, advancedRank = 0;
  std::optional<unsigned> advancedStart;
  SmallVector<int64_t> terms;
};
using AxisRequirements = SmallVector<bool, 4>;
struct WorksetTiling {
  DenseMap<Value, AxisRequirements> requirements;
  SmallVector<Operation *> writes;
  unsigned axis = 0;
};
struct AffineIndices {
  Value base;
  SmallVector<Value> steps;
};
class Construction {
public:
  Construction(ModuleOp original, ModuleOp target, dsa::ConfigurationAttr configuration, DictionaryAttr shapes)
      : analysis(original), target(target), b(target.getContext()), config(configuration), shapeBindings(shapes) {}

  LogicalResult lower(func::FuncOp source) {
    auto parameters = source->getAttrOfType<ArrayAttr>("intent.parameters");
    if (!parameters || parameters.size() != source.getNumArguments())
      return source.emitError("DSA construction requires canonical parameter metadata");
    DenseMap<int64_t, int64_t> fixedDimensions;
    llvm::SmallSet<StringRef, 8> boundNames;
    for (auto [argument, parameter] : llvm::zip(source.getArguments(), parameters)) {
      auto name = cast<ParameterAttr>(parameter).getName();
      auto binding = shapeBindings ? shapeBindings.getAs<DenseI64ArrayAttr>(name.getValue()) : DenseI64ArrayAttr();
      if (!binding) continue;
      auto view = dyn_cast<ViewType>(argument.getType());
      if (!view) return source.emitError("DSA shape binding must name a view parameter");
      auto tensor = cast<RankedTensorType>(view.getTensor());
      if (binding.size() != tensor.getRank()) return source.emitError("DSA shape binding rank mismatch for ") << name;
      auto identities = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions();
      for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
        int64_t extent = binding[axis];
        if (extent < -1) return source.emitError("DSA shape bindings use -1 for an unspecialized axis");
        if (extent < 0) continue;
        if (!tensor.isDynamicDim(axis) && tensor.getDimSize(axis) != extent)
          return source.emitError("DSA shape binding contradicts a source static extent");
        if (identities[axis] > 0) {
          auto [entry, inserted] = fixedDimensions.try_emplace(identities[axis], extent);
          if (!inserted && entry->second != extent) return source.emitError("DSA shape bindings disagree on a logical dimension");
        }
      }
      boundNames.insert(name.getValue());
    }
    if (shapeBindings && boundNames.size() != shapeBindings.size())
      return source.emitError("DSA shape binding names an unknown parameter");
    SmallVector<Type> arguments;
    SmallVector<Attribute> interface;
    SmallVector<Value> sourceArguments;
    for (auto [argument, parameter] : llvm::zip(source.getArguments(), parameters)) {
      auto name = cast<ParameterAttr>(parameter).getName();
      if (isa<ConstexprType>(argument.getType()) && argument.use_empty()) continue;
      if (auto view = dyn_cast<ViewType>(argument.getType())) {
        auto tensor = cast<RankedTensorType>(view.getTensor());
        Type element = tensor.getElementType();
        if (!element.isF16() && !element.isBF16() && !element.isF32() && !element.isInteger(32) && !element.isInteger(64) && !element.isInteger(1))
          return source.emitError("DSA construction does not implement this view storage type");
        auto shape = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
        if (!shape) return source.emitError("DSA view has no dimension identities");
        SmallVector<int64_t> specialized(tensor.getShape());
        for (int64_t axis = 0; axis < tensor.getRank(); ++axis)
          if (auto found = fixedDimensions.find(shape.getDimensions()[axis]); found != fixedDimensions.end()) specialized[axis] = found->second;
        arguments.push_back(MemRefType::get(specialized, tensor.getElementType()));
        interface.push_back(dsa::ViewArgumentAttr::get(b.getContext(), name, tensor.getElementType(),
            b.getDenseI64ArrayAttr(specialized), shape.getDimensions(), view.getAccess(), view.getConstraints()));
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
          auto physical = cast<MemRefType>(value.getType());
          Value size = physical.isDynamicDim(axis) ? Value(b.create<memref::DimOp>(source.getLoc(), value, axis))
                                                  : index(source.getLoc(), physical.getDimSize(axis));
          if (ids[axis] > 0) dimensions.try_emplace(ids[axis], size);
        }
      }
    }
    taskId = b.create<dsa::TaskIdOp>(source.getLoc(), b.getIndexType());
    taskCount = b.create<dsa::TaskCountOp>(source.getLoc(), b.getIndexType());
    SmallVector<ContractOp> matrices;
    source.walk([&](ContractOp matrix) { matrices.push_back(matrix); });
    structured = llvm::any_of(matrices, [&](ContractOp op) { return op->getBlock() != &source.front(); });
    source.walk([&](Operation *op) { structured |= isa<RegionFoldOp, RegionScanOp>(op); });
    if (matrices.empty()) source.walk([&](Operation *op) {
      for (Type type : op->getResultTypes()) if (auto tensor = dyn_cast<RankedTensorType>(type)) structured |= tensor.getRank() > 1;
    });
    if (!structured && !matrices.empty()) {
      if (matrices.size() != 1) return source.emitError("DSA top-level MatMul composition requires bounded local shapes");
      if (failed(lowerMatrix(source, matrices.front()))) return failure();
    } else {
      SmallVector<ParallelOp> roots;
      source.walk([&](ParallelOp op) { if (!op->getParentOfType<ParallelOp>()) roots.push_back(op); });
      if (!roots.empty()) {
        if (roots.size() != 1 || roots.front()->getBlock() != &source.front())
          return source.emitError("DSA block tasks currently require one outer parallel workset; multiple task phases need a target-wide join realization");
        auto worksets = analysis.logicalWorksets(source);
        if (failed(worksets)) return failure();
        if (structured && worksets->size() == 1 && worksets->front().isExact() &&
            !worksets->front().singleton && independentDomains(worksets->front())) {
          distributedWorkset = worksets->front();
          distributedRoot = roots.front();
        }
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
  Value add(Location loc, Value a, Value c) { return b.createOrFold<arith::AddIOp>(loc, a, c); }
  Value mul(Location loc, Value a, Value c) { return b.createOrFold<arith::MulIOp>(loc, a, c); }
  Value sub(Location loc, Value a, Value c) { return b.createOrFold<arith::SubIOp>(loc, a, c); }
  Value get(Value source) {
    Value value = values.lookupOrNull(source);
    if (!value && structured && source.getDefiningOp()) {
      if (failed(materialize(source.getDefiningOp()))) return {};
      value = values.lookupOrNull(source);
    }
    if (value && structured) if (auto tensor = dyn_cast<RankedTensorType>(source.getType())) {
      value = projectTensor(source.getLoc(), tensor, value, source);
      if (value) values.map(source, value);
    }
    return value;
  }
  LogicalResult materialize(Operation *op) {
    if (!materializing.insert(op).second) return op->emitError("cyclic DSA value materialization");
    LogicalResult result = lowerOperation(op);
    materializing.erase(op);
    return result;
  }
  Value asIndex(Value value, Location loc) {
    if (value && value.getType().isIndex()) return value;
    if (value && value.getType().isInteger(1)) value = b.create<arith::ExtUIOp>(loc, b.getI64Type(), value);
    if (value && isa<IntegerType>(value.getType())) return b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
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
  FailureOr<LocalShape> localShape(RankedTensorType type, Location loc) {
    auto ids = cast<TensorShapeAttr>(type.getEncoding()).getDimensions();
    LocalShape shape;
    for (int64_t axis = 0; axis < type.getRank(); ++axis) {
      if (type.getDimSize(axis) != 1) {
        auto bound = axisBindings.find(ids[axis]);
        if (bound != axisBindings.end()) { shape.push_back(bound->second); continue; }
      }
      Value size = extent(type, axis, loc);
      APInt constant;
      if (!size || !matchPattern(size, m_ConstantInt(&constant)) || constant.isNegative())
        return emitError(loc, "DSA local axis needs a selected execution slice or a compile-call shape binding; dimension ") << ids[axis], failure();
      int64_t capacity = constant.getSExtValue();
      shape.push_back({size, index(loc, 0), size, std::max<int64_t>(capacity, 1)});
    }
    int64_t elements = 1;
    for (const auto &axis : shape) {
      if (elements > std::numeric_limits<int64_t>::max() / axis.capacity)
        return emitError(loc, "DSA selected tensor capacity exceeds the address range"), failure();
      elements *= axis.capacity;
    }
    return shape;
  }
  Type storageElement(Type type) { return type.isIndex() ? b.getI64Type() : type; }
  FailureOr<LocalShape> localShape(Value value, Location loc) {
    if (auto selected = valueSlices.find(value); selected != valueSlices.end()) return selected->second;
    return localShape(cast<RankedTensorType>(value.getType()), loc);
  }
  Value allocateTensor(Location loc, Type element, const LocalShape &shape) {
    int64_t rows = 1;
    for (unsigned axis = 0; axis + 1 < shape.size(); ++axis) rows *= shape[axis].capacity;
    int64_t cols = shape.empty() ? 1 : shape.back().capacity;
    Value value = allocate(loc, storageElement(element), rows, cols);
    localShapes[value] = shape;
    Type stored = cast<MemRefType>(value.getType()).getElementType();
    Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(stored));
    b.create<dsa::FillOp>(loc, value, zero);
    return value;
  }
  bool sameIndex(Value a, Value c) {
    if (a == c) return true;
    APInt lhs, rhs;
    return matchPattern(a, m_ConstantInt(&lhs)) && matchPattern(c, m_ConstantInt(&rhs)) && lhs == rhs;
  }
  Value projectTensor(Location loc, RankedTensorType type, Value value, Value source = {}) {
    if (!localShapes.count(value)) return value;
    LocalShape original = localShapes.lookup(value), selected = original;
    auto ids = cast<TensorShapeAttr>(type.getEncoding()).getDimensions();
    bool changed = false;
    auto planned = valueSlices.find(source);
    for (int64_t axis = 0; axis < type.getRank(); ++axis) {
      auto binding = axisBindings.find(ids[axis]);
      if (type.getDimSize(axis) == 1 || (planned == valueSlices.end() && binding == axisBindings.end())) continue;
      const auto &current = original[axis];
      const auto &required = planned == valueSlices.end() ? binding->second : planned->second[axis];
      if (current.capacity == required.capacity && sameIndex(current.begin, required.begin) && sameIndex(current.count, required.count)) continue;
      if (!matchPattern(current.begin, m_Zero()) || !sameIndex(current.count, current.extent) || !sameIndex(current.extent, required.extent)) {
        emitError(loc, "DSA cached tensor projection requires an available complete logical axis"); return {};
      }
      selected[axis] = required; changed = true;
    }
    if (!changed) return value;
    auto physical = cast<MemRefType>(value.getType());
    Value result = allocateTensor(loc, type.getElementType(), selected);
    if (selected.size() > 2) {
      if (failed(eachElement(loc, selected, [&](ValueRange coordinates) {
        SmallVector<Value> source;
        for (unsigned axis = 0; axis < selected.size(); ++axis)
          source.push_back(add(loc, coordinates[axis], sub(loc, selected[axis].begin, original[axis].begin)));
        storeLocal(loc, loadLocal(loc, value, source), result, coordinates);
        return success();
      }))) return {};
      return result;
    }
    Value offset = index(loc, 0);
    if (selected.size() == 2)
      offset = mul(loc, sub(loc, selected[0].begin, original[0].begin), index(loc, physical.getDimSize(1)));
    if (!selected.empty()) offset = add(loc, offset, sub(loc, selected.back().begin, original.back().begin));
    b.create<dsa::LoadTileOp>(loc, value, result, offset,
        index(loc, selected.size() == 2 ? physical.getDimSize(1) : 0), index(loc, 1),
        selected.size() == 2 ? selected[0].count : index(loc, 1), selected.empty() ? index(loc, 1) : selected.back().count);
    return result;
  }
  SmallVector<Value> physicalCoordinates(Location loc, Value buffer, ValueRange coordinates) {
    Value row = index(loc, 0);
    if (coordinates.size() >= 2) row = coordinates.front();
    if (coordinates.size() > 2) {
      const auto &shape = localShapes.find(buffer)->second;
      for (unsigned axis = 1; axis + 1 < coordinates.size(); ++axis)
        row = add(loc, mul(loc, row, index(loc, shape[axis].capacity)), coordinates[axis]);
    }
    return {row, coordinates.empty() ? index(loc, 0) : coordinates.back()};
  }
  Value loadLocal(Location loc, Value value, ValueRange coordinates) {
    return b.create<memref::LoadOp>(loc, value, physicalCoordinates(loc, value, coordinates));
  }
  void storeLocal(Location loc, Value value, Value buffer, ValueRange coordinates) {
    value = scalarCast(loc, value, cast<MemRefType>(buffer.getType()).getElementType());
    b.create<memref::StoreOp>(loc, value, buffer, physicalCoordinates(loc, buffer, coordinates));
  }
  LogicalResult eachElement(Location loc, const LocalShape &shape,
      const std::function<LogicalResult(ValueRange)> &body) {
    SmallVector<Value> coordinates;
    std::function<LogicalResult(unsigned)> visit = [&](unsigned axis) -> LogicalResult {
      if (axis == shape.size()) return body(coordinates);
      return loop(loc, index(loc, 0), shape[axis].count, index(loc, 1), [&](Value i) {
        coordinates.push_back(i);
        LogicalResult result = visit(axis + 1);
        coordinates.pop_back();
        return result;
      });
    };
    return visit(0);
  }
  bool compactPrefix(const LocalShape &shape) {
    for (unsigned axis = 1; axis < shape.size(); ++axis) {
      APInt count;
      if (!matchPattern(shape[axis].count, m_ConstantInt(&count)) || count.getSExtValue() != shape[axis].capacity) return false;
    }
    return true;
  }
  bool completeShape(const LocalShape &shape) {
    return llvm::all_of(shape, [&](const LocalAxis &axis) {
      return matchPattern(axis.begin, m_Zero()) && sameIndex(axis.count, axis.extent);
    });
  }
  Value contiguousView(Location loc, Value source, const LocalShape &shape) {
    auto type = cast<MemRefType>(source.getType());
    int64_t rows = 1;
    for (unsigned axis = 0; axis + 1 < shape.size(); ++axis) rows *= shape[axis].capacity;
    int64_t columns = shape.empty() ? 1 : shape.back().capacity;
    auto targetType = MemRefType::get({rows, columns}, type.getElementType(), MemRefLayoutAttrInterface{}, type.getMemorySpace());
    Value result = b.create<memref::ReinterpretCastOp>(loc, targetType, source, int64_t(0),
        ArrayRef<int64_t>{rows, columns}, ArrayRef<int64_t>{columns, 1});
    localShapes[result] = shape;
    return result;
  }
  LogicalResult reshapeTensor(Operation *op, const LocalShape &shape) {
    Location loc = op->getLoc();
    Value input = get(op->getOperand(0));
    if (!input || !localShapes.count(input)) return op->emitError("DSA reshape input is unavailable");
    LocalShape original = localShapes.lookup(input);
    if (!completeShape(original) || !completeShape(shape))
      return op->emitError("DSA reshape requires a complete logical tensor before selecting execution slices");
    int64_t elements = 1;
    for (const auto &axis : shape) elements *= axis.capacity;
    Value output;
    if (compactPrefix(original) && compactPrefix(shape) && elements == cast<MemRefType>(input.getType()).getNumElements()) {
      output = contiguousView(loc, input, shape);
    } else {
      output = allocateTensor(loc, cast<RankedTensorType>(op->getResult(0).getType()).getElementType(), shape);
      if (failed(eachElement(loc, shape, [&](ValueRange coordinates) {
        Value linear = index(loc, 0);
        for (unsigned axis = 0; axis < shape.size(); ++axis)
          linear = add(loc, mul(loc, linear, shape[axis].count), coordinates[axis]);
        SmallVector<Value> source(original.size());
        for (int64_t axis = static_cast<int64_t>(original.size()) - 1; axis >= 0; --axis) {
          // A zero-sized tensor has no active iterations; use a nonzero divisor
          // so its unreachable indexing remains well-defined during folding.
          Value count = b.create<arith::MaxSIOp>(loc, original[axis].count, index(loc, 1));
          source[axis] = b.create<arith::RemSIOp>(loc, linear, count);
          linear = b.create<arith::DivSIOp>(loc, linear, count);
        }
        storeLocal(loc, loadLocal(loc, input, source), output, coordinates);
        return success();
      }))) return failure();
    }
    values.map(op->getResult(0), output); return success();
  }
  LogicalResult joinTensor(Operation *op, const LocalShape &shape) {
    Location loc = op->getLoc();
    Value lhs = get(op->getOperand(0)), rhs = get(op->getOperand(1));
    if (!lhs || !rhs || !localShapes.count(lhs) || !localShapes.count(rhs))
      return op->emitError("DSA join inputs are unavailable");
    LocalShape inputShape = localShapes.lookup(lhs), rightShape = localShapes.lookup(rhs);
    if (shape.size() != inputShape.size() + 1 || rightShape.size() != inputShape.size() ||
        shape.back().capacity != 2 || !matchPattern(shape.back().begin, m_Zero()) ||
        !sameIndex(shape.back().count, index(loc, 2)))
      return op->emitError("DSA join requires an unsliced trailing pair axis");
    for (unsigned axis = 0; axis < inputShape.size(); ++axis) {
      const LocalAxis *others[] = {&rightShape[axis], &shape[axis]};
      for (const auto *other : others)
        if (other->capacity != inputShape[axis].capacity || !sameIndex(other->begin, inputShape[axis].begin) ||
            !sameIndex(other->count, inputShape[axis].count))
          return op->emitError("DSA join inputs and output must have matching execution slices; axis ") << axis
              << ", input capacity=" << inputShape[axis].capacity << ", other capacity=" << other->capacity
              << ", input begin=" << inputShape[axis].begin << ", other begin=" << other->begin
              << ", input count=" << inputShape[axis].count << ", other count=" << other->count;
    }
    Value output = allocateTensor(loc, cast<RankedTensorType>(op->getResult(0).getType()).getElementType(), shape);
    if (!inputShape.empty() && compactPrefix(inputShape)) {
      auto physical = cast<MemRefType>(lhs.getType());
      LocalShape interleaved = inputShape;
      auto &last = interleaved.back();
      last.extent = mul(loc, last.extent, index(loc, 2));
      last.begin = mul(loc, last.begin, index(loc, 2));
      last.count = mul(loc, last.count, index(loc, 2));
      last.capacity *= 2;
      Value destination = contiguousView(loc, output, interleaved), rows = index(loc, 1);
      for (unsigned axis = 0; axis + 1 < inputShape.size(); ++axis) rows = mul(loc, rows, inputShape[axis].count);
      for (auto [slot, input] : llvm::enumerate(SmallVector<Value>{lhs, rhs}))
        b.create<dsa::StoreTileOp>(loc, input, destination, index(loc, slot), index(loc, 2 * physical.getDimSize(1)),
            index(loc, 2), rows, inputShape.back().count);
    } else if (failed(eachElement(loc, inputShape, [&](ValueRange coordinates) {
      SmallVector<Value> target(coordinates);
      target.push_back(index(loc, 0));
      storeLocal(loc, loadLocal(loc, lhs, coordinates), output, target);
      target.back() = index(loc, 1);
      storeLocal(loc, loadLocal(loc, rhs, coordinates), output, target);
      return success();
    }))) return failure();
    values.map(op->getResult(0), output); return success();
  }
  bool canDefer(Operation *op) {
    if (isa<RegionFoldOp, RegionScanOp, IfOp, ForOp, WhileOp, ParallelOp>(op)) return false;
    bool aggregate = llvm::any_of(op->getResultTypes(), [&](Type type) {
      return isa<RankedTensorType>(type) || static_cast<bool>(components(type));
    });
    if (!aggregate) return false;
    if (auto load = dyn_cast<ViewLoadOp>(op)) {
      auto view = dyn_cast<ViewType>(load.getInputs().front().getType());
      return view && view.getAccess() == 0;
    }
    return isMemoryEffectFree(op);
  }
  // Reuse the scalar numerical lowering for elementwise tensor operations.
  FailureOr<Value> scalarOperation(Operation *source, ValueRange operands) {
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
    state.addTypes(cast<RankedTensorType>(source->getResult(0).getType()).getElementType());
    Operation *scalar = Operation::create(state);
    auto saved = values;
    for (Value value : scalars) values.map(value, value);
    LogicalResult status = lowerOperation(scalar);
    Value result = values.lookupOrNull(scalar->getResult(0));
    values = std::move(saved);
    scalar->destroy();
    if (failed(status) || !result) return failure();
    return result;
  }
  LogicalResult tensorOperation(Operation *op) {
    Location loc = op->getLoc();
    Value result = op->getResult(0);
    auto type = cast<RankedTensorType>(result.getType());
    auto shape = localShape(result, loc);
    if (failed(shape)) return failure();
    if (isa<ReshapeOp>(op)) return reshapeTensor(op, *shape);
    if (isa<JoinOp>(op)) return joinTensor(op, *shape);
    if (auto matrix = dyn_cast<ContractOp>(op)) return localMatMul(matrix, *shape);
    if (auto indices = dyn_cast<IndicesOp>(op)) {
      Value source = indices->getOperand(0);
      if (!bindDomain(source) || shape->size() != 1) return op->emitError("DSA indices require a bound interval");
      Value output = allocateTensor(loc, type.getElementType(), *shape);
      Domain domain = domains.lookup(source);
      if (failed(eachElement(loc, *shape, [&](ValueRange coordinates) {
        Value coordinate = add(loc, domain.begin, mul(loc, add(loc, shape->front().begin, coordinates[0]), domain.step));
        storeLocal(loc, coordinate, output, coordinates);
        return success();
      }))) return failure();
      values.map(result, output); return success();
    }
    if (isa<BroadcastOp, FullOp>(op)) {
      Value input = get(op->getOperand(0));
      if (!input) return op->emitError("DSA broadcast input is unavailable");
      Value output = allocateTensor(loc, type.getElementType(), *shape);
      if (!isa<MemRefType>(input.getType())) {
        Value value = scalarCast(loc, input, storageElement(type.getElementType()));
        if (!value) return op->emitError("DSA broadcast element dtype is unavailable");
        b.create<dsa::FillOp>(loc, output, value);
      } else {
        auto inputType = cast<RankedTensorType>(op->getOperand(0).getType());
        if (inputType.getRank() > type.getRank()) return op->emitError("DSA broadcast cannot remove axes");
        unsigned leading = type.getRank() - inputType.getRank();
        if (failed(eachElement(loc, *shape, [&](ValueRange coordinates) {
          SmallVector<Value> source;
          for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
            bool singleton = inputType.getDimSize(axis) == 1 || matchPattern(localShapes.lookup(input)[axis].extent, m_One());
            source.push_back(singleton ? index(loc, 0) : coordinates[leading + axis]);
          }
          storeLocal(loc, loadLocal(loc, input, source), output, coordinates);
          return success();
        }))) return failure();
      }
      values.map(result, output); return success();
    }
    if (auto transpose = dyn_cast<TransposeOp>(op)) {
      Value input = get(transpose.getInput());
      if (!input) return op->emitError("DSA transpose input is unavailable");
      Value output = allocateTensor(loc, type.getElementType(), *shape);
      if (failed(eachElement(loc, *shape, [&](ValueRange coordinates) {
        SmallVector<Value> source(coordinates.size());
        for (auto [axis, perm] : llvm::enumerate(transpose.getPermutation()))
          source[cast<IntegerAttr>(perm).getInt()] = coordinates[axis];
        storeLocal(loc, loadLocal(loc, input, source), output, coordinates); return success();
      }))) return failure();
      values.map(result, output); return success();
    }
    if (!isa<UnaryOp, BinaryOp, CompareOp, SelectOp, MaskOp, CastOp>(op))
      return op->emitError("tensor operation has no DSA local implementation");
    SmallVector<Value> inputs;
    for (Value operand : op->getOperands()) {
      Value value = get(operand);
      if (!value) return op->emitError("DSA tensor operand is unavailable");
      inputs.push_back(value);
    }
    Value output = allocateTensor(loc, type.getElementType(), *shape);
    bool matching = llvm::all_of(inputs, [&](Value value) {
      auto buffer = dyn_cast<MemRefType>(value.getType());
      return buffer && buffer.getShape() == cast<MemRefType>(output.getType()).getShape();
    });
    bool floating = isa<FloatType>(type.getElementType());
    if (auto binary = dyn_cast<BinaryOp>(op); binary && matching && floating)
      b.create<dsa::BinaryOp>(loc, inputs[0], inputs[1], output, binary.getOperatorKindAttr(), binary.getApproximateAttr(), binary.getFlushToZeroAttr());
    else if (auto unary = dyn_cast<UnaryOp>(op); unary && matching && floating)
      b.create<dsa::UnaryOp>(loc, inputs[0], output, unary.getOperatorKindAttr(), unary.getApproximateAttr(), unary.getFlushToZeroAttr());
    else if (auto cast = dyn_cast<CastOp>(op); cast && matching) {
      if (cast.getRounding()) return cast.emitError("DSA explicit rounding is not implemented");
      b.create<dsa::CastOp>(loc, inputs[0], output);
    } else if (isa<SelectOp>(op) && matching)
      b.create<dsa::SelectOp>(loc, inputs[0], inputs[1], inputs[2], output);
    else if (isa<MaskOp>(op) && matching)
      b.create<dsa::SelectOp>(loc, inputs[1], inputs[0], inputs[2], output);
    else if (failed(eachElement(loc, *shape, [&](ValueRange coordinates) {
      SmallVector<Value> scalars;
      for (Value input : inputs) scalars.push_back(isa<MemRefType>(input.getType()) ? loadLocal(loc, input, coordinates) : input);
      auto value = scalarOperation(op, scalars);
      if (failed(value)) return failure();
      storeLocal(loc, *value, output, coordinates); return success();
    }))) return failure();
    values.map(result, output); return success();
  }
  LogicalResult localMatMul(ContractOp matrix, const LocalShape &shape, Value output = {}, bool stream = true) {
    Location loc = matrix.getLoc();
    auto leftType = cast<RankedTensorType>(matrix.getLhs().getType());
    auto rightType = cast<RankedTensorType>(matrix.getRhs().getType());
    auto pair = matrix.getReduce().size() == 1 ? dyn_cast<ArrayAttr>(matrix.getReduce()[0]) : ArrayAttr();
    if (leftType.getRank() < 1 || leftType.getRank() > 2 || rightType.getRank() < 1 || rightType.getRank() > 2 || !pair || pair.size() != 2 ||
        !matrix.getBatch().empty() ||
        !cast<RankedTensorType>(matrix.getResult().getType()).getElementType().isF32())
      return matrix.emitError("DSA local matrix/vector multiplication requires one paired axis and f32 accumulation");
    int64_t leftReduce = cast<IntegerAttr>(pair[0]).getInt(), rightReduce = cast<IntegerAttr>(pair[1]).getInt();
    if (stream && shape.size() == 2 && leftType.getRank() == 2 && rightType.getRank() == 2) {
      std::pair<Value, unsigned> roots[] = {{matrix.getLhs(), leftReduce}, {matrix.getRhs(), rightReduce}};
      if (auto plan = planExecutionSlices(*matrix->getBlock(), 0, roots)) {
        DenseMap<Value, LocalShape> complete;
        Value size;
        int64_t capacity = config.getTileK();
        bool available = true, needsSlicing = false;
        for (auto &entry : plan->requirements) {
          auto selected = localShape(entry.first, loc);
          if (failed(selected)) return failure();
          for (unsigned axis = 0; axis < selected->size(); ++axis) if (entry.second[axis]) {
            const auto &local = (*selected)[axis];
            available &= matchPattern(local.begin, m_Zero()) && sameIndex(local.count, local.extent) &&
                (!size || sameIndex(size, local.count));
            size = local.count;
            needsSlicing |= local.capacity > config.getTileK();
            capacity = std::min(capacity, local.capacity);
          }
          complete[entry.first] = *selected;
        }
        if (available && needsSlicing && size) {
          Value accumulator = allocateTensor(loc, b.getF32Type(), shape);
          auto savedValues = values; auto savedProducts = products; auto savedSlices = valueSlices;
          LogicalResult status = loop(loc, index(loc, 0), size, index(loc, capacity), [&](Value begin) {
            Value count = b.create<arith::MinSIOp>(loc, sub(loc, size, begin), index(loc, capacity));
            for (auto &entry : plan->requirements) {
              LocalShape selected = complete.lookup(entry.first);
              for (unsigned axis = 0; axis < selected.size(); ++axis)
                if (entry.second[axis]) selected[axis] = {selected[axis].extent, begin, count, capacity};
              valueSlices[entry.first] = std::move(selected);
            }
            return localMatMul(matrix, shape, accumulator, false);
          });
          values = std::move(savedValues); products = std::move(savedProducts); valueSlices = std::move(savedSlices);
          if (failed(status)) return failure();
          values.map(matrix.getResult(), accumulator);
          return success();
        }
      }
    }
    Value lhs = get(matrix.getLhs()), rhs = get(matrix.getRhs());
    if (!lhs || !rhs || !localShapes.count(lhs) || !localShapes.count(rhs)) return matrix.emitError("DSA local MatMul inputs are unavailable");
    auto normalize = [&](Value input, bool left, int64_t reduction) -> Value {
      LocalShape original = localShapes.lookup(input);
      bool transpose = original.size() == 2 && reduction == (left ? 0 : 1);
      bool columnVector = original.size() == 1 && !left;
      if (!transpose && !columnVector) return input;
      LocalAxis unit{index(loc, 1), index(loc, 0), index(loc, 1), 1};
      LocalShape normalized = transpose ? LocalShape{original[1], original[0]} : LocalShape{original[0], unit};
      Value output = allocateTensor(loc, cast<MemRefType>(input.getType()).getElementType(), normalized);
      if (failed(eachElement(loc, normalized, [&](ValueRange coordinates) {
        SmallVector<Value> from = transpose ? SmallVector<Value>{coordinates[1], coordinates[0]} : SmallVector<Value>{coordinates[0]};
        storeLocal(loc, loadLocal(loc, input, from), output, coordinates); return success();
      }))) return {};
      return output;
    };
    LocalShape leftShape = localShapes.lookup(lhs), rightShape = localShapes.lookup(rhs);
    lhs = normalize(lhs, true, leftReduce); rhs = normalize(rhs, false, rightReduce);
    if (!lhs || !rhs) return failure();
    if (!output) output = allocateTensor(loc, b.getF32Type(), shape);
    int64_t rows = cast<MemRefType>(lhs.getType()).getDimSize(0), columns = cast<MemRefType>(rhs.getType()).getDimSize(1);
    Value rowCount = leftShape.size() == 1 ? index(loc, 1) : leftShape[1 - leftReduce].count;
    Value columnCount = rightShape.size() == 1 ? index(loc, 1) : rightShape[1 - rightReduce].count;
    Value accumulator = output;
    if (cast<MemRefType>(output.getType()).getShape() != ArrayRef<int64_t>({rows, columns})) {
      accumulator = allocate(loc, b.getF32Type(), rows, columns);
      b.create<dsa::FillOp>(loc, accumulator, b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0)));
      b.create<dsa::FillOp>(loc, accumulator, b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0)));
    }
    b.create<dsa::MatMulOp>(loc, lhs, rhs, accumulator, rowCount, leftShape[leftReduce].count, columnCount);
    if (accumulator != output) {
      if (shape.size() != 1 || rightShape.size() != 1) return matrix.emitError("DSA matrix result projection is unavailable");
      if (failed(eachElement(loc, shape, [&](ValueRange coordinates) {
        storeLocal(loc, loadLocal(loc, accumulator, ValueRange{coordinates[0], index(loc, 0)}), output, coordinates); return success();
      }))) return failure();
    }
    values.map(matrix.getResult(), output); return success();
  }
  bool constantTrue(Value value) {
    Operation *op = value.getDefiningOp();
    if (!op) return false;
    if (isa<FullOp, BroadcastOp>(op)) return constantTrue(op->getOperand(0));
    auto constant = dyn_cast<ConstantOp>(op);
    auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
    return integer && integer.getType().isInteger(1) && integer.getValue().isOne();
  }
  FailureOr<AffineIndices> affineIndex(Value original) {
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
          if (singletonAxis(input, axis)) continue;
          if (!equalAxisExtent(input, axis, type, leading + axis)) return failure();
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
      if (auto valid = op->getAttrOfType<IntegerAttr>("valid_operand_index"))
        if (!constantTrue(op->getOperand(valid.getInt()))) return failure();
      auto relation = analysis.indexRelation(gather);
      if (failed(relation)) return failure();
      auto source = affineIndex(relation->source);
      if (failed(source)) return failure();
      auto axes = accessAxes(*relation);
      AffineIndices result{source->base, SmallVector<Value>(type.getRank(), index(loc, 0))};
      for (auto [position, term] : llvm::enumerate(relation->terms)) {
        if (term.kind == 1) continue;
        if (!term.sourceAxis) return failure();
        Value coefficient = source->steps[*term.sourceAxis];
        if (term.kind == 0) result.steps[axes.terms[position]] = coefficient;
        else if (term.kind == 4) {
          if (!bindDomain(term.operands.front())) return failure();
          Domain domain = domains.lookup(term.operands.front());
          result.base = add(loc, result.base, mul(loc, coefficient, domain.begin));
          result.steps[axes.terms[position]] = mul(loc, coefficient, domain.step);
        } else if (term.kind == 2) {
          int64_t literal = *term.staticValues.front();
          Value coordinate = index(loc, literal);
          if (literal < 0) {
            Value size = extent(cast<RankedTensorType>(relation->source.getType()), *term.sourceAxis, loc);
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
  AccessAxes accessAxes(const IndexRelationFact &relation) {
    AccessAxes axes;
    for (const auto &term : relation.terms) if (term.kind == 3)
      if (auto indices = dyn_cast<RankedTensorType>(term.operands.front().getType()))
        axes.advancedRank = std::max<unsigned>(axes.advancedRank, indices.getRank());
    for (const auto &term : relation.terms) {
      if (term.kind == 3 && isa<RankedTensorType>(term.operands.front().getType())) {
        if (!axes.advancedStart) { axes.advancedStart = axes.rank; axes.rank += axes.advancedRank; }
        axes.terms.push_back(*axes.advancedStart);
      } else if (term.kind == 0 || term.kind == 1 || term.kind == 4 || term.kind == 5) axes.terms.push_back(axes.rank++);
      else axes.terms.push_back(-1);
    }
    return axes;
  }
  LogicalResult tensorAccess(Operation *op) {
    auto relation = analysis.indexRelation(op);
    if (failed(relation)) return failure();
    Location loc = op->getLoc();
    bool store = isa<ViewStoreOp, ScatterUniqueOp>(op);
    bool external = isa<ViewType>(relation->source.getType());
    Value source = get(relation->source);
    if (!source) return op->emitError("DSA indexed source is unavailable");
    Value original = store ? op->getOperand(op->getAttrOfType<IntegerAttr>("value_operand_index").getInt()) : op->getResult(0);
    auto tensor = dyn_cast<RankedTensorType>(original.getType());
    LocalShape shape;
    if (tensor) {
      auto selected = localShape(original, loc);
      if (failed(selected)) return failure();
      shape = *selected;
    }
    Value data = store ? get(original) : allocateTensor(loc, tensor ? tensor.getElementType() : original.getType(), shape);
    if (!data) return op->emitError("DSA store data is unavailable");
    auto validIndex = op->getAttrOfType<IntegerAttr>("valid_operand_index");
    auto fillIndex = op->getAttrOfType<IntegerAttr>("fill_operand_index");
    Value valid = validIndex ? get(op->getOperand(validIndex.getInt())) : Value();
    Value fill = fillIndex ? get(op->getOperand(fillIndex.getInt())) : Value();
    if ((validIndex && !valid) || (fillIndex && !fill)) return op->emitError("DSA access predicate or fill is unavailable");

    auto axes = accessAxes(*relation);
    unsigned advancedRank = axes.advancedRank;
    auto advancedStart = axes.advancedStart;
    const auto &outputAxes = axes.terms;
    if (axes.rank != shape.size()) return op->emitError("DSA index relation requires rank-one basic regions and a broadcasted advanced-index group");
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
    bool rectangle = external && !validIndex && tensor && shape.size() <= 2;
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
      for (auto [termIndex, term] : llvm::enumerate(relation->terms)) {
        int64_t outputAxis = outputAxes[termIndex];
        if (term.kind == 1) continue;
        if (!term.sourceAxis) return op->emitError("DSA access has no source axis");
        Value step = stride(loc, source, *term.sourceAxis), coordinate;
        if (term.kind == 2) coordinate = staticCoordinate(term);
        else if (term.kind == 3) {
          if (auto found = affine.find(term.operands.front()); found != affine.end()) {
            coordinate = found->second.base;
            auto type = cast<RankedTensorType>(term.operands.front().getType());
            for (unsigned axis = 0; axis < type.getRank(); ++axis) {
              if (singletonAxis(type, axis)) continue;
              unsigned mapped = *advancedStart + advancedRank - type.getRank() + axis;
              Value coefficient = found->second.steps[axis];
              coordinate = add(loc, coordinate, mul(loc, shape[mapped].begin, coefficient));
              strides[mapped] = add(loc, strides[mapped], mul(loc, step, coefficient));
            }
          } else coordinate = asIndex(get(term.operands.front()), loc);
        }
        else {
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
      Value rows = shape.size() == 2 ? shape[0].count : index(loc, 1);
      Value cols = shape.empty() ? index(loc, 1) : shape.back().count;
      Value rowStride = strides.size() == 2 ? strides[0] : index(loc, 0);
      Value colStride = strides.empty() ? index(loc, 0) : strides.back();
      if (store) b.create<dsa::StoreTileOp>(loc, data, source, offset, rowStride, colStride, rows, cols);
      else { b.create<dsa::LoadTileOp>(loc, source, data, offset, rowStride, colStride, rows, cols); values.map(original, data); }
      return success();
    }
    auto readAt = [&](Value value, ValueRange coordinates) {
      return isa<MemRefType>(value.getType()) ? loadLocal(loc, value, coordinates) : value;
    };
    DenseMap<Value, Value> indexValues;
    for (const auto &term : relation->terms) if (term.kind == 3) {
      Value value = get(term.operands.front());
      if (!value) return op->emitError("DSA index operand is unavailable");
      indexValues[term.operands.front()] = value;
    }
    if (failed(eachElement(loc, shape, [&](ValueRange coordinates) -> LogicalResult {
      if (fill) storeLocal(loc, readAt(fill, coordinates), data, coordinates);
      auto access = [&]() -> LogicalResult {
        Value offset = index(loc, 0);
        SmallVector<Value> sourceCoordinates;
        for (auto [termIndex, term] : llvm::enumerate(relation->terms)) {
          int64_t outputAxis = outputAxes[termIndex];
          if (term.kind == 1) continue;
          if (!term.sourceAxis) return op->emitError("DSA indexing requires explicit source axes");
          Value coordinate;
          if (term.kind == 0 || term.kind == 4) {
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
                unsigned mapped = *advancedStart + advancedRank - indexType.getRank() + axis;
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

  void bindHelperValue(Value formal, ArrayRef<Value> fields, bool rebase = false, int64_t sourceAxis = -1) {
    SmallVector<Type> types; leaves(formal.getType(), types);
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
  FailureOr<SmallVector<Value>> helper(Block &block, ArrayRef<SmallVector<Value>> arguments,
                                      unsigned sourceCount = 0, int64_t sourceAxis = -1) {
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
  SmallVector<SmallVector<Value>> splitFields(TypeRange types, ValueRange fields) {
    SmallVector<SmallVector<Value>> result;
    unsigned offset = 0;
    for (Type type : types) {
      SmallVector<Type> schema; leaves(type, schema);
      result.push_back(llvm::to_vector(fields.slice(offset, schema.size())));
      offset += schema.size();
    }
    return result;
  }
  LogicalResult streamReduction(ReduceOp reduce, unsigned axis, int64_t dimension, const LocalShape &shape) {
    Location loc = reduce.getLoc();
    ValueRange sources = reduce.getInputs().take_front(reduce.getSourceCount());
    ValueRange identities = reduce.getInputs().slice(reduce.getSourceCount(), reduce.getIdentityCount());
    ValueRange captures = reduce.getInputs().take_back(reduce.getCaptureCount());
    SmallVector<Value> initial = flatten(identities), outputs;
    SmallVector<Type> elementTypes, resultTypes;
    for (Type type : identities.getTypes()) leaves(type, elementTypes);
    for (Type type : reduce->getResultTypes()) leaves(type, resultTypes);
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
  LogicalResult reduceTensor(ReduceOp reduce) {
    Location loc = reduce.getLoc();
    if (reduce.getAxes().size() != 1) return reduce.emitError("DSA local reduction currently requires one selected axis");
    unsigned axis = cast<IntegerAttr>(reduce.getAxes()[0]).getInt();
    ValueRange sources = reduce.getInputs().take_front(reduce.getSourceCount());
    ValueRange identities = reduce.getInputs().slice(reduce.getSourceCount(), reduce.getIdentityCount());
    ValueRange captures = reduce.getInputs().take_back(reduce.getCaptureCount());
    SmallVector<Type> sourceTypes;
    for (Type type : sources.getTypes()) leaves(type, sourceTypes);
    if (!sourceTypes.empty() && valueSlices.empty()) {
      auto first = dyn_cast<RankedTensorType>(sourceTypes.front());
      if (first && axis < first.getRank()) {
        auto shape = localShape(first, loc);
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
    for (Type type : reduce->getResultTypes()) leaves(type, resultTypes);
    if (resultTypes.size() != initial.size()) return reduce.emitError("DSA reduction result schema mismatch");
    SmallVector<Value> outputs;
    for (Type type : resultTypes) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      outputs.push_back(allocateTensor(loc, tensor ? tensor.getElementType() : type, resultShape));
    }
    auto combineOps = reduce.getCombine().front().without_terminator();
    auto binary = llvm::hasSingleElement(combineOps) ? dyn_cast<BinaryOp>(&*combineOps.begin()) : BinaryOp();
    bool simple = inputs.size() == 1 && reduce.getIdentityCount() == 1 && captures.empty() && binary &&
        !binary.getApproximate() && !binary.getFlushToZero() && axis + 1 == inputShape.size() &&
        binary.getLhs() == reduce.getCombine().front().getArgument(0) &&
        binary.getRhs() == reduce.getCombine().front().getArgument(1) &&
        reduce.getCombine().front().getTerminator()->getOperand(0) == binary.getResult();
    Type inputElement = cast<MemRefType>(inputs.front().getType()).getElementType();
    BinaryOperator kind = binary ? binary.getOperatorKind() : BinaryOperator::Add;
    bool boolean = inputElement.isInteger(1) && (kind == BinaryOperator::LogicalOr || kind == BinaryOperator::LogicalAnd);
    bool numeric = inputElement.isF32() && (kind == BinaryOperator::Add || kind == BinaryOperator::Maximum ||
        kind == BinaryOperator::Minimum || kind == BinaryOperator::MaximumNum || kind == BinaryOperator::MinimumNum);
    if (simple && (boolean || numeric)) {
      Value input = inputs.front();
      if (boolean) {
        Value promoted = allocateTensor(loc, b.getF32Type(), inputShape);
        b.create<dsa::CastOp>(loc, input, promoted); input = promoted;
        kind = kind == BinaryOperator::LogicalOr ? BinaryOperator::Maximum : BinaryOperator::Minimum;
      }
      int64_t capacity = (inputShape[axis].capacity + 31) / 32 * 32;
      Value row = allocate(loc, b.getF32Type(), 1, capacity), scratch = allocateLike(loc, row);
      Value reduced = allocate(loc, b.getF32Type(), 1, 1);
      Value identity = scalarCast(loc, initial.front(), b.getF32Type());
      if (failed(eachElement(loc, resultShape, [&](ValueRange coordinates) {
        SmallVector<Value> sourceCoordinates(coordinates);
        sourceCoordinates.push_back(index(loc, 0));
        auto physical = physicalCoordinates(loc, input, sourceCoordinates);
        Value offset = mul(loc, physical[0], index(loc, inputShape[axis].capacity));
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
        for (Type type : identities.getTypes()) leaves(type, scalarTypes);
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

  LogicalResult advanceScan(ScanOp scan, ValueRange slots, ValueRange next,
                           ValueRange elements, ArrayRef<SmallVector<Value>> captures) {
    ValueRange identities = scan.getInputs().slice(scan.getSourceCount(), scan.getIdentityCount());
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
  LogicalResult scanTensor(ScanOp scan) {
    Location loc = scan.getLoc();
    unsigned axis = scan.getAxis();
    ValueRange sources = scan.getInputs().take_front(scan.getSourceCount());
    ValueRange identities = scan.getInputs().slice(scan.getSourceCount(), scan.getIdentityCount());
    auto inputs = flatten(sources), initial = flatten(identities);
    SmallVector<Type> schema;
    for (Type type : identities.getTypes()) leaves(type, schema);
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
    for (Value capture : scan.getInputs().take_back(scan.getCaptureCount())) {
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

  bool canStreamScan(ScanOp scan, Block &block) {
    unsigned axis = scan.getAxis();
    auto sources = scan.getInputs().take_front(scan.getSourceCount());
    auto identities = scan.getInputs().slice(scan.getSourceCount(), scan.getIdentityCount());
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
        if (!after || !dependent.contains(&op) || llvm::count(ids(store.getInputs()[store.getValueOperandIndex()]), dimension) != 1)
          return false;
        hasOutput = true;
      } else if (auto load = dyn_cast<ViewLoadOp>(op)) {
        if (cast<ViewType>(load.getInputs().front().getType()).getAccess() != 0) return false;
      } else if (!isMemoryEffectFree(&op)) return false;
      if (isa<ScanOp, RegionFoldOp, RegionScanOp, ForOp, WhileOp, IfOp, ParallelOp, ReshapeOp, JoinOp>(op)) return false;
      for (Type type : op.getResultTypes()) {
        SmallVector<Type> fields; leaves(type, fields);
        for (Type field : fields) if (auto tensor = dyn_cast<RankedTensorType>(field))
          if (llvm::count(cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions().asArrayRef(), dimension) > 1) return false;
      }
      if (auto reduce = dyn_cast<ReduceOp>(op))
        for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
          SmallVector<Type> fields; leaves(source.getType(), fields);
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
  LogicalResult streamScan(ScanOp scan, Block &block) {
    Location loc = scan.getLoc();
    unsigned axis = scan.getAxis();
    auto sources = scan.getInputs().take_front(scan.getSourceCount());
    auto identities = scan.getInputs().slice(scan.getSourceCount(), scan.getIdentityCount());
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
    for (Value capture : scan.getInputs().take_back(scan.getCaptureCount())) {
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

  bool singletonAxis(RankedTensorType type, unsigned axis) {
    if (!type.isDynamicDim(axis)) return type.getDimSize(axis) == 1;
    auto ids = cast<TensorShapeAttr>(type.getEncoding()).getDimensions();
    Value size = dimensions.lookup(ids[axis]);
    return size && matchPattern(size, m_One());
  }
  bool equalAxisExtent(RankedTensorType lhs, unsigned a, RankedTensorType rhs, unsigned c) {
    if (!lhs.isDynamicDim(a) && !rhs.isDynamicDim(c)) return lhs.getDimSize(a) == rhs.getDimSize(c);
    auto lhsIds = cast<TensorShapeAttr>(lhs.getEncoding()).getDimensions();
    auto rhsIds = cast<TensorShapeAttr>(rhs.getEncoding()).getDimensions();
    if (lhsIds[a] > 0 && lhsIds[a] == rhsIds[c]) return true;
    Value l = dimensions.lookup(lhsIds[a]), r = dimensions.lookup(rhsIds[c]);
    if (l && r && sameIndex(l, r)) return true;
    APInt constant;
    if (!lhs.isDynamicDim(a) && r && matchPattern(r, m_ConstantInt(&constant))) return constant.getSExtValue() == lhs.getDimSize(a);
    if (!rhs.isDynamicDim(c) && l && matchPattern(l, m_ConstantInt(&constant))) return constant.getSExtValue() == rhs.getDimSize(c);
    return false;
  }
  std::optional<WorksetTiling> planExecutionSlices(Block &block, unsigned selectedAxis = 0,
      ArrayRef<std::pair<Value, unsigned>> sources = {}) {
    WorksetTiling plan;
    plan.axis = selectedAxis;
    DenseSet<Value> written;
    RankedTensorType outputType;
    if (sources.empty()) for (Operation &op : block.without_terminator()) {
      if (op.getNumRegions()) return std::nullopt;
      if (isa<ViewStoreOp, ScatterUniqueOp>(op)) {
        auto relation = analysis.indexRelation(&op);
        Value data = op.getOperand(op.getAttrOfType<IntegerAttr>("value_operand_index").getInt());
        auto type = dyn_cast<RankedTensorType>(data.getType());
        // Distinct writable views are disjoint in this target's launch ABI.
        // Reordering separate writes to one view needs an effect-footprint proof.
        if (failed(relation) || !type || selectedAxis >= type.getRank() || !written.insert(relation->source).second) return std::nullopt;
        if (outputType && !equalAxisExtent(outputType, selectedAxis, type, selectedAxis)) return std::nullopt;
        if (!outputType) outputType = type;
        plan.writes.push_back(&op);
      } else if (auto load = dyn_cast<ViewLoadOp>(op)) {
        if (cast<ViewType>(load.getInputs().front().getType()).getAccess() != 0) return std::nullopt;
      } else if (!isMemoryEffectFree(&op) && !isa<AssumeInBoundsOp>(op)) return std::nullopt;
    }
    if (sources.empty() && (plan.writes.empty() || singletonAxis(outputType, selectedAxis))) return std::nullopt;
    std::function<bool(Value, AxisRequirements)> require;
    std::function<bool(Value)> scalar;
    std::function<bool(Operation *, RankedTensorType, const AxisRequirements &)> access;
    DenseSet<Value> scalarSeen;
    scalar = [&](Value value) -> bool {
      if (auto tensor = dyn_cast<RankedTensorType>(value.getType())) return require(value, AxisRequirements(tensor.getRank(), false));
      if (components(value.getType())) return false;
      if (!scalarSeen.insert(value).second || values.lookupOrNull(value)) return true;
      Operation *op = value.getDefiningOp();
      if (!op || isa<DimOp>(op)) return true;
      if (op->getNumRegions()) return false;
      if (auto load = dyn_cast<ViewLoadOp>(op)) {
        if (cast<ViewType>(load.getInputs().front().getType()).getAccess() != 0) return false;
      } else if (!isMemoryEffectFree(op) && !isa<DomainOp, SubregionOp>(op)) return false;
      return llvm::all_of(op->getOperands(), scalar);
    };
    access = [&](Operation *op, RankedTensorType result, const AxisRequirements &requested) -> bool {
      auto relation = analysis.indexRelation(op);
      if (failed(relation)) return false;
      auto axes = accessAxes(*relation);
      if (axes.rank != requested.size()) return false;
      auto local = dyn_cast<RankedTensorType>(relation->source.getType());
      AxisRequirements sourceAxes(local ? local.getRank() : 0, false);
      for (auto [position, term] : llvm::enumerate(relation->terms)) {
        if (term.kind == 0 && local) sourceAxes[*term.sourceAxis] = requested[axes.terms[position]];
        if (term.kind == 3 && isa<RankedTensorType>(term.operands.front().getType())) {
          Value indices = term.operands.front();
          auto type = cast<RankedTensorType>(indices.getType());
          AxisRequirements indexAxes(type.getRank(), false);
          for (unsigned axis = 0; axis < type.getRank(); ++axis) {
            unsigned mapped = *axes.advancedStart + axes.advancedRank - type.getRank() + axis;
            if (!singletonAxis(type, axis) && requested[mapped]) {
              if (!equalAxisExtent(type, axis, result, mapped)) return false;
              indexAxes[axis] = true;
            }
          }
          if (!require(indices, indexAxes)) return false;
        } else for (Value operand : term.operands) if (!scalar(operand)) return false;
      }
      // Nonlocal gathers retain the complete indexed source axis. Full slices
      // project one result axis directly; inserted axes never consume a source axis.
      if (local && !require(relation->source, sourceAxes)) return false;
      for (StringRef attribute : {"valid_operand_index", "fill_operand_index"})
        if (auto operand = op->getAttrOfType<IntegerAttr>(attribute)) {
          Value value = op->getOperand(operand.getInt());
          if (isa<RankedTensorType>(value.getType())) {
            if (!require(value, requested)) return false;
          } else if (!scalar(value)) return false;
        }
      return true;
    };
    require = [&](Value value, AxisRequirements requested) -> bool {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || requested.size() != type.getRank()) return false;
      for (unsigned axis = 0; axis < requested.size(); ++axis)
        if (singletonAxis(type, axis)) requested[axis] = false;
      auto [entry, inserted] = plan.requirements.try_emplace(value, requested);
      if (!inserted) return entry->second == requested;
      // A pre-existing immutable SSA snapshot can be projected locally.
      if (values.lookupOrNull(value)) return true;
      Operation *op = value.getDefiningOp();
      if (!op || op->getNumRegions()) return false;
      if (isa<ViewLoadOp, GatherOp>(op)) {
        if (auto load = dyn_cast<ViewLoadOp>(op))
          if (cast<ViewType>(load.getInputs().front().getType()).getAccess() != 0) return false;
        return access(op, type, requested);
      }
      if (!isMemoryEffectFree(op)) return false;
      if (isa<IndicesOp, FullOp>(op)) return llvm::all_of(op->getOperands(), scalar);
      if (auto broadcast = dyn_cast<BroadcastOp>(op)) {
        Value input = broadcast->getOperand(0);
        auto source = dyn_cast<RankedTensorType>(input.getType());
        if (!source) return scalar(input);
        if (source.getRank() > type.getRank()) return false;
        AxisRequirements inputAxes(source.getRank(), false);
        unsigned leading = type.getRank() - source.getRank();
        for (unsigned axis = 0; axis < source.getRank(); ++axis)
          if (requested[leading + axis] && !singletonAxis(source, axis)) {
            if (!equalAxisExtent(source, axis, type, leading + axis)) return false;
            inputAxes[axis] = true;
          }
        return require(input, inputAxes);
      }
      if (auto transpose = dyn_cast<TransposeOp>(op)) {
        AxisRequirements inputAxes(type.getRank(), false);
        for (auto [axis, perm] : llvm::enumerate(transpose.getPermutation())) inputAxes[cast<IntegerAttr>(perm).getInt()] = requested[axis];
        return require(transpose.getInput(), inputAxes);
      }
      if (auto matrix = dyn_cast<ContractOp>(op)) {
        auto lhs = cast<RankedTensorType>(matrix.getLhs().getType()), rhs = cast<RankedTensorType>(matrix.getRhs().getType());
        AxisRequirements l(lhs.getRank(), false), r(rhs.getRank(), false);
        DenseSet<unsigned> leftReduced, rightReduced, rightBatched;
        DenseMap<unsigned, unsigned> batches;
        for (Attribute entry : matrix.getReduce()) {
          auto pair = cast<ArrayAttr>(entry);
          leftReduced.insert(cast<IntegerAttr>(pair[0]).getInt()); rightReduced.insert(cast<IntegerAttr>(pair[1]).getInt());
        }
        for (Attribute entry : matrix.getBatch()) {
          auto pair = cast<ArrayAttr>(entry);
          unsigned a = cast<IntegerAttr>(pair[0]).getInt(), c = cast<IntegerAttr>(pair[1]).getInt();
          batches[a] = c; rightBatched.insert(c);
        }
        unsigned resultAxis = 0;
        for (unsigned axis = 0; axis < l.size(); ++axis) if (!leftReduced.contains(axis)) {
          l[axis] = requested[resultAxis++];
          if (batches.count(axis)) r[batches.lookup(axis)] = l[axis];
        }
        for (unsigned axis = 0; axis < r.size(); ++axis)
          if (!rightReduced.contains(axis) && !rightBatched.contains(axis)) r[axis] = requested[resultAxis++];
        return resultAxis == requested.size() && require(matrix.getLhs(), l) && require(matrix.getRhs(), r);
      }
      if (!isa<UnaryOp, BinaryOp, CompareOp, SelectOp, MaskOp, CastOp>(op)) return false;
      for (Value input : op->getOperands()) {
        if (isa<RankedTensorType>(input.getType())) {
          if (!require(input, requested)) return false;
        } else if (!scalar(input)) return false;
      }
      return true;
    };
    for (Operation *write : plan.writes) {
      Value data = write->getOperand(write->getAttrOfType<IntegerAttr>("value_operand_index").getInt());
      auto type = cast<RankedTensorType>(data.getType());
      AxisRequirements requested(type.getRank(), false); requested[selectedAxis] = true;
      if (!require(data, requested) || !access(write, type, requested)) return std::nullopt;
    }
    for (auto [source, axis] : sources) {
      auto type = dyn_cast<RankedTensorType>(source.getType());
      if (!type || axis >= type.getRank()) return std::nullopt;
      AxisRequirements requested(type.getRank(), false); requested[axis] = true;
      if (!require(source, requested)) return std::nullopt;
    }
    if (sources.empty()) for (Operation &op : block.without_terminator()) {
      if (canDefer(&op) || isa<ViewStoreOp, ScatterUniqueOp, DimOp, AssumeInBoundsOp>(op)) continue;
      for (Value result : op.getResults()) if (!scalar(result)) return std::nullopt;
    }
    return plan;
  }
  LogicalResult lowerTiledWorkset(Block &block, const WorksetTiling &plan, bool prepared = false) {
    Location loc = block.getParentOp()->getLoc();
    // Scalar bounds and readonly scalar inputs keep their original evaluation
    // order. Tensor producers remain lazy until their selected slice is needed.
    if (!prepared) for (Operation &op : block.without_terminator()) {
      if (canDefer(&op) || isa<ViewStoreOp, ScatterUniqueOp>(op)) continue;
      if (failed(lowerOperation(&op))) return failure();
    }
    DenseMap<Value, LocalShape> complete;
    Value size;
    int64_t capacity = plan.axis == 0 ? config.getTileM() : config.getTileN();
    for (auto &entry : plan.requirements) {
      auto shape = localShape(entry.first, loc);
      if (failed(shape)) return failure();
      for (unsigned axis = 0; axis < shape->size(); ++axis) if (entry.second[axis]) {
        const auto &local = (*shape)[axis];
        if (!matchPattern(local.begin, m_Zero()) || !sameIndex(local.extent, local.count))
          return emitError(loc, "DSA workset tiling requires an available complete result axis");
        if (size && !sameIndex(size, local.count)) return emitError(loc, "DSA coordinated tile axes have different extents");
        size = local.count;
        capacity = std::min(capacity, local.capacity);
      }
      complete[entry.first] = *shape;
    }
    if (!size) return emitError(loc, "DSA workset tiling has no selected result axis");
    auto savedValues = values; auto savedProducts = products; auto savedSlices = valueSlices;
    LogicalResult status = loop(loc, index(loc, 0), size, index(loc, capacity), [&](Value begin) {
      Value count = b.create<arith::MinSIOp>(loc, sub(loc, size, begin), index(loc, capacity));
      for (auto &entry : plan.requirements) {
        LocalShape shape = complete.lookup(entry.first);
        for (unsigned axis = 0; axis < shape.size(); ++axis)
          if (entry.second[axis]) shape[axis] = {shape[axis].extent, begin, count, capacity};
        valueSlices[entry.first] = std::move(shape);
      }
      if (auto nested = planExecutionSlices(block, plan.axis + 1)) return lowerTiledWorkset(block, *nested, true);
      for (Operation *write : plan.writes) if (failed(lowerOperation(write))) return failure();
      return success();
    });
    values = std::move(savedValues); products = std::move(savedProducts); valueSlices = std::move(savedSlices);
    return status;
  }

  LogicalResult lowerStructuredBlock(Block &block) {
    for (Operation &op : block.without_terminator())
      if (auto scan = dyn_cast<ScanOp>(op); scan && canStreamScan(scan, block)) return streamScan(scan, block);
    RegionFoldOp fold;
    for (Operation &op : block.without_terminator()) if (auto region = dyn_cast<RegionFoldOp>(op)) {
      if (fold) return op.emitError("DSA multiple region folds in one workset need an explicit shared projection");
      fold = region;
    }
    if (!fold) {
      if (auto plan = planExecutionSlices(block)) return lowerTiledWorkset(block, *plan);
      return lowerOperations(block);
    }
    auto sourceType = cast<RankedTensorType>(fold.getInputs().front().getType());
    auto sourceIds = cast<TensorShapeAttr>(sourceType.getEncoding()).getDimensions();
    int64_t query = 0;
    for (Operation &op : block.without_terminator()) if (auto store = dyn_cast<ViewStoreOp>(op)) {
      auto type = dyn_cast<RankedTensorType>(store.getInputs()[store.getValueOperandIndex()].getType());
      if (!type || !type.getRank()) continue;
      int64_t candidate = cast<TensorShapeAttr>(type.getEncoding()).getDimensions()[0];
      if (candidate > 0 && !llvm::is_contained(sourceIds.asArrayRef(), candidate)) {
        if (query && query != candidate) return store.emitError("DSA fold outputs require different work ownership projections");
        query = candidate;
      }
    }
    if (!query || axisBindings.count(query)) return lowerOperations(block);
    // Slicing is legal only when this axis stays free throughout the author's
    // helpers. Reducing it or indexing it nonlocally requires another plan.
    bool independent = true;
    block.walk([&](Operation *op) {
      auto ids = [&](Value value) -> ArrayRef<int64_t> {
        auto type = dyn_cast<RankedTensorType>(value.getType());
        return type ? cast<TensorShapeAttr>(type.getEncoding()).getDimensions().asArrayRef() : ArrayRef<int64_t>();
      };
      if (isa<ForOp, WhileOp, IfOp, ScanOp, BufferLoadOp, BufferStoreOp>(op)) independent = false;
      for (Type result : op->getResultTypes()) {
        SmallVector<Type> fields; leaves(result, fields);
        for (Type field : fields) if (auto tensor = dyn_cast<RankedTensorType>(field))
          if (llvm::count(cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions().asArrayRef(), query) > 1) independent = false;
      }
      if (auto store = dyn_cast<ViewStoreOp>(op))
        if (llvm::count(ids(store.getInputs()[store.getValueOperandIndex()]), query) != 1) independent = false;
      if (isa<RegionFoldOp, RegionScanOp>(op) && analysis.regionSegment(op).dimensionIdentity == query) independent = false;
      if (auto load = dyn_cast<ViewLoadOp>(op))
        if (cast<ViewType>(load.getInputs().front().getType()).getAccess() != 0) independent = false;
      if (auto reduce = dyn_cast<ReduceOp>(op))
        for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
          SmallVector<Type> fields; leaves(source.getType(), fields);
          for (Type field : fields) if (auto tensor = dyn_cast<RankedTensorType>(field))
            for (Attribute axis : reduce.getAxes())
              if (cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions()[cast<IntegerAttr>(axis).getInt()] == query) independent = false;
        }
      if (auto matrix = dyn_cast<ContractOp>(op)) {
        for (Attribute entry : matrix.getReduce()) {
          auto pair = cast<ArrayAttr>(entry);
          independent &= ids(matrix.getLhs())[cast<IntegerAttr>(pair[0]).getInt()] != query;
          independent &= ids(matrix.getRhs())[cast<IntegerAttr>(pair[1]).getInt()] != query;
        }
        if (!matrix.getBatch().empty()) independent = false;
      }
      if (isa<GatherOp, ViewLoadOp, ViewStoreOp>(op)) {
        auto fact = analysis.indexRelation(op);
        if (failed(fact)) { independent = false; return; }
        auto sourceType = dyn_cast<RankedTensorType>(fact->source.getType());
        if (!sourceType) return;
        auto dims = ids(fact->source);
        for (const auto &term : fact->terms)
          if (term.sourceAxis && dims[*term.sourceAxis] == query && term.kind != 0) independent = false;
      }
      if (isa<ReshapeOp, JoinOp>(op)) independent = false;
    });
    if (!independent) return fold.emitError("DSA query partition requires an axis preserved by the region value graph");
    Value size = dimensions.lookup(query);
    if (!size) return fold.emitError("DSA query extent is unavailable");
    auto savedValues = values; auto savedProducts = products;
    auto savedDimensions = dimensions; auto savedAxes = axisBindings;
    auto status = loop(fold.getLoc(), index(fold.getLoc(), 0), size, index(fold.getLoc(), config.getTileM()), [&](Value begin) {
      Value count = b.create<arith::MinSIOp>(fold.getLoc(), sub(fold.getLoc(), size, begin), index(fold.getLoc(), config.getTileM()));
      axisBindings[query] = {size, begin, count, config.getTileM()};
      return lowerOperations(block);
    });
    values = std::move(savedValues); products = std::move(savedProducts);
    dimensions = std::move(savedDimensions); axisBindings = std::move(savedAxes);
    return status;
  }

  bool replayableSlice(Value value, int64_t dimension, DenseSet<Value> &visited) {
    if (!visited.insert(value).second) return true;
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor && !components(value.getType())) return true;
    Operation *op = value.getDefiningOp();
    if (!op) return false;
    if (auto load = dyn_cast<ViewLoadOp>(op)) {
      if (cast<ViewType>(load.getInputs().front().getType()).getAccess() != 0) return false;
    } else if (!isMemoryEffectFree(op)) return false;
    if (isa<RegionFoldOp, RegionScanOp, ScanOp, ReshapeOp, JoinOp>(op)) return false;
    auto containsAxis = [&](Value input, unsigned axis) {
      SmallVector<Type> fields; leaves(input.getType(), fields);
      for (Type field : fields) if (auto tensor = dyn_cast<RankedTensorType>(field))
        if (cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions()[axis] == dimension) return true;
      return false;
    };
    if (auto reduce = dyn_cast<ReduceOp>(op))
      for (Value input : reduce.getInputs().take_front(reduce.getSourceCount()))
        for (Attribute axis : reduce.getAxes()) if (containsAxis(input, cast<IntegerAttr>(axis).getInt())) return false;
    if (auto matrix = dyn_cast<ContractOp>(op)) {
      if (!matrix.getBatch().empty()) return false;
      for (Attribute entry : matrix.getReduce()) {
        auto pair = cast<ArrayAttr>(entry);
        if (containsAxis(matrix.getLhs(), cast<IntegerAttr>(pair[0]).getInt()) ||
            containsAxis(matrix.getRhs(), cast<IntegerAttr>(pair[1]).getInt())) return false;
      }
    }
    if (auto gather = dyn_cast<GatherOp>(op)) {
      auto fact = analysis.indexRelation(gather);
      if (failed(fact)) return false;
      for (const auto &term : fact->terms)
        if (term.sourceAxis && containsAxis(fact->source, *term.sourceAxis) && term.kind != 0) return false;
    }
    for (Value operand : op->getOperands()) if (!replayableSlice(operand, dimension, visited)) return false;
    return true;
  }
  void forgetReplayedTensors(Value value, DenseSet<Value> &visited) {
    if (!visited.insert(value).second || (!isa<RankedTensorType>(value.getType()) && !components(value.getType()))) return;
    Operation *op = value.getDefiningOp();
    if (!op) return;
    values.erase(value); products.erase(value);
    for (Value operand : op->getOperands()) forgetReplayedTensors(operand, visited);
  }

  FailureOr<SmallVector<Operation *>> regionConsumers(Operation *region, unsigned outputCount, int64_t dimension) {
    DenseSet<Operation *> consumers;
    SmallVector<Value> pending(region->getResults().take_front(outputCount));
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      for (Operation *user : value.getUsers()) {
        if (user->getBlock() != region->getBlock() || user == region->getBlock()->getTerminator())
          return region->emitError("DSA streamed region outputs require consumers in the same workset"), failure();
        if (!consumers.insert(user).second) continue;
        bool write = isa<ViewStoreOp, ScatterUniqueOp>(user);
        if (!write && (!isMemoryEffectFree(user) || user->getNumRegions()))
          return user->emitError("DSA streamed region consumer requires a pure position-preserving computation"), failure();
        if (isa<ReshapeOp, JoinOp>(user))
          return user->emitError("DSA streamed region consumer needs an explicit position mapping"), failure();
        for (Type type : user->getResultTypes()) {
          SmallVector<Type> fields; leaves(type, fields);
          for (Type field : fields) if (auto tensor = dyn_cast<RankedTensorType>(field))
            if (llvm::count(cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions().asArrayRef(), dimension) > 1)
              return user->emitError("DSA streamed region consumer couples distinct source positions"), failure();
        }
        if (auto matrix = dyn_cast<ContractOp>(user)) {
          for (Attribute entry : matrix.getReduce()) {
            auto pair = cast<ArrayAttr>(entry);
            for (auto [input, axis] : llvm::zip(matrix->getOperands(), pair)) {
              auto ids = cast<TensorShapeAttr>(cast<RankedTensorType>(input.getType()).getEncoding()).getDimensions();
              if (ids[cast<IntegerAttr>(axis).getInt()] == dimension)
                return user->emitError("DSA streamed region consumer reduces the source-position axis"), failure();
            }
          }
        }
        if (auto gather = dyn_cast<GatherOp>(user)) {
          auto relation = analysis.indexRelation(gather);
          if (failed(relation)) return failure();
          auto ids = cast<TensorShapeAttr>(cast<RankedTensorType>(relation->source.getType()).getEncoding()).getDimensions();
          for (const auto &term : relation->terms)
            if (term.sourceAxis && ids[*term.sourceAxis] == dimension && term.kind != 0)
              return user->emitError("DSA streamed region consumer performs a nonlocal lookup"), failure();
        }
        if (write) {
          auto data = user->getOperand(user->getAttrOfType<IntegerAttr>("value_operand_index").getInt());
          auto type = dyn_cast<RankedTensorType>(data.getType());
          if (!type || llvm::count(cast<TensorShapeAttr>(type.getEncoding()).getDimensions().asArrayRef(), dimension) != 1)
            return user->emitError("DSA streamed region output must preserve one source-position axis"), failure();
        }
        llvm::append_range(pending, user->getResults());
      }
    }
    SmallVector<Operation *> ordered;
    DenseSet<Value> checkedFinal;
    std::function<bool(Value)> requiresFinal = [&](Value value) {
      if (value.getDefiningOp() == region) return cast<OpResult>(value).getResultNumber() >= outputCount;
      if (!checkedFinal.insert(value).second) return false;
      Operation *definition = value.getDefiningOp();
      return definition && llvm::any_of(definition->getOperands(), requiresFinal);
    };
    for (Operation *op = region->getNextNode(); op && op != region->getBlock()->getTerminator(); op = op->getNextNode()) {
      if (consumers.contains(op)) {
        for (Value operand : op->getOperands()) {
          if (requiresFinal(operand))
            return op->emitError("DSA region output consumer requires the final state before traversal completes"), failure();
          if (operand.getDefiningOp() == region) {
            continue;
          }
          if (consumers.contains(operand.getDefiningOp())) continue;
          DenseSet<Value> visited;
          if (!replayableSlice(operand, dimension, visited))
            return op->emitError("DSA region consumer input needs complete materialization before streaming"), failure();
        }
        ordered.push_back(op);
      }
    }
    if (!ordered.empty()) {
      for (Operation *between = region->getNextNode(); between != ordered.back(); between = between->getNextNode()) {
        if (consumers.contains(between) || isMemoryEffectFree(between)) continue;
        if (auto load = dyn_cast<ViewLoadOp>(between))
          if (cast<ViewType>(load.getInputs().front().getType()).getAccess() == 0) continue;
        return between->emitError("DSA streaming output cannot cross an observable access"), failure();
      }
    }
    return ordered;
  }
  LogicalResult lowerRegion(Operation *op) {
    Location loc = op->getLoc();
    bool scan = isa<RegionScanOp>(op);
    unsigned sourceCount = op->getAttrOfType<IntegerAttr>("source_count").getInt();
    unsigned identityCount = op->getAttrOfType<IntegerAttr>("identity_count").getInt();
    unsigned stateCount = scan ? op->getAttrOfType<IntegerAttr>("state_count").getInt() : 0;
    unsigned captureCount = op->getAttrOfType<IntegerAttr>("capture_count").getInt();
    unsigned outputCount = scan ? op->getAttrOfType<IntegerAttr>("output_count").getInt() : 0;
    int64_t axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
    ValueRange sources = op->getOperands().take_front(sourceCount);
    ValueRange identities = op->getOperands().slice(sourceCount, identityCount);
    ValueRange states = op->getOperands().slice(sourceCount + identityCount, stateCount);
    ValueRange captures = op->getOperands().take_back(captureCount);
    auto fact = analysis.regionSegment(op);
    if (!fact.isExact()) return op->emitError("DSA region source has no exact canonical segment relation");
    auto sourceType = cast<RankedTensorType>(sources.front().getType());
    int64_t dimension = cast<TensorShapeAttr>(sourceType.getEncoding()).getDimensions()[axis];
    for (Value source : sources) {
      auto type = cast<RankedTensorType>(source.getType());
      auto ids = cast<TensorShapeAttr>(type.getEncoding()).getDimensions();
      DenseSet<Value> visited;
      if (llvm::count(ids.asArrayRef(), dimension) != 1 || !replayableSlice(source, dimension, visited))
        return op->emitError("DSA source slicing requires an independent axis and immutable replayable inputs; this source needs explicit snapshot materialization");
    }
    Value size = extent(sourceType, axis, loc);
    if (!size || axisBindings.count(dimension)) return op->emitError("DSA region source requires an unpartitioned logical traversal");
    auto initial = flatten(identities), initialState = flatten(states);
    auto slots = makeSlots(identities.getTypes(), loc), next = makeSlots(identities.getTypes(), loc);
    if (failed(slots) || failed(next) || initial.size() != slots->size()) return failure();
    for (auto [value, slot] : llvm::zip(initial, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
    SmallVector<SmallVector<Value>> captureFields;
    for (Value capture : captures) captureFields.push_back(flatten(ValueRange{capture}));
    auto consumers = regionConsumers(op, outputCount, dimension);
    if (failed(consumers)) return failure();
    auto savedValues = values; auto savedProducts = products; auto savedAxes = axisBindings;
    if (failed(loop(loc, index(loc, 0), size, index(loc, config.getRegionTile()), [&](Value begin) -> LogicalResult {
      Value count = b.create<arith::MinSIOp>(loc, sub(loc, size, begin), index(loc, config.getRegionTile()));
      axisBindings[dimension] = {size, begin, count, config.getRegionTile()};
      SmallVector<SmallVector<Value>> slices;
      DenseSet<Value> replayed;
      for (Value source : sources) forgetReplayedTensors(source, replayed);
      for (Value source : sources) {
        // Source expressions are replayed from immutable views in this slice.
        auto fields = flatten(ValueRange{source});
        if (fields.empty() || llvm::is_contained(fields, Value())) return failure();
        slices.push_back(std::move(fields));
      }
      auto arguments = slices; llvm::append_range(arguments, captureFields);
      auto part = helper(op->getRegion(0).front(), arguments, sourceCount, axis);
      if (failed(part)) return failure();
      if (scan) {
        auto applied = splitFields(identities.getTypes(), *slots);
        llvm::append_range(applied, splitFields(states.getTypes(), initialState));
        auto incoming = helper(op->getRegion(2).front(), applied);
        if (failed(incoming)) return failure();
        auto emittedArgs = slices;
        llvm::append_range(emittedArgs, splitFields(states.getTypes(), *incoming));
        llvm::append_range(emittedArgs, captureFields);
        auto emitted = helper(op->getRegion(3).front(), emittedArgs, sourceCount, axis);
        SmallVector<Type> outputTypes;
        ValueRange outputValues = op->getResults().take_front(outputCount);
        for (Type type : outputValues.getTypes()) leaves(type, outputTypes);
        if (failed(emitted) || emitted->size() != outputTypes.size()) return op->emitError("DSA scan output schema is unavailable");
        for (unsigned i = 0; i < outputTypes.size(); ++i) {
          // Reattach the helper-local slice to the complete output's source
          // coordinates before its view consumers form destination addresses.
          auto outputType = cast<RankedTensorType>(outputTypes[i]);
          auto outputIds = cast<TensorShapeAttr>(outputType.getEncoding()).getDimensions();
          LocalShape outputShape = localShapes.lookup((*emitted)[i]);
          for (int64_t outputAxis = 0; outputAxis < outputType.getRank(); ++outputAxis)
            if (outputIds[outputAxis] == dimension) outputShape[outputAxis] = {size, begin, count, config.getRegionTile()};
          localShapes[(*emitted)[i]] = std::move(outputShape);
        }
        auto outputFields = splitFields(outputValues.getTypes(), *emitted);
        for (auto [result, fields] : llvm::zip(outputValues, outputFields)) bindProduct(result, fields);
        for (Operation *consumer : *consumers)
          if (!canDefer(consumer) && failed(lowerOperation(consumer))) return failure();
      }
      auto combinedArgs = splitFields(identities.getTypes(), *slots);
      llvm::append_range(combinedArgs, splitFields(identities.getTypes(), *part));
      auto updated = helper(op->getRegion(1).front(), combinedArgs);
      if (failed(updated) || updated->size() != slots->size()) return failure();
      for (auto [value, slot] : llvm::zip(*updated, *next)) if (failed(copyTo(value, slot, loc))) return failure();
      for (auto [value, slot] : llvm::zip(*next, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
      return success();
    }))) return failure();
    values = std::move(savedValues); products = std::move(savedProducts); axisBindings = std::move(savedAxes);
    if (!scan) { bindSlots(op->getResults(), *slots, loc); return success(); }
    auto applied = splitFields(identities.getTypes(), *slots);
    llvm::append_range(applied, splitFields(states.getTypes(), initialState));
    auto finalState = helper(op->getRegion(2).front(), applied);
    if (failed(finalState)) return failure();
    auto grouped = splitFields(states.getTypes(), *finalState);
    for (auto [result, fields] : llvm::zip(op->getResults().drop_front(outputCount), grouped)) bindProduct(result, fields);
    for (Operation *consumer : *consumers) streamedOperations.insert(consumer);
    return success();
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
  bool bindDomain(Value value) {
    if (domains.count(value)) return true;
    Operation *definition = value.getDefiningOp();
    return definition && isa<DomainOp, SubregionOp>(definition) && succeeded(lowerOperation(definition));
  }
  // Bounds constrain the author's logical interval. They size local storage
  // without replacing the runtime extent or the source coordinate origin.
  std::optional<int64_t> upperDistance(Value end, Value begin) {
    if (sameIndex(end, begin)) return 0;
    APInt a, c;
    if (matchPattern(end, m_ConstantInt(&a)) && matchPattern(begin, m_ConstantInt(&c))) {
      APInt distance = a.sext(128) - c.sext(128);
      if (distance.isSignedIntN(64)) return std::max<int64_t>(0, distance.getSExtValue());
    }
    if (auto sum = end.getDefiningOp<arith::AddIOp>()) {
      Value extra;
      if (sameIndex(sum.getLhs(), begin)) extra = sum.getRhs();
      if (sameIndex(sum.getRhs(), begin)) extra = sum.getLhs();
      if (extra && matchPattern(extra, m_ConstantInt(&a)) && !a.isNegative()) return a.getSExtValue();
    }
    auto tighter = [](std::optional<int64_t> lhs, std::optional<int64_t> rhs) {
      return lhs && rhs ? std::optional<int64_t>(std::min(*lhs, *rhs)) : lhs ? lhs : rhs;
    };
    if (auto minimum = end.getDefiningOp<arith::MinSIOp>())
      return tighter(upperDistance(minimum.getLhs(), begin), upperDistance(minimum.getRhs(), begin));
    if (auto maximum = begin.getDefiningOp<arith::MaxSIOp>())
      return tighter(upperDistance(end, maximum.getLhs()), upperDistance(end, maximum.getRhs()));
    return std::nullopt;
  }
  bool independentDomains(const LogicalWorksetFact &workset) {
    DenseSet<Value> seen;
    std::function<bool(Value)> varies = [&](Value value) {
      if (llvm::is_contained(workset.coordinates, value)) return true;
      if (!seen.insert(value).second) return false;
      Operation *op = value.getDefiningOp();
      return op && llvm::any_of(op->getOperands(), varies);
    };
    for (Value domain : workset.domains) if (varies(domain)) return false;
    Operation *child = workset.parallel;
    while (Operation *parent = child->getParentOp()) {
      if (!isa<ParallelOp>(parent)) break;
      for (Operation &op : cast<ParallelOp>(parent).getBody().front().without_terminator()) {
        if (&op == child) break;
        if (auto load = dyn_cast<ViewLoadOp>(op)) {
          if (cast<ViewType>(load.getInputs().front().getType()).getAccess() == 0) continue;
        }
        if (!isMemoryEffectFree(&op) && !isa<AssumeInBoundsOp>(op)) return false;
      }
      child = parent;
    }
    return true;
  }
  LogicalResult distributeWorkset(Location loc) {
    const auto &workset = *distributedWorkset;
    SmallVector<Domain> intervals;
    SmallVector<Value> counts;
    Value total = index(loc, 1);
    for (Value source : workset.domains) {
      if (!bindDomain(source)) return emitError(loc, "DSA independent task domain is unavailable");
      Domain domain = domains.lookup(source);
      Value count = b.createOrFold<arith::MaxSIOp>(loc, sub(loc, domain.end, domain.begin), index(loc, 0));
      counts.push_back(count); intervals.push_back(domain); total = mul(loc, total, count);
    }
    if (matchPattern(total, m_Zero())) return success();
    auto savedValues = values; auto savedProducts = products; auto savedDimensions = dimensions;
    ++parallelDepth;
    LogicalResult status = loop(loc, taskId, total, taskCount, [&](Value task) {
      Value remaining = task;
      for (int64_t axis = intervals.size() - 1; axis >= 0; --axis) {
        Value coordinate = axis ? Value(b.create<arith::RemSIOp>(loc, remaining, counts[axis])) : remaining;
        values.map(workset.coordinates[axis], add(loc, intervals[axis].begin, coordinate));
        if (axis) remaining = b.create<arith::DivSIOp>(loc, remaining, counts[axis]);
      }
      return lowerBlock(*workset.body);
    });
    --parallelDepth;
    values = std::move(savedValues); products = std::move(savedProducts); dimensions = std::move(savedDimensions);
    return status;
  }
  Value scalarCast(Location loc, Value value, Type type) {
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
      if (components(input.getType())) {
        if (structured && !products.count(input) && input.getDefiningOp() && failed(materialize(input.getDefiningOp()))) return {};
        auto fields = products.lookup(input);
        if (structured) {
          SmallVector<Type> schema; leaves(input.getType(), schema);
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
          if (structured) {
            auto shape = localShape(tensor, loc);
            if (failed(shape)) return failure();
            slots.push_back(allocateTensor(loc, tensor.getElementType(), *shape));
          } else if (tensor.getRank() == 1 && activeCount) slots.push_back(allocate(loc, tensor.getElementType(), 1, config.getTile()));
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
    auto savedAxes = axisBindings;
    auto savedValues = values;
    auto savedProducts = products;
    auto savedStreamed = streamedOperations;
    auto restore = [&]() {
      dimensions = savedDimensions; axisBindings = savedAxes;
      values = savedValues; products = savedProducts;
      streamedOperations = savedStreamed;
    };
    auto slots = makeSlots(operation->getResultTypes(), loc);
    if (failed(slots)) return failure();
    if (auto conditional = dyn_cast<IfOp>(operation)) {
      auto target = b.create<scf::IfOp>(loc, get(conditional.getCondition()), true);
      for (auto [source, destination] : llvm::zip(operation->getRegions(), target->getRegions())) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&destination.front());
        if (failed(lowerResults(source.front(), *slots))) return failure();
        restore();
      }
    } else {
      auto forLoop = dyn_cast<ForOp>(operation);
      auto initial = flatten(forLoop ? operation->getOperands().drop_front() : operation->getOperands());
      if (initial.size() != slots->size()) return operation->emitError("DSA initial and result state schemas differ");
      auto next = makeSlots(operation->getResultTypes(), loc);
      if (failed(next)) return failure();
      for (auto [value, slot] : llvm::zip(initial, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
      savedValues = values; savedProducts = products;
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
        restore();
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
    restore();
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
    if (structured) return lowerStructuredBlock(block);
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
    for (Operation &op : block.without_terminator()) {
      if (structured && canDefer(&op)) continue;
      if (failed(lowerOperation(&op))) return failure();
    }
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
    if (streamedOperations.contains(operation)) return success();
    Location loc = operation->getLoc();
    if (structured) {
      if (isa<RegionFoldOp, RegionScanOp>(operation)) return lowerRegion(operation);
      if (auto reduce = dyn_cast<ReduceOp>(operation)) return reduceTensor(reduce);
      if (auto scan = dyn_cast<ScanOp>(operation)) return scanTensor(scan);
      if (isa<ViewLoadOp, ViewStoreOp, GatherOp, ScatterUniqueOp>(operation)) return tensorAccess(operation);
      if (operation->getNumResults() == 1 && isa<RankedTensorType>(operation->getResult(0).getType()) &&
          !isa<ExtractOp, IfOp, ForOp, WhileOp, ScanOp>(operation)) return tensorOperation(operation);
    }
    if (isa<IfOp, ForOp, WhileOp>(operation)) return orderedControl(operation);
    if (auto scan = dyn_cast<ScanOp>(operation)) return scanRow(scan);
    if (isa<MakeTupleOp, MakeRecordOp>(operation)) {
      products[operation->getResult(0)] = flatten(operation->getOperands());
      return success();
    }
    if (auto extract = dyn_cast<ExtractOp>(operation)) {
      if (structured && !products.count(extract.getProduct())) {
        auto def = extract.getProduct().getDefiningOp();
        if (!def || failed(materialize(def))) return extract.emitError("DSA record has no bound producer");
      }
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
      domains[domain.getResult()] = {begin, end, step, identity, upperDistance(end, begin)};
      if (identity > 0) dimensions[identity] = b.createOrFold<arith::MaxSIOp>(loc, sub(loc, end, begin), index(loc, 0));
      return success();
    }
    if (auto region = dyn_cast<SubregionOp>(operation)) {
      if (!bindDomain(region.getInputs().front()) || region.getExtentDimensions().size() != 1)
        return region.emitError("DSA subregion needs one bound source interval");
      Domain source = domains.lookup(region.getInputs().front());
      unsigned operand = 1;
      Value begin = region.getHasStart() ? asIndex(get(region.getInputs()[operand++]), loc) : source.begin;
      Value end = region.getHasStop() ? asIndex(get(region.getInputs()[operand]), loc) : source.end;
      if (!begin || !end) return region.emitError("DSA subregion bounds are unavailable");
      int64_t dimension = cast<IntegerAttr>(region.getExtentDimensions()[0]).getInt();
      auto capacity = upperDistance(end, begin);
      if (source.capacity) capacity = capacity ? std::min(*capacity, *source.capacity) : source.capacity;
      domains[region.getResult()] = {begin, end, source.step, dimension, capacity};
      Value count = sub(loc, end, begin);
      dimensions[dimension] = count;
      if (capacity) axisBindings[dimension] = {count, index(loc, 0), count, std::max<int64_t>(1, *capacity)};
      return success();
    }
    if (auto end = dyn_cast<RegionEndOp>(operation)) {
      if (!bindDomain(end.getSource())) return end.emitError("DSA region end needs a bound source interval");
      values.map(end.getResult(), domains.lookup(end.getSource()).end); return success();
    }
    if (auto parallel = dyn_cast<ParallelOp>(operation)) {
      if (parallel == distributedRoot) return distributeWorkset(loc);
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
      Value output;
      if (matrixResultType && type.getShape() == matrixResultType.getShape() && type.getEncoding() == matrixResultType.getEncoding())
        output = allocateLike(loc, matrixResultTile, type.getElementType());
      else if (type.getRank() == 1 && activeCount) output = allocate(loc, type.getElementType(), 1, config.getTile());
      else return operation->emitError("DSA broadcast requires a bound tensor tile");
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
      if (binary.getOperatorKind() == BinaryOperator::TrueDivide && binary.getApproximate() && lhs.getType().isF32()) {
        Value left = allocate(loc, b.getF32Type(), 1, 1), right = allocateLike(loc, left), output = allocateLike(loc, left);
        b.create<dsa::FillOp>(loc, left, lhs); b.create<dsa::FillOp>(loc, right, rhs);
        b.create<dsa::BinaryOp>(loc, left, right, output, binary.getOperatorKindAttr(), binary.getApproximateAttr(), binary.getFlushToZeroAttr());
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
    if (auto mask = dyn_cast<MaskOp>(operation)) {
      Value condition = get(mask.getPredicate()), value = get(mask.getValue()), fill = get(mask.getFill());
      if (!condition || !value || !fill || !condition.getType().isInteger(1)) return mask.emitError("DSA mask needs a scalar predicate");
      values.map(mask.getResult(), b.create<arith::SelectOp>(loc, condition, value, fill));
      return success();
    }
    if (auto unary = dyn_cast<UnaryOp>(operation)) {
      Value input = get(unary.getInput());
      if (!input) return unary.emitError("DSA unary operand is unavailable");
      if (!isa<MemRefType>(input.getType())) {
        if (unary.getOperatorKind() == UnaryOperator::Exp2 && unary.getApproximate() && input.getType().isF32()) {
          Value source = allocate(loc, b.getF32Type(), 1, 1), output = allocateLike(loc, source);
          b.create<dsa::FillOp>(loc, source, input);
          b.create<dsa::UnaryOp>(loc, source, output, unary.getOperatorKindAttr(), unary.getApproximateAttr(), unary.getFlushToZeroAttr());
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
      Value output = allocateLike(loc, input);
      b.create<dsa::UnaryOp>(loc, input, output, unary.getOperatorKindAttr(), unary.getApproximateAttr(), unary.getFlushToZeroAttr());
      values.map(unary.getResult(), output); return success();
    }
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      if (reduce.getSourceCount() != 1 || reduce.getIdentityCount() != 1 || reduce.getCaptureCount() ||
          reduce.getAxes().size() != 1 || cast<IntegerAttr>(reduce.getAxes()[0]).getInt() != 0 ||
          reduce->getNumResults() != 1 || !reduce->getResult(0).getType().isF32())
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
      else if (!isa<ViewLoadOp>(op) && failed(lowerOperation(&op))) return failure();
    }
    ViewStoreOp output;
    for (Operation *op : epilogue) {
      if (auto store = dyn_cast<ViewStoreOp>(op)) {
        if (output) return store.emitError("DSA matrix epilogue currently requires one output");
        output = store;
      } else if (!isa<ConstantOp, DimOp, BroadcastOp, FullOp, CastOp, UnaryOp, BinaryOp, SelectOp>(op))
        return op->emitError("DSA matrix epilogue requires local pointwise computation and a full output store");
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
        b.create<dsa::MatMulOp>(loc, lhs, rhs, accumulator, rows, depth, cols);
        return success();
      }))) return failure();
      values.map(matrix.getResult(), accumulator);
      auto previousType = matrixResultType;
      auto previousTile = matrixResultTile;
      matrixResultType = cast<RankedTensorType>(matrix.getResult().getType());
      matrixResultTile = accumulator;
      for (Operation *op : epilogue) {
        if (!isa<ViewStoreOp>(op)) { if (failed(lowerOperation(op))) return failure(); }
        else {
          Value result = get(output.getInputs()[output.getValueOperandIndex()]);
          if (!result) return output.emitError("DSA matrix result is unavailable");
          Value rowStride = stride(loc, destination, 0), colStride = stride(loc, destination, 1);
          b.create<dsa::StoreTileOp>(loc, result, destination,
              add(loc, mul(loc, m0, rowStride), mul(loc, n0, colStride)), rowStride, colStride, rows, cols);
        }
      }
      matrixResultType = previousType; matrixResultTile = previousTile;
      return success();
    });
  }

  CanonicalKernelAnalysis analysis;
  ModuleOp target;
  OpBuilder b;
  dsa::ConfigurationAttr config;
  DictionaryAttr shapeBindings;
  func::FuncOp function;
  IRMapping values;
  DenseMap<Value, SmallVector<Value>> products;
  DenseMap<int64_t, Value> dimensions;
  DenseMap<Value, Domain> domains;
  Value taskId, taskCount, activeBegin, activeCount;
  RankedTensorType matrixResultType;
  Value matrixResultTile;
  int64_t activeDimension = 0;
  unsigned parallelDepth = 0;
  SmallVector<int64_t> fullExtents;
  bool structured = false;
  DenseSet<Operation *> materializing;
  DenseMap<int64_t, LocalAxis> axisBindings;
  DenseMap<Value, LocalShape> localShapes;
  DenseMap<Value, LocalShape> valueSlices;
  DenseSet<Operation *> streamedOperations;
  std::optional<LogicalWorksetFact> distributedWorkset;
  ParallelOp distributedRoot;
};
}
LogicalResult lowerCanonicalKIRToDSA(ModuleOp module, dsa::ConfigurationAttr configuration, DictionaryAttr shapes) {
  CanonicalKernelAnalysis analysis(module);
  if (failed(analysis.verify())) return failure();
  OwningOpRef<ModuleOp> physical = ModuleOp::create(module.getLoc());
  Construction construction(module, *physical, configuration, shapes);
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (failed(construction.lower(function))) return failure();
  if (failed(dsa::verifyProgram(*physical))) return failure();
  module.getBodyRegion().takeBody(physical->getBodyRegion());
  module->setAttrs((*physical)->getAttrs());
  return success();
}
}
