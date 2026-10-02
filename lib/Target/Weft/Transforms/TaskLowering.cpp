#include "TaskLowering.h"
#include "Intent/Dialect/CPU/Transforms/Implementation.h"
#include "Views.h"
#include "TaskInterface.h"
#include "Quantization.h"
#include "Reductions.h"
#include "Intent/Dialect/CPU/Analysis/AxisRelations.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Analysis/ControlFlow.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/SubviewBounds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
namespace {

SmallVector<int64_t> shape(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type)) return llvm::to_vector(value.getShape().asArrayRef());
  if (auto value = dyn_cast<wk::ViewType>(type)) return llvm::to_vector(value.getShape().asArrayRef());
  if (auto value = dyn_cast<wk::SliceType>(type)) return llvm::to_vector(value.getShape().asArrayRef());
  return {};
}

SmallVector<int64_t> axes(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type)) return llvm::to_vector(value.getAxisIds().asArrayRef());
  if (auto value = dyn_cast<wk::ViewType>(type)) return llvm::to_vector(value.getAxisIds().asArrayRef());
  if (auto value = dyn_cast<wk::SliceType>(type)) return llvm::to_vector(value.getAxisIds().asArrayRef());
  return {};
}

Type element(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type)) return value.getElementType();
  return type;
}

} // namespace

class TaskLowering::Impl {
  struct LocalValue {
    Value value;
    SmallVector<OpFoldResult> sizes;
  };

public:
  Impl(func::FuncOp function, ModuleOp output,
               const llvm::DenseMap<Value, intent::QuantFormat> &formats,
               const cpu::ImplementationRegistry &implementations)
      : sourceFunction(function), relations(function),
        output(output), b(output.getContext()),
        formats(formats), implementations(implementations) {
    auto interface = getPublicInterface(function);
    for (auto [argument, schema] : llvm::zip(function.getArguments(), interface.getArguments())) {
      auto view = dyn_cast<intent::ViewType>(cast<PublicParameterAttr>(schema).getType());
      if (!view) continue;
      auto memory = cast<MemRefType>(argument.getType());
      for (unsigned axis = 0; axis < memory.getRank(); ++axis)
        if (memory.isDynamicDim(axis)) extentIds.try_emplace(publicViewDimensions(view)[axis], extentIds.size() + 1);
    }
    auto join = [&](Value lhs, Value rhs) {
      int64_t a = physicalAxis(relations.axes(lhs).back());
      int64_t c = physicalAxis(relations.axes(rhs).back());
      if (a != c) axisProjection[std::max(a, c)] = std::min(a, c);
    };
    function.walk([&](cpu::QuantizeOp op) { join(op.getInput(), op.getOutput()); });
    function.walk([&](cpu::QuantizedDotOp op) { join(op.getLhs(), op.getRhs()); });
    nextAxis = 1;
    function.walk([&](Operation *operation) {
      for (Value value : operation->getOperands())
        if (isa<MemRefType>(value.getType()))
          for (int64_t axis : relations.axes(value)) nextAxis = std::max(nextAxis, axis + 1);
    });
  }

  SmallVector<Value> shapeArguments(func::FuncOp function, OpBuilder &builder) {
    SmallVector<Value> result(extentIds.size());
    auto interface = getPublicInterface(function);
    for (BlockArgument argument : function.getArguments()) {
      auto memory = dyn_cast<MemRefType>(argument.getType());
      if (!memory) continue;
      auto ids = publicViewDimensions(getPublicView(interface, argument.getArgNumber()));
      for (auto [axis, extent] : llvm::enumerate(memory.getShape()))
        if (ShapedType::isDynamic(extent))
          result[extentIds.at(ids[axis]) - 1] = builder.create<memref::DimOp>(function.getLoc(), argument, axis);
    }
    return result;
  }

  LogicalResult lower(cpu::TasksOp tasks, StringRef name,
                      SmallVectorImpl<unsigned> &argumentPositions) {
    storage = std::make_unique<cpu::StorageAnalysis>(sourceFunction);
    currentTask = tasks;
    values.clear(); locals.clear(); localReferences.clear(); controlOwners.clear(); shapeValues.clear(); readOnlySupplies.clear(); viewAxes.clear();
    Location loc = tasks.getLoc();
    SmallVector<Attribute> names, accesses, symbols;
    SmallVector<int64_t> aliases;
    SmallVector<Type> types;
    auto memoryEffects = storage->effects(tasks);
    if (!memoryEffects.complete)
      return tasks.emitError("Weft task requires complete storage effects");
    auto formatStatus = tasks.walk([&](Operation *operation) {
      for (Value memory : operation->getOperands()) {
        if (!isa<MemRefType>(memory.getType())) continue;
        auto origins = storage->origins(memory);
        std::optional<intent::QuantFormat> selected;
        bool denseOrigin = false;
        for (Value origin : origins.values) {
          auto found = formats.find(origin);
          if (found == formats.end()) { denseOrigin = true; continue; }
          if (selected && *selected != found->second) {
            operation->emitError("Weft view merges incompatible encoded storage");
            return WalkResult::interrupt();
          }
          selected = found->second;
        }
        if (selected && (!origins.complete || denseOrigin)) {
          operation->emitError("Weft encoded view requires one representation across all storage origins");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (formatStatus.wasInterrupted()) return failure();
    for (unsigned position : argumentPositions) {
      Value capture = position == 0 ? tasks.getBody().front().getArgument(0) : tasks.getCaptures()[position - 1];
      names.push_back(b.getStringAttr(position == 0 ? "coordinate" : "capture_" + std::to_string(position - 1)));
      if (auto memory = dyn_cast<MemRefType>(capture.getType())) {
        auto ids = memoryAxes(capture);
        llvm::SmallSet<int64_t, 8> unique(ids.begin(), ids.end());
        if (unique.size() != ids.size())
          return tasks.emitError("Weft view requires independent logical axes; this reuse needs an explicit axis projection");
        SmallVector<int64_t> dimensions;
        for (auto [axis, extent] : llvm::enumerate(memory.getShape())) {
          if (ShapedType::isDynamic(extent)) {
            auto symbol = shapeSymbol(ValueBoundsConstraintSet::Variable(capture, axis));
            if (!symbol) return tasks.emitError("captured view extent has no external shape binding");
            dimensions.push_back(-*symbol);
          } else dimensions.push_back(extent);
        }
        auto format = quantizedFormat(capture);
        if (format) {
          int64_t bytes = *format == intent::QuantFormat::Q4K ? 144 : 292;
          if (memory.getShape().back() != bytes)
            return tasks.emitError("encoded capture requires a statically complete contiguous record byte span");
          dimensions.back() = 256;
        }
        types.push_back(wk::ViewType::get(b.getContext(), encoding(capture), array(dimensions), array(ids)));
        bool reads = false, writes = false;
        for (const cpu::StorageEffect &entry : memoryEffects.entries) {
          Value affected = entry.effect.getValue();
          if (!affected || !storage->disjoint(affected, capture)) {
            reads |= isa<MemoryEffects::Read>(entry.effect.getEffect());
            writes |= isa<MemoryEffects::Write>(entry.effect.getEffect());
          }
        }
        accesses.push_back(b.getStringAttr(reads ? (writes ? "readwrite" : "read")
                                               : (writes ? "write" : "none")));
        aliases.push_back(0);
      } else {
        types.push_back(scalarView(capture.getType()));
        accesses.push_back(b.getStringAttr("read"));
        aliases.push_back(0);
      }
    }
    for (unsigned symbol = 1; symbol <= extentIds.size(); ++symbol)
      symbols.push_back(b.getStringAttr("shape_" + std::to_string(symbol)));
    b.setInsertionPointToEnd(output.getBody());
    auto kernel = b.create<wk::KernelOp>(loc, name, b.getArrayAttr(names),
        b.getArrayAttr(accesses), array(aliases), b.getArrayAttr(symbols), "Intent CPU task");
    Block *body = new Block;
    kernel.getBody().push_back(body);
    for (Type type : types) body->addArgument(type, loc);
    b.setInsertionPointToStart(body);
    for (Attribute symbol : symbols)
      shapeValues.push_back(b.create<wk::SymbolOp>(loc, b.getIndexType(), cast<StringAttr>(symbol),
          b.getStringAttr("shape"), array({})));
    b.create<wk::RootDomainOp>(loc,
        wk::DomainType::get(b.getContext(), "root", 0, -1, 0, "root", "exact"));
    for (auto [position, target] : llvm::zip(argumentPositions, body->getArguments())) {
      Value source = tasks.getBody().front().getArgument(position);
      if (isa<MemRefType>(source.getType())) values.map(source, target);
      else {
        Value scalar = b.create<wk::SliceOp>(loc,
            sliceType(target, {}, {}),
            target, ValueRange{index(loc, 0)}, b.getArrayAttr({b.getStringAttr("index")}));
        Type storage = source.getType().isIndex()
            ? Type(IntegerType::get(b.getContext(), 64, IntegerType::Signed)) : source.getType();
        Value loaded = b.create<wk::AdmitOp>(loc, storage, scalar);
        if (source.getType().isIndex()) loaded = b.create<wk::CastOp>(loc, b.getIndexType(), loaded);
        values.map(source, loaded);
      }
    }
    if (failed(block(tasks.getBody().front()))) return failure();
    b.create<wk::ReturnOp>(loc, ValueRange{});
    return finalizeTaskInterface(kernel, argumentPositions);
  }

private:
  int64_t physicalAxis(int64_t axis) {
    while (axisProjection.count(axis)) axis = axisProjection.lookup(axis);
    return axis;
  }
  SmallVector<int64_t> memoryAxes(Value memory) {
    auto result = relations.axes(memory);
    for (int64_t &axis : result) axis = physicalAxis(axis);
    return result;
  }
  DenseI64ArrayAttr array(ArrayRef<int64_t> entries) { return b.getDenseI64ArrayAttr(entries); }
  wk::EncodingType dense(Type type) {
    std::string name;
    if (type.isIndex()) name = "i64";
    else if (auto integer = dyn_cast<IntegerType>(type))
      name = (integer.isUnsigned() ? "u" : "i") + std::to_string(integer.getWidth());
    else { llvm::raw_string_ostream stream(name); type.print(stream); }
    return wk::EncodingType::get(b.getContext(), name, "dense", "dense." + name, array({}));
  }
  wk::EncodingType encoding(Value memory) {
    auto format = quantizedFormat(memory);
    return format ? quantEncoding(b, *format)
                  : dense(cast<MemRefType>(memory.getType()).getElementType());
  }
  wk::SliceType sliceType(Value base, ArrayRef<int64_t> dimensions,
                         ArrayRef<int64_t> ids) {
    Type type = base.getType();
    Type encoding = isa<wk::ViewType>(type)
                        ? cast<wk::ViewType>(type).getEncoding()
                        : cast<wk::SliceType>(type).getEncoding();
    return wk::SliceType::get(b.getContext(), encoding, array(dimensions),
                             array(ids));
  }
  std::optional<intent::QuantFormat> quantizedFormat(Value memory) {
    for (Value origin : storage->origins(memory).values)
      if (auto found = formats.find(origin); found != formats.end())
        return found->second;
    return std::nullopt;
  }
  wk::ViewType scalarView(Type type) {
    return wk::ViewType::get(b.getContext(), dense(type), array({1}), array({nextAxis++}));
  }
  Type valueType(Type element, ArrayRef<int64_t> dimensions, ArrayRef<int64_t> ids) {
    element = scalarType(element);
    if (dimensions.empty()) return element;
    return wk::ValueType::get(b.getContext(), element, array(dimensions), array(ids));
  }
  Type scalarType(Type type) {
    if (auto integer = dyn_cast<IntegerType>(type); integer && integer.isSignless() && integer.getWidth() != 1)
      return IntegerType::get(b.getContext(), integer.getWidth(), IntegerType::Signed);
    return type;
  }
  Value index(Location loc, int64_t value) { return b.create<arith::ConstantIndexOp>(loc, value); }
  FailureOr<Type> resultType(Value memory, Type scalar = {}) {
    auto type = cast<MemRefType>(memory.getType());
    auto ids = memoryAxes(memory);
    SmallVector<int64_t> dimensions;
    for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
      if (ShapedType::isDynamic(extent)) {
        auto dimension = dimensionExtent(memory, axis);
        if (!dimension) return emitError(memory.getLoc(), "private value extent has no explicit shape binding"), failure();
        dimensions.push_back(*dimension);
      } else dimensions.push_back(extent);
    }
    Type element = scalar ? scalar : type.getElementType();
    if (element.isIndex()) element = IntegerType::get(b.getContext(), 64, IntegerType::Signed);
    return valueType(element, dimensions, ids);
  }
  bool sameBound(OpFoldResult lhs, OpFoldResult rhs) {
    if (lhs == rhs) return true;
    auto constant = [](OpFoldResult value) -> std::optional<int64_t> {
      if (auto attribute = dyn_cast<Attribute>(value)) return cast<IntegerAttr>(attribute).getInt();
      llvm::APInt integer;
      if (matchPattern(cast<Value>(value), m_ConstantInt(&integer))) return integer.getSExtValue();
      return std::nullopt;
    };
    auto a = constant(lhs), c = constant(rhs);
    if (a && c) return *a == *c;
    auto identity = [&](OpFoldResult bound) -> std::optional<int64_t> {
      if (auto value = dyn_cast<Value>(bound)) {
        auto found = llvm::find(shapeValues, value);
        if (found != shapeValues.end()) return -1 - (found - shapeValues.begin());
      }
      return nativeExtent(bound);
    };
    auto left = identity(lhs), right = identity(rhs);
    return left && right && *left == *right;
  }
  Value localRoot(Value memory) {
    if (controlOwners.contains(memory)) return memory;
    if (auto found = localReferences.find(memory); found != localReferences.end())
      return found->second;
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) return localRoot(cast.getSource());
    if (auto view = memory.getDefiningOp<memref::SubViewOp>()) return localRoot(view.getSource());
    if (memory.getDefiningOp() && isAxisView(memory.getDefiningOp())) {
      auto projection = queryAxisView(memory);
      if (succeeded(projection)) return localRoot(projection->source);
    }
    return storage->uniqueOrigin(memory);
  }

  bool sameStorageShape(Value first, Value second) {
    auto left = dyn_cast<MemRefType>(first.getType());
    auto right = dyn_cast<MemRefType>(second.getType());
    if (!left || !right || left.getRank() != right.getRank() ||
        left.getElementType() != right.getElementType()) return false;
    for (int64_t axis = 0; axis < left.getRank(); ++axis) {
      if (!left.isDynamicDim(axis) && !right.isDynamicDim(axis) &&
          left.getDimSize(axis) == right.getDimSize(axis)) continue;
      if (!cpu::haveEqualExtents(ValueBoundsConstraintSet::Variable(first, axis),
                                ValueBoundsConstraintSet::Variable(second, axis))) return false;
    }
    return true;
  }

  Value fullLocalOwner(Value memory) {
    Value root = storage->uniqueOrigin(memory);
    if (!root || !isLocal(root)) return {};
    if (!sameStorageShape(memory, root)) return {};
    llvm::SmallDenseSet<Value> visited;
    std::function<bool(Value)> complete = [&](Value value) {
      if (value == root || !visited.insert(value).second) return true;
      if (auto cast = value.getDefiningOp<memref::CastOp>())
        return complete(cast.getSource());
      if (auto view = value.getDefiningOp<memref::SubViewOp>()) {
        if (view.getDroppedDims().any()) return false;
        for (auto [axis, size] : llvm::enumerate(view.getMixedSizes()))
          if (!sameBound(view.getMixedOffsets()[axis], b.getIndexAttr(0)) ||
              !sameBound(view.getMixedStrides()[axis], b.getIndexAttr(1)) ||
              !cpu::haveEqualExtents(ValueBoundsConstraintSet::Variable(size),
                  ValueBoundsConstraintSet::Variable(view.getSource(), axis)))
            return false;
        return complete(view.getSource());
      }
      auto incoming = intent::queryControlFlowIncoming(value);
      return incoming.complete && !incoming.edges.empty() &&
          llvm::all_of(incoming.edges, [&](const intent::ControlFlowEdge &edge) {
            return edge.operand && complete(edge.operand->get());
          });
    };
    return complete(memory) ? root : Value{};
  }
  bool isLocal(Value memory) {
    Value root = localRoot(memory);
    if (controlOwners.contains(root)) return true;
    Operation *owner = root ? root.getDefiningOp() : nullptr;
    return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(owner) &&
           currentTask->isProperAncestor(owner);
  }

  std::optional<int64_t> nativeExtent(const cpu::ExtentExpression &extent) {
    AffineExpr expression = extent.expression.getResult(0);
    if (auto constant = dyn_cast<AffineConstantExpr>(expression))
      return constant.getValue() >= 0
          ? std::optional<int64_t>(constant.getValue()) : std::nullopt;

    // Weft shape entries are constants or references to the actual public shape
    // arguments. Proving an affine expression is not permission to invent a
    // new runtime symbol for it.
    unsigned position;
    if (auto dimension = dyn_cast<AffineDimExpr>(expression))
      position = dimension.getPosition();
    else if (auto symbol = dyn_cast<AffineSymbolExpr>(expression))
      position = extent.expression.getNumDims() + symbol.getPosition();
    else return std::nullopt;
    auto [value, axis] = extent.operands[position];
    auto argument = dyn_cast<BlockArgument>(value);
    auto function = argument
        ? dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp()) : func::FuncOp{};
    if (function != sourceFunction || !axis) return std::nullopt;
    auto publicView = getPublicView(getPublicInterface(function), argument.getArgNumber());
    if (!publicView) return std::nullopt;
    auto symbol = extentIds.find(publicViewDimensions(publicView)[*axis]);
    return symbol == extentIds.end()
        ? std::nullopt : std::optional<int64_t>(-symbol->second);
  }

  std::optional<int64_t> nativeExtent(OpFoldResult extent) {
    auto expression = cpu::queryExtent(extent);
    return succeeded(expression) ? nativeExtent(*expression) : std::nullopt;
  }

  std::optional<int64_t> dimensionExtent(Value memory, unsigned axis) {
    auto expression = cpu::queryExtent(ValueBoundsConstraintSet::Variable(memory, axis));
    return succeeded(expression) ? nativeExtent(*expression) : std::nullopt;
  }

  std::optional<int64_t>
  shapeSymbol(const ValueBoundsConstraintSet::Variable &extent) {
    if (auto expression = cpu::queryExtent(extent); succeeded(expression))
      if (auto native = nativeExtent(*expression); native && *native < 0)
        return -*native;
    // A dynamic host descriptor still needs an actual shape parameter even
    // when equality analysis proves its extent constant. Select only among the
    // existing public bindings; never synthesize a symbol for an expression.
    auto interface = getPublicInterface(sourceFunction);
    for (BlockArgument argument : sourceFunction.getArguments()) {
      auto memory = dyn_cast<MemRefType>(argument.getType());
      if (!memory) continue;
      auto ids = publicViewDimensions(getPublicView(interface, argument.getArgNumber()));
      for (unsigned axis = 0; axis < static_cast<unsigned>(memory.getRank()); ++axis)
        if (memory.isDynamicDim(axis) && cpu::haveEqualExtents(
                extent, ValueBoundsConstraintSet::Variable(argument, axis)))
          return extentIds.at(ids[axis]);
    }
    return std::nullopt;
  }

  FailureOr<Value> view(Value memory) {
    if (values.contains(memory)) {
      if (!viewAxes.count(memory)) {
        auto &mapping = viewAxes[memory];
        for (int64_t axis = 0; axis < cast<MemRefType>(memory.getType()).getRank(); ++axis) mapping.push_back(axis);
      }
      return values.lookup(memory);
    }
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) {
      auto source = view(cast.getSource());
      if (failed(source)) return failure();
      viewAxes[memory] = viewAxes.at(cast.getSource());
      values.map(memory, *source);
      return source;
    }
    if (memory.getDefiningOp() && isAxisView(memory.getDefiningOp())) {
      auto projection = queryAxisView(memory);
      if (failed(projection)) return failure();
      auto source = view(projection->source);
      if (failed(source)) return failure();
      auto &mapping = viewAxes[memory];
      for (auto axis : projection->sourceAxes)
        mapping.push_back(axis ? viewAxes.at(projection->source)[*axis] : -1);
      values.map(memory, *source);
      return source;
    }
    auto operation = memory.getDefiningOp<memref::SubViewOp>();
    if (!operation) return emitError(memory.getLoc(), "CPU memory has no explicit Weft view relation: ") << memory, failure();
    auto base = view(operation.getSource());
    if (failed(base)) return failure();
    auto sourceAxes = viewAxes.at(operation.getSource());
    auto baseShape = shape((*base).getType());
    SmallVector<OpFoldResult> nativeOffsets(baseShape.size(), b.getIndexAttr(0));
    SmallVector<OpFoldResult> nativeSizes(baseShape.size(), b.getIndexAttr(1));
    llvm::SmallBitVector dropped(baseShape.size());
    auto logicalDropped = operation.getDroppedDims();
    for (auto [axis, native] : llvm::enumerate(sourceAxes)) {
      if (native < 0) {
        if (!sameBound(operation.getMixedOffsets()[axis], b.getIndexAttr(0)) ||
            !sameBound(operation.getMixedSizes()[axis], b.getIndexAttr(1)))
          return operation.emitError("Weft inserted unit-axis subview must retain its single element"), failure();
        continue;
      }
      nativeOffsets[native] = operation.getMixedOffsets()[axis];
      nativeSizes[native] = operation.getMixedSizes()[axis];
      if (logicalDropped.test(axis)) dropped.set(native);
    }
    SmallVector<int64_t> mapping;
    for (auto [axis, native] : llvm::enumerate(sourceAxes))
      if (!logicalDropped.test(axis)) {
        int64_t shifted = native;
        for (int64_t before = 0; before < native; ++before) shifted -= dropped.test(before);
        mapping.push_back(shifted);
      }
    SmallVector<int64_t> offsets, extents, dimensions;
    SmallVector<Value> dynamic;
    auto append = [&](ArrayRef<OpFoldResult> bounds, SmallVectorImpl<int64_t> &statics) {
      for (OpFoldResult value : bounds) {
        if (auto attr = dyn_cast<Attribute>(value)) statics.push_back(cast<IntegerAttr>(attr).getInt());
        else {
          llvm::APInt integer;
          if (matchPattern(cast<Value>(value), m_ConstantInt(&integer))) statics.push_back(integer.getSExtValue());
          else { statics.push_back(-1); dynamic.push_back(values.lookup(cast<Value>(value))); }
        }
      }
    };
    for (OpFoldResult stride : operation.getMixedStrides())
      if (!sameBound(stride, b.getIndexAttr(1))) return operation.emitError("Weft subview requires unit coordinate steps"), failure();
    bool projectionOnly = true;
    for (unsigned axis = 0; axis < baseShape.size(); ++axis) {
      if (dropped.test(axis)) continue;
      auto size = nativeSizes[axis];
      bool full = nativeExtent(size) == baseShape[axis];
      projectionOnly &= sameBound(nativeOffsets[axis], b.getIndexAttr(0)) && full;
    }
    if (projectionOnly) {
      SmallVector<Attribute> selectors;
      SmallVector<Value> indices;
      SmallVector<int64_t> keptShape, keptAxes;
      auto baseAxes = axes((*base).getType());
      for (unsigned axis = 0; axis < baseShape.size(); ++axis) {
        selectors.push_back(b.getStringAttr(dropped.test(axis) ? "index" : "all"));
        if (dropped.test(axis)) {
          auto offset = nativeOffsets[axis];
          indices.push_back(isa<Attribute>(offset) ? index(operation.getLoc(), cast<IntegerAttr>(cast<Attribute>(offset)).getInt())
                                                   : values.lookup(cast<Value>(offset)));
        } else { keptShape.push_back(baseShape[axis]); keptAxes.push_back(baseAxes[axis]); }
      }
      Value selected = b.create<wk::SliceOp>(operation.getLoc(),
          sliceType(*base, keptShape, keptAxes),
          *base, indices, b.getArrayAttr(selectors));
      values.map(memory, selected);
      viewAxes[memory] = std::move(mapping);
      return selected;
    }
    append(nativeOffsets, offsets);
    for (OpFoldResult size : nativeSizes) {
      auto extent = nativeExtent(size);
      if (!extent) return operation.emitError("dynamic Weft subview extent has no shape binding"), failure();
      dimensions.push_back(*extent);
      if (*extent >= 0) extents.push_back(*extent);
      else {
        extents.push_back(-1);
        dynamic.push_back(values.lookup(cast<Value>(size)));
      }
    }
    auto format = quantizedFormat(memory);
    if (format) {
      int64_t bytes = *format == intent::QuantFormat::Q4K ? 144 : 292;
      if (offsets.back() != 0 || extents.back() != bytes)
        return operation.emitError("encoded view projection must retain its complete record bytes"), failure();
      extents.back() = dimensions.back() = 256;
    }
    auto ids = axes((*base).getType());
    Value selected = b.create<wk::SubviewOp>(operation.getLoc(),
        sliceType(*base, dimensions, ids),
        *base, dynamic, array(offsets), array(extents));
    if (dropped.any()) {
      SmallVector<Attribute> selectors;
      SmallVector<Value> indices;
      SmallVector<int64_t> keptShape, keptAxes;
      for (unsigned axis = 0; axis < ids.size(); ++axis) {
        if (dropped.test(axis)) {
          selectors.push_back(b.getStringAttr("index"));
          indices.push_back(index(operation.getLoc(), 0));
        } else {
          selectors.push_back(b.getStringAttr("all"));
          keptShape.push_back(dimensions[axis]); keptAxes.push_back(ids[axis]);
        }
      }
      selected = b.create<wk::SliceOp>(operation.getLoc(),
          sliceType(selected, keptShape, keptAxes),
          selected, indices, b.getArrayAttr(selectors));
    }
    values.map(memory, selected);
    viewAxes[memory] = std::move(mapping);
    return selected;
  }

  FailureOr<Value> reshapeViewValue(Value memory, Value value, Type target, SmallVector<int64_t> order) {
    auto sourceAxes = axes(value.getType());
    for (int64_t axis : sourceAxes) if (!llvm::is_contained(order, axis)) order.push_back(axis);
    SmallVector<int64_t> sourceDomain, reorderedDomain;
    auto sourceShape = shape(value.getType());
    for (auto [axis, extent] : llvm::zip(sourceAxes, sourceShape))
      if (extent != 1) sourceDomain.push_back(axis);
    for (int64_t axis : order)
      if (sourceShape[llvm::find(sourceAxes, axis) - sourceAxes.begin()] != 1) reorderedDomain.push_back(axis);
    // Moving a unit axis does not change the linear element sequence. Keep
    // the original order so the target need not realize a physical transpose.
    if (sourceDomain == reorderedDomain) order = sourceAxes;
    if (value.getType() == target && order == sourceAxes) return value;
    if (!isa<wk::ValueType>(value.getType()))
      return Value(b.create<wk::NewOp>(memory.getLoc(), target, value, true));
    if (!isa<wk::ValueType>(target)) {
      if (llvm::any_of(shape(value.getType()), [](int64_t size) { return size != 1; }))
        return emitError(memory.getLoc(), "Weft scalar view projection must contain one element"), failure();
      return Value(b.create<wk::ExtractOp>(memory.getLoc(), target, value,
          SmallVector<Value>(sourceAxes.size(), index(memory.getLoc(), 0)),
          b.getArrayAttr(SmallVector<Attribute>(sourceAxes.size(), b.getStringAttr("index")))));
    }
    bool dynamic = llvm::any_of(shape(value.getType()), [](int64_t size) { return size < 0; }) ||
                   llvm::any_of(shape(target), [](int64_t size) { return size < 0; });
    if (dynamic && (shape(value.getType()) != shape(target) || order != sourceAxes))
      return emitError(memory.getLoc(), "Weft axis permutation requires a statically shaped admitted tile"), failure();
    return Value(b.create<wk::ReshapeOp>(memory.getLoc(), target, value, array(order)));
  }

  // Positional memory consumers need the descriptor's logical axis order.
  // A contraction instead consumes named axes and can retain storage order.
  FailureOr<Value> projectViewValue(Value memory, Value value, Type nativeType, bool inverse = false) {
    auto logicalType = resultType(memory);
    if (failed(logicalType)) return failure();
    auto nativeAxes = axes(nativeType), logicalAxes = axes(*logicalType);
    auto &mapping = viewAxes.at(memory);
    SmallVector<int64_t> order;
    if (inverse) {
      for (unsigned native = 0; native < nativeAxes.size(); ++native)
        for (auto [logical, position] : llvm::enumerate(mapping))
          if (position == static_cast<int64_t>(native)) order.push_back(logicalAxes[logical]);
    } else {
      for (int64_t native : mapping) if (native >= 0) order.push_back(nativeAxes[native]);
    }
    return reshapeViewValue(memory, value, inverse ? nativeType : *logicalType, std::move(order));
  }

  SmallVector<Value> projectIndices(Value memory, ValueRange indices, unsigned nativeRank) {
    SmallVector<Value> projected(nativeRank, index(memory.getLoc(), 0));
    for (auto [logical, native] : llvm::enumerate(viewAxes.at(memory)))
      if (native >= 0) projected[native] = values.lookup(indices[logical]);
    return projected;
  }

  FailureOr<Value> dimension(Value memory, unsigned axis) {
    auto type = cast<MemRefType>(memory.getType());
    if (!type.isDynamicDim(axis)) return index(memory.getLoc(), type.getDimSize(axis));
    if (auto found = localReferences.find(memory); found != localReferences.end())
      return dimension(found->second, axis);
    if (controlOwners.contains(memory)) {
      auto found = locals.find(memory);
      if (found == locals.end()) return emitError(memory.getLoc(), "private control value has no dominating supply"), failure();
      OpFoldResult size = found->second.sizes[axis];
      return isa<Attribute>(size) ? index(memory.getLoc(), cast<IntegerAttr>(cast<Attribute>(size)).getInt())
                                  : cast<Value>(size);
    }
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) return dimension(cast.getSource(), axis);
    if (isAxisView(memory.getDefiningOp())) {
      auto projection = queryAxisView(memory);
      if (failed(projection)) return failure();
      return projection->sourceAxes[axis] ? dimension(projection->source, *projection->sourceAxes[axis])
                                          : FailureOr<Value>(index(memory.getLoc(), 1));
    }
    if (isLocal(memory)) {
      OpFoldResult size;
      if (auto projection = memory.getDefiningOp<memref::SubViewOp>()) {
        unsigned retained = 0;
        for (unsigned original = 0; original < projection.getSourceType().getRank(); ++original)
          if (!projection.getDroppedDims().test(original) && retained++ == axis)
            size = projection.getMixedSizes()[original];
      } else if (isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(memory.getDefiningOp())) {
        size = memory.getDefiningOp()->getOperand(type.getDynamicDimIndex(axis));
      }
      if (!size) return emitError(memory.getLoc(), "private view dimension has no explicit extent"), failure();
      return isa<Attribute>(size) ? index(memory.getLoc(), cast<IntegerAttr>(cast<Attribute>(size)).getInt())
                                  : values.lookup(cast<Value>(size));
    }
    auto region = view(memory);
    if (failed(region)) return failure();
    int64_t native = viewAxes.at(memory)[axis];
    if (native < 0) return index(memory.getLoc(), 1);
    return Value(b.create<wk::ExtentOp>(memory.getLoc(), b.getIndexType(), *region, native));
  }

  FailureOr<Value> readNative(Value memory) {
    if (!isa<MemRefType>(memory.getType())) return values.lookup(memory);
    if (!isLocal(memory)) {
      auto supply = readOnlySupplies.find(memory);
      if (supply != readOnlySupplies.end()) return supply->second;
      auto previous = operandReads.find(memory);
      if (previous != operandReads.end()) return previous->second;
      auto region = view(memory);
      if (failed(region)) return failure();
      Type element = cast<MemRefType>(memory.getType()).getElementType();
      if (element.isIndex()) element = IntegerType::get(b.getContext(), 64, IntegerType::Signed);
      Value loaded = b.create<wk::AdmitOp>(memory.getLoc(),
          valueType(element, shape((*region).getType()), axes((*region).getType())), *region);
      // Cache only the admitted snapshot. A positional read and a named-axis
      // contraction may consume this same value in different axis orders.
      operandReads[memory] = loaded;
      if (storage->preserves(currentTask, memory)) readOnlySupplies[memory] = loaded;
      return loaded;
    }
    Value root = localRoot(memory);
    auto found = locals.find(root);
    if (found == locals.end()) {
      emitError(memory.getLoc(), "Weft local value is read before a dominating complete supply");
      return failure();
    }
    if (!isa<wk::ValueType>(found->second.value.getType()) &&
        cast<MemRefType>(root.getType()).getRank() != 0) {
      // A complete uniform definition needs only its scalar until a consumer
      // selects a domain. In particular, a panel read never creates the full
      // allocation's register value just to extract that panel again.
      auto type = resultType(memory);
      if (failed(type)) return failure();
      auto supplied = alignValue(found->second.value, *type, memory.getLoc());
      if (failed(supplied)) return failure();
      auto &mapping = viewAxes[memory];
      mapping.clear();
      for (int64_t axis = 0; axis < cast<MemRefType>(memory.getType()).getRank(); ++axis)
        mapping.push_back(axis);
      return supplied;
    }
    if (memory == root || localReferences.lookup(memory) == root) {
      auto type = cast<MemRefType>(root.getType());
      unsigned dynamic = 0;
      for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
        OpFoldResult full = controlOwners.contains(root) ? found->second.sizes[axis] : ShapedType::isDynamic(extent)
            ? OpFoldResult(root.getDefiningOp()->getOperand(dynamic++))
            : OpFoldResult(b.getIndexAttr(extent));
        if (!sameBound(found->second.sizes[axis], full))
          return emitError(memory.getLoc(), "Weft local root read requires a complete initialized region"), failure();
      }
      auto &mapping = viewAxes[memory];
      mapping.clear();
      for (int64_t axis = 0; axis < type.getRank(); ++axis) mapping.push_back(axis);
      return found->second.value;
    }
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) {
      auto source = readNative(cast.getSource());
      if (failed(source)) return failure();
      viewAxes[memory] = viewAxes.at(cast.getSource());
      return source;
    }
    auto projection = localProjection(memory, found->second);
    if (failed(projection)) return failure();
    Value selected = b.create<wk::ExtractOp>(memory.getLoc(), projection->type, found->second.value,
                                            projectionIndices(*projection, memory.getLoc()), b.getArrayAttr(projection->selectors));
    return selected;
  }

  FailureOr<Value> read(Value memory) {
    auto native = readNative(memory);
    if (failed(native) || !isa<MemRefType>(memory.getType())) return native;
    return projectViewValue(memory, *native, (*native).getType());
  }

  FailureOr<Value> readNamedAxes(Value memory) {
    auto native = readNative(memory);
    if (failed(native) || !isa<MemRefType>(memory.getType())) return native;
    auto logical = resultType(memory);
    if (failed(logical)) return failure();
    auto logicalAxes = axes(*logical), logicalShape = shape(*logical);
    auto nativeShape = shape((*native).getType());
    auto mapping = viewAxes.at(memory);
    // Unit dimensions carry only coordinate zero. Reuse an existing unused
    // native unit before inserting one: [1,K] must stay [1,K], because even
    // an element-preserving [1,K] -> [K,1] can change the target partition.
    // This choice is local to the named-value representation; the descriptor
    // projection used by positional reads and writes remains unchanged.
    for (auto [logicalPosition, position] : llvm::enumerate(mapping)) {
      if (position >= 0 || logicalShape[logicalPosition] != 1) continue;
      for (unsigned candidate = 0; candidate < nativeShape.size(); ++candidate)
        if (nativeShape[candidate] == 1 && !llvm::is_contained(mapping, candidate)) {
          mapping[logicalPosition] = candidate;
          break;
        }
    }
    SmallVector<int64_t> dimensions, ids;
    for (unsigned position = 0; position < nativeShape.size(); ++position) {
      auto found = llvm::find(mapping, position);
      if (found == mapping.end()) {
        if (nativeShape[position] != 1)
          return emitError(memory.getLoc(), "named-axis supply lost a non-unit storage dimension"), failure();
        continue;
      }
      unsigned logicalPosition = found - mapping.begin();
      dimensions.push_back(nativeShape[position]);
      ids.push_back(logicalAxes[logicalPosition]);
    }
    for (auto [logicalPosition, position] : llvm::enumerate(mapping))
      if (position < 0) {
        if (logicalShape[logicalPosition] != 1)
          return emitError(memory.getLoc(), "named-axis supply can only insert unit dimensions"), failure();
        dimensions.push_back(1); ids.push_back(logicalAxes[logicalPosition]);
      }
    return reshapeViewValue(memory, *native,
        valueType(element((*native).getType()), dimensions, ids), axes((*native).getType()));
  }

  struct LocalProjection {
    Type type;
    SmallVector<Value> offsets;
    SmallVector<Attribute> selectors;
  };

  SmallVector<Value> projectionIndices(const LocalProjection &projection, Location loc) {
    SmallVector<Value> result;
    auto dimensions = shape(projection.type), ids = axes(projection.type);
    unsigned retained = 0;
    for (auto [selector, offset] : llvm::zip_equal(projection.selectors, projection.offsets)) {
      StringRef kind = cast<StringAttr>(selector).getValue();
      if (kind == "index") { result.push_back(offset); continue; }
      if (kind == "gather") {
        Type unsignedIndex = IntegerType::get(b.getContext(), 64, IntegerType::Unsigned);
        auto type = cast<wk::ValueType>(valueType(unsignedIndex, {dimensions[retained]}, {ids[retained]}));
        Value lane = b.create<wk::IotaOp>(loc, type, 0, dimensions[retained]);
        Value base = b.create<wk::CastOp>(loc, unsignedIndex, offset);
        result.push_back(b.create<wk::BinaryOp>(loc, type, lane, base, "add"));
      }
      ++retained;
    }
    return result;
  }

  FailureOr<LocalProjection> localProjection(Value memory, const LocalValue &state) {
    Value root = localRoot(memory);
    auto rootType = cast<MemRefType>(root.getType());
    SmallVector<Value> origins(rootType.getRank());
    SmallVector<bool> zeroOrigins(rootType.getRank(), true);
    for (Value &origin : origins) origin = index(memory.getLoc(), 0);
    SmallVector<OpFoldResult> sizes(state.sizes);
    SmallVector<int64_t> kept;
    for (unsigned axis = 0; axis < rootType.getRank(); ++axis) kept.push_back(axis);
    SmallVector<Value> chain;
    for (Value current = memory; current != root;) {
      if (auto found = localReferences.find(current); found != localReferences.end()) current = found->second;
      else if (auto cast = current.getDefiningOp<memref::CastOp>()) current = cast.getSource();
      else if (auto view = current.getDefiningOp<memref::SubViewOp>()) { chain.push_back(current); current = view.getSource(); }
      else if (current.getDefiningOp() && isAxisView(current.getDefiningOp())) {
        auto projection = queryAxisView(current);
        if (failed(projection)) return failure();
        chain.push_back(current); current = projection->source;
      }
      else return emitError(memory.getLoc(), "local window has no composed subview relation"), failure();
    }
    for (Value current : llvm::reverse(chain)) {
      SmallVector<int64_t> next;
      if (isAxisView(current.getDefiningOp())) {
        auto projection = queryAxisView(current);
        if (failed(projection)) return failure();
        for (auto source : projection->sourceAxes) next.push_back(source ? kept[*source] : -1);
        kept = std::move(next);
        continue;
      }
      auto view = current.getDefiningOp<memref::SubViewOp>();
      for (unsigned axis = 0; axis < kept.size(); ++axis) {
        int64_t original = kept[axis];
        if (!sameBound(view.getMixedStrides()[axis], b.getIndexAttr(1)))
          return view.emitError("private windows require unit coordinate steps"), failure();
        OpFoldResult offset = view.getMixedOffsets()[axis];
        if (original < 0) {
          if (!sameBound(offset, b.getIndexAttr(0)) || !sameBound(view.getMixedSizes()[axis], b.getIndexAttr(1)))
            return view.emitError("private inserted unit-axis window must retain its single element"), failure();
          if (!view.getDroppedDims().test(axis)) next.push_back(-1);
          continue;
        }
        zeroOrigins[original] = zeroOrigins[original] && sameBound(offset, b.getIndexAttr(0));
        Value value = isa<Attribute>(offset) ? index(memory.getLoc(), cast<IntegerAttr>(cast<Attribute>(offset)).getInt())
                                            : values.lookup(cast<Value>(offset));
        origins[original] = b.create<wk::BinaryOp>(memory.getLoc(), b.getIndexType(), origins[original], value, "add");
        sizes[original] = view.getMixedSizes()[axis];
        if (!view.getDroppedDims().test(axis)) next.push_back(original);
      }
      kept = std::move(next);
    }
    LocalProjection result;
    SmallVector<int64_t> dimensions, ids;
    auto rootAxes = memoryAxes(root);
    auto sourceAxes = axes(state.value.getType());
    auto sourceShape = shape(state.value.getType());
    SmallVector<int64_t> selectedRoots;
    for (auto [position, axis] : llvm::enumerate(sourceAxes)) {
      auto found = llvm::find(rootAxes, axis);
      if (found == rootAxes.end()) return emitError(memory.getLoc(), "private state axis lost its destination relation"), failure();
      unsigned original = found - rootAxes.begin();
      result.offsets.push_back(origins[original]);
      if (!llvm::is_contained(kept, original)) {
        result.selectors.push_back(b.getStringAttr("index"));
        continue;
      }
      selectedRoots.push_back(original);
      if (sameBound(sizes[original], state.sizes[original]) && zeroOrigins[original]) {
        result.selectors.push_back(b.getStringAttr("all"));
        dimensions.push_back(sourceShape[position]); ids.push_back(axis);
        continue;
      }
      std::optional<int64_t> size;
      if (auto attribute = dyn_cast<Attribute>(sizes[original])) size = cast<IntegerAttr>(attribute).getInt();
      else { llvm::APInt constant; if (matchPattern(cast<Value>(sizes[original]), m_ConstantInt(&constant))) size = constant.getSExtValue(); }
      if (!size || *size <= 0)
        return emitError(memory.getLoc(), "private window extent must be statically bounded by the selected implementation"), failure();
      result.selectors.push_back(b.getStringAttr("gather"));
      dimensions.push_back(*size); ids.push_back(axis);
    }
    result.type = valueType(element(state.value.getType()), dimensions, ids);
    auto &mapping = viewAxes[memory];
    mapping.clear();
    for (int64_t original : kept) {
      auto found = llvm::find(selectedRoots, original);
      mapping.push_back(original < 0 ? -1 : found - selectedRoots.begin());
    }
    return result;
  }

  FailureOr<Value> alignValue(Value value, Type target, Location loc) {
    if (element(value.getType()).isIndex() && element(target).isSignedInteger(64))
      value = b.create<wk::CastOp>(loc, valueType(element(target), shape(value.getType()), axes(value.getType())), value);
    if (value.getType() == target) return value;
    auto result = dyn_cast<wk::ValueType>(target);
    if (result && value.getType() == result.getElementType())
      return Value(b.create<wk::NewOp>(loc, target, value, true));
    auto input = dyn_cast<wk::ValueType>(value.getType());
    if (input && result && input.getElementType() == result.getElementType() &&
        llvm::all_of(input.getShape().asArrayRef(), [](int64_t extent) { return extent > 0; }) &&
        llvm::all_of(result.getShape().asArrayRef(), [](int64_t extent) { return extent > 0; })) {
      auto preserves = [](wk::ValueType from, wk::ValueType to) {
        for (auto [axis, extent] : llvm::zip(from.getAxisIds().asArrayRef(), from.getShape().asArrayRef())) {
          auto position = llvm::find(to.getAxisIds().asArrayRef(), axis);
          if (extent != 1 && (position == to.getAxisIds().asArrayRef().end() ||
              to.getShape()[position - to.getAxisIds().asArrayRef().begin()] != extent)) return false;
        }
        return true;
      };
      if (preserves(input, result) && preserves(result, input)) {
        SmallVector<int64_t> order;
        for (int64_t axis : result.getAxisIds().asArrayRef())
          if (llvm::is_contained(input.getAxisIds().asArrayRef(), axis)) order.push_back(axis);
        for (int64_t axis : input.getAxisIds().asArrayRef())
          if (!llvm::is_contained(order, axis)) order.push_back(axis);
        SmallVector<int64_t> inputDomain, resultDomain;
        for (auto [axis, extent] : llvm::zip(input.getAxisIds().asArrayRef(), input.getShape().asArrayRef()))
          if (extent != 1) inputDomain.push_back(axis);
        for (auto [axis, extent] : llvm::zip(result.getAxisIds().asArrayRef(), result.getShape().asArrayRef()))
          if (extent != 1) resultDomain.push_back(axis);
        if (inputDomain == resultDomain) order = llvm::to_vector(input.getAxisIds().asArrayRef());
        return Value(b.create<wk::ReshapeOp>(loc, result, value, array(order)));
      }
    }
    emitError(loc) << "Weft value does not preserve destination axes/extents: " << value.getType() << " -> " << target;
    return failure();
  }

  LogicalResult materializeLocal(Value root) {
    auto found = locals.find(root);
    if (found == locals.end() || isa<wk::ValueType>(found->second.value.getType()) ||
        cast<MemRefType>(root.getType()).getRank() == 0) return success();
    auto type = resultType(root);
    if (failed(type)) return failure();
    auto supplied = alignValue(found->second.value, *type, root.getLoc());
    if (failed(supplied)) return failure();
    found->second.value = *supplied;
    return success();
  }

  Value fillLocal(Value owner, Value scalar, Location loc,
                  const LocalProjection *projection = nullptr) {
    if (!isa<wk::ValueType>(owner.getType())) return scalar;
    auto dimensions = shape(projection ? projection->type : owner.getType());
    SmallVector<Value> coordinates;
    SmallVector<Attribute> selectors(shape(owner.getType()).size(), b.getStringAttr("index"));
    std::function<Value(unsigned, Value)> fill = [&](unsigned axis, Value current) -> Value {
      if (axis == dimensions.size()) {
        SmallVector<Value> selected;
        if (projection) {
          unsigned coordinate = 0;
          for (auto [selector, offset] : llvm::zip_equal(projection->selectors, projection->offsets)) {
            StringRef kind = cast<StringAttr>(selector).getValue();
            if (kind == "index") selected.push_back(offset);
            else selected.push_back(b.create<wk::BinaryOp>(loc, b.getIndexType(), offset,
                                                           coordinates[coordinate++], "add"));
          }
        } else selected = coordinates;
        for (Value &coordinate : selected)
          if (!coordinate.getType().isIndex())
            coordinate = b.create<wk::CastOp>(loc, b.getIndexType(), coordinate);
        return b.create<wk::UpdateOp>(loc, current.getType(), current, scalar,
                                     selected, b.getArrayAttr(selectors));
      }
      Value end = dimensions[axis] < 0 ? shapeValues[-dimensions[axis] - 1]
                                       : index(loc, dimensions[axis]);
      auto loop = b.create<scf::ForOp>(loc, index(loc, 0), end, index(loc, 1), ValueRange{current});
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(loop.getBody());
        coordinates.push_back(loop.getInductionVar());
        Value updated = fill(axis + 1, loop.getRegionIterArgs().front());
        coordinates.pop_back();
        b.create<scf::YieldOp>(loc, ValueRange{updated});
      }
      return loop.getResult(0);
    };
    return fill(0, owner);
  }

  LogicalResult write(Value memory, Value value, SmallVector<OpFoldResult> sizes = {}) {
    if (auto found = localReferences.find(memory); found != localReferences.end())
      return write(found->second, value, std::move(sizes));
    if (!isLocal(memory)) {
      auto region = view(memory);
      if (failed(region)) return failure();
      auto logicalType = resultType(memory);
      if (failed(logicalType)) return failure();
      auto logical = alignValue(value, *logicalType, memory.getLoc());
      if (failed(logical)) return failure();
      auto aligned = projectViewValue(memory, *logical,
          valueType(element((*logical).getType()), shape((*region).getType()), axes((*region).getType())), true);
      if (failed(aligned)) return failure();
      auto dimensions = shape((*region).getType()), ids = axes((*region).getType());
      if (wk::hasDynamicSubviewExtent<wk::SubviewOp, wk::SliceOp, wk::FieldOp>(*region) &&
          llvm::any_of(dimensions, [](int64_t extent) { return extent < 0; })) {
        SmallVector<Attribute> selectors;
        SmallVector<int64_t> fixedShape, fixedAxes;
        for (auto [extent, axis] : llvm::zip(dimensions, ids)) {
          selectors.push_back(b.getStringAttr(extent < 0 ? "index" : "all"));
          if (extent >= 0) { fixedShape.push_back(extent); fixedAxes.push_back(axis); }
        }
        auto selectedType = sliceType(*region, fixedShape, fixedAxes);
        auto selectedValue = valueType(element((*aligned).getType()), fixedShape, fixedAxes);
        SmallVector<Value> indices;
        // Keep the window's runtime extents in loop bounds and issue only
        // fixed-shape projections, as required by the Weft memory consumer.
        std::function<void(unsigned)> project = [&](unsigned axis) {
          if (axis == dimensions.size()) {
            Value selected = b.create<wk::SliceOp>(memory.getLoc(), selectedType, *region,
                indices, b.getArrayAttr(selectors));
            Value payload = b.create<wk::ExtractOp>(memory.getLoc(), selectedValue, *aligned,
                indices, b.getArrayAttr(selectors));
            b.create<wk::CommitOp>(memory.getLoc(), payload, selected);
            return;
          }
          if (dimensions[axis] >= 0) { project(axis + 1); return; }
          Value end = b.create<wk::ExtentOp>(memory.getLoc(), b.getIndexType(), *region, axis);
          auto loop = b.create<scf::ForOp>(memory.getLoc(), index(memory.getLoc(), 0), end, index(memory.getLoc(), 1));
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(loop.getBody());
          indices.push_back(loop.getInductionVar());
          project(axis + 1);
          indices.pop_back();
        };
        project(0);
        return success();
      }
      b.create<wk::CommitOp>(memory.getLoc(), *aligned, *region);
      return success();
    }
    Value root = localRoot(memory);
    if (memory != root) {
      if (auto cast = memory.getDefiningOp<memref::CastOp>()) return write(cast.getSource(), value, sizes);
      auto projection = memory.getDefiningOp<memref::SubViewOp>();
      auto found = locals.find(root);
      if (found != locals.end()) {
        if (failed(materializeLocal(root))) return failure();
        auto projected = localProjection(memory, found->second);
        if (failed(projected)) return failure();
        if (!isa<wk::ValueType>(value.getType())) {
          auto scalar = alignValue(value, element(projected->type), memory.getLoc());
          if (failed(scalar)) return failure();
          found->second.value = fillLocal(found->second.value, *scalar, memory.getLoc(), &*projected);
          return success();
        }
        auto logicalType = resultType(memory);
        if (failed(logicalType)) return failure();
        auto logical = alignValue(value, *logicalType, memory.getLoc());
        if (failed(logical)) return failure();
        auto aligned = projectViewValue(memory, *logical, projected->type, true);
        if (failed(aligned)) return failure();
        found->second.value = b.create<wk::UpdateOp>(memory.getLoc(), found->second.value.getType(),
            found->second.value, *aligned, projectionIndices(*projected, memory.getLoc()), b.getArrayAttr(projected->selectors));
        return success();
      }
      if (isAxisView(memory.getDefiningOp())) {
        auto axisView = queryAxisView(memory);
        if (failed(axisView)) return failure();
        auto sourceType = resultType(axisView->source), logicalType = resultType(memory);
        if (failed(sourceType) || failed(logicalType)) return failure();
        auto &mapping = viewAxes[memory];
        mapping.clear();
        for (auto axis : axisView->sourceAxes) mapping.push_back(axis ? *axis : -1);
        auto logical = alignValue(value, *logicalType, memory.getLoc());
        if (failed(logical)) return failure();
        auto supplied = projectViewValue(memory, *logical, *sourceType, true);
        return failed(supplied) ? failure() : write(axisView->source, *supplied);
      }
      if (!projection || projection.getSource() != root || projection.getDroppedDims().any() ||
          llvm::any_of(projection.getMixedOffsets(), [&](OpFoldResult offset) { return !sameBound(offset, b.getIndexAttr(0)); }))
        return emitError(memory.getLoc(), "Weft private supply must define one zero-based complete active region");
      sizes = projection.getMixedSizes();
    }
    if (sizes.empty()) {
      if (controlOwners.contains(root)) {
        auto found = locals.find(root);
        if (found == locals.end()) return emitError(memory.getLoc(), "private control write has no incoming value");
        sizes = found->second.sizes;
      } else {
      auto type = cast<MemRefType>(root.getType());
      Operation *allocation = root.getDefiningOp();
      unsigned dynamic = 0;
      for (int64_t extent : type.getShape()) {
        if (ShapedType::isDynamic(extent)) sizes.push_back(allocation->getOperand(dynamic++));
        else sizes.push_back(b.getIndexAttr(extent));
      }
      }
    }
    auto type = resultType(memory);
    if (failed(type)) return failure();
    auto current = locals.find(root);
    if (memory == root && !isa<wk::ValueType>(value.getType())) {
      Type scalar = element(*type);
      auto aligned = alignValue(value, scalar, memory.getLoc());
      if (failed(aligned)) return failure();
      if (current != locals.end() && isa<wk::ValueType>(current->second.value.getType()))
        current->second.value = fillLocal(current->second.value, *aligned, memory.getLoc());
      else locals[root] = {*aligned, sizes};
      return success();
    }
    auto aligned = alignValue(value, *type, memory.getLoc());
    if (failed(aligned)) return failure();
    if (current != locals.end() && isa<wk::ValueType>(*type)) {
      if ((isa<wk::ValueType>(current->second.value.getType()) &&
           current->second.value.getType() != *type) || current->second.sizes.size() != sizes.size() ||
          !llvm::all_of(llvm::zip(current->second.sizes, sizes), [&](auto bounds) {
            return sameBound(std::get<0>(bounds), std::get<1>(bounds));
          }))
        return emitError(memory.getLoc(), "complete private write must preserve its initialized owner and extents");
      if (isa<wk::ValueType>(current->second.value.getType()))
        current->second.value = b.create<wk::UpdateOp>(memory.getLoc(), *type,
            current->second.value, *aligned, ValueRange{},
            b.getArrayAttr(SmallVector<Attribute>(shape(*type).size(), b.getStringAttr("all"))));
      else current->second.value = *aligned;
      return success();
    }
    locals[root] = {*aligned, sizes};
    return success();
  }

  FailureOr<Type> pointwiseType(Value lhs, Value rhs) {
    if (element(lhs.getType()) != element(rhs.getType())) return failure();
    auto dimensions = shape(lhs.getType()), ids = axes(lhs.getType());
    auto rightAxes = axes(rhs.getType()), rightShape = shape(rhs.getType());
    for (auto [axis, extent] : llvm::zip(rightAxes, rightShape)) {
      auto it = llvm::find(ids, axis);
      if (it == ids.end()) { ids.push_back(axis); dimensions.push_back(extent); }
      else {
        auto &left = dimensions[it - ids.begin()];
        if (left == 1) left = extent;
        else if (extent != 1 && extent != left) return failure();
      }
    }
    return valueType(element(lhs.getType()), dimensions, ids);
  }

  FailureOr<Value> binary(Location loc, Value lhs, Value rhs, StringRef kind) {
    auto type = pointwiseType(lhs, rhs);
    if (failed(type)) { emitError(loc, "Weft pointwise operands have incompatible axis/extent relations"); return failure(); }
    return Value(b.create<wk::BinaryOp>(loc, *type, lhs, rhs, kind));
  }

  FailureOr<Value> expression(Operation *operation, IRMapping &mapping) {
    Location loc = operation->getLoc();
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      Type type = scalarType(constant.getType());
      if (type == constant.getType()) return b.clone(*constant, mapping)->getResult(0);
      Attribute value = constant.getValue();
      if (auto integer = dyn_cast<IntegerAttr>(value)) value = b.getIntegerAttr(type, integer.getValue());
      return Value(b.create<wk::ConstantOp>(loc, type, value));
    }
    SmallVector<Value> operands;
    for (Value operand : operation->getOperands()) operands.push_back(mapping.lookup(operand));
    StringRef kind;
    if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
      switch (compare.getPredicate()) {
      case arith::CmpIPredicate::eq: kind = "eq"; break;
      case arith::CmpIPredicate::ne: kind = "ne"; break;
      case arith::CmpIPredicate::slt: kind = "lt"; break;
      case arith::CmpIPredicate::sle: kind = "le"; break;
      case arith::CmpIPredicate::sgt: kind = "gt"; break;
      case arith::CmpIPredicate::sge: kind = "ge"; break;
      default: return compare.emitError("unsigned comparison requires unsigned Weft operands"), failure();
      }
    } else if (auto compare = dyn_cast<arith::CmpFOp>(operation)) {
      switch (compare.getPredicate()) {
      case arith::CmpFPredicate::OEQ: kind = "eq"; break;
      case arith::CmpFPredicate::UNE: kind = "ne"; break;
      case arith::CmpFPredicate::OLT: kind = "lt"; break;
      case arith::CmpFPredicate::OLE: kind = "le"; break;
      case arith::CmpFPredicate::OGT: kind = "gt"; break;
      case arith::CmpFPredicate::OGE: kind = "ge"; break;
      default: return compare.emitError("floating comparison has no exact Weft predicate mapping"), failure();
      }
    }
    if (!kind.empty()) {
      auto domain = pointwiseType(operands[0], operands[1]);
      if (failed(domain)) return operation->emitError("comparison domains disagree"), failure();
      return Value(b.create<wk::CompareOp>(loc,
          valueType(b.getI1Type(), shape(*domain), axes(*domain)), operands[0], operands[1], kind));
    }
    if (isa<arith::SelectOp>(operation)) {
      auto domain = pointwiseType(operands[1], operands[2]);
      if (failed(domain)) return operation->emitError("select branch domains disagree"), failure();
      auto dimensions = shape(*domain), ids = axes(*domain);
      for (auto [axis, extent] : llvm::zip(axes(operands[0].getType()), shape(operands[0].getType()))) {
        auto found = llvm::find(ids, axis);
        if (found == ids.end()) { ids.push_back(axis); dimensions.push_back(extent); }
        else if (dimensions[found - ids.begin()] == 1) dimensions[found - ids.begin()] = extent;
        else if (extent != 1 && extent != dimensions[found - ids.begin()])
          return operation->emitError("select predicate domain disagrees with branches"), failure();
      }
      return Value(b.create<wk::SelectOp>(loc, valueType(element(*domain), dimensions, ids),
          operands[0], operands[1], operands[2]));
    }
    if (isa<arith::AddFOp, arith::AddIOp>(operation)) kind = "add";
    else if (isa<arith::SubFOp, arith::SubIOp>(operation)) kind = "sub";
    else if (isa<arith::MulFOp, arith::MulIOp>(operation)) kind = "mul";
    else if (isa<arith::DivFOp, arith::DivSIOp>(operation)) kind = "div";
    else if (isa<arith::RemSIOp>(operation)) kind = "mod";
    else if (isa<arith::MaxNumFOp>(operation)) kind = "max";
    else if (isa<arith::MinNumFOp>(operation)) kind = "min";
    else if (isa<arith::MaximumFOp>(operation)) kind = "maximum";
    else if (isa<arith::MinimumFOp>(operation)) kind = "minimum";
    else if (isa<arith::AndIOp>(operation)) kind = "and";
    else if (isa<arith::OrIOp>(operation)) kind = "or";
    else if (isa<arith::XOrIOp>(operation)) kind = "xor";
    if (!kind.empty()) return binary(loc, operands[0], operands[1], kind);
    if (isa<arith::MinSIOp, arith::MaxSIOp>(operation))
      return binary(loc, operands[0], operands[1], isa<arith::MinSIOp>(operation) ? "min" : "max");
    if (auto divide = dyn_cast<arith::CeilDivSIOp>(operation)) {
      auto minusOne = binary(loc, operands[1], index(loc, 1), "sub");
      if (failed(minusOne)) return failure();
      auto sum = binary(loc, operands[0], *minusOne, "add");
      if (failed(sum)) return failure();
      return binary(loc, *sum, operands[1], "div");
    }
    if (isa<arith::IndexCastOp, arith::SIToFPOp, arith::FPToSIOp, arith::ExtFOp, arith::TruncFOp, arith::ExtSIOp>(operation))
      return Value(b.create<wk::CastOp>(loc, valueType(operation->getResult(0).getType(),
          shape(operands[0].getType()), axes(operands[0].getType())), operands[0]));
    if (isa<math::RsqrtOp>(operation)) kind = "rsqrt";
    else if (isa<math::ExpOp>(operation)) kind = "exp";
    else if (isa<arith::NegFOp>(operation)) kind = "neg";
    if (!kind.empty()) return Value(b.create<wk::UnaryOp>(loc, operands[0].getType(), operands[0], kind));
    operation->emitError("CPU operation has no Weft numerical representation");
    return failure();
  }

  FailureOr<Value> mappedInput(Value input, AffineMap map, ArrayRef<int64_t> loopAxes, bool namedAxes = false) {
    auto loaded = namedAxes ? readNamedAxes(input) : read(input);
    if (failed(loaded)) return failure();
    if (!isa<MemRefType>(input.getType())) return *loaded;
    auto dimensions = shape((*loaded).getType()), ids = axes((*loaded).getType());
    SmallVector<int64_t> keptShape, keptAxes, mappedAxes;
    SmallVector<Attribute> selectors;
    SmallVector<Value> indices;
    auto memoryIds = memoryAxes(input);
    for (auto [axis, id] : llvm::enumerate(ids)) {
      auto position = llvm::find(memoryIds, id);
      if (position == memoryIds.end()) return emitError(input.getLoc(), "Weft input lost its CPU coordinate relation"), failure();
      AffineExpr expression = map.getResult(position - memoryIds.begin());
      if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
        if (constant.getValue() != 0 || dimensions[axis] != 1)
          return emitError(input.getLoc(), "constant pointwise projection requires a singleton input axis"), failure();
        selectors.push_back(b.getStringAttr("index")); indices.push_back(index(input.getLoc(), 0));
      } else {
        auto dim = dyn_cast<AffineDimExpr>(expression);
        if (!dim) return emitError(input.getLoc(), "Weft computation requires an explicit dimension or singleton indexing map"), failure();
        selectors.push_back(b.getStringAttr("all")); keptShape.push_back(dimensions[axis]); keptAxes.push_back(ids[axis]);
        mappedAxes.push_back(loopAxes[dim.getPosition()]);
      }
    }
    Value result = *loaded;
    if (!indices.empty()) result = b.create<wk::ExtractOp>(input.getLoc(), valueType(element(result.getType()), keptShape, keptAxes),
        result, indices, b.getArrayAttr(selectors));
    if (mappedAxes != keptAxes) result = b.create<wk::ReshapeOp>(input.getLoc(),
        valueType(element(result.getType()), keptShape, mappedAxes), result, array(keptAxes));
    return result;
  }

  FailureOr<Value> reduceValue(Location loc, Value input, unsigned axis, StringRef kind) {
    auto dimensions = shape(input.getType()), ids = axes(input.getType());
    SmallVector<int64_t> outputShape(dimensions), outputAxes(ids);
    outputShape.erase(outputShape.begin() + axis); outputAxes.erase(outputAxes.begin() + axis);
    auto countTrue = [&](Value predicate) -> FailureOr<Value> {
      if (dimensions[axis] <= 0 || dimensions[axis] > int64_t(UINT32_MAX))
        return emitError(loc, "boolean reduction requires a statically bounded u32 contribution count"), failure();
      Type count = IntegerType::get(b.getContext(), 32, IntegerType::Unsigned);
      Value zero = b.create<wk::ConstantOp>(loc, count, b.getIntegerAttr(count, 0));
      Value one = b.create<wk::ConstantOp>(loc, count, b.getIntegerAttr(count, 1));
      Value bits = b.create<wk::SelectOp>(loc, valueType(count, dimensions, ids), predicate, one, zero);
      Value total = b.create<wk::ReduceOp>(loc, valueType(count, outputShape, outputAxes), bits, "add", axis);
      Value bound = kind == "and" ? Value(b.create<wk::ConstantOp>(loc, count, b.getIntegerAttr(count, dimensions[axis]))) : zero;
      return Value(b.create<wk::CompareOp>(loc, valueType(b.getI1Type(), outputShape, outputAxes),
                                         total, bound, kind == "and" ? "eq" : "ne"));
    };
    if (kind == "or" || kind == "and") return countTrue(input);
    bool propagating = kind == "maximum" || kind == "minimum";
    Value result = b.create<wk::ReduceOp>(loc, valueType(element(input.getType()), outputShape, outputAxes),
        input, kind == "maximum" ? "max" : kind == "minimum" ? "min" : kind, axis);
    if (propagating) {
      Value isNaN = b.create<wk::CompareOp>(loc, valueType(b.getI1Type(), dimensions, ids), input, input, "ne");
      auto anyNaN = countTrue(isNaN);
      if (failed(anyNaN)) return failure();
      auto floating = cast<FloatType>(element(input.getType()));
      Value nan = b.create<wk::ConstantOp>(loc, floating,
          FloatAttr::get(floating, llvm::APFloat::getQNaN(floating.getFloatSemantics())));
      result = b.create<wk::SelectOp>(loc, result.getType(), *anyNaN, nan, result);
    }
    return result;
  }

  FailureOr<Value> reductionContribution(Block &body,
                                         const NativeReduction &native,
                                         IRMapping &mapping) {
    for (Operation &operation : body.without_terminator()) {
      if (&operation == native.combine) continue;
      for (Value operand : operation.getOperands())
        if (!mapping.contains(operand))
          return operation.emitError("Weft reduction scalar capture has no current value binding"), failure();
      auto value = expression(&operation, mapping);
      if (failed(value)) return failure();
      mapping.map(operation.getResult(0), *value);
    }
    Value contribution = mapping.lookupOrNull(native.contribution);
    if (!contribution)
      return native.combine->emitError("Weft reduction contribution has no current value binding"), failure();
    return contribution;
  }

  LogicalResult generic(linalg::GenericOp operation) {
    if (operation.getOutputs().size() != 1 || operation.getNumResults())
      return operation.emitError("Weft CPU legalization requires one buffer-semantics result");
    Value destination = operation.getOutputs()[0];
    SmallVector<int64_t> loopAxes(operation.getNumLoops(), 0);
    auto maps = operation.getIndexingMapsArray();
    auto outputAxes = memoryAxes(destination);
    for (auto [axis, expression] : llvm::enumerate(maps.back().getResults())) {
      auto dim = dyn_cast<AffineDimExpr>(expression);
      if (!dim) return operation.emitError("Weft output traversal requires dimension projections");
      loopAxes[dim.getPosition()] = outputAxes[axis];
    }
    for (int64_t &axis : loopAxes) if (!axis) axis = nextAxis++;
    if (cpu::isMatrixContraction(operation)) {
      auto lhs = mappedInput(operation.getInputs()[0], maps[0], loopAxes, true);
      auto rhs = mappedInput(operation.getInputs()[1], maps[1], loopAxes, true);
      auto initial = read(destination);
      if (failed(lhs) || failed(rhs) || failed(initial)) return failure();
      auto definition = (*initial).getDefiningOp<wk::NewOp>();
      bool zero = definition && matchPattern(definition->getOperand(0), m_PosZeroFloat());
      if (definition && isa<IntegerType>(element((*initial).getType())))
        if (auto constant = definition->getOperand(0).getDefiningOp<wk::ConstantOp>())
          if (auto value = dyn_cast<IntegerAttr>(constant.getValue())) zero = value.getValue().isZero();
      bool integer = isa<IntegerType>(element((*initial).getType()));
      if (!zero && !integer)
        return operation.emitError("Weft contraction requires an explicit zero-initialized partial; nonzero fused accumulation has no equivalent canonical operation");
      int64_t reduction = loopAxes[2];
      Value term = b.create<wk::OuterContractOp>(operation.getLoc(), (*initial).getType(),
          *lhs, *rhs, array({reduction}), TypeAttr::get(element((*initial).getType())));
      if (!zero) {
        auto accumulated = binary(operation.getLoc(), *initial, term, "add");
        if (failed(accumulated)) return failure();
        term = *accumulated;
      }
      return write(destination, term);
    }
    auto iterators = operation.getIteratorTypesArray();
    SmallVector<unsigned> reductions;
    for (auto [axis, type] : llvm::enumerate(iterators))
      if (type == utils::IteratorType::reduction) reductions.push_back(axis);
    if (reductions.size() > 1)
      return operation.emitError("Weft structured reduction requires one explicit reduction axis");
    IRMapping mapping = values;
    Block &body = operation.getRegion().front();
    for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
      auto value = mappedInput(input, maps[number], loopAxes);
      if (failed(value)) return failure();
      mapping.map(body.getArgument(number), *value);
    }
    if (!reductions.empty()) {
      Value accumulator = body.getArguments().back();
      auto order = operation->getAttrOfType<cpu::ReductionOrderAttr>("intent_cpu.reduction_order");
      auto native = queryNativeReduction(operation, body, accumulator, order);
      if (failed(native)) return failure();
      auto contribution = reductionContribution(body, *native, mapping);
      if (failed(contribution)) return failure();
      auto ids = axes((*contribution).getType());
      auto position = llvm::find(ids, loopAxes[reductions[0]]);
      if (position == ids.end()) return operation.emitError("Weft reduction lost its current logical axis relation");
      auto reduced = reduceValue(operation.getLoc(), *contribution, position - ids.begin(), native->kind);
      auto initial = read(destination);
      if (failed(reduced) || failed(initial)) return failure();
      auto result = binary(operation.getLoc(), *initial, *reduced, native->kind);
      if (failed(result)) return failure();
      return write(destination, *result);
    }
    if (!operation.getIndexingMapsArray().back().isIdentity())
      return operation.emitError("Weft pointwise legalization requires an identity output traversal");
    if (!body.getArguments().back().use_empty()) return operation.emitError("Weft pointwise output is not a pure definition");
    for (Operation &nested : body.without_terminator()) {
      auto value = expression(&nested, mapping);
      if (failed(value)) return failure();
      mapping.map(nested.getResult(0), *value);
    }
    return write(destination, mapping.lookup(body.getTerminator()->getOperand(0)));
  }

  LogicalResult reduction(cpu::ReduceOp operation) {
    Block &body = operation.getCombine().front();
    auto native = queryNativeReduction(operation, body, body.getArgument(0), operation.getOrder());
    if (failed(native)) return failure();
    IRMapping mapping = values;
    SmallVector<int64_t> loopAxes(cast<AffineMapAttr>(operation.getIndexingMaps()[0]).getValue().getNumDims());
    for (int64_t &axis : loopAxes) axis = nextAxis++;
    for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
      auto value = mappedInput(input, cast<AffineMapAttr>(operation.getIndexingMaps()[number]).getValue(), loopAxes);
      if (failed(value)) return failure();
      mapping.map(body.getArgument(number + 1), *value);
    }
    auto contribution = reductionContribution(body, *native, mapping);
    if (failed(contribution)) return failure();
    if (shape((*contribution).getType()).size() != 1)
      return operation.emitError("Weft reduction requires one retained logical input axis");
    auto reduced = reduceValue(operation.getLoc(), *contribution, 0, native->kind);
    auto initial = read(operation.getInitial());
    if (failed(reduced) || failed(initial)) return failure();
    auto result = binary(operation.getLoc(), *initial, *reduced, native->kind);
    if (failed(result)) return failure();
    values.map(operation.getResult(), *result);
    return success();
  }

  LogicalResult block(Block &source) {
    auto enclosingSupplies = readOnlySupplies;
    for (Operation &operation : source.without_terminator())
      if (failed(lower(&operation))) {
        readOnlySupplies = std::move(enclosingSupplies);
        return failure();
      }
    // A supplied immutable input may serve later consumers in this block, but
    // a branch/loop-local SSA value cannot escape into its enclosing scope.
    readOnlySupplies = std::move(enclosingSupplies);
    return success();
  }

  FailureOr<SmallVector<Value>> writtenEnclosingLocals(Operation *scope) {
    SmallVector<Value> result;
    auto effects = storage->effects(scope);
    if (!effects.complete)
      return scope->emitError("Weft control requires complete storage effects"), failure();
    for (const cpu::StorageEffect &entry : effects.entries) {
      if (!isa<MemoryEffects::Write>(entry.effect.getEffect())) continue;
      Value memory = entry.effect.getValue();
      if (!memory)
        return scope->emitError("Weft control write has no storage target"), failure();
      Value owner = localRoot(memory);
      if (owner && isLocal(owner)) {
        Operation *definition = owner.getDefiningOp();
        if (auto argument = dyn_cast<BlockArgument>(owner))
          definition = argument.getOwner()->getParentOp();
        if (definition != scope && !scope->isProperAncestor(definition) &&
            !llvm::is_contained(result, owner)) result.push_back(owner);
        continue;
      }
      auto origins = storage->origins(memory);
      if (!origins.complete)
        return scope->emitError("Weft control write has unresolved storage origins"), failure();
      for (Value root : origins.values)
        if (isLocal(root) && !scope->isProperAncestor(root.getDefiningOp()) &&
            !llvm::is_contained(result, root)) result.push_back(root);
    }
    return result;
  }

  LogicalResult checkCarry(Value root, const LocalValue &before, Operation *scope) {
    if (isa<wk::ValueType>(before.value.getType()) && failed(materializeLocal(root)))
      return failure();
    auto found = locals.find(root);
    if (found == locals.end() || found->second.value.getType() != before.value.getType() ||
        found->second.sizes.size() != before.sizes.size() ||
        !llvm::all_of(llvm::zip(found->second.sizes, before.sizes), [&](auto bounds) {
          return sameBound(std::get<0>(bounds), std::get<1>(bounds));
        }))
      return scope->emitError("Weft control carry must preserve its complete initialized region and type");
    return success();
  }

  LogicalResult bindLocalReference(Value memory, Operation *scope) {
    Value root = fullLocalOwner(memory);
    if (!root)
      return scope->emitError("Weft task control requires a complete private storage owner; dynamic external descriptors belong to host control");
    if (memory != root) localReferences[memory] = root;
    return success();
  }

  bool immutableBorrow(Value memory) {
    auto view = storage->externalView(memory);
    if (!view || view.getAccess() != 0) return false;
    auto effects = storage->effects(currentTask);
    if (!effects.complete || effects.ordered) return false;
    for (const cpu::StorageEffect &entry : effects.entries) {
      Value affected = entry.effect.getValue();
      if (isa<MemoryEffects::Write>(entry.effect.getEffect()) &&
          (!affected || !storage->disjoint(affected, memory))) return false;
      // Native ownership has already proved that a mixed-origin conditional
      // release never frees its borrowed alternative. A direct release of the
      // borrowed storage does not have that ownership distinction.
      if (isa<MemoryEffects::Free>(entry.effect.getEffect()) &&
          (!affected || storage->uniqueOrigin(affected) == memory)) return false;
    }
    return true;
  }

  bool completePrivateValue(Value memory) {
    auto origins = storage->origins(memory);
    if (!origins.complete || origins.values.empty()) return false;
    for (Value origin : origins.values) {
      if (immutableBorrow(origin)) continue;
      Operation *allocation = origin.getDefiningOp();
      if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(allocation) ||
          !currentTask->isProperAncestor(allocation)) return false;
    }
    llvm::SmallDenseSet<Value> visited;
    std::function<bool(Value)> complete = [&](Value value) {
      if (!visited.insert(value).second) return true;
      auto sources = storage->origins(value);
      // The actual descriptor is admitted at its original projection; a
      // borrowed tile need not cover the public allocation's complete shape.
      if (sources.complete && !sources.values.empty() &&
          llvm::all_of(sources.values, [&](Value origin) { return immutableBorrow(origin); }))
        return true;
      if (!sameStorageShape(memory, value)) return false;
      if (llvm::is_contained(origins.values, value)) return true;
      if (auto cast = value.getDefiningOp<memref::CastOp>())
        return complete(cast.getSource());
      if (auto view = value.getDefiningOp<memref::SubViewOp>()) {
        if (view.getDroppedDims().any()) return false;
        for (auto [axis, size] : llvm::enumerate(view.getMixedSizes()))
          if (!sameBound(view.getMixedOffsets()[axis], b.getIndexAttr(0)) ||
              !sameBound(view.getMixedStrides()[axis], b.getIndexAttr(1)) ||
              !cpu::haveEqualExtents(ValueBoundsConstraintSet::Variable(size),
                  ValueBoundsConstraintSet::Variable(view.getSource(), axis))) return false;
        return complete(view.getSource());
      }
      auto incoming = intent::queryControlFlowIncoming(value);
      return incoming.complete && !incoming.edges.empty() &&
          llvm::all_of(incoming.edges, [&](const intent::ControlFlowEdge &edge) {
            return edge.operand && complete(edge.operand->get());
          });
    };
    return complete(memory);
  }

  LogicalResult isolateControlValue(Value memory, Operation *scope,
                                   ValueRange boundaries) {
    if (!completePrivateValue(memory))
      return scope->emitError("Weft task control requires complete private values; external runtime descriptors require host control");
    auto origins = storage->origins(memory);
    llvm::SmallDenseSet<Value> incomingAliases, outgoingAliases;
    for (Value boundary : boundaries) {
      if (!isa<MemRefType>(boundary.getType())) continue;
      auto aliases = storage->aliases(boundary);
      if (!aliases.complete)
        return scope->emitError("private control value has an unresolved storage escape");
      auto &allowed = isa<BlockArgument>(boundary) ? incomingAliases : outgoingAliases;
      allowed.insert(aliases.values.begin(), aliases.values.end());
    }
    auto taskEffects = storage->effects(currentTask);
    auto writesAliases = [&](const auto &aliases) {
      return llvm::any_of(taskEffects.entries, [&](const cpu::StorageEffect &entry) {
        return isa<MemoryEffects::Write>(entry.effect.getEffect()) &&
            (!entry.effect.getValue() || aliases.contains(entry.effect.getValue()));
      });
    };
    bool boundaryWrites = writesAliases(incomingAliases) || writesAliases(outgoingAliases);
    for (Value origin : origins.values) {
      auto aliases = storage->aliases(origin);
      if (!aliases.complete)
        return scope->emitError("private control storage has an unresolved alias or escape");
      Operation *definition = origin.getDefiningOp();
      bool createdInside = definition && scope->isProperAncestor(definition);
      bool mutatesIncoming = llvm::any_of(storage->effects(scope).entries,
          [&](const cpu::StorageEffect &entry) {
            if (!isa<MemoryEffects::Write>(entry.effect.getEffect())) return false;
            Value target = entry.effect.getValue();
            if (!target) return true;
            if (incomingAliases.contains(target)) return true;
            return !createdInside && llvm::is_contained(storage->origins(target).values, origin);
          });
      for (Value alias : aliases.values)
        for (Operation *user : alias.getUsers()) {
          if (!currentTask->isProperAncestor(user)) continue;
          if (user == scope || isa<RegionBranchOpInterface,
                  RegionBranchTerminatorOpInterface>(user) ||
              cpu::isStorageAliasOperation(user)) continue;
          auto effects = storage->effects(user);
          if (!effects.complete || effects.ordered)
            return scope->emitError("private control storage has an ordered or unknown observer");
          bool observes = llvm::any_of(effects.entries, [&](const cpu::StorageEffect &entry) {
            return (!entry.effect.getValue() || entry.effect.getValue() == alias) &&
                isa<MemoryEffects::Read, MemoryEffects::Write>(entry.effect.getEffect());
          });
          if (!observes) continue;
          if (scope->isProperAncestor(user)) {
            if (!isa<scf::IfOp>(scope) && !createdInside &&
                !incomingAliases.contains(alias) && mutatesIncoming)
              return scope->emitError("private control value has a separately observable incoming alias");
          } else {
            Operation *position = scope->getBlock()->findAncestorOpInBlock(*user);
            if (position && position->isBeforeInBlock(scope)) continue;
            bool writes = llvm::any_of(effects.entries, [&](const cpu::StorageEffect &entry) {
              return (!entry.effect.getValue() || entry.effect.getValue() == alias) &&
                  isa<MemoryEffects::Write>(entry.effect.getEffect());
            });
            if (!outgoingAliases.contains(alias) && (writes || boundaryWrites))
              return scope->emitError("private control value retains a separately observable outgoing alias");
          }
        }
    }
    // Distinct live descriptors cannot become independent value states when
    // writes through one may be observed through another descriptor.
    for (Value other : boundaries) {
      if (other == memory || !isa<MemRefType>(other.getType())) continue;
      auto argument = dyn_cast<BlockArgument>(memory);
      auto otherArgument = dyn_cast<BlockArgument>(other);
      bool simultaneous = argument && otherArgument
          ? argument.getOwner() == otherArgument.getOwner()
          : !argument && !otherArgument;
      if (!simultaneous || storage->disjoint(memory, other)) continue;
      llvm::SmallDenseSet<Value> sharedAliases;
      for (Value value : {memory, other}) {
        auto aliases = storage->aliases(value);
        sharedAliases.insert(aliases.values.begin(), aliases.values.end());
      }
      if (writesAliases(sharedAliases))
        return scope->emitError("private control values share an observable mutable storage identity");
    }
    return success();
  }

  LogicalResult prepareControlValues(Operation *scope, ValueRange boundaries) {
    for (Value value : boundaries) {
      if (!isa<MemRefType>(value.getType())) continue;
      Value owner = fullLocalOwner(value);
      if (owner && !scope->isProperAncestor(owner.getDefiningOp())) {
        if (failed(bindLocalReference(value, scope))) return failure();
      } else {
        if (failed(isolateControlValue(value, scope, boundaries))) return failure();
        controlOwners.insert(value);
      }
    }
    return success();
  }

  FailureOr<SmallVector<Type>> controlTypes(ValueRange boundaries) {
    SmallVector<Type> result;
    for (Value value : boundaries) {
      if (!isa<MemRefType>(value.getType())) result.push_back(scalarType(value.getType()));
      else if (controlOwners.contains(value)) {
        auto type = resultType(value);
        if (failed(type)) return failure();
        result.push_back(*type);
      }
    }
    return result;
  }

  Value controlOwner(Type type, Location loc, Value initial = {}) {
    Type scalar = element(type);
    Value zero;
    if (auto floating = dyn_cast<FloatType>(scalar))
      zero = b.create<arith::ConstantOp>(loc, b.getFloatAttr(floating, 0.0));
    else zero = b.create<wk::ConstantOp>(loc, scalar, b.getIntegerAttr(scalar, 0));
    Value owner = b.create<wk::NewOp>(loc, type, zero, true);
    if (initial)
      owner = b.create<wk::UpdateOp>(loc, type, owner, initial, ValueRange{},
          b.getArrayAttr(SmallVector<Attribute>(shape(type).size(), b.getStringAttr("all"))));
    return owner;
  }

  Value nativeOwner(Value value) {
    while (auto update = value.getDefiningOp<wk::UpdateOp>()) value = update.getInput();
    return value.getDefiningOp<wk::NewOp>() || value.getDefiningOp<wk::MaterializeOp>()
        ? value : Value{};
  }

  bool canReuseControlOwner(Value input, Value boundary, Operation *scope) {
    Value root = fullLocalOwner(input);
    if (!root || !locals.count(root)) return false;
    auto aliases = storage->aliases(root), forwarded = storage->aliases(boundary);
    if (!aliases.complete || !forwarded.complete) return false;
    llvm::SmallDenseSet<Value> transferred(forwarded.values.begin(), forwarded.values.end());
    for (Value alias : aliases.values) {
      if (transferred.contains(alias)) continue;
      for (Operation *user : alias.getUsers()) {
        if (user == scope || cpu::isStorageAliasOperation(user) ||
            isa<RegionBranchOpInterface, RegionBranchTerminatorOpInterface>(user)) continue;
        auto effects = storage->effects(user);
        if (!effects.complete || effects.ordered) return false;
        bool observes = llvm::any_of(effects.entries, [&](const cpu::StorageEffect &entry) {
          return (!entry.effect.getValue() || entry.effect.getValue() == alias) &&
              isa<MemoryEffects::Read, MemoryEffects::Write>(entry.effect.getEffect());
        });
        if (!observes) continue;
        Operation *position = scope->getBlock()->findAncestorOpInBlock(*user);
        if (!position || position == scope || !position->isBeforeInBlock(scope)) return false;
      }
    }
    return true;
  }

  void ownControlInputs(SmallVectorImpl<Value> &initial, ValueRange inputs,
                        ValueRange boundaries, Operation *scope) {
    llvm::SmallDenseSet<Value> reused;
    unsigned position = 0;
    for (auto [input, boundary] : llvm::zip_equal(inputs, boundaries)) {
      if (isa<MemRefType>(boundary.getType()) && !controlOwners.contains(boundary)) continue;
      Value &value = initial[position++];
      if (!isa<MemRefType>(boundary.getType()) || !isa<wk::ValueType>(value.getType())) continue;
      Value owner = nativeOwner(value);
      if (!owner || reused.contains(owner) || !canReuseControlOwner(input, boundary, scope))
        value = controlOwner(value.getType(), scope->getLoc(), value);
      else reused.insert(owner);
    }
  }

  FailureOr<Value> alignControlValue(Value value, Type target, Location loc) {
    if (value.getType() == target) return value;
    if (isa<wk::ValueType>(value.getType()) && isa<wk::ValueType>(target) &&
        element(value.getType()) == element(target) && shape(value.getType()) == shape(target))
      return Value(b.create<wk::ReshapeOp>(loc, target, value, array(axes(value.getType()))));
    return alignValue(value, target, loc);
  }

  FailureOr<SmallVector<Value>> controlValues(ValueRange inputs, ValueRange boundaries,
                                             ValueRange owners = {}) {
    SmallVector<Value> result;
    for (auto [input, boundary] : llvm::zip_equal(inputs, boundaries)) {
      if (!isa<MemRefType>(input.getType())) result.push_back(values.lookup(input));
      else if (controlOwners.contains(boundary)) {
        if (Value owner = fullLocalOwner(input); owner && !locals.count(owner))
          if (failed(bindLocalReference(input, boundary.getParentBlock()->getParentOp())))
            return failure();
        auto supplied = read(input);
        auto type = resultType(boundary);
        if (failed(supplied) || failed(type)) return failure();
        if (!owners.empty() && owners[result.size()]) *type = owners[result.size()].getType();
        auto aligned = alignControlValue(*supplied, *type, input.getLoc());
        if (failed(aligned)) return failure();
        result.push_back(*aligned);
      }
    }
    if (!owners.empty()) {
      auto exactOwner = [&](Value value, Value owner) {
        while (auto update = value.getDefiningOp<wk::UpdateOp>()) value = update.getInput();
        return value == owner;
      };
      auto referencesOwner = [&](Value value) {
        while (true) {
          if (llvm::is_contained(owners, value)) return true;
          if (auto update = value.getDefiningOp<wk::UpdateOp>()) value = update.getInput();
          else if (auto extract = value.getDefiningOp<wk::ExtractOp>()) value = extract.getInput();
          else if (auto reshape = value.getDefiningOp<wk::ReshapeOp>()) value = reshape.getInput();
          else return false;
        }
      };
      // Region exits are parallel assignments. Preserve every incoming owner
      // used by another output before any update changes its physical contents.
      for (auto [value, owner] : llvm::zip_equal(result, owners))
        if (owner && isa<wk::ValueType>(value.getType()) &&
            !exactOwner(value, owner) && referencesOwner(value))
          value = controlOwner(value.getType(), value.getLoc(), value);
      for (auto [value, owner] : llvm::zip_equal(result, owners))
        if (owner && isa<wk::ValueType>(value.getType()) && !exactOwner(value, owner))
          value = b.create<wk::UpdateOp>(value.getLoc(), value.getType(), owner, value, ValueRange{},
              b.getArrayAttr(SmallVector<Attribute>(shape(value.getType()).size(), b.getStringAttr("all"))));
    }
    return result;
  }

  LogicalResult mapControlValues(ValueRange source, ValueRange target) {
    unsigned position = 0;
    for (Value value : source) {
      if (!isa<MemRefType>(value.getType())) values.map(value, target[position++]);
      else if (controlOwners.contains(value)) {
        Value supplied = target[position++];
        auto type = resultType(value);
        if (failed(type)) return failure();
        auto aligned = alignControlValue(supplied, *type, value.getLoc());
        if (failed(aligned)) return failure();
        supplied = *aligned;
        SmallVector<OpFoldResult> sizes;
        for (int64_t extent : shape(supplied.getType()))
          sizes.push_back(extent < 0 ? OpFoldResult(shapeValues[-extent - 1])
                                     : OpFoldResult(b.getIndexAttr(extent)));
        locals[value] = {supplied, std::move(sizes)};
      }
    }
    return success();
  }

  LogicalResult lower(Operation *operation) {
    operandReads.clear();
    Location loc = operation->getLoc();
    if (isa<memref::AllocOp, memref::AllocaOp>(operation)) return success();
    if (isa<memref::SubViewOp, memref::CastOp>(operation) || isAxisView(operation)) {
      Value result = operation->getResult(0);
      if (isAxisView(operation) && failed(queryAxisView(result))) return failure();
      if (!isLocal(result) && failed(view(result))) return failure();
      return success();
    }
    if (auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(operation)) {
      SmallVector<Value> descriptors{metadata.getBaseBuffer(), metadata.getOffset()};
      llvm::append_range(descriptors, metadata.getStrides());
      for (Value descriptor : descriptors)
        for (Operation *user : descriptor.getUsers())
          if (!isa<memref::ReinterpretCastOp>(user))
            return metadata.emitError("Weft strided metadata may only supply a proved axis view; arbitrary address arithmetic is unsupported");
      for (auto [axis, size] : llvm::enumerate(metadata.getSizes())) {
        if (size.use_empty()) continue;
        auto extent = dimension(metadata.getSource(), axis);
        if (failed(extent)) return failure();
        values.map(size, *extent);
      }
      return success();
    }
    if (auto dealloc = dyn_cast<memref::DeallocOp>(operation)) { locals.erase(localRoot(dealloc.getMemref())); return success(); }
    if (auto dim = dyn_cast<memref::DimOp>(operation)) {
      auto axis = dim.getConstantIndex();
      if (!axis) return dim.emitError("Weft dimension requires a static axis position");
      auto extent = dimension(dim.getSource(), *axis);
      if (failed(extent)) return failure();
      values.map(dim.getResult(), *extent);
      return success();
    }
    if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
      auto value = read(copy.getSource());
      if (failed(value)) return failure();
      auto target = resultType(copy.getTarget());
      if (failed(target)) return failure();
      if (shape((*value).getType()) != shape(*target) || element((*value).getType()) != element(*target))
        return copy.emitError("positional copy requires the same ordered extents and element type");
      if (axes((*value).getType()) != axes(*target))
        *value = b.create<wk::ReshapeOp>(loc, *target, *value, array(axes((*value).getType())));
      if (isLocal(copy.getTarget())) {
        Value root = localRoot(copy.getTarget());
        if (root == copy.getTarget() && locals.count(root)) {
          if (failed(materializeLocal(root))) return failure();
          return write(copy.getTarget(), *value);
        }
        *value = b.create<wk::MaterializeOp>(loc, (*value).getType(), *value);
      }
      return write(copy.getTarget(), *value);
    }
    if (auto fill = dyn_cast<linalg::FillOp>(operation)) {
      Value destination = fill.getOutputs()[0];
      Value initial = values.lookup(fill.getInputs()[0]);
      if (isLocal(destination)) {
        auto type = resultType(destination);
        if (failed(type)) return failure();
        if (initial.getType().isIndex() && element(*type).isSignedInteger(64))
          initial = b.create<wk::CastOp>(loc, element(*type), initial);
        return write(destination, initial);
      }
      auto region = view(destination);
      if (failed(region)) return failure();
      auto dimensions = shape((*region).getType());
      SmallVector<Value> indices;
      std::function<void(unsigned)> fillAxis = [&](unsigned axis) {
        if (axis == dimensions.size()) {
          SmallVector<Attribute> selectors(dimensions.size(), b.getStringAttr("index"));
          Value selected = b.create<wk::SliceOp>(loc,
              sliceType(*region, {}, {}),
              *region, indices, b.getArrayAttr(selectors));
          b.create<wk::CommitOp>(loc, initial, selected);
          return;
        }
        Value end = b.create<wk::ExtentOp>(loc, b.getIndexType(), *region, axis);
        auto loop = b.create<scf::ForOp>(loc, index(loc, 0), end, index(loc, 1));
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(loop.getBody());
        indices.push_back(loop.getInductionVar());
        fillAxis(axis + 1);
        indices.pop_back();
      };
      fillAxis(0);
      return success();
    }
    if (isa<cpu::QuantizeOp, cpu::QuantizedDotOp>(operation)) {
      auto implementation = implementations.lookup(operation);
      if (failed(implementation)) return failure();
      if (!(*implementation)->expand) return operation->emitError("selected Weft implementation has no expansion");
      SmallVector<Value> arguments;
      auto operands = operation->getOperands();
      if (isa<cpu::QuantizedDotOp>(operation)) operands = operands.take_front(2);
      for (Value operand : operands) {
        auto supplied = view(operand);
        if (failed(supplied)) return operation->emitError("implementation requires supplied operand views");
        if (viewAxes.at(operand).size() != shape((*supplied).getType()).size())
          return operation->emitError("encoded implementation requires its declared storage rank");
        for (auto [axis, native] : llvm::enumerate(viewAxes.at(operand)))
          if (native != static_cast<int64_t>(axis))
            return operation->emitError("encoded implementation requires its declared storage-axis order");
        arguments.push_back(*supplied);
      }
      auto results = (*implementation)->expand(b, operation, arguments, nextAxis);
      if (failed(results)) return failure();
      if (auto dot = dyn_cast<cpu::QuantizedDotOp>(operation)) {
        if (results->size() != 1) return dot.emitError("quantized implementation must supply its complete output value");
        return write(dot.getOutput(), results->front());
      }
      if (results->size() != operation->getNumResults())
        return operation->emitError("implementation results disagree with the structured operation");
      values.map(operation->getResults(), *results);
      return success();
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      if (isLocal(store.getMemref()) && store.getIndices().empty())
        return write(store.getMemref(), values.lookup(store.getValue()));
      auto region = view(store.getMemref());
      if (failed(region)) return failure();
      auto indices = projectIndices(store.getMemref(), store.getIndices(), shape((*region).getType()).size());
      Value selected = b.create<wk::SliceOp>(loc,
          sliceType(*region, {}, {}),
          *region, indices, b.getArrayAttr(SmallVector<Attribute>(indices.size(), b.getStringAttr("index"))));
      b.create<wk::CommitOp>(loc, values.lookup(store.getValue()), selected);
      return success();
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(operation)) return generic(genericOp);
    if (auto reduce = dyn_cast<cpu::ReduceOp>(operation)) return reduction(reduce);
    if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
      if (failed(prepareControlValues(conditional, conditional.getResults()))) return failure();
      auto written = writtenEnclosingLocals(conditional);
      if (failed(written)) return failure();
      auto &roots = *written;
      for (Value root : roots)
        if (failed(materializeLocal(root))) return failure();
      auto savedLocals = locals;
      auto nativeTypes = controlTypes(conditional.getResults());
      if (failed(nativeTypes)) return failure();
      auto types = *nativeTypes;
      unsigned explicitResults = types.size();
      SmallVector<Value> owners;
      for (Type type : types)
        owners.push_back(isa<wk::ValueType>(type) ? controlOwner(type, loc) : Value{});
      for (Value root : roots) {
        auto found = savedLocals.find(root);
        if (found == savedLocals.end()) {
          auto type = resultType(root);
          if (failed(type)) return failure();
          types.push_back(*type);
        } else types.push_back(found->second.value.getType());
      }
      auto target = b.create<scf::IfOp>(loc, types, values.lookup(conditional.getCondition()),
          !types.empty() || !conditional.getElseRegion().empty());
      for (bool then : {true, false}) {
        Block *destination = then ? target.thenBlock() : target.getElseRegion().empty() ? nullptr : target.elseBlock();
        if (!destination) continue;
        locals = savedLocals;
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(destination);
        Block *source = then ? conditional.thenBlock() : conditional.getElseRegion().empty() ? nullptr : conditional.elseBlock();
        if (source && failed(block(*source))) return failure();
        auto supplied = source
            ? controlValues(source->getTerminator()->getOperands(), conditional.getResults(), owners)
            : FailureOr<SmallVector<Value>>(SmallVector<Value>{});
        if (failed(supplied)) return failure();
        auto results = *supplied;
        for (Value root : roots) {
          if (!locals.count(root)) return conditional.emitError("conditional local result is not initialized on every branch");
          if (failed(materializeLocal(root))) return failure();
          if (savedLocals.count(root) && failed(checkCarry(root, savedLocals.lookup(root), conditional))) return failure();
          results.push_back(locals.lookup(root).value);
        }
        if (!types.empty()) b.create<scf::YieldOp>(loc, results);
      }
      locals = savedLocals;
      if (failed(mapControlValues(conditional.getResults(), target.getResults().take_front(explicitResults)))) return failure();
      for (auto [root, value] : llvm::zip(roots, target.getResults().drop_front(explicitResults)))
        if (locals.count(root)) locals[root].value = value;
        else if (failed(write(root, value))) return failure();
      return success();
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      SmallVector<Value> boundaries(loop.getRegionIterArgs());
      llvm::append_range(boundaries, loop.getResults());
      if (failed(prepareControlValues(loop, boundaries))) return failure();
      auto supplied = controlValues(loop.getInitArgs(), loop.getRegionIterArgs());
      if (failed(supplied)) return failure();
      auto initial = *supplied;
      ownControlInputs(initial, loop.getInitArgs(), loop.getRegionIterArgs(), loop);
      unsigned explicitResults = initial.size();
      SmallVector<Value> roots;
      auto written = writtenEnclosingLocals(loop);
      if (failed(written)) return failure();
      for (Value root : *written) {
        if (!locals.count(root)) continue;
        if (failed(materializeLocal(root))) return failure();
        roots.push_back(root);
        initial.push_back(locals.lookup(root).value);
      }
      auto savedLocals = locals;
      auto target = b.create<scf::ForOp>(loc, values.lookup(loop.getLowerBound()),
          values.lookup(loop.getUpperBound()), values.lookup(loop.getStep()), initial);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(target.getBody());
        values.map(loop.getInductionVar(), target.getInductionVar());
        if (failed(mapControlValues(loop.getRegionIterArgs(), target.getRegionIterArgs().take_front(explicitResults)))) return failure();
        for (auto [root, value] : llvm::zip(roots, target.getRegionIterArgs().drop_front(explicitResults)))
          locals[root].value = value;
        if (failed(block(*loop.getBody()))) return failure();
        auto supplied = controlValues(loop.getBody()->getTerminator()->getOperands(), loop.getRegionIterArgs(),
            target.getRegionIterArgs().take_front(explicitResults));
        if (failed(supplied)) return failure();
        auto results = *supplied;
        for (Value root : roots) {
          if (failed(checkCarry(root, savedLocals.lookup(root), loop))) return failure();
          results.push_back(locals.lookup(root).value);
        }
        if (!initial.empty()) b.create<scf::YieldOp>(loc, results);
      }
      locals = std::move(savedLocals);
      if (failed(mapControlValues(loop.getResults(), target.getResults().take_front(explicitResults)))) return failure();
      for (auto [root, value] : llvm::zip(roots, target.getResults().drop_front(explicitResults)))
        locals[root].value = value;
      return success();
    }
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      SmallVector<Value> boundaries(loop.getBeforeArguments());
      llvm::append_range(boundaries, loop.getAfterArguments());
      llvm::append_range(boundaries, loop.getResults());
      if (failed(prepareControlValues(loop, boundaries))) return failure();
      auto supplied = controlValues(loop.getInits(), loop.getBeforeArguments());
      auto nativeTypes = controlTypes(loop.getResults());
      if (failed(supplied) || failed(nativeTypes)) return failure();
      auto initial = *supplied;
      ownControlInputs(initial, loop.getInits(), loop.getBeforeArguments(), loop);
      unsigned explicitInputs = initial.size();
      auto types = *nativeTypes;
      unsigned explicitResults = types.size();
      if (initial.size() != types.size() ||
          !llvm::all_of(llvm::zip(initial, types), [](auto pair) {
            Type left = std::get<0>(pair).getType(), right = std::get<1>(pair);
            return left == right || (isa<wk::ValueType>(left) && isa<wk::ValueType>(right) &&
                element(left) == element(right) && shape(left) == shape(right));
          }))
        return loop.emitError("Weft while requires one stable physical state schema across both regions");
      for (auto [value, type] : llvm::zip(initial, types)) type = value.getType();
      auto written = writtenEnclosingLocals(loop);
      if (failed(written)) return failure();
      SmallVector<Value> roots;
      for (Value root : *written)
        if (locals.count(root)) {
          if (failed(materializeLocal(root))) return failure();
          roots.push_back(root);
          initial.push_back(locals.lookup(root).value);
          types.push_back(locals.lookup(root).value.getType());
        }
      auto savedLocals = locals;
      auto target = b.create<scf::WhileOp>(loc, types, initial);
      Block &before = target.getBefore().emplaceBlock();
      Block &after = target.getAfter().emplaceBlock();
      for (Value value : initial) before.addArgument(value.getType(), loc);
      for (Type type : types) after.addArgument(type, loc);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&before);
        if (failed(mapControlValues(loop.getBeforeArguments(), before.getArguments().take_front(explicitInputs)))) return failure();
        for (auto [root, value] : llvm::zip(roots, before.getArguments().drop_front(explicitInputs)))
          locals[root].value = value;
        if (failed(block(loop.getBefore().front()))) return failure();
        auto condition = cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
        auto supplied = controlValues(condition.getArgs(), loop.getAfterArguments(),
            before.getArguments().take_front(explicitInputs));
        if (failed(supplied)) return failure();
        auto results = *supplied;
        for (Value root : roots) {
          if (failed(checkCarry(root, savedLocals.lookup(root), loop))) return failure();
          results.push_back(locals.lookup(root).value);
        }
        b.create<scf::ConditionOp>(loc, values.lookup(condition.getCondition()), results);
      }
      locals = savedLocals;
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&after);
        if (failed(mapControlValues(loop.getAfterArguments(), after.getArguments().take_front(explicitResults)))) return failure();
        for (auto [root, value] : llvm::zip(roots, after.getArguments().drop_front(explicitResults)))
          locals[root].value = value;
        if (failed(block(loop.getAfter().front()))) return failure();
        auto supplied = controlValues(loop.getAfter().front().getTerminator()->getOperands(), loop.getBeforeArguments(),
            after.getArguments().take_front(explicitResults));
        if (failed(supplied)) return failure();
        auto results = *supplied;
        for (Value root : roots) {
          if (failed(checkCarry(root, savedLocals.lookup(root), loop))) return failure();
          results.push_back(locals.lookup(root).value);
        }
        b.create<scf::YieldOp>(loc, results);
      }
      locals = std::move(savedLocals);
      if (failed(mapControlValues(loop.getResults(), target.getResults().take_front(explicitResults)))) return failure();
      for (auto [root, value] : llvm::zip(roots, target.getResults().drop_front(explicitResults)))
        locals[root].value = value;
      return success();
    }
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      if (isLocal(load.getMemref()) && load.getIndices().empty()) {
        auto value = read(load.getMemref());
        if (failed(value)) return failure();
        values.map(load.getResult(), *value);
        return success();
      }
      auto region = view(load.getMemref());
      if (failed(region)) return failure();
      auto indices = projectIndices(load.getMemref(), load.getIndices(), shape((*region).getType()).size());
      SmallVector<Attribute> selectors(indices.size(), b.getStringAttr("index"));
      Value selected = b.create<wk::SliceOp>(loc, sliceType(*region, {}, {}),
          *region, indices, b.getArrayAttr(selectors));
      values.map(load.getResult(), b.create<wk::AdmitOp>(loc, load.getType(), selected));
      return success();
    }
    if (operation->getNumResults() != 1) return operation->emitError("CPU task operation has no Weft representation");
    auto value = expression(operation, values);
    if (failed(value)) return failure();
    values.map(operation->getResult(0), *value);
    return success();
  }

  func::FuncOp sourceFunction;
  std::unique_ptr<cpu::StorageAnalysis> storage;
  cpu::TasksOp currentTask;
  cpu::AxisRelations relations;
  ModuleOp output;
  OpBuilder b;
  IRMapping values;
  llvm::DenseMap<Value, SmallVector<int64_t>> viewAxes;
  llvm::DenseMap<Value, LocalValue> locals;
  llvm::DenseMap<Value, Value> localReferences;
  llvm::DenseSet<Value> controlOwners;
  SmallVector<Value> shapeValues;
  llvm::DenseMap<Value, Value> operandReads;
  llvm::DenseMap<Value, Value> readOnlySupplies;
  const llvm::DenseMap<Value, intent::QuantFormat> &formats;
  const cpu::ImplementationRegistry &implementations;
  llvm::DenseMap<int64_t, int64_t> extentIds;
  llvm::DenseMap<int64_t, int64_t> axisProjection;
  int64_t nextAxis;
};

TaskLowering::TaskLowering(func::FuncOp function, ModuleOp output,
    const llvm::DenseMap<Value, intent::QuantFormat> &formats,
    const cpu::ImplementationRegistry &implementations)
    : implementation(std::make_unique<Impl>(function, output, formats, implementations)) {}

TaskLowering::~TaskLowering() = default;

SmallVector<Value> TaskLowering::shapeArguments(func::FuncOp function, OpBuilder &builder) {
  return implementation->shapeArguments(function, builder);
}

LogicalResult TaskLowering::lower(cpu::TasksOp task, StringRef name,
                                  SmallVectorImpl<unsigned> &argumentPositions) {
  return implementation->lower(task, name, argumentPositions);
}

} // namespace intent::weft_provider
