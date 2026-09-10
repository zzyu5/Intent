#include "Intent/Target/Weft/Transforms/Passes.h"
#include "Quantization.h"
#include "Intent/Target/Weft/Serialization/Serializer.h"
#include "Intent/Dialect/CPU/Analysis/AxisRelations.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/JSON.h"

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

struct LocalValue {
  Value value;
  SmallVector<OpFoldResult> sizes;
};

FailureOr<llvm::json::Object> taskABI(wk::KernelOp kernel) {
  llvm::json::Array arguments, symbols;
  for (Attribute symbol : kernel.getShapeSymbols())
    symbols.push_back(cast<StringAttr>(symbol).getValue().str());
  for (auto [index, argument] : llvm::enumerate(kernel.getBody().front().getArguments())) {
    auto view = cast<wk::ViewType>(argument.getType());
    auto encoding = cast<wk::EncodingType>(view.getEncoding());
    int64_t bytes, elements, alignment;
    std::string scalar;
    if (encoding.getKind() == "dense") {
      if (encoding.getFamily() == "f32") { scalar = "float"; bytes = 4; }
      else if (encoding.getFamily() == "i64") { scalar = "int64_t"; bytes = 8; }
      else if (encoding.getFamily() == "u8" || encoding.getFamily() == "i8") {
        scalar = encoding.getFamily() == "u8" ? "uint8_t" : "int8_t"; bytes = 1;
      } else return kernel.emitError("CPU task ABI has no native dense element representation"), failure();
      elements = 1; alignment = bytes;
    } else {
      wk::EncodingDeclOp declaration;
      for (auto candidate : kernel->getParentOfType<ModuleOp>().getOps<wk::EncodingDeclOp>())
        if (candidate.getSymName() == encoding.getFamily()) declaration = candidate;
      if (encoding.getKind() != "base" || !declaration)
        return kernel.emitError("CPU task ABI requires a declared base encoding"), failure();
      scalar = "uint8_t"; bytes = declaration.getStorageBits() / 8;
      elements = declaration.getElements(); alignment = declaration.getAlignment();
    }
    StringRef access = cast<StringAttr>(kernel.getArgAccess()[index]).getValue();
    bool writable = access == "write" || access == "readwrite";
    int64_t alias = kernel.getArgAliasSets()[index];
    bool restricted = alias >= 0 && llvm::count(kernel.getArgAliasSets(), alias) == 1;
    llvm::json::Array dimensions;
    for (int64_t extent : view.getShape().asArrayRef())
      dimensions.push_back(extent >= 0 ? std::to_string(extent)
          : cast<StringAttr>(kernel.getShapeSymbols()[-extent - 1]).getValue().str());
    arguments.push_back(llvm::json::Object{
        {"name", cast<StringAttr>(kernel.getArgNames()[index]).getValue().str()},
        {"c_type", (writable ? "" : "const ") + scalar + (restricted ? " *restrict" : " *")},
        {"encoding", encoding.getLayoutIdentity().str()}, {"shape", std::move(dimensions)},
        {"storage_bytes", bytes}, {"record_elements", elements}, {"alignment", alignment},
        {"interleave_rows", 0}, {"alias_set", alias}, {"writable", writable}});
  }
  return llvm::json::Object{{"symbol", kernel.getSymName().str()},
      {"arguments", std::move(arguments)}, {"shape_parameters", std::move(symbols)}};
}

class TaskLowering {
public:
  TaskLowering(func::FuncOp function, ModuleOp output,
               const llvm::DenseMap<Value, intent::QuantFormat> &formats,
               const cpu::ImplementationRegistry &implementations)
      : analysis(function), relations(function), output(output), b(output.getContext()),
        formats(formats), implementations(implementations) {
    auto interface = function->getAttrOfType<cpu::InterfaceAttr>("intent_cpu.interface");
    for (auto [argument, schema] : llvm::zip(function.getArguments(), interface.getArguments())) {
      auto view = dyn_cast<cpu::ViewArgumentAttr>(schema);
      if (!view) continue;
      auto memory = cast<MemRefType>(argument.getType());
      for (unsigned axis = 0; axis < memory.getRank(); ++axis)
        if (memory.isDynamicDim(axis)) extentIds.try_emplace(view.getDimensions()[axis], extentIds.size() + 1);
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
    auto interface = function->getAttrOfType<cpu::InterfaceAttr>("intent_cpu.interface");
    for (BlockArgument argument : function.getArguments()) {
      auto memory = dyn_cast<MemRefType>(argument.getType());
      if (!memory) continue;
      auto ids = cast<cpu::ViewArgumentAttr>(interface.getArguments()[argument.getArgNumber()]).getDimensions();
      for (auto [axis, extent] : llvm::enumerate(memory.getShape()))
        if (ShapedType::isDynamic(extent))
          result[extentIds.at(ids[axis]) - 1] = builder.create<memref::DimOp>(function.getLoc(), argument, axis);
    }
    return result;
  }

  LogicalResult lower(cpu::TasksOp tasks, StringRef name, ArrayRef<unsigned> argumentPositions,
                      llvm::json::Object &abi) {
    values.clear(); locals.clear();
    Location loc = tasks.getLoc();
    SmallVector<Attribute> names, accesses, symbols;
    SmallVector<int64_t> aliases;
    SmallVector<Type> types;
    auto memoryAccesses = analysis.accesses(tasks);
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
            auto symbol = dimensionSymbol(capture, axis);
            if (!symbol) return tasks.emitError("captured view extent has no external shape binding");
            dimensions.push_back(-*symbol);
          } else dimensions.push_back(extent);
        }
        auto format = formats.find(analysis.storageRoot(capture));
        if (format != formats.end()) {
          int64_t bytes = format->second == intent::QuantFormat::Q4K ? 144 : 292;
          if (memory.getShape().back() != bytes)
            return tasks.emitError("encoded capture requires a statically complete contiguous record byte span");
          dimensions.back() = 256;
        }
        types.push_back(wk::ViewType::get(b.getContext(), encoding(capture), array(dimensions), array(ids)));
        Value storage = analysis.storageRoot(capture);
        bool reads = false, writes = false;
        for (auto access : memoryAccesses)
          if (analysis.storageRoot(access.memory) == storage) {
            reads |= access.read; writes |= access.write;
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
      b.create<wk::SymbolOp>(loc, b.getIndexType(), cast<StringAttr>(symbol),
          b.getStringAttr("shape"), array({}));
    b.create<wk::RootDomainOp>(loc,
        wk::DomainType::get(b.getContext(), "root", 0, -1, 0, "root", "exact"));
    for (auto [position, target] : llvm::zip(argumentPositions, body->getArguments())) {
      Value source = tasks.getBody().front().getArgument(position);
      if (isa<MemRefType>(source.getType())) values.map(source, target);
      else {
        auto type = cast<wk::ViewType>(target.getType());
        Value scalar = b.create<wk::SliceOp>(loc,
            wk::SliceType::get(b.getContext(), type.getEncoding(), array({}), array({})),
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
    if (failed(verify(kernel))) return failure();
    auto interface = taskABI(kernel);
    if (failed(interface)) return failure();
    abi = std::move(*interface);
    return success();
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
  wk::EncodingType encoding(Value memory = {}) {
    if (!memory) return dense(b.getF32Type());
    auto format = formats.find(analysis.storageRoot(memory));
    return format == formats.end() ? dense(cast<MemRefType>(memory.getType()).getElementType())
                                  : quantEncoding(b, format->second);
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
        auto symbol = dimensionSymbol(memory, axis);
        if (!symbol) return emitError(memory.getLoc(), "private value extent has no explicit shape binding"), failure();
        dimensions.push_back(-*symbol);
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
    return a && c && *a == *c;
  }
  Value localRoot(Value memory) {
    while (true) {
      if (auto view = memory.getDefiningOp<memref::SubViewOp>()) memory = view.getSource();
      else if (auto cast = memory.getDefiningOp<memref::CastOp>()) memory = cast.getSource();
      else return memory;
    }
  }
  bool isLocal(Value memory) {
    return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(localRoot(memory).getDefiningOp());
  }

  std::optional<int64_t> extentSymbol(Value extent) {
    if (auto cast = extent.getDefiningOp<arith::IndexCastOp>(); cast &&
        (cast.getIn().getType().isInteger(64) || cast.getOut().getType().isInteger(64)))
      return extentSymbol(cast.getIn());
    if (auto maximum = extent.getDefiningOp<arith::MaxSIOp>()) {
      // A memref extent is nonnegative; max(extent, 0) is the same bound.
      if (matchPattern(maximum.getLhs(), m_Zero())) return extentSymbol(maximum.getRhs());
      if (matchPattern(maximum.getRhs(), m_Zero())) return extentSymbol(maximum.getLhs());
    }
    if (auto argument = dyn_cast<BlockArgument>(extent)) {
      auto tasks = dyn_cast<cpu::TasksOp>(argument.getOwner()->getParentOp());
      if (tasks && argument.getArgNumber())
        return extentSymbol(tasks.getCaptures()[argument.getArgNumber() - 1]);
      return std::nullopt;
    }
    auto dimension = extent.getDefiningOp<memref::DimOp>();
    if (!dimension || !dimension.getConstantIndex()) return std::nullopt;
    return dimensionSymbol(dimension.getSource(), *dimension.getConstantIndex());
  }

  std::optional<int64_t> dimensionSymbol(Value memory, unsigned axis) {
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) return dimensionSymbol(cast.getSource(), axis);
    if (auto argument = dyn_cast<BlockArgument>(memory)) {
      if (auto tasks = dyn_cast<cpu::TasksOp>(argument.getOwner()->getParentOp()))
        return dimensionSymbol(tasks.getCaptures()[argument.getArgNumber() - 1], axis);
      if (isa<func::FuncOp>(argument.getOwner()->getParentOp()) &&
          cast<MemRefType>(memory.getType()).isDynamicDim(axis)) {
        auto interface = argument.getOwner()->getParentOp()->getAttrOfType<cpu::InterfaceAttr>("intent_cpu.interface");
        auto dimension = cast<cpu::ViewArgumentAttr>(interface.getArguments()[argument.getArgNumber()]).getDimensions()[axis];
        return extentIds.at(dimension);
      }
    }
    if (auto subview = memory.getDefiningOp<memref::SubViewOp>()) {
      unsigned projected = 0;
      for (unsigned original = 0; original < subview.getSourceType().getRank(); ++original)
        if (!subview.getDroppedDims().test(original) && projected++ == axis) {
          auto size = dyn_cast<Value>(subview.getMixedSizes()[original]);
          return size ? extentSymbol(size) : std::nullopt;
        }
    }
    if (isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(memory.getDefiningOp())) {
      auto type = cast<MemRefType>(memory.getType());
      if (type.isDynamicDim(axis))
        return extentSymbol(memory.getDefiningOp()->getOperand(type.getDynamicDimIndex(axis)));
    }
    return std::nullopt;
  }

  FailureOr<Value> view(Value memory) {
    if (values.contains(memory)) return values.lookup(memory);
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) return view(cast.getSource());
    auto operation = memory.getDefiningOp<memref::SubViewOp>();
    if (!operation) return emitError(memory.getLoc(), "CPU memory has no explicit Weft view relation: ") << memory, failure();
    auto base = view(operation.getSource());
    if (failed(base)) return failure();
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
    auto dropped = operation.getDroppedDims();
    auto baseShape = shape((*base).getType());
    for (unsigned axis = 0; axis < baseShape.size(); ++axis) {
      if (dropped.test(axis)) continue;
      auto size = operation.getMixedSizes()[axis];
      bool full = baseShape[axis] >= 0 ? sameBound(size, b.getIndexAttr(baseShape[axis]))
          : isa<Value>(size) && extentSymbol(cast<Value>(size)) == -baseShape[axis];
      projectionOnly &= sameBound(operation.getMixedOffsets()[axis], b.getIndexAttr(0)) && full;
    }
    if (projectionOnly) {
      SmallVector<Attribute> selectors;
      SmallVector<Value> indices;
      SmallVector<int64_t> keptShape, keptAxes;
      auto baseAxes = axes((*base).getType());
      for (unsigned axis = 0; axis < baseShape.size(); ++axis) {
        selectors.push_back(b.getStringAttr(dropped.test(axis) ? "index" : "all"));
        if (dropped.test(axis)) {
          auto offset = operation.getMixedOffsets()[axis];
          indices.push_back(isa<Attribute>(offset) ? index(operation.getLoc(), cast<IntegerAttr>(cast<Attribute>(offset)).getInt())
                                                   : values.lookup(cast<Value>(offset)));
        } else { keptShape.push_back(baseShape[axis]); keptAxes.push_back(baseAxes[axis]); }
      }
      Value selected = b.create<wk::SliceOp>(operation.getLoc(),
          wk::SliceType::get(b.getContext(), encoding(memory), array(keptShape), array(keptAxes)),
          *base, indices, b.getArrayAttr(selectors));
      values.map(memory, selected);
      return selected;
    }
    append(operation.getMixedOffsets(), offsets);
    append(operation.getMixedSizes(), extents);
    auto format = formats.find(analysis.storageRoot(memory));
    if (format != formats.end()) {
      int64_t bytes = format->second == intent::QuantFormat::Q4K ? 144 : 292;
      if (offsets.back() != 0 || extents.back() != bytes)
        return operation.emitError("encoded view projection must retain its complete record bytes"), failure();
      extents.back() = 256;
    }
    auto ids = axes((*base).getType());
    for (auto [axis, extent] : llvm::enumerate(extents)) {
      if (extent >= 0) dimensions.push_back(extent);
      else {
        auto size = dyn_cast<Value>(operation.getMixedSizes()[axis]);
        auto symbol = size ? extentSymbol(size) : std::nullopt;
        if (!symbol) return operation.emitError("dynamic Weft subview extent has no shape binding"), failure();
        dimensions.push_back(-*symbol);
      }
    }
    Value selected = b.create<wk::SubviewOp>(operation.getLoc(),
        wk::SliceType::get(b.getContext(), encoding(memory), array(dimensions), array(ids)),
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
          wk::SliceType::get(b.getContext(), encoding(memory), array(keptShape), array(keptAxes)),
          selected, indices, b.getArrayAttr(selectors));
    }
    values.map(memory, selected);
    return selected;
  }

  FailureOr<Value> read(Value memory) {
    if (!isa<MemRefType>(memory.getType())) return values.lookup(memory);
    if (!isLocal(memory)) {
      auto previous = operandReads.find(memory);
      if (previous != operandReads.end()) return previous->second;
      auto region = view(memory);
      if (failed(region)) return failure();
      Type element = cast<MemRefType>(memory.getType()).getElementType();
      if (element.isIndex()) element = IntegerType::get(b.getContext(), 64, IntegerType::Signed);
      Value loaded = b.create<wk::AdmitOp>(memory.getLoc(),
          valueType(element, shape((*region).getType()), axes((*region).getType())), *region);
      operandReads[memory] = loaded;
      return loaded;
    }
    Value root = localRoot(memory);
    auto found = locals.find(root);
    if (found == locals.end()) {
      emitError(memory.getLoc(), "Weft local value is read before a dominating complete supply");
      return failure();
    }
    if (memory == root) {
      auto type = cast<MemRefType>(root.getType());
      unsigned dynamic = 0;
      for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
        OpFoldResult full = ShapedType::isDynamic(extent)
            ? OpFoldResult(root.getDefiningOp()->getOperand(dynamic++))
            : OpFoldResult(b.getIndexAttr(extent));
        if (!sameBound(found->second.sizes[axis], full))
          return emitError(memory.getLoc(), "Weft local root read requires a complete initialized region"), failure();
      }
      return found->second.value;
    }
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) return read(cast.getSource());
    auto projection = localProjection(memory, found->second);
    if (failed(projection)) return failure();
    return Value(b.create<wk::ExtractOp>(memory.getLoc(), projection->type, found->second.value,
                                        projection->indices, b.getArrayAttr(projection->selectors)));
  }

  struct LocalProjection {
    Type type;
    SmallVector<Value> indices;
    SmallVector<Attribute> selectors;
  };

  FailureOr<LocalProjection> localProjection(Value memory, const LocalValue &state) {
    Value root = localRoot(memory);
    auto rootType = cast<MemRefType>(root.getType());
    SmallVector<Value> origins(rootType.getRank());
    SmallVector<bool> zeroOrigins(rootType.getRank(), true);
    for (Value &origin : origins) origin = index(memory.getLoc(), 0);
    SmallVector<OpFoldResult> sizes(state.sizes);
    SmallVector<unsigned> kept;
    for (unsigned axis = 0; axis < rootType.getRank(); ++axis) kept.push_back(axis);
    SmallVector<memref::SubViewOp> chain;
    for (Value current = memory; current != root;) {
      if (auto cast = current.getDefiningOp<memref::CastOp>()) current = cast.getSource();
      else if (auto view = current.getDefiningOp<memref::SubViewOp>()) { chain.push_back(view); current = view.getSource(); }
      else return emitError(memory.getLoc(), "local window has no composed subview relation"), failure();
    }
    for (auto view : llvm::reverse(chain)) {
      SmallVector<unsigned> next;
      for (unsigned axis = 0; axis < kept.size(); ++axis) {
        unsigned original = kept[axis];
        if (!sameBound(view.getMixedStrides()[axis], b.getIndexAttr(1)))
          return view.emitError("private windows require unit coordinate steps"), failure();
        OpFoldResult offset = view.getMixedOffsets()[axis];
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
    for (auto [position, axis] : llvm::enumerate(sourceAxes)) {
      auto found = llvm::find(rootAxes, axis);
      if (found == rootAxes.end()) return emitError(memory.getLoc(), "private state axis lost its destination relation"), failure();
      unsigned original = found - rootAxes.begin();
      if (!llvm::is_contained(kept, original)) {
        result.selectors.push_back(b.getStringAttr("index")); result.indices.push_back(origins[original]);
        continue;
      }
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
      Type unsignedIndex = IntegerType::get(b.getContext(), 64, IntegerType::Unsigned);
      auto laneType = cast<wk::ValueType>(valueType(unsignedIndex, {*size}, {axis}));
      Value lane = b.create<wk::IotaOp>(memory.getLoc(), laneType, 0, *size);
      Value base = b.create<wk::CastOp>(memory.getLoc(), unsignedIndex, origins[original]);
      Value coordinate = b.create<wk::BinaryOp>(memory.getLoc(), laneType, lane, base, "add");
      result.selectors.push_back(b.getStringAttr("gather")); result.indices.push_back(coordinate);
      dimensions.push_back(*size); ids.push_back(axis);
    }
    result.type = valueType(element(state.value.getType()), dimensions, ids);
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

  LogicalResult write(Value memory, Value value, SmallVector<OpFoldResult> sizes = {}) {
    if (!isLocal(memory)) {
      auto region = view(memory);
      if (failed(region)) return failure();
      auto aligned = alignValue(value, valueType(element(value.getType()), shape((*region).getType()), axes((*region).getType())), memory.getLoc());
      if (failed(aligned)) return failure();
      b.create<wk::CommitOp>(memory.getLoc(), *aligned, *region);
      return success();
    }
    Value root = localRoot(memory);
    if (memory != root) {
      if (auto cast = memory.getDefiningOp<memref::CastOp>()) return write(cast.getSource(), value, sizes);
      auto projection = memory.getDefiningOp<memref::SubViewOp>();
      auto found = locals.find(root);
      if (found != locals.end()) {
        auto projected = localProjection(memory, found->second);
        if (failed(projected)) return failure();
        auto aligned = alignValue(value, projected->type, memory.getLoc());
        if (failed(aligned)) return failure();
        found->second.value = b.create<wk::UpdateOp>(memory.getLoc(), found->second.value.getType(),
            found->second.value, *aligned, projected->indices, b.getArrayAttr(projected->selectors));
        return success();
      }
      if (!projection || projection.getSource() != root || projection.getDroppedDims().any() ||
          llvm::any_of(projection.getMixedOffsets(), [&](OpFoldResult offset) { return !sameBound(offset, b.getIndexAttr(0)); }))
        return emitError(memory.getLoc(), "Weft private supply must define one zero-based complete active region");
      sizes = projection.getMixedSizes();
    }
    if (sizes.empty()) {
      auto type = cast<MemRefType>(root.getType());
      Operation *allocation = root.getDefiningOp();
      unsigned dynamic = 0;
      for (int64_t extent : type.getShape()) {
        if (ShapedType::isDynamic(extent)) sizes.push_back(allocation->getOperand(dynamic++));
        else sizes.push_back(b.getIndexAttr(extent));
      }
    }
    auto type = resultType(memory);
    if (failed(type)) return failure();
    auto aligned = alignValue(value, *type, memory.getLoc());
    if (failed(aligned)) return failure();
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
    if (isa<arith::IndexCastOp, arith::SIToFPOp, arith::FPToSIOp, arith::ExtFOp, arith::TruncFOp>(operation))
      return Value(b.create<wk::CastOp>(loc, valueType(operation->getResult(0).getType(),
          shape(operands[0].getType()), axes(operands[0].getType())), operands[0]));
    if (isa<math::RsqrtOp>(operation)) kind = "rsqrt";
    else if (isa<math::ExpOp>(operation)) kind = "exp";
    else if (isa<arith::NegFOp>(operation)) kind = "neg";
    if (!kind.empty()) return Value(b.create<wk::UnaryOp>(loc, operands[0].getType(), operands[0], kind));
    operation->emitError("CPU operation has no Weft numerical representation");
    return failure();
  }

  FailureOr<Value> mappedInput(Value input, AffineMap map, ArrayRef<int64_t> loopAxes) {
    auto loaded = read(input);
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
      auto lhs = mappedInput(operation.getInputs()[0], maps[0], loopAxes);
      auto rhs = mappedInput(operation.getInputs()[1], maps[1], loopAxes);
      auto initial = read(destination);
      if (failed(lhs) || failed(rhs) || failed(initial)) return failure();
      auto definition = (*initial).getDefiningOp<wk::NewOp>();
      if (!definition || !matchPattern(definition->getOperand(0), m_PosZeroFloat()))
        return operation.emitError("Weft contraction requires an explicit zero-initialized partial; nonzero fused accumulation has no equivalent canonical operation");
      int64_t reduction = loopAxes[2];
      Value term = b.create<wk::OuterContractOp>(operation.getLoc(), (*initial).getType(),
          *lhs, *rhs, array({reduction}), TypeAttr::get(b.getF32Type()));
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
      auto combine = body.getTerminator()->getOperand(0).getDefiningOp();
      if (!combine || !accumulator.hasOneUse() || combine->getNumOperands() != 2 ||
          !llvm::is_contained(combine->getOperands(), accumulator))
        return operation.emitError("Weft structured reduction requires a closed accumulator combine");
      StringRef kind;
      if (isa<arith::AddFOp, arith::AddIOp>(combine)) kind = "add";
      else if (isa<arith::MaximumFOp>(combine)) kind = "maximum";
      else if (isa<arith::MinimumFOp>(combine)) kind = "minimum";
      else if (isa<arith::MaxNumFOp>(combine)) kind = "max";
      else if (isa<arith::MinNumFOp>(combine)) kind = "min";
      else if (isa<arith::OrIOp>(combine) && accumulator.getType().isInteger(1)) kind = "or";
      else if (isa<arith::AndIOp>(combine) && accumulator.getType().isInteger(1)) kind = "and";
      else return operation.emitError("Weft reduction combine is not implemented");
      for (Operation &nested : body.without_terminator()) {
        if (&nested == combine) continue;
        auto value = expression(&nested, mapping);
        if (failed(value)) return failure();
        mapping.map(nested.getResult(0), *value);
      }
      Value contribution = mapping.lookup(combine->getOperand(combine->getOperand(0) == accumulator ? 1 : 0));
      auto ids = axes(contribution.getType());
      auto position = llvm::find(ids, loopAxes[reductions[0]]);
      if (position == ids.end()) return operation.emitError("Weft reduction lost its current logical axis relation");
      auto reduced = reduceValue(operation.getLoc(), contribution, position - ids.begin(), kind);
      auto initial = read(destination);
      if (failed(reduced) || failed(initial)) return failure();
      auto result = binary(operation.getLoc(), *initial, *reduced, kind);
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
    Operation *combine = body.getTerminator()->getOperand(0).getDefiningOp();
    auto initial = operation.getInitial().getDefiningOp<arith::ConstantOp>();
    auto identity = initial ? dyn_cast<FloatAttr>(initial.getValue()) : FloatAttr();
    bool additive = combine && isa<arith::AddFOp>(combine);
    bool maximum = combine && isa<arith::MaxNumFOp>(combine);
    Value accumulator = body.getArgument(0);
    if (!operation.getOrder().getAdjacentReassociation() || (!additive && !maximum) ||
        !identity || !accumulator.hasOneUse() ||
        !llvm::is_contained(combine->getOperands(), accumulator) ||
        (additive && !identity.getValue().isZero()) ||
        (maximum && !(identity.getValue().isInfinity() && identity.getValue().isNegative())))
      return operation.emitError("Weft reduction requires a closed additive/maximumNumber identity and accumulator combine");
    IRMapping mapping;
    SmallVector<int64_t> loopAxes(cast<AffineMapAttr>(operation.getIndexingMaps()[0]).getValue().getNumDims());
    for (int64_t &axis : loopAxes) axis = nextAxis++;
    for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
      auto value = mappedInput(input, cast<AffineMapAttr>(operation.getIndexingMaps()[number]).getValue(), loopAxes);
      if (failed(value)) return failure();
      mapping.map(body.getArgument(number + 1), *value);
    }
    for (Operation &nested : body.without_terminator()) {
      if (&nested == combine) continue;
      auto value = expression(&nested, mapping);
      if (failed(value)) return failure();
      mapping.map(nested.getResult(0), *value);
    }
    Value contribution = mapping.lookup(combine->getOperand(0) == accumulator
        ? combine->getOperand(1) : combine->getOperand(0));
    if (shape(contribution.getType()).size() != 1)
      return operation.emitError("Weft reduction requires one retained logical input axis");
    Value result = b.create<wk::ReduceOp>(operation.getLoc(), operation.getResult().getType(), contribution,
        additive ? "add" : "max", 0);
    if (maximum) {
      auto seeded = binary(operation.getLoc(), values.lookup(operation.getInitial()), result, "max");
      if (failed(seeded)) return failure();
      result = *seeded;
    }
    values.map(operation.getResult(), result);
    return success();
  }

  LogicalResult block(Block &source) {
    for (Operation &operation : source.without_terminator())
      if (failed(lower(&operation))) return failure();
    return success();
  }

  SmallVector<Value> writtenEnclosingLocals(Operation *scope) {
    SmallVector<Value> result;
    for (auto access : analysis.accesses(scope)) {
      Value root = localRoot(access.memory);
      if (access.write && isLocal(root) &&
          !scope->isProperAncestor(root.getDefiningOp()) &&
          !llvm::is_contained(result, root))
        result.push_back(root);
    }
    return result;
  }

  LogicalResult checkCarry(Value root, const LocalValue &before, Operation *scope) {
    auto found = locals.find(root);
    if (found == locals.end() || found->second.value.getType() != before.value.getType() ||
        found->second.sizes.size() != before.sizes.size() ||
        !llvm::all_of(llvm::zip(found->second.sizes, before.sizes), [&](auto bounds) {
          return sameBound(std::get<0>(bounds), std::get<1>(bounds));
        }))
      return scope->emitError("Weft control carry must preserve its complete initialized region and type");
    return success();
  }

  LogicalResult lower(Operation *operation) {
    operandReads.clear();
    Location loc = operation->getLoc();
    if (isa<memref::AllocOp, memref::AllocaOp>(operation)) return success();
    if (isa<memref::SubViewOp, memref::CastOp>(operation)) {
      Value result = operation->getResult(0);
      if (!isLocal(result) && failed(view(result))) return failure();
      return success();
    }
    if (auto dealloc = dyn_cast<memref::DeallocOp>(operation)) { locals.erase(localRoot(dealloc.getMemref())); return success(); }
    if (auto dimension = dyn_cast<memref::DimOp>(operation)) {
      auto axis = dimension.getConstantIndex();
      if (!axis) return dimension.emitError("Weft dimension requires a static axis position");
      if (isLocal(dimension.getSource())) {
        Value memory = dimension.getSource();
        while (auto cast = memory.getDefiningOp<memref::CastOp>()) memory = cast.getSource();
        OpFoldResult extent;
        if (auto subview = memory.getDefiningOp<memref::SubViewOp>()) {
          unsigned projected = 0;
          for (unsigned original = 0; original < subview.getSourceType().getRank(); ++original)
            if (!subview.getDroppedDims().test(original) && projected++ == *axis)
              extent = subview.getMixedSizes()[original];
        } else {
          auto type = cast<MemRefType>(memory.getType());
          extent = type.isDynamicDim(*axis)
              ? OpFoldResult(memory.getDefiningOp()->getOperand(type.getDynamicDimIndex(*axis)))
              : OpFoldResult(b.getIndexAttr(type.getDimSize(*axis)));
        }
        if (!extent) return dimension.emitError("local dimension has no explicit allocation or subview extent");
        values.map(dimension.getResult(), isa<Attribute>(extent)
            ? index(loc, cast<IntegerAttr>(cast<Attribute>(extent)).getInt())
            : values.lookup(cast<Value>(extent)));
        return success();
      }
      auto region = view(dimension.getSource());
      if (failed(region)) return failure();
      values.map(dimension.getResult(), b.create<wk::ExtentOp>(loc, b.getIndexType(), *region, *axis));
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
        auto current = locals.find(root);
        if (root == copy.getTarget() && current != locals.end() && isa<wk::ValueType>(*target)) {
          current->second.value = b.create<wk::UpdateOp>(loc, *target, current->second.value,
              *value, ValueRange{}, b.getArrayAttr(SmallVector<Attribute>(shape(*target).size(), b.getStringAttr("all"))));
          return success();
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
        Value result = b.create<wk::NewOp>(loc, *type, initial, true);
        return write(destination, result);
      }
      auto region = view(destination);
      if (failed(region)) return failure();
      auto dimensions = shape((*region).getType());
      SmallVector<Value> indices;
      std::function<void(unsigned)> fillAxis = [&](unsigned axis) {
        if (axis == dimensions.size()) {
          SmallVector<Attribute> selectors(dimensions.size(), b.getStringAttr("index"));
          Value selected = b.create<wk::SliceOp>(loc,
              wk::SliceType::get(b.getContext(), encoding(), array({}), array({})),
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
      for (Value operand : operation->getOperands()) {
        auto supplied = view(operand);
        if (failed(supplied)) return operation->emitError("implementation requires supplied operand views");
        arguments.push_back(*supplied);
      }
      auto results = (*implementation)->expand(b, operation, arguments, nextAxis);
      if (failed(results)) return failure();
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
      SmallVector<Value> indices;
      for (Value value : store.getIndices()) indices.push_back(values.lookup(value));
      Value selected = b.create<wk::SliceOp>(loc,
          wk::SliceType::get(b.getContext(), encoding(store.getMemref()), array({}), array({})),
          *region, indices, b.getArrayAttr(SmallVector<Attribute>(indices.size(), b.getStringAttr("index"))));
      b.create<wk::CommitOp>(loc, values.lookup(store.getValue()), selected);
      return success();
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(operation)) return generic(genericOp);
    if (auto reduce = dyn_cast<cpu::ReduceOp>(operation)) return reduction(reduce);
    if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
      auto roots = writtenEnclosingLocals(conditional);
      auto savedLocals = locals;
      SmallVector<Type> types;
      for (Type type : conditional.getResultTypes()) types.push_back(scalarType(type));
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
        SmallVector<Value> results;
        if (source)
          for (Value value : source->getTerminator()->getOperands()) results.push_back(values.lookup(value));
        for (Value root : roots) {
          if (!locals.count(root)) return conditional.emitError("conditional local result is not initialized on every branch");
          if (savedLocals.count(root) && failed(checkCarry(root, savedLocals.lookup(root), conditional))) return failure();
          results.push_back(locals.lookup(root).value);
        }
        if (!types.empty()) b.create<scf::YieldOp>(loc, results);
      }
      locals = savedLocals;
      for (auto [source, destination] : llvm::zip(conditional.getResults(), target.getResults().take_front(conditional.getNumResults()))) values.map(source, destination);
      for (auto [root, value] : llvm::zip(roots, target.getResults().drop_front(conditional.getNumResults())))
        if (locals.count(root)) locals[root].value = value;
        else if (failed(write(root, value))) return failure();
      return success();
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      SmallVector<Value> initial;
      for (Value value : loop.getInitArgs()) initial.push_back(values.lookup(value));
      auto savedLocals = locals;
      SmallVector<Value> roots;
      for (Value root : writtenEnclosingLocals(loop))
        if (locals.count(root)) { roots.push_back(root); initial.push_back(locals.lookup(root).value); }
      auto target = b.create<scf::ForOp>(loc, values.lookup(loop.getLowerBound()),
          values.lookup(loop.getUpperBound()), values.lookup(loop.getStep()), initial);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(target.getBody());
        values.map(loop.getInductionVar(), target.getInductionVar());
        for (auto [source, destination] : llvm::zip(loop.getRegionIterArgs(), target.getRegionIterArgs().take_front(loop.getNumRegionIterArgs()))) values.map(source, destination);
        for (auto [root, value] : llvm::zip(roots, target.getRegionIterArgs().drop_front(loop.getNumRegionIterArgs())))
          locals[root].value = value;
        if (failed(block(*loop.getBody()))) return failure();
        SmallVector<Value> results;
        for (Value value : loop.getBody()->getTerminator()->getOperands()) results.push_back(values.lookup(value));
        for (Value root : roots) {
          if (failed(checkCarry(root, savedLocals.lookup(root), loop))) return failure();
          results.push_back(locals.lookup(root).value);
        }
        if (!initial.empty()) b.create<scf::YieldOp>(loc, results);
      }
      locals = std::move(savedLocals);
      for (auto [source, destination] : llvm::zip(loop.getResults(), target.getResults().take_front(loop.getNumResults()))) values.map(source, destination);
      for (auto [root, value] : llvm::zip(roots, target.getResults().drop_front(loop.getNumResults())))
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
      SmallVector<Value> indices;
      SmallVector<Attribute> selectors(load.getIndices().size(), b.getStringAttr("index"));
      for (Value value : load.getIndices()) indices.push_back(values.lookup(value));
      Value selected = b.create<wk::SliceOp>(loc, wk::SliceType::get(b.getContext(), encoding(), array({}), array({})),
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

  cpu::PhysicalProgramAnalysis analysis;
  cpu::AxisRelations relations;
  ModuleOp output;
  OpBuilder b;
  IRMapping values;
  llvm::DenseMap<Value, LocalValue> locals;
  llvm::DenseMap<Value, Value> operandReads;
  const llvm::DenseMap<Value, intent::QuantFormat> &formats;
  const cpu::ImplementationRegistry &implementations;
  llvm::DenseMap<int64_t, int64_t> extentIds;
  llvm::DenseMap<int64_t, int64_t> axisProjection;
  int64_t nextAxis;
};

}

FailureOr<OwningOpRef<ModuleOp>> legalizeProgram(ModuleOp cpuProgram, std::string &metadata) {
  if (failed(cpu::verifyCPUProgram(cpuProgram, false))) return failure();
  cpuProgram.getContext()->loadDialect<wk::WEFTKernelDialect>();
  OwningOpRef<ModuleOp> output = ModuleOp::create(cpuProgram.getLoc());
  bool quantized = false;
  cpuProgram.walk([&](cpu::QuantizedDotOp) { quantized = true; });
  cpuProgram.walk([&](cpu::QuantizeOp) { quantized = true; });
  if (quantized) declareQuantEncodings(*output);
  llvm::json::Array interfaces;
  auto registry = implementations();
  llvm::json::Array parameters, candidates;
  SmallVector<func::FuncOp> functions(cpuProgram.getOps<func::FuncOp>());
  auto interface = functions.front()->getAttrOfType<cpu::InterfaceAttr>("intent_cpu.interface");
  cpu::PhysicalProgramAnalysis rootAnalysis(functions.front());
  llvm::DenseMap<Value, int64_t> alignments;
  functions.front().walk([&](cpu::QuantizedDotOp op) {
    alignments[rootAnalysis.storageRoot(op.getLhs())] = 2;
    alignments[rootAnalysis.storageRoot(op.getRhs())] = 4;
  });
  auto integers = [](DenseI64ArrayAttr entries) {
    llvm::json::Array result;
    for (int64_t entry : entries.asArrayRef()) result.push_back(entry);
    return result;
  };
  for (auto [index, parameter] : llvm::enumerate(interface.getArguments())) {
    if (auto view = dyn_cast<cpu::ViewArgumentAttr>(parameter))
      parameters.push_back(llvm::json::Object{{"name", view.getName().getValue().str()},
          {"kind", "view"}, {"dtype", view.getElementType().isF32() ? "f32" : "u8"},
          {"shape", integers(view.getShape())}, {"dimensions", integers(view.getDimensions())},
          {"alignment", std::max(int64_t(view.getElementType().isF32() ? 4 : 1), alignments.lookup(functions.front().getArgument(index)))},
          {"access", view.getAccess()}, {"alias", view.getAlias().getValue().str()}, {"noalias", view.getNoalias()}});
    else {
      auto scalar = cast<cpu::ScalarArgumentAttr>(parameter);
      parameters.push_back(llvm::json::Object{{"name", scalar.getName().getValue().str()},
          {"kind", "scalar"}, {"dtype", scalar.getType().isF32() ? "f32" : "i64"}});
    }
  }
  for (func::FuncOp function : functions) {
    auto config = function->getAttrOfType<cpu::ConfigurationAttr>("intent_cpu.configuration");
    llvm::json::Array implementations;
    for (Attribute entry : function->getAttrOfType<ArrayAttr>("intent_cpu.implementations")) {
      auto binding = cast<cpu::ImplementationAttr>(entry);
      llvm::json::Object values;
      for (NamedAttribute parameter : binding.getParameters())
        values[parameter.getName().getValue()] = cast<IntegerAttr>(parameter.getValue()).getInt();
      implementations.push_back(llvm::json::Object{{"name", binding.getName().getValue().str()}, {"parameters", std::move(values)}});
    }
    candidates.push_back(llvm::json::Object{{"entry", function.getName().str()},
        {"values", llvm::json::Array{config.getTaskGrain(), config.getTileM(), config.getTileN(), config.getTileK(), config.getRegionSize()}},
        {"implementations", std::move(implementations)}});
    cpu::PhysicalProgramAnalysis analysis(function);
    llvm::DenseMap<Value, intent::QuantFormat> formats;
    bool conflict = false;
    auto requireFormat = [&](Value memory, intent::QuantFormat format) {
      auto [it, inserted] = formats.try_emplace(analysis.storageRoot(memory), format);
      if (!inserted && it->second != format) conflict = true;
    };
    function.walk([&](cpu::QuantizeOp op) { requireFormat(op.getOutput(), op.getFormat()); });
    function.walk([&](cpu::QuantizedDotOp op) {
      requireFormat(op.getLhs(), op.getLhsFormat()); requireFormat(op.getRhs(), op.getRhsFormat());
    });
    if (conflict) return function.emitError("one CPU storage has incompatible quantized record interpretations"), failure();
    TaskLowering lowering(function, *output, formats, registry);
    OpBuilder host(function.getContext());
    host.setInsertionPointToStart(&function.front());
    auto shapeArguments = lowering.shapeArguments(function, host);
    SmallVector<cpu::TasksOp> tasks;
    function.walk([&](cpu::TasksOp operation) { tasks.push_back(operation); });
    if (tasks.empty()) return function.emitError("Weft generation requires an explicit CPU task interface"), failure();
    for (auto [ordinal, task] : llvm::enumerate(tasks)) {
      std::string name = function.getName().str() + "_task_" + std::to_string(ordinal);
      SmallVector<unsigned> argumentPositions;
      for (BlockArgument argument : task.getBody().front().getArguments())
        if (!argument.use_empty()) argumentPositions.push_back(argument.getArgNumber());
      llvm::json::Object abi;
      if (failed(lowering.lower(task, name, argumentPositions, abi))) return failure();
      host.setInsertionPoint(task);
      auto loc = task.getLoc();
      auto zero = host.create<arith::ConstantIndexOp>(loc, 0);
      auto one = host.create<arith::ConstantIndexOp>(loc, 1);
      auto loop = host.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{task.getCount()}, ValueRange{one});
      host.setInsertionPointToStart(loop.getBody());
      SmallVector<Value> arguments;
      SmallVector<Value> original{loop.getInductionVars()[0]};
      llvm::append_range(original, task.getCaptures());
      for (unsigned position : argumentPositions) {
        Value capture = original[position];
        if (isa<MemRefType>(capture.getType())) arguments.push_back(capture);
        else {
          Type type = capture.getType().isIndex() ? Type(host.getI64Type()) : capture.getType();
          Value box = host.create<memref::AllocaOp>(loc, MemRefType::get({1}, type));
          Value scalar = capture;
          if (capture.getType().isIndex()) scalar = host.create<arith::IndexCastOp>(loc, type, capture);
          host.create<memref::StoreOp>(loc, scalar, box, ValueRange{zero});
          arguments.push_back(box);
        }
      }
      llvm::append_range(arguments, shapeArguments);
      {
        OpBuilder::InsertionGuard guard(host);
        host.setInsertionPointToStart(cpuProgram.getBody());
        auto declaration = host.create<func::FuncOp>(loc, name,
            host.getFunctionType(TypeRange(arguments), {}));
        declaration.setPrivate();
        declaration->setAttr("cpu.external_runtime", host.getUnitAttr());
      }
      host.create<func::CallOp>(loc, name, TypeRange{}, arguments);
      interfaces.push_back(llvm::json::Object{{"cpu_entry", function.getName().str()},
          {"task_ordinal", static_cast<int64_t>(ordinal)}, {"weft_entry", name},
          {"coordinate_argument", llvm::is_contained(argumentPositions, 0u) ? 0 : -1},
          {"abi", std::move(abi)},
          {"argument_count", static_cast<int64_t>(argumentPositions.size())}});
      task.erase();
    }
    if (failed(cpu::materializeStructuredComputations(function))) return failure();
  }
  if (failed(verify(*output))) return failure();
  std::string hostSource;
  if (failed(serializeHostProgram(cpuProgram, hostSource))) return failure();
  llvm::raw_string_ostream stream(metadata);
  stream << llvm::json::Value(llvm::json::Object{{"kind", "weft-generation"},
      {"native", false}, {"host_source", hostSource}, {"tasks", std::move(interfaces)},
      {"parameters", std::move(parameters)}, {"candidates", std::move(candidates)},
      {"contiguous_views", interface.getContiguousViews()}, {"disjoint_outputs", interface.getDisjointOutputs()},
      {"workers", cpuProgram->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities").getWorkers()}});
  return std::move(output);
}

}
