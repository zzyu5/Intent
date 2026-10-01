#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "../../../Dialect/CPU/Transforms/Utilities.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <optional>
#include <string>

using namespace mlir;
namespace intent::mojo {
using namespace intent::cpu;
namespace {

std::optional<std::string> checkVector(Operation *, CapabilitiesAttr capabilities,
                                       const Configuration &config) {
  auto power = [](int64_t value) { return value > 0 && !(value & (value - 1)); };
  for (StringRef name : {"vector_width", "register_replicas", "reduction_replicas"})
    if (!config.local.get(name)) return "missing local parameter '" + name.str() + "'";
  for (NamedAttribute value : config.local)
    if (value.getName() != "vector_width" && value.getName() != "register_replicas" &&
        value.getName() != "reduction_replicas" && value.getName() != "micro_m" && value.getName() != "micro_n")
      return "unsupported local parameter '" + value.getName().getValue().str() + "'";
  int64_t width = config.parameter("vector_width");
  if (!power(width)) return "vector_width must be a positive power of two, got " + std::to_string(width);
  if (width > capabilities.getVectorBits() / 32)
    return "vector_width " + std::to_string(width) + " exceeds the target's " +
        std::to_string(capabilities.getVectorBits() / 32) + " f32 lanes";
  for (StringRef name : {"register_replicas", "reduction_replicas"}) {
    int64_t count = config.parameter(name);
    if (!power(count)) return name.str() + " must be a positive power of two, got " + std::to_string(count);
    if (count > 16) return name.str() + " exceeds the supported maximum of 16, got " + std::to_string(count);
  }
  return std::nullopt;
}

DictionaryAttr parameters(Builder &b, const Configuration &config) {
  return b.getDictionaryAttr({
      b.getNamedAttr("vector_width", config.local.get("vector_width")),
      b.getNamedAttr("register_replicas", config.local.get("register_replicas")),
      b.getNamedAttr("reduction_replicas", config.local.get("reduction_replicas"))});
}

Value subview(OpBuilder &b, Location loc, Value source,
              ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  return b.create<memref::SubViewOp>(loc, source, offsets, sizes,
      SmallVector<OpFoldResult>(offsets.size(), b.getIndexAttr(1)));
}

LogicalResult formTile(OpBuilder &b, linalg::GenericOp operation,
    const ContractionTile &tile, ConfigurationAttr configuration, ImplementationAttr binding) {
  Location loc = operation.getLoc();
  int64_t microM = implementationParameter(binding, "micro_m");
  int64_t microN = implementationParameter(binding, "micro_n");
  int64_t vectorWidth = implementationParameter(binding, "vector_width");
  const InputSupply *supplies[2] = {};
  for (const auto &input : tile.inputs) {
    if (input.operand > 1 || input.panelAxis != 1 ||
        input.panelSize != (input.operand == 0 ? configuration.getTileK() : vectorWidth * microN))
      return operation.emitError("Mojo contraction received an incompatible input representation");
    supplies[input.operand] = &input;
  }
  auto window = [&](unsigned operand, Value source, Value row, Value column,
                    ArrayRef<OpFoldResult> sizes, ArrayRef<int64_t> shape) -> Value {
    const InputSupply *supply = supplies[operand];
    if (!supply) return subview(b, loc, source, {row, column}, sizes);
    Value panel = index(b, loc, supply->panelSize);
    Value relativeRow = b.create<arith::SubIOp>(loc, row, supply->begins[0]);
    Value relativeColumn = b.create<arith::SubIOp>(loc, column, supply->begins[1]);
    SmallVector<OpFoldResult> offsets{b.create<arith::DivSIOp>(loc, relativeColumn, panel).getResult(), relativeRow,
        b.create<arith::RemSIOp>(loc, relativeColumn, panel).getResult()};
    SmallVector<OpFoldResult> packedSizes{b.getIndexAttr(1), sizes[0], sizes[1]};
    SmallVector<OpFoldResult> strides(3, b.getIndexAttr(1));
    auto type = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        shape, cast<MemRefType>(supply->storage.getType()), offsets, packedSizes, strides));
    return b.create<memref::SubViewOp>(loc, type, supply->storage, offsets, packedSizes, strides);
  };
  Value zero = index(b, loc, 0);
  Value mEnd = add(b, loc, tile.mBegin, tile.mCount);
  auto micro = [&](Value packed, Value m, Value n, int64_t rows, int64_t columns, int64_t width) {
    Value left = window(0, tile.lhs, m, tile.kBegin,
        {b.getIndexAttr(rows), tile.depth}, {rows, ShapedType::kDynamic});
    Value right = subview(b, loc, packed, {b.getIndexAttr(0), b.getIndexAttr(0)},
        {tile.depth, b.getIndexAttr(columns)});
    Value out = subview(b, loc, tile.output, {m, add(b, loc, tile.nBegin, n)},
        {b.getIndexAttr(rows), b.getIndexAttr(columns)});
    Type accumulator = cast<MemRefType>(tile.output.getType()).getElementType();
    Value partial = out;
    if (tile.first) {
      auto storage = b.create<memref::AllocaOp>(loc, MemRefType::get({rows, columns}, accumulator));
      storage.setAlignment(width * accumulator.getIntOrFloatBitWidth() / 8);
      b.create<linalg::FillOp>(loc, ValueRange{tile.initial}, ValueRange{storage});
      partial = storage;
    }
    auto contract = b.create<linalg::GenericOp>(loc, ValueRange{left, right}, ValueRange{partial},
        operation.getIndexingMapsArray(), operation.getIteratorTypesArray(),
        [&](OpBuilder &nested, Location loc, ValueRange arguments) {
          IRMapping mapping;
          Block &body = operation.getRegion().front();
          mapping.map(body.getArguments(), arguments);
          for (Operation &op : body.without_terminator()) {
            if (auto widen = dyn_cast<arith::ExtFOp>(op);
                widen && mapping.lookup(widen.getIn()).getType() == widen.getType())
              mapping.map(widen.getResult(), mapping.lookup(widen.getIn()));
            else nested.clone(op, mapping);
          }
          nested.create<linalg::YieldOp>(loc, mapping.lookup(body.getTerminator()->getOperand(0)));
        });
    contract->setAttr("intent_cpu.implementation", binding);
    contract->setAttr("intent_cpu.reduction_order", operation->getAttr("intent_cpu.reduction_order"));
    contract->setAttr("intent_cpu.microtile", MicrotileAttr::get(b.getContext(), rows, columns, width));
    if (!tile.first) return;
    b.create<linalg::GenericOp>(loc, ValueRange{partial}, ValueRange{out},
        SmallVector<AffineMap>(2, b.getMultiDimIdentityMap(2)),
        SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
        [](OpBuilder &nested, Location loc, ValueRange arguments) {
          nested.create<linalg::YieldOp>(loc, arguments[0]);
        });
  };
  struct RowRegion { int64_t rows; Value begin, end; };
  SmallVector<RowRegion> rowRegions;
  Value rowBegin = tile.mBegin;
  for (int64_t rows : {microM, int64_t{4}, int64_t{2}, int64_t{1}}) {
    if (!rowRegions.empty() && rows >= rowRegions.back().rows) continue;
    Value end = mEnd;
    if (rows != 1) {
      Value count = b.create<arith::SubIOp>(loc, mEnd, rowBegin);
      Value step = index(b, loc, rows);
      end = add(b, loc, rowBegin, multiply(b, loc, b.create<arith::DivSIOp>(loc, count, step), step));
    }
    rowRegions.push_back({rows, rowBegin, end});
    rowBegin = end;
  }
  auto rows = [&](Value n, int64_t columns, int64_t width) {
    Value column = add(b, loc, tile.nBegin, n);
    Value packed = window(1, tile.rhs, tile.kBegin, column,
        {tile.depth, b.getIndexAttr(columns)}, {ShapedType::kDynamic, columns});
    for (auto region : rowRegions)
      loop(b, loc, region.begin, region.end, region.rows,
          [&](Value m) { micro(packed, m, n, region.rows, columns, width); });
  };
  // The selected input representation is shared by all M microtiles.
  Value columnBegin = zero;
  if (const InputSupply *supply = supplies[1]) {
    Value panel = index(b, loc, supply->panelSize);
    Value relative = b.create<arith::SubIOp>(loc, tile.nBegin, supply->begins[1]);
    Value lane = b.create<arith::RemSIOp>(loc, relative, panel);
    Value aligned = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane, zero);
    Value room = b.create<arith::SubIOp>(loc, panel, lane);
    Value count = b.create<arith::MinSIOp>(loc, tile.nCount, room);
    columnBegin = b.create<arith::SelectOp>(loc, aligned, zero, count);
    loop(b, loc, zero, columnBegin, 1, [&](Value n) { rows(n, 1, 1); });
  }
  int64_t previousVectors = microN + 1;
  for (int64_t vectors : {microN, int64_t{2}, int64_t{1}}) {
    if (vectors >= previousVectors) continue;
    int64_t columns = vectorWidth * vectors;
    Value step = index(b, loc, columns);
    Value remaining = b.create<arith::SubIOp>(loc, tile.nCount, columnBegin);
    Value end = add(b, loc, columnBegin,
        multiply(b, loc, b.create<arith::DivSIOp>(loc, remaining, step), step));
    loop(b, loc, columnBegin, end, columns, [&](Value n) { rows(n, columns, vectorWidth); });
    columnBegin = end;
    previousVectors = vectors;
  }
  loop(b, loc, columnBegin, tile.nCount, 1, [&](Value n) { rows(n, 1, 1); });
  return success();
}

}

cpu::ImplementationRegistry implementations() {
  ImplementationRegistry result;
  for (StringRef family : {"mojo.region_contraction", "mojo.register_contraction"})
    result.addProfile(family, {"vector_width", "micro_m", "micro_n",
                               "register_replicas", "reduction_replicas"});
  for (StringRef family : {"mojo.region_vector", "mojo.vector"})
    result.addProfile(family, {"vector_width", "register_replicas", "reduction_replicas"});
  result.profile = [](func::FuncOp function) -> StringRef {
    bool contraction = false, region = false;
    function.walk([&](linalg::GenericOp op) { contraction |= isMatrixContraction(op); });
    function.walk([&](Operation *op) { region |= isa<cpu::RegionFoldOp, cpu::RegionScanOp>(op); });
    if (region) return contraction ? "mojo.region_contraction" : "mojo.region_vector";
    return contraction ? "mojo.register_contraction" : "mojo.vector";
  };
  Implementation contraction{"mojo.register_float", [](Operation *op) {
      auto generic = dyn_cast<linalg::GenericOp>(op);
      return generic && isMatrixContraction(generic) &&
          isa<FloatType>(cast<MemRefType>(generic.getOutputs()[0].getType()).getElementType());
    }, [](Operation *operation, CapabilitiesAttr capabilities, const Configuration &config) -> std::optional<std::string> {
      if (auto reason = checkVector(operation, capabilities, config)) return reason;
      for (StringRef name : {"micro_m", "micro_n"})
        if (!config.local.get(name)) return "missing local parameter '" + name.str() + "'";
      int64_t width = config.parameter("vector_width"), m = config.parameter("micro_m"), n = config.parameter("micro_n");
      int64_t bytes = cast<MemRefType>(cast<linalg::GenericOp>(operation).getOutputs()[0].getType()).getElementTypeBitWidth() / 8;
      if (config.tileN % width != 0)
        return "tile_n " + std::to_string(config.tileN) + " must be divisible by vector_width " + std::to_string(width);
      if (m > 8) return "micro_m exceeds the supported maximum of 8, got " + std::to_string(m);
      if (n > 4) return "micro_n exceeds the supported maximum of 4, got " + std::to_string(n);
      if (m * n > 24)
        return "micro_m * micro_n requires " + std::to_string(m * n) + " accumulators, exceeding the limit of 24";
      if (width * bytes * 8 > capabilities.getVectorBits())
        return "accumulator vector requires " + std::to_string(width * bytes * 8) +
            " bits, exceeding target vector_bits " + std::to_string(capabilities.getVectorBits());
      if (m * n * width > capabilities.getPrivateBytes() / bytes)
        return "microtile requires " + std::to_string(m * n * width) +
            " accumulator elements, exceeding private storage capacity " +
            std::to_string(capabilities.getPrivateBytes() / bytes);
      return std::nullopt;
    }, [](Builder &, const Configuration &config) { return config.local; }, formTile, {}};
  auto inputRequirements = [](InputReuse reuse) {
    return [reuse](linalg::GenericOp operation, ConfigurationAttr, ImplementationAttr binding) {
      int64_t width = implementationParameter(binding, "vector_width");
      Type element = cast<MemRefType>(operation.getInputs()[1].getType()).getElementType();
      int64_t bytes = element.getIntOrFloatBitWidth() / 8;
      return SmallVector<InputRequirement>{{1, element, 1, width * implementationParameter(binding, "micro_n"), width * bytes, reuse, 1}};
    };
  };
  auto directCheck = contraction.check;
  auto addInteger = [&](StringRef name, StringRef exactFloatName) {
    Implementation integer = contraction;
    integer.name = name;
    integer.applicable = [](Operation *op) {
      auto generic = dyn_cast<linalg::GenericOp>(op);
      return generic && isMatrixContraction(generic) &&
          cast<MemRefType>(generic.getOutputs()[0].getType()).getElementType().isSignlessInteger(32);
    };
    result.add(integer);
    integer.name = exactFloatName;
    auto integerCheck = integer.check;
    integer.check = [integerCheck](Operation *operation, CapabilitiesAttr capabilities, const Configuration &config)
        -> std::optional<std::string> {
      if (auto reason = integerCheck(operation, capabilities, config)) return reason;
      // Integer partials remain live across each local floating reduction.
      int64_t elements = config.parameter("micro_m") * config.parameter("micro_n") * config.parameter("vector_width");
      if (elements > capabilities.getPrivateBytes() / 8)
        return "simultaneous i32 and exact-f32 accumulator states require " + std::to_string(elements) +
            " elements each, exceeding private storage capacity " +
            std::to_string(capabilities.getPrivateBytes() / 8);
      return std::nullopt;
    };
    integer.parameters = [](Builder &b, const Configuration &config) {
      NamedAttrList fields(config.local);
      fields.append("exact_f32_chunk", b.getI64IntegerAttr(1024));
      return fields.getDictionary(b.getContext());
    };
    result.add(std::move(integer));
  };
  contraction.inputs = inputRequirements(InputReuse::Group);
  contraction.check = [directCheck](Operation *op, CapabilitiesAttr capabilities, const Configuration &config)
      -> std::optional<std::string> {
    if (auto reason = directCheck(op, capabilities, config)) return reason;
    Value source = cast<linalg::GenericOp>(op).getInputs()[1];
    int64_t bytes = cast<MemRefType>(source.getType()).getElementTypeBitWidth() / 8;
    int64_t limit = capabilities.getPrivateBytes() / bytes / config.parameter("vector_width") / config.parameter("micro_n");
    int64_t capacity = config.tileK;
    if (auto bound = constantDimensionUpperBound(source, 0); bound && *bound > 0)
      capacity = std::min(capacity, *bound);
    if (capacity > limit)
      return "group input panel requires " + std::to_string(capacity) + " reduction elements, exceeding the private storage limit of " +
          std::to_string(limit) + " reduction elements";
    return std::nullopt;
  };
  result.add(contraction);
  addInteger("mojo.register_integer", "mojo.register_integer_f32");
  contraction.name = "mojo.register_float_shared";
  contraction.inputs = inputRequirements(InputReuse::Consumers);
  auto sharedCheck = [directCheck](Operation *op, CapabilitiesAttr capabilities, const Configuration &config)
      -> std::optional<std::string> {
    if (auto reason = directCheck(op, capabilities, config)) return reason;
    int64_t panel = config.parameter("vector_width") * config.parameter("micro_n");
    if (config.tileN % panel != 0)
      return "tile_n " + std::to_string(config.tileN) + " must be divisible by the shared input panel width " +
          std::to_string(panel);
    return std::nullopt;
  };
  contraction.check = sharedCheck;
  result.add(contraction);
  addInteger("mojo.register_integer_shared", "mojo.register_integer_f32_shared");
  contraction.name = "mojo.register_float_direct";
  contraction.check = directCheck;
  contraction.contraction.unitInnerStride[1] = true;
  contraction.inputs = {};
  result.add(contraction);
  addInteger("mojo.register_integer_direct", "mojo.register_integer_f32_direct");
  contraction.name = "mojo.register_float_widened";
  contraction.contraction.unitInnerStride[1] = false;
  contraction.check = [sharedCheck](Operation *op, CapabilitiesAttr capabilities, const Configuration &config)
      -> std::optional<std::string> {
    if (auto reason = sharedCheck(op, capabilities, config)) return reason;
    auto operation = cast<linalg::GenericOp>(op);
    Type accumulator = cast<MemRefType>(operation.getOutputs()[0].getType()).getElementType();
    if (!llvm::any_of(operation.getInputs(), [&](Value input) {
      return cast<MemRefType>(input.getType()).getElementType() != accumulator;
    })) return "widened input supply requires an input element type different from the accumulator type";
    return std::nullopt;
  };
  contraction.inputs = [](linalg::GenericOp operation, ConfigurationAttr config, ImplementationAttr binding) {
    int64_t width = implementationParameter(binding, "vector_width");
    Type element = cast<MemRefType>(operation.getOutputs()[0].getType()).getElementType();
    int64_t alignment = width * element.getIntOrFloatBitWidth() / 8;
    return SmallVector<InputRequirement>{
        {0, element, 1, config.getTileK(), alignment, InputReuse::Consumers, config.getTileK()},
        {1, element, 1, width * implementationParameter(binding, "micro_n"), alignment, InputReuse::Consumers, 1}};
  };
  result.add(std::move(contraction));
  Implementation vector{"mojo.vector", [](Operation *op) {
      if (auto generic = dyn_cast<linalg::GenericOp>(op)) return !isMatrixContraction(generic);
      return isa<cpu::ReduceOp, cpu::HistogramOp, func::FuncOp>(op);
    }, checkVector, parameters, {}, {}};
  vector.worksetRows = [](linalg::GenericOp operation, ImplementationAttr) {
    return registerContractionRows(operation);
  };
  result.add(std::move(vector));
  auto scanParameters = [](bool vectorized) {
    return [vectorized](Builder &b, const Configuration &config) {
      NamedAttrList fields(parameters(b, config));
      fields.append("scan_width", b.getI64IntegerAttr(vectorized ? config.parameter("vector_width") : 1));
      return fields.getDictionary(b.getContext());
    };
  };
  result.add({"mojo.scan_scalar", [](Operation *op) { return isa<cpu::ScanOp>(op); },
      checkVector, scanParameters(false), {}, {}});
  result.add({"mojo.scan_vector", [](Operation *op) { return isa<cpu::ScanOp>(op); },
      [](Operation *op, CapabilitiesAttr capabilities, const Configuration &config) -> std::optional<std::string> {
        if (auto reason = checkVector(op, capabilities, config)) return reason;
        if (!supportsVectorScan(cast<cpu::ScanOp>(op)))
          return "vector scan requires a source-form innermost scan with supported storage and an elementwise combine";
        return std::nullopt;
      }, scanParameters(true), {}, {}});
  return result;
}

}
