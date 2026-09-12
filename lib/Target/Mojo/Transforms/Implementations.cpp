#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "../../../Dialect/CPU/Transforms/Utilities.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
namespace intent::mojo {
using namespace intent::cpu;
namespace {

bool vectorLegal(Operation *, CapabilitiesAttr capabilities, const Configuration &config) {
  auto power = [](int64_t value) { return value > 0 && !(value & (value - 1)); };
  if (!config.local.get("vector_width") || !config.local.get("register_replicas") ||
      !config.local.get("reduction_replicas")) return false;
  for (NamedAttribute value : config.local)
    if (value.getName() != "vector_width" && value.getName() != "register_replicas" &&
        value.getName() != "reduction_replicas" && value.getName() != "micro_m" && value.getName() != "micro_n") return false;
  return power(config.parameter("vector_width")) && config.parameter("vector_width") <= capabilities.getVectorBits() / 32 &&
      power(config.parameter("register_replicas")) && config.parameter("register_replicas") <= 16 &&
      power(config.parameter("reduction_replicas")) && config.parameter("reduction_replicas") <= 16;
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
    const ContractionTile &tile, ConfigurationAttr, ImplementationAttr binding) {
  Location loc = operation.getLoc();
  int64_t microM = implementationParameter(binding, "micro_m");
  int64_t microN = implementationParameter(binding, "micro_n");
  int64_t vectorWidth = implementationParameter(binding, "vector_width");
  const InputSupply *supply = nullptr;
  for (const auto &input : tile.inputs) {
    if (input.operand != 1 || input.panelAxis != 1 || input.panelSize != vectorWidth * microN)
      return operation.emitError("Mojo contraction received an incompatible input representation");
    supply = &input;
  }
  Value zero = index(b, loc, 0);
  Value mEnd = add(b, loc, tile.mBegin, tile.mCount);
  auto micro = [&](Value packed, Value m, Value n, int64_t rows, int64_t columns, int64_t width) {
    Value left = subview(b, loc, tile.lhs, {m, tile.kBegin}, {b.getIndexAttr(rows), tile.depth});
    Value right = subview(b, loc, packed, {b.getIndexAttr(0), b.getIndexAttr(0)},
        {tile.depth, b.getIndexAttr(columns)});
    Value out = subview(b, loc, tile.output, {m, add(b, loc, tile.nBegin, n)},
        {b.getIndexAttr(rows), b.getIndexAttr(columns)});
    Type accumulator = cast<MemRefType>(tile.output.getType()).getElementType();
    auto partial = b.create<memref::AllocaOp>(loc, MemRefType::get({rows, columns}, accumulator));
    partial.setAlignment(width * accumulator.getIntOrFloatBitWidth() / 8);
    b.create<linalg::FillOp>(loc, ValueRange{tile.initial}, ValueRange{partial});
    auto contract = b.create<linalg::GenericOp>(loc, ValueRange{left, right}, ValueRange{partial},
        operation.getIndexingMapsArray(), operation.getIteratorTypesArray(),
        [&](OpBuilder &nested, Location loc, ValueRange arguments) {
          IRMapping mapping;
          Block &body = operation.getRegion().front();
          mapping.map(body.getArguments(), arguments);
          for (Operation &op : body.without_terminator()) nested.clone(op, mapping);
          nested.create<linalg::YieldOp>(loc, mapping.lookup(body.getTerminator()->getOperand(0)));
        });
    contract->setAttr("intent_cpu.implementation", binding);
    contract->setAttr("intent_cpu.microtile", MicrotileAttr::get(b.getContext(), rows, columns, width));
    SmallVector<Value> inputs{partial};
    if (!tile.first) inputs.insert(inputs.begin(), out);
    b.create<linalg::GenericOp>(loc, inputs, ValueRange{out},
        SmallVector<AffineMap>(inputs.size() + 1, b.getMultiDimIdentityMap(2)),
        SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
        [&](OpBuilder &nested, Location loc, ValueRange arguments) {
          Value value = arguments[0];
          if (!tile.first) value = nested.create<arith::AddFOp>(loc, value, arguments[1]);
          nested.create<linalg::YieldOp>(loc, value);
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
    Value packed;
    if (supply) {
      Value panel = index(b, loc, supply->panelSize);
      Value relativeColumn = b.create<arith::SubIOp>(loc, column, supply->begins[1]);
      Value relativeK = b.create<arith::SubIOp>(loc, tile.kBegin, supply->begins[0]);
      SmallVector<OpFoldResult> offsets{b.create<arith::DivSIOp>(loc, relativeColumn, panel).getResult(), relativeK,
          b.create<arith::RemSIOp>(loc, relativeColumn, panel).getResult()};
      SmallVector<OpFoldResult> sizes{b.getIndexAttr(1), tile.depth, b.getIndexAttr(columns)};
      SmallVector<OpFoldResult> strides(3, b.getIndexAttr(1));
      auto type = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          {ShapedType::kDynamic, columns}, cast<MemRefType>(supply->storage.getType()), offsets, sizes, strides));
      packed = b.create<memref::SubViewOp>(loc, type, supply->storage, offsets, sizes, strides);
    } else {
      packed = subview(b, loc, tile.rhs, {tile.kBegin, column}, {tile.depth, b.getIndexAttr(columns)});
    }
    for (auto region : rowRegions)
      loop(b, loc, region.begin, region.end, region.rows,
          [&](Value m) { micro(packed, m, n, region.rows, columns, width); });
  };
  // The selected input representation is shared by all M microtiles.
  Value columnBegin = zero;
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
  result.profile = [](func::FuncOp function) -> StringRef {
    bool contraction = false, region = false;
    function.walk([&](linalg::GenericOp op) { contraction |= isMatrixContraction(op); });
    function.walk([&](Operation *op) { region |= isa<cpu::RegionFoldOp, cpu::RegionScanOp>(op); });
    if (region) return contraction ? "mojo.region_contract_float" : "mojo.region_vector";
    return contraction ? "mojo.register_float" : "mojo.vector";
  };
  Implementation contraction{"mojo.register_float", [](Operation *op) {
      auto generic = dyn_cast<linalg::GenericOp>(op);
      return generic && isMatrixContraction(generic) &&
          isa<FloatType>(cast<MemRefType>(generic.getOutputs()[0].getType()).getElementType());
    }, [](Operation *operation, CapabilitiesAttr capabilities, const Configuration &config) {
      if (!vectorLegal(operation, capabilities, config) || !config.local.get("micro_m") || !config.local.get("micro_n")) return false;
      int64_t width = config.parameter("vector_width"), m = config.parameter("micro_m"), n = config.parameter("micro_n");
      int64_t bytes = cast<MemRefType>(cast<linalg::GenericOp>(operation).getOutputs()[0].getType()).getElementTypeBitWidth() / 8;
      return config.tileN % width == 0 && m <= 8 && n <= 4 && m * n <= 24 &&
          width * bytes * 8 <= capabilities.getVectorBits() &&
          m * n * width <= capabilities.getPrivateBytes() / bytes;
    }, [](Builder &, const Configuration &config) { return config.local; }, formTile, {}};
  auto inputRequirements = [](InputReuse reuse) {
    return [reuse](linalg::GenericOp operation, ConfigurationAttr, ImplementationAttr binding) {
      int64_t width = implementationParameter(binding, "vector_width");
      int64_t bytes = cast<MemRefType>(operation.getInputs()[1].getType()).getElementTypeBitWidth() / 8;
      return SmallVector<InputRequirement>{{1, 1, width * implementationParameter(binding, "micro_n"), width * bytes, reuse}};
    };
  };
  auto directLegal = contraction.legal;
  contraction.inputs = inputRequirements(InputReuse::Group);
  contraction.legal = [directLegal](Operation *op, CapabilitiesAttr capabilities, const Configuration &config) {
    int64_t bytes = cast<MemRefType>(cast<linalg::GenericOp>(op).getInputs()[1].getType()).getElementTypeBitWidth() / 8;
    return directLegal(op, capabilities, config) && config.tileK <= capabilities.getPrivateBytes() / bytes /
        config.parameter("vector_width") / config.parameter("micro_n");
  };
  result.add(contraction);
  contraction.name = "mojo.register_float_shared";
  contraction.inputs = inputRequirements(InputReuse::Consumers);
  contraction.legal = [directLegal](Operation *op, CapabilitiesAttr capabilities, const Configuration &config) {
    return directLegal(op, capabilities, config) &&
        config.tileN % (config.parameter("vector_width") * config.parameter("micro_n")) == 0;
  };
  result.add(contraction);
  contraction.name = "mojo.register_float_direct";
  contraction.legal = directLegal;
  contraction.inputs = {};
  result.add(std::move(contraction));
  result.add({"mojo.vector", [](Operation *op) {
      if (auto generic = dyn_cast<linalg::GenericOp>(op)) return !isMatrixContraction(generic);
      return isa<cpu::ReduceOp>(op);
    }, vectorLegal, parameters, {}, {}});
  return result;
}

}
