#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "../../../Dialect/CPU/Transforms/Utilities.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
namespace intent::mojo {
using namespace intent::cpu;
namespace {

bool vectorLegal(CapabilitiesAttr capabilities, const Configuration &config) {
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
    const ContractionTile &tile, ConfigurationAttr shared, ImplementationAttr binding) {
  Location loc = operation.getLoc();
  int64_t microM = implementationParameter(binding, "micro_m");
  int64_t microN = implementationParameter(binding, "micro_n");
  int64_t vectorWidth = implementationParameter(binding, "vector_width");
  Value zero = index(b, loc, 0);
  Value mEnd = add(b, loc, tile.mBegin, tile.mCount);
  auto micro = [&](Value packed, Value m, Value n, int64_t rows, int64_t columns, int64_t width) {
    Value left = subview(b, loc, tile.lhs, {m, tile.kBegin}, {b.getIndexAttr(rows), tile.depth});
    Value right = subview(b, loc, packed, {b.getIndexAttr(0), b.getIndexAttr(0)},
        {tile.depth, b.getIndexAttr(columns)});
    Value out = subview(b, loc, tile.output, {m, add(b, loc, tile.nBegin, n)},
        {b.getIndexAttr(rows), b.getIndexAttr(columns)});
    auto partial = b.create<memref::AllocaOp>(loc, MemRefType::get({rows, columns}, b.getF32Type()));
    partial.setAlignment(width * 4);
    b.create<linalg::FillOp>(loc, ValueRange{tile.initial}, ValueRange{partial});
    auto contract = b.create<linalg::GenericOp>(loc, ValueRange{left, right}, ValueRange{partial},
        operation.getIndexingMapsArray(), operation.getIteratorTypesArray(),
        [](OpBuilder &nested, Location loc, ValueRange arguments) {
          Value value = nested.create<math::FmaOp>(loc, arguments[0], arguments[1], arguments[2]);
          nested.create<linalg::YieldOp>(loc, value);
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
    auto packed = b.create<memref::AllocaOp>(loc,
        MemRefType::get({shared.getTileK(), columns}, b.getF32Type()));
    packed.setAlignment(width * 4);
    Value source = subview(b, loc, tile.rhs, {tile.kBegin, add(b, loc, tile.nBegin, n)},
        {tile.depth, b.getIndexAttr(columns)});
    Value destination = subview(b, loc, packed, {b.getIndexAttr(0), b.getIndexAttr(0)},
        {tile.depth, b.getIndexAttr(columns)});
    b.create<memref::CopyOp>(loc, source, destination);
    for (auto region : rowRegions)
      loop(b, loc, region.begin, region.end, region.rows,
          [&](Value m) { micro(packed, m, n, region.rows, columns, width); });
  };
  // This implementation shares its packed B panel across the outer M block.
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
    bool contraction = false;
    function.walk([&](linalg::GenericOp op) { contraction |= isMatrixContraction(op); });
    return contraction ? "mojo.register_f32" : "mojo.vector_f32";
  };
  result.add({"mojo.register_f32", [](Operation *op) {
      auto generic = dyn_cast<linalg::GenericOp>(op);
      return generic && isMatrixContraction(generic) &&
          cast<MemRefType>(generic.getInputs()[0].getType()).getElementType().isF32();
    }, [](CapabilitiesAttr capabilities, const Configuration &config) {
      if (!vectorLegal(capabilities, config) || !config.local.get("micro_m") || !config.local.get("micro_n")) return false;
      int64_t width = config.parameter("vector_width"), m = config.parameter("micro_m"), n = config.parameter("micro_n");
      return config.tileN % width == 0 && m <= 8 && n <= 4 && m * n <= 24 &&
          config.tileK <= capabilities.getPrivateBytes() / 4 / width / n;
    }, [](Builder &, const Configuration &config) { return config.local; }, formTile, {}});
  result.add({"mojo.vector_f32", [](Operation *op) {
      if (auto generic = dyn_cast<linalg::GenericOp>(op)) return !isMatrixContraction(generic);
      return isa<cpu::ReduceOp>(op);
    }, vectorLegal, parameters, {}, {}});
  return result;
}

}
