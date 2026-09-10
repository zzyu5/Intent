#include "Intent/Target/Weft/Transforms/Passes.h"
#include "Quantization.h"
#include "../../../Dialect/CPU/Transforms/Utilities.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
namespace intent::weft_provider {
using namespace intent::cpu;
namespace {

LogicalResult formTile(OpBuilder &b, linalg::GenericOp operation,
    const ContractionTile &tile, ConfigurationAttr, ImplementationAttr binding) {
  Location loc = operation.getLoc();
  auto view = [&](Value source, Value m, Value n, OpFoldResult rows, OpFoldResult columns) -> Value {
    return b.create<memref::SubViewOp>(loc, source,
        ArrayRef<OpFoldResult>{m, n}, ArrayRef<OpFoldResult>{rows, columns},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
  };
  loop(b, loc, tile.mBegin, add(b, loc, tile.mBegin, tile.mCount), 1, [&](Value m) {
    loop(b, loc, tile.nBegin, add(b, loc, tile.nBegin, tile.nCount), 1, [&](Value n) {
      Value lhs = view(tile.lhs, m, tile.kBegin, b.getIndexAttr(1), tile.depth);
      Value rhs = view(tile.rhs, tile.kBegin, n, tile.depth, b.getIndexAttr(1));
      Value out = view(tile.output, m, n, b.getIndexAttr(1), b.getIndexAttr(1));
      auto partial = b.create<memref::AllocaOp>(loc, MemRefType::get({1, 1}, b.getF32Type()));
      b.create<linalg::FillOp>(loc, ValueRange{tile.initial}, ValueRange{partial});
      auto term = b.create<linalg::GenericOp>(loc, ValueRange{lhs, rhs}, ValueRange{partial},
          operation.getIndexingMapsArray(), operation.getIteratorTypesArray(),
          [](OpBuilder &nested, Location loc, ValueRange args) {
            nested.create<linalg::YieldOp>(loc, nested.create<math::FmaOp>(loc, args[0], args[1], args[2]).getResult());
          });
      term->setAttr("intent_cpu.implementation", binding);
      SmallVector<Value> inputs{partial};
      if (!tile.first) inputs.insert(inputs.begin(), out);
      b.create<linalg::GenericOp>(loc, inputs, ValueRange{out},
          SmallVector<AffineMap>(inputs.size() + 1, b.getMultiDimIdentityMap(2)),
          SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
          [&](OpBuilder &nested, Location loc, ValueRange args) {
            Value result = args[0];
            if (!tile.first) result = nested.create<arith::AddFOp>(loc, result, args[1]);
            nested.create<linalg::YieldOp>(loc, result);
          });
    });
  });
  return success();
}

}

cpu::ImplementationRegistry implementations() {
  ImplementationRegistry result;
  result.profile = [](func::FuncOp function) -> StringRef {
    bool quantize = false, contraction = false, region = false;
    function.walk([&](cpu::QuantizeOp) { quantize = true; });
    function.walk([&](linalg::GenericOp op) { contraction |= isMatrixContraction(op); });
    function.walk([&](Operation *op) { region |= isa<cpu::RegionFoldOp, cpu::RegionScanOp>(op); });
    if (region) return contraction ? "weft.region_contract_f32" : "weft.region_structured";
    return quantize ? "weft.q8_k" : contraction ? "weft.contract_f32" : "weft.structured";
  };
  auto noParameters = [](Builder &b, const Configuration &) { return b.getDictionaryAttr({}); };
  auto legal = [](CapabilitiesAttr, const Configuration &) { return true; };
  result.add({"weft.q8_k", [](Operation *op) { return isa<cpu::QuantizeOp>(op); },
      [](CapabilitiesAttr, const Configuration &config) {
        return config.local.size() == 1 && config.local.get("chunk") &&
            (config.parameter("chunk") == 32 || config.parameter("chunk") == 64);
      }, [](Builder &b, const Configuration &config) {
        return config.local;
      }, {}, [](OpBuilder &b, Operation *operation, ValueRange args, int64_t &nextAxis)
          -> FailureOr<SmallVector<Value>> {
        if (failed(expandQuantize(b, cast<cpu::QuantizeOp>(operation), args[0], args[1], nextAxis)))
          return failure();
        return SmallVector<Value>{};
      }});
  result.add({"weft.q4_k_q8_k", [](Operation *op) { return isa<cpu::QuantizedDotOp>(op); },
      legal, noParameters, {}, [](OpBuilder &b, Operation *operation, ValueRange args, int64_t &nextAxis)
          -> FailureOr<SmallVector<Value>> {
        auto value = expandQuantizedDot(b, cast<cpu::QuantizedDotOp>(operation), args[0], args[1], nextAxis);
        if (failed(value)) return failure();
        return SmallVector<Value>{*value};
      }});
  result.add({"weft.contract_f32", [](Operation *op) {
      auto generic = dyn_cast<linalg::GenericOp>(op);
      return generic && isMatrixContraction(generic);
    }, legal, noParameters, formTile, {}, {true, true, true}});
  result.add({"weft.structured_f32", [](Operation *op) {
      if (auto generic = dyn_cast<linalg::GenericOp>(op)) return !isMatrixContraction(generic);
      return isa<cpu::ReduceOp>(op);
    }, legal, [](Builder &b, const Configuration &config) {
      return b.getDictionaryAttr({b.getNamedAttr("panel", b.getI64IntegerAttr(std::min<int64_t>(4, config.tileN)))});
    }, {}, {}, {}, [](ImplementationAttr binding) {
      return implementationParameter(binding, "panel");
    }});
  return result;
}

}
