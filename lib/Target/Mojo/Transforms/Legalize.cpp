#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "Intent/Transforms/CPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace intent::mojo {
namespace {

bool supportedType(Type type) {
  if (auto vector = dyn_cast<VectorType>(type)) {
    int64_t width = vector.getNumElements();
    return vector.getRank() == 1 && vector.getElementType().isF32() &&
        width > 0 && (width & (width - 1)) == 0;
  }
  if (auto memory = dyn_cast<MemRefType>(type)) return memory.getElementType().isF32();
  return type.isIndex() || type.isF32() || type.isInteger(64);
}

LogicalResult checkSurface(ModuleOp module) {
  bool invalid = false;
  module.walk([&](Operation *operation) {
    if (isa<ModuleOp, func::FuncOp, func::ReturnOp, scf::YieldOp, scf::ReduceOp>(operation)) return;
    bool supported = isa<arith::ConstantOp, arith::AddFOp, arith::AddIOp,
        arith::SubFOp, arith::SubIOp, arith::MulFOp, arith::MulIOp,
        arith::DivFOp, arith::DivSIOp, arith::FloorDivSIOp, arith::RemSIOp,
        arith::MinSIOp, arith::MaxSIOp, arith::CeilDivSIOp, arith::NegFOp,
        arith::IndexCastOp, math::FmaOp, math::SqrtOp, memref::DimOp,
        memref::SubViewOp, memref::CastOp, memref::LoadOp, memref::StoreOp,
        memref::AllocaOp, memref::AllocOp, memref::DeallocOp,
        vector::LoadOp, vector::StoreOp, vector::BroadcastOp, vector::ShuffleOp,
        vector::ExtractElementOp, scf::ForOp, scf::ParallelOp>(operation);
    supported &= llvm::all_of(operation->getOperandTypes(), supportedType);
    supported &= llvm::all_of(operation->getResultTypes(), supportedType);
    if (auto constant = dyn_cast<arith::ConstantOp>(operation))
      supported &= isa<IntegerAttr, FloatAttr, DenseFPElementsAttr>(constant.getValue());
    if (auto dimension = dyn_cast<memref::DimOp>(operation))
      supported &= dimension.getConstantIndex().has_value();
    if (auto stack = dyn_cast<memref::AllocaOp>(operation))
      supported &= stack.getType().hasStaticShape();
    if (auto parallel = dyn_cast<scf::ParallelOp>(operation))
      supported &= parallel.getNumResults() == 0 && parallel.getNumLoops() == 1 &&
          matchPattern(parallel.getLowerBound()[0], m_Zero()) &&
          matchPattern(parallel.getStep()[0], m_One());
    if (!supported) {
      operation->emitError("current operation/type has no supported Mojo CPU surface form");
      invalid = true;
    }
  });
  return failure(invalid);
}

}

LogicalResult legalizeProgram(ModuleOp module) {
  if (failed(cpu::verifyCPUProgram(module, true)) || failed(checkSurface(module))) return failure();
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto enter = builder.create<func::FuncOp>(module.getLoc(), "intent_cpu_enter_ieee",
      builder.getFunctionType({}, {builder.getI32Type()}));
  auto leave = builder.create<func::FuncOp>(module.getLoc(), "intent_cpu_leave_ieee",
      builder.getFunctionType({builder.getI32Type()}, {}));
  enter.setPrivate(); leave.setPrivate();
  enter->setAttr("cpu.external_runtime", builder.getUnitAttr());
  leave->setAttr("cpu.external_runtime", builder.getUnitAttr());
  SmallVector<Block *> scopes;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal()) continue;
    scopes.push_back(&function.front());
    function.walk([&](scf::ParallelOp parallel) { scopes.push_back(parallel.getBody()); });
  }
  for (Block *scope : scopes) {
    builder.setInsertionPointToStart(scope);
    Value previous = builder.create<func::CallOp>(module.getLoc(), enter, ValueRange{}).getResult(0);
    builder.setInsertionPoint(scope->getTerminator());
    builder.create<func::CallOp>(module.getLoc(), leave, ValueRange{previous});
  }
  return cpu::verifyCPUProgram(module, true);
}

}
