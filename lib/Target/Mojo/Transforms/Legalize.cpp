#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "Intent/Transforms/CPU/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace intent::mojo {

LogicalResult legalizeProgram(ModuleOp module) {
  if (failed(cpu::verifyCPUProgram(module, true))) return failure();
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
