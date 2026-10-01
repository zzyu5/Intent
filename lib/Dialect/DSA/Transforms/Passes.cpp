#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
namespace intent::dsa {

#define GEN_PASS_DEF_DSACOLLECTIVEGATHERSUPPLY
#define GEN_PASS_DEF_DSAMATRIXSUPPLY
#define GEN_PASS_DEF_DSANORMALIZEINDICES
#include "Intent/Dialect/DSA/Transforms/Passes.h.inc"

namespace {

LogicalResult finishGroup(ModuleOp module, StringRef name, LogicalResult result) {
  if (failed(result)) return module.emitError() << "DSA transformation failed: " << name;
  if (failed(verifyProgram(module))) return module.emitError() << "DSA postcondition failed: " << name;
  return success();
}

struct CollectiveGatherSupplyPass
    : impl::DSACollectiveGatherSupplyBase<CollectiveGatherSupplyPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyProgram(module))) return signalPassFailure();
    auto function = *module.getOps<func::FuncOp>().begin();
    if (failed(finishGroup(module, getArgument(), realizeCollectiveGatherSupply(function)))) signalPassFailure();
  }
};

struct MatrixSupplyPass : impl::DSAMatrixSupplyBase<MatrixSupplyPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyProgram(module))) return signalPassFailure();
    auto function = *module.getOps<func::FuncOp>().begin();
    if (failed(finishGroup(module, getArgument(), realizeMatrixSupply(function)))) signalPassFailure();
  }
};

struct NormalizeIndicesPass : impl::DSANormalizeIndicesBase<NormalizeIndicesPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyProgram(module))) return signalPassFailure();
    auto function = *module.getOps<func::FuncOp>().begin();
    LogicalResult result = success();
    if (normalizeLinearIndices(function)) {
      OpPassManager cleanup(ModuleOp::getOperationName());
      cleanup.addPass(createCanonicalizerPass());
      cleanup.addPass(createCSEPass());
      result = runPipeline(cleanup, module);
    }
    if (failed(finishGroup(module, getArgument(), result))) signalPassFailure();
  }
};
} // namespace

void buildDSAPipeline(OpPassManager &manager) {
  manager.addPass(createDSACollectiveGatherSupply());
  manager.addPass(createDSAMatrixSupply());
  manager.addPass(createDSANormalizeIndices());
}

void registerDSAPipelines() {
  PassPipelineRegistration<>("intent-dsa", "Form a complete shared DSA program",
      [](OpPassManager &manager) { buildDSAPipeline(manager); });
}

} // namespace intent::dsa
