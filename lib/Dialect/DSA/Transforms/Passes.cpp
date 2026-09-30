#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Transforms/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
namespace intent::dsa {
namespace {

LogicalResult finishGroup(ModuleOp module, StringRef name, LogicalResult result) {
  if (failed(result)) return module.emitError() << "DSA transformation failed: " << name;
  if (failed(verifyProgram(module))) return module.emitError() << "DSA postcondition failed: " << name;
  return success();
}

class CollectiveGatherSupplyPass
    : public PassWrapper<CollectiveGatherSupplyPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollectiveGatherSupplyPass)
  StringRef getArgument() const final { return "intent-dsa-collective-gather-supply"; }
  StringRef getDescription() const final { return "Form collective row supply from current DSA address relations"; }
  void runOnOperation() final {
    auto module = getOperation();
    auto function = *module.getOps<func::FuncOp>().begin();
    if (failed(finishGroup(module, getArgument(), realizeCollectiveGatherSupply(function)))) signalPassFailure();
  }
};

class MatrixSupplyPass : public PassWrapper<MatrixSupplyPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatrixSupplyPass)
  StringRef getArgument() const final { return "intent-dsa-matrix-supply"; }
  StringRef getDescription() const final { return "Form resident or shared supply for a complete DSA matrix traversal"; }
  void runOnOperation() final {
    auto module = getOperation();
    auto function = *module.getOps<func::FuncOp>().begin();
    if (failed(finishGroup(module, getArgument(), realizeMatrixSupply(function)))) signalPassFailure();
  }
};

class NormalizeIndicesPass : public PassWrapper<NormalizeIndicesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeIndicesPass)
  StringRef getArgument() const final { return "intent-dsa-normalize-indices"; }
  StringRef getDescription() const final { return "Normalize exact DSA index programs and canonicalize the result"; }
  void runOnOperation() final {
    auto module = getOperation();
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

void registerDSAPasses() {
  PassRegistration<CollectiveGatherSupplyPass>();
  PassRegistration<MatrixSupplyPass>();
  PassRegistration<NormalizeIndicesPass>();
}

LogicalResult runProgramTransforms(ModuleOp module) {
  if (failed(verifyProgram(module))) return failure();
  PassManager manager(module.getContext(), ModuleOp::getOperationName());
  manager.addPass(std::make_unique<CollectiveGatherSupplyPass>());
  manager.addPass(std::make_unique<MatrixSupplyPass>());
  manager.addPass(std::make_unique<NormalizeIndicesPass>());
  if (failed(intent::configurePassManager(manager))) return failure();
  return manager.run(module);
}

} // namespace intent::dsa
