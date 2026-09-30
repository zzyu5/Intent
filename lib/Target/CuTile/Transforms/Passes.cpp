#include "Intent/Target/CuTile/Transforms/Passes.h"
#include "Legalize.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "Intent/Transforms/PassManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace intent::cutile {
namespace {

void programDialects(DialectRegistry &registry) {
  registry.insert<gpu::IntentGPUDialect, IntentCuTileDialect,
                  arith::ArithDialect, cf::ControlFlowDialect,
                  func::FuncDialect, scf::SCFDialect>();
}

LogicalResult finishGroup(ModuleOp module, StringRef name,
                          LogicalResult result) {
  if (failed(result))
    return module.emitError() << "cuTile transformation failed: " << name;
  if (failed(mlir::verify(module)))
    return module.emitError() << "cuTile structural postcondition failed: " << name;
  return success();
}

class PrepareProgramPass
    : public PassWrapper<PrepareProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareProgramPass)
  void getDependentDialects(DialectRegistry &registry) const final {
    programDialects(registry);
  }
  StringRef getArgument() const final { return "intent-cutile-prepare-program"; }
  StringRef getDescription() const final {
    return "Prepare cuTile matrix shapes, buffers and workspace accesses";
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), prepareProgram(module))))
      signalPassFailure();
  }
};

class NativeProgramPass
    : public PassWrapper<NativeProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeProgramPass)
  void getDependentDialects(DialectRegistry &registry) const final {
    programDialects(registry);
  }
  NativeProgramPass() = default;
  explicit NativeProgramPass(const gpu::TuningProfiles &profiles)
      : profiles(&profiles) {}
  StringRef getArgument() const final { return "intent-cutile-native-program"; }
  StringRef getDescription() const final {
    return "Form native cuTile accesses and computations from the current GPU program";
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (!profiles) {
      module.emitError("cuTile native program requires compilation tuning profiles");
      return signalPassFailure();
    }
    if (failed(finishGroup(module, getArgument(),
                           formNativeProgram(module, *profiles))))
      signalPassFailure();
  }
private:
  const gpu::TuningProfiles *profiles = nullptr;
};

class FinalizeProgramPass
    : public PassWrapper<FinalizeProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeProgramPass)
  void getDependentDialects(DialectRegistry &registry) const final {
    programDialects(registry);
  }
  StringRef getArgument() const final { return "intent-cutile-finalize-program"; }
  StringRef getDescription() const final {
    return "Close cuTile configurations, loops and array views and verify the surface";
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), finalizeProgram(module))))
      signalPassFailure();
  }
};

} // namespace

void registerCuTilePasses() {
  PassRegistration<PrepareProgramPass>();
  PassRegistration<NativeProgramPass>();
  PassRegistration<FinalizeProgramPass>();
}

LogicalResult legalizeGPUProgram(ModuleOp module,
                                const gpu::TuningProfiles &profiles) {
  if (failed(gpu::verifyGPUProgram(module))) return failure();
  PassManager manager(module.getContext(), ModuleOp::getOperationName());
  manager.addPass(std::make_unique<PrepareProgramPass>());
  manager.addPass(std::make_unique<NativeProgramPass>(profiles));
  manager.addPass(std::make_unique<FinalizeProgramPass>());
  if (failed(intent::configurePassManager(manager))) return failure();
  return manager.run(module);
}

} // namespace intent::cutile
