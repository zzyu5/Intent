#include "Intent/Target/Triton/Transforms/Passes.h"
#include "Legalization.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "Intent/Target/Triton/IR/TritonDialect.h"
#include "Intent/Transforms/PassManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace intent::triton {
namespace {

void programDialects(DialectRegistry &registry) {
  registry.insert<gpu::IntentGPUDialect, IntentTritonDialect,
                  arith::ArithDialect, cf::ControlFlowDialect,
                  func::FuncDialect, scf::SCFDialect>();
}

LogicalResult finishGroup(ModuleOp module, StringRef name,
                          LogicalResult result) {
  if (failed(result))
    return module.emitError() << "Triton transformation failed: " << name;
  // Once native forms are present, their operation verifiers own the local
  // contracts. The shared verifier intentionally rejects provider dialects.
  if (failed(mlir::verify(module)))
    return module.emitError() << "Triton structural postcondition failed: " << name;
  return success();
}

class ProgramGridPass
    : public PassWrapper<ProgramGridPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProgramGridPass)
  void getDependentDialects(DialectRegistry &registry) const final {
    programDialects(registry);
  }
  StringRef getArgument() const final { return "intent-triton-program-grid"; }
  StringRef getDescription() const final { return "Legalize the Triton program grid"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), legalizeProgramGrid(module))) ||
        failed(gpu::verifyGPUProgram(module)))
      signalPassFailure();
  }
};

class PrepareMemoryPass
    : public PassWrapper<PrepareMemoryPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareMemoryPass)
  void getDependentDialects(DialectRegistry &registry) const final {
    programDialects(registry);
  }
  StringRef getArgument() const final { return "intent-triton-prepare-memory"; }
  StringRef getDescription() const final {
    return "Prepare Triton buffers, indexed accesses and matrix shapes";
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), prepareTritonMemory(module))))
      signalPassFailure();
  }
};

class NativeFormsPass
    : public PassWrapper<NativeFormsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeFormsPass)
  void getDependentDialects(DialectRegistry &registry) const final {
    programDialects(registry);
  }
  NativeFormsPass() = default;
  explicit NativeFormsPass(const gpu::TuningProfiles &profiles)
      : profiles(&profiles) {}
  StringRef getArgument() const final { return "intent-triton-native-forms"; }
  StringRef getDescription() const final {
    return "Form native Triton accesses, supply and complete configurations";
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (!profiles) {
      module.emitError("Triton native forms require compilation tuning profiles");
      return signalPassFailure();
    }
    if (failed(finishGroup(module, getArgument(),
                           formTritonProgram(module, *profiles))))
      signalPassFailure();
  }
private:
  // Immutable compile-call input, valid for the entire synchronous manager run.
  const gpu::TuningProfiles *profiles = nullptr;
};

class FinalizeProgramPass
    : public PassWrapper<FinalizeProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeProgramPass)
  void getDependentDialects(DialectRegistry &registry) const final {
    programDialects(registry);
  }
  StringRef getArgument() const final { return "intent-triton-finalize-program"; }
  StringRef getDescription() const final {
    return "Close Triton callbacks and resource constraints and verify the surface";
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), finalizeTritonProgram(module))))
      signalPassFailure();
  }
};

} // namespace

void registerTritonPasses() {
  PassRegistration<ProgramGridPass>();
  PassRegistration<PrepareMemoryPass>();
  PassRegistration<NativeFormsPass>();
  PassRegistration<FinalizeProgramPass>();
}

LogicalResult legalizeGPUProgram(ModuleOp module,
                                const gpu::TuningProfiles &profiles) {
  if (failed(gpu::verifyGPUProgram(module))) return failure();
  PassManager manager(module.getContext(), ModuleOp::getOperationName());
  manager.addPass(std::make_unique<ProgramGridPass>());
  manager.addPass(std::make_unique<PrepareMemoryPass>());
  manager.addPass(std::make_unique<NativeFormsPass>(profiles));
  manager.addPass(std::make_unique<FinalizeProgramPass>());
  if (failed(intent::configurePassManager(manager))) return failure();
  return manager.run(module);
}

} // namespace intent::triton
