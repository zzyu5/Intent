#include "PassDetail.h"
#include "Intent/Transforms/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
namespace intent::bangc {
namespace {
LogicalResult finishGroup(ModuleOp module, StringRef name, LogicalResult result) {
  if (failed(result)) return module.emitError() << "BANG C transformation failed: " << name;
  if (failed(dsa::verifyProgram(module))) return module.emitError() << "BANG C postcondition failed: " << name;
  return success();
}

class NativeComputationsPass : public PassWrapper<NativeComputationsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeComputationsPass)
  StringRef getArgument() const final { return "intent-bangc-native-computations"; }
  StringRef getDescription() const final { return "Realize BANG C numerical and matrix computations"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), realizeNativeComputations(module)))) {
      signalPassFailure();
      return;
    }
  }
};

class NativeWorkspacePass : public PassWrapper<NativeWorkspacePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeWorkspacePass)
  StringRef getArgument() const final { return "intent-bangc-native-workspace"; }
  StringRef getDescription() const final { return "Realize workspace required by selected BANG C computations"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), realizeNativeWorkspace(module)))) {
      signalPassFailure();
      return;
    }
  }
};

class NativeImplementationsPass : public PassWrapper<NativeImplementationsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeImplementationsPass)
  StringRef getArgument() const final { return "intent-bangc-native-implementations"; }
  StringRef getDescription() const final { return "Select BANG C native implementations and matrix layouts"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), selectNativeImplementations(module)))) {
      signalPassFailure();
      return;
    }
  }
};

class LocalCompositionPass : public PassWrapper<LocalCompositionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LocalCompositionPass)
  StringRef getArgument() const final { return "intent-bangc-local-composition"; }
  StringRef getDescription() const final { return "Compose local values and accesses in the BANG C program"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), composeLocalProgram(module)))) {
      signalPassFailure();
      return;
    }
  }
};

class SupplySynchronizationPass : public PassWrapper<SupplySynchronizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SupplySynchronizationPass)
  StringRef getArgument() const final { return "intent-bangc-supply-synchronization"; }
  StringRef getDescription() const final { return "Schedule BANG C supply, participants and synchronization"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), scheduleProgramSupply(module)))) {
      signalPassFailure();
      return;
    }
  }
};

class StorageBindingPass : public PassWrapper<StorageBindingPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StorageBindingPass)
  StringRef getArgument() const final { return "intent-bangc-storage-binding"; }
  StringRef getDescription() const final { return "Bind final BANG C physical storage and verify target legality"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), bindProgramStorage(module)))) {
      signalPassFailure();
      return;
    }
    if (failed(verifyProgram(module))) {
      module.emitError() << "BANG C target postcondition failed: " << getArgument();
      signalPassFailure();
    }
  }
};

} // namespace

void registerBangCPasses() {
  PassRegistration<NativeComputationsPass>();
  PassRegistration<NativeWorkspacePass>();
  PassRegistration<NativeImplementationsPass>();
  PassRegistration<LocalCompositionPass>();
  PassRegistration<SupplySynchronizationPass>();
  PassRegistration<StorageBindingPass>();
}

LogicalResult legalizeProgram(ModuleOp module, StringRef architecture) {
  if (architecture != "mtp_372")
    return module.emitError("BANG C currently has a bound implementation profile for mtp_372");
  if (failed(dsa::verifyProgram(module))) return failure();
  module->setAttr("bangc.architecture", StringAttr::get(module.getContext(), architecture));
  PassManager manager(module.getContext(), ModuleOp::getOperationName());
  manager.addPass(std::make_unique<NativeComputationsPass>());
  manager.addPass(std::make_unique<NativeWorkspacePass>());
  manager.addPass(std::make_unique<NativeImplementationsPass>());
  manager.addPass(std::make_unique<LocalCompositionPass>());
  manager.addPass(std::make_unique<SupplySynchronizationPass>());
  manager.addPass(std::make_unique<StorageBindingPass>());
  if (failed(intent::configurePassManager(manager))) return failure();
  return manager.run(module);
}

} // namespace intent::bangc
