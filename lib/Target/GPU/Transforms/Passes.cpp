#include "Intent/Target/GPU/Transforms/Passes.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/GPU/Realization/PhysicalProgram.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace intent::gpu {
namespace {

class ConstructPhysicalProgramPass final
    : public PassWrapper<ConstructPhysicalProgramPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstructPhysicalProgramPass)

  explicit ConstructPhysicalProgramPass(const DeviceCapabilities &device)
      : device(device) {}

  StringRef getArgument() const final { return "intent-construct-gpu-program"; }
  StringRef getDescription() const final {
    return "Convert canonical Intent Kernel IR into one executable GPU physical program";
  }

  void runOnOperation() final {
    if (failed(constructPhysicalProgram(getOperation(), device)))
      signalPassFailure();
  }

private:
  DeviceCapabilities device;
};

class VerifyPhysicalProgramPass final
    : public PassWrapper<VerifyPhysicalProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyPhysicalProgramPass)

  StringRef getArgument() const final { return "intent-verify-gpu-program"; }
  StringRef getDescription() const final {
    return "Verify the complete GPU physical-program contract";
  }

  void runOnOperation() final {
    SmallVector<plan::ProgramOp> programs(
        getOperation().getOps<plan::ProgramOp>());
    if (programs.size() != 1 || failed(plan::verifyGpuProgram(programs.front())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass>
createConstructPhysicalProgramPass(const DeviceCapabilities &device) {
  return std::make_unique<ConstructPhysicalProgramPass>(device);
}

std::unique_ptr<Pass> createVerifyPhysicalProgramPass() {
  return std::make_unique<VerifyPhysicalProgramPass>();
}

LogicalResult runPhysicalProgramPipeline(ModuleOp module,
                                         const DeviceCapabilities &device) {
  PassManager manager(module.getContext());
  manager.enableVerifier(true);
  manager.addPass(createConstructPhysicalProgramPass(device));
  manager.addPass(createVerifyPhysicalProgramPass());
  manager.addPass(createRefinePrivateBufferResidencyPass(device));
  manager.addPass(createVerifyPhysicalProgramPass());
  manager.addPass(createRefinePersistentTraversalPass());
  manager.addPass(createVerifyPhysicalProgramPass());
  manager.addPass(createRefineBoundaryNeutralizationPass());
  manager.addPass(createVerifyPhysicalProgramPass());
  return manager.run(module);
}

} // namespace intent::gpu
