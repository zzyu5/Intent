#include "Intent/Target/GPU/Transforms/Passes.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/GPU/Realization/PhysicalProgram.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace intent::gpu {
namespace {

class FormAutomaticBlockingPass final
    : public PassWrapper<FormAutomaticBlockingPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormAutomaticBlockingPass)
  StringRef getArgument() const final {
    return "intent-form-gpu-automatic-blocking";
  }
  StringRef getDescription() const final {
    return "Form compiler-owned GPU blocking for full logical tensor domains";
  }
  void runOnOperation() final {
    SmallVector<plan::ProgramOp> programs(
        getOperation().getOps<plan::ProgramOp>());
    if (programs.size() != 1) {
      getOperation().emitError(
          "automatic blocking requires exactly one physical program");
      signalPassFailure();
      return;
    }
    FailureOr<std::unique_ptr<PhysicalProgramAnalysis>> analysis =
        PhysicalProgramAnalysis::compute(programs.front());
    if (failed(analysis) ||
        failed(formAutomaticBlocking(programs.front(),
                                     (*analysis)->getFacts())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFormAutomaticBlockingPass() {
  return std::make_unique<FormAutomaticBlockingPass>();
}

} // namespace intent::gpu
