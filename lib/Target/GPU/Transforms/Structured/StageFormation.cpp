#include "Intent/Target/GPU/Transforms/Passes.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/GPU/Realization/PhysicalProgram.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace intent::gpu {
namespace {

class FormStagesPass final
    : public PassWrapper<FormStagesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormStagesPass)
  StringRef getArgument() const final { return "intent-form-gpu-stages"; }
  StringRef getDescription() const final {
    return "Form compiler-private stage slices and their physical axes";
  }
  void runOnOperation() final {
    SmallVector<plan::ProgramOp> programs(
        getOperation().getOps<plan::ProgramOp>());
    if (programs.size() != 1) {
      getOperation().emitError(
          "stage formation requires exactly one physical program");
      signalPassFailure();
      return;
    }
    FailureOr<std::unique_ptr<PhysicalProgramAnalysis>> analysis =
        PhysicalProgramAnalysis::compute(programs.front());
    if (failed(analysis) ||
        failed(reconcileStages(programs.front(), (*analysis)->getFacts())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFormStagesPass() {
  return std::make_unique<FormStagesPass>();
}

} // namespace intent::gpu
