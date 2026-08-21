#include "Intent/Target/GPU/Transforms/Passes.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/GPU/Realization/PhysicalProgram.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace intent::gpu {
namespace {

class ReconcileAccessRangesPass final
    : public PassWrapper<ReconcileAccessRangesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReconcileAccessRangesPass)
  StringRef getArgument() const final {
    return "intent-reconcile-gpu-access-ranges";
  }
  StringRef getDescription() const final {
    return "Project selected execution ranges onto physical transfer footprints";
  }
  void runOnOperation() final {
    SmallVector<plan::ProgramOp> programs(
        getOperation().getOps<plan::ProgramOp>());
    if (programs.size() != 1) {
      getOperation().emitError(
          "access realization requires exactly one physical program");
      signalPassFailure();
      return;
    }
    FailureOr<std::unique_ptr<PhysicalProgramAnalysis>> analysis =
        PhysicalProgramAnalysis::compute(programs.front());
    if (failed(analysis) ||
        failed(reconcileAccessRanges(programs.front(),
                                     (*analysis)->getFacts())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createReconcileAccessRangesPass() {
  return std::make_unique<ReconcileAccessRangesPass>();
}

} // namespace intent::gpu
