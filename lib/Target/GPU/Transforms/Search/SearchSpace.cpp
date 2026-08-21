#include "Intent/Target/GPU/Transforms/Passes.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/GPU/Realization/PhysicalProgram.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace intent::gpu {
namespace {

class MaterializeSearchSpacePass final
    : public PassWrapper<MaterializeSearchSpacePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeSearchSpacePass)

  StringRef getArgument() const final {
    return "intent-materialize-gpu-search-space";
  }
  StringRef getDescription() const final {
    return "Derive the parameter search space from the final shared GPU physical program";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (!module.getOps<plan::SearchSpaceOp>().empty()) {
      module.emitError("GPU search space must be materialized exactly once");
      signalPassFailure();
      return;
    }
    SmallVector<plan::ProgramOp> programs(module.getOps<plan::ProgramOp>());
    if (programs.size() != 1) {
      module.emitError("GPU search-space materialization requires one physical program");
      signalPassFailure();
      return;
    }
    FailureOr<std::unique_ptr<PhysicalProgramAnalysis>> analysis =
        PhysicalProgramAnalysis::compute(programs.front());
    if (failed(analysis)) {
      signalPassFailure();
      return;
    }
    if (failed(materializeSearchSpace(
            module, (*analysis)->getFacts(), programs.front())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createMaterializeSearchSpacePass() {
  return std::make_unique<MaterializeSearchSpacePass>();
}

} // namespace intent::gpu
