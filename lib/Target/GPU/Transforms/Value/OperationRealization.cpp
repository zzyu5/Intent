#include "Intent/Target/GPU/Transforms/Passes.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace intent::gpu {
namespace {

LogicalResult refineValues(plan::ProgramOp program) {
  FailureOr<std::unique_ptr<PhysicalProgramAnalysis>> analysis =
      PhysicalProgramAnalysis::compute(program);
  if (failed(analysis))
    return failure();
  target::KernelModel &kernel = (*analysis)->getKernel();
  OpBuilder builder(program.getContext());
  for (plan::ReductionOp reduction :
       program.getBody().getOps<plan::ReductionOp>())
    reduction->setAttr("result_space",
                       builder.getStringAttr("private_fragment"));
  for (plan::ScanOp scan : program.getBody().getOps<plan::ScanOp>())
    scan->setAttr("carry_space", builder.getStringAttr("private_scalar"));
  for (plan::PointwiseOp pointwise :
       program.getBody().getOps<plan::PointwiseOp>()) {
    Operation *operation = kernel.nodes.lookup(pointwise.getNode());
    if (!operation || operation->getNumResults() != 1)
      return pointwise.emitOpError(
          "does not bind one physical pointwise result");
    StringRef space = isa<RankedTensorType>(operation->getResult(0).getType())
                          ? StringRef("private_fragment")
                          : StringRef("private_scalar");
    pointwise->setAttr("result_space", builder.getStringAttr(space));
  }
  for (plan::SparseContractOp sparse :
       program.getBody().getOps<plan::SparseContractOp>()) {
    sparse->setAttr("compressed_space", builder.getStringAttr("shared"));
    sparse->setAttr("metadata_space", builder.getStringAttr("shared"));
    sparse->setAttr("rhs_space", builder.getStringAttr("shared"));
    sparse->setAttr("accumulator_space",
                    builder.getStringAttr("private_fragment"));
  }
  return success();
}

class RefineValueRealizationPass final
    : public PassWrapper<RefineValueRealizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RefineValueRealizationPass)
  StringRef getArgument() const final {
    return "intent-refine-gpu-value-realization";
  }
  StringRef getDescription() const final {
    return "Select pointwise, reduction, scan-carry, and sparse value residency";
  }
  void runOnOperation() final {
    SmallVector<plan::ProgramOp> programs(
        getOperation().getOps<plan::ProgramOp>());
    if (programs.size() != 1 || failed(refineValues(programs.front())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createRefineValueRealizationPass() {
  return std::make_unique<RefineValueRealizationPass>();
}

} // namespace intent::gpu
