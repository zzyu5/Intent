#include "Intent/Target/GPU/Transforms/Passes.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace intent::gpu {
namespace {

LogicalResult refineTransfers(plan::ProgramOp program) {
  FailureOr<std::unique_ptr<PhysicalProgramAnalysis>> analysis =
      PhysicalProgramAnalysis::compute(program);
  if (failed(analysis))
    return failure();
  target::KernelModel &kernel = (*analysis)->getKernel();
  OpBuilder builder(program.getContext());
  for (plan::TransferOp transfer :
       program.getBody().getOps<plan::TransferOp>()) {
    Operation *operation = kernel.nodes.lookup(transfer.getNode());
    if (!operation)
      return transfer.emitOpError("does not bind a physical transfer operation");
    StringRef name = target::semanticOperationName(*operation);
    bool load = name == "intent.view_load";
    bool returnedAtomic =
        name == "intent.atomic_cas" ||
        (name == "intent.atomic_add" && operation->getNumResults() == 1);
    bool hasResult = load || returnedAtomic;
    if (hasResult && operation->getNumResults() != 1)
      return transfer.emitOpError("has no unique physical transfer result");
    StringRef resultSpace = "none";
    if (hasResult)
      resultSpace = isa<RankedTensorType>(operation->getResult(0).getType())
                        ? StringRef("private_fragment")
                        : StringRef("private_scalar");
    StringRef coverageSpace =
        transfer.getTensorIndexing() == "compact" ? resultSpace
                                                   : StringRef("none");
    transfer->setAttr("materialization", builder.getStringAttr("direct"));
    transfer->setAttr("result_space", builder.getStringAttr(resultSpace));
    transfer->setAttr("coverage_space", builder.getStringAttr(coverageSpace));
  }
  return success();
}

class RefineTransferRealizationPass final
    : public PassWrapper<RefineTransferRealizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RefineTransferRealizationPass)
  StringRef getArgument() const final {
    return "intent-refine-gpu-transfer-realization";
  }
  StringRef getDescription() const final {
    return "Select transfer result and compact-coverage realization";
  }
  void runOnOperation() final {
    SmallVector<plan::ProgramOp> programs(
        getOperation().getOps<plan::ProgramOp>());
    if (programs.size() != 1 || failed(refineTransfers(programs.front())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createRefineTransferRealizationPass() {
  return std::make_unique<RefineTransferRealizationPass>();
}

} // namespace intent::gpu
