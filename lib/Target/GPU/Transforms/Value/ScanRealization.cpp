#include "Intent/Target/GPU/Transforms/Passes.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace intent::gpu {
namespace {

LogicalResult refineScans(plan::ProgramOp program) {
  FailureOr<std::unique_ptr<PhysicalProgramAnalysis>> analysis =
      PhysicalProgramAnalysis::compute(program);
  if (failed(analysis))
    return failure();
  target::KernelFacts &facts = (*analysis)->getFacts();
  llvm::DenseMap<int64_t, plan::ScanOp> bindings;
  for (plan::ScanOp scan : program.getBody().getOps<plan::ScanOp>())
    bindings[scan.getNode()] = scan;
  OpBuilder builder(program.getContext());
  for (const auto &entry : facts.scans) {
    Operation &operation = *entry.first;
    FailureOr<int64_t> node = target::getNodeID(operation, "scan realization");
    plan::ScanOp binding =
        succeeded(node) ? bindings.lookup(*node) : plan::ScanOp();
    if (failed(node) || !binding)
      return operation.emitOpError("has no physical scan binding");
    bool workspace = entry.second.scalarConsumers;
    binding->setAttr("result_space",
                     builder.getStringAttr(workspace ? "private_workspace"
                                                     : "private_fragment"));
    binding->setAttr("materialization",
                     builder.getStringAttr(workspace ? "scalar_access"
                                                     : "fragment_access"));
    binding->setAttr(
        "producers",
        builder.getDenseI64ArrayAttr(workspace
                                         ? ArrayRef<int64_t>(entry.second.producers)
                                         : ArrayRef<int64_t>()));
    binding->setAttr(
        "materialized_values",
        builder.getDenseI64ArrayAttr(
            workspace ? ArrayRef<int64_t>(entry.second.materializedValues)
                      : ArrayRef<int64_t>()));
  }
  return success();
}

class RefineScanRealizationPass final
    : public PassWrapper<RefineScanRealizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RefineScanRealizationPass)
  StringRef getArgument() const final {
    return "intent-refine-gpu-scan-realization";
  }
  StringRef getDescription() const final {
    return "Refine scan result materialization and replay slices";
  }
  void runOnOperation() final {
    SmallVector<plan::ProgramOp> programs(
        getOperation().getOps<plan::ProgramOp>());
    if (programs.size() != 1 || failed(refineScans(programs.front())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createRefineScanRealizationPass() {
  return std::make_unique<RefineScanRealizationPass>();
}

} // namespace intent::gpu
