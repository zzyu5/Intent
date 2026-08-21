#include "Intent/Target/Common/Lowering/Driver.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace intent::target {
namespace {

LogicalResult materializeTargetProgram(ModuleOp module,
                                       const MaterializationTarget &target) {
  if (target.provider.empty() || !target.addProviderPasses ||
      !target.verifyProviderProgram || !target.materializeProgramSource)
    return module.emitError(
        "target materialization has an incomplete implementation");
  if (failed(verify(module)))
    return failure();

  SmallVector<plan::ProgramOp> programs(
      module.getOps<plan::ProgramOp>());
  SmallVector<plan::SearchSpaceOp> searchSpaces(
      module.getOps<plan::SearchSpaceOp>());
  if (programs.size() != 1 || searchSpaces.size() > 1)
    return module.emitError()
           << target.displayName
           << " materialization requires one physical program and at most one search space";
  if (failed(plan::verifyGpuProgram(programs.front())))
    return failure();
  for (plan::TargetProgramOp existing :
       programs.front().getBody().getOps<plan::TargetProgramOp>())
    return existing.emitOpError(
        "physical program is already materialized for a provider");

  plan::SearchSpaceOp searchSpace =
      searchSpaces.empty() ? plan::SearchSpaceOp() : searchSpaces.front();
  if (searchSpace &&
      (searchSpace.getEntry() != programs.front().getEntry() ||
       searchSpace.getTarget() != programs.front().getTarget()))
    return searchSpace.emitOpError("does not match the physical program");
  if (searchSpace && failed(plan::verifyGpuSearchSpace(searchSpace)))
    return failure();

  FailureOr<func::FuncOp> entry = plan::getPhysicalEntry(programs.front());
  FailureOr<KernelModel> kernel =
      succeeded(entry) ? analyzeKernel(*entry)
                       : FailureOr<KernelModel>(failure());
  if (failed(entry) || failed(kernel))
    return failure();
  if (failed(target.verifyProviderProgram(*kernel, programs.front(),
                                          searchSpace)))
    return failure();
  std::string source;
  llvm::raw_string_ostream stream(source);
  if (failed(target.materializeProgramSource(
          std::move(*kernel), programs.front(), searchSpace, stream)))
    return failure();
  stream.flush();
  OpBuilder builder(module.getContext());
  builder.setInsertionPoint(programs.front().getBody().front().getTerminator());
  builder.create<plan::TargetProgramOp>(
      programs.front().getLoc(), builder.getStringAttr(target.provider),
      builder.getStringAttr(source));
  return plan::verifyGpuProgram(programs.front());
}

class MaterializeTargetProgramPass final
    : public PassWrapper<MaterializeTargetProgramPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeTargetProgramPass)

  explicit MaterializeTargetProgramPass(const MaterializationTarget &target)
      : target(target) {}

  StringRef getArgument() const final {
    return "intent-materialize-target-program";
  }
  StringRef getDescription() const final {
    return "Materialize one provider-legal program from the current physical program";
  }
  void runOnOperation() final {
    if (failed(materializeTargetProgram(getOperation(), target)))
      signalPassFailure();
  }

private:
  MaterializationTarget target;
};

} // namespace

std::unique_ptr<Pass>
createMaterializeTargetProgramPass(const MaterializationTarget &target) {
  return std::make_unique<MaterializeTargetProgramPass>(target);
}

LogicalResult runTargetMaterializationPipeline(
    ModuleOp module, const MaterializationTarget &target) {
  PassManager manager(module.getContext());
  manager.enableVerifier(true);
  target.addProviderPasses(manager);
  manager.addPass(createMaterializeTargetProgramPass(target));
  return manager.run(module);
}

LogicalResult translateTargetProgram(ModuleOp module, StringRef provider,
                                     llvm::raw_ostream &output) {
  if (failed(verify(module)))
    return failure();
  SmallVector<plan::ProgramOp> programs(module.getOps<plan::ProgramOp>());
  if (programs.size() != 1)
    return module.emitError(
        "target translation requires exactly one physical program");
  FailureOr<plan::TargetProgramOp> target =
      plan::getTargetProgram(programs.front(), provider);
  if (failed(target))
    return failure();
  output << (*target).getSource();
  return success();
}

} // namespace intent::target
