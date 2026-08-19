#include "Intent/Target/Common/Emission/Driver.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace intent::target {

LogicalResult emitTargetSource(ModuleOp module, llvm::raw_ostream &output,
                               const EmissionTarget &target) {
  if (!target.emitKernelSource)
    return module.emitError("target emission has an incomplete implementation");
  if (failed(verify(module)))
    return failure();

  SmallVector<plan::ProgramOp> programs(
      module.getOps<plan::ProgramOp>());
  SmallVector<plan::SearchSpaceOp> searchSpaces(
      module.getOps<plan::SearchSpaceOp>());
  if (programs.size() != 1 || searchSpaces.size() > 1)
    return module.emitError()
           << target.displayName
           << " emission requires one physical program and at most one search space";
  if (failed(plan::verifyGpuProgram(programs.front())))
    return failure();

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
  return target.emitKernelSource(std::move(*kernel), programs.front(),
                                 searchSpace, output);
}

} // namespace intent::target
