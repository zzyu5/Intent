#include "Intent/Target/Triton/Translate.h"

#include "Intent/Analysis/StableSoftmax.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Transforms/Passes.h"
#include "StableSoftmax.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace intent {
namespace triton {

LogicalResult emitTritonSource(ModuleOp module, llvm::raw_ostream &output) {
  if (failed(verifyKernelModule(module)) || failed(verify(module)))
    return failure();
  SmallVector<plan::PlanOp> plans(module.getOps<plan::PlanOp>());
  if (plans.size() != 1)
    return module.emitError("Triton translation requires exactly one Physical Plan");
  FailureOr<StableSoftmaxMatch> matched = matchStableSoftmax(module);
  if (failed(matched))
    return failure();
  return emitStableSoftmaxSource(*matched, plans.front(), output);
}

} // namespace triton
} // namespace intent
