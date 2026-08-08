#ifndef INTENT_LIB_TARGET_TRITON_STABLESOFTMAX_H
#define INTENT_LIB_TARGET_TRITON_STABLESOFTMAX_H

#include "Intent/Analysis/StableSoftmax.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace intent {
namespace triton {

mlir::LogicalResult
emitStableSoftmaxSource(StableSoftmaxMatch &softmax,
                        plan::PlanOp physicalPlan,
                        llvm::raw_ostream &output);

} // namespace triton
} // namespace intent

#endif
