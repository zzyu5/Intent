#ifndef INTENT_TARGET_GPU_REALIZATION_ANALYSIS_H
#define INTENT_TARGET_GPU_REALIZATION_ANALYSIS_H

#include "Intent/Target/Common/Realization/KernelFacts.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::gpu::realization {

mlir::LogicalResult analyzeOperations(intent::target::KernelFacts &facts);

} // namespace intent::gpu::realization

#endif
