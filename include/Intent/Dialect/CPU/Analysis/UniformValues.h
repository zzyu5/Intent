#ifndef INTENT_DIALECT_CPU_ANALYSIS_UNIFORMVALUES_H
#define INTENT_DIALECT_CPU_ANALYSIS_UNIFORMVALUES_H

#include "Intent/Analysis/UniformValues.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace intent::cpu {

mlir::Attribute foldUniformComputation(mlir::linalg::GenericOp operation,
    const UniformBindings &operands, const UniformBindings &scalarFacts = UniformBindings());

}
#endif
