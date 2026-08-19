#ifndef INTENT_TARGET_COMMON_LOWERING_COMBINER_H
#define INTENT_TARGET_COMMON_LOWERING_COMBINER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include <functional>
#include <string>

namespace intent::target::lowering {

struct CombinerUse {
  mlir::func::FuncOp function;
  unsigned componentCount = 0;
  unsigned captureCount = 0;
};

using CombinerExpressionEmitter = std::function<mlir::FailureOr<std::string>(
    mlir::Operation &, llvm::ArrayRef<std::string>)>;
using CombinerCastSpelling =
    std::function<mlir::FailureOr<std::string>(mlir::Type)>;

bool hasGenericCombiner(mlir::Operation &operation);
mlir::FailureOr<CombinerUse> resolveCombiner(mlir::Operation &operation);
mlir::FailureOr<llvm::SmallVector<mlir::func::FuncOp>>
collectCombiners(mlir::func::FuncOp entry);
mlir::FailureOr<std::string>
combinerProjectionName(mlir::Operation &operation);
mlir::FailureOr<std::string>
renderPythonCombiner(mlir::func::FuncOp function, llvm::StringRef emittedName,
                     llvm::StringRef decorator,
                     CombinerExpressionEmitter emitExpression);
mlir::FailureOr<std::string>
renderPythonCombinerProjection(mlir::Operation &operation,
                               llvm::StringRef decorator,
                               llvm::StringRef selectSpelling);
mlir::FailureOr<std::string> renderPythonPointwiseExpression(
    mlir::Operation &operation, llvm::ArrayRef<std::string> operands,
    llvm::StringRef lowering, CombinerCastSpelling castSpelling);

} // namespace intent::target::lowering

#endif
