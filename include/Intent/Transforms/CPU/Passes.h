#ifndef INTENT_TRANSFORMS_CPU_PASSES_H
#define INTENT_TRANSFORMS_CPU_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringRef.h"

namespace intent::cpu {

struct Configuration {
  int64_t vectorWidth;
  int64_t taskGrain;
  int64_t tileM;
  int64_t tileN;
  int64_t tileK;
  int64_t microM;
};

mlir::LogicalResult fuseIntermediateBuffers(mlir::func::FuncOp function);
mlir::LogicalResult blockContractions(mlir::func::FuncOp function,
                                    const Configuration &configuration);
mlir::LogicalResult vectorizeLoops(mlir::func::FuncOp function, int64_t width);
mlir::LogicalResult partitionTasks(mlir::func::FuncOp function, int64_t grain);
mlir::LogicalResult verifyCPUProgram(mlir::ModuleOp module, bool realized);
mlir::LogicalResult runCPUPasses(mlir::ModuleOp module, int64_t vectorBits,
                               int64_t workers, llvm::StringRef defaults,
                               llvm::StringRef overrides);

}

#endif
