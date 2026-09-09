#ifndef INTENT_DIALECT_CPU_TRANSFORMS_PASSES_H
#define INTENT_DIALECT_CPU_TRANSFORMS_PASSES_H

#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
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
  int64_t microN;
};

mlir::LogicalResult fuseStructuredComputations(mlir::func::FuncOp function);
void forwardCPUOutputs(mlir::func::FuncOp function);
mlir::LogicalResult materializeStructuredComputations(mlir::func::FuncOp function);
mlir::LogicalResult materializeRegisterContractions(mlir::func::FuncOp function);
mlir::LogicalResult materializeCPUProgram(mlir::ModuleOp module);
mlir::LogicalResult fuseIntermediateBuffers(mlir::func::FuncOp function);
mlir::LogicalResult blockContractions(mlir::func::FuncOp function,
                                    const Configuration &configuration);
mlir::LogicalResult vectorizeLoops(mlir::func::FuncOp function, int64_t width);
mlir::LogicalResult partitionTasks(mlir::func::FuncOp function, int64_t grain);
mlir::LogicalResult isolateTasks(mlir::func::FuncOp function);
mlir::LogicalResult materializeTaskLoops(mlir::func::FuncOp function);
mlir::LogicalResult runCPUPasses(mlir::ModuleOp module, int64_t vectorBits,
                               int64_t workers, llvm::StringRef defaults,
                               llvm::StringRef overrides);

}

#endif
