#ifndef INTENT_DIALECT_CPU_TRANSFORMS_PASSES_H
#define INTENT_DIALECT_CPU_TRANSFORMS_PASSES_H

#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringRef.h"

namespace intent::cpu {

class ImplementationRegistry;

struct Configuration {
  int64_t taskGrain;
  int64_t tileM;
  int64_t tileN;
  int64_t tileK;
  int64_t regionSize;
  mlir::DictionaryAttr local;

  int64_t parameter(llvm::StringRef name) const {
    return mlir::cast<mlir::IntegerAttr>(local.get(name)).getInt();
  }
};

mlir::LogicalResult fuseStructuredComputations(mlir::func::FuncOp function);
void forwardCPUOutputs(mlir::func::FuncOp function);
mlir::LogicalResult materializeStructuredComputations(mlir::func::FuncOp function);
mlir::LogicalResult fuseIntermediateBuffers(mlir::func::FuncOp function);
mlir::LogicalResult fuseReductionTraversals(mlir::func::FuncOp function);
mlir::LogicalResult blockContractions(mlir::func::FuncOp function,
                                    const Configuration &configuration,
                                    const ImplementationRegistry &implementations);
mlir::LogicalResult vectorizeLoops(mlir::func::FuncOp function, int64_t width,
                                   int64_t replicas, int64_t reductionReplicas);
mlir::LogicalResult partitionTasks(mlir::func::FuncOp function, int64_t grain);
mlir::LogicalResult isolateTasks(mlir::func::FuncOp function);
mlir::LogicalResult realizeRegions(mlir::func::FuncOp function, int64_t segmentSize);
mlir::LogicalResult materializeTaskLoops(mlir::func::FuncOp function);
mlir::LogicalResult runCPUPasses(mlir::ModuleOp module, int64_t vectorBits,
                               int64_t workers, llvm::StringRef defaults,
                               llvm::StringRef overrides,
                               const ImplementationRegistry &implementations);

}

#endif
