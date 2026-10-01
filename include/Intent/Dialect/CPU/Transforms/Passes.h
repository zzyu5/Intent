#ifndef INTENT_DIALECT_CPU_TRANSFORMS_PASSES_H
#define INTENT_DIALECT_CPU_TRANSFORMS_PASSES_H

#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Transforms/Configuration.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "llvm/ADT/StringRef.h"

namespace intent::cpu {

class ImplementationRegistry;

#define GEN_PASS_DECL
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

struct CPUCompilationOptions : mlir::PassPipelineOptions<CPUCompilationOptions> {
  Option<std::string> provider{*this, "provider", llvm::cl::desc("Compiled CPU source provider")};
  Option<std::string> defaults{*this, "defaults", llvm::cl::desc("Provider default tuning profile file")};
  Option<std::string> overrides{*this, "overrides", llvm::cl::desc("Tuning profile overrides file")};
  Option<int64_t> vectorBits{*this, "vector-bits", llvm::cl::init(0)};
  Option<int64_t> workers{*this, "workers", llvm::cl::init(0)};
  Option<int64_t> privateBytes{*this, "private-bytes", llvm::cl::init(262144)};
  Option<bool> matrixI8I32{*this, "matrix-i8-i32", llvm::cl::init(false)};
};

void buildCPUPipeline(mlir::OpPassManager &manager, const CPUCompilationOptions &options);

void registerCPUPasses();
mlir::LogicalResult normalizeContractions(mlir::func::FuncOp function);
mlir::LogicalResult materializeCPUConfigurations(
    mlir::ModuleOp module, const ImplementationRegistry &implementations,
    llvm::StringRef defaults, llvm::StringRef overrides);

mlir::LogicalResult fuseStructuredComputations(mlir::func::FuncOp function);
void eraseDeadPrivateBuffers(mlir::func::FuncOp function);
mlir::LogicalResult reusePrivateStorage(mlir::func::FuncOp function);
mlir::LogicalResult foldUniformComputations(mlir::func::FuncOp function);
mlir::LogicalResult reusePreparedInputs(mlir::func::FuncOp function,
                                      const ImplementationRegistry &implementations);
mlir::LogicalResult groupRegionComputations(mlir::func::FuncOp function,
                                          const Configuration &configuration);
mlir::LogicalResult groupQuantizedDots(mlir::func::FuncOp function,
                                     const ImplementationRegistry &implementations);
mlir::LogicalResult groupWorksetComputations(mlir::func::FuncOp function,
                                           const ImplementationRegistry &implementations);
mlir::LogicalResult exposeStructuredWorksets(mlir::func::FuncOp function,
    const ImplementationRegistry &implementations, llvm::ArrayRef<mlir::Value> leadingExtents = {});
void forwardCPUOutputs(mlir::func::FuncOp function);
mlir::LogicalResult materializeStructuredComputations(mlir::func::FuncOp function);
mlir::LogicalResult realizeSliceScans(mlir::func::FuncOp function);
mlir::LogicalResult realizeHistograms(mlir::func::FuncOp function);
mlir::LogicalResult fuseIntermediateBuffers(mlir::func::FuncOp function);
mlir::LogicalResult fuseReductionTraversals(mlir::func::FuncOp function);
mlir::LogicalResult blockContractions(mlir::func::FuncOp function,
                                    const Configuration &configuration,
                                    const ImplementationRegistry &implementations);
mlir::LogicalResult blockStructuredComputations(mlir::func::FuncOp function,
                                    const ImplementationRegistry &implementations);
mlir::LogicalResult vectorizeLoops(mlir::func::FuncOp function, int64_t width,
                                   int64_t replicas, int64_t reductionReplicas);
mlir::LogicalResult partitionTasks(mlir::func::FuncOp function, int64_t grain,
                                 const ImplementationRegistry &implementations);
mlir::LogicalResult isolateTasks(mlir::func::FuncOp function);
mlir::LogicalResult realizeRegions(mlir::func::FuncOp function, const Configuration &configuration,
                                  const ImplementationRegistry &implementations);
mlir::LogicalResult materializeTaskDispatches(mlir::func::FuncOp function);

}

#endif
