#ifndef INTENT_DIALECT_CPU_ANALYSIS_PHYSICALPROGRAM_H
#define INTENT_DIALECT_CPU_ANALYSIS_PHYSICALPROGRAM_H

#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include <optional>

namespace intent::cpu {

struct AllocationFacts {
  mlir::Value value;
  mlir::Operation *owner;
  mlir::Operation *writer;
  std::optional<int64_t> bytes;
  bool stack;
};

struct MemoryAccess {
  mlir::Operation *operation;
  mlir::Value memory;
  bool read;
  bool write;
};

// A query object belongs to the current IR snapshot; recreate after a rewrite.
class PhysicalProgramAnalysis {
public:
  explicit PhysicalProgramAnalysis(mlir::func::FuncOp function) : function(function) {}
  mlir::Value storageRoot(mlir::Value memory);
  ViewArgumentAttr externalView(mlir::Value memory);
  bool isReadOnly(mlir::Value memory);
  bool mayReadAt(mlir::Value memory, mlir::Operation *from,
                 mlir::Operation *to);
  llvm::SmallVector<AllocationFacts> allocations();
  llvm::SmallVector<MemoryAccess> accesses(mlir::Operation *scope);
  mlir::LogicalResult verify(bool realized);

private:
  mlir::func::FuncOp function;
};

bool isMatrixContraction(mlir::linalg::GenericOp operation);
mlir::LogicalResult verifyCPUProgram(mlir::ModuleOp module, bool realized);

}
#endif
