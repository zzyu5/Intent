#ifndef INTENT_CPU_TRANSFORMS_IMPLEMENTATION_INPUTS_H
#define INTENT_CPU_TRANSFORMS_IMPLEMENTATION_INPUTS_H

#include "Intent/Dialect/CPU/Transforms/Implementation.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace intent::cpu {

class ImplementationInputs {
public:
  explicit ImplementationInputs(mlir::func::FuncOp function) : function(function) {}
  mlir::FailureOr<llvm::SmallVector<InputSupply>> prepare(mlir::linalg::GenericOp operation,
      llvm::ArrayRef<InputRequirement> requirements);
  mlir::FailureOr<InputSupply> prepareCaptured(mlir::linalg::GenericOp operation,
      mlir::memref::LoadOp input, const InputRequirement &requirement, mlir::Operation *scope);
  mlir::FailureOr<llvm::SmallVector<InputSupply>> prepareGroup(mlir::OpBuilder &builder,
      mlir::linalg::GenericOp operation, const ContractionTile &tile,
      ConfigurationAttr configuration, llvm::ArrayRef<InputRequirement> requirements);

private:
  mlir::Operation *consumerScope(mlir::Value source, mlir::linalg::GenericOp operation);
  InputSupply materialize(mlir::Value source, const InputRequirement &requirement, mlir::Operation *scope);
  struct Prepared {
    mlir::Value source;
    InputRequirement requirement;
    mlir::memref::AllocOp allocation;
    mlir::memref::DeallocOp end;
  };
  mlir::func::FuncOp function;
  llvm::SmallVector<Prepared> prepared;
  llvm::SmallVector<mlir::Operation *> guardedLoops;
};

}
#endif
