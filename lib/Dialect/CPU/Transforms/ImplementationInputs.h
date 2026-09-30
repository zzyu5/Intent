#ifndef INTENT_CPU_TRANSFORMS_IMPLEMENTATION_INPUTS_H
#define INTENT_CPU_TRANSFORMS_IMPLEMENTATION_INPUTS_H

#include "Intent/Dialect/CPU/Transforms/Implementation.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intent::cpu {

struct ConsumerWindow {
  mlir::memref::SubViewOp view;
  unsigned axis;
  bool transposed;
};

std::optional<ConsumerWindow> consumerWindow(mlir::Value source,
    const InputRequirement &requirement, mlir::Operation *consumer);
bool hasIndependentWindowCoordinates(mlir::memref::SubViewOp window,
    mlir::Operation *loop, mlir::Value groupCoordinate = {});

class ImplementationInputs {
public:
  explicit ImplementationInputs(mlir::func::FuncOp function) : function(function) {}
  mlir::FailureOr<llvm::SmallVector<InputSupply>> prepare(mlir::linalg::GenericOp operation,
      llvm::ArrayRef<InputRequirement> requirements, const Implementation &implementation);
  mlir::FailureOr<InputSupply> prepareCaptured(mlir::linalg::GenericOp operation,
      mlir::memref::LoadOp input, const InputRequirement &requirement, mlir::Operation *scope);
  mlir::FailureOr<llvm::SmallVector<InputSupply>> prepareGroup(mlir::OpBuilder &builder,
      mlir::linalg::GenericOp operation, const ContractionTile &tile,
      ConfigurationAttr configuration, llvm::ArrayRef<InputRequirement> requirements);

private:
  std::optional<InputSupply> prepareWindow(mlir::Value source, const InputRequirement &requirement,
      mlir::linalg::GenericOp operation);
  void guardLoop(mlir::scf::ForOp loop);
  mlir::Operation *consumerScope(mlir::Value source, mlir::linalg::GenericOp operation,
      const InputRequirement &requirement, const Implementation &implementation);
  InputSupply materialize(mlir::Value source, const InputRequirement &requirement, mlir::Operation *scope);
  struct Prepared {
    mlir::Value source;
    InputRequirement requirement;
    mlir::memref::AllocOp allocation;
  };
  struct PreparedWindow {
    mlir::Value source;
    InputRequirement requirement;
    unsigned axis;
    bool transposed;
    mlir::scf::ForOp scope;
    mlir::memref::AllocOp storage;
    mlir::memref::AllocOp initialized;
  };
  mlir::func::FuncOp function;
  llvm::SmallVector<Prepared> prepared;
  llvm::SmallVector<PreparedWindow> windows;
  llvm::SmallVector<mlir::Operation *> guardedLoops;
};

}
#endif
