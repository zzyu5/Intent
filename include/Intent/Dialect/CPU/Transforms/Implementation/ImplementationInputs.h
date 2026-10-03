#ifndef INTENT_DIALECT_CPU_TRANSFORMS_IMPLEMENTATIONINPUTS_H
#define INTENT_DIALECT_CPU_TRANSFORMS_IMPLEMENTATIONINPUTS_H

#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include <memory>

namespace mlir::memref {
class LoadOp;
}

namespace intent::cpu {

// One context per function owns prepared input snapshots and their reuse.
class ImplementationInputs {
public:
  explicit ImplementationInputs(mlir::func::FuncOp function);
  ~ImplementationInputs();

  mlir::FailureOr<llvm::SmallVector<InputSupply>> prepare(
      mlir::linalg::GenericOp operation,
      llvm::ArrayRef<InputRequirement> requirements,
      const Implementation &implementation);
  mlir::FailureOr<InputSupply> prepareCaptured(
      mlir::linalg::GenericOp operation, mlir::memref::LoadOp input,
      const InputRequirement &requirement, mlir::Operation *scope);
  mlir::FailureOr<llvm::SmallVector<InputSupply>> prepareGroup(
      mlir::OpBuilder &builder, mlir::linalg::GenericOp operation,
      const ContractionTile &tile, ConfigurationAttr configuration,
      llvm::ArrayRef<InputRequirement> requirements);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

mlir::LogicalResult reusePreparedInputs(
    mlir::func::FuncOp function, const ImplementationRegistry &implementations);
mlir::LogicalResult groupQuantizedDots(
    mlir::func::FuncOp function, const ImplementationRegistry &implementations);

} // namespace intent::cpu
#endif
