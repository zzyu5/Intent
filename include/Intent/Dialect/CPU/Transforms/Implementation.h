#ifndef INTENT_DIALECT_CPU_TRANSFORMS_IMPLEMENTATION_H
#define INTENT_DIALECT_CPU_TRANSFORMS_IMPLEMENTATION_H

#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include <functional>

namespace intent::cpu {

enum class InputReuse { Group, Consumers };

// Storage order is [panel, unsplit source axes..., lane within panel]. A group
// supply covers one compute group; consumer reuse preserves a source snapshot.
struct InputRequirement {
  unsigned operand;
  unsigned panelAxis;
  int64_t panelSize;
  int64_t alignment;
  InputReuse reuse;
};

struct InputSupply {
  unsigned operand;
  unsigned panelAxis;
  int64_t panelSize;
  mlir::Value storage;
  llvm::SmallVector<mlir::Value> begins; // Logical source coordinates of the supplied window.
};

struct ContractionTile {
  mlir::Value lhs, rhs, output, initial;
  mlir::Value mBegin, mCount, nBegin, nCount, kBegin, depth;
  bool first;
  llvm::ArrayRef<InputSupply> inputs;
};

struct ContractionRequirements {
  bool completePrivateInitialization = false;
  bool staticReductionExtent = false;
  bool staticParallelExtent = false;
};

struct Implementation {
  llvm::StringRef name;
  std::function<bool(mlir::Operation *)> applicable;
  std::function<bool(mlir::Operation *, CapabilitiesAttr, const Configuration &)> legal;
  std::function<mlir::DictionaryAttr(mlir::Builder &, const Configuration &)> parameters;
  std::function<mlir::LogicalResult(mlir::OpBuilder &, mlir::linalg::GenericOp,
      const ContractionTile &, ConfigurationAttr, ImplementationAttr)> formTile;
  std::function<mlir::FailureOr<llvm::SmallVector<mlir::Value>>(
      mlir::OpBuilder &, mlir::Operation *, mlir::ValueRange, int64_t &)> expand;
  ContractionRequirements contraction;
  std::function<int64_t(ImplementationAttr)> parallelWindow;
  bool requiresMatrixI8I32 = false;
  std::function<llvm::SmallVector<InputRequirement>(mlir::linalg::GenericOp,
      ConfigurationAttr, ImplementationAttr)> inputs;
};

class ImplementationRegistry {
public:
  std::function<llvm::StringRef(mlir::func::FuncOp)> profile;
  void add(Implementation implementation) { implementations.push_back(std::move(implementation)); }
  mlir::FailureOr<const Implementation *> lookup(mlir::Operation *operation) const;
  llvm::SmallVector<llvm::SmallVector<ImplementationAttr>> candidates(
      mlir::func::FuncOp function, CapabilitiesAttr capabilities,
      const Configuration &configuration) const;
  mlir::LogicalResult bind(mlir::func::FuncOp function, CapabilitiesAttr capabilities,
                           const Configuration &configuration,
                           llvm::ArrayRef<ImplementationAttr> bindings) const;

private:
  llvm::SmallVector<Implementation> implementations;
};

bool needsImplementation(mlir::Operation *operation);
int64_t implementationParameter(ImplementationAttr binding, llvm::StringRef name);

}
#endif
