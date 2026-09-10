#ifndef INTENT_DIALECT_CPU_TRANSFORMS_IMPLEMENTATION_H
#define INTENT_DIALECT_CPU_TRANSFORMS_IMPLEMENTATION_H

#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include <functional>

namespace intent::cpu {

struct ContractionTile {
  mlir::Value lhs, rhs, output, initial;
  mlir::Value mBegin, mCount, nBegin, nCount, kBegin, depth;
  bool first;
};

struct ContractionRequirements {
  bool completePrivateInitialization = false;
  bool staticReductionExtent = false;
  bool staticParallelExtent = false;
};

struct Implementation {
  llvm::StringRef name;
  std::function<bool(mlir::Operation *)> applicable;
  std::function<bool(CapabilitiesAttr, const Configuration &)> legal;
  std::function<mlir::DictionaryAttr(mlir::Builder &, const Configuration &)> parameters;
  std::function<mlir::LogicalResult(mlir::OpBuilder &, mlir::linalg::GenericOp,
      const ContractionTile &, ConfigurationAttr, ImplementationAttr)> formTile;
  std::function<mlir::FailureOr<llvm::SmallVector<mlir::Value>>(
      mlir::OpBuilder &, mlir::Operation *, mlir::ValueRange, int64_t &)> expand;
  ContractionRequirements contraction;
  std::function<int64_t(ImplementationAttr)> parallelWindow;
};

class ImplementationRegistry {
public:
  std::function<llvm::StringRef(mlir::func::FuncOp)> profile;
  void add(Implementation implementation) { implementations.push_back(std::move(implementation)); }
  mlir::FailureOr<const Implementation *> select(mlir::Operation *operation) const;
  mlir::FailureOr<const Implementation *> lookup(mlir::Operation *operation) const;
  mlir::LogicalResult bind(mlir::func::FuncOp function, CapabilitiesAttr capabilities,
                           const Configuration &configuration) const;
  bool legal(mlir::func::FuncOp function, CapabilitiesAttr capabilities,
             const Configuration &configuration) const;

private:
  llvm::SmallVector<Implementation> implementations;
};

bool needsImplementation(mlir::Operation *operation);
int64_t implementationParameter(ImplementationAttr binding, llvm::StringRef name);

}
#endif
