#ifndef INTENT_DIALECT_GPU_ANALYSIS_PHYSICALPARAMETERS_H
#define INTENT_DIALECT_GPU_ANALYSIS_PHYSICALPARAMETERS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {

struct PhysicalParameterDomain {
  ParameterOp operation;
  ParameterAttr definition;
  bool coverage;
  bool provider;

  mlir::StringAttr name() const { return definition.getName(); }
  ParameterRole role() const {
    return static_cast<ParameterRole>(definition.getRole());
  }
  llvm::ArrayRef<int64_t> candidates() const {
    return definition.getCandidates().asArrayRef();
  }
};

// A read-only snapshot of declarations in the current physical kernel. Rebuild
// after changing a declaration or its candidate domain. Configs remain IR
// attributes; this analysis neither chooses candidates nor owns execution data.
class PhysicalParameterSpace {
public:
  static mlir::FailureOr<PhysicalParameterSpace> read(mlir::func::FuncOp kernel);
  llvm::ArrayRef<PhysicalParameterDomain> domains() const { return parameters; }
  const PhysicalParameterDomain *find(ParameterRole role) const;
  mlir::FailureOr<llvm::SmallVector<mlir::DictionaryAttr>>
  sharedConfigurations() const;

private:
  mlir::func::FuncOp kernel;
  llvm::SmallVector<PhysicalParameterDomain> parameters;
};

} // namespace intent::gpu
#endif
