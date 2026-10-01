#ifndef INTENT_GPU_ANALYSIS_CONFIGURATIONS_H
#define INTENT_GPU_ANALYSIS_CONFIGURATIONS_H

#include "Intent/Dialect/GPU/IR/ConfigurationParameterOpInterface.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {

// Read-only declarations from the current kernel. Rebuild after changing any
// parameter, provider choice or binding phase. Candidate rows remain in IR.
class ConfigurationSpace {
public:
  static mlir::FailureOr<ConfigurationSpace> read(mlir::func::FuncOp kernel);
  llvm::ArrayRef<ConfigurationParameterOpInterface> parameters() const {
    return declarations;
  }
  mlir::LogicalResult verifyBindings(mlir::DictionaryAttr bindings,
                                    ConfigurationStage stage) const;
  mlir::FailureOr<llvm::SmallVector<mlir::DictionaryAttr>>
  configurations(ConfigurationStage stage) const;

private:
  mlir::func::FuncOp kernel;
  llvm::SmallVector<ConfigurationParameterOpInterface> declarations;
};

} // namespace intent::gpu
#endif
