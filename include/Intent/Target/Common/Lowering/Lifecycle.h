#ifndef INTENT_TARGET_COMMON_LOWERING_LIFECYCLE_H
#define INTENT_TARGET_COMMON_LOWERING_LIFECYCLE_H

#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::target {

class TargetProgramMaterializer {
public:
  virtual ~TargetProgramMaterializer() = default;

  virtual mlir::LogicalResult prepare() = 0;
  virtual void emitImports() = 0;
  virtual mlir::LogicalResult emitHelpers() = 0;
  virtual mlir::LogicalResult emitKernelHeader() = 0;
  virtual mlir::LogicalResult
  registerOperationHandlers(OperationHandlerRegistry &registry) = 0;
  virtual mlir::LogicalResult emitWrapper() = 0;

  virtual mlir::func::FuncOp entry() const = 0;
  virtual llvm::StringRef stage() const = 0;
  virtual llvm::raw_ostream &stream() = 0;

  void setOperationRegistry(const OperationHandlerRegistry *value) {
    registry = value;
  }

protected:
  const OperationHandlerRegistry *operationRegistry() const { return registry; }

private:
  const OperationHandlerRegistry *registry = nullptr;
};

mlir::LogicalResult materializeSource(TargetProgramMaterializer &emitter);

} // namespace intent::target

#endif
