#ifndef INTENT_DIALECT_GPU_SERIALIZATION_INTERFACE_H
#define INTENT_DIALECT_GPU_SERIALIZATION_INTERFACE_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <string>

namespace intent::gpu {

struct ViewArgument {
  unsigned abi;
  std::string name;
  ViewType type;
  bool workspace;
};

struct ScalarArgument {
  unsigned abi;
  std::string name;
  std::string kind;
  mlir::Type type;
};

struct MetadataArgument {
  unsigned abi;
  std::string name;
  std::string kind;
  unsigned sourceABI;
  unsigned sourceAxis;
  std::optional<int64_t> dimension;
};

// ABI order and view contracts are facts of the final physical program. Names
// here are the public IR names; a provider separately spells its kernel names.
struct KernelInterface {
  llvm::SmallVector<ViewArgument> views;
  llvm::SmallVector<ScalarArgument> scalars;
  llvm::SmallVector<MetadataArgument> metadata;
  llvm::SmallVector<unsigned> publicArguments;
};

mlir::FailureOr<KernelInterface> readInterface(mlir::func::FuncOp kernel);

// The expression tree preserves IR symbols and operators. It is data for the
// runtime evaluator, never a generated Python expression or a new policy.
llvm::json::Value serializeExpression(PhysicalExprAttr expression);

// Returns {provider, interface}; the provider appends its own named object for
// native launch details. The callback reports the names actually serialized,
// including packed-argument projections when those are part of the native ABI.
// Bindings outside gpu.parameter declarations require the provider's verifier;
// its callback sees exactly those bindings, including an empty dictionary.
mlir::FailureOr<llvm::json::Object> serializeInterface(
    mlir::func::FuncOp kernel, llvm::StringRef provider,
    llvm::function_ref<std::string(mlir::Value)> kernelName,
    llvm::function_ref<mlir::LogicalResult(mlir::DictionaryAttr)>
        verifyProviderBindings = {});

} // namespace intent::gpu
#endif
