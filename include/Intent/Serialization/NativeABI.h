#ifndef INTENT_SERIALIZATION_NATIVEABI_H
#define INTENT_SERIALIZATION_NATIVEABI_H

#include "Intent/Dialect/Intent/IR/Interface.h"
#include <optional>

namespace intent {

enum class NativeSlotRole { Pointer, Extent, Stride, Scalar };

struct NativeSlot {
  NativeSlotRole role;
  unsigned parameter;
  std::optional<unsigned> axis;
  mlir::Type carrier;
  mlir::Type element;

  std::string name() const;
};

struct NativeABI {
  llvm::SmallVector<NativeSlot> slots;
  llvm::json::Array serialize() const;
};

// CPU/DSA native entry flattening. The family has already verified each public
// argument's physical storage; only the provider decides its scalar C carrier.
mlir::FailureOr<NativeABI> queryNativeABI(
    mlir::func::FuncOp function, InterfaceAttr interface,
    llvm::function_ref<mlir::FailureOr<mlir::Type>(mlir::Type)> scalarCarrier);

} // namespace intent
#endif
