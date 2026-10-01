#ifndef INTENT_DIALECT_GPU_SERIALIZATION_PYTHON_H
#define INTENT_DIALECT_GPU_SERIALIZATION_PYTHON_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace intent::gpu {

struct PythonScalarSyntax {
  llvm::StringRef prefix;
  llvm::StringRef boolean;
  // An empty spelling preserves a provider's unsupported scalar type.
  llvm::StringRef float64;
  llvm::StringRef float8E4M3FN;
  llvm::StringRef float8E5M2;
};

std::string pythonScalarType(mlir::Type type, const PythonScalarSyntax &syntax);

struct PythonExpressionSyntax {
  // Empty ceilDivide/select names request their ordinary Python expression.
  llvm::StringRef ceilDivide;
  llvm::StringRef minimum;
  llvm::StringRef maximum;
  llvm::StringRef select;
  llvm::StringRef nextPowerOfTwo;
  bool clampNextPowerOfTwo;
};

std::string pythonExpression(
    PhysicalExprAttr expression, const PythonExpressionSyntax &syntax,
    llvm::function_ref<std::string(PhysicalExprAttr)> symbol);

// The callback only spells a positive infinity; this function preserves its
// sign and owns bool/integer, finite floating-point and NaN spelling.
std::string pythonLiteral(
    mlir::Attribute value,
    llvm::function_ref<std::string(mlir::Type)> infinity = nullptr);

} // namespace intent::gpu
#endif
