#include "Intent/Dialect/GPU/Serialization/Python.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include <cmath>
#include <iomanip>
#include <sstream>

using namespace mlir;

namespace intent::gpu {

std::string pythonScalarType(Type type, const PythonScalarSyntax &syntax) {
  if (type.isIndex()) return (syntax.prefix + "int64").str();
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() == 1) return (syntax.prefix + syntax.boolean).str();
    StringRef sign = integer.isUnsigned() ? "u" : "";
    return (syntax.prefix + sign + "int" + Twine(integer.getWidth())).str();
  }
  StringRef spelling;
  if (isa<Float16Type>(type)) spelling = "float16";
  else if (isa<BFloat16Type>(type)) spelling = "bfloat16";
  else if (isa<Float32Type>(type)) spelling = "float32";
  else if (isa<Float64Type>(type)) spelling = syntax.float64;
  else if (isa<Float8E4M3FNType>(type)) spelling = syntax.float8E4M3FN;
  else if (isa<Float8E5M2Type>(type)) spelling = syntax.float8E5M2;
  return spelling.empty() ? std::string() : (syntax.prefix + spelling).str();
}

std::string pythonExpression(
    PhysicalExprAttr expression, const PythonExpressionSyntax &syntax,
    llvm::function_ref<std::string(PhysicalExprAttr)> symbol) {
  auto kind = expression.getKind();
  if (kind == PhysicalExprKind::Constant)
    return std::to_string(expression.getValue());
  if (kind == PhysicalExprKind::Parameter ||
      kind == PhysicalExprKind::Dimension ||
      kind == PhysicalExprKind::ScalarABI)
    return symbol(expression);

  SmallVector<std::string> operands;
  for (Attribute operand : expression.getOperands())
    operands.push_back(pythonExpression(cast<PhysicalExprAttr>(operand), syntax, symbol));
  auto infix = [&](StringRef operation) {
    return "(" + operands[0] + " " + operation.str() + " " + operands[1] + ")";
  };
  auto call = [&](StringRef function) {
    std::string result = function.str() + "(";
    for (auto [index, operand] : llvm::enumerate(operands)) {
      if (index) result += ", ";
      result += operand;
    }
    return result + ")";
  };
  switch (kind) {
  case PhysicalExprKind::Add: return infix("+");
  case PhysicalExprKind::Subtract: return infix("-");
  case PhysicalExprKind::Multiply: return infix("*");
  case PhysicalExprKind::FloorDiv: return infix("//");
  case PhysicalExprKind::CeilDiv:
    if (!syntax.ceilDivide.empty()) return call(syntax.ceilDivide);
    return "((" + operands[0] + " + " + operands[1] + " - 1) // " + operands[1] + ")";
  case PhysicalExprKind::Minimum: return call(syntax.minimum);
  case PhysicalExprKind::Maximum: return call(syntax.maximum);
  case PhysicalExprKind::Select:
    if (!syntax.select.empty()) return call(syntax.select);
    return "(" + operands[1] + " if " + operands[0] + " else " + operands[2] + ")";
  case PhysicalExprKind::NextPowerOfTwo:
    if (syntax.clampNextPowerOfTwo)
      operands[0] = "max(" + operands[0] + ", 1)";
    return call(syntax.nextPowerOfTwo);
  default: llvm_unreachable("unverified physical expression kind");
  }
}

std::string pythonLiteral(Attribute value,
                          llvm::function_ref<std::string(Type)> infinity) {
  if (auto integer = dyn_cast<IntegerAttr>(value)) {
    if (integer.getType().isInteger(1))
      return integer.getInt() ? "True" : "False";
    return std::to_string(integer.getInt());
  }
  if (auto floating = dyn_cast<FloatAttr>(value)) {
    double number = floating.getValueAsDouble();
    if (std::isnan(number)) return "float(\"nan\")";
    if (std::isinf(number))
      return (std::signbit(number) ? "-" : "") +
             (infinity ? infinity(floating.getType()) : "float(\"inf\")");
    std::ostringstream stream;
    stream << std::setprecision(17) << number;
    std::string result = stream.str();
    if (result.find_first_of(".eE") == std::string::npos) result += ".0";
    return result;
  }
  return {};
}

} // namespace intent::gpu
