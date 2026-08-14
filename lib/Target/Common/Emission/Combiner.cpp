#include "Intent/Target/Common/Emission/Combiner.h"

#include "Intent/Target/Common/Analysis/Record.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace intent::target::emission {
namespace {

FailureOr<std::string> constantExpression(Operation &operation) {
  Attribute value = operation.getAttr("intent.value");
  if (auto integer = dyn_cast_or_null<IntegerAttr>(value)) {
    if (operation.getNumResults() == 1 &&
        operation.getResult(0).getType().isInteger(1))
      return integer.getValue().isZero() ? std::string("False")
                                         : std::string("True");
    return std::to_string(integer.getInt());
  }
  auto floating = dyn_cast_or_null<FloatAttr>(value);
  if (!floating)
    return operation.emitOpError("has no Python combiner literal spelling");
  const llvm::APFloat &number = floating.getValue();
  if (number.isNaN())
    return std::string("float('nan')");
  if (number.isInfinity())
    return number.isNegative() ? std::string("-float('inf')")
                               : std::string("float('inf')");
  llvm::SmallString<32> spelling;
  number.toString(spelling);
  return spelling.str().str();
}

} // namespace

bool hasGenericCombiner(Operation &operation) {
  return isa_and_nonnull<FlatSymbolRefAttr>(operation.getAttr("intent.combine"));
}

FailureOr<CombinerUse> resolveCombiner(Operation &operation) {
  auto reference = operation.getAttrOfType<FlatSymbolRefAttr>("intent.combine");
  auto components =
      operation.getAttrOfType<IntegerAttr>("intent.component_count");
  auto captures = operation.getAttrOfType<IntegerAttr>("intent.capture_count");
  auto module = operation.getParentOfType<ModuleOp>();
  auto function = reference && module
                      ? module.lookupSymbol<func::FuncOp>(reference.getValue())
                      : func::FuncOp();
  auto role = function
                  ? function->getAttrOfType<StringAttr>("intent.role")
                  : StringAttr();
  if (!reference || !components || components.getInt() <= 0 || !captures ||
      captures.getInt() < 0 || !function || !role ||
      role.getValue() != "combiner") {
    operation.emitOpError("does not reference a canonical typed combiner");
    return failure();
  }
  unsigned componentCount = static_cast<unsigned>(components.getInt());
  unsigned captureCount = static_cast<unsigned>(captures.getInt());
  if (operation.getNumOperands() != 2 * componentCount + captureCount ||
      operation.getNumResults() != componentCount ||
      function.getNumArguments() != 2 * componentCount + captureCount ||
      function.getNumResults() != componentCount) {
    operation.emitOpError("has a mismatched typed combiner ABI");
    return failure();
  }
  auto elementType = [](Type type) {
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      return tensor.getElementType();
    return type;
  };
  for (unsigned component = 0; component < componentCount; ++component) {
    Type identity = operation.getOperand(componentCount + component).getType();
    if (elementType(operation.getOperand(component).getType()) != identity ||
        function.getArgument(component).getType() != identity ||
        function.getArgument(componentCount + component).getType() != identity ||
        function.getResultTypes()[component] != identity ||
        elementType(operation.getResult(component).getType()) != identity) {
      operation.emitOpError("has a mismatched combiner accumulator type");
      return failure();
    }
  }
  for (unsigned capture = 0; capture < captureCount; ++capture) {
    Type type = operation.getOperand(2 * componentCount + capture).getType();
    if (!isa<IntegerType, IndexType, FloatType>(type) ||
        function.getArgument(2 * componentCount + capture).getType() != type) {
      operation.emitOpError("has a non-scalar or mismatched combiner capture");
      return failure();
    }
  }
  return CombinerUse{function, componentCount, captureCount};
}

FailureOr<SmallVector<func::FuncOp>> collectCombiners(func::FuncOp entry) {
  SmallVector<func::FuncOp> functions;
  DenseSet<Operation *> seen;
  WalkResult walk = entry.walk([&](Operation *operation) {
    if (!hasGenericCombiner(*operation))
      return WalkResult::advance();
    if (operation->getAttrOfType<StringAttr>("intent.combine_builtin"))
      return WalkResult::advance();
    FailureOr<CombinerUse> combiner = resolveCombiner(*operation);
    if (failed(combiner))
      return WalkResult::interrupt();
    if (seen.insert(combiner->function.getOperation()).second)
      functions.push_back(combiner->function);
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return failure();
  return functions;
}

FailureOr<std::string> combinerProjectionName(Operation &operation) {
  FailureOr<CombinerUse> combiner = resolveCombiner(operation);
  auto node = operation.getAttrOfType<IntegerAttr>("intent.node");
  if (failed(combiner) || !node || node.getInt() < 0)
    return operation.emitOpError(
        "has no stable node identity for its combiner projection");
  return combiner->function.getName().str() + "_projection_" +
         std::to_string(node.getInt());
}

FailureOr<std::string>
renderPythonCombiner(func::FuncOp function, StringRef emittedName,
                     StringRef decorator,
                     CombinerExpressionEmitter emitExpression) {
  if (!llvm::hasSingleElement(function.getBody())) {
    function.emitOpError("combiner source rendering requires one block");
    return failure();
  }
  DenseMap<Value, std::string> expressions;
  std::string source;
  llvm::raw_string_ostream output(source);
  if (!decorator.empty())
    output << decorator << "\n";
  output << "def " << emittedName << "(";
  for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
    if (index)
      output << ", ";
    std::string name = "arg_" + std::to_string(index);
    output << name;
    expressions[argument] = std::move(name);
  }
  output << "):\n";

  unsigned temporary = 0;
  unsigned returnCount = 0;
  SmallVector<std::string> returns;
  for (Operation &operation : function.getBody().front()) {
    StringRef name = operation.getName().getStringRef();
    if (name == "intent.make_record")
      continue;
    if (name == "intent.return") {
      if (++returnCount != 1 || &operation != &function.getBody().front().back()) {
        operation.emitOpError(
            "combiner source rendering requires one final return");
        return failure();
      }
      for (Value operand : operation.getOperands()) {
        auto found = expressions.find(operand);
        if (found == expressions.end()) {
          operation.emitOpError("returns an unrendered combiner value");
          return failure();
        }
        returns.push_back(found->second);
      }
      continue;
    }

    FailureOr<std::string> expression = failure();
    if (name == "intent.constant") {
      expression = constantExpression(operation);
    } else if (name == "intent.extract") {
      FailureOr<Value> field = target::resolveRecordField(operation);
      auto found = succeeded(field) ? expressions.find(*field) : expressions.end();
      if (failed(field) || found == expressions.end())
        return operation.emitOpError(
            "record field has no rendered combiner source");
      expression = found->second;
    } else {
      SmallVector<std::string> operands;
      for (Value operand : operation.getOperands()) {
        auto found = expressions.find(operand);
        if (found == expressions.end())
          return operation.emitOpError(
              "uses an unrendered combiner operand");
        operands.push_back(found->second);
      }
      expression = emitExpression(operation, operands);
    }
    if (failed(expression) || operation.getNumResults() != 1)
      return operation.emitOpError(
          "does not have a mechanical Python combiner expression");
    std::string result = "value_" + std::to_string(temporary++);
    output << "    " << result << " = " << *expression << "\n";
    expressions[operation.getResult(0)] = std::move(result);
  }
  if (returns.empty()) {
    function.emitOpError("combiner source rendering found no return values");
    return failure();
  }
  output << "    return ";
  if (returns.size() == 1)
    output << returns.front();
  else
    output << "(" << llvm::join(returns, ", ") << ")";
  output << "\n\n\n";
  output.flush();
  return source;
}

FailureOr<std::string>
renderPythonCombinerProjection(Operation &operation, StringRef decorator,
                               StringRef selectSpelling) {
  FailureOr<CombinerUse> combiner = resolveCombiner(operation);
  FailureOr<std::string> name = combinerProjectionName(operation);
  if (failed(combiner) || failed(name) || combiner->captureCount == 0 ||
      selectSpelling.empty())
    return operation.emitOpError(
        "has no runtime-capture combiner projection to render");

  const unsigned components = combiner->componentCount;
  const unsigned captures = combiner->captureCount;
  const unsigned extended = components + captures + 1;
  std::string source;
  llvm::raw_string_ostream output(source);
  if (!decorator.empty())
    output << decorator << "\n";
  output << "def " << *name << "(";
  for (unsigned index = 0; index < 2 * extended; ++index) {
    if (index)
      output << ", ";
    output << (index < extended ? "lhs_" : "rhs_")
           << (index < extended ? index : index - extended);
  }
  output << "):\n";
  for (unsigned capture = 0; capture < captures; ++capture) {
    unsigned position = components + capture;
    output << "    capture_" << capture << " = " << selectSpelling
           << "(lhs_" << (extended - 1) << ", lhs_" << position
           << ", rhs_" << position << ")\n";
  }
  output << "    capture_valid = lhs_" << (extended - 1) << " | rhs_"
         << (extended - 1) << "\n";
  output << "    ";
  for (unsigned component = 0; component < components; ++component) {
    if (component)
      output << ", ";
    output << "result_" << component;
  }
  output << " = " << combiner->function.getName() << "(";
  bool first = true;
  auto argument = [&](StringRef value) {
    if (!first)
      output << ", ";
    output << value;
    first = false;
  };
  for (unsigned component = 0; component < components; ++component)
    argument("lhs_" + std::to_string(component));
  for (unsigned component = 0; component < components; ++component)
    argument("rhs_" + std::to_string(component));
  for (unsigned capture = 0; capture < captures; ++capture)
    argument("capture_" + std::to_string(capture));
  output << ")\n    return ";
  first = true;
  for (unsigned component = 0; component < components; ++component)
    argument("result_" + std::to_string(component));
  for (unsigned capture = 0; capture < captures; ++capture)
    argument("capture_" + std::to_string(capture));
  argument("capture_valid");
  output << "\n\n\n";
  output.flush();
  return source;
}

FailureOr<std::string> renderPythonPointwiseExpression(
    Operation &operation, ArrayRef<std::string> operands, StringRef lowering,
    CombinerCastSpelling castSpelling) {
  auto binary = [&](StringRef symbol) -> FailureOr<std::string> {
    if (operands.size() != 2)
      return failure();
    return "(" + operands[0] + " " + symbol.str() + " " + operands[1] + ")";
  };
  if (lowering == "python_add")
    return binary("+");
  if (lowering == "python_subtract")
    return binary("-");
  if (lowering == "python_multiply")
    return binary("*");
  if (lowering == "python_true_divide")
    return binary("/");
  if (lowering == "python_floor_divide")
    return binary("//");
  if (lowering == "python_remainder")
    return binary("%");
  if (lowering == "python_bitwise_and" || lowering == "python_logical_and")
    return binary("&");
  if (lowering == "python_bitwise_or" || lowering == "python_logical_or")
    return binary("|");
  if (lowering == "python_bitwise_xor")
    return binary("^");
  if (lowering == "python_left_shift")
    return binary("<<");
  if (lowering == "python_right_shift")
    return binary(">>");
  if (lowering == "python_equal")
    return binary("==");
  if (lowering == "python_not_equal")
    return binary("!=");
  if (lowering == "python_less")
    return binary("<");
  if (lowering == "python_less_equal")
    return binary("<=");
  if (lowering == "python_greater")
    return binary(">");
  if (lowering == "python_greater_equal")
    return binary(">=");
  if (lowering == "python_negate" && operands.size() == 1)
    return "(-" + operands[0] + ")";
  if (lowering == "python_not" && operands.size() == 1)
    return "(~" + operands[0] + ")";
  if ((lowering.ends_with(".cast") || lowering.ends_with(".astype")) &&
      operands.size() == 1 && operation.getNumResults() == 1) {
    FailureOr<std::string> dtype = castSpelling(operation.getResult(0).getType());
    if (failed(dtype))
      return failure();
    return lowering.str() + "(" + operands[0] + ", " + *dtype + ")";
  }
  if (lowering.ends_with(".where") && operands.size() == 3)
    return lowering.str() + "(" + operands[0] + ", " + operands[1] +
           ", " + operands[2] + ")";
  if (operands.size() == 1)
    return lowering.str() + "(" + operands[0] + ")";
  if (operands.size() == 2)
    return lowering.str() + "(" + operands[0] + ", " + operands[1] + ")";
  return operation.emitOpError(
      "has no mechanical Python pointwise expression spelling");
}

} // namespace intent::target::emission
