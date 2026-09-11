#ifndef INTENT_ANALYSIS_UNIFORMVALUES_H
#define INTENT_ANALYSIS_UNIFORMVALUES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>
#include <optional>

namespace intent {

enum class UniformKind {
  Unknown, Constant, Forward, Join, Aggregate, Extract, Select, Cast, Bitcast,
  Negate, Not, Add, Subtract, Multiply, And, Or, Xor, Exp,
  Maximum, Minimum, MaximumNum, MinimumNum, Compare, Fold, Contract
};
enum class UniformPredicate { Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual, Ordered, Unordered };

// Dialect adapters describe current typed def-use, never a replacement program.
struct UniformExpression {
  UniformKind kind = UniformKind::Unknown;
  mlir::Type type;
  mlir::Attribute literal;
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<mlir::Value> parameters;
  llvm::SmallVector<mlir::Value> yields;
  unsigned stateCount = 0;
  unsigned result = 0;
  bool nonempty = false;
  bool unsignedInput = false;
  bool unsignedOutput = false;
  bool unorderedTrue = false;
  UniformPredicate predicate = UniformPredicate::Equal;
};

using UniformBindings = llvm::DenseMap<mlir::Value, mlir::Attribute>;
bool equalUniformConstants(mlir::Attribute lhs, mlir::Attribute rhs);
mlir::Attribute uniformZero(mlir::Type type);
std::optional<bool> uniformBoolean(mlir::Attribute value);
UniformExpression describeScalarValue(mlir::Value value);

class UniformValueAnalysis {
public:
  using Describe = std::function<UniformExpression(mlir::Value)>;
  explicit UniformValueAnalysis(Describe describe) : describe(std::move(describe)) {}
  mlir::Attribute evaluate(mlir::Value value, const UniformBindings &bindings = UniformBindings()) const;
  mlir::Attribute fold(const UniformExpression &expression, const UniformBindings &bindings = UniformBindings()) const;

private:
  Describe describe;
  mlir::Attribute evaluate(mlir::Value value, const UniformBindings &bindings,
                          llvm::SmallDenseSet<mlir::Value> &visiting) const;
  mlir::Attribute fold(const UniformExpression &expression, const UniformBindings &bindings,
                      llvm::SmallDenseSet<mlir::Value> &visiting) const;
};

}
#endif
