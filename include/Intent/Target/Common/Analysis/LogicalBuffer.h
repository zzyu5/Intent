#ifndef INTENT_TARGET_COMMON_ANALYSIS_LOGICALBUFFER_H
#define INTENT_TARGET_COMMON_ANALYSIS_LOGICALBUFFER_H

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include <optional>

namespace intent::target {

struct LogicalBufferInfo {
  llvm::SmallVector<int64_t> shape;
  mlir::Type elementType;
};

struct LogicalBufferIndex {
  std::optional<unsigned> operand;
  std::optional<int64_t> constant;
};

inline mlir::FailureOr<LogicalBufferInfo>
getLogicalBufferInfo(mlir::Operation &operation) {
  if (operation.getName().getStringRef() != "intent.buffer" ||
      operation.getNumOperands() != 1 || operation.getNumResults() != 1 ||
      !mlir::isa<intent::BufferType>(operation.getResult(0).getType()))
    return operation.emitOpError(
        "private logical buffer requires one initializer and one buffer result");
  mlir::Type elementType = operation.getOperand(0).getType();
  if (!mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(
          elementType))
    return operation.emitOpError(
        "private logical buffer requires a scalar initializer");
  auto resultShapes =
      operation.getAttrOfType<mlir::ArrayAttr>("intent.result_shapes");
  auto shape = resultShapes && resultShapes.size() == 1
                   ? mlir::dyn_cast<mlir::ArrayAttr>(resultShapes[0])
                   : mlir::ArrayAttr();
  if (!shape || shape.empty())
    return operation.emitOpError(
        "private logical buffer requires a nonempty static shape");
  LogicalBufferInfo result;
  result.elementType = elementType;
  for (mlir::Attribute attribute : shape) {
    auto extent = mlir::dyn_cast<mlir::StringAttr>(attribute);
    int64_t value = 0;
    if (!extent || extent.getValue().getAsInteger(10, value) || value <= 0)
      return operation.emitOpError(
          "private logical buffer requires positive static extents");
    result.shape.push_back(value);
  }
  return result;
}

inline mlir::FailureOr<LogicalBufferIndex>
getLogicalBufferIndex(mlir::Operation &operation) {
  mlir::FailureOr<llvm::SmallVector<IndexTerm>> relation =
      parseIndexRelation(operation);
  if (mlir::failed(relation) || relation->size() != 1)
    return operation.emitOpError(
        "private logical buffer requires one scalar index");
  const IndexTerm &term = relation->front();
  if (term.kind == "value_index" && term.operands.size() == 1 &&
      term.operands.front() &&
      *term.operands.front() < operation.getNumOperands())
    return LogicalBufferIndex{*term.operands.front(), std::nullopt};
  if (term.kind == "static_index" && term.staticValues.size() == 1 &&
      term.staticValues.front().has_value())
    return LogicalBufferIndex{std::nullopt, *term.staticValues.front()};
  return operation.emitOpError(
      "private logical buffer index is not canonical");
}

} // namespace intent::target

#endif
