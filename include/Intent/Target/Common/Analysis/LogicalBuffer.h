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
#include <limits>

namespace intent::target {

struct LogicalBufferInfo {
  llvm::SmallVector<int64_t> shape;
  mlir::Type elementType;
};

struct LogicalBufferIndex {
  std::optional<unsigned> operand;
  std::optional<int64_t> constant;
};

inline mlir::FailureOr<int64_t>
logicalBufferElementCount(const LogicalBufferInfo &info,
                          mlir::Operation &operation) {
  int64_t count = 1;
  for (int64_t extent : info.shape) {
    if (count > std::numeric_limits<int64_t>::max() / extent)
      return operation.emitOpError(
          "private logical-buffer extent overflows target allocation");
    count *= extent;
  }
  return count;
}

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

inline mlir::FailureOr<llvm::SmallVector<LogicalBufferIndex>>
getLogicalBufferIndices(mlir::Operation &operation, size_t expectedRank) {
  mlir::FailureOr<llvm::SmallVector<IndexTerm>> relation =
      parseIndexRelation(operation);
  if (mlir::failed(relation) || relation->size() != expectedRank)
    return operation.emitOpError(
        "private logical buffer index rank does not match its shape");
  llvm::SmallVector<LogicalBufferIndex> result;
  result.reserve(relation->size());
  for (const IndexTerm &term : *relation) {
    if (term.kind == "value_index" && term.operands.size() == 1 &&
        term.operands.front() &&
        *term.operands.front() < operation.getNumOperands()) {
      result.push_back(
          LogicalBufferIndex{*term.operands.front(), std::nullopt});
      continue;
    }
    if (term.kind == "static_index" && term.staticValues.size() == 1 &&
        term.staticValues.front().has_value()) {
      result.push_back(
          LogicalBufferIndex{std::nullopt, *term.staticValues.front()});
      continue;
    }
    return operation.emitOpError(
        "private logical buffer index is not canonical");
  }
  return result;
}

inline mlir::FailureOr<LogicalBufferIndex>
getLogicalBufferIndex(mlir::Operation &operation) {
  mlir::FailureOr<llvm::SmallVector<LogicalBufferIndex>> indices =
      getLogicalBufferIndices(operation, 1);
  if (mlir::failed(indices))
    return mlir::failure();
  return indices->front();
}

} // namespace intent::target

#endif
