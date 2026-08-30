#include "Intent/Dialect/Intent/IR/IntentOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>

using namespace mlir;

namespace intent {
namespace {

bool isIntegerLike(Type type) {
  return isa<IntegerType, IndexType, LogicalIndexType>(type);
}

RankedTensorType getTensorSchema(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor;
  if (auto view = dyn_cast<ViewType>(type))
    return dyn_cast<RankedTensorType>(view.getTensor());
  if (auto buffer = dyn_cast<BufferType>(type))
    return dyn_cast<RankedTensorType>(buffer.getTensor());
  return {};
}

Type getElementType(Type type) {
  if (auto tensor = getTensorSchema(type))
    return tensor.getElementType();
  return type;
}

std::optional<unsigned> getLogicalRank(Type type) {
  if (auto tensor = getTensorSchema(type))
    return tensor.getRank();
  if (auto domain = dyn_cast<DomainType>(type))
    return domain.getRank();
  if (auto region = dyn_cast<RegionType>(type))
    return region.getRank();
  return std::nullopt;
}

std::optional<uint64_t> getCoordinateSource(Type type) {
  if (auto domain = dyn_cast<DomainType>(type))
    return domain.getOriginId();
  if (auto region = dyn_cast<RegionType>(type))
    return region.getSourceId();
  return std::nullopt;
}

FailureOr<int64_t> getCount(Operation *operation, StringRef name) {
  auto attribute = operation->getAttrOfType<IntegerAttr>(name);
  if (!attribute || attribute.getInt() < 0) {
    operation->emitOpError() << "requires non-negative integer attribute '"
                             << name << "'";
    return failure();
  }
  return attribute.getInt();
}

std::optional<int64_t> getConstantInteger(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getName().getStringRef() != "intent.constant")
    return std::nullopt;
  if (auto integer = definition->getAttrOfType<IntegerAttr>("value"))
    return integer.getInt();
  return std::nullopt;
}

FailureOr<SmallVector<int64_t>> getIntegerArray(Operation *operation,
                                                StringRef name) {
  auto array = operation->getAttrOfType<ArrayAttr>(name);
  if (!array) {
    operation->emitOpError() << "requires array attribute '" << name << "'";
    return failure();
  }
  SmallVector<int64_t> values;
  values.reserve(array.size());
  for (Attribute attribute : array) {
    auto integer = dyn_cast<IntegerAttr>(attribute);
    if (!integer) {
      operation->emitOpError() << "attribute '" << name
                               << "' must contain integers";
      return failure();
    }
    values.push_back(integer.getInt());
  }
  return values;
}

FailureOr<SmallVector<std::pair<unsigned, unsigned>>>
getAxisPairs(Operation *operation, StringRef name, unsigned lhsRank,
             unsigned rhsRank) {
  auto array = operation->getAttrOfType<ArrayAttr>(name);
  if (!array) {
    operation->emitOpError() << "requires axis-pair attribute '" << name << "'";
    return failure();
  }
  SmallVector<std::pair<unsigned, unsigned>> pairs;
  for (Attribute attribute : array) {
    auto pair = dyn_cast<ArrayAttr>(attribute);
    if (!pair || pair.size() != 2 || !isa<IntegerAttr>(pair[0]) ||
        !isa<IntegerAttr>(pair[1])) {
      operation->emitOpError() << "attribute '" << name
                               << "' must contain integer axis pairs";
      return failure();
    }
    int64_t lhs = cast<IntegerAttr>(pair[0]).getInt();
    int64_t rhs = cast<IntegerAttr>(pair[1]).getInt();
    if (lhs < 0 || rhs < 0 || lhs >= lhsRank || rhs >= rhsRank) {
      operation->emitOpError() << "attribute '" << name
                               << "' contains an out-of-range axis";
      return failure();
    }
    pairs.emplace_back(static_cast<unsigned>(lhs), static_cast<unsigned>(rhs));
  }
  return pairs;
}

DenseI64ArrayAttr getDimensionIDs(RankedTensorType tensor) {
  auto encoding = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
  return encoding ? encoding.getDimensions() : DenseI64ArrayAttr();
}

std::optional<int64_t> getDimensionID(RankedTensorType tensor, unsigned axis) {
  if (!ShapedType::isDynamic(tensor.getDimSize(axis)))
    return std::nullopt;
  DenseI64ArrayAttr ids = getDimensionIDs(tensor);
  if (!ids || axis >= ids.size() || ids[axis] <= 0)
    return std::nullopt;
  return ids[axis];
}

bool extentValueMatchesAxis(Value extent, RankedTensorType tensor,
                            unsigned axis) {
  if (auto constant = getConstantInteger(extent))
    return !tensor.isDynamicDim(axis) && tensor.getDimSize(axis) == *constant;
  Operation *definition = extent.getDefiningOp();
  auto dimension = definition &&
                           definition->getName().getStringRef() == "intent.dim"
                       ? definition->getAttrOfType<IntegerAttr>("dimension")
                       : IntegerAttr();
  std::optional<int64_t> tensorDimension = getDimensionID(tensor, axis);
  return dimension && tensorDimension && dimension.getInt() == *tensorDimension;
}

bool sameDimension(RankedTensorType lhs, unsigned lhsAxis,
                   RankedTensorType rhs, unsigned rhsAxis) {
  int64_t left = lhs.getDimSize(lhsAxis);
  int64_t right = rhs.getDimSize(rhsAxis);
  if (!ShapedType::isDynamic(left) || !ShapedType::isDynamic(right))
    return left == right;
  DenseI64ArrayAttr lhsIDs = getDimensionIDs(lhs);
  DenseI64ArrayAttr rhsIDs = getDimensionIDs(rhs);
  return lhsIDs && rhsIDs && lhsIDs[lhsAxis] > 0 &&
         lhsIDs[lhsAxis] == rhsIDs[rhsAxis];
}

LogicalResult verifySliceAssembly(Operation *operation, Type sliceType,
                                  Type resultType, int64_t sliceDimensionID,
                                  RankedTensorType source,
                                  unsigned sourceAxis) {
  if (auto sliceTuple = dyn_cast<intent::TupleType>(sliceType)) {
    auto resultTuple = dyn_cast<intent::TupleType>(resultType);
    if (!resultTuple || sliceTuple.getComponentTypes().size() !=
                            resultTuple.getComponentTypes().size())
      return operation->emitOpError(
          "region-scan tuple output/result schemas disagree");
    for (auto [sliceComponent, resultComponent] :
         llvm::zip(sliceTuple.getComponentTypes(),
                   resultTuple.getComponentTypes()))
      if (failed(verifySliceAssembly(
              operation, cast<TypeAttr>(sliceComponent).getValue(),
              cast<TypeAttr>(resultComponent).getValue(), sliceDimensionID,
              source, sourceAxis)))
        return failure();
    return success();
  }
  if (auto sliceRecord = dyn_cast<RecordType>(sliceType)) {
    auto resultRecord = dyn_cast<RecordType>(resultType);
    if (!resultRecord || sliceRecord.getFieldNames() != resultRecord.getFieldNames() ||
        sliceRecord.getFieldTypes().size() != resultRecord.getFieldTypes().size())
      return operation->emitOpError(
          "region-scan record output/result schemas disagree");
    for (auto [sliceField, resultField] :
         llvm::zip(sliceRecord.getFieldTypes(), resultRecord.getFieldTypes()))
      if (failed(verifySliceAssembly(
              operation, cast<TypeAttr>(sliceField).getValue(),
              cast<TypeAttr>(resultField).getValue(), sliceDimensionID,
              source, sourceAxis)))
        return failure();
    return success();
  }
  auto slice = dyn_cast<RankedTensorType>(sliceType);
  auto result = dyn_cast<RankedTensorType>(resultType);
  if (!slice || !result || slice.getRank() != result.getRank() ||
      slice.getElementType() != result.getElementType())
    return operation->emitOpError(
        "region-scan output must be a tensor or structural tensor product");
  std::optional<unsigned> assembledAxis;
  for (unsigned axis = 0; axis < slice.getRank(); ++axis) {
    if (getDimensionID(slice, axis) == sliceDimensionID) {
      if (assembledAxis)
        return operation->emitOpError(
            "region-scan output slice extent must occur exactly once");
      assembledAxis = axis;
      if (!sameDimension(source, sourceAxis, result, axis))
        return operation->emitOpError(
            "region-scan result does not restore the full source extent");
    } else if (!sameDimension(slice, axis, result, axis)) {
      return operation->emitOpError(
          "region-scan output assembly changed a non-source dimension");
    }
  }
  return assembledAxis
             ? success()
             : operation->emitOpError(
                   "region-scan output slice does not carry the source slice extent");
}

bool sameTensorShape(RankedTensorType lhs, RankedTensorType rhs) {
  if (!lhs || !rhs || lhs.getRank() != rhs.getRank())
    return false;
  for (unsigned axis = 0; axis < lhs.getRank(); ++axis)
    if (!sameDimension(lhs, axis, rhs, axis))
      return false;
  return true;
}

bool compatibleElementType(Type lhs, Type rhs) {
  if (lhs == rhs)
    return true;
  return (isa<LogicalIndexType>(lhs) && isa<IndexType>(rhs)) ||
         (isa<IndexType>(lhs) && isa<LogicalIndexType>(rhs));
}

bool sameDataSchema(Type lhs, Type rhs, bool compareElements = true) {
  auto lhsTensor = dyn_cast<RankedTensorType>(lhs);
  auto rhsTensor = dyn_cast<RankedTensorType>(rhs);
  if (static_cast<bool>(lhsTensor) != static_cast<bool>(rhsTensor))
    return false;
  if (lhsTensor && rhsTensor)
    return sameTensorShape(lhsTensor, rhsTensor) &&
           (!compareElements || compatibleElementType(lhsTensor.getElementType(),
                                                      rhsTensor.getElementType()));
  return !compareElements || compatibleElementType(lhs, rhs);
}

bool isBooleanData(Type type) { return getElementType(type).isInteger(1); }

bool isNumericData(Type type) {
  Type element = getElementType(type);
  return isa<IntegerType, IndexType, LogicalIndexType, FloatType>(element) &&
         !element.isInteger(1);
}

FailureOr<SmallVector<unsigned>>
verifyShapeRelation(Operation *operation, RankedTensorType result,
                    bool allowInferred) {
  auto relation = operation->getAttrOfType<ShapeRelationAttr>("shape");
  ArrayAttr axes = relation ? relation.getAxes() : ArrayAttr();
  if (!axes || axes.size() != static_cast<size_t>(result.getRank())) {
    operation->emitOpError("shape relation must describe every result axis");
    return failure();
  }
  SmallVector<unsigned> dynamicOperands;
  unsigned inferredCount = 0;
  for (auto [axis, attribute] : llvm::enumerate(axes)) {
    auto entry = dyn_cast<ShapeExprAttr>(attribute);
    if (!entry) {
      operation->emitOpError("shape relation entry has an invalid kind");
      return failure();
    }
    if (entry.getKind() == 0) {
      if (result.getDimSize(axis) != entry.getPayload()) {
        operation->emitOpError(
            "static shape relation disagrees with result type");
        return failure();
      }
      continue;
    }
    auto ids = getDimensionIDs(result);
    if (entry.getDimension() <= 0 || !ids ||
        ids[axis] != entry.getDimension() ||
        !ShapedType::isDynamic(result.getDimSize(axis))) {
      operation->emitOpError(
          "dynamic/inferred shape relation must bind the result dimension identity");
      return failure();
    }
    if (entry.getKind() == 2) {
      if (!allowInferred || ++inferredCount > 1) {
        operation->emitOpError(
            "shape relation permits at most one inferred reshape extent");
        return failure();
      }
      continue;
    }
    int64_t operand = entry.getPayload();
    if (operand < 0 || operand >= operation->getNumOperands() ||
        !isIntegerLike(operation->getOperand(operand).getType())) {
      operation->emitOpError(
          "dynamic shape relation references an invalid extent operand");
      return failure();
    }
    dynamicOperands.push_back(static_cast<unsigned>(operand));
  }
  return dynamicOperands;
}

FailureOr<SmallVector<int64_t>> getExtentDimensions(Operation *operation,
                                                    unsigned rank) {
  FailureOr<SmallVector<int64_t>> dimensions =
      getIntegerArray(operation, "extent_dimensions");
  if (failed(dimensions) || dimensions->size() != rank) {
    operation->emitOpError(
        "logical iteration extent identities must match the source rank");
    return failure();
  }
  for (int64_t identity : *dimensions)
    if (identity < 0) {
      operation->emitOpError(
          "logical iteration extent identity must be non-negative");
      return failure();
    }
  return dimensions;
}

FailureOr<SmallVector<int64_t>> getIterationExtentDimensions(Value source) {
  Operation *definition = source.getDefiningOp();
  auto rank = getLogicalRank(source.getType());
  if (!definition || !rank)
    return failure();
  return getExtentDimensions(definition, *rank);
}

LogicalResult verifyShapeOperands(Operation *operation,
                                  ArrayRef<unsigned> dynamicOperands,
                                  unsigned firstShapeOperand) {
  if (operation->getNumOperands() !=
      firstShapeOperand + dynamicOperands.size())
    return operation->emitOpError(
        "shape extent operands do not match the canonical relation");
  for (auto [offset, position] : llvm::enumerate(dynamicOperands))
    if (position != firstShapeOperand + offset)
      return operation->emitOpError(
          "shape extent operands must be canonical and position ordered");
  return success();
}

SmallVector<Type> getOperandTypes(Operation *operation, unsigned offset = 0,
                                  std::optional<unsigned> count = std::nullopt) {
  SmallVector<Type> types;
  unsigned end = count ? offset + *count : operation->getNumOperands();
  for (unsigned index = offset; index < end; ++index)
    types.push_back(operation->getOperand(index).getType());
  return types;
}

SmallVector<Type> getResultTypes(Operation *operation, unsigned offset = 0,
                                 std::optional<unsigned> count = std::nullopt) {
  SmallVector<Type> types;
  unsigned end = count ? offset + *count : operation->getNumResults();
  for (unsigned index = offset; index < end; ++index)
    types.push_back(operation->getResult(index).getType());
  return types;
}

LogicalResult verifyOneBlock(Operation *owner, Region &region,
                             StringRef terminator) {
  if (!llvm::hasSingleElement(region) || region.front().empty())
    return owner->emitOpError("structured region must contain one non-empty block");
  if (region.front().back().getName().getStringRef() != terminator)
    return owner->emitOpError() << "structured region must end in "
                                << terminator;
  return success();
}

LogicalResult verifyYieldSchema(Operation *owner, Region &region,
                                TypeRange arguments, TypeRange yielded,
                                bool pure) {
  if (failed(verifyOneBlock(owner, region, "intent.yield")))
    return failure();
  Block &block = region.front();
  if (!llvm::equal(block.getArgumentTypes(), arguments))
    return owner->emitOpError("structured region argument schema is not canonical");
  Operation &terminator = block.back();
  if (!llvm::equal(terminator.getOperandTypes(), yielded))
    return owner->emitOpError("structured region yield schema is not canonical");
  if (!pure)
    return success();
  WalkResult walk = region.walk([&](Operation *nested) {
    StringRef name = nested->getName().getStringRef();
    if (name == "intent.yield")
      return WalkResult::advance();
    if (name == "intent.random_bits" || name == "intent.buffer" ||
        name == "intent.view_load" || name == "intent.view_store" ||
        name == "intent.buffer_load" || name == "intent.buffer_store" ||
        name == "intent.scatter_unique" ||
        name == "intent.scatter_reduce" || name.starts_with("intent.atomic_")) {
      nested->emitOpError("is not legal in a pure structured helper region");
      return WalkResult::interrupt();
    }
    // Recursive control is legal only when every nested operation has already
    // passed this same walk; the container itself carries recursive effects.
    if (name == "intent.if" || name == "intent.for" ||
        name == "intent.while" || name == "intent.parallel")
      return WalkResult::advance();
    if (!isMemoryEffectFree(nested)) {
      nested->emitOpError("has observable effects in a pure structured helper region");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return walk.wasInterrupted() ? failure() : success();
}

struct IndexRelationInfo {
  unsigned resultRank = 0;
  DenseI64ArrayAttr resultDimensions;
  llvm::DenseSet<unsigned> operandPositions;
};

FailureOr<IndexRelationInfo> verifyIndexRelation(Operation *operation) {
  auto relation = operation->getAttrOfType<IndexRelationAttr>("index");
  if (!relation)
    return operation->emitOpError("requires a typed index relation");
  ArrayAttr terms = relation.getTerms();
  if (operation->getNumOperands() == 0)
    return operation->emitOpError("index relation requires an addressable source");
  auto ranked = getTensorSchema(operation->getOperand(0).getType());
  if (!ranked || relation.getSourceRank() != ranked.getRank())
    return operation->emitOpError(
        "index relation source rank does not match its first operand");

  unsigned consumed = 0;
  unsigned basicResultRank = 0;
  unsigned advancedResultRank = 0;
  SmallVector<std::pair<RankedTensorType, unsigned>> advancedShape;
  IndexRelationInfo result;
  result.resultDimensions = relation.getResultDimensions();
  for (Attribute attribute : terms) {
    auto term = dyn_cast<IndexTermAttr>(attribute);
    if (!term)
      return operation->emitOpError("index relation term has an invalid schema");
    int64_t termKind = term.getKind();
    ArrayRef<int64_t> operands = term.getOperandPositions().asArrayRef();
    ArrayRef<int64_t> values = term.getStaticValues().asArrayRef();
    if (termKind != 1)
      ++consumed;
    for (int64_t position : operands) {
      if (position == -1)
        continue;
      if (position < 0 || position >= operation->getNumOperands())
        return operation->emitOpError(
            "index relation references an invalid operand position");
      Type type = operation->getOperand(position).getType();
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (termKind != 4 && !isIntegerLike(type) &&
          (!tensor || !isIntegerLike(tensor.getElementType())))
        return operation->emitOpError(
            "dynamic index relation operand must be integer/index typed");
      result.operandPositions.insert(static_cast<unsigned>(position));
    }
    if ((termKind == 0 || termKind == 1) &&
        (!operands.empty() || !values.empty()))
      return operation->emitOpError(
          "full-slice/new-axis terms cannot carry payload");
    if (termKind == 0 || termKind == 1 || termKind == 5)
      ++basicResultRank;
    if (termKind == 2 &&
        (!operands.empty() || values.size() != 1 ||
         values[0] == std::numeric_limits<int64_t>::min()))
      return operation->emitOpError(
          "static-index term requires one integer literal");
    if ((termKind == 3 || termKind == 4) &&
        (operands.size() != 1 || operands[0] < 0 || !values.empty()))
      return operation->emitOpError(
          "value/region index term requires one dynamic operand");
    if (termKind == 4) {
      int64_t position = operands[0];
      if (!isa<DomainType, RegionType>(
              operation->getOperand(position).getType()))
        return operation->emitOpError(
            "region index term must reference a domain or subregion operand");
      auto rank = getLogicalRank(operation->getOperand(position).getType());
      if (!rank)
        return operation->emitOpError(
            "region index term has no logical rank");
      basicResultRank += *rank;
    }
    if (termKind == 3) {
      int64_t position = operands[0];
      if (auto tensor = dyn_cast<RankedTensorType>(
              operation->getOperand(position).getType())) {
        if (advancedShape.empty()) {
          advancedShape.reserve(tensor.getRank());
          for (unsigned axis = 0; axis < tensor.getRank(); ++axis)
            advancedShape.emplace_back(tensor, axis);
        } else {
          unsigned mergedRank = std::max<unsigned>(advancedShape.size(),
                                                   tensor.getRank());
          SmallVector<std::pair<RankedTensorType, unsigned>> merged(mergedRank);
          for (unsigned trailing = 0; trailing < mergedRank; ++trailing) {
            bool hasLeft = trailing < advancedShape.size();
            bool hasRight = trailing < static_cast<unsigned>(tensor.getRank());
            auto left = hasLeft
                            ? advancedShape[advancedShape.size() - 1 - trailing]
                            : std::pair<RankedTensorType, unsigned>();
            unsigned rightAxis =
                hasRight ? tensor.getRank() - 1 - trailing : 0;
            bool leftOne = hasLeft && left.first.getDimSize(left.second) == 1;
            bool rightOne = hasRight && tensor.getDimSize(rightAxis) == 1;
            if (hasLeft && hasRight && !leftOne && !rightOne &&
                !sameDimension(left.first, left.second, tensor, rightAxis))
              return operation->emitOpError(
                  "tensor index operands have incompatible broadcast extents");
            if (!hasLeft)
              merged[mergedRank - 1 - trailing] =
                  std::make_pair(tensor, rightAxis);
            else if (!hasRight)
              merged[mergedRank - 1 - trailing] = left;
            else
              merged[mergedRank - 1 - trailing] =
                  leftOne ? std::make_pair(tensor, rightAxis) : left;
          }
          advancedShape = std::move(merged);
        }
        advancedResultRank = advancedShape.size();
      }
    }
    if (termKind == 5) {
      if (operands.size() != 3 || values.size() != 3)
        return operation->emitOpError(
            "slice term requires start/stop/step payload slots");
      constexpr int64_t absent = std::numeric_limits<int64_t>::min();
      for (auto [operand, value] : llvm::zip(operands, values))
        if (operand >= 0 && value != absent)
          return operation->emitOpError(
              "slice payload cannot be dynamic and static simultaneously");
      if (values[2] == 0)
        return operation->emitOpError("slice step cannot be zero");
    }
  }
  if (consumed != relation.getSourceRank())
    return operation->emitOpError(
        "index relation must consume each source axis exactly once");
  result.resultRank = basicResultRank + advancedResultRank;
  if (result.resultRank != relation.getResultRank())
    return operation->emitOpError(
        "index relation result-rank fact disagrees with its typed terms");
  return result;
}

std::optional<unsigned> getDataRank(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getRank();
  if (isa<IntegerType, IndexType, FloatType, LogicalIndexType>(type))
    return 0;
  return std::nullopt;
}

LogicalResult verifyIndexedData(Operation *operation, Type type,
                                Type elementType,
                                const IndexRelationInfo &relation,
                                StringRef subject) {
  auto actualRank = getDataRank(type);
  if (!actualRank || *actualRank != relation.resultRank ||
      !compatibleElementType(getElementType(type), elementType))
    return operation->emitOpError()
           << subject << " does not match the indexed element/rank schema";
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return success();
  DenseI64ArrayAttr actualDimensions = getDimensionIDs(tensor);
  for (unsigned axis = 0; axis < relation.resultRank; ++axis) {
    int64_t expected = relation.resultDimensions[axis];
    if (expected <= 0 || !actualDimensions ||
        actualDimensions[axis] != expected)
      return operation->emitOpError()
             << subject << " lost an index-relation result dimension identity";
  }
  return success();
}

LogicalResult verifyOperandPartition(
    Operation *operation, const IndexRelationInfo &relation,
    ArrayRef<unsigned> reserved) {
  llvm::DenseSet<unsigned> reservedSet(reserved.begin(), reserved.end());
  for (unsigned position = 1; position < operation->getNumOperands(); ++position) {
    bool indexed = relation.operandPositions.contains(position);
    bool special = reservedSet.contains(position);
    if (indexed == special)
      return operation->emitOpError(
          "index/value/validity operand partition is not canonical");
  }
  return success();
}

LogicalResult verifyProductOperation(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  if (name == "intent.make_tuple") {
    auto result = dyn_cast<intent::TupleType>(operation->getResult(0).getType());
    if (!result ||
        result.getComponentTypes().size() != operation->getNumOperands())
      return operation->emitOpError("tuple operands do not match result schema");
    for (auto [operand, typeAttribute] :
         llvm::zip(operation->getOperands(), result.getComponentTypes()))
      if (operand.getType() != cast<TypeAttr>(typeAttribute).getValue())
        return operation->emitOpError("tuple component operand has the wrong type");
    return success();
  }
  if (name == "intent.make_record") {
    auto result = dyn_cast<RecordType>(operation->getResult(0).getType());
    if (!result || result.getFieldTypes().size() != operation->getNumOperands())
      return operation->emitOpError("record operands do not match result schema");
    for (auto [operand, typeAttribute] :
         llvm::zip(operation->getOperands(), result.getFieldTypes()))
      if (operand.getType() != cast<TypeAttr>(typeAttribute).getValue())
        return operation->emitOpError("record field operand has the wrong type");
    return success();
  }
  auto field = operation->getAttrOfType<IntegerAttr>("field");
  if (!field || field.getInt() < 0)
    return operation->emitOpError("product projection requires a valid component");
  if (auto tuple = dyn_cast<intent::TupleType>(operation->getOperand(0).getType())) {
    if (static_cast<size_t>(field.getInt()) >= tuple.getComponentTypes().size())
      return operation->emitOpError("tuple projection references an invalid component");
    if (operation->getResult(0).getType() !=
        cast<TypeAttr>(tuple.getComponentTypes()[field.getInt()]).getValue())
      return operation->emitOpError("tuple projection result type is incorrect");
    return success();
  }
  auto record = dyn_cast<RecordType>(operation->getOperand(0).getType());
  if (!record ||
      static_cast<size_t>(field.getInt()) >= record.getFieldTypes().size())
    return operation->emitOpError("record projection references an invalid field");
  if (operation->getResult(0).getType() !=
      cast<TypeAttr>(record.getFieldTypes()[field.getInt()]).getValue())
    return operation->emitOpError("record projection result type is incorrect");
  return success();
}

LogicalResult verifyDataOperation(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  if (name == "intent.reshape" || name == "intent.broadcast" ||
      name == "intent.full") {
    auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
    if (!result || operation->getNumOperands() == 0)
      return operation->emitOpError(
          "shape operation requires a source/fill and ranked tensor result");
    FailureOr<SmallVector<unsigned>> dynamicOperands =
        verifyShapeRelation(operation, result, name == "intent.reshape");
    if (failed(dynamicOperands) ||
        failed(verifyShapeOperands(operation, *dynamicOperands, 1)))
      return failure();
    Type sourceType = operation->getOperand(0).getType();
    if (name == "intent.full") {
      if (!compatibleElementType(sourceType, result.getElementType()) ||
          !isa<IntegerType, IndexType, FloatType, LogicalIndexType>(sourceType))
        return operation->emitOpError(
            "full fill/result element schema is invalid");
      return success();
    }
    auto source = dyn_cast<RankedTensorType>(sourceType);
    if (name == "intent.reshape") {
      if (!source || source.getElementType() != result.getElementType())
        return operation->emitOpError(
            "reshape must preserve ranked tensor element type");
      if (source.hasStaticShape() && result.hasStaticShape() &&
          source.getNumElements() != result.getNumElements())
        return operation->emitOpError(
            "reshape must preserve logical element count");
      return success();
    }
    if (!compatibleElementType(getElementType(sourceType),
                               result.getElementType()))
      return operation->emitOpError("broadcast must preserve element type");
    if (!source)
      return isa<IntegerType, IndexType, FloatType, LogicalIndexType>(sourceType)
                 ? success()
                 : operation->emitOpError(
                       "broadcast source must be scalar or ranked tensor");
    if (source.getRank() > result.getRank())
      return operation->emitOpError("broadcast cannot reduce rank");
    unsigned offset = result.getRank() - source.getRank();
    for (unsigned axis = 0; axis < source.getRank(); ++axis)
      if (source.getDimSize(axis) != 1 &&
          !sameDimension(source, axis, result, axis + offset))
        return operation->emitOpError(
            "broadcast axis is neither size one nor source-identical");
    return success();
  }

  Type result = operation->getResult(0).getType();
  if (name == "intent.unary") {
    Type input = operation->getOperand(0).getType();
    auto kind = operation->getAttrOfType<UnaryOperatorAttr>("operator_kind");
    if (!kind || !sameDataSchema(input, result))
      return operation->emitOpError("unary operator/schema is invalid");
    if (kind.getValue() == UnaryOperator::Not)
      return isBooleanData(input)
                 ? success()
                 : operation->emitOpError("logical not requires bool data");
    if (kind.getValue() == UnaryOperator::Negate ||
        kind.getValue() == UnaryOperator::Abs)
      return isNumericData(input)
                 ? success()
                 : operation->emitOpError("numeric unary requires numeric data");
    return isa<FloatType>(getElementType(input))
               ? success()
               : operation->emitOpError(
                     "transcendental unary operation requires floating data");
  }

  if (name == "intent.binary") {
    Type lhs = operation->getOperand(0).getType();
    Type rhs = operation->getOperand(1).getType();
    auto kind = operation->getAttrOfType<BinaryOperatorAttr>("operator_kind");
    if (!kind || !sameDataSchema(lhs, rhs, false) ||
        !sameDataSchema(lhs, result, false))
      return operation->emitOpError("binary operator/schema is invalid");
    BinaryOperator value = kind.getValue();
    if (value == BinaryOperator::LogicalAnd ||
        value == BinaryOperator::LogicalOr)
      return isBooleanData(lhs) && isBooleanData(rhs) && isBooleanData(result)
                 ? success()
                 : operation->emitOpError(
                       "logical binary operation requires bool data");
    if (value == BinaryOperator::BitwiseAnd ||
        value == BinaryOperator::BitwiseOr ||
        value == BinaryOperator::BitwiseXor ||
        value == BinaryOperator::LeftShift ||
        value == BinaryOperator::RightShift) {
      Type lhsElement = getElementType(lhs);
      Type rhsElement = getElementType(rhs);
      Type resultElement = getElementType(result);
      return isa<IntegerType, IndexType, LogicalIndexType>(lhsElement) &&
                     isa<IntegerType, IndexType, LogicalIndexType>(rhsElement) &&
                     isa<IntegerType, IndexType, LogicalIndexType>(resultElement)
                 ? success()
                 : operation->emitOpError(
                       "bitwise binary operation requires integer/index data");
    }
    if (!isNumericData(lhs) || !isNumericData(rhs) || !isNumericData(result))
      return operation->emitOpError(
          "arithmetic binary operation requires numeric data");
    if (value == BinaryOperator::TrueDivide &&
        !isa<FloatType>(getElementType(lhs)))
      return operation->emitOpError(
          "true division requires explicitly floating operands");
    if ((value == BinaryOperator::FloorDivide ||
         value == BinaryOperator::Remainder) &&
        !isa<IntegerType, IndexType, LogicalIndexType>(getElementType(lhs)))
      return operation->emitOpError(
          "floor division/remainder require integer/index operands");
    if ((value == BinaryOperator::MaximumNum ||
         value == BinaryOperator::MinimumNum) &&
        !isa<FloatType>(getElementType(lhs)))
      return operation->emitOpError(
          "NaN-selecting min/max require floating operands");
    return success();
  }

  if (name == "intent.compare") {
    Type lhs = operation->getOperand(0).getType();
    Type rhs = operation->getOperand(1).getType();
    auto predicate =
        operation->getAttrOfType<ComparePredicateAttr>("predicate");
    if (!predicate || !sameDataSchema(lhs, rhs, false) ||
        !sameDataSchema(lhs, result, false) || !isBooleanData(result) ||
        (!sameDataSchema(lhs, rhs) &&
         !(isNumericData(lhs) && isNumericData(rhs))))
      return operation->emitOpError("comparison schema/predicate is invalid");
    return success();
  }

  if (name == "intent.select" || name == "intent.mask") {
    unsigned conditionIndex = name == "intent.select" ? 0 : 1;
    unsigned lhsIndex = name == "intent.select" ? 1 : 0;
    unsigned rhsIndex = 2;
    Type condition = operation->getOperand(conditionIndex).getType();
    Type lhs = operation->getOperand(lhsIndex).getType();
    Type rhs = operation->getOperand(rhsIndex).getType();
    if (!isBooleanData(condition) || !sameDataSchema(condition, lhs, false) ||
        !sameDataSchema(lhs, rhs) || !sameDataSchema(lhs, result))
      return operation->emitOpError(
          "select/mask operands must already have one canonical broadcast schema");
    return success();
  }

  if (name == "intent.cast" || name == "intent.bitcast") {
    Type input = operation->getOperand(0).getType();
    if (!sameDataSchema(input, result, false))
      return operation->emitOpError("cast must preserve logical shape");
    if (name == "intent.cast") {
      auto rounding = operation->getAttrOfType<IntegerAttr>("rounding");
      if (rounding && rounding.getInt() != 0)
        return operation->emitOpError("cast rounding is outside the language enum");
      return success();
    }
    auto width = [](Type type) -> std::optional<unsigned> {
      if (auto integer = dyn_cast<IntegerType>(type))
        return integer.getWidth();
      if (auto floating = dyn_cast<FloatType>(type))
        return floating.getWidth();
      return std::nullopt;
    };
    Type sourceElement = getElementType(input);
    Type resultElement = getElementType(result);
    auto sourceWidth = width(sourceElement);
    auto resultWidth = width(resultElement);
    if (!sourceWidth || !resultWidth || sourceElement.isInteger(1) ||
        resultElement.isInteger(1) || *sourceWidth != *resultWidth)
      return operation->emitOpError(
          "bitcast requires equal-width non-bool elements");
    return success();
  }
  return success();
}

LogicalResult verifyControlOperation(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  if (name == "intent.parallel" || name == "intent.for") {
    bool parallel = name == "intent.parallel";
    if (operation->getNumOperands() < 1 || operation->getNumRegions() != 1)
      return operation->emitOpError("iteration requires a source and one body");
    Type sourceType = operation->getOperand(0).getType();
    auto rank = getLogicalRank(sourceType);
    auto source = getCoordinateSource(sourceType);
    if (!rank || !source || failed(verifyOneBlock(operation, operation->getRegion(0),
                                                   "intent.yield")))
      return failure();
    Block &body = operation->getRegion(0).front();
    unsigned carryCount = operation->getNumOperands() - 1;
    if (parallel && (carryCount != 0 || operation->getNumResults() != 0))
      return operation->emitOpError("parallel iteration cannot carry SSA values");
    if (!parallel && operation->getNumResults() != carryCount)
      return operation->emitOpError("for-loop carry/result arity is inconsistent");
    if (body.getNumArguments() != *rank + carryCount)
      return operation->emitOpError("iteration body argument schema is inconsistent");
    for (unsigned axis = 0; axis < *rank; ++axis) {
      auto index = dyn_cast<LogicalIndexType>(body.getArgument(axis).getType());
      if (!index || index.getSourceId() != *source || index.getAxis() != axis)
        return operation->emitOpError(
            "iteration coordinate lost its source/axis provenance");
    }
    for (unsigned index = 0; index < carryCount; ++index) {
      Type initial = operation->getOperand(index + 1).getType();
      if (body.getArgument(*rank + index).getType() != initial ||
          operation->getResult(index).getType() != initial)
        return operation->emitOpError("for-loop carry type is not stable");
    }
    Operation &yield = body.back();
    if (yield.getNumOperands() != carryCount ||
        !llvm::equal(yield.getOperandTypes(), operation->getResultTypes()))
      return operation->emitOpError("iteration yield schema is inconsistent");
    return success();
  }
  if (name == "intent.if") {
    if (operation->getNumRegions() != 2 ||
        !operation->getOperand(0).getType().isInteger(1))
      return operation->emitOpError("if requires i1 condition and two branches");
    for (Region &region : operation->getRegions()) {
      if (failed(verifyOneBlock(operation, region, "intent.yield")))
        return failure();
      Block &block = region.front();
      if (block.getNumArguments() != 0 ||
          !llvm::equal(block.back().getOperandTypes(), operation->getResultTypes()))
        return operation->emitOpError("if branch yield schema is inconsistent");
    }
    return success();
  }
  if (name == "intent.while") {
    if (operation->getNumRegions() != 2 ||
        operation->getNumOperands() != operation->getNumResults())
      return operation->emitOpError("while carry/result schema is inconsistent");
    if (!llvm::equal(operation->getOperandTypes(), operation->getResultTypes()))
      return operation->emitOpError("while carry types are not stable");
    Region &before = operation->getRegion(0);
    Region &after = operation->getRegion(1);
    if (failed(verifyOneBlock(operation, before, "intent.condition")) ||
        failed(verifyOneBlock(operation, after, "intent.yield")))
      return failure();
    if (!llvm::equal(before.front().getArgumentTypes(), operation->getOperandTypes()) ||
        !llvm::equal(after.front().getArgumentTypes(), operation->getOperandTypes()))
      return operation->emitOpError("while region argument schema is inconsistent");
    Operation &condition = before.front().back();
    SmallVector<Type> conditionTypes;
    for (Value value : condition.getOperands().drop_front())
      conditionTypes.push_back(value.getType());
    if (condition.getNumOperands() != operation->getNumOperands() + 1 ||
        !condition.getOperand(0).getType().isInteger(1) ||
        !llvm::equal(conditionTypes, operation->getOperandTypes()))
      return operation->emitOpError("while condition does not forward its carries");
    if (!llvm::equal(after.front().back().getOperandTypes(),
                     operation->getResultTypes()))
      return operation->emitOpError("while yield schema is inconsistent");
    return success();
  }
  return success();
}

LogicalResult verifyReduceOrScan(Operation *operation, bool scan) {
  FailureOr<int64_t> sourceCount = getCount(operation, "source_count");
  FailureOr<int64_t> identityCount = getCount(operation, "identity_count");
  FailureOr<int64_t> captureCount = getCount(operation, "capture_count");
  if (failed(sourceCount) || failed(identityCount) || failed(captureCount) ||
      *sourceCount <= 0 || *sourceCount != *identityCount ||
      operation->getNumOperands() !=
          static_cast<unsigned>(*sourceCount + *identityCount + *captureCount) ||
      operation->getNumResults() != static_cast<unsigned>(*identityCount) ||
      operation->getNumRegions() != 1)
    return operation->emitOpError("reduce/scan component partition is inconsistent");

  SmallVector<int64_t> axes;
  if (scan) {
    auto axis = operation->getAttrOfType<IntegerAttr>("axis");
    if (!axis || axis.getInt() < 0)
      return operation->emitOpError("scan requires a non-negative axis");
    axes.push_back(axis.getInt());
  } else {
    FailureOr<SmallVector<int64_t>> values = getIntegerArray(operation, "axes");
    if (failed(values) || values->empty())
      return operation->emitOpError("reduce requires one or more axes");
    axes = *values;
  }

  SmallVector<Type> accumulatorTypes;
  llvm::DenseSet<int64_t> uniqueAxes;
  for (int64_t axis : axes)
    if (axis < 0 || !uniqueAxes.insert(axis).second)
      return operation->emitOpError("reduce/scan axes must be unique and non-negative");
  for (int64_t index = 0; index < *sourceCount; ++index) {
    auto source = dyn_cast<RankedTensorType>(operation->getOperand(index).getType());
    Type identity = operation->getOperand(*sourceCount + index).getType();
    if (!source || getElementType(identity) != source.getElementType())
      return operation->emitOpError(
          "reduce/scan source and identity component types disagree");
    for (int64_t axis : axes)
      if (axis >= source.getRank())
        return operation->emitOpError("reduce/scan axis is outside source rank");
    Type result = operation->getResult(index).getType();
    if (scan) {
      auto identityTensor = dyn_cast<RankedTensorType>(identity);
      bool scalarIdentity = identity == source.getElementType();
      bool sliceIdentity = identityTensor &&
                           identityTensor.getRank() + 1 == source.getRank() &&
                           identityTensor.getElementType() == source.getElementType();
      if (sliceIdentity) {
        unsigned identityAxis = 0;
        for (unsigned sourceAxis = 0; sourceAxis < source.getRank(); ++sourceAxis) {
          if (sourceAxis == static_cast<unsigned>(axes.front()))
            continue;
          if (!sameDimension(source, sourceAxis, identityTensor, identityAxis++))
            sliceIdentity = false;
        }
      }
      if ((!scalarIdentity && !sliceIdentity) || result != source)
        return operation->emitOpError("scan result must preserve source type");
    } else {
      auto resultTensor = dyn_cast<RankedTensorType>(result);
      unsigned resultRank = source.getRank() - axes.size();
      bool scalarIdentity = identity == source.getElementType();
      bool resultIdentity = identity == result;
      if ((resultRank == 0 && (!scalarIdentity || result != source.getElementType())) ||
          (resultRank != 0 &&
           (!resultTensor || resultTensor.getRank() != resultRank ||
            resultTensor.getElementType() != source.getElementType() ||
            (!scalarIdentity && !resultIdentity))))
        return operation->emitOpError("reduce result type does not match removed axes");
      if (resultTensor) {
        unsigned resultAxis = 0;
        for (unsigned sourceAxis = 0; sourceAxis < source.getRank(); ++sourceAxis) {
          if (uniqueAxes.contains(sourceAxis))
            continue;
          if (!sameDimension(source, sourceAxis, resultTensor, resultAxis++))
            return operation->emitOpError(
                "reduce result dimension lost source identity");
        }
      }
    }
    accumulatorTypes.push_back(identity);
  }
  SmallVector<Type> arguments(accumulatorTypes);
  arguments.append(accumulatorTypes);
  SmallVector<Type> captures = getOperandTypes(
      operation, static_cast<unsigned>(*sourceCount + *identityCount));
  arguments.append(captures);
  return verifyYieldSchema(operation, operation->getRegion(0), arguments,
                           accumulatorTypes, true);
}

LogicalResult verifyRegionFold(Operation *operation) {
  FailureOr<int64_t> sourceCount = getCount(operation, "source_count");
  FailureOr<int64_t> identityCount = getCount(operation, "identity_count");
  FailureOr<int64_t> captureCount = getCount(operation, "capture_count");
  auto axis = operation->getAttrOfType<IntegerAttr>("axis");
  if (failed(sourceCount) || failed(identityCount) || failed(captureCount) ||
      *sourceCount <= 0 || *identityCount <= 0 || !axis || axis.getInt() < 0 ||
      operation->getNumOperands() !=
          static_cast<unsigned>(*sourceCount + *identityCount + *captureCount) ||
      operation->getNumResults() != static_cast<unsigned>(*identityCount) ||
      operation->getNumRegions() != 2)
    return operation->emitOpError("region-fold component partition is inconsistent");
  if (failed(verifyOneBlock(operation, operation->getRegion(0), "intent.yield")) ||
      failed(verifyOneBlock(operation, operation->getRegion(1), "intent.yield")))
    return failure();
  SmallVector<Type> sliceTypes;
  RankedTensorType first;
  for (int64_t index = 0; index < *sourceCount; ++index) {
    auto source = dyn_cast<RankedTensorType>(operation->getOperand(index).getType());
    if (!source || axis.getInt() >= source.getRank())
      return operation->emitOpError("region-fold source axis is invalid");
    if (!first)
      first = source;
    else if (!sameDimension(first, axis.getInt(), source, axis.getInt()))
      return operation->emitOpError(
          "region-fold sources do not share one logical source extent");
    sliceTypes.push_back(operation->getRegion(0).front().getArgument(index).getType());
    auto slice = dyn_cast<RankedTensorType>(sliceTypes.back());
    if (!slice || slice.getRank() != source.getRank() ||
        slice.getElementType() != source.getElementType())
      return operation->emitOpError("region-fold summarize slice type is invalid");
    for (unsigned dimension = 0; dimension < source.getRank(); ++dimension)
      if (dimension != static_cast<unsigned>(axis.getInt()) &&
          !sameDimension(source, dimension, slice, dimension))
        return operation->emitOpError(
            "region-fold summarize slice changed a non-source dimension");
    if (sliceTypes.size() > 1) {
      auto firstSlice = cast<RankedTensorType>(sliceTypes.front());
      if (getDimensionID(firstSlice, axis.getInt()) !=
          getDimensionID(slice, axis.getInt()))
        return operation->emitOpError(
            "region-fold source components are not sliced in lockstep");
    }
  }
  SmallVector<Type> identities = getOperandTypes(
      operation, static_cast<unsigned>(*sourceCount),
      static_cast<unsigned>(*identityCount));
  if (!llvm::equal(identities, operation->getResultTypes()))
    return operation->emitOpError("region-fold identity/result schema disagrees");
  SmallVector<Type> captures = getOperandTypes(
      operation, static_cast<unsigned>(*sourceCount + *identityCount));
  SmallVector<Type> summarizeArguments(sliceTypes);
  summarizeArguments.append(captures);
  if (failed(verifyYieldSchema(operation, operation->getRegion(0),
                               summarizeArguments, identities, true)))
    return failure();
  SmallVector<Type> combineArguments(identities);
  combineArguments.append(identities);
  return verifyYieldSchema(operation, operation->getRegion(1), combineArguments,
                           identities, true);
}

LogicalResult verifyRegionScan(Operation *operation) {
  FailureOr<int64_t> sourceCount = getCount(operation, "source_count");
  FailureOr<int64_t> identityCount = getCount(operation, "identity_count");
  FailureOr<int64_t> stateCount = getCount(operation, "state_count");
  FailureOr<int64_t> captureCount = getCount(operation, "capture_count");
  FailureOr<int64_t> outputCount = getCount(operation, "output_count");
  auto axis = operation->getAttrOfType<IntegerAttr>("axis");
  if (failed(sourceCount) || failed(identityCount) || failed(stateCount) ||
      failed(captureCount) || failed(outputCount) || *sourceCount <= 0 ||
      *identityCount <= 0 || *stateCount <= 0 || *outputCount <= 0 || !axis ||
      axis.getInt() < 0 ||
      operation->getNumOperands() != static_cast<unsigned>(
          *sourceCount + *identityCount + *stateCount + *captureCount) ||
      operation->getNumResults() !=
          static_cast<unsigned>(*outputCount + *stateCount) ||
      operation->getNumRegions() != 4)
    return operation->emitOpError("region-scan component partition is inconsistent");
  for (Region &region : operation->getRegions())
    if (failed(verifyOneBlock(operation, region, "intent.yield")))
      return failure();

  SmallVector<Type> slices;
  RankedTensorType first;
  for (int64_t index = 0; index < *sourceCount; ++index) {
    auto source = dyn_cast<RankedTensorType>(operation->getOperand(index).getType());
    if (!source || axis.getInt() >= source.getRank())
      return operation->emitOpError("region-scan source axis is invalid");
    if (!first)
      first = source;
    else if (!sameDimension(first, axis.getInt(), source, axis.getInt()))
      return operation->emitOpError(
          "region-scan sources do not share one logical source extent");
    Type sliceType = operation->getRegion(0).front().getArgument(index).getType();
    auto slice = dyn_cast<RankedTensorType>(sliceType);
    if (!slice || slice.getRank() != source.getRank() ||
        slice.getElementType() != source.getElementType())
      return operation->emitOpError("region-scan summarize slice type is invalid");
    for (unsigned dimension = 0; dimension < source.getRank(); ++dimension)
      if (dimension != static_cast<unsigned>(axis.getInt()) &&
          !sameDimension(source, dimension, slice, dimension))
        return operation->emitOpError(
            "region-scan summarize slice changed a non-source dimension");
    if (!slices.empty()) {
      auto firstSlice = cast<RankedTensorType>(slices.front());
      if (getDimensionID(firstSlice, axis.getInt()) !=
          getDimensionID(slice, axis.getInt()))
        return operation->emitOpError(
            "region-scan source components are not sliced in lockstep");
    }
    slices.push_back(sliceType);
  }
  SmallVector<Type> transitions = getOperandTypes(
      operation, static_cast<unsigned>(*sourceCount),
      static_cast<unsigned>(*identityCount));
  SmallVector<Type> states = getOperandTypes(
      operation, static_cast<unsigned>(*sourceCount + *identityCount),
      static_cast<unsigned>(*stateCount));
  SmallVector<Type> captures = getOperandTypes(
      operation,
      static_cast<unsigned>(*sourceCount + *identityCount + *stateCount));
  SmallVector<Type> finalStates = getResultTypes(
      operation, static_cast<unsigned>(*outputCount),
      static_cast<unsigned>(*stateCount));
  if (!llvm::equal(states, finalStates))
    return operation->emitOpError("region-scan final-state schema is inconsistent");
  SmallVector<Type> summarizeArguments(slices);
  summarizeArguments.append(captures);
  if (failed(verifyYieldSchema(operation, operation->getRegion(0),
                               summarizeArguments, transitions, true)))
    return failure();
  SmallVector<Type> combineArguments(transitions);
  combineArguments.append(transitions);
  if (failed(verifyYieldSchema(operation, operation->getRegion(1),
                               combineArguments, transitions, true)))
    return failure();
  SmallVector<Type> applyArguments(transitions);
  applyArguments.append(states);
  if (failed(verifyYieldSchema(operation, operation->getRegion(2), applyArguments,
                               states, true)))
    return failure();
  SmallVector<Type> emitArguments(slices);
  emitArguments.append(states);
  emitArguments.append(captures);
  Region &emit = operation->getRegion(3);
  if (!llvm::equal(emit.front().getArgumentTypes(), emitArguments) ||
      emit.front().back().getNumOperands() != *outputCount)
    return operation->emitOpError("region-scan emit schema is inconsistent");
  if (failed(verifyYieldSchema(operation, emit, emitArguments,
                               emit.front().back().getOperandTypes(), true)))
    return failure();
  auto firstSlice = cast<RankedTensorType>(slices.front());
  std::optional<int64_t> sliceDimensionID =
      getDimensionID(firstSlice, axis.getInt());
  if (!sliceDimensionID)
    return operation->emitOpError(
        "region-scan slice axis requires a stable dynamic extent identity");
  for (unsigned index = 0; index < static_cast<unsigned>(*outputCount); ++index)
    if (failed(verifySliceAssembly(
            operation, emit.front().back().getOperand(index).getType(),
            operation->getResult(index).getType(), *sliceDimensionID, first,
            axis.getInt())))
      return failure();
  return success();
}

LogicalResult verifyContract(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  auto lhs = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  unsigned rhsIndex = name == "intent.scaled_contract"
                          ? 2
                          : name == "intent.sparse_contract"
                                ? 2
                                : 1;
  auto rhs = dyn_cast<RankedTensorType>(operation->getOperand(rhsIndex).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!lhs || !rhs || !result)
    return operation->emitOpError("contract family requires ranked tensor data");
  FailureOr<SmallVector<std::pair<unsigned, unsigned>>> reductions =
      getAxisPairs(operation, "reduce", lhs.getRank(), rhs.getRank());
  FailureOr<SmallVector<std::pair<unsigned, unsigned>>> batches =
      getAxisPairs(operation, "batch", lhs.getRank(), rhs.getRank());
  if (failed(reductions) || failed(batches) || reductions->empty())
    return operation->emitOpError("contract requires at least one reduction pair");
  if (name == "intent.scaled_contract") {
    auto lhsScale = dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
    auto rhsScale = dyn_cast<RankedTensorType>(operation->getOperand(3).getType());
    auto lhsGroup = operation->getAttrOfType<IntegerAttr>("lhs_group_size");
    auto rhsGroup = operation->getAttrOfType<IntegerAttr>("rhs_group_size");
    auto lhsFormat = operation->getAttrOfType<ScaledFormatAttr>("lhs_format");
    auto rhsFormat = operation->getAttrOfType<ScaledFormatAttr>("rhs_format");
    bool fixedAxes =
        reductions->size() == 2 && batches->empty() &&
        (*reductions)[0] == std::make_pair(1u, 0u) &&
        (*reductions)[1] == std::make_pair(2u, 1u);
    if (!lhsScale || !rhsScale || lhs.getRank() != 3 ||
        lhsScale.getRank() != 2 || rhs.getRank() != 3 ||
        rhsScale.getRank() != 2 || !fixedAxes || !lhsGroup || !rhsGroup ||
        lhsGroup.getInt() <= 0 || lhsGroup != rhsGroup || !lhsFormat ||
        !rhsFormat)
      return operation->emitOpError(
          "scaled contract requires the closed [M,G,C]/[M,G] x [G,C,N]/[N,G] schema");
    auto carrierExtent = [](ScaledFormat format,
                            int64_t group) -> std::optional<int64_t> {
      int64_t packing = format == ScaledFormat::E2M1 ? 2 : 1;
      if (group % packing != 0)
        return std::nullopt;
      return group / packing;
    };
    std::optional<int64_t> lhsCarrier =
        carrierExtent(lhsFormat.getValue(), lhsGroup.getInt());
    std::optional<int64_t> rhsCarrier =
        carrierExtent(rhsFormat.getValue(), rhsGroup.getInt());
    if (!lhsCarrier || !rhsCarrier || lhs.getDimSize(2) != *lhsCarrier ||
        rhs.getDimSize(1) != *rhsCarrier ||
        !sameDimension(lhs, 0, lhsScale, 0) ||
        !sameDimension(lhs, 1, lhsScale, 1) ||
        !sameDimension(lhs, 1, rhs, 0) ||
        !sameDimension(lhs, 1, rhsScale, 1) ||
        !sameDimension(rhs, 2, rhsScale, 0) || result.getRank() != 2 ||
        !sameDimension(lhs, 0, result, 0) ||
        !sameDimension(rhs, 2, result, 1) || !isNumericData(lhsScale) ||
        !isNumericData(rhsScale) || !isNumericData(result))
      return operation->emitOpError(
          "scaled contract operands violate the closed scale-axis relation");
    return success();
  }
  llvm::DenseSet<unsigned> lhsUsed;
  llvm::DenseSet<unsigned> rhsUsed;
  bool sparse = name == "intent.sparse_contract";
  auto sparseFormat =
      sparse ? operation->getAttrOfType<SparseFormatAttr>("format")
             : SparseFormatAttr();
  for (auto [left, right] : *reductions) {
    if (!lhsUsed.insert(left).second || !rhsUsed.insert(right).second ||
        (!(sparse && sparseFormat &&
           sparseFormat.getCompressionAxis() == left) &&
         !sameDimension(lhs, left, rhs, right)))
      return operation->emitOpError(
          "contract reduction axes must be unique and extent-compatible");
  }
  for (auto [left, right] : *batches) {
    if (!lhsUsed.insert(left).second || !rhsUsed.insert(right).second ||
        !sameDimension(lhs, left, rhs, right))
      return operation->emitOpError(
          "contract batch axes must be unique, disjoint, and compatible");
  }
  unsigned expectedRank = batches->size() + lhs.getRank() + rhs.getRank() -
                          2 * reductions->size() - 2 * batches->size();
  if (result.getRank() != expectedRank)
    return operation->emitOpError("contract result rank is inconsistent");
  SmallVector<std::pair<RankedTensorType, unsigned>> expectedAxes;
  for (unsigned axis = 0; axis < lhs.getRank(); ++axis)
    if (!llvm::any_of(*reductions,
                      [&](auto pair) { return pair.first == axis; }))
      expectedAxes.emplace_back(lhs, axis);
  for (unsigned axis = 0; axis < rhs.getRank(); ++axis) {
    bool reduction = llvm::any_of(
        *reductions, [&](auto pair) { return pair.second == axis; });
    bool batch = llvm::any_of(*batches,
                             [&](auto pair) { return pair.second == axis; });
    if (!reduction && !batch)
      expectedAxes.emplace_back(rhs, axis);
  }
  if (expectedAxes.size() != static_cast<unsigned>(result.getRank()))
    return operation->emitOpError("contract result axis schema is inconsistent");
  for (auto [resultAxis, expected] : llvm::enumerate(expectedAxes))
    if (!sameDimension(expected.first, expected.second, result, resultAxis))
      return operation->emitOpError(
          "contract result dimension lost its free/batch axis identity");
  if (!isNumericData(result))
    return operation->emitOpError(
        "contract accumulator/result element type must be numeric");
  if (name == "intent.sparse_contract") {
    auto format = operation->getAttrOfType<SparseFormatAttr>("format");
    if (!format || format.getCompressionAxis() >= lhs.getRank() ||
        !operation->getOperand(3).getType().isIndex())
      return operation->emitOpError("sparse contract format schema is invalid");
    if (!llvm::any_of(*reductions, [&](auto pair) {
          return pair.first == format.getCompressionAxis();
        }))
      return operation->emitOpError(
          "sparse compression axis must be a contraction reduction axis");
    auto reduction = llvm::find_if(*reductions, [&](auto pair) {
      return pair.first == format.getCompressionAxis();
    });
    Value logicalExtent = operation->getOperand(3);
    if (!extentValueMatchesAxis(logicalExtent, rhs, reduction->second))
      return operation->emitOpError(
          "sparse logical extent must equal its paired dense reduction axis");
    if (auto constant = getConstantInteger(logicalExtent)) {
      int64_t group = format.getKind() == 0 ? 2 : 4;
      if (*constant < 0 || *constant % group != 0)
        return operation->emitOpError(
            "sparse logical extent violates the closed format group size");
    }
    Type metadata = operation->getOperand(1).getType();
    if (format.getKind() == 0) {
      auto positions = dyn_cast<RankedTensorType>(metadata);
      if (!positions || !isIntegerLike(positions.getElementType()))
        return operation->emitOpError(
            "one-of-two metadata must be a logical-index tensor");
    } else {
      auto positions = dyn_cast<RecordType>(metadata);
      if (!positions || positions.getFieldNames().size() != 2 ||
          cast<StringAttr>(positions.getFieldNames()[0]).getValue() != "first" ||
          cast<StringAttr>(positions.getFieldNames()[1]).getValue() != "second")
        return operation->emitOpError(
            "two-of-four metadata must be the {first, second} record");
      auto first = dyn_cast<RankedTensorType>(
          cast<TypeAttr>(positions.getFieldTypes()[0]).getValue());
      auto second = dyn_cast<RankedTensorType>(
          cast<TypeAttr>(positions.getFieldTypes()[1]).getValue());
      if (!first || !second || !sameTensorShape(first, second) ||
          !isIntegerLike(first.getElementType()) ||
          !isIntegerLike(second.getElementType()))
        return operation->emitOpError(
            "two-of-four metadata fields must be shape-identical logical-index tensors");
    }
  }
  return success();
}

LogicalResult verifyAccess(Operation *operation) {
  FailureOr<IndexRelationInfo> relation = verifyIndexRelation(operation);
  if (failed(relation))
    return failure();
  StringRef name = operation->getName().getStringRef();
  Type sourceType = operation->getOperand(0).getType();
  auto source = getTensorSchema(sourceType);
  if (!source)
    return operation->emitOpError(
        "indexed access requires a ranked source/destination schema");

  bool load = name == "intent.view_load" || name == "intent.buffer_load" ||
              name == "intent.gather";
  bool viewAccess = name == "intent.view_load" || name == "intent.view_store";
  bool bufferAccess =
      name == "intent.buffer_load" || name == "intent.buffer_store";
  bool scatter = name == "intent.scatter_unique" ||
                 name == "intent.scatter_reduce";
  auto view = dyn_cast<ViewType>(sourceType);
  if ((viewAccess || scatter) && !view)
    return operation->emitOpError(
        "external access requires an external view destination/source");
  if (bufferAccess && !isa<BufferType>(sourceType))
    return operation->emitOpError(
        "logical-buffer access requires a logical buffer source");
  if (name == "intent.gather" && !isa<RankedTensorType>(sourceType))
    return operation->emitOpError("pure gather requires an immutable tensor source");
  if (view && load && view.getAccess() == 1)
    return operation->emitOpError("Out-only external view cannot be read");
  if (view && !load && view.getAccess() == 0)
    return operation->emitOpError("In-only external view cannot be written");

  if (load) {
    if (operation->getNumResults() != 1 ||
        failed(verifyIndexedData(operation, operation->getResult(0).getType(),
                                 source.getElementType(), *relation,
                                 "indexed read result")))
      return failure();
    SmallVector<unsigned> reserved;
    auto valid = operation->getAttrOfType<IntegerAttr>("valid_operand_index");
    auto fill = operation->getAttrOfType<IntegerAttr>("fill_operand_index");
    if (static_cast<bool>(valid) != static_cast<bool>(fill))
      return operation->emitOpError(
          "indexed read must provide both validity and fill or neither");
    if (valid) {
      if (valid.getInt() <= 0 || fill.getInt() <= 0 ||
          valid.getInt() >= operation->getNumOperands() ||
          fill.getInt() >= operation->getNumOperands() ||
          valid.getInt() == fill.getInt())
        return operation->emitOpError(
            "indexed read validity/fill operand indices are invalid");
      Type resultType = operation->getResult(0).getType();
      Type validType = operation->getOperand(valid.getInt()).getType();
      Type fillType = operation->getOperand(fill.getInt()).getType();
      if (!isBooleanData(validType) ||
          !sameDataSchema(validType, resultType, false) ||
          !sameDataSchema(fillType, resultType))
        return operation->emitOpError(
            "indexed read validity/fill must match the canonical result shape");
      reserved.push_back(valid.getInt());
      reserved.push_back(fill.getInt());
    }
    return verifyOperandPartition(operation, *relation, reserved);
  }

  auto valueIndex =
      operation->getAttrOfType<IntegerAttr>("value_operand_index");
  if (!valueIndex || valueIndex.getInt() <= 0 ||
      valueIndex.getInt() >= operation->getNumOperands())
    return operation->emitOpError("indexed write value operand index is invalid");
  if (failed(verifyIndexedData(
          operation, operation->getOperand(valueIndex.getInt()).getType(),
          source.getElementType(), *relation, "indexed write value")))
    return failure();
  if (failed(verifyOperandPartition(
          operation, *relation,
          {static_cast<unsigned>(valueIndex.getInt())})))
    return failure();
  if (name != "intent.scatter_reduce")
    return success();
  if (operation->getNumRegions() != 1)
    return operation->emitOpError(
        "scatter-reduce requires one typed combine region");
  Type valueType = source.getElementType();
  SmallVector<Type> arguments{valueType, valueType};
  SmallVector<Type> yielded{valueType};
  return verifyYieldSchema(operation, operation->getRegion(0), arguments,
                           yielded, true);
}

LogicalResult verifyAtomic(Operation *operation) {
  FailureOr<IndexRelationInfo> relation = verifyIndexRelation(operation);
  if (failed(relation))
    return failure();
  Type targetType = operation->getOperand(0).getType();
  auto target = getTensorSchema(targetType);
  if (!target || !isa<ViewType, BufferType>(targetType))
    return operation->emitOpError("atomic target must be a view or logical buffer");
  auto order = operation->getAttrOfType<AtomicOrderingAttr>("ordering");
  if (!order)
    return operation->emitOpError("atomic ordering is outside the language enum");
  StringRef name = operation->getName().getStringRef();
  if (auto view = dyn_cast<ViewType>(targetType); view && view.getAccess() != 2)
    return operation->emitOpError(
        "external atomic target must use InOut access semantics");
  if (name == "intent.atomic_load") {
    if (order.getValue() != AtomicOrdering::Relaxed &&
        order.getValue() != AtomicOrdering::Acquire)
      return operation->emitOpError("atomic load allows relaxed or acquire ordering");
    if (failed(verifyIndexedData(operation, operation->getResult(0).getType(),
                                 target.getElementType(), *relation,
                                 "atomic load result")))
      return failure();
    return verifyOperandPartition(operation, *relation, {});
  }
  StringRef valueName = name == "intent.atomic_compare_exchange"
                            ? "expected_operand"
                            : "value_operand";
  auto valueIndex = operation->getAttrOfType<IntegerAttr>(valueName);
  if (!valueIndex || valueIndex.getInt() <= 0 ||
      valueIndex.getInt() >= operation->getNumOperands())
    return operation->emitOpError("atomic value operand index is invalid");
  Type valueType = operation->getOperand(valueIndex.getInt()).getType();
  if (failed(verifyIndexedData(operation, valueType, target.getElementType(),
                               *relation, "atomic value")))
    return failure();
  if (name == "intent.atomic_store") {
    if (order.getValue() != AtomicOrdering::Relaxed &&
        order.getValue() != AtomicOrdering::Release)
      return operation->emitOpError("atomic store allows relaxed or release ordering");
    return verifyOperandPartition(
        operation, *relation,
        {static_cast<unsigned>(valueIndex.getInt())});
  }
  if (name == "intent.atomic_rmw") {
    auto kind = operation->getAttrOfType<AtomicRMWKindAttr>("kind");
    if (!kind ||
        operation->getResult(0).getType() != valueType)
      return operation->emitOpError("atomic RMW schema is invalid");
    return verifyOperandPartition(
        operation, *relation,
        {static_cast<unsigned>(valueIndex.getInt())});
  }
  auto desired = operation->getAttrOfType<IntegerAttr>("desired_operand");
  auto result = dyn_cast<RecordType>(operation->getResult(0).getType());
  Type successType =
      result && result.getFieldTypes().size() == 2
          ? cast<TypeAttr>(result.getFieldTypes()[1]).getValue()
          : Type();
  if (!desired || desired.getInt() != valueIndex.getInt() + 1 ||
      desired.getInt() >= operation->getNumOperands() ||
      operation->getOperand(desired.getInt()).getType() != valueType || !result ||
      result.getFieldNames().size() != 2 ||
      cast<StringAttr>(result.getFieldNames()[0]).getValue() != "old_value" ||
      cast<StringAttr>(result.getFieldNames()[1]).getValue() != "success" ||
      cast<TypeAttr>(result.getFieldTypes()[0]).getValue() != valueType ||
      !successType || !getElementType(successType).isInteger(1) ||
      !sameDataSchema(successType, valueType, false))
    return operation->emitOpError("compare-exchange result schema is invalid");
  return verifyOperandPartition(
      operation, *relation,
      {static_cast<unsigned>(valueIndex.getInt()),
       static_cast<unsigned>(desired.getInt())});
}

LogicalResult verifyCanonicalOperation(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  if (name == "intent.parallel" || name == "intent.if" ||
      name == "intent.for" || name == "intent.while")
    return verifyControlOperation(operation);
  if (name == "intent.make_tuple" || name == "intent.make_record" ||
      name == "intent.extract")
    return verifyProductOperation(operation);
  if (name == "intent.unary" || name == "intent.binary" ||
      name == "intent.compare" || name == "intent.select" ||
      name == "intent.mask" || name == "intent.cast" ||
      name == "intent.bitcast" || name == "intent.reshape" ||
      name == "intent.broadcast" || name == "intent.full")
    return verifyDataOperation(operation);
  if (name == "intent.reduce")
    return verifyReduceOrScan(operation, false);
  if (name == "intent.scan")
    return verifyReduceOrScan(operation, true);
  if (name == "intent.region_fold")
    return verifyRegionFold(operation);
  if (name == "intent.region_scan")
    return verifyRegionScan(operation);
  if (name == "intent.contract" || name == "intent.scaled_contract" ||
      name == "intent.sparse_contract")
    return verifyContract(operation);
  if (name == "intent.view_load" || name == "intent.view_store" ||
      name == "intent.gather" || name == "intent.scatter_unique" ||
      name == "intent.scatter_reduce" || name == "intent.buffer_load" ||
      name == "intent.buffer_store")
    return verifyAccess(operation);
  if (name.starts_with("intent.atomic_"))
    return verifyAtomic(operation);
  if (name == "intent.constant") {
    Attribute value = operation->getAttr("value");
    if (!value)
      return operation->emitOpError("constant requires a canonical value");
    Type result = operation->getResult(0).getType();
    if ((isa<IntegerAttr>(value) && !isa<IntegerType, IndexType>(result)) ||
        (isa<FloatAttr>(value) && !isa<FloatType>(result)))
      return operation->emitOpError("constant attribute/result type mismatch");
    return success();
  }
  if (name == "intent.dim") {
    auto axis = operation->getAttrOfType<IntegerAttr>("axis");
    auto dimension = operation->getAttrOfType<IntegerAttr>("dimension");
    auto rank = getLogicalRank(operation->getOperand(0).getType());
    if (!axis || axis.getInt() < 0 || !dimension || dimension.getInt() <= 0 ||
        !rank || axis.getInt() >= *rank)
      return operation->emitOpError("dimension query axis is invalid");
    Type sourceType = operation->getOperand(0).getType();
    if (auto source = getTensorSchema(sourceType)) {
      auto ids = getDimensionIDs(source);
      if (!source.isDynamicDim(axis.getInt()) || !ids ||
          ids[axis.getInt()] != dimension.getInt())
        return operation->emitOpError(
            "dimension query identity does not match its tensor source axis");
    } else {
      FailureOr<SmallVector<int64_t>> dimensions =
          getIterationExtentDimensions(operation->getOperand(0));
      if (failed(dimensions) ||
          (*dimensions)[axis.getInt()] != dimension.getInt())
        return operation->emitOpError(
            "dimension query identity does not match its iteration source axis");
    }
    return success();
  }
  if (name == "intent.domain") {
    auto result = dyn_cast<DomainType>(operation->getResult(0).getType());
    if (!result || result.getRank() != 1 ||
        (operation->getNumOperands() != 2 && operation->getNumOperands() != 3))
      return operation->emitOpError("domain requires start/stop[/step] and rank one");
    if (failed(getExtentDimensions(operation, 1)))
      return failure();
    for (Value bound : operation->getOperands())
      if (!isIntegerLike(bound.getType()))
        return operation->emitOpError("domain bound must be integer/index typed");
    return success();
  }
  if (name == "intent.domain_product") {
    auto result = dyn_cast<DomainType>(operation->getResult(0).getType());
    unsigned rank = 0;
    SmallVector<int64_t> expectedDimensions;
    for (Value operand : operation->getOperands()) {
      auto domain = dyn_cast<DomainType>(operand.getType());
      if (!domain)
        return operation->emitOpError("domain product requires domains");
      rank += domain.getRank();
      FailureOr<SmallVector<int64_t>> dimensions =
          getIterationExtentDimensions(operand);
      if (failed(dimensions))
        return operation->emitOpError(
            "domain product cannot recover a source extent identity");
      expectedDimensions.append(*dimensions);
    }
    if (!result || operation->getNumOperands() == 0 || result.getRank() != rank)
      return operation->emitOpError("domain product result rank is invalid");
    FailureOr<SmallVector<int64_t>> dimensions =
        getExtentDimensions(operation, rank);
    if (failed(dimensions) || *dimensions != expectedDimensions)
      return operation->emitOpError(
          "domain product extent identities must concatenate its operands");
    return success();
  }
  if (name == "intent.subregion") {
    Type source = operation->getOperand(0).getType();
    auto result = dyn_cast<RegionType>(operation->getResult(0).getType());
    auto hasStart = operation->getAttrOfType<BoolAttr>("has_start");
    auto hasStop = operation->getAttrOfType<BoolAttr>("has_stop");
    auto sourceID = getCoordinateSource(source);
    auto rank = getLogicalRank(source);
    unsigned expected = 1 + (hasStart && hasStart.getValue()) +
                        (hasStop && hasStop.getValue());
    if (!result || !hasStart || !hasStop || !sourceID || !rank || *rank != 1 ||
        operation->getNumOperands() != expected ||
        result.getSourceId() != *sourceID || result.getRank() != *rank)
      return operation->emitOpError("subregion source/bound schema is invalid");
    FailureOr<SmallVector<int64_t>> dimensions =
        getExtentDimensions(operation, *rank);
    if (failed(dimensions))
      return failure();
    if (!hasStart.getValue() && !hasStop.getValue()) {
      FailureOr<SmallVector<int64_t>> sourceDimensions =
          getIterationExtentDimensions(operation->getOperand(0));
      if (failed(sourceDimensions) || *dimensions != *sourceDimensions)
        return operation->emitOpError(
            "identity subregion must preserve source extent identities");
    } else if (llvm::any_of(*dimensions,
                            [](int64_t identity) { return identity <= 0; })) {
      return operation->emitOpError(
          "bounded subregion requires fresh dynamic extent identities");
    }
    for (Value bound : operation->getOperands().drop_front())
      if (!isIntegerLike(bound.getType()))
        return operation->emitOpError("subregion bound must be integer/index typed");
    return success();
  }
  if (name == "intent.indices") {
    auto sourceRank = getLogicalRank(operation->getOperand(0).getType());
    auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
    auto axis = operation->getAttrOfType<IntegerAttr>("tensor_axis");
    bool tensorSource = isa<RankedTensorType>(operation->getOperand(0).getType());
    if (!sourceRank || !result || !result.getElementType().isIndex() ||
        result.getRank() != *sourceRank || tensorSource != static_cast<bool>(axis) ||
        (axis && (axis.getInt() < 0 || axis.getInt() >= *sourceRank)))
      return operation->emitOpError("indices result/source schema is invalid");
    if (auto source = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
        source && !sameTensorShape(source, result))
      return operation->emitOpError(
          "tensor indices must preserve the source logical shape");
    if (!tensorSource) {
      FailureOr<SmallVector<int64_t>> dimensions =
          getIterationExtentDimensions(operation->getOperand(0));
      auto resultIDs = getDimensionIDs(result);
      if (failed(dimensions) ||
          dimensions->size() != static_cast<size_t>(result.getRank()))
        return operation->emitOpError(
            "indices cannot recover its source extent identities");
      for (auto [resultAxis, identity] : llvm::enumerate(*dimensions)) {
        if (identity <= 0 || !resultIDs || resultIDs[resultAxis] != identity)
          return operation->emitOpError(
              "indices result extent identity differs from its source");
      }
    }
    return success();
  }
  if (name == "intent.region_end") {
    auto rank = getLogicalRank(operation->getOperand(0).getType());
    return rank && *rank == 1 ? success()
                              : operation->emitOpError("region_end requires rank one");
  }
  if (name == "intent.assume_in_bounds") {
    auto axis = operation->getAttrOfType<IntegerAttr>("axis");
    auto rank = getLogicalRank(operation->getOperand(1).getType());
    if (!axis || axis.getInt() < 0 || !rank || axis.getInt() >= *rank ||
        !isIntegerLike(getElementType(operation->getOperand(0).getType())))
      return operation->emitOpError("in-bounds precondition schema is invalid");
    return success();
  }
  if (name == "intent.transpose") {
    auto input = getTensorSchema(operation->getOperand(0).getType());
    auto result = getTensorSchema(operation->getResult(0).getType());
    FailureOr<SmallVector<int64_t>> permutation =
        getIntegerArray(operation, "permutation");
    if (!input || !result || failed(permutation) ||
        permutation->size() != static_cast<size_t>(input.getRank()) ||
        result.getRank() != input.getRank())
      return operation->emitOpError("transpose permutation/rank is invalid");
    llvm::DenseSet<int64_t> axes;
    for (auto [resultAxis, sourceAxis] : llvm::enumerate(*permutation))
      if (sourceAxis < 0 || sourceAxis >= input.getRank() ||
          !axes.insert(sourceAxis).second ||
          !sameDimension(input, sourceAxis, result, resultAxis))
        return operation->emitOpError("transpose permutation is not a typed bijection");
    return success();
  }
  if (name == "intent.join") {
    auto lhs = getTensorSchema(operation->getOperand(0).getType());
    auto rhs = getTensorSchema(operation->getOperand(1).getType());
    auto result = getTensorSchema(operation->getResult(0).getType());
    if (!lhs || !rhs || !result || !sameTensorShape(lhs, rhs) ||
        result.getRank() != lhs.getRank() + 1 ||
        result.getElementType() != lhs.getElementType() ||
        result.getDimSize(result.getRank() - 1) != 2)
      return operation->emitOpError("join requires equal inputs and trailing extent two");
    for (unsigned axis = 0; axis < lhs.getRank(); ++axis)
      if (!sameDimension(lhs, axis, result, axis))
        return operation->emitOpError(
            "join result prefix lost source dimension identity");
    return success();
  }
  if (name == "intent.buffer") {
    auto buffer = dyn_cast<BufferType>(operation->getResult(0).getType());
    auto tensor = buffer ? dyn_cast<RankedTensorType>(buffer.getTensor())
                         : RankedTensorType();
    if (!buffer || !tensor)
      return operation->emitOpError("logical buffer result schema is invalid");
    FailureOr<SmallVector<unsigned>> dynamicOperands =
        verifyShapeRelation(operation, tensor, false);
    if (failed(dynamicOperands))
      return failure();
    for (auto [offset, position] : llvm::enumerate(*dynamicOperands))
      if (position != offset)
        return operation->emitOpError(
            "buffer shape extent operands must be canonical and position ordered");
    auto initial = operation->getAttrOfType<IntegerAttr>("initial_operand");
    unsigned expectedOperands = dynamicOperands->size() + (initial ? 1 : 0);
    if (operation->getNumOperands() != expectedOperands ||
        (initial && initial.getInt() != static_cast<int64_t>(dynamicOperands->size())))
      return operation->emitOpError(
          "logical buffer shape/initializer operand partition is invalid");
    if (initial) {
      Type initializer = operation->getOperand(initial.getInt()).getType();
      bool scalar = compatibleElementType(initializer, tensor.getElementType());
      auto initializerTensor = dyn_cast<RankedTensorType>(initializer);
      bool complete = initializerTensor &&
                      initializerTensor.getElementType() == tensor.getElementType() &&
                      sameTensorShape(initializerTensor, tensor);
      if (!scalar && !complete)
        return operation->emitOpError(
            "logical buffer initializer must be an element scalar or complete value");
    }
    return success();
  }
  if (name == "intent.histogram") {
    auto values = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
    auto valid = dyn_cast<RankedTensorType>(operation->getOperand(2).getType());
    auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
    auto resultDimension =
        operation->getAttrOfType<IntegerAttr>("result_dimension");
    if (!values || !valid || !result || !resultDimension ||
        !isIntegerLike(values.getElementType()) ||
        !valid.getElementType().isInteger(1) || !sameTensorShape(values, valid) ||
        result.getRank() != 1 || !isa<IntegerType>(result.getElementType()) ||
        result.getElementType().isInteger(1))
      return operation->emitOpError("histogram value/valid/result schema is invalid");
    Value bins = operation->getOperand(1);
    if (Operation *definition = bins.getDefiningOp()) {
      if (definition->getName().getStringRef() == "intent.constant") {
        auto value = definition->getAttrOfType<IntegerAttr>("value");
        auto ids = getDimensionIDs(result);
        if (!value || value.getInt() <= 0 || !ids ||
            resultDimension.getInt() <= 0 ||
            ids[0] != resultDimension.getInt() ||
            result.getDimSize(0) != value.getInt())
          return operation->emitOpError(
              "histogram result extent must equal its static bin count");
        return success();
      }
    }
    if (!ShapedType::isDynamic(result.getDimSize(0)))
      return operation->emitOpError(
          "runtime histogram bin count requires a dynamic result extent");
    auto ids = getDimensionIDs(result);
    if (!ids || resultDimension.getInt() <= 0 ||
        ids[0] != resultDimension.getInt())
      return operation->emitOpError(
          "runtime histogram bin count lost its dimension identity");
    return success();
  }
  if (name == "intent.random_bits") {
    Type counter = operation->getOperand(1).getType();
    Type result = operation->getResult(0).getType();
    if (!operation->getOperand(0).getType().isUnsignedInteger(64) ||
        !isIntegerLike(getElementType(counter)) ||
        !getElementType(result).isUnsignedInteger(32))
      return operation->emitOpError("Philox bits seed/counter/result schema is invalid");
    auto counterTensor = dyn_cast<RankedTensorType>(counter);
    auto resultTensor = dyn_cast<RankedTensorType>(result);
    if (static_cast<bool>(counterTensor) != static_cast<bool>(resultTensor) ||
        (counterTensor && !sameTensorShape(counterTensor, resultTensor)))
      return operation->emitOpError("Philox bits result must preserve counter shape");
    return success();
  }
  return success();
}

} // namespace

#define INTENT_DEFINE_VERIFY(OP)                                               \
  LogicalResult OP::verify() { return verifyCanonicalOperation(getOperation()); }

INTENT_DEFINE_VERIFY(AssumeInBoundsOp)
INTENT_DEFINE_VERIFY(AtomicCompareExchangeOp)
INTENT_DEFINE_VERIFY(AtomicLoadOp)
INTENT_DEFINE_VERIFY(AtomicRMWOp)
INTENT_DEFINE_VERIFY(AtomicStoreOp)
INTENT_DEFINE_VERIFY(BinaryOp)
INTENT_DEFINE_VERIFY(BitcastOp)
INTENT_DEFINE_VERIFY(BroadcastOp)
INTENT_DEFINE_VERIFY(BufferLoadOp)
INTENT_DEFINE_VERIFY(BufferOp)
INTENT_DEFINE_VERIFY(BufferStoreOp)
INTENT_DEFINE_VERIFY(CastOp)
INTENT_DEFINE_VERIFY(CompareOp)
INTENT_DEFINE_VERIFY(ConditionOp)
INTENT_DEFINE_VERIFY(ConstantOp)
INTENT_DEFINE_VERIFY(ContractOp)
INTENT_DEFINE_VERIFY(DimOp)
INTENT_DEFINE_VERIFY(DomainOp)
INTENT_DEFINE_VERIFY(DomainProductOp)
INTENT_DEFINE_VERIFY(ExtractOp)
INTENT_DEFINE_VERIFY(ForOp)
INTENT_DEFINE_VERIFY(FullOp)
INTENT_DEFINE_VERIFY(GatherOp)
INTENT_DEFINE_VERIFY(HistogramOp)
INTENT_DEFINE_VERIFY(IfOp)
INTENT_DEFINE_VERIFY(IndicesOp)
INTENT_DEFINE_VERIFY(JoinOp)
INTENT_DEFINE_VERIFY(MakeRecordOp)
INTENT_DEFINE_VERIFY(MakeTupleOp)
INTENT_DEFINE_VERIFY(MaskOp)
INTENT_DEFINE_VERIFY(ParallelOp)
INTENT_DEFINE_VERIFY(RandomBitsOp)
INTENT_DEFINE_VERIFY(ReduceOp)
INTENT_DEFINE_VERIFY(RegionEndOp)
INTENT_DEFINE_VERIFY(RegionFoldOp)
INTENT_DEFINE_VERIFY(RegionScanOp)
INTENT_DEFINE_VERIFY(ReshapeOp)
INTENT_DEFINE_VERIFY(ReturnOp)
INTENT_DEFINE_VERIFY(ScaledContractOp)
INTENT_DEFINE_VERIFY(ScanOp)
INTENT_DEFINE_VERIFY(ScatterReduceOp)
INTENT_DEFINE_VERIFY(ScatterUniqueOp)
INTENT_DEFINE_VERIFY(SelectOp)
INTENT_DEFINE_VERIFY(SparseContractOp)
INTENT_DEFINE_VERIFY(SubregionOp)
INTENT_DEFINE_VERIFY(TransposeOp)
INTENT_DEFINE_VERIFY(UnaryOp)
INTENT_DEFINE_VERIFY(ViewLoadOp)
INTENT_DEFINE_VERIFY(ViewStoreOp)
INTENT_DEFINE_VERIFY(WhileOp)
INTENT_DEFINE_VERIFY(YieldOp)

#undef INTENT_DEFINE_VERIFY

} // namespace intent

#define GET_OP_CLASSES
#include "Intent/Dialect/Intent/IR/IntentOps.cpp.inc"
