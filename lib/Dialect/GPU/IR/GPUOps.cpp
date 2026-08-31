#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

Type elementType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragment.getElementType();
  return type;
}

bool sameShape(Type lhs, Type rhs) {
  auto left = dyn_cast<FragmentType>(lhs);
  auto right = dyn_cast<FragmentType>(rhs);
  if (static_cast<bool>(left) != static_cast<bool>(right))
    return false;
  return !left || (left.getShape() == right.getShape() &&
                   left.getAxisMaps() == right.getAxisMaps() &&
                   left.getValidity() == right.getValidity() &&
                   left.getOwner() == right.getOwner());
}

bool sameExecutionShape(Type lhs, Type rhs) {
  auto left = dyn_cast<FragmentType>(lhs);
  auto right = dyn_cast<FragmentType>(rhs);
  if (static_cast<bool>(left) != static_cast<bool>(right))
    return false;
  return !left || (left.getShape() == right.getShape() &&
                   left.getValidity() == right.getValidity() &&
                   left.getOwner() == right.getOwner());
}

bool valueMatchesPhysicalExtent(Value value, PhysicalExprAttr extent) {
  if (auto physical = value.getDefiningOp<PhysicalExprOp>())
    return physical.getExpression() == extent;
  auto kind = static_cast<PhysicalExprKind>(extent.getKind());
  if (kind == PhysicalExprKind::Constant) {
    auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
    return constant && constant.value() == extent.getValue();
  }
  if (kind == PhysicalExprKind::Parameter) {
    auto parameter = value.getDefiningOp<ParameterOp>();
    return parameter && parameter.getParameter().getName() == extent.getSymbol();
  }
  return false;
}

struct PhysicalProduct {
  llvm::APInt constant = llvm::APInt(256, 1);
  SmallVector<PhysicalExprAttr> orderedTerms;
  bool valid = true;
};

void collectPhysicalProduct(PhysicalExprAttr value, PhysicalProduct &product) {
  if (!product.valid)
    return;
  auto kind = static_cast<PhysicalExprKind>(value.getKind());
  if (kind == PhysicalExprKind::Multiply) {
    if (value.getOperands().size() != 2) {
      product.valid = false;
      return;
    }
    collectPhysicalProduct(cast<PhysicalExprAttr>(value.getOperands()[0]),
                           product);
    collectPhysicalProduct(cast<PhysicalExprAttr>(value.getOperands()[1]),
                           product);
    return;
  }
  if (kind == PhysicalExprKind::Constant) {
    if (value.getValue() < 0) {
      product.valid = false;
      return;
    }
    product.constant *= llvm::APInt(256, value.getValue());
    return;
  }
  product.orderedTerms.push_back(value);
}

bool sameElementCount(FragmentType lhs, FragmentType rhs) {
  PhysicalProduct left;
  PhysicalProduct right;
  for (Attribute extent : lhs.getShape())
    collectPhysicalProduct(cast<PhysicalExprAttr>(extent), left);
  for (Attribute extent : rhs.getShape())
    collectPhysicalProduct(cast<PhysicalExprAttr>(extent), right);
  return left.valid && right.valid && left.constant == right.constant &&
         left.orderedTerms == right.orderedTerms;
}

bool sameElementCount(ArrayRef<Attribute> lhs, ArrayRef<Attribute> rhs) {
  PhysicalProduct left;
  PhysicalProduct right;
  for (Attribute extent : lhs)
    collectPhysicalProduct(cast<PhysicalExprAttr>(extent), left);
  for (Attribute extent : rhs)
    collectPhysicalProduct(cast<PhysicalExprAttr>(extent), right);
  return left.valid && right.valid && left.constant == right.constant &&
         left.orderedTerms == right.orderedTerms;
}

LogicalResult verifyDataSchemas(Operation *operation, TypeRange operands,
                                Type result, bool resultIsPredicate = false) {
  for (auto [index, operand] : llvm::enumerate(operands))
    if (!sameShape(operand, result))
      {
        InFlightDiagnostic diagnostic = operation->emitOpError(
            "physical data shapes/ownership disagree: operand=");
        diagnostic << operand << ", result=" << result;
        if (Operation *definition =
                operation->getOperand(index).getDefiningOp())
          diagnostic << ", producer=" << *definition;
        return failure();
      }
  if (resultIsPredicate) {
    if (!elementType(result).isInteger(1))
      return operation->emitOpError("physical predicate result must be i1");
  } else {
    for (Type operand : operands)
      if (elementType(operand) != elementType(result))
        return operation->emitOpError("physical data element types disagree");
  }
  return success();
}

unsigned rankOf(Type type) {
  if (auto view = dyn_cast<ViewType>(type))
    return view.getRank();
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragment.getShape().size();
  if (auto buffer = dyn_cast<BufferType>(type))
    return buffer.getShape().size();
  return 0;
}

Type resourceElementType(Type type) {
  if (auto view = dyn_cast<ViewType>(type))
    return view.getElementType();
  if (auto buffer = dyn_cast<BufferType>(type))
    return buffer.getElementType();
  return {};
}

LogicalResult verifyContractAxes(Operation *owner, FragmentType lhs,
                                 FragmentType rhs, FragmentType result,
                                 ArrayRef<int64_t> lhsReduction,
                                 ArrayRef<int64_t> rhsReduction,
                                 ArrayRef<int64_t> lhsBatch,
                                 ArrayRef<int64_t> rhsBatch,
                                 std::optional<int64_t> lhsExtentException =
                                     std::nullopt) {
  if (lhsReduction.empty() || lhsReduction.size() != rhsReduction.size() ||
      lhsBatch.size() != rhsBatch.size())
    return owner->emitOpError("physical contract axis-pair counts disagree");
  llvm::DenseSet<int64_t> lhsAxes;
  llvm::DenseSet<int64_t> rhsAxes;
  auto verifyPairs = [&](ArrayRef<int64_t> leftAxes,
                         ArrayRef<int64_t> rightAxes,
                         StringRef role) -> LogicalResult {
    for (auto [left, right] : llvm::zip(leftAxes, rightAxes)) {
      if (left < 0 || right < 0 ||
          static_cast<size_t>(left) >= lhs.getShape().size() ||
          static_cast<size_t>(right) >= rhs.getShape().size())
        return owner->emitOpError()
               << "physical contract " << role
               << " axis is out of range: lhs_axis=" << left
               << ", rhs_axis=" << right;
      if (!lhsAxes.insert(left).second || !rhsAxes.insert(right).second)
        return owner->emitOpError()
               << "physical contract " << role << " axis is repeated";
      if (lhs.getShape()[left] != rhs.getShape()[right] &&
          (!lhsExtentException || left != *lhsExtentException))
        return owner->emitOpError()
               << "physical contract " << role
               << " extents disagree: lhs_axis=" << left
               << ", lhs_extent=" << lhs.getShape()[left]
               << ", rhs_axis=" << right
               << ", rhs_extent=" << rhs.getShape()[right];
    }
    return success();
  };
  if (failed(verifyPairs(lhsReduction, rhsReduction, "reduction")) ||
      failed(verifyPairs(lhsBatch, rhsBatch, "batch")))
    return failure();
  size_t expectedRank = lhsBatch.size() +
                        (lhs.getShape().size() - lhsAxes.size()) +
                        (rhs.getShape().size() - rhsAxes.size());
  return result.getShape().size() == expectedRank
             ? success()
             : owner->emitOpError(
                   "physical contract result rank disagrees with batch/free axes");
}

} // namespace

static FailureOr<ArrayAttr> inferReshapeReassociationImpl(
    MLIRContext *context, ArrayRef<Attribute> sourceShape,
    ArrayRef<Attribute> resultShape, ArrayRef<Attribute> sourceMappings,
    ArrayRef<Attribute> resultMappings) {
  unsigned sourceRank = sourceShape.size();
  unsigned resultRank = resultShape.size();
  unsigned sourceBegin = 0;
  unsigned resultBegin = 0;
  SmallVector<Attribute> groups;
  auto relationScore = [&](unsigned sourceBegin, unsigned sourceEnd,
                           unsigned resultBegin, unsigned resultEnd) {
    if (sourceMappings.size() != sourceShape.size() ||
        resultMappings.size() != resultShape.size())
      return 0u;
    unsigned score = 0;
    for (unsigned resultAxis = resultBegin; resultAxis < resultEnd;
         ++resultAxis) {
      if (sameElementCount(ArrayRef<Attribute>(),
                           resultShape.slice(resultAxis, 1)))
        continue;
      auto resultMap = dyn_cast<AxisMapAttr>(resultMappings[resultAxis]);
      if (!resultMap)
        continue;
      unsigned best = 0;
      for (unsigned sourceAxis = sourceBegin; sourceAxis < sourceEnd;
           ++sourceAxis) {
        auto sourceMap = dyn_cast<AxisMapAttr>(sourceMappings[sourceAxis]);
        if (!sourceMap)
          continue;
        if (sourceMap.getSourceId() == resultMap.getSourceId() &&
            sourceMap.getSourceAxis() == resultMap.getSourceAxis() &&
            sourceMap.getDerived() == resultMap.getDerived()) {
          best = 2;
          break;
        }
        if (sourceMap.getDimensionId() > 0 &&
            sourceMap.getDimensionId() == resultMap.getDimensionId())
          best = std::max(best, 1u);
      }
      score += best;
    }
    return score;
  };
  while (sourceBegin < sourceRank && resultBegin < resultRank) {
    bool sourceUnit = sameElementCount(
        sourceShape.slice(sourceBegin, 1), ArrayRef<Attribute>());
    bool resultUnit = sameElementCount(
        ArrayRef<Attribute>(), resultShape.slice(resultBegin, 1));
    // Unit axes are real row-major structure, not an ambiguous factor of the
    // neighboring dynamic extent.  Preserve insertion/removal explicitly so a
    // later physical extent refinement cannot turn `[N] -> [1, N]` into
    // `[N] -> [N, N]` merely because both result axes carry the same logical
    // dimension identity.
    if (resultUnit && !sourceUnit) {
      groups.push_back(ReshapeGroupAttr::get(
          context, DenseI64ArrayAttr::get(context, {}),
          DenseI64ArrayAttr::get(context,
                                 {static_cast<int64_t>(resultBegin)})));
      ++resultBegin;
      continue;
    }
    if (sourceUnit && !resultUnit) {
      groups.push_back(ReshapeGroupAttr::get(
          context,
          DenseI64ArrayAttr::get(context,
                                 {static_cast<int64_t>(sourceBegin)}),
          DenseI64ArrayAttr::get(context, {})));
      ++sourceBegin;
      continue;
    }
    std::optional<std::pair<unsigned, unsigned>> match;
    unsigned bestScore = 0;
    for (unsigned sourceEnd = sourceBegin + 1;
         sourceEnd <= sourceRank; ++sourceEnd) {
      for (unsigned resultEnd = resultBegin + 1;
           resultEnd <= resultRank; ++resultEnd) {
        if (!sameElementCount(
                sourceShape.slice(sourceBegin, sourceEnd - sourceBegin),
                resultShape.slice(resultBegin, resultEnd - resultBegin)))
          continue;
        unsigned score =
            relationScore(sourceBegin, sourceEnd, resultBegin, resultEnd);
        if (!match || score > bestScore) {
          match = std::make_pair(sourceEnd, resultEnd);
          bestScore = score;
        }
      }
    }
    if (!match)
      return failure();
    SmallVector<int64_t> sourceAxes;
    SmallVector<int64_t> resultAxes;
    for (unsigned axis = sourceBegin; axis < match->first; ++axis)
      sourceAxes.push_back(axis);
    for (unsigned axis = resultBegin; axis < match->second; ++axis)
      resultAxes.push_back(axis);
    groups.push_back(ReshapeGroupAttr::get(
        context, DenseI64ArrayAttr::get(context, sourceAxes),
        DenseI64ArrayAttr::get(context, resultAxes)));
    sourceBegin = match->first;
    resultBegin = match->second;
  }
  if (sourceBegin < sourceRank) {
    ArrayRef<Attribute> remaining = sourceShape.drop_front(sourceBegin);
    if (!sameElementCount(remaining, ArrayRef<Attribute>()))
      return failure();
    SmallVector<int64_t> sourceAxes;
    for (unsigned axis = sourceBegin; axis < sourceRank; ++axis)
      sourceAxes.push_back(axis);
    groups.push_back(ReshapeGroupAttr::get(
        context, DenseI64ArrayAttr::get(context, sourceAxes),
        DenseI64ArrayAttr::get(context, {})));
    sourceBegin = sourceRank;
  }
  if (resultBegin < resultRank) {
    ArrayRef<Attribute> remaining = resultShape.drop_front(resultBegin);
    if (!sameElementCount(ArrayRef<Attribute>(), remaining))
      return failure();
    SmallVector<int64_t> resultAxes;
    for (unsigned axis = resultBegin; axis < resultRank; ++axis)
      resultAxes.push_back(axis);
    groups.push_back(ReshapeGroupAttr::get(
        context, DenseI64ArrayAttr::get(context, {}),
        DenseI64ArrayAttr::get(context, resultAxes)));
    resultBegin = resultRank;
  }
  if (sourceBegin != sourceRank || resultBegin != resultRank)
    return failure();
  return ArrayAttr::get(context, groups);
}

FailureOr<ArrayAttr>
inferReshapeReassociation(MLIRContext *context,
                          ArrayRef<Attribute> sourceShape,
                          ArrayRef<Attribute> resultShape) {
  return inferReshapeReassociationImpl(context, sourceShape, resultShape, {},
                                       {});
}

FailureOr<ArrayAttr>
inferReshapeReassociation(FragmentType source, FragmentType result,
                          unsigned sourcePrefix, unsigned resultPrefix) {
  if (!source || !result || sourcePrefix > source.getShape().size() ||
      resultPrefix > result.getShape().size())
    return failure();
  return inferReshapeReassociationImpl(
      source.getContext(), source.getShape().getValue().drop_front(sourcePrefix),
      result.getShape().getValue().drop_front(resultPrefix),
      source.getAxisMaps().getValue().drop_front(sourcePrefix),
      result.getAxisMaps().getValue().drop_front(resultPrefix));
}

LogicalResult ParameterOp::verify() {
  return getParameter().getCandidates().empty()
             ? emitOpError("parameter has no legal candidate")
             : success();
}

LogicalResult PhysicalExprOp::verify() { return success(); }

LogicalResult ProgramIdOp::verify() {
  return success();
}

LogicalResult WorksetCoordinateOp::verify() {
  const int64_t sourceId = getSourceIdAttr().getInt();
  const int64_t sourceAxis = getSourceAxisAttr().getInt();
  const int64_t sourceRank = getSourceRankAttr().getInt();
  const int64_t dimensionId = getDimensionIdAttr().getInt();
  if (sourceId <= 0)
    return emitOpError(
        "workset coordinate requires logical source provenance");
  if (sourceRank <= 0 || sourceAxis < 0 || sourceAxis >= sourceRank)
    return emitOpError(
        "workset coordinate source axis is outside its logical source rank");
  if (dimensionId <= 0)
    return emitOpError(
        "workset coordinate requires a logical dimension identity");
  auto worksetAxis = (*this)->getAttrOfType<IntegerAttr>(worksetAxisAttr);
  if (!worksetAxis || worksetAxis.getInt() < 0)
    return emitOpError(
        "workset coordinate requires a non-negative physical workset axis");
  return success();
}

LogicalResult DelinearizeOp::verify() {
  if (getExtents().empty() || getExtents().size() != getCoordinates().size() ||
      getLaunchExtents().size() != getExtents().size())
    return emitOpError(
        "delinearization requires one runtime and launch extent per coordinate");
  for (Attribute extent : getLaunchExtents())
    if (!isa<PhysicalExprAttr>(extent))
      return emitOpError("delinearization launch extents must be typed expressions");
  if (auto roles = (*this)->getAttrOfType<DenseI64ArrayAttr>(coordinateRolesAttr)) {
    if (static_cast<size_t>(roles.size()) != getCoordinates().size())
      return emitOpError(
          "delinearization requires one physical role per coordinate");
    for (int64_t role : roles.asArrayRef())
      if (role < static_cast<int64_t>(CoordinateRole::Unspecified) ||
          role > static_cast<int64_t>(CoordinateRole::IndirectTraversal))
        return emitOpError("delinearization coordinate role is invalid");
  }
  return success();
}

LogicalResult DimOp::verify() {
  return static_cast<uint64_t>(getAxis()) < getView().getType().getRank()
             ? success()
             : emitOpError("view dimension axis is out of range");
}

LogicalResult RangeOp::verify() {
  return getResult().getType().getSourceId() != 0
             ? success()
             : emitOpError("physical range requires logical source provenance");
}

LogicalResult RangeBoundOp::verify() {
  return getBound() <= 2 ? success()
                         : emitOpError("range bound selector is invalid");
}

LogicalResult MakeRangeOp::verify() {
  auto result = getResult().getType();
  if (result.getShape().size() != 1 ||
      result.getElementType() != getStart().getType() ||
      getExtent().getType() != getStart().getType() ||
      getStep().getType() != getStart().getType() ||
      getLogicalStart().getType() != getStart().getType() ||
      getLogicalStop().getType() != getStart().getType())
    return emitOpError("physical range must produce a rank-one index fragment");
  auto mapping = dyn_cast<AxisMapAttr>(result.getAxisMaps()[0]);
  if (!mapping || mapping.getSourceId() != static_cast<uint64_t>(getSourceId()) ||
      mapping.getSourceAxis() != static_cast<uint32_t>(getSourceAxis()) ||
      mapping.getDerived() != getDerived())
    return emitOpError("physical range result lost logical provenance");
  auto physicalExtent = dyn_cast<PhysicalExprAttr>(result.getShape()[0]);
  if (!physicalExtent ||
      !valueMatchesPhysicalExtent(getExtent(), physicalExtent)) {
    InFlightDiagnostic diagnostic = emitOpError(
        "physical range extent does not match its result fragment extent");
    diagnostic << "; result_extent=" << result.getShape()[0]
               << ", extent_operand=" << getExtent()
               << ", source_id=" << getSourceId()
               << ", source_axis=" << getSourceAxis();
    if (Operation *producer = getExtent().getDefiningOp())
      diagnostic << ", extent_producer=" << producer->getName();
    if (Operation *parent = (*this)->getParentOp())
      diagnostic << ", parent=" << parent->getName();
    return failure();
  }
  return success();
}

LogicalResult SplatOp::verify() {
  return getResult().getType().getElementType() == getValue().getType()
             ? success()
             : emitOpError("splat element/result type disagree");
}

LogicalResult BroadcastOp::verify() {
  if (auto input = dyn_cast<FragmentType>(getValue().getType())) {
    auto result = getResult().getType();
    if (input.getElementType() != result.getElementType() ||
        input.getOwner() != result.getOwner() ||
        input.getShape().size() > result.getShape().size())
      return emitOpError("broadcast physical schema is invalid");
    BroadcastProjection projection = queryBroadcastProjection(input, result);
    if (!projection.isExact())
      return emitOpError(
                 projection.state == BroadcastProjectionState::Ambiguous
                     ? "broadcast physical axis projection is ambiguous"
                     : "broadcast physical axis projection is unknown")
             << "; input=" << input << "; result=" << result;
  } else if (getValue().getType() != getResult().getType().getElementType()) {
      return emitOpError("scalar broadcast element type disagrees: value=")
             << getValue().getType()
             << ", result_element=" << getResult().getType().getElementType()
             << ", value_producer="
             << (getValue().getDefiningOp()
                     ? getValue().getDefiningOp()->getName().getStringRef()
                     : StringRef("block argument"))
             << ", result=" << getResult().getType();
  }
  return success();
}

LogicalResult UnaryOp::verify() {
  if (!sameShape(getInput().getType(), getResult().getType()) ||
      elementType(getInput().getType()) != elementType(getResult().getType()))
    return emitOpError("unary physical schema is invalid");
  return success();
}

LogicalResult BinaryOp::verify() {
  return verifyDataSchemas(getOperation(), {getLhs().getType(), getRhs().getType()},
                           getResult().getType());
}

LogicalResult CompareOp::verify() {
  if (!sameShape(getLhs().getType(), getRhs().getType()) ||
      !sameExecutionShape(getLhs().getType(), getResult().getType()) ||
      elementType(getLhs().getType()) != elementType(getRhs().getType()) ||
      !elementType(getResult().getType()).isInteger(1)) {
    return emitOpError("comparison physical schema is invalid: lhs=")
           << getLhs().getType() << ", rhs=" << getRhs().getType()
           << ", result=" << getResult().getType() << ", lhs_producer="
           << (getLhs().getDefiningOp()
                   ? getLhs().getDefiningOp()->getName().getStringRef()
                   : StringRef("block argument"))
           << ", rhs_producer="
           << (getRhs().getDefiningOp()
                   ? getRhs().getDefiningOp()->getName().getStringRef()
                   : StringRef("block argument"));
  }
  return success();
}

LogicalResult SelectOp::verify() {
  if (!sameShape(getCondition().getType(), getTrueValue().getType()) ||
      !sameShape(getTrueValue().getType(), getFalseValue().getType()) ||
      !sameShape(getTrueValue().getType(), getResult().getType()) ||
      !elementType(getCondition().getType()).isInteger(1) ||
      elementType(getTrueValue().getType()) != elementType(getFalseValue().getType()) ||
      elementType(getTrueValue().getType()) != elementType(getResult().getType()))
    return emitOpError("select physical schema is invalid");
  return success();
}

LogicalResult CastOp::verify() {
  if (sameShape(getValue().getType(), getResult().getType()))
    return success();
  return emitOpError("cast must preserve physical shape and ownership: ")
         << getValue().getType() << " vs " << getResult().getType();
}

LogicalResult BitcastOp::verify() {
  if (!sameShape(getValue().getType(), getResult().getType()))
    return emitOpError("bitcast must preserve physical shape and ownership");
  auto width = [](Type type) -> std::optional<unsigned> {
    if (auto integer = dyn_cast<IntegerType>(type))
      return integer.getWidth();
    if (auto floating = dyn_cast<FloatType>(type))
      return floating.getWidth();
    return std::nullopt;
  };
  auto source = width(elementType(getValue().getType()));
  auto target = width(elementType(getResult().getType()));
  return source && target && *source == *target
             ? success()
             : emitOpError("bitcast element widths disagree");
}

LogicalResult ReshapeOp::verify() {
  auto source = dyn_cast<FragmentType>(getValue().getType());
  auto result = dyn_cast<FragmentType>(getResult().getType());
  if (!source || !result || source.getElementType() != result.getElementType() ||
      source.getOwner() != result.getOwner() || !getReassociation() ||
      !sameElementCount(source, result)) {
    return emitOpError("reshape physical schema is invalid: source=")
           << getValue().getType() << ", result=" << getResult().getType()
           << ", reassociation=" << getReassociation();
  }
  unsigned nextSource = 0;
  unsigned nextResult = 0;
  for (Attribute attribute : getReassociation()) {
    auto group = dyn_cast<ReshapeGroupAttr>(attribute);
    if (!group ||
        (!group.getSourceAxes().empty() &&
         group.getSourceAxes()[0] != nextSource) ||
        (!group.getResultAxes().empty() &&
         group.getResultAxes()[0] != nextResult))
      return emitOpError(
          "reshape reassociation must consecutively cover both physical shapes");
    ArrayRef<int64_t> sourceAxes = group.getSourceAxes().asArrayRef();
    ArrayRef<int64_t> resultAxes = group.getResultAxes().asArrayRef();
    if (!sourceAxes.empty())
      nextSource = sourceAxes.back() + 1;
    if (!resultAxes.empty())
      nextResult = resultAxes.back() + 1;
  }
  unsigned logicalSourceRank = nextSource;
  unsigned logicalResultRank = nextResult;
  if (logicalSourceRank > source.getShape().size() ||
      logicalResultRank > result.getShape().size())
    return emitOpError("reshape reassociation axis is out of bounds");
  unsigned sourcePrefix = source.getShape().size() - logicalSourceRank;
  unsigned resultPrefix = result.getShape().size() - logicalResultRank;
  if (sourcePrefix != resultPrefix)
    return emitOpError(
        "reshape must preserve its physical execution prefix: source=")
           << source << ", result=" << result;
  for (unsigned axis = 0; axis < sourcePrefix; ++axis)
    if (source.getShape()[axis] != result.getShape()[axis] ||
        source.getAxisMaps()[axis] != result.getAxisMaps()[axis])
      return emitOpError(
          "reshape changed a physical execution-prefix axis");

  nextSource = 0;
  nextResult = 0;
  for (Attribute attribute : getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    ArrayRef<int64_t> sourceAxes = group.getSourceAxes().asArrayRef();
    ArrayRef<int64_t> resultAxes = group.getResultAxes().asArrayRef();
    SmallVector<Attribute> sourceExtents;
    SmallVector<Attribute> resultExtents;
    for (int64_t axis : sourceAxes)
      sourceExtents.push_back(source.getShape()[sourcePrefix + axis]);
    for (int64_t axis : resultAxes)
      resultExtents.push_back(result.getShape()[resultPrefix + axis]);
    if (!sameElementCount(sourceExtents, resultExtents))
      return emitOpError(
          "reshape reassociation group does not preserve row-major elements");
    if (!sourceAxes.empty())
      nextSource = sourceAxes.back() + 1;
    if (!resultAxes.empty())
      nextResult = resultAxes.back() + 1;
  }
  if (nextSource != logicalSourceRank || nextResult != logicalResultRank)
    return emitOpError(
        "reshape reassociation does not cover every logical physical axis");
  return success();
}

LogicalResult TransposeOp::verify() {
  auto source = getValue().getType();
  auto result = getResult().getType();
  if (source.getShape().size() != result.getShape().size() ||
      getPermutation().size() != source.getShape().size() ||
      source.getElementType() != result.getElementType() ||
      source.getOwner() != result.getOwner())
    return emitOpError("transpose physical rank/type/owner is invalid");
  llvm::DenseSet<int64_t> axes;
  for (auto [target, input] : llvm::enumerate(getPermutation()))
    if (input < 0 || static_cast<size_t>(input) >= source.getShape().size() ||
        !axes.insert(input).second || source.getShape()[input] != result.getShape()[target])
      return emitOpError("transpose permutation does not map physical extents");
  return success();
}

LogicalResult JoinOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto result = getResult().getType();
  if (lhs != rhs || getAxis() != lhs.getShape().size() ||
      result.getShape().size() != lhs.getShape().size() + 1 ||
      result.getElementType() != lhs.getElementType() ||
      result.getOwner() != lhs.getOwner())
    return emitOpError("join physical schema is invalid");
  for (unsigned axis = 0; axis < lhs.getShape().size(); ++axis)
    if (result.getShape()[axis] != lhs.getShape()[axis])
      return emitOpError("join changed a pre-existing physical extent");
  auto trailing = dyn_cast<PhysicalExprAttr>(
      result.getShape()[result.getShape().size() - 1]);
  if (!trailing ||
      trailing.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) ||
      trailing.getValue() != 2)
    return emitOpError("join trailing physical extent must be exactly two");
  return success();
}

LogicalResult MakeRecordOp::verify() {
  auto result = getResult().getType();
  if (result.getFieldTypes().size() != getFields().size())
    return emitOpError("record fields do not match its physical type");
  for (auto [index, field, type] :
       llvm::enumerate(getFields(), result.getFieldTypes()))
    if (field.getType() != cast<TypeAttr>(type).getValue())
      return emitOpError("record field has the wrong physical type")
             << "; field=" << index << "; actual=" << field.getType()
             << "; expected=" << cast<TypeAttr>(type).getValue();
  return success();
}

LogicalResult ExtractOp::verify() {
  auto record = getRecord().getType();
  if (getField() >= record.getFieldTypes().size())
    return emitOpError("record projection is outside its physical schema")
           << "; field=" << getField()
           << "; field_count=" << record.getFieldTypes().size();
  Type expected =
      cast<TypeAttr>(record.getFieldTypes()[getField()]).getValue();
  if (getResult().getType() != expected)
    return emitOpError("record projection is outside its physical schema")
           << "; field=" << getField()
           << "; actual=" << getResult().getType()
           << "; expected=" << expected;
  return success();
}

LogicalResult LoadOp::verify() {
  unsigned coordinateCount = getCoordinates().size();
  if (coordinateCount != rankOf(getResource().getType()) ||
      getSourceAxes().size() != coordinateCount)
    return emitOpError("load coordinate partition/rank is inconsistent");
  bool hasValid = static_cast<bool>(getValid());
  bool hasFill = static_cast<bool>(getFill());
  if (hasValid != hasFill)
    return emitOpError("load requires validity and fill together");
  if (hasValid && (!elementType(getValid().getType()).isInteger(1) ||
                   !sameShape(getValid().getType(), getResult().getType()) ||
                   !sameShape(getFill().getType(), getResult().getType())))
    return emitOpError("load validity/fill physical schema is invalid: result=")
           << getResult().getType() << ", valid=" << getValid().getType()
           << ", fill=" << getFill().getType();
  llvm::DenseSet<int64_t> axes;
  for (int64_t axis : getSourceAxes())
    if (axis < 0 || axis >= static_cast<int64_t>(coordinateCount) ||
        !axes.insert(axis).second)
      return emitOpError("load source-axis mapping is not a bijection");
  Type resourceElement = dyn_cast<ViewType>(getResource().getType())
                             ? cast<ViewType>(getResource().getType()).getElementType()
                             : cast<BufferType>(getResource().getType()).getElementType();
  return resourceElement == elementType(getResult().getType())
             ? success()
             : emitOpError("load resource/result element types disagree");
}

LogicalResult GatherOp::verify() {
  auto source = dyn_cast<FragmentType>(getSource().getType());
  if (!source || getCoordinates().empty() ||
      getSourceAxes().size() != getCoordinates().size())
    return emitOpError("gather source/coordinate relation is inconsistent");
  llvm::DenseSet<int64_t> axes;
  for (int64_t axis : getSourceAxes())
    if (axis < 0 || axis >= static_cast<int64_t>(source.getShape().size()) ||
        !axes.insert(axis).second)
      return emitOpError("gather source-axis relation is not a unique subset");
  bool hasValid = static_cast<bool>(getValid());
  bool hasFill = static_cast<bool>(getFill());
  if (hasValid != hasFill)
    return emitOpError("gather requires validity and fill together");
  if (hasValid && (!elementType(getValid().getType()).isInteger(1) ||
                   !sameShape(getValid().getType(), getResult().getType()) ||
                   !sameShape(getFill().getType(), getResult().getType())))
    return emitOpError("gather validity/fill physical schema is invalid");
  return source.getElementType() == elementType(getResult().getType())
             ? success()
             : emitOpError("gather source/result element types disagree");
}

LogicalResult AssumeInBoundsOp::verify() {
  return getAxis() < rankOf(getResource().getType())
             ? success()
             : emitOpError("assumed source axis is outside the resource rank");
}

void LoadOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

LogicalResult StoreOp::verify() {
  unsigned coordinateCount = getCoordinates().size();
  if (coordinateCount != rankOf(getResource().getType()) ||
      getSourceAxes().size() != coordinateCount)
    return emitOpError("store coordinate/effect schema is inconsistent");
  if (getValid() && (!elementType(getValid().getType()).isInteger(1) ||
                     !sameShape(getValid().getType(), getValue().getType())))
    return emitOpError("store validity must match its value fragment");
  llvm::DenseSet<int64_t> axes;
  for (int64_t axis : getSourceAxes())
    if (axis < 0 || axis >= static_cast<int64_t>(coordinateCount) ||
        !axes.insert(axis).second)
      return emitOpError("store source-axis mapping is not a bijection");
  Type resourceElement = dyn_cast<ViewType>(getResource().getType())
                             ? cast<ViewType>(getResource().getType()).getElementType()
                             : cast<BufferType>(getResource().getType()).getElementType();
  return resourceElement == elementType(getValue().getType())
             ? success()
             : emitOpError("store resource/value element types disagree");
}

void StoreOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult ContractOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto accumulator = getAccumulator().getType();
  auto result = getResult().getType();
  if (accumulator != result || lhs.getOwner() != rhs.getOwner() ||
      lhs.getOwner() != result.getOwner()) {
    InFlightDiagnostic diagnostic =
        emitOpError("physical contract relation/ownership is inconsistent");
    diagnostic << "; lhs_owner=" << lhs.getOwner()
               << ", rhs_owner=" << rhs.getOwner()
               << ", result_owner=" << result.getOwner()
               << ", accumulator=" << accumulator << ", result=" << result;
    return failure();
  }
  return verifyContractAxes(getOperation(), lhs, rhs, result,
                            getLhsReductionAxes(), getRhsReductionAxes(),
                            getLhsBatchAxes(), getRhsBatchAxes());
}

namespace {

LogicalResult verifyHelperRegion(Operation *owner, Region &region,
                                 TypeRange argumentTypes,
                                 TypeRange resultTypes) {
  if (!llvm::hasSingleElement(region))
    return owner->emitOpError("physical helper requires one block");
  Block &block = region.front();
  if (!llvm::equal(block.getArgumentTypes(), argumentTypes) || block.empty())
    return owner->emitOpError("physical helper arguments disagree with its schema");
  auto yield = dyn_cast<YieldOp>(block.back());
  if (!yield || !llvm::equal(yield.getOperandTypes(), resultTypes)) {
    InFlightDiagnostic diagnostic =
        owner->emitOpError("physical helper yield disagrees with its schema");
    diagnostic << "; expected=[";
    for (Type type : resultTypes)
      diagnostic << type << ", ";
    diagnostic << "]";
    if (yield) {
      diagnostic << "; actual=[";
      for (Type type : yield.getOperandTypes())
        diagnostic << type << ", ";
      diagnostic << "]";
    } else {
      diagnostic << "; actual=<no yield>";
    }
    return failure();
  }
  WalkResult effects = region.walk([&](Operation *nested) {
    if (isa<YieldOp>(nested))
      return WalkResult::advance();
    if (!isMemoryEffectFree(nested)) {
      nested->emitOpError("is effectful inside a physical pure helper");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return effects.wasInterrupted() ? failure() : success();
}

LogicalResult verifySegmentSlice(Operation *owner, Type sourceType,
                                 Type sliceType, uint64_t axis,
                                 ParameterAttr segment) {
  auto source = dyn_cast<FragmentType>(sourceType);
  auto slice = dyn_cast<FragmentType>(sliceType);
  if (!source || !slice || source.getElementType() != slice.getElementType() ||
      source.getOwner() != slice.getOwner() ||
      source.getShape().size() != slice.getShape().size() ||
      axis >= source.getShape().size())
    return owner->emitOpError(
        "physical region source slice lost rank/type/coordinate mapping");
  for (unsigned dimension = 0; dimension < source.getShape().size(); ++dimension) {
    if (dimension == axis) {
      auto extent = dyn_cast<PhysicalExprAttr>(slice.getShape()[dimension]);
      if (!extent ||
          extent.getKind() !=
              static_cast<uint32_t>(PhysicalExprKind::Parameter) ||
          extent.getSymbol() != segment.getName())
        return owner->emitOpError(
                   "physical region slice axis is not bound to its segment parameter: slice_extent=")
               << slice.getShape()[dimension]
               << ", segment=" << segment.getName();
    } else if (source.getShape()[dimension] != slice.getShape()[dimension] ||
               source.getAxisMaps()[dimension] !=
                   slice.getAxisMaps()[dimension]) {
      InFlightDiagnostic diagnostic = owner->emitOpError(
          "physical region slice changed a non-segment extent or coordinate mapping");
      diagnostic << "; axis=" << dimension
                 << ", source_extent=" << source.getShape()[dimension]
                 << ", slice_extent=" << slice.getShape()[dimension]
                 << ", source_mapping=" << source.getAxisMaps()[dimension]
                 << ", slice_mapping=" << slice.getAxisMaps()[dimension];
      return failure();
    }
  }
  return success();
}

bool preservesSliceAssembly(Type slice, Type result) {
  if (auto sliceFragment = dyn_cast<FragmentType>(slice)) {
    auto resultFragment = dyn_cast<FragmentType>(result);
    return resultFragment &&
           sliceFragment.getElementType() == resultFragment.getElementType() &&
           sliceFragment.getShape().size() ==
               resultFragment.getShape().size() &&
           sliceFragment.getAxisMaps() == resultFragment.getAxisMaps();
  }
  auto sliceRecord = dyn_cast<RecordType>(slice);
  auto resultRecord = dyn_cast<RecordType>(result);
  if (!sliceRecord || !resultRecord ||
      sliceRecord.getFieldNames() != resultRecord.getFieldNames() ||
      sliceRecord.getFieldTypes().size() !=
          resultRecord.getFieldTypes().size())
    return false;
  return llvm::all_of(
      llvm::zip(sliceRecord.getFieldTypes(), resultRecord.getFieldTypes()),
      [](auto fields) {
        return preservesSliceAssembly(
            cast<TypeAttr>(std::get<0>(fields)).getValue(),
            cast<TypeAttr>(std::get<1>(fields)).getValue());
      });
}

LogicalResult verifyReduceLike(Operation *owner, ValueRange inputs,
                               ResultRange results, Region &combine,
                               uint64_t sourceCount, uint64_t identityCount,
                               uint64_t captureCount) {
  if (sourceCount == 0 || sourceCount != identityCount ||
      inputs.size() != sourceCount + identityCount + captureCount ||
      results.size() != identityCount)
    return owner->emitOpError("physical reduce/scan component partition is invalid");
  SmallVector<Type> accumulators;
  for (unsigned index = 0; index < identityCount; ++index) {
    Type identity = inputs[sourceCount + index].getType();
    if (identity != results[index].getType())
      return owner->emitOpError("physical identity/result type disagrees");
    accumulators.push_back(identity);
  }
  SmallVector<Type> arguments(accumulators);
  arguments.append(accumulators);
  for (Value capture : inputs.drop_front(sourceCount + identityCount))
    arguments.push_back(capture.getType());
  return verifyHelperRegion(owner, combine, arguments, accumulators);
}

bool typeCarriesAxis(Type type, int64_t axis) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return axis >= 0 &&
           axis < static_cast<int64_t>(fragment.getShape().size());
  auto record = dyn_cast<RecordType>(type);
  return record && llvm::all_of(record.getFieldTypes(), [&](Attribute field) {
           return typeCarriesAxis(cast<TypeAttr>(field).getValue(), axis);
         });
}

} // namespace

LogicalResult ReduceOp::verify() {
  if (getAxes().empty())
    return emitOpError("physical reduce requires at least one axis");
  llvm::DenseSet<int64_t> axes;
  for (int64_t axis : getAxes()) {
    if (!axes.insert(axis).second)
      return emitOpError("physical reduce axes must be unique");
    for (Value source : getInputs().take_front(getSourceCount()))
      if (!typeCarriesAxis(source.getType(), axis))
        return emitOpError("physical reduce axis is outside a source schema");
  }
  return verifyReduceLike(getOperation(), getInputs(), getResults(), getCombine(),
                          getSourceCount(), getIdentityCount(), getCaptureCount());
}

LogicalResult ScanOp::verify() {
  for (Value source : getInputs().take_front(getSourceCount()))
    if (!typeCarriesAxis(source.getType(), getAxis()))
      return emitOpError("physical scan axis is outside a source schema");
  return verifyReduceLike(getOperation(), getInputs(), getResults(), getCombine(),
                          getSourceCount(), getIdentityCount(), getCaptureCount());
}

LogicalResult RegionFoldOp::verify() {
  uint64_t sourceCount = getSourceCount();
  uint64_t identityCount = getIdentityCount();
  uint64_t captureCount = getCaptureCount();
  if (sourceCount == 0 || identityCount == 0 ||
      getInputs().size() != sourceCount + identityCount + captureCount ||
      getResults().size() != identityCount || !getSegment())
    return emitOpError("physical region-fold component partition is invalid");
  SmallVector<Type> summaries;
  for (unsigned index = 0; index < identityCount; ++index) {
    Type identity = getInputs()[sourceCount + index].getType();
    if (identity != getResults()[index].getType())
      return emitOpError("region-fold identity/result type disagrees: identity=")
             << identity << ", result=" << getResults()[index].getType();
    summaries.push_back(identity);
  }
  if (!llvm::hasSingleElement(getSummarize()) ||
      getSummarize().front().getNumArguments() != sourceCount + captureCount)
    return emitOpError("region-fold summarizer argument partition is invalid");
  SmallVector<Type> summarizeArguments;
  for (unsigned index = 0; index < sourceCount; ++index) {
    Type slice = getSummarize().front().getArgument(index).getType();
    if (failed(verifySegmentSlice(getOperation(), getInputs()[index].getType(),
                                  slice, getAxis(), getSegment())))
      return failure();
    summarizeArguments.push_back(slice);
  }
  for (Value capture : getInputs().drop_front(sourceCount + identityCount))
    summarizeArguments.push_back(capture.getType());
  auto summarizeYield = dyn_cast<YieldOp>(getSummarize().front().back());
  if (!summarizeYield ||
      !llvm::equal(summarizeYield.getOperandTypes(), summaries))
    return emitOpError(
        "region-fold summarizer yield disagrees with its summary schema");
  if (failed(verifyHelperRegion(getOperation(), getSummarize(), summarizeArguments,
                                summaries)))
    return failure();
  SmallVector<Type> combineArguments(summaries);
  combineArguments.append(summaries);
  auto combineYield = dyn_cast<YieldOp>(getCombine().front().back());
  if (!combineYield || !llvm::equal(combineYield.getOperandTypes(), summaries)) {
    InFlightDiagnostic diagnostic = emitOpError(
        "region-fold combine yield disagrees with its summary schema");
    diagnostic << "; expected=[";
    for (Type type : summaries)
      diagnostic << type << ", ";
    diagnostic << "]";
    if (combineYield) {
      diagnostic << "; actual=[";
      for (Type type : combineYield.getOperandTypes())
        diagnostic << type << ", ";
      diagnostic << "]";
    }
    return failure();
  }
  return verifyHelperRegion(getOperation(), getCombine(), combineArguments,
                            summaries);
}

LogicalResult RegionScanOp::verify() {
  uint64_t sourceCount = getSourceCount();
  uint64_t identityCount = getIdentityCount();
  uint64_t stateCount = getStateCount();
  uint64_t captureCount = getCaptureCount();
  uint64_t outputCount = getOutputCount();
  if (sourceCount == 0 || identityCount == 0 || stateCount == 0 ||
      outputCount == 0 ||
      getInputs().size() != sourceCount + identityCount + stateCount + captureCount ||
      getResults().size() != outputCount + stateCount || !getSegment())
    return emitOpError("physical region-scan component partition is invalid");
  if (!llvm::hasSingleElement(getSummarize()) ||
      getSummarize().front().getNumArguments() != sourceCount + captureCount)
    return emitOpError("region-scan summarizer argument partition is invalid");
  SmallVector<Type> slices;
  for (unsigned index = 0; index < sourceCount; ++index) {
    Type slice = getSummarize().front().getArgument(index).getType();
    if (failed(verifySegmentSlice(getOperation(), getInputs()[index].getType(),
                                  slice, getAxis(), getSegment())))
      return failure();
    slices.push_back(slice);
  }
  SmallVector<Type> transitions;
  for (unsigned index = 0; index < identityCount; ++index)
    transitions.push_back(getInputs()[sourceCount + index].getType());
  SmallVector<Type> states;
  unsigned stateOffset = sourceCount + identityCount;
  for (unsigned index = 0; index < stateCount; ++index) {
    Type state = getInputs()[stateOffset + index].getType();
    if (state != getResults()[outputCount + index].getType())
      return emitOpError("region-scan final-state type disagrees: input=")
             << state << ", result="
             << getResults()[outputCount + index].getType();
    states.push_back(state);
  }
  SmallVector<Type> captures;
  unsigned captureOffset = stateOffset + stateCount;
  for (Value capture : getInputs().drop_front(captureOffset))
    captures.push_back(capture.getType());
  SmallVector<Type> summarizeArguments(slices);
  summarizeArguments.append(captures);
  if (failed(verifyHelperRegion(getOperation(), getSummarize(),
                                summarizeArguments, transitions)))
    return failure();
  SmallVector<Type> combineArguments(transitions);
  combineArguments.append(transitions);
  if (failed(verifyHelperRegion(getOperation(), getCombine(), combineArguments,
                                transitions)))
    return failure();
  SmallVector<Type> applyArguments(transitions);
  applyArguments.append(states);
  if (failed(verifyHelperRegion(getOperation(), getApply(), applyArguments,
                                states)))
    return failure();
  SmallVector<Type> emitArguments(slices);
  emitArguments.append(states);
  emitArguments.append(captures);
  if (!llvm::hasSingleElement(getEmit()) || getEmit().front().empty())
    return emitOpError("region-scan emitter is empty");
  auto yield = dyn_cast<YieldOp>(getEmit().front().back());
  if (!yield || yield.getValues().size() != outputCount)
    return emitOpError("region-scan emitter output count disagrees");
  SmallVector<Type> emitted(yield.getOperandTypes());
  if (failed(verifyHelperRegion(getOperation(), getEmit(), emitArguments,
                                emitted)))
    return failure();
  for (auto [slice, result] : llvm::zip(emitted, getResults().take_front(outputCount))) {
    if (!preservesSliceAssembly(slice, result.getType()))
      return emitOpError("region-scan output assembly relation is invalid");
  }
  return success();
}

LogicalResult ScaledContractOp::verify() {
  auto lhs = getLhs().getType();
  auto lhsScale = getLhsScale().getType();
  auto rhs = getRhs().getType();
  auto rhsScale = getRhsScale().getType();
  auto result = getResult().getType();
  auto sameLogicalAxis = [](FragmentType left, unsigned leftAxis,
                            FragmentType right, unsigned rightAxis) {
    auto lhsMap = cast<AxisMapAttr>(left.getAxisMaps()[leftAxis]);
    auto rhsMap = cast<AxisMapAttr>(right.getAxisMaps()[rightAxis]);
    return lhsMap.getDimensionId() > 0 &&
           lhsMap.getDimensionId() == rhsMap.getDimensionId();
  };
  auto constantExtent = [](FragmentType value,
                           unsigned axis) -> std::optional<int64_t> {
    auto extent = cast<PhysicalExprAttr>(value.getShape()[axis]);
    return extent.getKind() ==
                   static_cast<uint32_t>(PhysicalExprKind::Constant)
               ? std::optional<int64_t>(extent.getValue())
               : std::nullopt;
  };
  auto carrierExtent = [](ScaledFormat format,
                          uint64_t group) -> std::optional<int64_t> {
    uint64_t packing = format == ScaledFormat::E2M1 ? 2 : 1;
    return group % packing == 0
               ? std::optional<int64_t>(group / packing)
               : std::nullopt;
  };
  ArrayRef<int64_t> lhsReduction = getLhsReductionAxes();
  ArrayRef<int64_t> rhsReduction = getRhsReductionAxes();
  bool fixedAxes = lhsReduction.size() == 2 && rhsReduction.size() == 2 &&
                   lhsReduction[0] == 1 && lhsReduction[1] == 2 &&
                   rhsReduction[0] == 0 && rhsReduction[1] == 1 &&
                   getLhsBatchAxes().empty() && getRhsBatchAxes().empty();
  if (getLhsGroupSize() == 0 ||
      getLhsGroupSize() != getRhsGroupSize() ||
      getAccumulator().getType() != result || lhs.getOwner() != rhs.getOwner() ||
      lhs.getOwner() != lhsScale.getOwner() ||
      lhs.getOwner() != rhsScale.getOwner() || lhs.getOwner() != result.getOwner() ||
      lhs.getShape().size() != 3 || lhsScale.getShape().size() != 2 ||
      rhs.getShape().size() != 3 || rhsScale.getShape().size() != 2 ||
      result.getShape().size() != 2 || !fixedAxes)
    return emitOpError("scaled-contract physical schema is invalid");
  std::optional<int64_t> lhsCarrier =
      carrierExtent(getLhsFormat(), getLhsGroupSize());
  std::optional<int64_t> rhsCarrier =
      carrierExtent(getRhsFormat(), getRhsGroupSize());
  if (!lhsCarrier || !rhsCarrier || constantExtent(lhs, 2) != lhsCarrier ||
      constantExtent(rhs, 1) != rhsCarrier ||
      !sameLogicalAxis(lhs, 0, lhsScale, 0) ||
      !sameLogicalAxis(lhs, 1, lhsScale, 1) ||
      !sameLogicalAxis(lhs, 1, rhs, 0) ||
      !sameLogicalAxis(lhs, 1, rhsScale, 1) ||
      !sameLogicalAxis(rhs, 2, rhsScale, 0) ||
      !sameLogicalAxis(lhs, 0, result, 0) ||
      !sameLogicalAxis(rhs, 2, result, 1))
    return emitOpError("scaled-contract physical scale-axis relation is invalid");
  return success();
}

LogicalResult SparseContractOp::verify() {
  auto lhs = getCompressed().getType();
  auto rhs = getRhs().getType();
  auto result = getResult().getType();
  if (!getFormat() ||
      getFormat().getCompressionAxis() >=
          getCompressed().getType().getShape().size() ||
      getAccumulator().getType() != result || lhs.getOwner() != rhs.getOwner() ||
      lhs.getOwner() != result.getOwner())
    return emitOpError("sparse-contract physical schema is invalid");
  return verifyContractAxes(getOperation(), lhs, rhs, result,
                            getLhsReductionAxes(), getRhsReductionAxes(),
                            getLhsBatchAxes(), getRhsBatchAxes(),
                            static_cast<int64_t>(
                                getFormat().getCompressionAxis()));
}

LogicalResult HistogramOp::verify() {
  if (!getValues().getType().getElementType().isIntOrIndex() ||
      !getValid().getType().getElementType().isInteger(1) ||
      getValues().getType().getShape() != getValid().getType().getShape() ||
      getResult().getType().getShape().size() != 1)
    return emitOpError("histogram physical schema is invalid");
  return success();
}

LogicalResult ScatterReduceOp::verify() {
  if (getCoordinates().size() != rankOf(getResource().getType()) ||
      getSourceAxes().size() != getCoordinates().size())
    return emitOpError("scatter-reduce address rank is invalid");
  if (resourceElementType(getResource().getType()) !=
          elementType(getValue().getType()) ||
      (getValid() &&
       (!elementType(getValid().getType()).isInteger(1) ||
        !sameShape(getValid().getType(), getValue().getType()))))
    return emitOpError("scatter-reduce value/validity schema is invalid");
  llvm::DenseSet<int64_t> axes;
  for (int64_t axis : getSourceAxes())
    if (axis < 0 ||
        axis >= static_cast<int64_t>(rankOf(getResource().getType())) ||
        !axes.insert(axis).second)
      return emitOpError(
          "scatter-reduce source-axis mapping is not a bijection");
  SmallVector<Type> arguments{getValue().getType(), getValue().getType()};
  SmallVector<Type> results{getValue().getType()};
  return verifyHelperRegion(getOperation(), getCombine(), arguments, results);
}

void ScatterReduceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

namespace {

LogicalResult verifyAtomicAddress(Operation *owner, Type resource,
                                  ValueRange coordinates, Value valid,
                                  ArrayRef<int64_t> sourceAxes,
                                  AtomicOrdering ordering,
                                  AtomicSharingDomain sharing) {
  if (coordinates.size() != rankOf(resource) ||
      sourceAxes.size() != coordinates.size())
    return owner->emitOpError("atomic physical address/order schema is invalid");
  bool orderingLegal =
      (isa<AtomicLoadOp>(owner) &&
       (ordering == AtomicOrdering::Relaxed ||
        ordering == AtomicOrdering::Acquire)) ||
      (isa<AtomicStoreOp>(owner) &&
       (ordering == AtomicOrdering::Relaxed ||
        ordering == AtomicOrdering::Release)) ||
      isa<AtomicRMWOp, AtomicCompareExchangeOp>(owner);
  if (!orderingLegal)
    return owner->emitOpError(
        "atomic ordering is illegal for this physical operation");
  if (auto buffer = dyn_cast<BufferType>(resource)) {
    AtomicSharingDomain expected =
        buffer.getScope().getValue() == BufferScope::InvocationWorkspace
            ? AtomicSharingDomain::KernelInvocation
            : AtomicSharingDomain::ProgramInstance;
    if (sharing != expected)
      return owner->emitOpError(
          "atomic sharing domain disagrees with its physical allocation scope");
  } else if (isa<ViewType>(resource) &&
             sharing != AtomicSharingDomain::KernelInvocation) {
    return owner->emitOpError(
        "external-view atomic must cover the kernel invocation");
  }
  if (valid && !elementType(valid.getType()).isInteger(1))
    return owner->emitOpError("atomic validity must be a predicate");
  llvm::DenseSet<int64_t> axes;
  for (int64_t axis : sourceAxes)
    if (axis < 0 || axis >= static_cast<int64_t>(rankOf(resource)) ||
        !axes.insert(axis).second)
      return owner->emitOpError(
          "atomic source-axis mapping is not a bijection");
  return success();
}

} // namespace

LogicalResult AtomicLoadOp::verify() {
  if (resourceElementType(getResource().getType()) !=
          elementType(getResult().getType()) ||
      (getValid() && !sameShape(getValid().getType(), getResult().getType())))
    return emitOpError("atomic-load resource/result element types disagree");
  return verifyAtomicAddress(getOperation(), getResource().getType(),
                             getCoordinates(), getValid(), getSourceAxes(),
                             getOrdering(), getSharing());
}

LogicalResult AtomicStoreOp::verify() {
  if (resourceElementType(getResource().getType()) !=
          elementType(getValue().getType()) ||
      (getValid() && !sameShape(getValid().getType(), getValue().getType())))
    return emitOpError("atomic-store resource/value element types disagree");
  return verifyAtomicAddress(getOperation(), getResource().getType(),
                             getCoordinates(), getValid(), getSourceAxes(),
                             getOrdering(), getSharing());
}

LogicalResult AtomicRMWOp::verify() {
  if (getResult().getType() != getValue().getType() ||
      resourceElementType(getResource().getType()) !=
          elementType(getValue().getType()) ||
      (getValid() && !sameShape(getValid().getType(), getValue().getType())))
    return emitOpError("atomic RMW physical value/kind schema is invalid")
           << "; resource element="
           << resourceElementType(getResource().getType())
           << ", value=" << getValue().getType()
           << ", result=" << getResult().getType()
           << ", kind=" << stringifyAtomicRMWKind(getKind());
  return verifyAtomicAddress(getOperation(), getResource().getType(),
                             getCoordinates(), getValid(), getSourceAxes(),
                             getOrdering(), getSharing());
}

LogicalResult AtomicCompareExchangeOp::verify() {
  auto result = getResult().getType();
  if (getExpected().getType() != getDesired().getType() ||
      resourceElementType(getResource().getType()) !=
          elementType(getExpected().getType()) ||
      (getValid() && !sameShape(getValid().getType(), getExpected().getType())) ||
      result.getFieldTypes().size() != 2 ||
      cast<TypeAttr>(result.getFieldTypes()[0]).getValue() !=
          getExpected().getType() ||
      !elementType(cast<TypeAttr>(result.getFieldTypes()[1]).getValue())
           .isInteger(1))
    return emitOpError("compare-exchange physical result schema is invalid");
  return verifyAtomicAddress(getOperation(), getResource().getType(),
                             getCoordinates(), getValid(), getSourceAxes(),
                             getOrdering(), getSharing());
}

LogicalResult RandomBitsOp::verify() {
  return elementType(getResult().getType()).isUnsignedInteger(32) &&
                 sameShape(getCounter().getType(), getResult().getType())
             ? success()
             : emitOpError("Philox result must be a shape-identical u32 value");
}

LogicalResult BufferOp::verify() {
  auto type = getResult().getType();
  if (type.getWorkspace())
    return emitOpError(
        "invocation workspace must be an explicit hidden ABI resource");
  if ((type.getInitialization().getValue() ==
       BufferInitialization::FullValue) !=
      static_cast<bool>(getInitialValue()))
    return emitOpError(
        "buffer initializer disagrees with its initialization obligation");
  if (getInitialValue() &&
      elementType(getInitialValue().getType()) != getResult().getType().getElementType())
    return emitOpError("buffer initializer element type disagrees");
  return success();
}

void BufferOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get());
  if (getInitialValue())
    effects.emplace_back(MemoryEffects::Write::get());
}

} // namespace intent::gpu

#define GET_OP_CLASSES
#include "Intent/Dialect/GPU/IR/GPUOps.cpp.inc"
