#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::gpu {

ParameterAttr ParameterOp::getDeclaration() {
  return lookupParameterDeclaration(getOperation(), getReference());
}

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
  auto kind = extent.getKind();
  if (kind == PhysicalExprKind::Constant) {
    auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
    return constant && constant.value() == extent.getValue();
  }
  if (kind == PhysicalExprKind::Parameter) {
    auto parameter = value.getDefiningOp<ParameterOp>();
    return parameter && parameter.getReference() == extent.getParameterReference();
  }
  return false;
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

LogicalResult verifyWritableResource(Operation *owner, Type resource) {
  auto view = dyn_cast<ViewType>(resource);
  return !view || view.getAccess() != 0
             ? success()
             : owner->emitOpError("In-only external view cannot be written");
}

LogicalResult verifyResourceSharing(Operation *owner, Type resource,
                                    AtomicSharingDomain sharing) {
  AtomicSharingDomain expected;
  if (auto buffer = dyn_cast<BufferType>(resource))
    expected = buffer.getScope().getValue() == BufferScope::InvocationWorkspace
                   ? AtomicSharingDomain::KernelInvocation
                   : AtomicSharingDomain::ProgramInstance;
  else if (isa<ViewType>(resource))
    expected = AtomicSharingDomain::KernelInvocation;
  else
    return owner->emitOpError("sharing requires a physical resource");
  return sharing == expected
             ? success()
             : owner->emitOpError(
                   "sharing domain disagrees with the physical resource scope");
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
  if (sourceShape == resultShape) {
    for (unsigned axis = 0; axis < sourceRank; ++axis) {
      auto axes = DenseI64ArrayAttr::get(context, {static_cast<int64_t>(axis)});
      groups.push_back(ReshapeGroupAttr::get(context, axes, axes));
    }
    return ArrayAttr::get(context, groups);
  }
  auto relationScore = [&](unsigned sourceBegin, unsigned sourceEnd,
                           unsigned resultBegin, unsigned resultEnd) {
    if (sourceMappings.size() != sourceShape.size() ||
        resultMappings.size() != resultShape.size())
      return 0u;
    unsigned score = 0;
    for (unsigned resultAxis = resultBegin; resultAxis < resultEnd;
         ++resultAxis) {
      if (haveEqualPhysicalElementCounts(ArrayRef<Attribute>(),
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
    bool sourceUnit = haveEqualPhysicalElementCounts(
        sourceShape.slice(sourceBegin, 1), ArrayRef<Attribute>());
    bool resultUnit = haveEqualPhysicalElementCounts(
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
        if (!haveEqualPhysicalElementCounts(
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
    if (!haveEqualPhysicalElementCounts(remaining, ArrayRef<Attribute>()))
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
    if (!haveEqualPhysicalElementCounts(ArrayRef<Attribute>(), remaining))
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
  ParameterAttr declaration = getDeclaration();
  if (!declaration)
    return emitOpError("references an undeclared kernel parameter") << "; reference=" << getReference();
  if (!declaration.isExtent())
    return emitOpError("index parameter read requires a positive index declaration");
  return success();
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
  if (getExtents().empty() || getExtents().size() != getCoordinates().size())
    return emitOpError(
        "delinearization requires one runtime extent per coordinate");
  return success();
}

std::optional<DecodedCoordinate> queryDecodedCoordinate(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    if (auto group = dyn_cast<ExecutionGroupOp>(argument.getOwner()->getParentOp()))
      return DecodedCoordinate{group.getLinear(), group.getExtents(),
                               argument.getArgNumber()};
  if (auto result = dyn_cast<OpResult>(value))
    if (auto decode = dyn_cast<DelinearizeOp>(result.getOwner()))
      return DecodedCoordinate{decode.getLinear(), decode.getExtents(),
                               result.getResultNumber()};
  return std::nullopt;
}

void ExecutionGroupOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent())
    regions.emplace_back(&getBody());
  else
    regions.emplace_back();
}

LogicalResult ExecutionGroupOp::verify() {
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("requires one execution block");
  if (getExtents().empty() || getExtents().size() != getCoordinates().size() ||
      getLaunchExtents().size() != getExtents().size() ||
      getCoordinateRoles().size() != getExtents().size())
    return emitOpError("requires one runtime extent, launch extent, role and block argument per coordinate");
  for (BlockArgument coordinate : getCoordinates())
    if (!coordinate.getType().isIndex())
      return emitOpError("execution coordinates must have index type");
  for (Attribute extent : getLaunchExtents())
    if (!isa<PhysicalExprAttr>(extent))
      return emitOpError("launch extents must be typed expressions");
  for (int64_t role : getCoordinateRoles())
    if (role < static_cast<int64_t>(CoordinateRole::Unspecified) ||
        role > static_cast<int64_t>(CoordinateRole::IndirectTraversal))
      return emitOpError("coordinate role is invalid");
  if (getGroupId() < 0)
    return emitOpError("requires a nonnegative execution group identity");
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

LogicalResult ViewOverlapOp::verify() {
  auto kernel = (*this)->getParentOfType<func::FuncOp>();
  if (!kernel || !kernel->hasAttr(kernelAttr) ||
      (*this)->getBlock() != &kernel.front())
    return emitOpError("must be declared in a physical GPU kernel entry block");
  if (getLhs() == getRhs())
    return emitOpError("requires two distinct external view arguments");
  for (Value value : getOperands()) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &kernel.front())
      return emitOpError("requires external view ABI arguments");
    if (!getPublicView(argument))
      return emitOpError("requires external view ABI arguments");
  }
  return success();
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
      return emitOpError("broadcast physical schema is invalid")
             << "; input=" << input << "; result=" << result;
    auto relations = queryFragmentOperandRelations(getOperation());
    if (failed(relations) || !relations->front().hasCompatibleExtents())
      return emitOpError("broadcast physical axis projection is unknown or incompatible")
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
  return verifyPointwiseMathMode(getOperation(), elementType(getInput().getType()));
}

LogicalResult BinaryOp::verify() {
  if (failed(verifyPointwiseMathMode(getOperation(), elementType(getLhs().getType()))))
    return failure();
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
      source.getOwner() != result.getOwner() || !getReassociation()) {
    return emitOpError("reshape physical schema is invalid: source=")
           << getValue().getType() << ", result=" << getResult().getType()
           << ", reassociation=" << getReassociation();
  }
  auto relations = queryFragmentOperandRelations(getOperation());
  if (failed(relations))
    return emitOpError(
        "reshape reassociation must cover consecutive axes and preserve the "
        "execution-prefix rank");
  const auto &relation = relations->front();
  for (const auto &group : relation.groups)
    if (group.kind == FragmentAxisRelationKind::Corresponding &&
        source.getAxisMaps()[group.sourceAxes[0]] !=
            result.getAxisMaps()[group.resultAxes[0]])
      return emitOpError("reshape changed a physical execution-prefix axis");
  if (!relation.hasCompatibleExtents())
    return emitOpError(
        "reshape reassociation group does not preserve row-major elements");
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
  auto relations = queryFragmentOperandRelations(getOperation());
  if (failed(relations) || !relations->front().hasCompatibleExtents())
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
      trailing.getKind() != PhysicalExprKind::Constant ||
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

LogicalResult verifyAccessSchema(AccessOpInterface access) {
  Type resource = access.getAccessResource().getType();
  Type payload = access.getAccessValueType();
  Value valid = access.getAccessValidity(), fill = access.getAccessFill();
  unsigned coordinateCount = access.getAccessCoordinates().size();
  bool gather = access.getAccessKind() == AccessKind::Gather;
  if ((gather && (!isa<FragmentType>(resource) || !coordinateCount)) ||
      (!gather && coordinateCount != rankOf(resource)) ||
      access.getAccessSourceAxes().size() != coordinateCount)
    return access.emitOpError("access coordinate partition/rank is inconsistent");
  if (access.getAccessFillMutable() && bool(valid) != bool(fill))
    return access.emitOpError("read requires validity and fill together");
  if (valid && (!elementType(valid.getType()).isInteger(1) ||
                !sameShape(valid.getType(), payload)))
    return access.emitOpError("access validity must match its value schema")
           << "; value=" << payload << "; valid=" << valid.getType();
  if (fill && !sameShape(fill.getType(), payload))
    return access.emitOpError("read fill must match its value schema")
           << "; value=" << payload << "; fill=" << fill.getType();
  llvm::DenseSet<int64_t> axes;
  for (int64_t axis : access.getAccessSourceAxes())
    if (axis < 0 || axis >= static_cast<int64_t>(rankOf(resource)) ||
        !axes.insert(axis).second)
      return access.emitOpError(gather ? "gather source axes must be a unique subset"
                                       : "memory source axes must be a bijection");
  Type resourceElement = gather ? cast<FragmentType>(resource).getElementType()
                                : resourceElementType(resource);
  if (resourceElement != elementType(payload))
    return access.emitOpError("access resource/value element types disagree");
  for (auto [index, coordinate] : llvm::enumerate(access.getAccessCoordinates())) {
    auto projection = queryAccessCoordinateProjection(access, index);
    if (!projection.isExact())
      return access.emitOpError(
          "access coordinate has no exact projection into its payload lanes")
             << "; coordinate_slot=" << index
             << "; resource_axis=" << access.getAccessSourceAxes()[index]
             << "; coordinate=" << coordinate.getType()
             << "; payload=" << payload;
  }
  return success();
}

LogicalResult AssumeInBoundsOp::verify() {
  if (!isa<IntegerType, IndexType>(elementType(getIndex().getType())))
    return emitOpError("in-bounds assumption requires an integer/index value");
  return getAxis() < rankOf(getResource().getType())
             ? success()
             : emitOpError("assumed source axis is outside the resource rank");
}

void LoadOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

LogicalResult StoreOp::verify() {
  return verifyWritableResource(getOperation(), getResource().getType());
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
                                 ParameterRefAttr segment) {
  auto declaration = lookupParameterDeclaration(owner, segment);
  if (!declaration || !declaration.isExtent())
    return owner->emitOpError("region segment requires a declared positive index parameter");
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
              PhysicalExprKind::Parameter ||
          extent.getParameterReference() != segment)
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

LogicalResult verifyReduceLike(StructuredOpInterface operation) {
  SmallVector<Type> accumulators;
  for (auto [identity, result] : llvm::zip_equal(operation.getIdentities(), operation->getResults())) {
    if (identity.getType() != result.getType())
      return operation->emitOpError("physical identity/result type disagrees");
    accumulators.push_back(identity.getType());
  }
  return verifyHelperRegion(operation, operation.getCombine(),
                            operation.getCombineArgumentTypes(), accumulators);
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

LogicalResult verifyReductionResultRelations(Operation *owner,
                                              ValueRange sources,
                                              ResultRange results,
                                              ArrayRef<int64_t> axes) {
  if (sources.size() != results.size())
    return owner->emitOpError("physical reduction needs one result relation per source component");
  for (auto [source, result] : llvm::zip_equal(sources, results)) {
    auto expected = inferCollectiveResultType(source.getType(), axes,
                                               elementType(result.getType()));
    if (failed(expected) || result.getType() != *expected)
      return owner->emitOpError("physical reduction result does not preserve source free axes")
             << "; source=" << source.getType() << "; result=" << result.getType();
  }
  return success();
}

} // namespace

FailureOr<Type> inferCollectiveResultType(Type source, ArrayRef<int64_t> reducedAxes,
                                         Type resultElement) {
  auto fragment = dyn_cast<FragmentType>(source);
  if (!fragment) return failure();
  llvm::SmallDenseSet<int64_t> reduced;
  for (int64_t axis : reducedAxes)
    if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()) ||
        !reduced.insert(axis).second)
      return failure();
  SmallVector<Attribute> shape, mappings;
  for (auto [axis, extent] : llvm::enumerate(fragment.getShape())) {
    if (reduced.contains(axis)) continue;
    shape.push_back(extent);
    auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(source.getContext(), mapping.getSourceId(),
        mapping.getSourceAxis(), mapping.getDimensionId(), mappings.size(),
        mapping.getDerived()));
  }
  if (shape.empty()) return resultElement;
  return Type(FragmentType::get(source.getContext(), resultElement,
      ArrayAttr::get(source.getContext(), shape), ArrayAttr::get(source.getContext(), mappings),
      fragment.getValidity(), fragment.getOwner()));
}

LogicalResult inferScalarCollectiveResultTypes(std::optional<Location> location,
    ValueRange sources, int64_t axis, bool scan, SmallVectorImpl<Type> &results) {
  if (sources.empty()) return emitOptionalError(location, "native collective requires a source");
  auto first = dyn_cast<FragmentType>(sources.front().getType());
  if (!first || axis < 0 || axis >= static_cast<int64_t>(first.getShape().size()))
    return emitOptionalError(location, "collective axis is outside its source fragment");
  SmallVector<int64_t, 1> axes;
  if (!scan) axes.push_back(axis);
  for (Value value : sources) {
    auto source = dyn_cast<FragmentType>(value.getType());
    if (!source || source.getShape() != first.getShape())
      return emitOptionalError(location, "native collective requires equal source fragment shapes");
    auto expected = inferCollectiveResultType(source, axes, source.getElementType());
    if (failed(expected)) return failure();
    results.push_back(*expected);
  }
  return success();
}

LogicalResult verifyScalarCollective(Operation *owner, ValueRange sources,
                                     ValueRange identities, ResultRange results,
                                     Region &combine, int64_t axis, bool scan) {
  unsigned count = sources.size();
  if (!count || identities.size() != count || results.size() != count ||
      !llvm::hasSingleElement(combine))
    return owner->emitOpError(
        "requires paired sources/identities and one scalar combine block");
  SmallVector<Type> expected;
  if (failed(inferScalarCollectiveResultTypes(owner->getLoc(), sources, axis, scan, expected)))
    return failure();
  Block &block = combine.front();
  if (block.empty())
    return owner->emitOpError("scalar combine block must yield its results");
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (block.getNumArguments() != 2 * count || !yield ||
      yield.getValues().size() != count)
    return owner->emitOpError("scalar combine arity disagrees with sources");
  for (unsigned i = 0; i < count; ++i) {
    auto source = dyn_cast<FragmentType>(sources[i].getType());
    if (identities[i].getType() != results[i].getType())
      return owner->emitOpError(
          "native collective requires equal source shapes and exact identity/result types");
    Type element = source.getElementType();
    if (block.getArgument(i).getType() != element ||
        block.getArgument(count + i).getType() != element ||
        yield.getValues()[i].getType() != element)
      return owner->emitOpError(
          "callback arguments and yields must be source element types");
    if (results[i].getType() != expected[i])
      return owner->emitOpError(
          "collective result must preserve its source free-axis relation");
  }
  for (Operation &operation : block) {
    if (operation.getNumRegions())
      return owner->emitOpError(
          "native callback must be a closed elementwise block");
    for (Value result : operation.getResults())
      if (isa<FragmentType>(result.getType()))
        return owner->emitOpError(
            "native callback still contains a fragment value");
  }
  return success();
}

LogicalResult ReduceOp::inferReturnTypes(MLIRContext *context,
    std::optional<Location> location, ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions, SmallVectorImpl<Type> &results) {
  Adaptor adaptor(operands, attributes, properties, regions);
  if (adaptor.getSources().empty() || adaptor.getSources().size() != adaptor.getIdentities().size())
    return failure();
  for (auto [source, identity] : llvm::zip_equal(adaptor.getSources(), adaptor.getIdentities())) {
    auto type = inferCollectiveResultType(source.getType(), adaptor.getAxes(), elementType(identity.getType()));
    if (failed(type)) return failure();
    results.push_back(*type);
  }
  return success();
}

LogicalResult ScanOp::inferReturnTypes(MLIRContext *context,
    std::optional<Location> location, ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions, SmallVectorImpl<Type> &results) {
  Adaptor adaptor(operands, attributes, properties, regions);
  if (adaptor.getSources().empty() || adaptor.getSources().size() != adaptor.getIdentities().size())
    return failure();
  llvm::append_range(results, adaptor.getSources().getTypes());
  return success();
}

LogicalResult ReduceOp::verify() {
  auto structured = cast<StructuredOpInterface>(getOperation());
  if (failed(verifyStructuredArity(structured))) return failure();
  if (getAxes().empty()) return emitOpError("physical reduce requires at least one axis");
  llvm::DenseSet<int64_t> axes;
  for (int64_t axis : getAxes()) {
    if (!axes.insert(axis).second) return emitOpError("physical reduce axes must be unique");
    for (Value source : getSources())
      if (!typeCarriesAxis(source.getType(), axis))
        return emitOpError("physical reduce axis is outside a source schema");
  }
  if (failed(verifyReductionResultRelations(getOperation(), getSources(), getResults(), getAxes())))
    return failure();
  return verifyReduceLike(structured);
}

LogicalResult ScanOp::verify() {
  auto structured = cast<StructuredOpInterface>(getOperation());
  if (failed(verifyStructuredArity(structured))) return failure();
  for (auto [source, result] : llvm::zip_equal(getSources(), getResults())) {
    if (source.getType() != result.getType())
      return emitOpError("physical scan result must preserve its source type");
    if (!typeCarriesAxis(source.getType(), getAxis()))
      return emitOpError("physical scan axis is outside a source schema");
  }
  return verifyReduceLike(structured);
}

LogicalResult RegionFoldOp::verify() {
  auto structured = cast<StructuredOpInterface>(getOperation());
  if (failed(verifyStructuredArity(structured))) return failure();
  SmallVector<Type> summaries;
  for (auto [identity, result] : llvm::zip_equal(getIdentities(), getResults())) {
    if (identity.getType() != result.getType())
      return emitOpError("region-fold identity/result type disagrees: identity=")
             << identity.getType() << ", result=" << result.getType();
    summaries.push_back(identity.getType());
  }
  SmallVector<Type> slices;
  for (auto [source, slice] : llvm::zip_equal(getSources(), structured.getSummarizeSources())) {
    if (failed(verifySegmentSlice(getOperation(), source.getType(), slice.getType(), getAxis(), getSegment())))
      return failure();
    slices.push_back(slice.getType());
  }
  if (failed(verifyHelperRegion(getOperation(), getSummarize(),
                                structured.getSummarizeArgumentTypes(slices), summaries))) return failure();
  return verifyHelperRegion(getOperation(), getCombine(),
                            structured.getCombineArgumentTypes(), summaries);
}

LogicalResult RegionScanOp::verify() {
  auto structured = cast<StructuredOpInterface>(getOperation());
  if (failed(verifyStructuredArity(structured))) return failure();
  SmallVector<Type> slices, transitions, states;
  for (auto [source, slice] : llvm::zip_equal(getSources(), structured.getSummarizeSources())) {
    if (failed(verifySegmentSlice(getOperation(), source.getType(), slice.getType(), getAxis(), getSegment())))
      return failure();
    slices.push_back(slice.getType());
  }
  for (Value identity : getIdentities()) transitions.push_back(identity.getType());
  for (auto [initial, result] : llvm::zip_equal(getInitialStates(), getFinalStates())) {
    if (initial.getType() != result.getType())
      return emitOpError("region-scan final-state type disagrees: input=")
             << initial.getType() << ", result=" << result.getType();
    states.push_back(initial.getType());
  }
  if (failed(verifyHelperRegion(getOperation(), getSummarize(),
                                structured.getSummarizeArgumentTypes(slices), transitions))) return failure();
  if (failed(verifyHelperRegion(getOperation(), getCombine(),
                                structured.getCombineArgumentTypes(), transitions))) return failure();
  if (failed(verifyHelperRegion(getOperation(), getApply(),
                                structured.getApplyArgumentTypes(), states))) return failure();
  SmallVector<Type> emitted(structured.getEmitYields().getTypes());
  if (failed(verifyHelperRegion(getOperation(), getEmit(),
                                structured.getEmitArgumentTypes(slices), emitted))) return failure();
  for (auto [slice, result] : llvm::zip_equal(emitted, getEmittedResults()))
    if (!preservesSliceAssembly(slice, result.getType()))
      return emitOpError("region-scan output assembly relation is invalid");
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
  auto samePhysicalAxis = [](FragmentType left, unsigned leftAxis,
                             FragmentType right, unsigned rightAxis) {
    return left.getShape()[leftAxis] == right.getShape()[rightAxis];
  };
  auto constantExtent = [](FragmentType value,
                           unsigned axis) -> std::optional<int64_t> {
    auto extent = cast<PhysicalExprAttr>(value.getShape()[axis]);
    return extent.getKind() ==
                   PhysicalExprKind::Constant
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
      !samePhysicalAxis(lhs, 0, lhsScale, 0) ||
      !sameLogicalAxis(lhs, 1, lhsScale, 1) ||
      !samePhysicalAxis(lhs, 1, lhsScale, 1) ||
      !sameLogicalAxis(lhs, 1, rhs, 0) ||
      !samePhysicalAxis(lhs, 1, rhs, 0) ||
      !sameLogicalAxis(lhs, 1, rhsScale, 1) ||
      !samePhysicalAxis(lhs, 1, rhsScale, 1) ||
      !sameLogicalAxis(rhs, 2, rhsScale, 0) ||
      !samePhysicalAxis(rhs, 2, rhsScale, 0) ||
      !sameLogicalAxis(lhs, 0, result, 0) ||
      !samePhysicalAxis(lhs, 0, result, 0) ||
      !sameLogicalAxis(rhs, 2, result, 1) ||
      !samePhysicalAxis(rhs, 2, result, 1))
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
      !isa<IntegerType>(getResult().getType().getElementType()) ||
      getValues().getType().getShape() != getValid().getType().getShape() ||
      getResult().getType().getShape().size() != 1)
    return emitOpError("histogram physical schema is invalid");
  return success();
}

LogicalResult ScatterReduceOp::verify() {
  if (failed(verifyWritableResource(getOperation(), getResource().getType())) ||
      failed(verifyResourceSharing(getOperation(), getResource().getType(),
                                   getSharing())))
    return failure();
  SmallVector<Type> arguments{getValue().getType(), getValue().getType()};
  SmallVector<Type> results{getValue().getType()};
  return verifyHelperRegion(getOperation(), getCombine(), arguments, results);
}

void ScatterReduceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

namespace {

LogicalResult verifyAtomicSemantics(AccessOpInterface access,
                                    AtomicOrdering ordering,
                                    AtomicSharingDomain sharing) {
  Operation *owner = access;
  Type resource = access.getAccessResource().getType();
  bool orderingLegal =
      (access.getAccessKind() == AccessKind::AtomicLoad &&
       (ordering == AtomicOrdering::Relaxed ||
        ordering == AtomicOrdering::Acquire)) ||
      (access.getAccessKind() == AccessKind::AtomicStore &&
       (ordering == AtomicOrdering::Relaxed ||
        ordering == AtomicOrdering::Release)) ||
      access.getAccessKind() == AccessKind::AtomicRMW ||
      access.getAccessKind() == AccessKind::AtomicCompareExchange;
  if (!orderingLegal)
    return owner->emitOpError(
        "atomic ordering is illegal for this physical operation");
  if (failed(verifyResourceSharing(owner, resource, sharing)))
    return failure();
  if (auto view = dyn_cast<ViewType>(resource); view && view.getAccess() != 2)
    return owner->emitOpError(
        "external atomic target must use InOut access semantics");
  return success();
}

} // namespace

LogicalResult AtomicLoadOp::verify() {
  return verifyAtomicSemantics(cast<AccessOpInterface>(getOperation()),
                               getOrdering(), getSharing());
}

LogicalResult AtomicStoreOp::verify() {
  return verifyAtomicSemantics(cast<AccessOpInterface>(getOperation()),
                               getOrdering(), getSharing());
}

LogicalResult AtomicRMWOp::verify() {
  if (getResult().getType() != getValue().getType())
    return emitOpError("atomic RMW physical value/kind schema is invalid")
           << "; resource element="
           << resourceElementType(getResource().getType())
           << ", value=" << getValue().getType()
           << ", result=" << getResult().getType()
           << ", kind=" << stringifyAtomicRMWKind(getKind());
  return verifyAtomicSemantics(cast<AccessOpInterface>(getOperation()),
                               getOrdering(), getSharing());
}

LogicalResult AtomicCompareExchangeOp::verify() {
  auto result = getResult().getType();
  if (getExpected().getType() != getDesired().getType() ||
      result.getFieldTypes().size() != 2 ||
      cast<TypeAttr>(result.getFieldTypes()[0]).getValue() !=
          getExpected().getType() ||
      !elementType(cast<TypeAttr>(result.getFieldTypes()[1]).getValue())
           .isInteger(1))
    return emitOpError("compare-exchange physical result schema is invalid");
  return verifyAtomicSemantics(cast<AccessOpInterface>(getOperation()),
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
  if (type.isInvocationWorkspace())
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
  if (Value initialValue = getInitialValue())
    if (auto initial = dyn_cast<FragmentType>(initialValue.getType());
        initial && (initial.getShape() != type.getShape() ||
                    initial.getOwner() != type.getOwner()))
      return emitOpError(
          "buffer full-value initializer must cover its physical shape and owner")
          << "; initializer=" << initial << "; buffer=" << type;
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
