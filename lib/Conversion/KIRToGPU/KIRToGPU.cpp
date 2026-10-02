#include "Intent/Dialect/GPU/Transforms/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/ExecutionGroups.h"
#include "Intent/Conversion/KIRToGPU/KIRToGPU.h"

#include "Intent/Analysis/CanonicalKernel.h"
#include "Intent/Interfaces/StructuredOpInterface.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Transforms/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/Intent/IR/IntentAttrs.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <functional>

using namespace mlir;

namespace intent {
namespace {

using gpu::AxisMapAttr;
using gpu::FragmentType;
using gpu::PhysicalExprAttr;
using gpu::PhysicalExprKind;

DenseI64ArrayAttr dimensionIds(RankedTensorType tensor) {
  auto shape = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
  return shape ? shape.getDimensions() : DenseI64ArrayAttr();
}

RankedTensorType viewTensor(Value value) {
  auto view = dyn_cast<ViewType>(value.getType());
  return view ? dyn_cast<RankedTensorType>(view.getTensor()) : RankedTensorType();
}

bool isCanonicalEffect(Operation *operation) {
  if (auto buffer = dyn_cast<BufferOp>(operation))
    return static_cast<bool>(buffer.getInitial());
  return isa<ViewStoreOp, BufferStoreOp, ScatterUniqueOp, ScatterReduceOp,
             AtomicStoreOp, AtomicRMWOp, AtomicCompareExchangeOp>(operation);
}

ArrayAttr observableEffectOrigins(func::FuncOp function) {
  SmallVector<Attribute> origins;
  function.walk([&](Operation *operation) {
    if (!isCanonicalEffect(operation))
      return;
    if (auto node = operation->getAttrOfType<IntegerAttr>("intent.node"))
      origins.push_back(node);
  });
  return ArrayAttr::get(function.getContext(), origins);
}

PhysicalExprAttr expression(MLIRContext *context, PhysicalExprKind kind,
                            int64_t value = 0, StringRef symbol = {},
                            ArrayRef<Attribute> operands = {}) {
  return PhysicalExprAttr::get(
      context, kind, value,
      kind == PhysicalExprKind::Parameter
          ? Attribute(gpu::ParameterRefAttr::get(context, StringAttr::get(context, symbol)))
          : Attribute(StringAttr::get(context, symbol)),
      ArrayAttr::get(context, operands));
}

PhysicalExprAttr parameterExpression(MLIRContext *context, StringRef name) {
  return expression(context, PhysicalExprKind::Parameter, 0, name);
}

PhysicalExprAttr dimensionExpression(func::FuncOp function, int64_t dimension) {
  return PhysicalExprAttr::get(function.getContext(), PhysicalExprKind::Dimension,
      dimension, gpu::ArgumentRefAttr::get(function.getContext(),
          function.getNumArguments() + dimension),
      ArrayAttr::get(function.getContext(), {}));
}

PhysicalExprAttr binaryExpression(MLIRContext *context, PhysicalExprKind kind,
                                  PhysicalExprAttr lhs,
                                  PhysicalExprAttr rhs) {
  auto leftKind = lhs.getKind();
  auto rightKind = rhs.getKind();
  bool leftConstant = leftKind == PhysicalExprKind::Constant;
  bool rightConstant = rightKind == PhysicalExprKind::Constant;
  if (kind == PhysicalExprKind::Multiply && leftConstant && rightConstant)
    return expression(context, PhysicalExprKind::Constant,
                      lhs.getValue() * rhs.getValue());
  if ((kind == PhysicalExprKind::Add ||
       kind == PhysicalExprKind::Subtract) &&
      rightConstant && rhs.getValue() == 0)
    return lhs;
  if (kind == PhysicalExprKind::Add && leftConstant && lhs.getValue() == 0)
    return rhs;
  if (kind == PhysicalExprKind::Multiply) {
    if ((leftConstant && lhs.getValue() == 0) ||
        (rightConstant && rhs.getValue() == 0))
      return expression(context, PhysicalExprKind::Constant, 0);
    if (leftConstant && lhs.getValue() == 1)
      return rhs;
    if (rightConstant && rhs.getValue() == 1)
      return lhs;
  }
  if ((kind == PhysicalExprKind::CeilDiv ||
       kind == PhysicalExprKind::FloorDiv) &&
      rightConstant && rhs.getValue() == 1)
    return lhs;
  return expression(context, kind, 0, {}, {lhs, rhs});
}

struct PhysicalAxisIdentity {
  uint64_t sourceId;
  uint32_t sourceAxis;
  int64_t dimensionId;
  bool derived = false;
};

FragmentType fragmentType(MLIRContext *context, Type element,
                          ArrayRef<PhysicalExprAttr> shape,
                          ArrayRef<PhysicalAxisIdentity> axes,
                          uint64_t owner = 1) {
  SmallVector<Attribute> extents(shape.begin(), shape.end());
  SmallVector<Attribute> mappings;
  for (auto [fragmentAxis, mapping] : llvm::enumerate(axes))
    mappings.push_back(AxisMapAttr::get(
        context, mapping.sourceId, mapping.sourceAxis, mapping.dimensionId,
        fragmentAxis, mapping.derived));
  return FragmentType::get(context, element, ArrayAttr::get(context, extents),
                           ArrayAttr::get(context, mappings), 1, owner);
}

Value createBinary(OpBuilder &builder, Location location, Type result, Value lhs,
                   Value rhs, BinaryOperator kind) {
  auto left = lhs.getDefiningOp<arith::ConstantOp>();
  auto right = rhs.getDefiningOp<arith::ConstantOp>();
  auto leftInteger = left ? dyn_cast<IntegerAttr>(left.getValue()) : IntegerAttr();
  auto rightInteger = right ? dyn_cast<IntegerAttr>(right.getValue()) : IntegerAttr();
  if (leftInteger && rightInteger) {
    int64_t l = leftInteger.getInt();
    int64_t r = rightInteger.getInt();
    std::optional<int64_t> folded;
    if (kind == BinaryOperator::Add)
      folded = l + r;
    else if (kind == BinaryOperator::Subtract)
      folded = l - r;
    else if (kind == BinaryOperator::Multiply)
      folded = l * r;
    else if (kind == BinaryOperator::FloorDivide && r != 0)
      folded = l / r;
    else if (kind == BinaryOperator::LogicalAnd)
      folded = l && r;
    if (folded)
      return builder.create<arith::ConstantOp>(
          location, result, builder.getIntegerAttr(result, *folded));
  }
  return builder.create<gpu::BinaryOp>(location, result, lhs, rhs, kind);
}

bool samePhysicalShape(gpu::FragmentType lhs, gpu::FragmentType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getAxisMaps() == rhs.getAxisMaps() &&
         lhs.getValidity() == rhs.getValidity() &&
         lhs.getOwner() == rhs.getOwner();
}

FailureOr<Value> retargetBroadcast(OpBuilder &builder, Location location,
                                   Value value, gpu::FragmentType target) {
  auto source = dyn_cast<gpu::FragmentType>(value.getType());
  if (!source)
    return failure();
  if (samePhysicalShape(source, target))
    return value;
  auto result = gpu::FragmentType::get(
      value.getContext(), source.getElementType(), target.getShape(),
      target.getAxisMaps(), target.getValidity(), target.getOwner());
  Operation *replacement = nullptr;
  if (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>())
    replacement = builder.create<gpu::BroadcastOp>(location, result,
                                                    broadcast.getValue());
  else if (auto splat = value.getDefiningOp<gpu::SplatOp>())
    replacement = builder.create<gpu::SplatOp>(location, result,
                                                splat.getValue());
  else
    replacement = builder.create<gpu::BroadcastOp>(location, result, value);
  if (Operation *definition = value.getDefiningOp())
    if (Attribute origin = definition->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
  return replacement->getResult(0);
}

FailureOr<gpu::FragmentType>
mergeBroadcastExtents(gpu::FragmentType source,
                      gpu::FragmentType relationTarget) {
  gpu::BroadcastProjection projection =
      gpu::queryAxisProjection(source, relationTarget);
  if (!projection.isExact() ||
      source.getValidity() != relationTarget.getValidity() ||
      source.getOwner() != relationTarget.getOwner())
    return failure();
  SmallVector<Attribute> shape(relationTarget.getShape().begin(),
                               relationTarget.getShape().end());
  for (auto [targetAxis, sourceAxis] :
       llvm::enumerate(projection.targetToSource)) {
    if (!sourceAxis || source.getShape()[*sourceAxis] == shape[targetAxis])
      continue;
    auto sourceExtent = cast<gpu::PhysicalExprAttr>(
        source.getShape()[*sourceAxis]);
    auto targetExtent = cast<gpu::PhysicalExprAttr>(shape[targetAxis]);
    bool sourceUnit =
        sourceExtent.getKind() ==
            gpu::PhysicalExprKind::Constant &&
        sourceExtent.getValue() == 1;
    bool targetUnit =
        targetExtent.getKind() ==
            gpu::PhysicalExprKind::Constant &&
        targetExtent.getValue() == 1;
    if (sourceUnit)
      continue;
    if (!targetUnit)
      return failure();
    shape[targetAxis] = sourceExtent;
  }
  return gpu::FragmentType::get(
      relationTarget.getContext(), relationTarget.getElementType(),
      ArrayAttr::get(relationTarget.getContext(), shape),
      relationTarget.getAxisMaps(), relationTarget.getValidity(),
      relationTarget.getOwner());
}

bool carriesLogicalDimensions(gpu::FragmentType fragment,
                              RankedTensorType logical) {
  DenseI64ArrayAttr dimensions = dimensionIds(logical);
  if (!dimensions ||
      static_cast<size_t>(dimensions.size()) != fragment.getAxisMaps().size())
    return false;
  for (auto [axis, mapping] : llvm::enumerate(fragment.getAxisMaps()))
    if (cast<gpu::AxisMapAttr>(mapping).getDimensionId() != dimensions[axis])
      return false;
  return true;
}

FailureOr<Value> projectPositionalValue(OpBuilder &builder, Location location,
                                       Value identity, Type targetType) {
  if (identity.getType() == targetType)
    return identity;
  auto source = dyn_cast<gpu::FragmentType>(identity.getType());
  auto target = dyn_cast<gpu::FragmentType>(targetType);
  if (source && target && source.getElementType() == target.getElementType() &&
      source.getShape() == target.getShape() &&
      source.getOwner() == target.getOwner() &&
      source.getValidity() == target.getValidity()) {
    SmallVector<Attribute> groups;
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
      groups.push_back(gpu::ReshapeGroupAttr::get(
          builder.getContext(), builder.getDenseI64ArrayAttr({axis}),
          builder.getDenseI64ArrayAttr({axis})));
    return Value(builder.create<gpu::ReshapeOp>(
        location, target, identity, builder.getArrayAttr(groups)));
  }
  return gpu::projectPhysicalValueToSchema(builder, location, identity,
                                           targetType);
}

LogicalResult alignElementwiseOperands(OpBuilder &builder, Location location,
                                       Value &lhs, Value &rhs,
                                       Type logicalLhs = Type(),
                                       Type logicalRhs = Type()) {
  auto left = dyn_cast<gpu::FragmentType>(lhs.getType());
  auto right = dyn_cast<gpu::FragmentType>(rhs.getType());
  if (!left && right &&
      isa<IntegerType, FloatType, IndexType>(lhs.getType())) {
    auto target = gpu::FragmentType::get(
        lhs.getContext(), lhs.getType(), right.getShape(), right.getAxisMaps(),
        right.getValidity(), right.getOwner());
    lhs = builder.create<gpu::SplatOp>(location, target, lhs);
    left = target;
  }
  if (left && !right &&
      isa<IntegerType, FloatType, IndexType>(rhs.getType())) {
    auto target = gpu::FragmentType::get(
        rhs.getContext(), rhs.getType(), left.getShape(), left.getAxisMaps(),
        left.getValidity(), left.getOwner());
    rhs = builder.create<gpu::SplatOp>(location, target, rhs);
    right = target;
  }
  if (!left || !right || samePhysicalShape(left, right))
    return success();
  auto leftLogical = dyn_cast_or_null<RankedTensorType>(logicalLhs);
  auto rightLogical = dyn_cast_or_null<RankedTensorType>(logicalRhs);
  if (leftLogical && rightLogical &&
      dimensionIds(leftLogical) == dimensionIds(rightLogical) &&
      left.getOwner() == right.getOwner() &&
      left.getValidity() == right.getValidity()) {
    bool leftMatches = carriesLogicalDimensions(left, leftLogical);
    bool rightMatches = carriesLogicalDimensions(right, rightLogical);
    if (leftMatches != rightMatches &&
        gpu::queryAxisProjection(leftMatches ? right : left,
                                 leftMatches ? left : right).isExact()) {
      gpu::FragmentType relationTarget = leftMatches ? left : right;
      gpu::FragmentType extentSource = leftMatches ? right : left;
      FailureOr<gpu::FragmentType> target =
          mergeBroadcastExtents(extentSource, relationTarget);
      if (failed(target))
        return failure();
      FailureOr<Value> alignedLeft =
          retargetBroadcast(builder, location, lhs, *target);
      FailureOr<Value> alignedRight =
          retargetBroadcast(builder, location, rhs, *target);
      if (failed(alignedLeft) || failed(alignedRight))
        return failure();
      lhs = *alignedLeft;
      rhs = *alignedRight;
      return success();
    }
  }
  if (leftLogical && rightLogical &&
      leftLogical.getShape() == rightLogical.getShape() &&
      left.getShape().size() == static_cast<size_t>(leftLogical.getRank()) &&
      right.getShape().size() == static_cast<size_t>(rightLogical.getRank()) &&
      left.getShape() == right.getShape() &&
      left.getOwner() == right.getOwner() &&
      left.getValidity() == right.getValidity()) {
    bool sameDimensions = true;
    auto leftDimensions = dimensionIds(leftLogical);
    auto rightDimensions = dimensionIds(rightLogical);
    for (unsigned axis = 0; axis < left.getShape().size(); ++axis)
      if (leftLogical.isDynamicDim(axis))
        sameDimensions &= leftDimensions && rightDimensions &&
                          leftDimensions[axis] == rightDimensions[axis];
    if (sameDimensions) {
      SmallVector<Attribute> mappings;
      for (auto [leftAttribute, rightAttribute] :
           llvm::zip(left.getAxisMaps(), right.getAxisMaps())) {
        auto leftMapping = cast<gpu::AxisMapAttr>(leftAttribute);
        auto rightMapping = cast<gpu::AxisMapAttr>(rightAttribute);
        // A broadcast-derived identity must not replace a source coordinate.
        mappings.push_back(leftMapping.getDerived() && !rightMapping.getDerived()
                               ? rightAttribute
                               : leftAttribute);
      }
      auto target = gpu::FragmentType::get(
          lhs.getContext(), left.getElementType(), left.getShape(),
          builder.getArrayAttr(mappings), left.getValidity(), left.getOwner());
      auto rebind = [&](Value value) -> FailureOr<Value> {
        auto source = cast<gpu::FragmentType>(value.getType());
        auto rebound = gpu::FragmentType::get(
            value.getContext(), source.getElementType(), target.getShape(),
            target.getAxisMaps(), target.getValidity(), target.getOwner());
        if (source == rebound)
          return value;
        auto reassociation = gpu::inferReshapeReassociation(source, rebound);
        if (failed(reassociation))
          return failure();
        return Value(builder.create<gpu::ReshapeOp>(
            location, rebound, value, *reassociation));
      };
      // Canonical pointwise operands already refer to the same logical
      // positions. Preserve lane order while rebinding their provenance.
      FailureOr<Value> alignedLeft = rebind(lhs);
      FailureOr<Value> alignedRight = rebind(rhs);
      if (failed(alignedLeft) || failed(alignedRight))
        return failure();
      lhs = *alignedLeft;
      rhs = *alignedRight;
      return success();
    }
  }
  auto leftBroadcast = lhs.getDefiningOp<gpu::BroadcastOp>();
  auto rightBroadcast = rhs.getDefiningOp<gpu::BroadcastOp>();
  if (leftBroadcast && rightBroadcast &&
      left.getShape().size() == right.getShape().size()) {
    auto inheritedAxis = [](gpu::BroadcastOp broadcast,
                            gpu::FragmentType result,
                            unsigned resultAxis) {
      auto source = dyn_cast<gpu::FragmentType>(broadcast.getValue().getType());
      if (!source || source.getShape().size() > result.getShape().size())
        return false;
      unsigned offset = result.getShape().size() - source.getShape().size();
      if (resultAxis < offset)
        return false;
      unsigned sourceAxis = resultAxis - offset;
      auto sourceMapping =
          cast<gpu::AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
      auto resultMapping =
          cast<gpu::AxisMapAttr>(result.getAxisMaps()[resultAxis]);
      return source.getShape()[sourceAxis] == result.getShape()[resultAxis] &&
             sourceMapping.getSourceId() == resultMapping.getSourceId() &&
             sourceMapping.getSourceAxis() == resultMapping.getSourceAxis();
    };
    SmallVector<Attribute> shape(left.getShape().begin(), left.getShape().end());
    SmallVector<Attribute> mappings(left.getAxisMaps().begin(),
                                    left.getAxisMaps().end());
    for (unsigned axis = 0; axis < shape.size(); ++axis) {
      bool leftInherited = inheritedAxis(leftBroadcast, left, axis);
      bool rightInherited = inheritedAxis(rightBroadcast, right, axis);
      if (leftInherited && rightInherited) {
        auto leftMapping = cast<gpu::AxisMapAttr>(mappings[axis]);
        auto rightMapping =
            cast<gpu::AxisMapAttr>(right.getAxisMaps()[axis]);
        if (shape[axis] != right.getShape()[axis])
          return failure();
        if (leftMapping.getSourceId() != rightMapping.getSourceId() ||
            leftMapping.getSourceAxis() != rightMapping.getSourceAxis()) {
          auto extent = dyn_cast<gpu::PhysicalExprAttr>(shape[axis]);
          if (!extent ||
              extent.getKind() !=
                  gpu::PhysicalExprKind::Constant ||
              extent.getValue() != 1)
            return failure();
        }
        continue;
      }
      if (rightInherited) {
        shape[axis] = right.getShape()[axis];
        mappings[axis] = right.getAxisMaps()[axis];
        continue;
      }
      if (!leftInherited && shape[axis] != right.getShape()[axis])
        return failure();
    }
    auto leftTarget = gpu::FragmentType::get(
        lhs.getContext(), left.getElementType(), builder.getArrayAttr(shape),
        builder.getArrayAttr(mappings), left.getValidity(), left.getOwner());
    auto rightTarget = gpu::FragmentType::get(
        rhs.getContext(), right.getElementType(), builder.getArrayAttr(shape),
        builder.getArrayAttr(mappings), right.getValidity(), right.getOwner());
    FailureOr<Value> alignedLeft =
        retargetBroadcast(builder, location, lhs, leftTarget);
    FailureOr<Value> alignedRight =
        retargetBroadcast(builder, location, rhs, rightTarget);
    if (failed(alignedLeft) || failed(alignedRight))
      return failure();
    lhs = *alignedLeft;
    rhs = *alignedRight;
    return success();
  }
  // A common pointwise schema is selected by the typed broadcast relation,
  // not by which operand happens to have a BroadcastOp producer.  In
  // particular, a structured segment extent is authoritative over a
  // construction-time singleton even when the segmented value is the operand
  // being rematerialized.  Choosing the non-broadcast producer unconditionally
  // would replace that extent with one and construct an invalid BroadcastOp.
  bool leftToRight = gpu::queryBroadcastProjection(left, right).isExact();
  bool rightToLeft = gpu::queryBroadcastProjection(right, left).isExact();
  if (leftToRight || rightToLeft) {
    bool projectLeft = leftToRight && !rightToLeft;
    Value &projected = projectLeft ? lhs : rhs;
    gpu::FragmentType target = projectLeft ? right : left;
    FailureOr<Value> aligned =
        retargetBroadcast(builder, location, projected, target);
    if (failed(aligned))
      return failure();
    projected = *aligned;
    return success();
  }
  // Result-axis identities are local provenance for a value produced by a
  // canonical tensor operation.  Two same-rank pointwise operands can carry
  // different result identities for the same logical axis (for example, two
  // independently formed contraction results).  Align those corresponding
  // axes by position.  Source-coordinate identities are different: distinct
  // sources must survive as distinct physical axes, even when their extents
  // happen to be equal.
  auto isResultAxisIdentity = [](gpu::AxisMapAttr mapping) {
    return mapping.getDerived();
  };
  if (left.getShape() == right.getShape() &&
      left.getAxisMaps().size() == right.getAxisMaps().size() &&
      left.getOwner() == right.getOwner() &&
      left.getValidity() == right.getValidity()) {
    bool positionallyAligned = true;
    for (auto [leftAttribute, rightAttribute] :
         llvm::zip(left.getAxisMaps(), right.getAxisMaps())) {
      auto leftMapping = cast<gpu::AxisMapAttr>(leftAttribute);
      auto rightMapping = cast<gpu::AxisMapAttr>(rightAttribute);
      bool sameIdentity =
          leftMapping.getSourceId() == rightMapping.getSourceId() &&
          leftMapping.getSourceAxis() == rightMapping.getSourceAxis() &&
          leftMapping.getDerived() == rightMapping.getDerived();
      positionallyAligned &=
          sameIdentity ||
          (isResultAxisIdentity(leftMapping) &&
           isResultAxisIdentity(rightMapping));
    }
    if (positionallyAligned) {
      FailureOr<Value> aligned = retargetBroadcast(builder, location, rhs, left);
      if (failed(aligned))
        return failure();
      rhs = *aligned;
      return success();
    }
  }
  if (left.getOwner() != right.getOwner() ||
      left.getValidity() != right.getValidity())
    return failure();
  SmallVector<Attribute> shape(left.getShape().begin(), left.getShape().end());
  SmallVector<Attribute> mappings;
  for (Attribute attribute : left.getAxisMaps()) {
    auto mapping = cast<gpu::AxisMapAttr>(attribute);
    mappings.push_back(gpu::AxisMapAttr::get(
        lhs.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
  }
  for (auto [axis, attribute] : llvm::enumerate(right.getAxisMaps())) {
    auto mapping = cast<gpu::AxisMapAttr>(attribute);
    auto found = llvm::find_if(mappings, [&](Attribute existing) {
      auto current = cast<gpu::AxisMapAttr>(existing);
      return current.getSourceId() == mapping.getSourceId() &&
             current.getSourceAxis() == mapping.getSourceAxis() &&
             current.getDerived() == mapping.getDerived();
    });
    if (found != mappings.end()) {
      unsigned existingAxis = std::distance(mappings.begin(), found);
      if (shape[existingAxis] != right.getShape()[axis])
        return failure();
      continue;
    }
    shape.push_back(right.getShape()[axis]);
    mappings.push_back(gpu::AxisMapAttr::get(
        lhs.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
  }
  auto leftTarget = gpu::FragmentType::get(
      lhs.getContext(), left.getElementType(), builder.getArrayAttr(shape),
      builder.getArrayAttr(mappings), left.getValidity(), left.getOwner());
  auto rightTarget = gpu::FragmentType::get(
      rhs.getContext(), right.getElementType(), builder.getArrayAttr(shape),
      builder.getArrayAttr(mappings), right.getValidity(), right.getOwner());
  FailureOr<Value> alignedLeft =
      retargetBroadcast(builder, location, lhs, leftTarget);
  FailureOr<Value> alignedRight =
      retargetBroadcast(builder, location, rhs, rightTarget);
  if (failed(alignedLeft) || failed(alignedRight))
    return failure();
  lhs = *alignedLeft;
  rhs = *alignedRight;
  return success();
}

Value createCompare(OpBuilder &builder, Location location, Type result, Value lhs,
                    Value rhs, ComparePredicate predicate) {
  return builder.create<gpu::CompareOp>(location, result, lhs, rhs, predicate);
}

struct MetadataBinding {
  int64_t dimension;
  unsigned sourceABI;
  unsigned sourceAxis;
};

struct PhysicalABI {
  SmallVector<Type> arguments;
  SmallVector<DictionaryAttr> argumentAttrs;
  SmallVector<std::optional<unsigned>> physicalArgumentForSource;
  InterfaceAttr interface;
  llvm::DenseMap<int64_t, MetadataBinding> dimensions;
  SmallVector<int64_t> dimensionOrder;
};

FailureOr<PhysicalABI> buildPhysicalABI(func::FuncOp function,
                                        OpBuilder &builder) {
  PhysicalABI result;
  MLIRContext *context = function.getContext();
  Block &sourceEntry = function.getBody().front();

  auto interface = buildPublicInterface(function);
  if (failed(interface)) return failure();
  result.interface = *interface;
  // Source argument identities and logical dimension identities occupy disjoint
  // ranges. Neither is a physical slot; later appends allocate fresh identities.
  uint64_t nextReference = sourceEntry.getNumArguments() + 1;
  for (BlockArgument argument : sourceEntry.getArguments())
    if (auto tensor = viewTensor(argument))
      if (auto ids = dimensionIds(tensor))
        for (int64_t identity : ids.asArrayRef())
          nextReference = std::max(nextReference,
              uint64_t(sourceEntry.getNumArguments()) + identity + 1);
  SmallVector<std::pair<Type, DictionaryAttr>> strideArguments;
  auto bindingAttrs = [&](gpu::ArgumentBindingAttr binding) {
    return builder.getDictionaryAttr({builder.getNamedAttr(gpu::argumentBindingAttr, binding)});
  };
  unsigned publicOrdinal = 0;

  for (auto [abi, argument] : llvm::enumerate(sourceEntry.getArguments())) {
    auto parameter = getSourceParameter(argument);
    if (!parameter)
      return function.emitError("canonical argument has no source parameter binding");
    if (isa<ConstexprType>(argument.getType())) {
      if (!argument.use_empty())
        return function.emitError("constexpr parameter must be specialized before physical construction");
      result.physicalArgumentForSource.push_back(std::nullopt);
      continue;
    }
    result.physicalArgumentForSource.push_back(result.arguments.size());
    auto reference = gpu::ArgumentRefAttr::get(context, abi + 1);
    auto binding = gpu::ArgumentBindingAttr::get(context, reference,
        gpu::ArgumentKind::Public, builder.getI64IntegerAttr(publicOrdinal++),
        gpu::ArgumentRefAttr{}, IntegerAttr{}, IntegerAttr{});
    result.argumentAttrs.push_back(bindingAttrs(binding));
    auto logicalView = dyn_cast<ViewType>(argument.getType());
    if (!logicalView) {
      Type physicalType = argument.getType();
      if (!isa<IntegerType, IndexType, FloatType>(physicalType))
        return function.emitError("runtime scalar ABI parameter has a non-scalar type");
      result.arguments.push_back(physicalType);
      continue;
    }
    auto tensor = dyn_cast<RankedTensorType>(logicalView.getTensor());
    DenseI64ArrayAttr canonicalIds =
        tensor ? dimensionIds(tensor) : DenseI64ArrayAttr();
    if (!tensor || parameter.getOriginId() < 0 ||
        (canonicalIds && canonicalIds.size() != tensor.getRank()))
      return function.emitError(
          "view lost canonical source or dimension identities");
    uint64_t sourceId = static_cast<uint64_t>(parameter.getOriginId()) + 1;
    SmallVector<int64_t> ids(tensor.getRank(), 0);
    if (canonicalIds)
      llvm::copy(canonicalIds.asArrayRef(), ids.begin());
    for (unsigned axis = 0; axis < static_cast<unsigned>(tensor.getRank()); ++axis)
      if (tensor.isDynamicDim(axis) && ids[axis] <= 0)
        return function.emitError(
            "dynamic view axis lost its canonical dimension identity");
    SmallVector<Attribute> strides;
    SmallVector<Attribute> extents;
    for (unsigned axis = 0; axis < static_cast<unsigned>(tensor.getRank()); ++axis) {
      auto strideReference = gpu::ArgumentRefAttr::get(context, nextReference++);
      strides.push_back(PhysicalExprAttr::get(context, PhysicalExprKind::ScalarABI,
          0, strideReference, builder.getArrayAttr({})));
      strideArguments.emplace_back(builder.getIndexType(), bindingAttrs(
          gpu::ArgumentBindingAttr::get(context, strideReference, gpu::ArgumentKind::Stride,
              IntegerAttr{}, reference, builder.getI64IntegerAttr(axis), IntegerAttr{})));
      int64_t dimension = ids[axis];
      extents.push_back(
          tensor.isDynamicDim(axis)
              ? Attribute(dimensionExpression(function, dimension))
              : Attribute(expression(context, PhysicalExprKind::Constant,
                                     tensor.getDimSize(axis))));
      if (dimension > 0) {
        auto [binding, inserted] = result.dimensions.try_emplace(
            dimension,
            MetadataBinding{dimension, static_cast<unsigned>(abi), axis});
        // Automatic output allocation must derive shared dimensions from an
        // input, including inputs declared after the output in the source ABI.
        if (!inserted && logicalView.getAccess() != 1 &&
            cast<ViewType>(sourceEntry.getArgument(binding->second.sourceABI)
                               .getType()).getAccess() == 1) {
          binding->second.sourceABI = abi;
          binding->second.sourceAxis = axis;
        }
      }
    }
    auto layout = gpu::ViewLayoutAttr::get(
        context, ArrayAttr::get(context, extents),
        DenseI64ArrayAttr::get(context, ids),
        ArrayAttr::get(context, strides));
    result.arguments.push_back(gpu::ViewType::get(
        context, tensor.getElementType(), logicalView.getAccess(),
        sourceId, layout));
  }

  for (auto &entry : result.dimensions)
    result.dimensionOrder.push_back(entry.first);
  llvm::sort(result.dimensionOrder);
  for (int64_t dimension : result.dimensionOrder) {
    MetadataBinding &binding = result.dimensions.find(dimension)->second;
    result.arguments.push_back(builder.getIndexType());
    result.argumentAttrs.push_back(bindingAttrs(gpu::ArgumentBindingAttr::get(
        context, dimensionExpression(function, dimension).getArgumentReference(),
        gpu::ArgumentKind::Dimension, IntegerAttr{},
        gpu::ArgumentRefAttr::get(context, binding.sourceABI + 1),
        builder.getI64IntegerAttr(binding.sourceAxis), builder.getI64IntegerAttr(dimension))));
  }
  for (auto [type, attributes] : strideArguments) {
    result.arguments.push_back(type);
    result.argumentAttrs.push_back(attributes);
  }
  return result;
}

FailureOr<PhysicalExprAttr> launchExpression(Value value,
                                             func::FuncOp function = {}) {
  MLIRContext *context = value.getContext();
  Operation *definition = value.getDefiningOp();
  if (!definition) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || !function || argument.getOwner() != &function.getBody().front())
      return failure();
    if (!getSourceParameter(argument)) return failure();
    Type type = value.getType();
    if (!isa<IntegerType, IndexType>(type))
      return failure();
    return PhysicalExprAttr::get(context, PhysicalExprKind::ScalarABI, 0,
        gpu::ArgumentRefAttr::get(context, argument.getArgNumber() + 1),
        ArrayAttr::get(context, {}));
  }
  if (!definition)
    return failure();
  if (auto constant = dyn_cast<intent::ConstantOp>(definition)) {
    auto integer = dyn_cast<IntegerAttr>(constant.getValue());
    if (!integer)
      return failure();
    return expression(context, PhysicalExprKind::Constant, integer.getInt());
  }
  if (auto dim = dyn_cast<intent::DimOp>(definition)) {
    RankedTensorType tensor = viewTensor(dim.getSource());
    if (tensor && dim.getAxis() < static_cast<uint64_t>(tensor.getRank()) &&
        !tensor.isDynamicDim(dim.getAxis()))
      return expression(context, PhysicalExprKind::Constant,
                        tensor.getDimSize(dim.getAxis()));
    if (tensor && dim.getDimension() > 0)
      return dimensionExpression(function, dim.getDimension());
    if (auto domain = dim.getSource().getDefiningOp<intent::DomainOp>()) {
      FailureOr<PhysicalExprAttr> start =
          launchExpression(domain.getBounds()[0], function);
      FailureOr<PhysicalExprAttr> stop =
          launchExpression(domain.getBounds()[1], function);
      FailureOr<PhysicalExprAttr> step =
          domain.getBounds().size() == 3
              ? launchExpression(domain.getBounds()[2], function)
              : FailureOr<PhysicalExprAttr>(
                    expression(context, PhysicalExprKind::Constant, 1));
      if (failed(start) || failed(stop) || failed(step))
        return failure();
      if (start->getKind() ==
              PhysicalExprKind::Constant &&
          start->getValue() == 0 &&
          step->getKind() ==
              PhysicalExprKind::Constant &&
          step->getValue() == 1)
        return *stop;
      PhysicalExprAttr distance =
          binaryExpression(context, PhysicalExprKind::Subtract, *stop, *start);
      return binaryExpression(context, PhysicalExprKind::CeilDiv, distance,
                              *step);
    }
    return failure();
  }
  if (auto binary = dyn_cast<intent::BinaryOp>(definition)) {
    FailureOr<PhysicalExprAttr> lhs = launchExpression(binary.getLhs(), function);
    FailureOr<PhysicalExprAttr> rhs = launchExpression(binary.getRhs(), function);
    if (failed(lhs) || failed(rhs))
      return failure();
    PhysicalExprKind kind;
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add:
      kind = PhysicalExprKind::Add;
      break;
    case BinaryOperator::Subtract:
      kind = PhysicalExprKind::Subtract;
      break;
    case BinaryOperator::Multiply:
      kind = PhysicalExprKind::Multiply;
      break;
    case BinaryOperator::FloorDivide:
      kind = PhysicalExprKind::FloorDiv;
      break;
    case BinaryOperator::Maximum:
    case BinaryOperator::MaximumNum:
      kind = PhysicalExprKind::Maximum;
      break;
    case BinaryOperator::Minimum:
    case BinaryOperator::MinimumNum:
      kind = PhysicalExprKind::Minimum;
      break;
    default:
      return failure();
    }
    return binaryExpression(context, kind, *lhs, *rhs);
  }
  return failure();
}

FailureOr<PhysicalExprAttr> logicalDimensionExpression(func::FuncOp function,
                                                      int64_t dimension) {
  PhysicalExprAttr identity =
      dimensionExpression(function, dimension);
  PhysicalExprAttr resolved;
  bool conflict = false;
  auto observe = [&](Value value) {
    FailureOr<PhysicalExprAttr> candidate = launchExpression(value, function);
    if (failed(candidate) || *candidate == identity)
      return;
    if (resolved && resolved != *candidate)
      conflict = true;
    else
      resolved = *candidate;
  };
  function.walk([&](Operation *operation) {
    if (auto dim = dyn_cast<intent::DimOp>(operation)) {
      if (dim.getDimension() == dimension)
        observe(dim.getResult());
      return;
    }
    ArrayAttr relations;
    if (auto reshape = dyn_cast<intent::ReshapeOp>(operation))
      relations = reshape.getShape().getAxes();
    else if (auto broadcast = dyn_cast<intent::BroadcastOp>(operation))
      relations = broadcast.getShape().getAxes();
    else if (auto full = dyn_cast<intent::FullOp>(operation))
      relations = full.getShape().getAxes();
    if (!relations)
      return;
    for (Attribute attribute : relations) {
      auto relation = cast<intent::ShapeExprAttr>(attribute);
      if (relation.getKind() == 1 && relation.getDimension() == dimension)
        observe(operation->getOperand(relation.getPayload()));
    }
  });
  if (conflict)
    return failure();
  return resolved ? resolved : identity;
}

std::optional<int64_t> sourceExtentDimension(Value source) {
  while (auto subregion = source.getDefiningOp<intent::SubregionOp>())
    source = subregion.getInputs().front();
  auto domain = source.getDefiningOp<intent::DomainOp>();
  if (!domain || domain.getExtentDimensions().size() != 1)
    return std::nullopt;
  int64_t identity =
      cast<IntegerAttr>(domain.getExtentDimensions()[0]).getInt();
  return identity > 0 ? std::optional<int64_t>(identity) : std::nullopt;
}

FailureOr<PhysicalAxisIdentity>
resultAxisIdentity(Operation *operation, unsigned resultIndex = 0,
                   unsigned axis = 0);

FailureOr<PhysicalAxisIdentity>
physicalAxisIdentity(Operation *origin, int64_t logicalIdentity,
                     unsigned logicalAxis) {
  if (!origin || logicalIdentity <= 0)
    return failure();
  std::optional<std::pair<uint64_t, uint32_t>> local;
  bool localConflict = false;
  for (Value operand : origin->getOperands()) {
    std::optional<int64_t> extentIdentity = sourceExtentDimension(operand);
    if (!extentIdentity || *extentIdentity != logicalIdentity)
      continue;
    std::optional<std::pair<uint64_t, uint32_t>> candidate;
    if (auto domain = dyn_cast<DomainType>(operand.getType())) {
      if (domain.getRank() == 1)
        candidate = std::make_pair(domain.getOriginId(), uint32_t{0});
    } else if (auto region = dyn_cast<RegionType>(operand.getType())) {
      if (region.getRank() == 1)
        candidate = std::make_pair(region.getSourceId(), uint32_t{0});
    }
    if (!candidate)
      continue;
    if (local && *local != *candidate)
      localConflict = true;
    else
      local = candidate;
  }
  if (localConflict)
    return failure();
  if (local)
    return PhysicalAxisIdentity{local->first, local->second, logicalIdentity,
                                /*derived=*/false};
  func::FuncOp function = origin->getParentOfType<func::FuncOp>();
  if (!function)
    return resultAxisIdentity(origin, /*resultIndex=*/0, logicalAxis);
  std::optional<std::pair<uint64_t, uint32_t>> physical;
  bool conflict = false;
  function.walk([&](intent::IndicesOp indices) {
    auto result = dyn_cast<RankedTensorType>(indices.getResult().getType());
    DenseI64ArrayAttr identities = result ? dimensionIds(result)
                                          : DenseI64ArrayAttr();
    if (!result || !identities)
      return;
    for (unsigned axis = 0; axis < identities.size(); ++axis) {
      if (identities[axis] != logicalIdentity)
        continue;
      std::optional<std::pair<uint64_t, uint32_t>> candidate;
      if (auto domain = dyn_cast<DomainType>(indices.getSource().getType()))
        candidate = std::make_pair(domain.getOriginId(), axis);
      else if (auto region = dyn_cast<RegionType>(indices.getSource().getType()))
        candidate = std::make_pair(region.getSourceId(), axis);
      if (!candidate)
        continue;
      if (physical && *physical != *candidate)
        conflict = true;
      else
        physical = candidate;
    }
  });
  if (conflict) {
    FailureOr<PhysicalAxisIdentity> identity =
        resultAxisIdentity(origin, /*resultIndex=*/0, logicalAxis);
    if (failed(identity))
      return failure();
    identity->dimensionId = logicalIdentity;
    return *identity;
  }
  if (physical)
    return PhysicalAxisIdentity{physical->first, physical->second,
                                logicalIdentity, /*derived=*/false};
  FailureOr<PhysicalAxisIdentity> identity =
      resultAxisIdentity(origin, /*resultIndex=*/0, logicalAxis);
  if (failed(identity))
    return failure();
  identity->dimensionId = logicalIdentity;
  return *identity;
}

std::optional<int64_t> integerConstant(Value value) {
  Attribute attribute;
  if (auto constant = value.getDefiningOp<intent::ConstantOp>())
    attribute = constant.getValue();
  else if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    attribute = constant.getValue();
  auto integer = dyn_cast_or_null<IntegerAttr>(attribute);
  return integer ? std::optional<int64_t>(integer.getInt()) : std::nullopt;
}

std::optional<int64_t> subregionStaticExtentBound(Operation *origin,
                                                  int64_t logicalIdentity) {
  if (!origin || logicalIdentity <= 0)
    return std::nullopt;
  func::FuncOp function = origin->getParentOfType<func::FuncOp>();
  if (!function)
    return std::nullopt;
  std::optional<int64_t> result;
  bool conflict = false;
  function.walk([&](intent::SubregionOp subregion) {
    bool definesIdentity = llvm::any_of(
        subregion.getExtentDimensions(), [&](Attribute attribute) {
          return cast<IntegerAttr>(attribute).getInt() == logicalIdentity;
        });
    if (!definesIdentity || !subregion.getHasStart() ||
        !subregion.getHasStop())
      return;
    Value start = subregion.getInputs()[1];
    Value stop = subregion.getInputs()[2];
    auto addedExtent = [&](Value candidate) -> std::optional<int64_t> {
      auto add = candidate.getDefiningOp<intent::BinaryOp>();
      if (!add || add.getOperatorKind() != BinaryOperator::Add)
        return std::nullopt;
      if (add.getLhs() == start)
        return integerConstant(add.getRhs());
      if (add.getRhs() == start)
        return integerConstant(add.getLhs());
      return std::nullopt;
    };
    std::optional<int64_t> bound = addedExtent(stop);
    if (!bound) {
      auto minimum = stop.getDefiningOp<intent::BinaryOp>();
      if (minimum &&
          (minimum.getOperatorKind() == BinaryOperator::Minimum ||
           minimum.getOperatorKind() == BinaryOperator::MinimumNum)) {
        bound = addedExtent(minimum.getLhs());
        if (!bound)
          bound = addedExtent(minimum.getRhs());
      }
    }
    if (!bound) {
      auto difference = stop.getDefiningOp<intent::BinaryOp>();
      if (difference &&
          difference.getOperatorKind() == BinaryOperator::Subtract) {
        Value base = difference.getRhs();
        auto minimum = difference.getLhs().getDefiningOp<intent::BinaryOp>();
        if (minimum &&
            (minimum.getOperatorKind() == BinaryOperator::Minimum ||
             minimum.getOperatorKind() == BinaryOperator::MinimumNum)) {
          auto distanceFromBase = [&](Value candidate)
              -> std::optional<int64_t> {
            auto add = candidate.getDefiningOp<intent::BinaryOp>();
            if (!add || add.getOperatorKind() != BinaryOperator::Add)
              return std::nullopt;
            if (add.getLhs() == base)
              return integerConstant(add.getRhs());
            if (add.getRhs() == base)
              return integerConstant(add.getLhs());
            return std::nullopt;
          };
          bound = distanceFromBase(minimum.getLhs());
          if (!bound)
            bound = distanceFromBase(minimum.getRhs());
        }
      }
    }
    if (!bound || *bound <= 0)
      return;
    if (result && *result != *bound)
      conflict = true;
    else
      result = *bound;
  });
  return conflict ? std::nullopt : result;
}

FailureOr<PhysicalExprAttr> fragmentExtentExpression(RankedTensorType tensor,
                                                     Operation *origin,
                                                     unsigned axis) {
  DenseI64ArrayAttr identities = dimensionIds(tensor);
  if (!origin || !identities || axis >= identities.size() || identities[axis] <= 0)
    return failure();
  if (std::optional<int64_t> staticBound =
          subregionStaticExtentBound(origin, identities[axis]))
    return expression(tensor.getContext(), PhysicalExprKind::Constant,
                      *staticBound);
  // Initial scalar ownership needs no choice between equal-length source
  // occurrences. Their own ranges retain the logical coordinates and bounds.
  return expression(origin->getContext(), PhysicalExprKind::Constant, 1);
}

Type convertScalarType(Type type, uint64_t owner = 1) {
  if (isa<IntegerType, FloatType, IndexType>(type))
    return type;
  if (isa<LogicalIndexType>(type))
    return IndexType::get(type.getContext());
  if (auto record = dyn_cast<intent::RecordType>(type))
    return gpu::RecordType::get(type.getContext(), record.getFieldNames(),
                                record.getFieldTypes(), owner);
  if (auto tuple = dyn_cast<intent::TupleType>(type)) {
    SmallVector<Attribute> names;
    for (unsigned index = 0; index < tuple.getComponentTypes().size(); ++index)
      names.push_back(StringAttr::get(type.getContext(),
                                     ("_" + Twine(index)).str()));
    return gpu::RecordType::get(type.getContext(),
                                ArrayAttr::get(type.getContext(), names),
                                tuple.getComponentTypes(), owner);
  }
  return {};
}

FailureOr<PhysicalAxisIdentity> resultAxisIdentity(Operation *operation,
                                                   unsigned resultIndex,
                                                   unsigned axis) {
  if (!operation || resultIndex >= operation->getNumResults())
    return failure();
  auto results = operation->getAttrOfType<ArrayAttr>("intent.result_nodes");
  if (!results || resultIndex >= results.size())
    return failure();
  auto value = dyn_cast<IntegerAttr>(results[resultIndex]);
  if (!value || value.getInt() < 0 || axis > std::numeric_limits<uint32_t>::max())
    return failure();
  return PhysicalAxisIdentity{static_cast<uint64_t>(value.getInt()) + 1,
                              static_cast<uint32_t>(axis),
                              /*dimensionId=*/0, /*derived=*/true};
}

FailureOr<FragmentType> convertTensorType(RankedTensorType tensor,
                                          Operation *origin,
                                          std::optional<FragmentType> prototype =
                                              std::nullopt,
                                          uint64_t owner = 1,
                                          unsigned resultIndex = 0) {
  MLIRContext *context = tensor.getContext();
  DenseI64ArrayAttr dimensions = dimensionIds(tensor);
  SmallVector<PhysicalExprAttr> shape;
  SmallVector<PhysicalAxisIdentity> mappings;
  for (unsigned axis = 0; axis < static_cast<unsigned>(tensor.getRank()); ++axis) {
    if (!dimensions || dimensions.size() != static_cast<unsigned>(tensor.getRank()) ||
        dimensions[axis] <= 0)
      return failure();
    if (tensor.isDynamicDim(axis)) {
      FailureOr<PhysicalExprAttr> extent =
          fragmentExtentExpression(tensor, origin, axis);
      if (failed(extent))
        return failure();
      shape.push_back(*extent);
      FailureOr<PhysicalAxisIdentity> mapping =
          physicalAxisIdentity(origin, dimensions[axis], axis);
      if (failed(mapping))
        return failure();
      mappings.push_back(*mapping);
    } else {
      shape.push_back(expression(context, PhysicalExprKind::Constant,
                                 tensor.getDimSize(axis)));
      FailureOr<PhysicalAxisIdentity> mapping =
          physicalAxisIdentity(origin, dimensions[axis], axis);
      if (failed(mapping))
        return failure();
      mappings.push_back(*mapping);
    }
  }
  if (prototype && prototype->getShape().size() == shape.size()) {
    bool sameExtents = true;
    for (auto [left, right] : llvm::zip(prototype->getShape(), shape))
      sameExtents &= left == right;
    if (sameExtents) {
      SmallVector<Attribute> extents(shape.begin(), shape.end());
      SmallVector<Attribute> remapped;
      for (auto [axis, attribute] :
           llvm::enumerate(prototype->getAxisMaps())) {
        auto mapping = cast<AxisMapAttr>(attribute);
        remapped.push_back(AxisMapAttr::get(
            context, mapping.getSourceId(), mapping.getSourceAxis(),
            dimensions[axis], axis, mapping.getDerived()));
      }
      return FragmentType::get(context, tensor.getElementType(),
                               ArrayAttr::get(context, extents),
                               ArrayAttr::get(context, remapped),
                               prototype->getValidity(),
                               prototype->getOwner());
    }
  }
  return fragmentType(context, tensor.getElementType(), shape, mappings, owner);
}

FailureOr<FragmentType> convertSegmentSliceType(RankedTensorType tensor,
                                                FragmentType prototype,
                                                unsigned axis,
                                                StringRef parameter) {
  if (axis >= prototype.getShape().size() ||
      tensor.getRank() != static_cast<int64_t>(prototype.getShape().size()) ||
      tensor.getElementType() != prototype.getElementType())
    return failure();
  SmallVector<Attribute> shape(prototype.getShape().begin(),
                               prototype.getShape().end());
  SmallVector<Attribute> mappings(prototype.getAxisMaps().begin(),
                                  prototype.getAxisMaps().end());
  DenseI64ArrayAttr dimensions = dimensionIds(tensor);
  if (!dimensions || dimensions.size() != tensor.getRank() ||
      dimensions[axis] <= 0)
    return failure();
  shape[axis] = parameterExpression(tensor.getContext(), parameter);
  // The explicit structured-region boundary creates a helper-local slice
  // dimension.  Keep the source component's non-segment provenance, but give
  // the sliced axis its canonical helper dimension identity.  Otherwise one
  // logical source used simultaneously as an outer ownership axis and an
  // inner segment axis would collapse to one physical extent authority.
  auto mapping = cast<AxisMapAttr>(prototype.getAxisMaps()[axis]);
  mappings[axis] = AxisMapAttr::get(
      tensor.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
      dimensions[axis], axis, mapping.getDerived());
  return FragmentType::get(tensor.getContext(), tensor.getElementType(),
                           ArrayAttr::get(tensor.getContext(), shape),
                           ArrayAttr::get(tensor.getContext(), mappings),
                           prototype.getValidity(),
                           prototype.getOwner());
}

FailureOr<Type> regionAssemblyType(Type result, Type slice) {
  if (auto resultFragment = dyn_cast<FragmentType>(result)) {
    auto sliceFragment = dyn_cast<FragmentType>(slice);
    if (!sliceFragment ||
        resultFragment.getElementType() != sliceFragment.getElementType() ||
        resultFragment.getShape().size() != sliceFragment.getShape().size() ||
        resultFragment.getOwner() != sliceFragment.getOwner())
      return failure();
    return Type(FragmentType::get(
        result.getContext(), resultFragment.getElementType(),
        resultFragment.getShape(), sliceFragment.getAxisMaps(),
        resultFragment.getValidity(), resultFragment.getOwner()));
  }
  auto resultRecord = dyn_cast<gpu::RecordType>(result);
  auto sliceRecord = dyn_cast<gpu::RecordType>(slice);
  if (!resultRecord || !sliceRecord ||
      resultRecord.getFieldNames() != sliceRecord.getFieldNames() ||
      resultRecord.getFieldTypes().size() !=
          sliceRecord.getFieldTypes().size() ||
      resultRecord.getOwner() != sliceRecord.getOwner())
    return result == slice ? FailureOr<Type>(result)
                           : FailureOr<Type>(failure());
  SmallVector<Attribute> fields;
  for (auto [whole, part] : llvm::zip(resultRecord.getFieldTypes(),
                                      sliceRecord.getFieldTypes())) {
    FailureOr<Type> field = regionAssemblyType(
        cast<TypeAttr>(whole).getValue(), cast<TypeAttr>(part).getValue());
    if (failed(field))
      return failure();
    fields.push_back(TypeAttr::get(*field));
  }
  return Type(gpu::RecordType::get(
      result.getContext(), resultRecord.getFieldNames(),
      ArrayAttr::get(result.getContext(), fields), resultRecord.getOwner()));
}

FailureOr<Type> convertDataType(Type type, Operation *origin,
                                std::optional<Type> prototype = std::nullopt,
                                uint64_t owner = 1,
                                unsigned resultIndex = 0) {
  if (isa<IntegerType, FloatType, IndexType, LogicalIndexType>(type)) {
    Type element = convertScalarType(type, owner);
    if (prototype)
      if (auto fragment = dyn_cast<FragmentType>(*prototype))
        return Type(FragmentType::get(
            type.getContext(), element, fragment.getShape(),
            fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner()));
    return element;
  }
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    std::optional<FragmentType> fragment;
    if (prototype)
      if (auto candidate = dyn_cast<FragmentType>(*prototype))
        fragment = candidate;
    FailureOr<FragmentType> converted =
        convertTensorType(tensor, origin, fragment, owner, resultIndex);
    if (failed(converted))
      return failure();
    return Type(*converted);
  }
  if (auto record = dyn_cast<intent::RecordType>(type)) {
    auto prototypeRecord =
        prototype ? dyn_cast<gpu::RecordType>(*prototype) : gpu::RecordType();
    if (prototypeRecord &&
        (prototypeRecord.getFieldNames() != record.getFieldNames() ||
         prototypeRecord.getFieldTypes().size() != record.getFieldTypes().size()))
      return failure();
    SmallVector<Attribute> fields;
    for (auto [index, field] : llvm::enumerate(record.getFieldTypes())) {
      std::optional<Type> fieldPrototype;
      if (prototypeRecord)
        fieldPrototype = cast<TypeAttr>(
                             prototypeRecord.getFieldTypes()[index])
                             .getValue();
      FailureOr<Type> converted =
          convertDataType(cast<TypeAttr>(field).getValue(), origin,
                          fieldPrototype,
                          prototypeRecord ? prototypeRecord.getOwner() : owner,
                          resultIndex);
      if (failed(converted))
        return failure();
      fields.push_back(TypeAttr::get(*converted));
    }
    return Type(gpu::RecordType::get(type.getContext(), record.getFieldNames(),
                                     ArrayAttr::get(type.getContext(), fields),
                                     prototypeRecord ? prototypeRecord.getOwner()
                                                     : owner));
  }
  if (auto tuple = dyn_cast<intent::TupleType>(type)) {
    auto prototypeRecord =
        prototype ? dyn_cast<gpu::RecordType>(*prototype) : gpu::RecordType();
    if (prototypeRecord &&
        prototypeRecord.getFieldTypes().size() !=
            tuple.getComponentTypes().size())
      return failure();
    SmallVector<Attribute> names;
    SmallVector<Attribute> fields;
    for (auto [index, field] : llvm::enumerate(tuple.getComponentTypes())) {
      names.push_back(StringAttr::get(type.getContext(),
                                     ("_" + Twine(index)).str()));
      std::optional<Type> fieldPrototype;
      if (prototypeRecord)
        fieldPrototype = cast<TypeAttr>(
                             prototypeRecord.getFieldTypes()[index])
                             .getValue();
      FailureOr<Type> converted =
          convertDataType(cast<TypeAttr>(field).getValue(), origin,
                          fieldPrototype,
                          prototypeRecord ? prototypeRecord.getOwner() : owner,
                          resultIndex);
      if (failed(converted))
        return failure();
      fields.push_back(TypeAttr::get(*converted));
    }
    return Type(gpu::RecordType::get(
        type.getContext(), ArrayAttr::get(type.getContext(), names),
        ArrayAttr::get(type.getContext(), fields),
        prototypeRecord ? prototypeRecord.getOwner() : owner));
  }
  return failure();
}

FailureOr<Type> convertReductionResultType(Type logical, Operation *origin,
                                           Value source,
                                           ArrayRef<int64_t> reducedAxes,
                                           unsigned resultIndex) {
  auto sourceType = dyn_cast<FragmentType>(source.getType());
  auto logicalType = dyn_cast<RankedTensorType>(logical);
  if (!sourceType || !logicalType)
    return convertDataType(logical, origin, std::nullopt, 1, resultIndex);
  if (sourceType.getShape().size() < reducedAxes.size())
    return failure();
  if (logicalType.getRank() !=
      static_cast<int64_t>(sourceType.getShape().size() - reducedAxes.size()))
    return failure();
  return gpu::inferCollectiveResultType(sourceType, reducedAxes,
                                        logicalType.getElementType());
}

FailureOr<Type> convertContractResultType(
    Type logical, Value lhs, Value rhs, ArrayRef<int64_t> lhsReduction,
    ArrayRef<int64_t> rhsReduction, ArrayRef<int64_t> lhsBatch,
    ArrayRef<int64_t> rhsBatch) {
  auto tensor = dyn_cast<RankedTensorType>(logical);
  auto left = dyn_cast<FragmentType>(lhs.getType());
  auto right = dyn_cast<FragmentType>(rhs.getType());
  if (!tensor || !left || !right || left.getOwner() != right.getOwner() ||
      lhsReduction.size() != rhsReduction.size() ||
      lhsBatch.size() != rhsBatch.size())
    return failure();

  SmallVector<bool> leftReduced(left.getShape().size(), false);
  SmallVector<bool> rightReduced(right.getShape().size(), false);
  SmallVector<bool> rightBatched(right.getShape().size(), false);
  for (auto [lhsAxis, rhsAxis] : llvm::zip(lhsReduction, rhsReduction)) {
    if (lhsAxis < 0 || rhsAxis < 0 ||
        lhsAxis >= static_cast<int64_t>(leftReduced.size()) ||
        rhsAxis >= static_cast<int64_t>(rightReduced.size()) ||
        leftReduced[lhsAxis] || rightReduced[rhsAxis])
      return failure();
    leftReduced[lhsAxis] = true;
    rightReduced[rhsAxis] = true;
  }
  for (auto [lhsAxis, rhsAxis] : llvm::zip(lhsBatch, rhsBatch)) {
    if (lhsAxis < 0 || rhsAxis < 0 ||
        lhsAxis >= static_cast<int64_t>(leftReduced.size()) ||
        rhsAxis >= static_cast<int64_t>(rightReduced.size()) ||
        leftReduced[lhsAxis] || rightReduced[rhsAxis] || rightBatched[rhsAxis])
      return failure();
    rightBatched[rhsAxis] = true;
  }

  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  auto appendAxis = [&](FragmentType source, unsigned axis) {
    unsigned resultAxis = shape.size();
    shape.push_back(source.getShape()[axis]);
    auto sourceMapping = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(
        tensor.getContext(), sourceMapping.getSourceId(),
        sourceMapping.getSourceAxis(), sourceMapping.getDimensionId(), resultAxis,
        sourceMapping.getDerived()));
  };
  for (unsigned axis = 0; axis < leftReduced.size(); ++axis)
    if (!leftReduced[axis])
      appendAxis(left, axis);
  for (unsigned axis = 0; axis < rightReduced.size(); ++axis)
    if (!rightReduced[axis] && !rightBatched[axis])
      appendAxis(right, axis);
  if (shape.size() != static_cast<unsigned>(tensor.getRank()))
    return failure();
  return Type(FragmentType::get(
      tensor.getContext(), tensor.getElementType(),
      ArrayAttr::get(tensor.getContext(), shape),
      ArrayAttr::get(tensor.getContext(), mappings), /*validity=*/1,
      left.getOwner()));
}

struct IterationAxis {
  Value start;
  Value stop;
  Value step;
  Value coordinatePrototype;
  Value source;
};

LogicalResult collectIterationAxes(
    Value source, SmallVectorImpl<IterationAxis> &axes);

class ScalarRegionLowering {
public:
  ScalarRegionLowering(OpBuilder &builder, llvm::DenseMap<Value, Value> values,
                       ArrayRef<Value> views,
                       llvm::DenseMap<int64_t, Value> dimensions,
                       llvm::DenseMap<StringAttr, Value> parameters,
                       CanonicalKernelAnalysis &canonicalAnalysis,
                       func::FuncOp physicalKernel)
      : builder(builder), values(std::move(values)), views(views),
        dimensions(std::move(dimensions)), parameters(std::move(parameters)),
        canonicalAnalysis(canonicalAnalysis), physicalKernel(physicalKernel) {}

  FailureOr<SmallVector<Value>> lowerBlock(Block &source) {
    for (Operation &operation : source) {
      if (auto yield = dyn_cast<intent::YieldOp>(operation)) {
        SmallVector<Value> results;
        for (Value value : yield.getInputs()) {
          FailureOr<Value> lowered = get(value);
          if (failed(lowered))
            return failure();
          results.push_back(*lowered);
        }
        return results;
      }
      if (isa<intent::ReturnOp>(operation))
        return SmallVector<Value>{};
      if (failed(lower(&operation)))
        return failure();
    }
    return SmallVector<Value>{};
  }

  FailureOr<SmallVector<Value>> lowerWorksetBlock(Block &source) {
    SmallVector<Operation *> ancestors;
    for (Operation *parent = source.getParentOp();
         parent && !isa<func::FuncOp>(parent); parent = parent->getParentOp())
      ancestors.push_back(parent);
    // Workset extraction must retain the lexical constraints used by its accesses.
    for (Operation *ancestor : llvm::reverse(ancestors))
      for (Operation &operation : *ancestor->getBlock()) {
        if (&operation == ancestor)
          break;
        if (isa<intent::AssumeInBoundsOp>(operation) &&
            failed(lower(&operation)))
          return failure();
      }
    return lowerBlock(source);
  }

  llvm::DenseMap<Value, Value> &mapping() { return values; }
  FailureOr<Value> lowerValue(Value source) { return get(source); }
  FailureOr<Value> lowerIndexValue(Value source) {
    FailureOr<Value> value = get(source);
    return succeeded(value) ? asIndex(source.getLoc(), *value)
                            : FailureOr<Value>(failure());
  }

private:
  FailureOr<Value> get(Value source) {
    auto found = values.find(source);
    if (found != values.end())
      return found->second;
    if (Operation *definition = source.getDefiningOp()) {
      if (failed(lower(definition)))
        return failure();
      found = values.find(source);
      if (found != values.end())
        return found->second;
    }
    return failure();
  }

  FailureOr<Type> elementwiseResultType(Type logical, Operation *origin,
                                        Value prototype) {
    auto fragment = dyn_cast<gpu::FragmentType>(prototype.getType());
    if (!fragment)
      return convertDataType(logical, origin, prototype.getType());
    Type element;
    if (auto tensor = dyn_cast<RankedTensorType>(logical))
      element = tensor.getElementType();
    else if (isa<IntegerType, FloatType, IndexType, LogicalIndexType>(logical))
      element = convertScalarType(logical, fragment.getOwner());
    else
      return failure();
    return Type(gpu::FragmentType::get(
        logical.getContext(), element, fragment.getShape(),
        fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner()));
  }

  void mapResults(Operation *source, Operation *target) {
    for (auto [from, to] : llvm::zip(source->getResults(), target->getResults()))
      values[from] = to;
    if (Attribute node = source->getAttr("intent.node"))
      target->setAttr(gpu::originAttr, node);
  }

  FailureOr<gpu::ParameterAttr>
  getOrCreateRegionSegment(Operation *operation) {
    RegionSegmentFact fact = canonicalAnalysis.regionSegment(operation);
    if (!fact.isExact())
      return operation->emitOpError(
          "region segmentation has no exact canonical source relation");
    std::string name =
        ("SEGMENT_N" + Twine(fact.operationIdentity) + "_D" +
         Twine(fact.dimensionIdentity))
            .str();
    Type sourceType = operation->getOperand(0).getType();
    if (auto tensor = dyn_cast<RankedTensorType>(sourceType))
      sourceType = tensor.getElementType();
    if (!isa<IntegerType, FloatType>(sourceType))
      return operation->emitOpError(
          "region segment has no scalar element type for its physical parameter");
    gpu::ParameterCategory category = gpu::ParameterCategory::Scan;
    if (auto fold = dyn_cast<intent::RegionFoldOp>(operation)) {
      bool contraction = false;
      fold.getSummarize().walk([&](Operation *nested) {
        contraction |= isa<intent::ContractOp, intent::ScaledContractOp,
                           intent::SparseContractOp>(nested);
      });
      category = contraction ? gpu::ParameterCategory::RegionContraction
                             : gpu::ParameterCategory::RegionReduction;
    }
    auto schema = gpu::ParameterAttr::get(
        operation->getContext(), builder.getStringAttr(name), builder.getIndexType(),
        gpu::ParameterRole::ScanChunk,
        category,
        sourceType.getIntOrFloatBitWidth(),
        builder.getDenseI64ArrayAttr(
            {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
             32768, 65536}),
        gpu::ConfigurationBindingPhase::Shared,
        gpu::ParameterBindingAttr::get(
            operation->getContext(), builder.getI64IntegerAttr(fact.dimensionIdentity),
            {}, {}, {}, false, false));
    func::FuncOp physical = physicalKernel;

    auto reference = gpu::declareParameter(physical, schema);
    if (failed(reference))
      return failure();
    if (!parameters.count(schema.getName())) {
      OpBuilder declarationBuilder(&physical.getBody().front(),
                                   physical.getBody().front().begin());
      auto read = gpu::materializeParameter(
          declarationBuilder, operation->getLoc(), *reference);
      read->setAttr(
          gpu::originAttr,
          declarationBuilder.getI64IntegerAttr(fact.operationIdentity));
      parameters[schema.getName()] = read.getResult();
    }
    return schema;
  }

  LogicalResult lowerPureRegion(Region &source, Region &target,
                                ArrayRef<Type> argumentTypes,
                                ArrayRef<Type> resultTypes = {}) {
    if (!target.empty())
      return failure();
    OpBuilder::InsertionGuard guard(builder);
    Block *block = builder.createBlock(
        &target, target.end(), argumentTypes,
        SmallVector<Location>(argumentTypes.size(),
                              source.getParentOp()->getLoc()));
    auto childValues = values;
    for (auto [from, to] :
         llvm::zip(source.front().getArguments(), block->getArguments()))
      childValues[from] = to;
    builder.setInsertionPointToStart(block);
    ScalarRegionLowering child(builder, std::move(childValues), views,
                               dimensions, parameters, canonicalAnalysis,
                               physicalKernel);
    FailureOr<SmallVector<Value>> yielded = child.lowerBlock(source.front());
    if (failed(yielded))
      return failure();
    if (!resultTypes.empty()) {
      if (yielded->size() != resultTypes.size())
        return failure();
      for (auto [index, type] : llvm::enumerate(resultTypes)) {
        FailureOr<Value> projected = projectPositionalValue(
            builder, source.getParentOp()->getLoc(), (*yielded)[index], type);
        if (failed(projected))
          return failure();
        (*yielded)[index] = *projected;
      }
    }
    builder.create<gpu::YieldOp>(source.getParentOp()->getLoc(), *yielded);
    return success();
  }

  void attachOrigin(Operation *source, Operation *target) {
    if (Attribute node = source->getAttr("intent.node"))
      target->setAttr(gpu::originAttr, node);
  }

  Value rangeExtent(Location location, Value start, Value stop, Value step) {
    if (integerConstant(start) == 0 && integerConstant(step) == 1)
      return stop;
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    Value distance = createBinary(builder, location, builder.getIndexType(), stop,
                                  start, BinaryOperator::Subtract);
    Value adjusted = createBinary(
        builder, location, builder.getIndexType(), distance,
        createBinary(builder, location, builder.getIndexType(), step, one,
                     BinaryOperator::Subtract),
        BinaryOperator::Add);
    return createBinary(builder, location, builder.getIndexType(), adjusted, step,
                        BinaryOperator::FloorDivide);
  }

  Value rangeBound(Location location, Value range, unsigned bound) {
    if (auto operation = range.getDefiningOp<gpu::RangeOp>()) {
      if (bound == 0)
        return operation.getStart();
      if (bound == 1)
        return operation.getStop();
      if (bound == 2)
        return operation.getStep();
    }
    return builder.create<gpu::RangeBoundOp>(location, builder.getIndexType(),
                                              range, bound);
  }

  FailureOr<Value> physicalExtentValue(Location location,
                                       PhysicalExprAttr expression) {
    auto kind = expression.getKind();
    if (kind == PhysicalExprKind::Constant)
      return Value(builder.create<arith::ConstantIndexOp>(location,
                                                           expression.getValue()));
    if (kind == PhysicalExprKind::Dimension) {
      int64_t identity = expression.getValue();
      if (identity <= 0)
        return failure();
      auto found = dimensions.find(identity);
      return found == dimensions.end() ? FailureOr<Value>(failure())
                                       : FailureOr<Value>(found->second);
    }
    if (kind == PhysicalExprKind::Parameter) {
      auto found = parameters.find(expression.getParameterReference().getName());
      return found == parameters.end() ? FailureOr<Value>(failure())
                                       : FailureOr<Value>(found->second);
    }
    func::FuncOp function = physicalKernel;
    if (kind == PhysicalExprKind::ScalarABI) {
      if (auto argument = gpu::resolveArgument(function, expression.getArgumentReference()))
        return Value(argument);
      return failure();
    }
    return Value(builder.create<gpu::PhysicalExprOp>(
        location, builder.getIndexType(), expression));
  }

  FailureOr<PhysicalExprAttr> physicalShapeExpression(Value value,
                                                      Operation *origin) {
    func::FuncOp function = origin->getParentOfType<func::FuncOp>();
    FailureOr<PhysicalExprAttr> launched = launchExpression(value, function);
    if (succeeded(launched))
      return *launched;
    Operation *definition = value.getDefiningOp();
    if (auto dim = dyn_cast_or_null<intent::DimOp>(definition)) {
      FailureOr<Value> lowered = get(dim.getSource());
      auto fragment = succeeded(lowered)
                          ? dyn_cast<gpu::FragmentType>((*lowered).getType())
                          : gpu::FragmentType();
      if (!fragment || dim.getDimension() <= 0)
        return failure();
      PhysicalExprAttr extent;
      for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps())) {
        auto mapping = dyn_cast<gpu::AxisMapAttr>(attribute);
        if (!mapping || mapping.getDimensionId() !=
                            static_cast<int64_t>(dim.getDimension()))
          continue;
        auto candidate =
            cast<PhysicalExprAttr>(fragment.getShape()[axis]);
        if (extent && extent != candidate)
          return failure();
        extent = candidate;
      }
      return extent ? FailureOr<PhysicalExprAttr>(extent)
                    : FailureOr<PhysicalExprAttr>(failure());
    }
    if (auto binary = dyn_cast_or_null<intent::BinaryOp>(definition)) {
      FailureOr<PhysicalExprAttr> lhs =
          physicalShapeExpression(binary.getLhs(), origin);
      FailureOr<PhysicalExprAttr> rhs =
          physicalShapeExpression(binary.getRhs(), origin);
      if (failed(lhs) || failed(rhs))
        return failure();
      PhysicalExprKind kind;
      switch (binary.getOperatorKind()) {
      case BinaryOperator::Add:
        kind = PhysicalExprKind::Add;
        break;
      case BinaryOperator::Subtract:
        kind = PhysicalExprKind::Subtract;
        break;
      case BinaryOperator::Multiply:
        kind = PhysicalExprKind::Multiply;
        break;
      case BinaryOperator::FloorDivide:
        kind = PhysicalExprKind::FloorDiv;
        break;
      case BinaryOperator::Maximum:
      case BinaryOperator::MaximumNum:
        kind = PhysicalExprKind::Maximum;
        break;
      case BinaryOperator::Minimum:
      case BinaryOperator::MinimumNum:
        kind = PhysicalExprKind::Minimum;
        break;
      default:
        return failure();
      }
      return binaryExpression(origin->getContext(), kind, *lhs, *rhs);
    }
    return failure();
  }

  FailureOr<Value> asIndex(Location location, Value value) {
    if (value.getType().isIndex())
      return value;
    if (!isa<IntegerType>(value.getType()))
      return failure();
    return Value(builder.create<gpu::CastOp>(location, builder.getIndexType(),
                                             value));
  }

  FailureOr<Value> asLogicalIndex(Location location, Value value) {
    if (!isa<gpu::FragmentType>(value.getType()))
      return asIndex(location, value);
    auto source = cast<gpu::FragmentType>(value.getType());
    if (source.getElementType().isIndex())
      return value;
    if (!isa<IntegerType>(source.getElementType()))
      return failure();
    auto target = gpu::FragmentType::get(
        value.getContext(), builder.getIndexType(), source.getShape(),
        source.getAxisMaps(), source.getValidity(), source.getOwner());
    return Value(builder.create<gpu::CastOp>(location, target, value));
  }

  FailureOr<unsigned> physicalResourceAxis(Type logicalResource,
                                           Type physicalResource,
                                           unsigned logicalAxis) {
    RankedTensorType logicalTensor;
    if (auto view = dyn_cast<intent::ViewType>(logicalResource))
      logicalTensor = dyn_cast<RankedTensorType>(view.getTensor());
    else if (auto buffer = dyn_cast<intent::BufferType>(logicalResource))
      logicalTensor = dyn_cast<RankedTensorType>(buffer.getTensor());
    else
      logicalTensor = dyn_cast<RankedTensorType>(logicalResource);
    if (!logicalTensor || logicalAxis >= static_cast<unsigned>(logicalTensor.getRank()))
      return failure();

    unsigned physicalRank = 0;
    if (auto view = dyn_cast<gpu::ViewType>(physicalResource))
      physicalRank = view.getRank();
    else if (auto buffer = dyn_cast<gpu::BufferType>(physicalResource))
      physicalRank = buffer.getShape().size();
    else if (auto fragment = dyn_cast<gpu::FragmentType>(physicalResource))
      physicalRank = fragment.getShape().size();
    else
      return failure();
    unsigned logicalRank = logicalTensor.getRank();
    if (physicalRank < logicalRank)
      return failure();
    if (!isa<gpu::FragmentType>(physicalResource) && physicalRank != logicalRank)
      return failure();
    return physicalRank - logicalRank + logicalAxis;
  }

  FailureOr<Value> resourceExtent(Location location, Value resource,
                                  unsigned axis) {
    PhysicalExprAttr extent;
    if (auto view = dyn_cast<gpu::ViewType>(resource.getType())) {
      if (axis >= view.getRank())
        return failure();
      extent = cast<PhysicalExprAttr>(view.getLayout().getExtents()[axis]);
      if (extent.getKind() ==
          PhysicalExprKind::Constant)
        return physicalExtentValue(location, extent);
      return Value(builder.create<gpu::DimOp>(location, builder.getIndexType(),
                                              resource, axis));
    } else if (auto buffer = dyn_cast<gpu::BufferType>(resource.getType())) {
      if (axis >= buffer.getShape().size())
        return failure();
      extent = cast<PhysicalExprAttr>(buffer.getShape()[axis]);
    } else if (auto fragment = dyn_cast<gpu::FragmentType>(resource.getType())) {
      if (axis >= fragment.getShape().size())
        return failure();
      gpu::PhysicalProgramAnalysis analysis(physicalKernel);
      if (analysis.axisRealization(resource, axis).constructionScalarSeed) {
        auto mapping = cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[axis]);
        auto dimension = dimensions.find(mapping.getDimensionId());
        if (dimension != dimensions.end())
          return dimension->second;
        FailureOr<gpu::MakeRangeOp> range =
            gpu::queryExactLogicalRange(analysis.axisRanges(resource, axis));
        if (failed(range))
          return failure();
        return rangeExtent(location, range->getLogicalStart(),
                           range->getLogicalStop(), range->getStep());
      }
      extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    } else {
      return failure();
    }
    return physicalExtentValue(location, extent);
  }

  void bindExtentDimensions(ArrayAttr identities, Value extent) {
    if (!identities || identities.size() != 1)
      return;
    int64_t identity = cast<IntegerAttr>(identities[0]).getInt();
    if (identity > 0)
      dimensions[identity] = extent;
  }

  FailureOr<SmallVector<Value>> accessCoordinates(Operation *operation) {
    auto relation = canonicalAnalysis.indexRelation(operation);
    if (failed(relation))
      return failure();
    SmallVector<Value> coordinates;
    FailureOr<Value> resource = get(relation->source);
    if (failed(resource))
      return failure();
    Value stored = cast<IndexedAccessOpInterface>(operation).getStoredValue();
    gpu::FragmentType valuePrototype;
    if (isa<ViewStoreOp, BufferStoreOp, ScatterUniqueOp, ScatterReduceOp>(operation)) {
      FailureOr<Value> value = get(stored);
      if (succeeded(value))
        valuePrototype = dyn_cast<gpu::FragmentType>((*value).getType());
    }
    auto resultExtent = [&](size_t axis,
                            PhysicalExprAttr fallback)
        -> FailureOr<PhysicalExprAttr> {
      if (valuePrototype) {
        size_t resultRank = relation->resultDimensionIdentities.size();
        if (valuePrototype.getShape().size() < resultRank)
          return failure();
        // A scalar logical index may already have been promoted to a physical
        // fragment.  Those ownership axes lead the value fragment but are not
        // part of the logical access result.  Relation result axis zero must
        // therefore address the first axis after that exact lifted prefix.
        size_t liftedRank = valuePrototype.getShape().size() - resultRank;
        size_t prototypeAxis = liftedRank + axis;
        if (prototypeAxis < valuePrototype.getShape().size())
          return cast<PhysicalExprAttr>(
              valuePrototype.getShape()[prototypeAxis]);
      }
      if (operation->getNumResults() == 0)
        return fallback;
      auto tensor = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
      if (!tensor || axis >= static_cast<size_t>(tensor.getRank()))
        return fallback;
      if (!tensor.isDynamicDim(axis))
        return expression(operation->getContext(), PhysicalExprKind::Constant,
                          tensor.getDimSize(axis));
      FailureOr<PhysicalExprAttr> extent =
          fragmentExtentExpression(tensor, operation, axis);
      return extent;
    };
    auto makeRange = [&](unsigned sourceAxis, Value start, Value stop,
                         Value step, uint64_t sourceId,
                         int64_t dimensionId,
                         bool derived,
                         PhysicalExprAttr physicalExtent) -> FailureOr<Value> {
      FailureOr<Value> physicalStart = asIndex(operation->getLoc(), start);
      FailureOr<Value> physicalStop = asIndex(operation->getLoc(), stop);
      FailureOr<Value> physicalStep = asIndex(operation->getLoc(), step);
      if (failed(physicalStart) || failed(physicalStop) || failed(physicalStep))
        return failure();
      start = *physicalStart;
      stop = *physicalStop;
      step = *physicalStep;
      FailureOr<Value> extent =
          physicalExtentValue(operation->getLoc(), physicalExtent);
      if (failed(extent))
        return failure();
      auto type = fragmentType(operation->getContext(), builder.getIndexType(),
                               {physicalExtent},
                               {{sourceId, sourceAxis, dimensionId, derived}});
      return Value(builder.create<gpu::MakeRangeOp>(
          operation->getLoc(), type, start, *extent, step, start, stop, sourceId,
          sourceAxis, derived));
    };
    unsigned sourceAxis = 0;
    unsigned resultAxis = 0;
    unsigned basicResultAxes = 0;
    for (const IndexTermFact &term : relation->terms) {
      int64_t kind = term.kind;
      if (kind == 0 || kind == 1 || kind == 4 || kind == 5)
        ++basicResultAxes;
    }
    if (basicResultAxes > relation->resultDimensionIdentities.size())
      return failure();
    unsigned advancedRank =
        relation->resultDimensionIdentities.size() - basicResultAxes;
    std::optional<unsigned> advancedStart;
    for (const IndexTermFact &term : relation->terms) {
      if (term.kind == 1) {
        ++resultAxis;
        continue;
      }
      FailureOr<unsigned> physicalSourceAxis = physicalResourceAxis(
          relation->source.getType(), (*resource).getType(), sourceAxis);
      if (failed(physicalSourceAxis))
        return failure();
      if (term.kind == 0) {
        auto view = dyn_cast<gpu::ViewType>((*resource).getType());
        auto buffer = dyn_cast<gpu::BufferType>((*resource).getType());
        auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
        if ((!view && !buffer && !fragment) ||
            (view && *physicalSourceAxis >= view.getRank()) ||
            (buffer && *physicalSourceAxis >= buffer.getShape().size()) ||
            (fragment &&
             *physicalSourceAxis >= fragment.getShape().size())) {
          operation->emitOpError(
              "full-slice term has no matching physical source axis");
          return failure();
        }
        Value start = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 0);
        Value stop;
        uint64_t sourceId;
        PhysicalExprAttr extent;
        uint32_t logicalSourceAxis = sourceAxis;
        bool derived = false;
        if (view || buffer) {
          // A buffer is a canonical value result, using the same derived
          // identity convention as resultAxisIdentity.
          sourceId = view ? view.getSourceId() : buffer.getInstance() + 1;
          derived = static_cast<bool>(buffer);
          extent = cast<PhysicalExprAttr>(
              view ? view.getLayout().getExtents()[*physicalSourceAxis]
                   : buffer.getShape()[*physicalSourceAxis]);
          FailureOr<Value> physicalStop =
              physicalExtentValue(operation->getLoc(), extent);
          if (failed(physicalStop)) {
            operation->emitOpError(
                "resource full-slice extent is not materialized in the current program: ")
                << extent;
            return failure();
          }
          stop = *physicalStop;
        } else {
          auto mapping =
              cast<gpu::AxisMapAttr>(
                  fragment.getAxisMaps()[*physicalSourceAxis]);
          FailureOr<Value> physicalExtent = physicalExtentValue(
              operation->getLoc(),
              cast<PhysicalExprAttr>(
                  fragment.getShape()[*physicalSourceAxis]));
          if (failed(physicalExtent)) {
            operation->emitOpError(
                "fragment full-slice extent is not materialized in the current program: ")
                << fragment.getShape()[*physicalSourceAxis];
            return failure();
          }
          stop = *physicalExtent;
          sourceId = mapping.getSourceId();
          logicalSourceAxis = mapping.getSourceAxis();
          derived = mapping.getDerived();
          extent =
              cast<PhysicalExprAttr>(
                  fragment.getShape()[*physicalSourceAxis]);
        }
        int64_t dimension = 0;
        if (view) {
          auto dimensions = view.getLayout().getDimensionIds();
          if (*physicalSourceAxis >= dimensions.size())
            return failure();
          dimension = dimensions[*physicalSourceAxis];
        } else if (buffer) {
          dimension = relation->resultDimensionIdentities[resultAxis];
        } else {
          dimension = cast<gpu::AxisMapAttr>(
                          fragment.getAxisMaps()[*physicalSourceAxis])
                          .getDimensionId();
        }
        if (dimension <= 0)
          return failure();
        // A full slice of an already-physical fragment consumes that
        // fragment's current extent.  Reconstructing the helper-local logical
        // result dimension here would discard an enclosing region segment.
        if (view || buffer) {
          FailureOr<PhysicalExprAttr> resultPhysicalExtent =
              resultExtent(resultAxis, extent);
          if (failed(resultPhysicalExtent))
            return failure();
          extent = *resultPhysicalExtent;
        }
        Value step = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 1);
        FailureOr<Value> coordinate = makeRange(
            logicalSourceAxis, start, stop, step, sourceId, dimension, derived,
            extent);
        if (failed(coordinate))
          return failure();
        coordinates.push_back(*coordinate);
        ++sourceAxis;
        ++resultAxis;
        continue;
      }
      if (term.kind == 2) {
        int64_t literal = *term.staticValues[0];
        Value coordinate = builder.create<arith::ConstantIndexOp>(
            operation->getLoc(), literal);
        if (literal < 0) {
          FailureOr<Value> extent =
              resourceExtent(operation->getLoc(), *resource,
                             *physicalSourceAxis);
          if (failed(extent))
            return failure();
          coordinate = createBinary(builder, operation->getLoc(),
                                    builder.getIndexType(), *extent, coordinate,
                                    BinaryOperator::Add);
        }
        coordinates.push_back(coordinate);
        ++sourceAxis;
        continue;
      }
      if (term.kind == 3) {
        Value index = term.operands[0];
        FailureOr<Value> coordinate = get(index);
        if (failed(coordinate))
          return failure();
        Value physicalCoordinate = *coordinate;
        if (auto fragment = dyn_cast<gpu::FragmentType>(physicalCoordinate.getType())) {
          auto logicalCoordinate =
              dyn_cast<RankedTensorType>(index.getType());
          if (logicalCoordinate) {
            if (fragment.getShape().size() > advancedRank)
              return failure();
            if (!advancedStart) {
              advancedStart = resultAxis;
              resultAxis += advancedRank;
            }
          }
          SmallVector<Attribute> mappings;
          for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps())) {
            auto mapping = cast<gpu::AxisMapAttr>(attribute);
            int64_t dimension = mapping.getDimensionId();
            if (logicalCoordinate)
              dimension = relation->resultDimensionIdentities[
                  *advancedStart + advancedRank - fragment.getShape().size() + axis];
            mappings.push_back(gpu::AxisMapAttr::get(
                operation->getContext(), mapping.getSourceId(),
                mapping.getSourceAxis(), dimension,
                mappings.size(), mapping.getDerived()));
          }
          auto target = gpu::FragmentType::get(
              operation->getContext(), fragment.getElementType(),
              fragment.getShape(), builder.getArrayAttr(mappings),
              fragment.getValidity(), fragment.getOwner());
          if (target != fragment) {
            auto reassociation = gpu::inferReshapeReassociation(fragment, target);
            if (failed(reassociation))
              return failure();
            physicalCoordinate = builder.create<gpu::ReshapeOp>(
                operation->getLoc(), target, physicalCoordinate, *reassociation);
          }
        }
        FailureOr<Value> logicalIndex =
            asLogicalIndex(operation->getLoc(), physicalCoordinate);
        if (failed(logicalIndex))
          return failure();
        coordinates.push_back(*logicalIndex);
        ++sourceAxis;
        continue;
      }
      if (term.kind == 4) {
        FailureOr<Value> range = get(term.operands[0]);
        if (failed(range) || !isa<gpu::RangeType>((*range).getType()))
          return failure();
        Value start = rangeBound(operation->getLoc(), *range, 0);
        Value stop = rangeBound(operation->getLoc(), *range, 1);
        Value step = rangeBound(operation->getLoc(), *range, 2);
        auto rangeType = cast<gpu::RangeType>((*range).getType());
        PhysicalExprAttr fallback = expression(
            operation->getContext(), PhysicalExprKind::Constant, 1);
        FailureOr<PhysicalExprAttr> physicalExtent =
            resultExtent(resultAxis, fallback);
        if (failed(physicalExtent))
          return failure();
        PhysicalExprAttr extent = *physicalExtent;
        uint64_t sourceId = rangeType.getSourceId();
        uint32_t logicalSourceAxis = rangeType.getSourceAxis();
        bool derived = rangeType.getDerived();
        auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
        auto logical =
            dyn_cast<RankedTensorType>(relation->source.getType());
        const bool fragmentIndex = static_cast<bool>(fragment);
        unsigned logicalAxis = sourceAxis;
        unsigned fragmentAxis = *physicalSourceAxis;
        if (!fragment && valuePrototype) {
          auto valueType = dyn_cast<RankedTensorType>(stored.getType());
          if (valueType &&
              valueType.getRank() == relation->resultDimensionIdentities.size()) {
            fragment = valuePrototype;
            logical = valueType;
            logicalAxis = resultAxis;
            fragmentAxis =
                fragment.getShape().size() - valueType.getRank() + resultAxis;
          }
        }
        if (fragment && logical && integerConstant(start) == 0 &&
            integerConstant(step) == 1) {
          DenseI64ArrayAttr identities = dimensionIds(logical);
          auto mapping = cast<gpu::AxisMapAttr>(
              fragment.getAxisMaps()[fragmentAxis]);
          bool coversSource = false;
          if (logical.isDynamicDim(logicalAxis)) {
            if (identities) {
              auto binding = dimensions.find(identities[logicalAxis]);
              coversSource =
                  binding != dimensions.end() && stop == binding->second;
            }
          } else {
            coversSource = integerConstant(stop) == logical.getDimSize(logicalAxis);
          }
          if (coversSource && identities &&
              mapping.getDimensionId() == identities[logicalAxis] &&
              rangeType.getDimensionId() == identities[logicalAxis]) {
            // Full logical indexing preserves the value's axis relation. Reads
            // use local fragment coordinates; writes retain logical bounds so
            // ownership advances their coordinates with the producing tile.
            sourceId = mapping.getSourceId();
            logicalSourceAxis = mapping.getSourceAxis();
            derived = mapping.getDerived();
            extent = cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
            if (fragmentIndex) {
              FailureOr<Value> fragmentStop =
                  physicalExtentValue(operation->getLoc(), extent);
              if (failed(fragmentStop))
                return failure();
              stop = *fragmentStop;
            }
          }
        }
        FailureOr<Value> coordinate = makeRange(
            logicalSourceAxis, start, stop, step, sourceId,
            rangeType.getDimensionId(), derived, extent);
        if (failed(coordinate))
          return failure();
        for (StringRef name : {gpu::sourceSubregionAttr,
                               gpu::sourceSubregionBoundAttr})
          if (Attribute value = (*range).getDefiningOp()->getAttr(name))
            coordinate->getDefiningOp()->setAttr(name, value);
        coordinates.push_back(*coordinate);
        ++sourceAxis;
        ++resultAxis;
        continue;
      }
      if (term.kind == 5) {
        SmallVector<Value> bounds;
        for (unsigned component = 0; component < 3; ++component) {
          Value dynamic = term.operands[component];
          std::optional<int64_t> literal = term.staticValues[component];
          if (dynamic) {
            FailureOr<Value> value = get(dynamic);
            if (failed(value))
              return failure();
            bounds.push_back(*value);
          } else if (literal) {
            bounds.push_back(builder.create<arith::ConstantIndexOp>(
                operation->getLoc(), *literal));
          } else {
            auto view = dyn_cast<gpu::ViewType>((*resource).getType());
            auto buffer = dyn_cast<gpu::BufferType>((*resource).getType());
            auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
            if ((!view && !buffer && !fragment) ||
                (view && *physicalSourceAxis >= view.getRank()) ||
                (buffer && *physicalSourceAxis >= buffer.getShape().size()) ||
                (fragment &&
                 *physicalSourceAxis >= fragment.getShape().size()))
              return failure();
            if (component == 0) {
              bounds.push_back(builder.create<arith::ConstantIndexOp>(
                  operation->getLoc(), 0));
            } else if (component == 1) {
              if (view || buffer) {
                FailureOr<Value> physicalExtent = physicalExtentValue(
                    operation->getLoc(),
                    cast<PhysicalExprAttr>(
                        view ? view.getLayout().getExtents()[*physicalSourceAxis]
                             : buffer.getShape()[*physicalSourceAxis]));
                if (failed(physicalExtent))
                  return failure();
                bounds.push_back(*physicalExtent);
              } else {
                FailureOr<Value> physicalExtent = physicalExtentValue(
                    operation->getLoc(), cast<PhysicalExprAttr>(
                                             fragment.getShape()[*physicalSourceAxis]));
                if (failed(physicalExtent))
                  return failure();
                bounds.push_back(*physicalExtent);
              }
            } else {
              bounds.push_back(builder.create<arith::ConstantIndexOp>(
                  operation->getLoc(), 1));
            }
          }
        }
        auto view = dyn_cast<gpu::ViewType>((*resource).getType());
        auto buffer = dyn_cast<gpu::BufferType>((*resource).getType());
        auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
        if ((!view && !buffer && !fragment) ||
            (view && *physicalSourceAxis >= view.getRank()) ||
            (buffer && *physicalSourceAxis >= buffer.getShape().size()) ||
            (fragment &&
             *physicalSourceAxis >= fragment.getShape().size()))
          return failure();
        PhysicalExprAttr fallback = cast<PhysicalExprAttr>(
            view     ? view.getLayout().getExtents()[*physicalSourceAxis]
            : buffer ? buffer.getShape()[*physicalSourceAxis]
                     : fragment.getShape()[*physicalSourceAxis]);
        FailureOr<PhysicalExprAttr> physicalExtent =
            resultExtent(resultAxis, fallback);
        if (failed(physicalExtent))
          return failure();
        PhysicalExprAttr extent = *physicalExtent;
        uint64_t sourceId;
        uint32_t logicalSourceAxis = sourceAxis;
        int64_t dimension = 0;
        bool derived = false;
        if (view) {
          sourceId = view.getSourceId();
          auto dimensions = view.getLayout().getDimensionIds();
          if (*physicalSourceAxis >= dimensions.size())
            return failure();
          dimension = dimensions[*physicalSourceAxis];
        } else if (buffer) {
          sourceId = buffer.getInstance() + 1;
          derived = true;
        } else {
          auto mapping =
              cast<gpu::AxisMapAttr>(
                  fragment.getAxisMaps()[*physicalSourceAxis]);
          sourceId = mapping.getSourceId();
          logicalSourceAxis = mapping.getSourceAxis();
          dimension = mapping.getDimensionId();
          derived = mapping.getDerived();
        }
        dimension = relation->resultDimensionIdentities[resultAxis];
        if (dimension <= 0)
          return failure();
        FailureOr<Value> coordinate = makeRange(
            logicalSourceAxis, bounds[0], bounds[1], bounds[2], sourceId,
            dimension, derived, extent);
        if (failed(coordinate))
          return failure();
        coordinates.push_back(*coordinate);
        ++sourceAxis;
        ++resultAxis;
        continue;
      }
      return failure();
    }
    return coordinates;
  }

  FailureOr<SmallVector<int64_t>> sourceAxes(Operation *operation) {
    auto relation = canonicalAnalysis.indexRelation(operation);
    if (failed(relation))
      return failure();
    FailureOr<Value> resource = get(relation->source);
    if (failed(resource))
      return failure();
    SmallVector<int64_t> axes;
    unsigned sourceAxis = 0;
    for (const IndexTermFact &term : relation->terms) {
      if (term.kind == 1)
        continue;
      FailureOr<unsigned> physicalSourceAxis = physicalResourceAxis(
          relation->source.getType(), (*resource).getType(), sourceAxis++);
      if (failed(physicalSourceAxis))
        return failure();
      axes.push_back(*physicalSourceAxis);
    }
    return axes;
  }

  FailureOr<Type> accessResultType(Operation *operation, Type logical,
                                   ArrayRef<Value> coordinates,
                                   std::optional<Type> prototype = std::nullopt) {
    auto relation = canonicalAnalysis.indexRelation(operation);
    if (failed(relation))
      return failure();
    SmallVector<Attribute> liftedShape;
    SmallVector<Attribute> liftedMappings;
    uint64_t liftedOwner = 1;
    if (prototype) {
      auto fragment = dyn_cast<gpu::FragmentType>(*prototype);
      size_t logicalRank = relation->resultDimensionIdentities.size();
      if (fragment && fragment.getShape().size() >= logicalRank) {
        size_t prefixRank = fragment.getShape().size() - logicalRank;
        liftedOwner = fragment.getOwner();
        for (unsigned axis = 0; axis < prefixRank; ++axis) {
          auto mapping = cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[axis]);
          liftedShape.push_back(fragment.getShape()[axis]);
          liftedMappings.push_back(gpu::AxisMapAttr::get(
              operation->getContext(), mapping.getSourceId(),
              mapping.getSourceAxis(), mapping.getDimensionId(), axis,
              mapping.getDerived()));
        }
      }
    }
    unsigned coordinateIndex = 0;
    for (const IndexTermFact &term : relation->terms) {
      if (term.kind == 1)
        continue;
      if (coordinateIndex >= coordinates.size())
        return failure();
      Value current = coordinates[coordinateIndex++];
      if (term.kind != 3)
        continue;
      if (isa<RankedTensorType>(term.operands[0].getType()))
        continue;
      auto fragment = dyn_cast<gpu::FragmentType>(current.getType());
      if (!fragment)
        continue;
      liftedOwner = fragment.getOwner();
      for (auto [axis, mappingAttribute] :
           llvm::enumerate(fragment.getAxisMaps())) {
        auto mapping = cast<gpu::AxisMapAttr>(mappingAttribute);
        bool exists = llvm::any_of(liftedMappings, [&](Attribute existing) {
          auto value = cast<gpu::AxisMapAttr>(existing);
          return value.getSourceId() == mapping.getSourceId() &&
                 value.getSourceAxis() == mapping.getSourceAxis() &&
                 value.getDerived() == mapping.getDerived();
        });
        if (exists)
          continue;
        liftedShape.push_back(fragment.getShape()[axis]);
        liftedMappings.push_back(gpu::AxisMapAttr::get(
            operation->getContext(), mapping.getSourceId(),
            mapping.getSourceAxis(), mapping.getDimensionId(),
            liftedMappings.size(), mapping.getDerived()));
      }
    }
    FailureOr<Type> converted = failure();
    auto logicalTensor = dyn_cast<RankedTensorType>(logical);
    auto prototypeFragment =
        prototype ? dyn_cast<gpu::FragmentType>(*prototype) : gpu::FragmentType();
    if (logicalTensor && prototypeFragment) {
      // A write value may already carry physical ownership axes introduced by
      // scalar logical indices.  The index relation describes only its result
      // tail.  Split that tail from the prototype before combining it with the
      // lifted prefix below; using the complete prototype here would append the
      // ownership axes twice.
      size_t resultRank = relation->resultDimensionIdentities.size();
      if (prototypeFragment.getShape().size() < resultRank)
        return failure();
      size_t tailStart = prototypeFragment.getShape().size() - resultRank;
      SmallVector<Attribute> tailShape(
          prototypeFragment.getShape().begin() + tailStart,
          prototypeFragment.getShape().end());
      SmallVector<Attribute> tailMappings;
      for (unsigned axis = 0; axis < resultRank; ++axis) {
        auto mapping = cast<gpu::AxisMapAttr>(
            prototypeFragment.getAxisMaps()[tailStart + axis]);
        tailMappings.push_back(gpu::AxisMapAttr::get(
            operation->getContext(), mapping.getSourceId(),
            mapping.getSourceAxis(), mapping.getDimensionId(), axis,
            mapping.getDerived()));
      }
      converted = Type(gpu::FragmentType::get(
          operation->getContext(), logicalTensor.getElementType(),
          builder.getArrayAttr(tailShape), builder.getArrayAttr(tailMappings),
          prototypeFragment.getValidity(), prototypeFragment.getOwner()));
    } else if (logicalTensor) {
      ArrayRef<int64_t> dimensions = relation->resultDimensionIdentities;
      if (dimensions.size() != static_cast<size_t>(logicalTensor.getRank()))
        return failure();
      SmallVector<Attribute> shape;
      SmallVector<Attribute> mappings;
      for (unsigned axis = 0; axis < dimensions.size(); ++axis) {
        PhysicalExprAttr extent;
        if (logicalTensor.isDynamicDim(axis)) {
          FailureOr<PhysicalExprAttr> dynamicExtent =
              fragmentExtentExpression(logicalTensor, operation, axis);
          if (failed(dynamicExtent))
            return failure();
          extent = *dynamicExtent;
        } else {
          extent = expression(operation->getContext(),
                              PhysicalExprKind::Constant,
                              logicalTensor.getDimSize(axis));
        }
        FailureOr<PhysicalAxisIdentity> identity =
            resultAxisIdentity(operation, /*resultIndex=*/0, axis);
        if (!extent || dimensions[axis] <= 0 || failed(identity))
          return failure();
        shape.push_back(extent);
        mappings.push_back(gpu::AxisMapAttr::get(
            operation->getContext(), identity->sourceId, identity->sourceAxis,
            dimensions[axis], axis, identity->derived));
      }
      converted = Type(gpu::FragmentType::get(
          operation->getContext(), logicalTensor.getElementType(),
          builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
          /*validity=*/1, liftedOwner));
    } else {
      converted = convertDataType(logical, operation, prototype);
    }
    if (failed(converted)) {
      operation->emitOpError(
          "access result logical type has no physical fragment schema");
      return failure();
    }
    if (!liftedShape.empty()) {
      Type element = *converted;
      SmallVector<Attribute> shape(liftedShape.begin(), liftedShape.end());
      SmallVector<Attribute> mappings(liftedMappings.begin(),
                                     liftedMappings.end());
      uint32_t validity = 1;
      if (auto fragment = dyn_cast<gpu::FragmentType>(*converted)) {
        element = fragment.getElementType();
        validity = fragment.getValidity();
        for (auto [axis, mappingAttribute] :
             llvm::enumerate(fragment.getAxisMaps())) {
          auto mapping = cast<gpu::AxisMapAttr>(mappingAttribute);
          bool exists = llvm::any_of(mappings, [&](Attribute existing) {
            auto value = cast<gpu::AxisMapAttr>(existing);
            return value.getSourceId() == mapping.getSourceId() &&
                   value.getSourceAxis() == mapping.getSourceAxis() &&
                   value.getDerived() == mapping.getDerived();
          });
          if (exists)
            continue;
          shape.push_back(fragment.getShape()[axis]);
          mappings.push_back(gpu::AxisMapAttr::get(
              operation->getContext(), mapping.getSourceId(),
              mapping.getSourceAxis(), mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
        }
      }
      *converted = gpu::FragmentType::get(
          operation->getContext(), element,
          builder.getArrayAttr(shape), builder.getArrayAttr(mappings), validity,
          liftedOwner);
    }
    auto fragment = dyn_cast<gpu::FragmentType>(*converted);
    if (!fragment)
      return *converted;
    SmallVector<Attribute> shape(fragment.getShape().begin(),
                                 fragment.getShape().end());
    SmallVector<Attribute> mappings(liftedMappings.begin(),
                                    liftedMappings.end());
    unsigned coordinate = 0;
    unsigned resultAxis = liftedMappings.size();
    const unsigned liftedRank = resultAxis;
    unsigned basicResultAxes = 0;
    for (const IndexTermFact &term : relation->terms) {
      int64_t kind = term.kind;
      if (kind == 0 || kind == 1 || kind == 4 || kind == 5)
        ++basicResultAxes;
    }
    if (basicResultAxes > relation->resultDimensionIdentities.size())
      return failure();
    unsigned advancedRank =
        relation->resultDimensionIdentities.size() - basicResultAxes;
    bool advancedMapped = false;
    auto resultDimension = [&](size_t axis) -> FailureOr<int64_t> {
      ArrayRef<int64_t> dimensions = relation->resultDimensionIdentities;
      if (axis < liftedRank || axis - liftedRank >= dimensions.size())
        return failure();
      return dimensions[axis - liftedRank];
    };
    auto resultIdentity = [&](unsigned axis)
        -> FailureOr<PhysicalAxisIdentity> {
      if (operation->getNumResults() != 0)
        return resultAxisIdentity(operation, /*resultIndex=*/0, axis);
      if (axis >= fragment.getAxisMaps().size())
        return failure();
      auto mapping = cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[axis]);
      return PhysicalAxisIdentity{mapping.getSourceId(), mapping.getSourceAxis(),
                                  mapping.getDimensionId(),
                                  mapping.getDerived()};
    };
    auto appendCoordinate = [&](Value value) -> LogicalResult {
      auto source = dyn_cast<gpu::FragmentType>(value.getType());
      if (!source)
        return failure();
      for (Attribute attribute : source.getAxisMaps()) {
        auto mapping = cast<gpu::AxisMapAttr>(attribute);
        FailureOr<int64_t> dimension = resultDimension(resultAxis);
        if (failed(dimension))
          return failure();
        shape[resultAxis] = source.getShape()[mapping.getFragmentAxis()];
        mappings.push_back(gpu::AxisMapAttr::get(
            operation->getContext(), mapping.getSourceId(),
            mapping.getSourceAxis(), *dimension, resultAxis++,
            mapping.getDerived()));
      }
      return success();
    };
    auto advancedAxisMapping = [&](unsigned advancedAxis)
        -> gpu::AxisMapAttr {
      ArrayRef<int64_t> resultDimensions = relation->resultDimensionIdentities;
      if (advancedAxis >= resultDimensions.size())
        return {};
      gpu::AxisMapAttr selected;
      unsigned coordinateIndex = 0;
      for (const IndexTermFact &term : relation->terms) {
        if (term.kind == 1)
          continue;
        if (coordinateIndex >= coordinates.size())
          return {};
        Value current = coordinates[coordinateIndex++];
        if (term.kind != 3)
          continue;
        if (!isa<RankedTensorType>(term.operands[0].getType()))
          continue;
        auto logical =
            cast<RankedTensorType>(term.operands[0].getType());
        auto source = dyn_cast<gpu::FragmentType>(current.getType());
        DenseI64ArrayAttr logicalDimensions = dimensionIds(logical);
        if (!source || !logicalDimensions ||
            logical.getRank() > static_cast<int64_t>(advancedRank))
          continue;
        unsigned alignedStart = advancedRank - logical.getRank();
        if (advancedAxis < alignedStart)
          continue;
        unsigned localAxis = advancedAxis - alignedStart;
        if (localAxis >= static_cast<unsigned>(logical.getRank()))
          continue;
        // A singleton advanced-index operand is broadcast along this result
        // axis and therefore carries no coordinate variation for it.  Let the
        // non-singleton operand provide the axis relation instead of treating
        // the two broadcast operands as conflicting coordinate authorities.
        if (!logical.isDynamicDim(localAxis) &&
            logical.getDimSize(localAxis) == 1)
          continue;
        if (logicalDimensions[localAxis] <= 0 ||
            source.getShape().size() <
                static_cast<unsigned>(logical.getRank()))
          return {};
        unsigned physicalAxis =
            source.getShape().size() - logical.getRank() + localAxis;
        auto mapping = cast<gpu::AxisMapAttr>(
            source.getAxisMaps()[physicalAxis]);
        auto sameMapping = [](gpu::AxisMapAttr lhs, gpu::AxisMapAttr rhs) {
          return lhs.getSourceId() == rhs.getSourceId() &&
                 lhs.getSourceAxis() == rhs.getSourceAxis() &&
                 lhs.getDimensionId() == rhs.getDimensionId() &&
                 lhs.getDerived() == rhs.getDerived();
        };
        if (selected && !sameMapping(selected, mapping))
          return {};
        selected = mapping;
      }
      return selected;
    };
    for (const IndexTermFact &term : relation->terms) {
      if (term.kind == 1) {
        FailureOr<int64_t> dimension = resultDimension(resultAxis);
        FailureOr<PhysicalAxisIdentity> identity = resultIdentity(resultAxis);
        if (failed(dimension) || *dimension <= 0 || failed(identity))
          return failure();
        mappings.push_back(gpu::AxisMapAttr::get(
            operation->getContext(), identity->sourceId, identity->sourceAxis,
            *dimension, resultAxis, identity->derived));
        ++resultAxis;
        continue;
      }
      if (coordinate >= coordinates.size())
        return failure();
      Value current = coordinates[coordinate++];
      if (term.kind == 0 || term.kind == 4 ||
          term.kind == 5) {
        if (failed(appendCoordinate(current))) {
          operation->emitOpError(
              "access range coordinate cannot map a result axis");
          return failure();
        }
        continue;
      }
      if (term.kind == 3 && isa<gpu::FragmentType>(current.getType())) {
        if (!isa<RankedTensorType>(term.operands[0].getType()))
          continue;
        if (!advancedMapped) {
          for (unsigned axis = 0; axis < advancedRank; ++axis) {
            FailureOr<int64_t> dimension = resultDimension(resultAxis);
            if (failed(dimension))
              return failure();
            PhysicalAxisIdentity identity;
            if (auto mapping = advancedAxisMapping(axis)) {
              identity = PhysicalAxisIdentity{
                  mapping.getSourceId(), mapping.getSourceAxis(), *dimension,
                  mapping.getDerived()};
            } else {
              FailureOr<PhysicalAxisIdentity> derivedIdentity =
                  resultIdentity(resultAxis);
              if (failed(derivedIdentity))
                return failure();
              identity = *derivedIdentity;
              identity.dimensionId = *dimension;
            }
            mappings.push_back(gpu::AxisMapAttr::get(
                operation->getContext(), identity.sourceId, identity.sourceAxis,
                *dimension, resultAxis++, identity.derived));
          }
          advancedMapped = true;
        }
        continue;
      }
      if (term.kind != 2 && term.kind != 3)
        return failure();
    }
    if (coordinate != coordinates.size() ||
        resultAxis != fragment.getShape().size()) {
      operation->emitOpError()
          << "access result rank/coordinate partition disagrees: consumed "
          << coordinate << " of " << coordinates.size() << " coordinates and "
          << resultAxis << " of " << fragment.getShape().size()
          << " result axes";
      return failure();
    }
    return Type(gpu::FragmentType::get(
        operation->getContext(), fragment.getElementType(),
        builder.getArrayAttr(shape),
        builder.getArrayAttr(mappings), fragment.getValidity(),
        fragment.getOwner()));
  }

  FailureOr<Value> accessValue(Operation *operation, Value logical,
                               ArrayRef<Value> coordinates) {
    FailureOr<Value> value = get(logical);
    if (failed(value))
      return failure();
    FailureOr<Type> relationType = accessResultType(
        operation, logical.getType(), coordinates, (*value).getType());
    if (failed(relationType))
      return failure();
    if ((*value).getType() == *relationType)
      return *value;
    auto source = dyn_cast<gpu::FragmentType>((*value).getType());
    auto target = dyn_cast<gpu::FragmentType>(*relationType);
    if (!source && target && (*value).getType() == target.getElementType())
      return Value(builder.create<gpu::SplatOp>(operation->getLoc(), target,
                                                *value));
    if (!source || !target) {
      operation->emitOpError()
          << "indexed write value/result relation is not a fragment: value="
          << (*value).getType() << ", relation=" << *relationType;
      return failure();
    }
    if (source.getElementType() != target.getElementType() ||
        source.getShape().size() != target.getShape().size()) {
      operation->emitOpError()
          << "indexed write value cannot adopt its result relation: " << source
          << " vs " << target;
      return failure();
    }
    if (source.getShape() == target.getShape() &&
        !gpu::queryAxisProjection(source, target).isExact()) {
      // Canonical assignment has already aligned logical tensor axes by
      // position.  Rebinding those axes to destination coordinates preserves
      // lane order; matching old domain identities would imply a transpose.
      SmallVector<Attribute> groups;
      for (unsigned axis = 0; axis < source.getShape().size(); ++axis) {
        auto axes = builder.getDenseI64ArrayAttr({static_cast<int64_t>(axis)});
        groups.push_back(gpu::ReshapeGroupAttr::get(
            operation->getContext(), axes, axes));
      }
      auto projected = builder.create<gpu::ReshapeOp>(
          operation->getLoc(), target, *value, builder.getArrayAttr(groups));
      if (Operation *definition = (*value).getDefiningOp())
        if (Attribute origin = definition->getAttr(gpu::originAttr))
          projected->setAttr(gpu::originAttr, origin);
      return projected.getResult();
    }
    return retargetBroadcast(builder, operation->getLoc(), *value, target);
  }

  FailureOr<Value> projectAccessOperand(Location location, Value value,
                                        gpu::FragmentType accessType) {
    if (!value)
      return value;
    Type element = value.getType();
    if (auto source = dyn_cast<gpu::FragmentType>(element))
      element = source.getElementType();
    auto target = gpu::FragmentType::get(
        value.getContext(), element, accessType.getShape(),
        accessType.getAxisMaps(), accessType.getValidity(), accessType.getOwner());
    if (value.getType() == target)
      return value;
    if (!isa<gpu::FragmentType>(value.getType()))
      return Value(builder.create<gpu::SplatOp>(location, target, value));
    return retargetBroadcast(builder, location, value, target);
  }

  SmallVector<bool> canonicalAccessAxisProofs(Operation *operation,
                                              Value resource) {
    auto relation = canonicalAnalysis.indexRelation(operation);
    if (failed(relation) || !resource)
      return {};

    unsigned physicalRank = 0;
    if (auto view = dyn_cast<gpu::ViewType>(resource.getType()))
      physicalRank = view.getRank();
    else if (auto buffer = dyn_cast<gpu::BufferType>(resource.getType()))
      physicalRank = buffer.getShape().size();
    else if (auto fragment = dyn_cast<gpu::FragmentType>(resource.getType()))
      physicalRank = fragment.getShape().size();
    else
      return {};
    SmallVector<bool> proven(physicalRank, false);

    RankedTensorType source;
    Type sourceType = relation->source.getType();
    if (auto view = dyn_cast<intent::ViewType>(sourceType))
      source = dyn_cast<RankedTensorType>(view.getTensor());
    else if (auto buffer = dyn_cast<intent::BufferType>(sourceType))
      source = dyn_cast<RankedTensorType>(buffer.getTensor());
    else
      source = dyn_cast<RankedTensorType>(sourceType);

    SmallVector<Value> indexedValues(proven.size());
    unsigned sourceAxis = 0;
    for (const IndexTermFact &term : relation->terms) {
      if (term.kind == 1)
        continue;
      FailureOr<unsigned> physicalSourceAxis = physicalResourceAxis(
          relation->source.getType(), resource.getType(), sourceAxis);
      if (failed(physicalSourceAxis) || *physicalSourceAxis >= proven.size())
        return {};
      if (term.kind == 0 && isa<gpu::FragmentType>(resource.getType())) {
        // A full slice of an already materialized fragment addresses its
        // complete physical value. Any logical tail was resolved when that
        // value was produced; rebuilding a resource bound from its
        // construction-time extent creates a competing validity relation when
        // pointwise ownership later widens the fragment.
        proven[*physicalSourceAxis] = true;
      } else if (term.kind == 2 && source &&
                 sourceAxis < static_cast<unsigned>(source.getRank()) &&
                 !source.isDynamicDim(sourceAxis)) {
        int64_t index = *term.staticValues[0];
        int64_t extent = source.getDimSize(sourceAxis);
        proven[*physicalSourceAxis] = -extent <= index && index < extent;
      } else if (term.kind == 3) {
        indexedValues[*physicalSourceAxis] = term.operands[0];
      }
      ++sourceAxis;
    }
    if (sourceAxis != relation->sourceRank)
      return {};

    func::FuncOp function = operation->getParentOfType<func::FuncOp>();
    if (!function)
      return proven;
    DominanceInfo dominance(function);
    function.walk([&](intent::AssumeInBoundsOp assumption) {
      if (assumption.getView() != relation->source)
        return;
      FailureOr<unsigned> axis = physicalResourceAxis(
          assumption.getView().getType(), resource.getType(),
          assumption.getAxis());
      if (failed(axis) || *axis >= proven.size() || proven[*axis] ||
          !indexedValues[*axis] ||
          assumption.getIndex() != indexedValues[*axis] ||
          !dominance.properlyDominates(assumption.getOperation(), operation))
        return;
      proven[*axis] = true;
    });
    return proven;
  }

  FailureOr<Value> materializeAccessValidity(
      Operation *operation, Value resource, ArrayRef<Value> coordinates,
      ArrayRef<int64_t> sourceAxes, Type payloadType, Value existing) {
    if (coordinates.size() != sourceAxes.size())
      return failure();
    SmallVector<bool> proven = canonicalAccessAxisProofs(operation, resource);
    if (proven.empty() && !coordinates.empty())
      return failure();

    auto payloadFragment = dyn_cast<gpu::FragmentType>(payloadType);
    Type predicateType = builder.getI1Type();
    if (payloadFragment)
      predicateType = gpu::FragmentType::get(
          operation->getContext(), builder.getI1Type(),
          payloadFragment.getShape(), payloadFragment.getAxisMaps(),
          payloadFragment.getValidity(), payloadFragment.getOwner());

    if (existing) {
      if (payloadFragment) {
        FailureOr<Value> projected =
            projectAccessOperand(operation->getLoc(), existing, payloadFragment);
        if (failed(projected))
          return failure();
        existing = *projected;
      }
      if (existing.getType() != predicateType)
        return failure();
    }

    // Basic index terms have an explicit Cartesian result position.  Preserve
    // that occurrence when equal-length regions share one source identity;
    // broadcasting by the coordinate type alone would select the trailing axis.
    SmallVector<std::optional<unsigned>> coordinateAxes(coordinates.size());
    auto relation = canonicalAnalysis.indexRelation(operation);
    if (payloadFragment && succeeded(relation) &&
        relation->resultDimensionIdentities.size() == payloadFragment.getShape().size()) {
      unsigned coordinateIndex = 0;
      unsigned resultAxis = 0;
      bool cartesian = true;
      for (const IndexTermFact &term : relation->terms) {
        if (term.kind == 1) {
          ++resultAxis;
          continue;
        }
        if (coordinateIndex >= coordinates.size()) {
          cartesian = false;
          break;
        }
        auto type = dyn_cast<gpu::FragmentType>(coordinates[coordinateIndex].getType());
        if (term.kind == 0 || term.kind == 4 || term.kind == 5) {
          if (!type || type.getShape().size() != 1 ||
              resultAxis >= payloadFragment.getShape().size()) {
            cartesian = false;
            break;
          }
          auto source = cast<gpu::AxisMapAttr>(type.getAxisMaps()[0]);
          auto target = cast<gpu::AxisMapAttr>(payloadFragment.getAxisMaps()[resultAxis]);
          if (!(gpu::sourceAxisIdentity(source) == gpu::sourceAxisIdentity(target)) ||
              source.getDimensionId() != target.getDimensionId() ||
              target.getDimensionId() != relation->resultDimensionIdentities[resultAxis] ||
              type.getShape()[0] != payloadFragment.getShape()[resultAxis] ||
              type.getOwner() != payloadFragment.getOwner()) {
            cartesian = false;
            break;
          }
          coordinateAxes[coordinateIndex] = resultAxis++;
        } else if (type) {
          cartesian = false;
          break;
        }
        ++coordinateIndex;
      }
      if (!cartesian || coordinateIndex != coordinates.size() ||
          resultAxis != payloadFragment.getShape().size())
        std::fill(coordinateAxes.begin(), coordinateAxes.end(), std::nullopt);
    }

    Value valid = existing;
    for (auto [coordinateIndex, coordinate] : llvm::enumerate(coordinates)) {
      int64_t sourceAxis = sourceAxes[coordinateIndex];
      if (sourceAxis < 0 ||
          sourceAxis >= static_cast<int64_t>(proven.size()))
        return failure();
      if (proven[sourceAxis])
        continue;
      FailureOr<Value> extent =
          resourceExtent(operation->getLoc(), resource, sourceAxis);
      if (failed(extent)) {
        operation->emitOpError("indexed resource extent is unavailable")
            << "; resource=" << resource << "; axis=" << sourceAxis;
        return failure();
      }

      Value index = coordinate;
      Value upper = *extent;
      Value zero = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 0);
      bool nonNegative = false;
      if (std::optional<int64_t> constant = integerConstant(coordinate))
        nonNegative = *constant >= 0;
      if (auto range = coordinate.getDefiningOp<gpu::MakeRangeOp>()) {
        std::optional<int64_t> start = integerConstant(range.getStart());
        std::optional<int64_t> step = integerConstant(range.getStep());
        nonNegative = start && step && *start >= 0 && *step > 0;
      }
      FailureOr<Value> logicalIndex =
          asLogicalIndex(operation->getLoc(), index);
      if (failed(logicalIndex))
        return failure();
      index = *logicalIndex;

      Type axisPredicateType = builder.getI1Type();
      if (auto indexType = dyn_cast<gpu::FragmentType>(index.getType())) {
        axisPredicateType = gpu::FragmentType::get(
            operation->getContext(), builder.getI1Type(), indexType.getShape(),
            indexType.getAxisMaps(), indexType.getValidity(),
            indexType.getOwner());
        upper = builder.create<gpu::SplatOp>(operation->getLoc(), indexType,
                                             upper);
        zero = builder.create<gpu::SplatOp>(operation->getLoc(), indexType,
                                            zero);
      }

      Value upperBound = createCompare(
          builder, operation->getLoc(), axisPredicateType, index, upper,
          ComparePredicate::Lt);
      Value axisValid = upperBound;
      if (!nonNegative) {
        Value lowerBound = createCompare(
            builder, operation->getLoc(), axisPredicateType, index, zero,
            ComparePredicate::Ge);
        axisValid = createBinary(builder, operation->getLoc(), axisPredicateType,
                                 lowerBound, upperBound,
                                 BinaryOperator::LogicalAnd);
      }
      if (payloadFragment) {
        auto axis = coordinateAxes[coordinateIndex];
        FailureOr<Value> projected = axis
            ? gpu::projectPredicateToFragmentAxis(builder, operation->getLoc(),
                                                  axisValid, payloadFragment, *axis)
            : projectAccessOperand(operation->getLoc(), axisValid, payloadFragment);
        if (failed(projected)) {
          operation->emitOpError("index bounds have no payload projection")
              << "; index=" << index << "; payload=" << payloadFragment;
          return failure();
        }
        axisValid = *projected;
      }
      if (axisValid.getType() != predicateType)
        return failure();
      valid = valid ? createBinary(builder, operation->getLoc(), predicateType,
                                   valid, axisValid,
                                   BinaryOperator::LogicalAnd)
                    : axisValid;
    }
    return valid;
  }

  FailureOr<Value> zeroAccessFill(Location location, Type type) {
    auto fragment = dyn_cast<gpu::FragmentType>(type);
    Type element = fragment ? fragment.getElementType() : type;
    TypedAttr zero;
    if (auto integer = dyn_cast<IntegerType>(element))
      zero = builder.getIntegerAttr(integer, 0);
    else if (auto floating = dyn_cast<FloatType>(element))
      zero = builder.getFloatAttr(floating, 0.0);
    else if (isa<IndexType>(element))
      zero = builder.getIndexAttr(0);
    if (!zero)
      return failure();
    FailureOr<Value> scalar =
        gpu::materializeScalarConstant(builder, location, zero, element);
    if (failed(scalar))
      return failure();
    if (!fragment)
      return *scalar;
    return Value(builder.create<gpu::SplatOp>(location, fragment, *scalar));
  }

  LogicalResult lower(Operation *operation) {
    if (operation->getNumResults() > 0) {
      bool complete = llvm::all_of(operation->getResults(),
                                   [&](Value result) { return values.count(result); });
      if (complete)
        return success();
    }
    Location location = operation->getLoc();
    if (auto constant = dyn_cast<intent::ConstantOp>(operation)) {
      FailureOr<Value> value = gpu::materializeScalarConstant(
          builder, location, constant.getValue(), constant.getResult().getType());
      if (failed(value))
        return constant.emitOpError(
            "constant cannot be represented by its physical result type");
      values[constant.getResult()] = *value;
      if (Operation *target = value->getDefiningOp())
        attachOrigin(operation, target);
      return success();
    }
    if (auto dim = dyn_cast<intent::DimOp>(operation)) {
      if (values.count(dim.getResult()))
        return success();
      auto binding = dimensions.find(dim.getDimension());
      if (binding != dimensions.end()) {
        values[dim.getResult()] = binding->second;
        return success();
      }
      FailureOr<Value> source = get(dim.getSource());
      if (failed(source))
        return dim.emitOpError(
            "GPU construction lost a launch-visible dimension binding");
      binding = dimensions.find(dim.getDimension());
      if (binding != dimensions.end()) {
        values[dim.getResult()] = binding->second;
        return success();
      }
      if (isa<gpu::RangeType>((*source).getType())) {
        Value start = rangeBound(location, *source, 0);
        Value stop = rangeBound(location, *source, 1);
        Value step = rangeBound(location, *source, 2);
        Value extent = rangeExtent(location, start, stop, step);
        values[dim.getResult()] = extent;
        if (dim.getDimension() > 0)
          dimensions[dim.getDimension()] = extent;
        return success();
      }
      if (auto fragment = dyn_cast<gpu::FragmentType>((*source).getType())) {
        if (dim.getAxis() >= fragment.getShape().size())
          return dim.emitOpError("fragment dimension axis is outside its rank");
        FailureOr<Value> extent = physicalExtentValue(
            location,
            cast<PhysicalExprAttr>(fragment.getShape()[dim.getAxis()]));
        if (failed(extent))
          return dim.emitOpError(
              "fragment dimension has no materialized physical extent");
        values[dim.getResult()] = *extent;
        if (dim.getDimension() > 0)
          dimensions[dim.getDimension()] = *extent;
        return success();
      }
      if (!isa<gpu::ViewType>((*source).getType()))
        return dim.emitOpError(
            "GPU construction has no exact runtime extent for this dimension");
      auto view = cast<gpu::ViewType>((*source).getType());
      auto extent = cast<PhysicalExprAttr>(
          view.getLayout().getExtents()[dim.getAxis()]);
      if (extent.getKind() ==
          PhysicalExprKind::Constant) {
        values[dim.getResult()] = builder.create<arith::ConstantIndexOp>(
            location, extent.getValue());
        return success();
      }
      auto target = builder.create<gpu::DimOp>(location, builder.getIndexType(),
                                               *source, dim.getAxis());
      mapResults(operation, target);
      return success();
    }
    if (auto domain = dyn_cast<intent::DomainOp>(operation)) {
      if (values.count(domain.getResult()))
        return success();
      FailureOr<Value> start = get(domain.getBounds()[0]);
      FailureOr<Value> stop = get(domain.getBounds()[1]);
      FailureOr<Value> step =
          domain.getBounds().size() == 3
              ? get(domain.getBounds()[2])
              : FailureOr<Value>(builder.create<arith::ConstantIndexOp>(
                    location, 1));
      if (failed(start) || failed(stop) || failed(step))
        return domain.emitOpError("physical domain bounds are unavailable");
      start = asIndex(location, *start);
      stop = asIndex(location, *stop);
      step = asIndex(location, *step);
      if (failed(start) || failed(stop) || failed(step))
        return domain.emitOpError(
            "physical domain bounds are not integer coordinates");
      if (domain.getExtentDimensions().size() != 1)
        return domain.emitOpError(
            "physical domain requires one logical dimension identity");
      int64_t identity =
          cast<IntegerAttr>(domain.getExtentDimensions()[0]).getInt();
      if (identity <= 0)
        return domain.emitOpError(
            "physical domain has no logical dimension identity");
      auto type = gpu::RangeType::get(
          operation->getContext(), domain.getResult().getType().getOriginId(), 0,
          identity, /*derived=*/false);
      auto target =
          builder.create<gpu::RangeOp>(location, type, *start, *stop, *step);
      mapResults(operation, target);
      bindExtentDimensions(domain.getExtentDimensions(),
                           rangeExtent(location, *start, *stop, *step));
      return success();
    }
    if (isa<intent::DomainProductOp>(operation))
      return success();
    if (auto subregion = dyn_cast<intent::SubregionOp>(operation)) {
      FailureOr<Value> source = get(subregion.getInputs().front());
      if (failed(source) || !isa<gpu::RangeType>((*source).getType()))
        return subregion.emitOpError(
            "physical subregion source is not an executable range");
      Value start = rangeBound(location, *source, 0);
      Value stop = rangeBound(location, *source, 1);
      Value step = rangeBound(location, *source, 2);
      unsigned operand = 1;
      if (subregion.getHasStart()) {
        FailureOr<Value> explicitStart = get(subregion.getInputs()[operand++]);
        if (failed(explicitStart))
          return failure();
        FailureOr<Value> physicalStart = asIndex(location, *explicitStart);
        if (failed(physicalStart))
          return subregion.emitOpError(
              "subregion start is not an integer coordinate");
        start = *physicalStart;
      }
      if (subregion.getHasStop()) {
        FailureOr<Value> explicitStop = get(subregion.getInputs()[operand]);
        if (failed(explicitStop))
          return failure();
        FailureOr<Value> physicalStop = asIndex(location, *explicitStop);
        if (failed(physicalStop))
          return subregion.emitOpError(
              "subregion stop is not an integer coordinate");
        stop = *physicalStop;
      }
      auto logical = subregion.getResult().getType();
      auto sourceRange = cast<gpu::RangeType>((*source).getType());
      if (subregion.getExtentDimensions().size() != 1)
        return subregion.emitOpError(
            "physical subregion requires one logical extent identity");
      int64_t extentDimension =
          cast<IntegerAttr>(subregion.getExtentDimensions()[0]).getInt();
      if (extentDimension <= 0)
        return subregion.emitOpError(
            "physical subregion has no logical extent identity");
      auto type = gpu::RangeType::get(
          operation->getContext(), logical.getSourceId(), 0,
          extentDimension, sourceRange.getDerived());
      auto target =
          builder.create<gpu::RangeOp>(location, type, start, stop, step);
      if (sourceRange.getDimensionId() <= 0)
        return subregion.emitOpError(
            "physical subregion source has no parent dimension identity");
      target->setAttr(
          gpu::sourceSubregionAttr,
          builder.getI64IntegerAttr(sourceRange.getDimensionId()));
      if (std::optional<int64_t> bound =
              subregionStaticExtentBound(operation, extentDimension))
        target->setAttr(gpu::sourceSubregionBoundAttr,
                        builder.getI64IntegerAttr(*bound));
      mapResults(operation, target);
      bindExtentDimensions(subregion.getExtentDimensions(),
                           rangeExtent(location, start, stop, step));
      return success();
    }
    if (auto end = dyn_cast<intent::RegionEndOp>(operation)) {
      FailureOr<Value> source = get(end.getSource());
      if (failed(source) || !isa<gpu::RangeType>((*source).getType()))
        return end.emitOpError("region end source is not a physical range");
      Value target = rangeBound(location, *source, 1);
      values[end.getResult()] = target;
      return success();
    }
    if (auto indices = dyn_cast<intent::IndicesOp>(operation)) {
      FailureOr<Value> source = get(indices.getSource());
      if (failed(source))
        return indices.emitOpError("indices physical source is unavailable");
      if (auto fragment = dyn_cast<gpu::FragmentType>((*source).getType())) {
        auto axisAttr = operation->getAttrOfType<IntegerAttr>("tensor_axis");
        if (!axisAttr || axisAttr.getInt() < 0 ||
            axisAttr.getInt() >= static_cast<int64_t>(fragment.getShape().size()))
          return indices.emitOpError(
              "tensor indices require one explicit physical source axis");
        unsigned axis = axisAttr.getInt();
        auto mapping =
            cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[axis]);
        auto extent = cast<gpu::PhysicalExprAttr>(fragment.getShape()[axis]);
        FailureOr<Value> physicalExtent =
            physicalExtentValue(location, extent);
        if (failed(physicalExtent))
          return indices.emitOpError(
              "tensor index axis extent is not materialized");
        Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
        Value one = builder.create<arith::ConstantIndexOp>(location, 1);
        Value start = zero, logicalStart = zero, step = one, logicalStop;
        gpu::MakeRangeOp sourceRange;
        gpu::PhysicalRangeFact sourceRanges =
            gpu::PhysicalProgramAnalysis(physicalKernel).axisRanges(*source, axis);
        if (!sourceRanges.roots.empty()) {
          FailureOr<gpu::MakeRangeOp> range =
              gpu::queryExactLogicalRange(sourceRanges);
          if (failed(range))
            return indices.emitOpError(
                "tensor index axis has ambiguous logical ranges");
          sourceRange = *range;
          start = range->getStart();
          logicalStart = range->getLogicalStart();
          logicalStop = range->getLogicalStop();
          step = range->getStep();
        } else {
          auto logical = cast<RankedTensorType>(indices.getSource().getType());
          auto identities = dimensionIds(logical);
          PhysicalExprAttr logicalExtent;
          if (logical.isDynamicDim(axis)) {
            auto function = operation->getParentOfType<func::FuncOp>();
            FailureOr<PhysicalExprAttr> resolved =
                logicalDimensionExpression(function, identities[axis]);
            if (failed(resolved))
              return indices.emitOpError(
                  "tensor index axis has conflicting logical extent relations");
            logicalExtent = *resolved;
          } else {
            logicalExtent = expression(operation->getContext(),
                                       PhysicalExprKind::Constant,
                                       logical.getDimSize(axis));
          }
          FailureOr<Value> stop = physicalExtentValue(location, logicalExtent);
          if (failed(stop))
            return indices.emitOpError(
                "tensor index axis has no logical extent authority");
          logicalStop = *stop;
        }
        auto coordinateType = gpu::FragmentType::get(
            operation->getContext(), builder.getIndexType(),
            builder.getArrayAttr({extent}),
            builder.getArrayAttr({gpu::AxisMapAttr::get(
                operation->getContext(), mapping.getSourceId(),
                mapping.getSourceAxis(), mapping.getDimensionId(), 0,
                mapping.getDerived())}),
            fragment.getValidity(), fragment.getOwner());
        auto coordinateRange = builder.create<gpu::MakeRangeOp>(
            location, coordinateType, start, *physicalExtent, step, logicalStart,
            logicalStop,
            mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDerived());
        if (sourceRange)
          for (StringRef name : {gpu::sourceSubregionAttr,
                                 gpu::sourceSubregionBoundAttr})
            if (Attribute attribute = sourceRange->getAttr(name))
              coordinateRange->setAttr(name, attribute);
        Value coordinate = coordinateRange.getResult();
        Value origin = builder.create<gpu::BroadcastOp>(
            location, coordinateType, logicalStart);
        Value stride = builder.create<gpu::BroadcastOp>(
            location, coordinateType, step);
        coordinate = builder.create<gpu::BinaryOp>(
            location, coordinateType, coordinate, origin, BinaryOperator::Subtract);
        coordinate = builder.create<gpu::BinaryOp>(
            location, coordinateType, coordinate, stride,
            BinaryOperator::FloorDivide);
        auto resultType = gpu::FragmentType::get(
            operation->getContext(), builder.getIndexType(), fragment.getShape(),
            fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
        auto target = builder.create<gpu::BroadcastOp>(location, resultType,
                                                       coordinate);
        mapResults(operation, target);
        return success();
      }
      if (!isa<gpu::RangeType>((*source).getType()))
        return indices.emitOpError(
            "indices source is neither a logical range nor a tensor fragment");
      Value start = rangeBound(location, *source, 0);
      Value stop = rangeBound(location, *source, 1);
      Value step = rangeBound(location, *source, 2);
      FailureOr<Type> result =
          convertDataType(indices.getResult().getType(), operation);
      if (failed(result) || !isa<gpu::FragmentType>(*result))
        return indices.emitOpError("indices result has no physical fragment type");
      auto rangeType = cast<gpu::RangeType>((*source).getType());
      auto genericType = cast<gpu::FragmentType>(*result);
      if (genericType.getShape().size() != 1)
        return indices.emitOpError(
            "range indices require one physical fragment extent");
      FailureOr<Value> extent = physicalExtentValue(
          location, cast<gpu::PhysicalExprAttr>(genericType.getShape()[0]));
      if (failed(extent))
        return indices.emitOpError(
            "range indices physical extent is not materialized");
      auto resultType = gpu::FragmentType::get(
          operation->getContext(), genericType.getElementType(),
          genericType.getShape(),
          builder.getArrayAttr({gpu::AxisMapAttr::get(
              operation->getContext(), rangeType.getSourceId(),
              rangeType.getSourceAxis(), rangeType.getDimensionId(), 0,
              rangeType.getDerived())}),
          genericType.getValidity(), genericType.getOwner());
      auto target = builder.create<gpu::MakeRangeOp>(
          location, resultType, start, *extent, step, start, stop,
          rangeType.getSourceId(), rangeType.getSourceAxis(),
          rangeType.getDerived());
      for (StringRef name : {gpu::sourceSubregionAttr,
                             gpu::sourceSubregionBoundAttr})
        if (Attribute value = (*source).getDefiningOp()->getAttr(name))
          target->setAttr(name, value);
      mapResults(operation, target);
      return success();
    }
    if (auto full = dyn_cast<intent::FullOp>(operation)) {
      FailureOr<Value> fill = get(full.getInputs().front());
      FailureOr<Type> result =
          convertDataType(full.getResult().getType(), operation);
      if (failed(fill) || failed(result) || !isa<gpu::FragmentType>(*result))
        return full.emitOpError("full has no physical fragment realization");
      auto targetType = cast<gpu::FragmentType>(*result);
      SmallVector<Attribute> shape(targetType.getShape().begin(),
                                   targetType.getShape().end());
      SmallVector<Attribute> mappings(targetType.getAxisMaps().begin(),
                                      targetType.getAxisMaps().end());
      for (auto [resultAxis, attribute] :
           llvm::enumerate(full.getShape().getAxes())) {
        auto relation = cast<intent::ShapeExprAttr>(attribute);
        if (relation.getKind() != 1 || relation.getPayload() < 0 ||
            static_cast<size_t>(relation.getPayload()) >=
                full.getInputs().size())
          continue;
        auto dim = full.getInputs()[relation.getPayload()]
                       .getDefiningOp<intent::DimOp>();
        if (!dim)
          continue;
        FailureOr<Value> source = get(dim.getSource());
        auto fragment = succeeded(source)
                            ? dyn_cast<gpu::FragmentType>((*source).getType())
                            : gpu::FragmentType();
        if (!fragment || dim.getAxis() >= fragment.getShape().size())
          continue;
        auto sourceMapping =
            cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[dim.getAxis()]);
        shape[resultAxis] = fragment.getShape()[dim.getAxis()];
        mappings[resultAxis] = gpu::AxisMapAttr::get(
            operation->getContext(), sourceMapping.getSourceId(),
            sourceMapping.getSourceAxis(), sourceMapping.getDimensionId(),
            resultAxis, sourceMapping.getDerived());
      }
      targetType = gpu::FragmentType::get(
          operation->getContext(), targetType.getElementType(),
          builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
          targetType.getValidity(), targetType.getOwner());
      auto target =
          builder.create<gpu::SplatOp>(location, targetType, *fill);
      mapResults(operation, target);
      return success();
    }
    if (auto broadcastOp = dyn_cast<intent::BroadcastOp>(operation)) {
      FailureOr<Value> input = get(broadcastOp.getInputs().front());
      FailureOr<Type> result =
          convertDataType(broadcastOp.getResult().getType(), operation);
      if (failed(input) || failed(result) || !isa<gpu::FragmentType>(*result))
        return broadcastOp.emitOpError(
            "broadcast has no physical fragment realization");
      auto targetType = cast<gpu::FragmentType>(*result);
      auto sourceType = dyn_cast<gpu::FragmentType>((*input).getType());
      auto logicalSource =
          dyn_cast<RankedTensorType>(broadcastOp.getInputs().front().getType());
      auto logicalResult =
          dyn_cast<RankedTensorType>(broadcastOp.getResult().getType());
      if (sourceType && logicalResult) {
        unsigned logicalSourceRank = logicalSource ? logicalSource.getRank() : 0;
        if (logicalSourceRank > static_cast<unsigned>(logicalResult.getRank()))
          return broadcastOp.emitOpError(
              "broadcast source rank exceeds result rank");
        if (sourceType.getShape().size() < logicalSourceRank)
          return broadcastOp.emitOpError(
              "broadcast source lost a logical physical axis");

        // A scalar or tensor value can already be lifted over the current
        // physical workset before the logical broadcast is constructed.  Such
        // axes are not part of the logical source rank, but they remain part of
        // the value and therefore must survive the broadcast.  Prefix the
        // source-only workset axes, then retain the logical result axis order.
        SmallVector<Attribute> shape;
        SmallVector<Attribute> mappings;
        auto appendAxis = [&](Attribute extent, gpu::AxisMapAttr mapping) {
          mappings.push_back(gpu::AxisMapAttr::get(
              operation->getContext(), mapping.getSourceId(),
              mapping.getSourceAxis(), mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
          shape.push_back(extent);
        };
        unsigned worksetRank = sourceType.getShape().size() - logicalSourceRank;
        for (unsigned sourceAxis = 0; sourceAxis < worksetRank; ++sourceAxis)
          appendAxis(sourceType.getShape()[sourceAxis],
                     cast<gpu::AxisMapAttr>(
                         sourceType.getAxisMaps()[sourceAxis]));
        unsigned offset = logicalResult.getRank() - logicalSourceRank;
        for (auto [resultAxis, attribute] :
             llvm::enumerate(targetType.getAxisMaps())) {
          auto resultMapping = cast<gpu::AxisMapAttr>(attribute);
          Attribute extent = targetType.getShape()[resultAxis];
          if (logicalSource && resultAxis >= offset) {
            unsigned logicalSourceAxis = resultAxis - offset;
            unsigned physicalSourceAxis = worksetRank + logicalSourceAxis;
            bool expandsSingleton =
                !logicalSource.isDynamicDim(logicalSourceAxis) &&
                logicalSource.getDimSize(logicalSourceAxis) == 1 &&
                (logicalResult.isDynamicDim(resultAxis) ||
                 logicalResult.getDimSize(resultAxis) != 1);
            if (!expandsSingleton)
              extent = sourceType.getShape()[physicalSourceAxis];
            if (!expandsSingleton) {
              auto sourceMapping = cast<gpu::AxisMapAttr>(
                  sourceType.getAxisMaps()[physicalSourceAxis]);
              resultMapping = gpu::AxisMapAttr::get(
                  operation->getContext(), sourceMapping.getSourceId(),
                  sourceMapping.getSourceAxis(), sourceMapping.getDimensionId(),
                  resultAxis, sourceMapping.getDerived());
            }
          }
          appendAxis(extent, resultMapping);
        }
        targetType = gpu::FragmentType::get(
            operation->getContext(), targetType.getElementType(),
            builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
            targetType.getValidity(), sourceType.getOwner());
      }
      auto target =
          builder.create<gpu::BroadcastOp>(location, targetType, *input);
      mapResults(operation, target);
      return success();
    }
    if (auto reshape = dyn_cast<intent::ReshapeOp>(operation)) {
      FailureOr<Value> input = get(reshape.getInputs().front());
      auto source = succeeded(input)
                        ? dyn_cast<gpu::FragmentType>((*input).getType())
                        : gpu::FragmentType();
      auto logicalSource =
          dyn_cast<RankedTensorType>(reshape.getInputs().front().getType());
      auto logicalResult =
          dyn_cast<RankedTensorType>(reshape.getResult().getType());
      if (failed(input) || !source || !logicalSource || !logicalResult)
        return reshape.emitOpError(
            "reshape has no physical fragment realization");
      unsigned physicalSourceRank = source.getShape().size();
      unsigned logicalSourceRank = logicalSource.getRank();
      if (physicalSourceRank < logicalSourceRank)
        return reshape.emitOpError("reshape source lost a logical physical axis");
      unsigned worksetRank = physicalSourceRank - logicalSourceRank;
      auto resultMapping = [&](unsigned logicalResultAxis,
                               unsigned physicalResultAxis)
          -> FailureOr<Attribute> {
        DenseI64ArrayAttr dimensions = dimensionIds(logicalResult);
        FailureOr<PhysicalAxisIdentity> identity =
            resultAxisIdentity(operation, /*resultIndex=*/0, logicalResultAxis);
        if (!dimensions || logicalResultAxis >= dimensions.size() ||
            dimensions[logicalResultAxis] <= 0 || failed(identity))
          return failure();
        return Attribute(gpu::AxisMapAttr::get(
            operation->getContext(), identity->sourceId, identity->sourceAxis,
            dimensions[logicalResultAxis], physicalResultAxis,
            identity->derived));
      };
      SmallVector<Attribute> shape;
      SmallVector<Attribute> mappings;
      for (unsigned axis = 0; axis < worksetRank; ++axis) {
        shape.push_back(source.getShape()[axis]);
        auto mapping = cast<gpu::AxisMapAttr>(source.getAxisMaps()[axis]);
        mappings.push_back(gpu::AxisMapAttr::get(
            operation->getContext(), mapping.getSourceId(),
            mapping.getSourceAxis(), mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
      }
      unsigned resultRank = logicalResult.getRank();
      ArrayAttr relation = reshape.getShape().getAxes();
      if (!relation || relation.size() != resultRank)
        return reshape.emitOpError(
            "reshape result has no complete canonical shape relation");
      func::FuncOp function = operation->getParentOfType<func::FuncOp>();
      DenseI64ArrayAttr logicalSourceDimensions = dimensionIds(logicalSource);
      if (!function || !logicalSourceDimensions ||
          logicalSourceDimensions.size() !=
              static_cast<int64_t>(logicalSourceRank))
        return reshape.emitOpError(
            "reshape source has no complete logical extent relation");
      SmallVector<PhysicalExprAttr> logicalSourceExtents;
      for (unsigned axis = 0; axis < logicalSourceRank; ++axis) {
        PhysicalExprAttr extent;
        if (!logicalSource.isDynamicDim(axis)) {
          extent = expression(operation->getContext(),
                              PhysicalExprKind::Constant,
                              logicalSource.getDimSize(axis));
        } else if (logicalSourceDimensions[axis] > 0) {
          FailureOr<PhysicalExprAttr> resolved = logicalDimensionExpression(
              function, logicalSourceDimensions[axis]);
          if (failed(resolved))
            return reshape.emitOpError(
                "reshape source dimension has conflicting logical extent relations");
          extent = *resolved;
        } else {
          return reshape.emitOpError(
              "reshape source axis has no logical extent identity");
        }
        logicalSourceExtents.push_back(extent);
      }

      SmallVector<PhysicalExprAttr> logicalResultExtents;
      std::optional<unsigned> inferredAxis;
      PhysicalExprAttr knownResultProduct = expression(
          operation->getContext(), PhysicalExprKind::Constant, 1);
      for (auto [axis, attribute] : llvm::enumerate(relation)) {
        auto extent = dyn_cast<intent::ShapeExprAttr>(attribute);
        if (!extent)
          return reshape.emitOpError(
              "reshape result shape relation has an invalid entry");
        PhysicalExprAttr physical;
        if (extent.getKind() == 0) {
          physical = expression(operation->getContext(),
                                PhysicalExprKind::Constant,
                                extent.getPayload());
        } else if (extent.getKind() == 1) {
          int64_t operand = extent.getPayload();
          if (operand < 0 ||
              operand >= static_cast<int64_t>(operation->getNumOperands()))
            return reshape.emitOpError(
                "reshape shape relation references an invalid extent operand");
          if (extent.getDimension() <= 0)
            return reshape.emitOpError(
                "reshape result extent has no typed logical expression");
          FailureOr<PhysicalExprAttr> resolved = logicalDimensionExpression(
              function, extent.getDimension());
          if (failed(resolved))
            return reshape.emitOpError(
                "reshape result dimension has conflicting logical extent relations");
          physical = *resolved;
        } else if (extent.getKind() == 2) {
          if (inferredAxis)
            return reshape.emitOpError(
                "reshape has more than one inferred logical extent");
          inferredAxis = axis;
          logicalResultExtents.push_back(PhysicalExprAttr());
          continue;
        } else {
          return reshape.emitOpError(
              "reshape shape relation uses an unknown extent kind");
        }
        logicalResultExtents.push_back(physical);
        knownResultProduct = binaryExpression(
            operation->getContext(), PhysicalExprKind::Multiply,
            knownResultProduct, physical);
      }
      PhysicalExprAttr logicalSourceProduct = expression(
          operation->getContext(), PhysicalExprKind::Constant, 1);
      for (PhysicalExprAttr extent : logicalSourceExtents)
        logicalSourceProduct = binaryExpression(
            operation->getContext(), PhysicalExprKind::Multiply,
            logicalSourceProduct, extent);
      if (inferredAxis) {
        logicalResultExtents[*inferredAxis] = binaryExpression(
              operation->getContext(), PhysicalExprKind::FloorDiv,
              logicalSourceProduct, knownResultProduct);
      }

      FailureOr<ArrayAttr> reassociation = gpu::inferReshapeReassociation(
          operation->getContext(),
          SmallVector<Attribute>(logicalSourceExtents.begin(),
                                 logicalSourceExtents.end()),
          SmallVector<Attribute>(logicalResultExtents.begin(),
                                 logicalResultExtents.end()));
      if (failed(reassociation))
        return reshape.emitOpError(
            "canonical reshape has no exact row-major logical relation");

      SmallVector<PhysicalExprAttr> physicalResultExtents(resultRank);
      SmallVector<Attribute> physicalResultMappings;
      physicalResultMappings.reserve(resultRank);
      for (unsigned axis = 0; axis < resultRank; ++axis) {
        FailureOr<Attribute> mapping =
            resultMapping(axis, worksetRank + axis);
        if (failed(mapping))
          return reshape.emitOpError(
              "reshape result axis has no coordinate identity");
        physicalResultMappings.push_back(*mapping);
      }
      for (Attribute attribute : *reassociation) {
        auto group = cast<gpu::ReshapeGroupAttr>(attribute);
        PhysicalExprAttr sourceProduct = expression(
            operation->getContext(), PhysicalExprKind::Constant, 1);
        for (int64_t logicalSourceAxis :
             group.getSourceAxes().asArrayRef())
          sourceProduct = binaryExpression(
              operation->getContext(), PhysicalExprKind::Multiply,
              sourceProduct,
              cast<PhysicalExprAttr>(
                  source.getShape()[worksetRank + logicalSourceAxis]));
        PhysicalExprAttr knownProduct = expression(
            operation->getContext(), PhysicalExprKind::Constant, 1);
        SmallVector<unsigned> unresolved;
        for (int64_t logicalResultAxis :
             group.getResultAxes().asArrayRef()) {
          PhysicalExprAttr logicalExtent =
              logicalResultExtents[logicalResultAxis];
          if (logicalExtent.getKind() ==
                  PhysicalExprKind::Constant &&
              logicalExtent.getValue() == 1) {
            physicalResultExtents[logicalResultAxis] = logicalExtent;
            continue;
          }
          std::optional<unsigned> matchedSource;
          for (int64_t logicalSourceAxis :
               group.getSourceAxes().asArrayRef()) {
            if (logicalSourceExtents[logicalSourceAxis] != logicalExtent)
              continue;
            if (matchedSource) {
              matchedSource.reset();
              break;
            }
            matchedSource = logicalSourceAxis;
          }
          if (!matchedSource) {
            unresolved.push_back(logicalResultAxis);
            continue;
          }
          unsigned sourceAxis = worksetRank + *matchedSource;
          PhysicalExprAttr physicalExtent = cast<PhysicalExprAttr>(
              source.getShape()[sourceAxis]);
          physicalResultExtents[logicalResultAxis] = physicalExtent;
          knownProduct = binaryExpression(
              operation->getContext(), PhysicalExprKind::Multiply,
              knownProduct, physicalExtent);
          auto mapping = cast<gpu::AxisMapAttr>(
              source.getAxisMaps()[sourceAxis]);
          physicalResultMappings[logicalResultAxis] = gpu::AxisMapAttr::get(
              operation->getContext(), mapping.getSourceId(),
              mapping.getSourceAxis(), mapping.getDimensionId(),
              worksetRank + logicalResultAxis, mapping.getDerived());
        }
        if (unresolved.size() > 1) {
          auto knownKind =
              knownProduct.getKind();
          bool allConstant = knownKind == PhysicalExprKind::Constant;
          int64_t logicalConstantProduct =
              allConstant ? knownProduct.getValue() : 0;
          for (unsigned axis : unresolved) {
            PhysicalExprAttr logicalExtent = logicalResultExtents[axis];
            if (logicalExtent.getKind() !=
                PhysicalExprKind::Constant) {
              allConstant = false;
              continue;
            }
            if (allConstant)
              logicalConstantProduct *= logicalExtent.getValue();
          }
          auto sourceKind = sourceProduct.getKind();
          if (allConstant && sourceKind == PhysicalExprKind::Constant &&
              logicalConstantProduct == sourceProduct.getValue()) {
            for (unsigned axis : unresolved)
              physicalResultExtents[axis] = logicalResultExtents[axis];
            knownProduct = sourceProduct;
            unresolved.clear();
          } else if (sourceKind == PhysicalExprKind::Constant &&
                     sourceProduct.getValue() == 1) {
            PhysicalExprAttr one = expression(
                operation->getContext(), PhysicalExprKind::Constant, 1);
            for (unsigned axis : unresolved)
              physicalResultExtents[axis] = one;
            knownProduct = sourceProduct;
            unresolved.clear();
          } else {
            return reshape.emitOpError(
                "reshape physical group has more than one unbound result axis");
          }
        }
        if (unresolved.size() == 1) {
          physicalResultExtents[unresolved.front()] = binaryExpression(
              operation->getContext(), PhysicalExprKind::FloorDiv,
              sourceProduct, knownProduct);
          knownProduct = sourceProduct;
        }
        if (knownProduct != sourceProduct)
          return reshape.emitOpError(
              "reshape physical group does not preserve row-major elements");
      }
      for (auto [axis, extent] : llvm::enumerate(physicalResultExtents)) {
        if (!extent)
          return reshape.emitOpError(
              "reshape physical result axis has no extent decision");
        shape.push_back(extent);
        mappings.push_back(physicalResultMappings[axis]);
      }

      auto result = gpu::FragmentType::get(
          operation->getContext(), source.getElementType(),
          builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
          source.getValidity(), source.getOwner());
      auto target = builder.create<gpu::ReshapeOp>(
          location, result, *input, *reassociation);
      mapResults(operation, target);
      return success();
    }
    if (auto transpose = dyn_cast<intent::TransposeOp>(operation)) {
      FailureOr<Value> input = get(transpose.getInput());
      auto source = succeeded(input)
                        ? dyn_cast<gpu::FragmentType>((*input).getType())
                        : gpu::FragmentType();
      auto logical = dyn_cast<RankedTensorType>(transpose.getInput().getType());
      if (failed(input) || !source || !logical ||
          source.getShape().size() < static_cast<size_t>(logical.getRank()))
        return transpose.emitOpError(
            "transpose has no physical fragment realization");
      SmallVector<int64_t> permutation;
      for (Attribute axis : transpose.getPermutation())
        permutation.push_back(cast<IntegerAttr>(axis).getInt());
      unsigned prefix = source.getShape().size() - logical.getRank();
      SmallVector<Attribute> shape;
      SmallVector<Attribute> mappings;
      auto appendAxis = [&](unsigned sourceAxis) {
        shape.push_back(source.getShape()[sourceAxis]);
        auto mapping =
            cast<gpu::AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
        mappings.push_back(gpu::AxisMapAttr::get(
            operation->getContext(), mapping.getSourceId(),
            mapping.getSourceAxis(), mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
      };
      for (unsigned axis = 0; axis < prefix; ++axis)
        appendAxis(axis);
      SmallVector<int64_t> physicalPermutation;
      for (unsigned axis = 0; axis < prefix; ++axis)
        physicalPermutation.push_back(axis);
      for (int64_t axis : permutation) {
        if (axis < 0 || axis >= logical.getRank())
          return transpose.emitOpError(
              "transpose permutation is outside the logical rank");
        appendAxis(prefix + axis);
        physicalPermutation.push_back(prefix + axis);
      }
      auto result = gpu::FragmentType::get(
          operation->getContext(), source.getElementType(),
          builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
          source.getValidity(), source.getOwner());
      auto target = builder.create<gpu::TransposeOp>(
          location, result, *input, physicalPermutation);
      mapResults(operation, target);
      return success();
    }
    if (auto join = dyn_cast<intent::JoinOp>(operation)) {
      FailureOr<Value> lhs = get(join.getLhs());
      FailureOr<Value> rhs = get(join.getRhs());
      if (failed(lhs) || failed(rhs) ||
          failed(alignElementwiseOperands(builder, location, *lhs, *rhs)))
        return join.emitOpError(
            "join operands have no common physical fragment schema");
      auto left = succeeded(lhs) ? dyn_cast<gpu::FragmentType>((*lhs).getType())
                                 : gpu::FragmentType();
      auto right = succeeded(rhs) ? dyn_cast<gpu::FragmentType>((*rhs).getType())
                                  : gpu::FragmentType();
      auto logicalLeft = dyn_cast<RankedTensorType>(join.getLhs().getType());
      auto logicalRight = dyn_cast<RankedTensorType>(join.getRhs().getType());
      auto logical = dyn_cast<RankedTensorType>(join.getResult().getType());
      if (!left || !right || left != right ||
          !logicalLeft || !logicalRight || !logical ||
          logicalLeft.getRank() != logicalRight.getRank() ||
          logical.getRank() != logicalLeft.getRank() + 1)
        return join.emitOpError("join has no physical fragment realization")
               << "; lhs=" << (left ? Type(left) : Type())
               << ", rhs=" << (right ? Type(right) : Type())
               << ", logical_lhs=" << join.getLhs().getType()
               << ", logical_result=" << join.getResult().getType();
      FailureOr<PhysicalAxisIdentity> trailingIdentity = resultAxisIdentity(
          operation, /*resultIndex=*/0, logical.getRank() - 1);
      DenseI64ArrayAttr dimensions = dimensionIds(logical);
      if (!dimensions || dimensions.size() != logical.getRank() ||
          dimensions[logical.getRank() - 1] <= 0 || failed(trailingIdentity))
        return join.emitOpError(
            "join trailing axis has no canonical coordinate identity");
      SmallVector<Attribute> shape(left.getShape().begin(),
                                   left.getShape().end());
      shape.push_back(expression(operation->getContext(),
                                 PhysicalExprKind::Constant, 2));
      SmallVector<Attribute> mappings(left.getAxisMaps().begin(),
                                      left.getAxisMaps().end());
      mappings.push_back(gpu::AxisMapAttr::get(
          operation->getContext(), trailingIdentity->sourceId,
          trailingIdentity->sourceAxis,
          dimensions[logical.getRank() - 1],
          left.getShape().size(), trailingIdentity->derived));
      auto result = gpu::FragmentType::get(
          operation->getContext(), left.getElementType(),
          builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
          left.getValidity(), left.getOwner());
      auto target = builder.create<gpu::JoinOp>(
          location, result, *lhs, *rhs, result.getShape().size() - 1);
      mapResults(operation, target);
      return success();
    }
    if (auto unary = dyn_cast<intent::UnaryOp>(operation)) {
      FailureOr<Value> input = get(unary.getInput());
      if (failed(input))
        return failure();
      FailureOr<Type> result =
          elementwiseResultType(unary.getResult().getType(), operation, *input);
      if (failed(result))
        return unary.emitOpError("unary value has no physical data type");
      auto target = builder.create<gpu::UnaryOp>(location, *result, *input,
          unary.getOperatorKind(), unary.getApproximate(), unary.getFlushToZero());
      mapResults(operation, target);
      return success();
    }
    if (auto binary = dyn_cast<intent::BinaryOp>(operation)) {
      FailureOr<Value> lhs = get(binary.getLhs());
      FailureOr<Value> rhs = get(binary.getRhs());
      if (failed(lhs) || failed(rhs))
        return binary.emitOpError("binary operands have no physical data schema");
      if (failed(alignElementwiseOperands(
              builder, location, *lhs, *rhs, binary.getLhs().getType(),
              binary.getRhs().getType())))
        return binary.emitOpError(
                   "binary operands have incompatible physical schemas; lhs=")
               << (*lhs).getType() << ", rhs=" << (*rhs).getType();
      FailureOr<Type> result =
          elementwiseResultType(binary.getResult().getType(), operation, *lhs);
      if (failed(result))
        return binary.emitOpError("binary result has no physical data schema");
      auto target = builder.create<gpu::BinaryOp>(location, *result, *lhs, *rhs,
          binary.getOperatorKind(), binary.getApproximate(), binary.getFlushToZero());
      mapResults(operation, target);
      return success();
    }
    if (auto compare = dyn_cast<intent::CompareOp>(operation)) {
      FailureOr<Value> lhs = get(compare.getLhs());
      FailureOr<Value> rhs = get(compare.getRhs());
      if (failed(lhs) || failed(rhs) ||
          failed(alignElementwiseOperands(
              builder, location, *lhs, *rhs, compare.getLhs().getType(),
              compare.getRhs().getType())))
        return compare.emitOpError("comparison operands are unavailable");
      // A comparison changes only the element type.  Its predicate executes
      // over exactly the same physical fragment as the aligned operands.
      FailureOr<Type> result =
          elementwiseResultType(compare.getResult().getType(), operation, *lhs);
      if (failed(result))
        return compare.emitOpError("comparison result is unavailable");
      auto target = builder.create<gpu::CompareOp>(location, *result, *lhs, *rhs,
                                                    compare.getPredicate());
      mapResults(operation, target);
      return success();
    }
    if (auto select = dyn_cast<intent::SelectOp>(operation)) {
      FailureOr<Value> condition = get(select.getCondition());
      FailureOr<Value> trueValue = get(select.getTrueValue());
      FailureOr<Value> falseValue = get(select.getFalseValue());
      if (failed(condition) || failed(trueValue) || failed(falseValue) ||
          failed(alignElementwiseOperands(
              builder, location, *trueValue, *falseValue,
              select.getTrueValue().getType(),
              select.getFalseValue().getType())))
        return select.emitOpError("select operands are unavailable");
      if (auto target = dyn_cast<gpu::FragmentType>((*trueValue).getType())) {
        auto predicate = dyn_cast<gpu::FragmentType>((*condition).getType());
        if (predicate && !samePhysicalShape(predicate, target)) {
          auto logicalPredicate = dyn_cast<RankedTensorType>(select.getCondition().getType());
          auto logicalValue = dyn_cast<RankedTensorType>(select.getTrueValue().getType());
          auto predicateTarget = gpu::FragmentType::get(
              predicate.getContext(), predicate.getElementType(), target.getShape(),
              target.getAxisMaps(), target.getValidity(), target.getOwner());
          // Canonical select operands are already broadcast by logical axis.
          // Preserve that positional value instead of stripping its broadcast
          // and matching a reused row coordinate to the wrong matrix axis.
          bool positional = logicalPredicate && logicalValue &&
              dimensionIds(logicalPredicate) == dimensionIds(logicalValue) &&
              predicate.getShape() == target.getShape();
          FailureOr<Value> aligned = positional
              ? projectPositionalValue(builder, location, *condition, predicateTarget)
              : retargetBroadcast(builder, location, *condition, target);
          if (failed(aligned))
            return select.emitOpError(
                       "select predicate cannot adopt the selected physical axes: ")
                   << predicate << " vs " << target;
          *condition = *aligned;
        }
      }
      FailureOr<Type> result = elementwiseResultType(
          select.getResult().getType(), operation, *trueValue);
      if (failed(result))
        return select.emitOpError("select result is unavailable");
      auto target = builder.create<gpu::SelectOp>(
          location, *result, *condition, *trueValue, *falseValue);
      mapResults(operation, target);
      return success();
    }
    if (auto cast = dyn_cast<intent::CastOp>(operation)) {
      FailureOr<Value> input = get(cast.getInput());
      FailureOr<Type> result =
          failed(input) ? FailureOr<Type>(failure())
                        : elementwiseResultType(cast.getResult().getType(),
                                                operation, *input);
      if (failed(input) || failed(result))
        return cast.emitOpError("cast input/result is unavailable");
      if (!isa<gpu::FragmentType>(*result)) {
        if (auto constant = (*input).getDefiningOp<arith::ConstantOp>()) {
          FailureOr<Value> folded = gpu::materializeScalarConstant(
              builder, location, constant.getValue(), *result);
          if (succeeded(folded)) {
            values[cast.getResult()] = *folded;
            return success();
          }
        }
      }
      auto target = builder.create<gpu::CastOp>(location, *result, *input);
      mapResults(operation, target);
      return success();
    }
    if (auto bitcast = dyn_cast<intent::BitcastOp>(operation)) {
      FailureOr<Value> input = get(bitcast.getInput());
      FailureOr<Type> result =
          failed(input) ? FailureOr<Type>(failure())
                        : elementwiseResultType(bitcast.getResult().getType(),
                                                operation, *input);
      if (failed(input) || failed(result))
        return bitcast.emitOpError("bitcast input/result is unavailable");
      auto target = builder.create<gpu::BitcastOp>(location, *result, *input);
      mapResults(operation, target);
      return success();
    }
    if (auto mask = dyn_cast<intent::MaskOp>(operation)) {
      FailureOr<Value> value = get(mask.getValue());
      FailureOr<Value> predicate = get(mask.getPredicate());
      FailureOr<Value> fill = get(mask.getFill());
      if (failed(value) || failed(predicate) || failed(fill) ||
          failed(alignElementwiseOperands(
              builder, location, *value, *fill, mask.getValue().getType(),
              mask.getFill().getType())))
        return mask.emitOpError("mask operands are unavailable");
      if (auto targetType = dyn_cast<gpu::FragmentType>((*value).getType())) {
        auto predicateType =
            dyn_cast<gpu::FragmentType>((*predicate).getType());
        if (predicateType && !samePhysicalShape(predicateType, targetType)) {
          FailureOr<Value> aligned =
              retargetBroadcast(builder, location, *predicate, targetType);
          if (failed(aligned))
            return mask.emitOpError(
                "mask predicate cannot adopt the masked physical axes");
          *predicate = *aligned;
        }
      }
      FailureOr<Type> result =
          failed(value) ? FailureOr<Type>(failure())
                        : elementwiseResultType(mask.getResult().getType(),
                                                operation, *value);
      if (failed(result))
        return mask.emitOpError("mask operands are unavailable");
      auto target = builder.create<gpu::SelectOp>(
          location, *result, *predicate, *value, *fill);
      mapResults(operation, target);
      return success();
    }
    if (auto schema = dyn_cast<StructuredOpInterface>(operation)) {
      auto lowerOperands = [&](ValueRange operands) -> FailureOr<SmallVector<Value>> {
        SmallVector<Value> values;
        for (Value operand : operands) {
          auto value = get(operand);
          if (failed(value))
            return operation->emitOpError("structured physical operand is unavailable"), failure();
          values.push_back(*value);
        }
        return values;
      };
      auto sources = lowerOperands(schema.getSources());
      auto identities = lowerOperands(schema.getIdentities());
      auto initialStates = lowerOperands(schema.getInitialStates());
      auto captures = lowerOperands(schema.getCaptures());
      if (failed(sources) || failed(identities) || failed(initialStates) || failed(captures))
        return failure();
      auto kind = schema.getStructuredKind();
      auto axes = schema.getIterationAxes();
      const bool region = schema.getSummarizeRegion() != nullptr;
      const bool scan = kind == StructuredOpKind::RegionScan;
      SmallVector<Type> results;
      for (auto [index, logical] : llvm::enumerate(operation->getResultTypes())) {
        FailureOr<Type> converted = failure();
        if (kind == StructuredOpKind::Reduce) {
          if (index >= sources->size())
            return operation->emitOpError("reduce result has no corresponding physical source");
          converted = convertReductionResultType(logical, operation, (*sources)[index], axes, index);
        } else {
          std::optional<Type> prototype;
          if (kind == StructuredOpKind::Scan) {
            if (index >= sources->size())
              return operation->emitOpError("scan result has no corresponding physical source");
            prototype = (*sources)[index].getType();
          } else if (kind == StructuredOpKind::RegionFold) {
            prototype = (*identities)[index].getType();
          } else if (index >= schema.getEmittedResults().size()) {
            unsigned state = index - schema.getEmittedResults().size();
            if (state >= initialStates->size())
              return operation->emitOpError("region-scan final-state result has no matching state operand");
            prototype = (*initialStates)[state].getType();
          }
          converted = convertDataType(logical, operation, prototype, /*owner=*/1, index);
        }
        if (failed(converted))
          return operation->emitOpError("structured result has no physical schema")
              << "; result=" << index << "; logical_type=" << logical;
        results.push_back(*converted);
      }
      if (!scan) {
        for (auto [index, identity] : llvm::enumerate(*identities)) {
          auto projected = projectPositionalValue(builder, location, identity, results[index]);
          if (failed(projected))
            return operation->emitOpError("structured identity cannot adopt its physical accumulator schema");
          (*identities)[index] = *projected;
        }
      }
      gpu::ParameterAttr segment;
      if (region) {
        auto parameter = getOrCreateRegionSegment(operation);
        if (failed(parameter)) return failure();
        segment = *parameter;
      }
      auto created = [&]() -> FailureOr<Operation *> {
        switch (kind) {
        case StructuredOpKind::Reduce:
          return builder.create<gpu::ReduceOp>(location, *sources, *identities,
              *captures, axes).getOperation();
        case StructuredOpKind::Scan: {
          auto logical = cast<intent::ScanOp>(operation);
          return builder.create<gpu::ScanOp>(location, *sources, *identities, *captures,
              logical.getAxis(), logical.getInclusive(), logical.getReverse()).getOperation();
        }
        case StructuredOpKind::RegionFold:
          return builder.create<gpu::RegionFoldOp>(location, results, *sources, *identities,
              *captures, axes.front(), segment.getReference()).getOperation();
        case StructuredOpKind::RegionScan:
          return builder.create<gpu::RegionScanOp>(location,
              TypeRange(results).take_front(schema.getEmittedResults().size()),
              TypeRange(results).drop_front(schema.getEmittedResults().size()),
              *sources, *identities, *initialStates, *captures, axes.front(), segment.getReference()).getOperation();
        }
        return operation->emitOpError("unknown structured operation kind"), failure();
      }();
      if (failed(created)) return failure();
      Operation *target = *created;
      auto physical = cast<StructuredOpInterface>(target);
      if (!region) {
        if (failed(lowerPureRegion(schema.getCombine(), physical.getCombine(),
                physical.getCombineArgumentTypes(), results)))
          return failure();
      } else {
        SmallVector<Type> slices;
        for (auto [formal, value] : llvm::zip(schema.getSummarizeSources(), *sources)) {
          auto tensor = dyn_cast<RankedTensorType>(formal.getType());
          auto prototype = dyn_cast<FragmentType>(value.getType());
          auto converted = tensor && prototype
              ? convertSegmentSliceType(tensor, prototype, axes.front(), segment.getName().getValue())
              : FailureOr<FragmentType>(failure());
          if (failed(converted))
            return operation->emitOpError("region source slice has no physical fragment schema");
          slices.push_back(*converted);
        }
        SmallVector<Type> summaryTypes;
        if (scan) llvm::append_range(summaryTypes, TypeRange(*identities));
        else summaryTypes = results;
        if (failed(lowerPureRegion(*schema.getSummarizeRegion(), *physical.getSummarizeRegion(),
                physical.getSummarizeArgumentTypes(slices), summaryTypes))) return failure();
        if (failed(lowerPureRegion(schema.getCombine(), physical.getCombine(),
                physical.getCombineArgumentTypes(), summaryTypes))) return failure();
        if (scan) {
          SmallVector<Type> stateTypes;
          llvm::append_range(stateTypes, TypeRange(*initialStates));
          if (failed(lowerPureRegion(*schema.getApplyRegion(), *physical.getApplyRegion(),
                  physical.getApplyArgumentTypes(), stateTypes))) return failure();
          if (failed(lowerPureRegion(*schema.getEmitRegion(), *physical.getEmitRegion(),
                  physical.getEmitArgumentTypes(slices))))
            return failure();
          auto emitted = physical.getEmitYields();
          if (emitted.size() != physical.getEmittedResults().size())
            return operation->emitOpError("region-scan emitter has no complete physical output schema");
          for (auto [index, result, slice] : llvm::enumerate(physical.getEmittedResults(), emitted)) {
            auto assembled = regionAssemblyType(result.getType(), slice.getType());
            if (failed(assembled))
              return operation->emitOpError("region-scan emitted slice cannot define its assembled output relation")
                  << "; result_index=" << index << "; slice=" << slice.getType()
                  << "; result=" << result.getType();
            result.setType(*assembled);
          }
        }
      }
      mapResults(operation, target);
      return success();
    }
    auto axisPairs = [&](ArrayAttr pairs, SmallVectorImpl<int64_t> &lhs,
                         SmallVectorImpl<int64_t> &rhs) {
      for (Attribute attribute : pairs) {
        auto pair = cast<ArrayAttr>(attribute);
        lhs.push_back(cast<IntegerAttr>(pair[0]).getInt());
        rhs.push_back(cast<IntegerAttr>(pair[1]).getInt());
      }
    };
    auto zeroAccumulator = [&](FragmentType type) -> FailureOr<Value> {
      Type element = type.getElementType();
      Value zero;
      if (isa<FloatType>(element))
        zero = builder.create<arith::ConstantOp>(
            location, element, builder.getFloatAttr(element, 0.0));
      else if (auto integer = dyn_cast<IntegerType>(element))
        zero = builder.create<arith::ConstantOp>(
            location, element, builder.getIntegerAttr(integer, 0));
      else
        return failure();
      return Value(builder.create<gpu::SplatOp>(location, type, zero));
    };
    if (auto contract = dyn_cast<intent::ContractOp>(operation)) {
      FailureOr<Value> lhs = get(contract.getLhs());
      FailureOr<Value> rhs = get(contract.getRhs());
      SmallVector<int64_t> lhsReduction, rhsReduction, lhsBatch, rhsBatch;
      axisPairs(contract.getReduce(), lhsReduction, rhsReduction);
      axisPairs(contract.getBatch(), lhsBatch, rhsBatch);
      FailureOr<Type> result = failed(lhs) || failed(rhs)
                                   ? FailureOr<Type>(failure())
                                     : convertContractResultType(
                                         contract.getResult().getType(), *lhs, *rhs,
                                         lhsReduction, rhsReduction, lhsBatch,
                                         rhsBatch);
      if (failed(lhs) || failed(rhs) || failed(result) ||
          (succeeded(lhs) && !isa<gpu::FragmentType>((*lhs).getType())) ||
          (succeeded(rhs) && !isa<gpu::FragmentType>((*rhs).getType())) ||
          (succeeded(result) && !isa<gpu::FragmentType>(*result))) {
        InFlightDiagnostic diagnostic = contract.emitOpError(
            "contract operands/results have no physical fragment schema");
        diagnostic << "; lhs=";
        if (succeeded(lhs))
          diagnostic << (*lhs).getType();
        else
          diagnostic << "unavailable";
        diagnostic << ", rhs=";
        if (succeeded(rhs))
          diagnostic << (*rhs).getType();
        else
          diagnostic << "unavailable";
        diagnostic << ", result=";
        if (succeeded(result))
          diagnostic << *result;
        else
          diagnostic << "unavailable";
        return failure();
      }
      FailureOr<Value> accumulator =
          zeroAccumulator(cast<gpu::FragmentType>(*result));
      if (failed(accumulator))
        return contract.emitOpError("contract accumulator dtype is unsupported");
      auto target = builder.create<gpu::ContractOp>(
          location, *result, *lhs, *rhs, *accumulator, lhsReduction,
          rhsReduction, lhsBatch, rhsBatch);
      mapResults(operation, target);
      return success();
    }
    if (auto contract = dyn_cast<intent::ScaledContractOp>(operation)) {
      FailureOr<Value> lhs = get(contract.getLhs());
      FailureOr<Value> lhsScale = get(contract.getLhsScale());
      FailureOr<Value> rhs = get(contract.getRhs());
      FailureOr<Value> rhsScale = get(contract.getRhsScale());
      SmallVector<int64_t> lhsReduction, rhsReduction, lhsBatch, rhsBatch;
      axisPairs(contract.getReduce(), lhsReduction, rhsReduction);
      axisPairs(contract.getBatch(), lhsBatch, rhsBatch);
      FailureOr<Type> result = failed(lhs) || failed(rhs)
                                   ? FailureOr<Type>(failure())
                                     : convertContractResultType(
                                         contract.getResult().getType(), *lhs, *rhs,
                                         lhsReduction, rhsReduction, lhsBatch,
                                         rhsBatch);
      if (failed(lhs) || failed(lhsScale) || failed(rhs) || failed(rhsScale) ||
          failed(result) || !isa<gpu::FragmentType>((*lhs).getType()) ||
          !isa<gpu::FragmentType>((*lhsScale).getType()) ||
          !isa<gpu::FragmentType>((*rhs).getType()) ||
          !isa<gpu::FragmentType>((*rhsScale).getType()) ||
          !isa<gpu::FragmentType>(*result))
        return contract.emitOpError(
            "scaled contract operands/results have no physical fragment schema");
      FailureOr<Value> accumulator =
          zeroAccumulator(cast<gpu::FragmentType>(*result));
      if (failed(accumulator))
        return contract.emitOpError(
            "scaled contract accumulator dtype is unsupported");
      auto target = builder.create<gpu::ScaledContractOp>(
          location, *result, *lhs, *lhsScale, *rhs, *rhsScale, *accumulator,
          lhsReduction, rhsReduction, lhsBatch, rhsBatch,
          contract.getLhsFormat(), contract.getRhsFormat(),
          contract.getLhsGroupSize(), contract.getRhsGroupSize());
      mapResults(operation, target);
      return success();
    }
    if (auto contract = dyn_cast<intent::SparseContractOp>(operation)) {
      FailureOr<Value> compressed = get(contract.getCompressed());
      FailureOr<Value> metadata = get(contract.getMetadata());
      FailureOr<Value> rhs = get(contract.getRhs());
      FailureOr<Value> logicalExtent = get(contract.getLogicalExtent());
      SmallVector<int64_t> lhsReduction, rhsReduction, lhsBatch, rhsBatch;
      axisPairs(contract.getReduce(), lhsReduction, rhsReduction);
      axisPairs(contract.getBatch(), lhsBatch, rhsBatch);
      FailureOr<Type> result = failed(compressed) || failed(rhs)
                                   ? FailureOr<Type>(failure())
                                   : convertContractResultType(
                                         contract.getResult().getType(),
                                         *compressed, *rhs, lhsReduction,
                                         rhsReduction, lhsBatch, rhsBatch);
      if (failed(compressed) || failed(metadata) || failed(rhs) ||
          failed(logicalExtent) || failed(result) ||
          !isa<gpu::FragmentType>((*compressed).getType()) ||
          !isa<gpu::FragmentType>((*rhs).getType()) ||
          !isa<gpu::FragmentType>(*result))
        return contract.emitOpError(
            "sparse contract operands/results have no physical schema");
      FailureOr<Value> accumulator =
          zeroAccumulator(cast<gpu::FragmentType>(*result));
      if (failed(accumulator))
        return contract.emitOpError(
            "sparse contract accumulator dtype is unsupported");
      auto format = contract.getFormat();
      auto target = builder.create<gpu::SparseContractOp>(
          location, *result, *compressed, *metadata, *rhs, *logicalExtent,
          *accumulator, format,
          lhsReduction, rhsReduction, lhsBatch, rhsBatch);
      mapResults(operation, target);
      return success();
    }
    if (auto histogram = dyn_cast<intent::HistogramOp>(operation)) {
      FailureOr<Value> values = get(histogram.getValues());
      FailureOr<Value> bins = get(histogram.getBins());
      FailureOr<Value> valid = get(histogram.getValid());
      FailureOr<Type> result =
          convertDataType(histogram.getResult().getType(), operation);
      if (failed(values) || failed(bins) || failed(valid) || failed(result) ||
          !isa<gpu::FragmentType>((*values).getType()) ||
          !isa<gpu::FragmentType>((*valid).getType()) ||
          !isa<gpu::FragmentType>(*result))
        return histogram.emitOpError(
            "histogram operands/results have no physical fragment schema");
      auto target = builder.create<gpu::HistogramOp>(
          location, *result, *values, *bins, *valid);
      mapResults(operation, target);
      return success();
    }
    if (auto random = dyn_cast<intent::RandomBitsOp>(operation)) {
      FailureOr<Value> seed = get(random.getSeed());
      FailureOr<Value> counter = get(random.getLogicalCounter());
      FailureOr<Type> result =
          failed(counter)
              ? FailureOr<Type>(failure())
              : convertDataType(random.getResult().getType(), operation,
                                (*counter).getType());
      if (failed(seed) || failed(counter) || failed(result))
        return random.emitOpError("Philox operands are unavailable");
      auto seedType = dyn_cast<IntegerType>((*seed).getType());
      if (!seedType || seedType.getWidth() != 64)
        return random.emitOpError("Philox seed is not a 64-bit integer");
      if (seedType != builder.getI64Type())
        *seed = builder.create<gpu::BitcastOp>(location, builder.getI64Type(),
                                               *seed);
      auto target = builder.create<gpu::RandomBitsOp>(location, *result, *seed,
                                                       *counter);
      mapResults(operation, target);
      return success();
    }
    if (auto buffer = dyn_cast<intent::BufferOp>(operation)) {
      auto logical = cast<intent::BufferType>(buffer.getResult().getType());
      auto tensor = cast<RankedTensorType>(logical.getTensor());
      SmallVector<Attribute> shape;
      for (Attribute attribute : buffer.getShape().getAxes()) {
        auto axis = cast<intent::ShapeExprAttr>(attribute);
        FailureOr<PhysicalExprAttr> extent = failure();
        if (axis.getKind() == 0)
          extent = expression(operation->getContext(),
                              PhysicalExprKind::Constant, axis.getPayload());
        else if (axis.getKind() == 1)
          extent = launchExpression(buffer.getExtents()[axis.getPayload()],
                                    operation->getParentOfType<func::FuncOp>());
        if (failed(extent))
          return buffer.emitOpError(
              "logical buffer allocation requires a launch-visible extent");
        shape.push_back(*extent);
      }
      LogicalBufferFact allocation = canonicalAnalysis.logicalBuffer(operation);
      if (!allocation.isExact())
        return buffer.emitOpError(
            "logical buffer has no exact canonical allocation fact");
      gpu::BufferScope scope =
          allocation.scope == LogicalBufferScope::ProgramPrivate
              ? gpu::BufferScope::ProgramPrivate
              : gpu::BufferScope::IterationPrivate;
      auto physicalType = gpu::BufferType::get(
          operation->getContext(), tensor.getElementType(), builder.getArrayAttr(shape),
          gpu::BufferScopeAttr::get(operation->getContext(), scope),
          allocation.instanceIdentity,
          /*owner=*/1,
          gpu::BufferInitializationAttr::get(
              operation->getContext(), gpu::BufferInitialization::FirstWrite),
          /*visibility=*/0);
      Value initial;
      if (buffer.getInitial()) {
        FailureOr<Value> lowered =
            get(buffer.getInitial());
        if (failed(lowered))
          return buffer.emitOpError("logical buffer initializer is unavailable");
        initial = *lowered;
      }
      // Materialize initialization as an ordered write in the shared program.
      if (initial && !isa<gpu::FragmentType>(initial.getType()) && !shape.empty()) {
        auto payload = convertTensorType(tensor, operation);
        if (failed(payload))
          return buffer.emitOpError("scalar initializer has no buffer shape");
        initial = builder.create<gpu::SplatOp>(location, *payload, initial);
      }
      if (initial)
        for (unsigned axis = 0; axis < shape.size(); ++axis)
          if (cast<gpu::FragmentType>(initial.getType()).getShape()[axis] !=
                  shape[axis] &&
              failed(gpu::realizeFullCoverageDimension(physicalKernel, initial,
                                                      axis)))
            return buffer.emitOpError(
                "buffer initializer has no complete physical value");
      auto target = builder.create<gpu::BufferOp>(location, physicalType, Value());
      if (initial) {
        auto payload = dyn_cast<gpu::FragmentType>(initial.getType());
        SmallVector<Value> coordinates;
        SmallVector<int64_t> axes;
        Value valid;
        Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
        Value one = builder.create<arith::ConstantIndexOp>(location, 1);
        for (unsigned axis = 0; axis < shape.size(); ++axis) {
          auto mapping = cast<gpu::AxisMapAttr>(payload.getAxisMaps()[axis]);
          auto coordinateType = gpu::FragmentType::get(
              operation->getContext(), builder.getIndexType(),
              builder.getArrayAttr({payload.getShape()[axis]}),
              builder.getArrayAttr({gpu::AxisMapAttr::get(
                  operation->getContext(), mapping.getSourceId(),
                  mapping.getSourceAxis(), mapping.getDimensionId(), 0,
                  mapping.getDerived())}),
              payload.getValidity(), payload.getOwner());
          Value stop = builder.create<gpu::PhysicalExprOp>(
              location, builder.getIndexType(), cast<PhysicalExprAttr>(shape[axis]));
          Value width = builder.create<gpu::PhysicalExprOp>(
              location, builder.getIndexType(),
              cast<PhysicalExprAttr>(payload.getShape()[axis]));
          Value coordinate = builder.create<gpu::MakeRangeOp>(
              location, coordinateType, zero, width, one, zero, stop,
              mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived());
          coordinates.push_back(coordinate);
          axes.push_back(axis);
          auto predicateType = gpu::FragmentType::get(
              operation->getContext(), builder.getI1Type(),
              coordinateType.getShape(), coordinateType.getAxisMaps(),
              coordinateType.getValidity(), coordinateType.getOwner());
          Value end = builder.create<gpu::SplatOp>(location, coordinateType, stop);
          Value predicate = builder.create<gpu::CompareOp>(
              location, predicateType, coordinate, end, ComparePredicate::Lt);
          auto projected = gpu::projectPredicateToFragmentAxis(
              builder, location, predicate, payload, axis);
          if (failed(projected))
            return buffer.emitOpError("buffer initializer has no tail projection");
          valid = valid ? createBinary(builder, location, (*projected).getType(),
                                        valid, *projected, BinaryOperator::LogicalAnd)
                        : *projected;
        }
        auto initialized = builder.create<gpu::StoreOp>(
            location, target, coordinates, initial, valid, axes);
        attachOrigin(operation, initialized);
      }
      mapResults(operation, target);
      return success();
    }
    if (auto gather = dyn_cast<intent::GatherOp>(operation)) {
      FailureOr<Value> source = get(gather.getSource());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Type> result = failed(coordinates)
                                   ? FailureOr<Type>(failure())
                                   : accessResultType(
                                         operation, gather.getResult().getType(),
                                         *coordinates);
      if (failed(source))
        return gather.emitOpError("gather physical source is unavailable");
      if (failed(coordinates))
        return gather.emitOpError(
            "gather index relation has no physical coordinate realization");
      if (failed(axes))
        return gather.emitOpError(
            "gather index relation has no physical source-axis mapping");
      if (failed(result))
        return gather.emitOpError(
            "gather result cannot preserve the physical coordinate relation");
      Value valid;
      Value fill;
      if (gather.getValid()) {
        FailureOr<Value> lowered =
            get(gather.getValid());
        if (failed(lowered))
          return failure();
        valid = *lowered;
      }
      if (gather.getFill()) {
        FailureOr<Value> lowered =
            get(gather.getFill());
        if (failed(lowered))
          return failure();
        fill = *lowered;
      }
      if (auto target = dyn_cast<gpu::FragmentType>(*result)) {
        for (Value *operand : {&valid, &fill}) {
          if (!*operand)
            continue;
          FailureOr<Value> aligned =
              projectAccessOperand(location, *operand, target);
          if (failed(aligned))
            return gather.emitOpError(
                "gather validity/fill cannot adopt its result relation");
          *operand = *aligned;
        }
      }
      FailureOr<Value> bounded = materializeAccessValidity(
          operation, *source, *coordinates, *axes, *result, valid);
      if (failed(bounded))
        return gather.emitOpError(
            "gather resource bounds cannot be materialized in its result relation");
      valid = *bounded;
      if (valid && !fill) {
        FailureOr<Value> zero = zeroAccessFill(location, *result);
        if (failed(zero))
          return gather.emitOpError(
              "gather resource-bounds validity has no typed fill");
        fill = *zero;
      }
      if (coordinates->empty() && axes->empty() && source->getType() == *result) {
        Value target = *source;
        if (valid) {
          target = builder.create<gpu::SelectOp>(location, *result, valid,
                                                  target, fill);
          attachOrigin(operation, target.getDefiningOp());
        }
        values[gather.getResult()] = target;
        return success();
      }
      if (auto scalarTensor = dyn_cast<gpu::FragmentType>(source->getType());
          scalarTensor && scalarTensor.getShape().empty() &&
          coordinates->empty() && axes->empty()) {
        FailureOr<PhysicalAxisIdentity> identity = resultAxisIdentity(operation, 0, 0);
        if (failed(identity))
          return gather.emitOpError("scalar extraction has no physical value identity");
        auto singleton = fragmentType(
            operation->getContext(), scalarTensor.getElementType(),
            {expression(operation->getContext(), PhysicalExprKind::Constant, 1)},
            {*identity}, scalarTensor.getOwner());
        auto reassociation = builder.getArrayAttr({gpu::ReshapeGroupAttr::get(
            operation->getContext(), builder.getDenseI64ArrayAttr({}),
            builder.getDenseI64ArrayAttr({0}))});
        source = Value(builder.create<gpu::ReshapeOp>(
            location, singleton, *source, reassociation));
        coordinates->push_back(builder.create<arith::ConstantIndexOp>(location, 0));
        axes->push_back(0);
      }
      auto target = builder.create<gpu::GatherOp>(
          location, *result, *source, *coordinates, valid, fill, *axes);
      mapResults(operation, target);
      return success();
    }
    if (auto load = dyn_cast<intent::ViewLoadOp>(operation)) {
      FailureOr<Value> resource = get(load.getSource());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Type> result = failed(coordinates)
                                   ? FailureOr<Type>(failure())
                                   : accessResultType(
                                         operation, load.getResult().getType(),
                                         *coordinates);
      if (failed(resource))
        return load.emitOpError("view-load physical resource is unavailable");
      if (failed(coordinates))
        return load.emitOpError(
            "view-load index relation has no physical coordinate realization");
      if (failed(axes))
        return load.emitOpError(
            "view-load index relation has no physical source-axis mapping");
      if (failed(result))
        return load.emitOpError(
            "view-load result cannot preserve the physical coordinate relation");
      Value valid;
      Value fill;
      if (load.getValid()) {
        FailureOr<Value> lowered =
            get(load.getValid());
        if (failed(lowered))
          return failure();
        valid = *lowered;
      }
      if (load.getFill()) {
        FailureOr<Value> lowered =
            get(load.getFill());
        if (failed(lowered))
          return failure();
        fill = *lowered;
      }
      if (auto target = dyn_cast<gpu::FragmentType>(*result)) {
        for (Value *operand : {&valid, &fill}) {
          if (!*operand)
            continue;
          FailureOr<Value> aligned =
              projectAccessOperand(location, *operand, target);
          if (failed(aligned))
            return load.emitOpError(
                "view-load validity/fill cannot adopt its result relation");
          *operand = *aligned;
        }
      }
      FailureOr<Value> bounded = materializeAccessValidity(
          operation, *resource, *coordinates, *axes, *result, valid);
      if (failed(bounded))
        return load.emitOpError(
            "view-load resource bounds cannot be materialized in its result relation");
      valid = *bounded;
      if (valid && !fill) {
        FailureOr<Value> zero = zeroAccessFill(location, *result);
        if (failed(zero))
          return load.emitOpError(
              "view-load resource-bounds validity has no typed fill");
        fill = *zero;
      }
      auto target = builder.create<gpu::LoadOp>(
          location, *result, *resource, *coordinates, valid,
          fill, *axes);
      mapResults(operation, target);
      return success();
    }
    if (auto load = dyn_cast<intent::BufferLoadOp>(operation)) {
      FailureOr<Value> resource = get(load.getSource());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Type> result = failed(coordinates)
                                   ? FailureOr<Type>(failure())
                                   : accessResultType(
                                         operation, load.getResult().getType(),
                                         *coordinates);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(result))
        return load.emitOpError("buffer load physical relation is unavailable");
      FailureOr<Value> valid = materializeAccessValidity(
          operation, *resource, *coordinates, *axes, *result, Value());
      if (failed(valid))
        return load.emitOpError(
            "buffer-load resource bounds cannot be materialized in its result relation");
      Value fill;
      if (*valid) {
        FailureOr<Value> zero = zeroAccessFill(location, *result);
        if (failed(zero))
          return load.emitOpError(
              "buffer-load resource-bounds validity has no typed fill");
        fill = *zero;
      }
      auto target = builder.create<gpu::LoadOp>(
          location, *result, *resource, *coordinates, *valid, fill, *axes);
      mapResults(operation, target);
      return success();
    }
    if (auto store = dyn_cast<intent::ViewStoreOp>(operation)) {
      FailureOr<Value> resource = get(store.getSource());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            store.getValue(),
                            *coordinates);
      if (failed(resource))
        return store.emitOpError("view-store physical resource is unavailable");
      if (failed(coordinates))
        return store.emitOpError(
            "view-store index relation has no physical coordinate realization");
      if (failed(axes))
        return store.emitOpError(
            "view-store index relation has no physical source-axis mapping");
      if (failed(value))
        return store.emitOpError(
            "view-store value cannot preserve its physical result relation");
      FailureOr<Value> valid = materializeAccessValidity(
          operation, *resource, *coordinates, *axes, (*value).getType(), Value());
      if (failed(valid))
        return store.emitOpError(
            "view-store resource bounds cannot be materialized in its value relation");
      auto target = builder.create<gpu::StoreOp>(
          location, *resource, *coordinates, *value, *valid, *axes);
      if (Attribute node = operation->getAttr("intent.node"))
        target->setAttr(gpu::originAttr, node);
      return success();
    }
    if (auto store = dyn_cast<intent::BufferStoreOp>(operation)) {
      FailureOr<Value> resource = get(store.getSource());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            store.getValue(),
                            *coordinates);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value))
        return store.emitOpError("buffer store physical relation is unavailable");
      FailureOr<Value> valid = materializeAccessValidity(
          operation, *resource, *coordinates, *axes, (*value).getType(), Value());
      if (failed(valid))
        return store.emitOpError(
            "buffer-store resource bounds cannot be materialized in its value relation");
      auto target = builder.create<gpu::StoreOp>(
          location, *resource, *coordinates, *value, *valid, *axes);
      attachOrigin(operation, target);
      return success();
    }
    if (auto store = dyn_cast<intent::ScatterUniqueOp>(operation)) {
      FailureOr<Value> resource = get(store.getSource());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            store.getValue(),
                            *coordinates);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value))
        return store.emitOpError("unique scatter is not a scalar physical access");
      FailureOr<Value> valid = materializeAccessValidity(
          operation, *resource, *coordinates, *axes, (*value).getType(), Value());
      if (failed(valid))
        return store.emitOpError(
            "unique-scatter resource bounds cannot be materialized in its value relation");
      auto target = builder.create<gpu::StoreOp>(
          location, *resource, *coordinates, *value, *valid, *axes);
      if (Attribute node = operation->getAttr("intent.node"))
        target->setAttr(gpu::originAttr, node);
      return success();
    }
    if (auto scatter = dyn_cast<intent::ScatterReduceOp>(operation)) {
      FailureOr<Value> resource = get(scatter.getSource());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            scatter.getValue(),
                            *coordinates);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value))
        return scatter.emitOpError(
            "scatter-reduce physical relation is unavailable");
      FailureOr<Value> valid = materializeAccessValidity(
          operation, *resource, *coordinates, *axes, (*value).getType(), Value());
      if (failed(valid))
        return scatter.emitOpError(
            "scatter-reduce resource bounds cannot be materialized in its value relation");
      OperationState state(location, gpu::ScatterReduceOp::getOperationName());
      state.addOperands(*resource);
      state.addOperands(*coordinates);
      state.addOperands(*value);
      if (*valid)
        state.addOperands(*valid);
      state.addAttribute("source_axes", builder.getDenseI64ArrayAttr(*axes));
      state.addAttribute(
          "sharing", gpu::AtomicSharingDomainAttr::get(
                         operation->getContext(),
                         isa<gpu::ViewType>((*resource).getType())
                             ? gpu::AtomicSharingDomain::KernelInvocation
                             : gpu::AtomicSharingDomain::ProgramInstance));
      state.addAttribute("operandSegmentSizes",
                         builder.getDenseI32ArrayAttr(
                             {1, static_cast<int32_t>(coordinates->size()), 1,
                              static_cast<int32_t>(static_cast<bool>(*valid))}));
      state.addRegion();
      Operation *raw = builder.create(state);
      auto target = cast<gpu::ScatterReduceOp>(raw);
      SmallVector<Type> arguments{(*value).getType(), (*value).getType()};
      if (failed(lowerPureRegion(scatter.getCombine(), target.getCombine(),
                                 arguments)))
        return failure();
      attachOrigin(operation, raw);
      return success();
    }
    if (auto atomic = dyn_cast<intent::AtomicLoadOp>(operation)) {
      FailureOr<Value> resource = get(atomic.getSource());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Type> result =
          failed(coordinates)
              ? FailureOr<Type>(failure())
              : accessResultType(operation, atomic.getResult().getType(),
                                 *coordinates);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(result))
        return atomic.emitOpError("atomic load address/result is unavailable");
      gpu::AtomicSharingDomain sharing =
          isa<gpu::ViewType>((*resource).getType())
              ? gpu::AtomicSharingDomain::KernelInvocation
              : gpu::AtomicSharingDomain::ProgramInstance;
      FailureOr<Value> valid = materializeAccessValidity(
          operation, *resource, *coordinates, *axes, *result, Value());
      if (failed(valid))
        return atomic.emitOpError(
            "atomic-load resource bounds cannot be materialized in its result relation");
      auto target = builder.create<gpu::AtomicLoadOp>(
          location, *result, *resource, *coordinates, *valid,
          atomic.getOrdering(), sharing, *axes);
      mapResults(operation, target);
      return success();
    }
    if (auto atomic = dyn_cast<intent::AtomicStoreOp>(operation)) {
      FailureOr<Value> resource = get(atomic.getSource());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            atomic.getValue(),
                            *coordinates);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value))
        return atomic.emitOpError("atomic store address/value is unavailable");
      gpu::AtomicSharingDomain sharing =
          isa<gpu::ViewType>((*resource).getType())
              ? gpu::AtomicSharingDomain::KernelInvocation
              : gpu::AtomicSharingDomain::ProgramInstance;
      FailureOr<Value> valid = materializeAccessValidity(
          operation, *resource, *coordinates, *axes, (*value).getType(), Value());
      if (failed(valid))
        return atomic.emitOpError(
            "atomic-store resource bounds cannot be materialized in its value relation");
      auto target = builder.create<gpu::AtomicStoreOp>(
          location, *resource, *coordinates, *value, *valid,
          atomic.getOrdering(), sharing, *axes);
      if (Attribute node = operation->getAttr("intent.node"))
        target->setAttr(gpu::originAttr, node);
      return success();
    }
    if (auto atomic = dyn_cast<intent::AtomicRMWOp>(operation)) {
      FailureOr<Value> resource = get(atomic.getSource());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            atomic.getValue(),
                            *coordinates);
      FailureOr<Type> result = failed(value)
                                   ? FailureOr<Type>(failure())
                                   : FailureOr<Type>((*value).getType());
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value) || failed(result))
        return atomic.emitOpError("atomic RMW address/value is unavailable");
      gpu::AtomicSharingDomain sharing =
          isa<gpu::ViewType>((*resource).getType())
              ? gpu::AtomicSharingDomain::KernelInvocation
              : gpu::AtomicSharingDomain::ProgramInstance;
      FailureOr<Value> valid = materializeAccessValidity(
          operation, *resource, *coordinates, *axes, (*value).getType(), Value());
      if (failed(valid))
        return atomic.emitOpError(
            "atomic-RMW resource bounds cannot be materialized in its value relation");
      auto target = builder.create<gpu::AtomicRMWOp>(
          location, *result, *resource, *coordinates, *value, *valid,
          atomic.getKind(), atomic.getOrdering(), sharing, *axes);
      mapResults(operation, target);
      return success();
    }
    if (auto atomic = dyn_cast<intent::AtomicCompareExchangeOp>(operation)) {
      FailureOr<Value> resource = get(atomic.getSource());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> expected =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            atomic.getExpected(),
                            *coordinates);
      FailureOr<Value> desired =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            atomic.getDesired(),
                            *coordinates);
      FailureOr<Type> result =
          convertDataType(atomic.getResult().getType(), operation);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(expected) || failed(desired) || failed(result))
        return atomic.emitOpError("compare-exchange address/value is unavailable");
      gpu::AtomicSharingDomain sharing =
          isa<gpu::ViewType>((*resource).getType())
              ? gpu::AtomicSharingDomain::KernelInvocation
              : gpu::AtomicSharingDomain::ProgramInstance;
      FailureOr<Value> valid = materializeAccessValidity(
          operation, *resource, *coordinates, *axes, (*expected).getType(),
          Value());
      if (failed(valid))
        return atomic.emitOpError(
            "compare-exchange resource bounds cannot be materialized in its value relation");
      auto target = builder.create<gpu::AtomicCompareExchangeOp>(
          location, *result, *resource, *coordinates, *expected, *desired,
          *valid, atomic.getOrdering(), sharing, *axes);
      mapResults(operation, target);
      return success();
    }
    if (auto record = dyn_cast<intent::MakeRecordOp>(operation)) {
      SmallVector<Value> fields;
      for (Value field : record.getFields()) {
        FailureOr<Value> lowered = get(field);
        if (failed(lowered))
          return failure();
        fields.push_back(*lowered);
      }
      FailureOr<Type> result =
          convertDataType(record.getResult().getType(), operation);
      if (failed(result))
        return record.emitOpError("record contains an unphysical field");
      auto schema = cast<gpu::RecordType>(*result);
      SmallVector<Attribute> fieldTypes;
      for (Value field : fields)
        fieldTypes.push_back(TypeAttr::get(field.getType()));
      auto resultType = gpu::RecordType::get(
          operation->getContext(), schema.getFieldNames(),
          builder.getArrayAttr(fieldTypes), schema.getOwner());
      auto target =
          builder.create<gpu::MakeRecordOp>(location, resultType, fields);
      mapResults(operation, target);
      return success();
    }
    if (auto tuple = dyn_cast<intent::MakeTupleOp>(operation)) {
      SmallVector<Value> fields;
      for (Value field : tuple.getComponents()) {
        FailureOr<Value> lowered = get(field);
        if (failed(lowered))
          return failure();
        fields.push_back(*lowered);
      }
      FailureOr<Type> result =
          convertDataType(tuple.getResult().getType(), operation);
      if (failed(result))
        return tuple.emitOpError("tuple contains an unphysical field");
      auto schema = cast<gpu::RecordType>(*result);
      SmallVector<Attribute> fieldTypes;
      for (Value field : fields)
        fieldTypes.push_back(TypeAttr::get(field.getType()));
      auto resultType = gpu::RecordType::get(
          operation->getContext(), schema.getFieldNames(),
          builder.getArrayAttr(fieldTypes), schema.getOwner());
      auto target =
          builder.create<gpu::MakeRecordOp>(location, resultType, fields);
      mapResults(operation, target);
      return success();
    }
    if (auto extract = dyn_cast<intent::ExtractOp>(operation)) {
      FailureOr<Value> product = get(extract.getProduct());
      if (failed(product))
        return extract.emitOpError("product projection is unavailable");
      auto recordType = dyn_cast<gpu::RecordType>((*product).getType());
      int64_t field = extract.getField();
      if (!recordType || field < 0 ||
          field >= static_cast<int64_t>(recordType.getFieldTypes().size()))
        return extract.emitOpError("product projection field is unavailable");
      Type result = cast<TypeAttr>(recordType.getFieldTypes()[field]).getValue();
      auto target = builder.create<gpu::ExtractOp>(location, result, *product,
                                                    extract.getField());
      mapResults(operation, target);
      return success();
    }
    if (auto ifOperation = dyn_cast<intent::IfOp>(operation)) {
      FailureOr<Value> condition = get(ifOperation.getCondition());
      if (failed(condition))
        return failure();
      SmallVector<Type> resultTypes;
      for (auto [index, type] :
           llvm::enumerate(ifOperation.getResultTypes())) {
        FailureOr<Type> converted =
            convertDataType(type, operation, std::nullopt, /*owner=*/1, index);
        if (failed(converted))
          return ifOperation.emitOpError("if carries an unphysical value");
        resultTypes.push_back(*converted);
      }
      auto target = builder.create<scf::IfOp>(location, resultTypes, *condition,
                                               /*withElseRegion=*/true);
      auto lowerBranch = [&](Region &targetRegion, Region &sourceRegion) {
        Block &targetBlock = targetRegion.front();
        if (!targetBlock.empty() && isa<scf::YieldOp>(targetBlock.back()))
          targetBlock.back().erase();
        OpBuilder nested(&targetBlock, targetBlock.begin());
        ScalarRegionLowering child(nested, values, views, dimensions, parameters,
                                   canonicalAnalysis, physicalKernel);
        FailureOr<SmallVector<Value>> yielded =
            child.lowerBlock(sourceRegion.front());
        if (failed(yielded) || yielded->size() != resultTypes.size())
          return failure();
        for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
          FailureOr<Value> projected = projectPositionalValue(
              nested, location, (*yielded)[index], resultType);
          if (failed(projected))
            return failure();
          (*yielded)[index] = *projected;
        }
        nested.create<scf::YieldOp>(location, *yielded);
        return success();
      };
      if (failed(lowerBranch(target.getThenRegion(),
                             ifOperation.getThenRegion())) ||
          failed(lowerBranch(target.getElseRegion(),
                             ifOperation.getElseRegion())))
        return failure();
      mapResults(operation, target);
      return success();
    }
    if (isa<intent::ForOp, intent::ParallelOp>(operation)) {
      // Parallel regions remaining inside an execution group may be serialized.
      // Keep their enclosing ordered control and resource environment intact.
      auto forOperation = dyn_cast<intent::ForOp>(operation);
      auto parallel = dyn_cast<intent::ParallelOp>(operation);
      Value sourceDomain = forOperation ? forOperation.getSource() : parallel.getSource();
      Region &sourceRegion = forOperation ? forOperation.getBody() : parallel.getBody();
      ValueRange inductionVariables = forOperation ? ValueRange(forOperation.getInductionVars())
                                                  : ValueRange(sourceRegion.front().getArguments());
      ValueRange iterArguments = forOperation ? ValueRange(forOperation.getRegionIterArgs()) : ValueRange{};
      SmallVector<IterationAxis> axes;
      if (failed(collectIterationAxes(sourceDomain, axes)) ||
          axes.empty())
        return operation->emitOpError(
            "iteration source has no exact domain/subregion relation");
      SmallVector<Value> lowers, uppers, steps;
      for (IterationAxis axis : axes) {
        FailureOr<Value> lower = get(axis.start);
        FailureOr<Value> upper = get(axis.stop);
        FailureOr<Value> prototype = get(axis.coordinatePrototype);
        if (failed(lower) || failed(upper) || failed(prototype))
          return operation->emitOpError(
              "physical loop bounds are unavailable");
        Type coordinateType = (*prototype).getType();
        auto alignBound = [&](Value value) -> FailureOr<Value> {
          if (value.getType() == coordinateType)
            return value;
          if (!isa<IntegerType, IndexType>(value.getType()) ||
              !isa<IntegerType, IndexType>(coordinateType))
            return failure();
          return Value(builder.create<gpu::CastOp>(
              location, coordinateType, value));
        };
        lower = alignBound(*lower);
        upper = alignBound(*upper);
        if (failed(lower) || failed(upper))
          return operation->emitOpError(
              "physical subregion bounds cannot adopt their source coordinate type");
        Value step;
        if (axis.step) {
          FailureOr<Value> lowered = get(axis.step);
          if (failed(lowered))
            return operation->emitOpError(
                "physical loop step is unavailable");
          FailureOr<Value> aligned = alignBound(*lowered);
          if (failed(aligned))
            return operation->emitOpError(
                "physical loop step cannot adopt its source coordinate type");
          step = *aligned;
        } else if ((*lower).getType().isIndex()) {
          step = builder.create<arith::ConstantIndexOp>(location, 1);
        } else if (auto integer = dyn_cast<IntegerType>((*lower).getType())) {
          step = builder.create<arith::ConstantOp>(
              location, integer, builder.getIntegerAttr(integer, 1));
        }
        if (!step || (*lower).getType() != (*upper).getType() ||
            (*lower).getType() != step.getType())
          return operation->emitOpError(
              "physical loop bounds must share one scalar type");
        lowers.push_back(*lower);
        uppers.push_back(*upper);
        steps.push_back(step);
      }
      SmallVector<Value> initial;
      ValueRange initArgs = forOperation ? ValueRange(forOperation.getInitArgs()) : ValueRange{};
      for (Value input : initArgs) {
        FailureOr<Value> lowered = get(input);
        if (failed(lowered))
          return failure();
        initial.push_back(*lowered);
      }
      bool nestedFailed = false;
      std::function<SmallVector<Value>(OpBuilder &, unsigned,
                                       SmallVector<Value>, ValueRange)>
          lowerAxis;
      lowerAxis = [&](OpBuilder &nested, unsigned axis,
                      SmallVector<Value> coordinates,
                      ValueRange carries) -> SmallVector<Value> {
        if (axis == axes.size()) {
          auto childValues = values;
          Block &source = sourceRegion.front();
          for (auto [argument, coordinate] :
               llvm::zip(inductionVariables, coordinates))
            childValues[argument] = coordinate;
          for (auto [argument, carry] : llvm::zip(iterArguments, carries))
            childValues[argument] = carry;
          ScalarRegionLowering child(nested, std::move(childValues), views,
                                     dimensions, parameters, canonicalAnalysis,
                                     physicalKernel);
          FailureOr<SmallVector<Value>> yielded = child.lowerBlock(source);
          if (failed(yielded)) {
            nestedFailed = true;
            return {};
          }
          return *yielded;
        }
        SmallVector<Value> loopResults;
        auto loop = nested.create<scf::ForOp>(
            location, lowers[axis], uppers[axis], steps[axis], carries,
            [](OpBuilder &bodyBuilder, Location location, Value,
                ValueRange innerCarries) {
              bodyBuilder.create<scf::YieldOp>(location, innerCarries);
            });
        OpBuilder bodyBuilder(loop.getBody()->getTerminator());
        SmallVector<Value> nextCoordinates(coordinates);
        nextCoordinates.push_back(loop.getInductionVar());
        SmallVector<Value> yielded = lowerAxis(
            bodyBuilder, axis + 1, std::move(nextCoordinates),
            loop.getRegionIterArgs());
        if (nestedFailed) {
          loop.erase();
          return {};
        }
        loop.getBody()->getTerminator()->setOperands(yielded);
        if (isa<intent::ParallelOp>(operation))
          loop->setAttr(gpu::independentIterationAttr, nested.getUnitAttr());
        auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
        if (!yield) {
          nestedFailed = true;
          loop.erase();
          return {};
        }
        for (auto [index, item] : llvm::enumerate(llvm::zip(
                 loop.getInitArgs(), yield.getOperands()))) {
          Value initial = std::get<0>(item);
          Value yielded = std::get<1>(item);
          if (initial.getType() == yielded.getType())
            continue;
          // The loop-carried value keeps the physical relation established by
          // its initializer, recursively for record-valued algorithm state. A
          // body expression may temporarily acquire result-local axis
          // identities, but those identities must not replace the source
          // provenance carried across iterations.
          OpBuilder before(yield);
          FailureOr<Value> aligned = projectPositionalValue(
              before, location, yielded, initial.getType());
          if (failed(aligned)) {
            operation->emitOpError(
                "loop update cannot preserve its physical carry relation")
                << "; initial=" << initial.getType()
                << "; yielded=" << yielded.getType();
            nestedFailed = true;
            break;
          }
          yield->setOperand(index, *aligned);
        }
        if (nestedFailed) {
          loop.erase();
          return {};
        }
        loopResults.append(loop.getResults().begin(), loop.getResults().end());
        return loopResults;
      };
      SmallVector<Value> resultValues =
          lowerAxis(builder, 0, SmallVector<Value>{}, initial);
      if (nestedFailed)
        return failure();
      for (auto [source, target] :
           llvm::zip(operation->getResults(), resultValues))
        values[source] = target;
      return success();
    }
    if (auto whileOperation = dyn_cast<intent::WhileOp>(operation)) {
      SmallVector<Value> initial;
      SmallVector<Type> resultTypes;
      for (Value input : whileOperation.getInitArgs()) {
        FailureOr<Value> lowered = get(input);
        if (failed(lowered))
          return failure();
        initial.push_back(*lowered);
        resultTypes.push_back((*lowered).getType());
      }
      auto target = builder.create<scf::WhileOp>(location, resultTypes, initial);
      {
        OpBuilder::InsertionGuard guard(builder);
        Block *before = builder.createBlock(
            &target.getBefore(), target.getBefore().end(), resultTypes,
            SmallVector<Location>(resultTypes.size(), location));
        auto childValues = values;
        for (auto [source, targetArgument] : llvm::zip(
                 whileOperation.getBeforeArguments(),
                 before->getArguments()))
          childValues[source] = targetArgument;
        builder.setInsertionPointToStart(before);
        ScalarRegionLowering child(builder, std::move(childValues), views,
                                   dimensions, parameters, canonicalAnalysis,
                                   physicalKernel);
        Block &source = whileOperation.getBefore().front();
        for (Operation &nested : source.without_terminator())
          if (failed(child.lower(&nested)))
            return failure();
        auto condition = cast<intent::ConditionOp>(source.getTerminator());
        FailureOr<Value> predicate = child.get(condition.getCondition());
        SmallVector<Value> forwarded;
        for (Value value : condition.getArgs()) {
          FailureOr<Value> lowered = child.get(value);
          if (failed(lowered))
            return failure();
          forwarded.push_back(*lowered);
        }
        if (failed(predicate))
          return failure();
        builder.create<scf::ConditionOp>(location, *predicate, forwarded);
      }
      {
        OpBuilder::InsertionGuard guard(builder);
        Block *after = builder.createBlock(
            &target.getAfter(), target.getAfter().end(), resultTypes,
            SmallVector<Location>(resultTypes.size(), location));
        auto childValues = values;
        for (auto [source, targetArgument] : llvm::zip(
                 whileOperation.getAfterArguments(),
                 after->getArguments()))
          childValues[source] = targetArgument;
        builder.setInsertionPointToStart(after);
        ScalarRegionLowering child(builder, std::move(childValues), views,
                                   dimensions, parameters, canonicalAnalysis,
                                   physicalKernel);
        FailureOr<SmallVector<Value>> yielded =
            child.lowerBlock(whileOperation.getAfter().front());
        if (failed(yielded))
          return failure();
        builder.create<scf::YieldOp>(location, *yielded);
      }
      mapResults(operation, target);
      return success();
    }
    if (auto assumption = dyn_cast<intent::AssumeInBoundsOp>(operation)) {
      FailureOr<Value> index = get(assumption.getIndex());
      FailureOr<Value> resource = get(assumption.getView());
      if (failed(index) || failed(resource))
        return assumption.emitOpError(
            "in-bounds assertion lost its physical value or resource");
      index = asLogicalIndex(location, *index);
      if (failed(index))
        return assumption.emitOpError(
            "in-bounds assertion index has no physical logical-index schema");
      FailureOr<unsigned> axis = physicalResourceAxis(
          assumption.getView().getType(), (*resource).getType(),
          assumption.getAxis());
      if (failed(axis))
        return assumption.emitOpError(
            "in-bounds assertion axis has no physical resource mapping");
      auto target = builder.create<gpu::AssumeInBoundsOp>(
          location, *index, *resource, *axis);
      attachOrigin(operation, target);
      return success();
    }
    return operation->emitOpError(
        "has no scalar shared-GPU construction; tensor/structured families require physical co-realization");
  }

  OpBuilder &builder;
  llvm::DenseMap<Value, Value> values;
  ArrayRef<Value> views;
  llvm::DenseMap<int64_t, Value> dimensions;
  llvm::DenseMap<StringAttr, Value> parameters;
  CanonicalKernelAnalysis &canonicalAnalysis;
  func::FuncOp physicalKernel;
};

struct ParallelWorkset {
  intent::ParallelOp operation;
  Block *body = nullptr;
  bool singleton = false;
  SmallVector<IterationAxis> axes;
  SmallVector<BlockArgument> coordinateArguments;
  SmallVector<PhysicalExprAttr> launchExtents;
  PhysicalExprAttr launchLength;
};

LogicalResult collectIterationAxes(
    Value source, SmallVectorImpl<IterationAxis> &axes) {
  if (auto domain = source.getDefiningOp<intent::DomainOp>()) {
    axes.push_back({domain.getBounds()[0], domain.getBounds()[1],
                    domain.getBounds().size() == 3 ? domain.getBounds()[2]
                                                   : Value(),
                    domain.getBounds()[0], source});
    return success();
  }
  if (auto product = source.getDefiningOp<intent::DomainProductOp>()) {
    for (Value component : product.getDomains())
      if (failed(collectIterationAxes(component, axes)))
        return failure();
    return success();
  }
  auto subregion = source.getDefiningOp<intent::SubregionOp>();
  if (!subregion || subregion.getInputs().empty())
    return failure();
  SmallVector<IterationAxis> parent;
  if (failed(collectIterationAxes(subregion.getInputs().front(), parent)) ||
      parent.size() != 1)
    return failure();
  unsigned operand = 1;
  if (subregion.getHasStart())
    parent.front().start = subregion.getInputs()[operand++];
  if (subregion.getHasStop())
    parent.front().stop = subregion.getInputs()[operand++];
  if (operand != subregion.getInputs().size())
    return failure();
  parent.front().source = source;
  axes.append(parent.begin(), parent.end());
  return success();
}

LogicalResult finalizeParallelWorkset(ParallelWorkset &workset,
                                      func::FuncOp function) {
  MLIRContext *context = workset.operation.getContext();
  for (const IterationAxis &axis : workset.axes) {
    FailureOr<PhysicalExprAttr> start =
        launchExpression(axis.start, function);
    FailureOr<PhysicalExprAttr> stop =
        launchExpression(axis.stop, function);
    FailureOr<PhysicalExprAttr> step =
        axis.step
            ? launchExpression(axis.step, function)
            : FailureOr<PhysicalExprAttr>(
                  expression(context, PhysicalExprKind::Constant, 1));
    if (failed(start) || failed(stop) || failed(step))
      return axis.source.getDefiningOp()->emitOpError(
          "parallel workset bound is not a launch-visible typed expression");
    PhysicalExprAttr distance = binaryExpression(
        context, PhysicalExprKind::Subtract, *stop, *start);
    workset.launchExtents.push_back(binaryExpression(
        context, PhysicalExprKind::CeilDiv, distance, *step));
  }
  workset.launchLength = workset.launchExtents.front();
  for (PhysicalExprAttr extent : llvm::drop_begin(workset.launchExtents))
    workset.launchLength = binaryExpression(
        context, PhysicalExprKind::Multiply, workset.launchLength, extent);
  if (workset.coordinateArguments.size() != workset.axes.size())
    return workset.operation.emitOpError(
        "parallel workset coordinates do not match its domain product");
  return success();
}

bool capturesScanPrefixAxis(const LogicalWorksetFact &workset,
                            CanonicalKernelAnalysis &analysis,
                            intent::ParallelOp &scope) {
  if (workset.singleton || !workset.parallel || !workset.body)
    return false;
  SmallVector<CoordinateOrigin> worksetCoordinates;
  for (BlockArgument coordinate : workset.coordinates) {
    CoordinateProvenance provenance = analysis.coordinateProvenance(coordinate);
    if (provenance.known)
      worksetCoordinates.append(provenance.origins.begin(),
                                provenance.origins.end());
  }
  bool captured = false;
  workset.body->walk([&](intent::GatherOp gather) {
    SmallVector<Value> pending{gather.getSource()};
    llvm::DenseSet<Value> visited;
    SmallVector<intent::ScanOp> scans;
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      if (!visited.insert(value).second)
        continue;
      if (auto scan = value.getDefiningOp<intent::ScanOp>()) {
        if (!workset.parallel->isProperAncestor(scan))
          scans.push_back(scan);
        continue;
      }
      Operation *producer = value.getDefiningOp();
      auto tensor = dyn_cast<RankedTensorType>(value.getType());
      if (!producer || !tensor ||
          !isa<intent::UnaryOp, intent::BinaryOp, intent::CompareOp,
               intent::SelectOp, intent::CastOp, intent::BitcastOp>(producer))
        continue;
      for (Value input : producer->getOperands()) {
        auto operand = dyn_cast<RankedTensorType>(input.getType());
        if (operand && operand.getShape() == tensor.getShape() &&
            dimensionIds(operand) == dimensionIds(tensor))
          pending.push_back(input);
      }
    }
    FailureOr<IndexRelationFact> relation = analysis.indexRelation(gather);
    if (failed(relation))
      return WalkResult::advance();
    for (intent::ScanOp scan : scans) {
      for (const IndexTermFact &term : relation->terms) {
        if (!term.sourceAxis || *term.sourceAxis != scan.getAxis() ||
            !term.coordinate.known)
          continue;
        if (llvm::any_of(term.coordinate.origins, [&](const auto &origin) {
              return llvm::is_contained(worksetCoordinates, origin);
            })) {
          auto enclosing = gather->getParentOfType<intent::ParallelOp>();
          while (enclosing && !enclosing->isProperAncestor(scan))
            enclosing = enclosing->getParentOfType<intent::ParallelOp>();
          if (!captured)
            scope = enclosing;
          else if (!scope || !enclosing)
            scope = {};
          else if (enclosing->isProperAncestor(scope))
            scope = enclosing;
          captured = true;
          break;
        }
      }
    }
    return WalkResult::advance();
  });
  return captured;
}

LogicalResult constructGPUProgram(ModuleOp module,
                                  const GPUCapabilities &capabilities,
                                  func::FuncOp function) {
  Attribute sourceOrigin = function->getAttr("intent.source");
  if (!sourceOrigin)
    return function.emitError(
        "canonical kernel has no stable source origin for physical construction");
  CanonicalKernelAnalysis canonicalAnalysis(module);
  FailureOr<SmallVector<LogicalWorksetFact, 4>> logicalWorksets =
      canonicalAnalysis.logicalWorksets(function);
  if (failed(logicalWorksets))
    return failure();
  SmallVector<intent::ParallelOp> prefixScopes;
  bool wholeBodyPrefix = false;
  for (const LogicalWorksetFact &workset : *logicalWorksets) {
    intent::ParallelOp scope;
    if (!capturesScanPrefixAxis(workset, canonicalAnalysis, scope))
      continue;
    if (!scope) {
      wholeBodyPrefix = true;
      break;
    }
    SmallVector<IterationAxis> axes;
    for (auto [domain, coordinate] :
         llvm::zip(workset.domains, workset.coordinates)) {
      if (failed(collectIterationAxes(domain, axes)))
        return failure();
      if (coordinate == scope.getBody().front().getArguments().back())
        break;
    }
    if (llvm::any_of(axes, [&](const IterationAxis &axis) {
          return failed(launchExpression(axis.start, function)) ||
                 failed(launchExpression(axis.stop, function)) ||
                 (axis.step && failed(launchExpression(axis.step, function)));
        })) {
      wholeBodyPrefix = true;
      break;
    }
    if (llvm::any_of(prefixScopes, [&](intent::ParallelOp previous) {
          return previous->isAncestor(scope);
        }))
      continue;
    llvm::erase_if(prefixScopes, [&](intent::ParallelOp previous) {
      return scope->isProperAncestor(previous);
    });
    prefixScopes.push_back(scope);
  }
  if (wholeBodyPrefix) {
    LogicalWorksetFact singleton;
    singleton.state = CanonicalFactState::Exact;
    singleton.body = &function.getBody().front();
    singleton.singleton = true;
    logicalWorksets->clear();
    logicalWorksets->push_back(std::move(singleton));
  } else if (!prefixScopes.empty()) {
    // The prefix and its consumers share one instance; enclosing independent
    // coordinates remain launch axes. A scope also owns all sibling worksets.
    SmallVector<LogicalWorksetFact, 4> grouped;
    llvm::SmallPtrSet<Operation *, 4> emitted;
    for (LogicalWorksetFact workset : *logicalWorksets) {
      intent::ParallelOp scope;
      for (intent::ParallelOp candidate : prefixScopes)
        if (workset.parallel && candidate->isAncestor(workset.parallel)) {
          scope = candidate;
          break;
        }
      if (!scope) {
        grouped.push_back(std::move(workset));
        continue;
      }
      if (!emitted.insert(scope.getOperation()).second)
        continue;
      Block &body = scope.getBody().front();
      auto first = llvm::find(workset.coordinates, body.getArgument(0));
      size_t offset = std::distance(workset.coordinates.begin(), first);
      size_t count = offset + body.getNumArguments();
      if (count > workset.coordinates.size() || count > workset.domains.size() ||
          !llvm::equal(ArrayRef<BlockArgument>(workset.coordinates).slice(
                           offset, body.getNumArguments()), body.getArguments()))
        return scope.emitOpError(
            "captured prefix scope has no matching workset coordinate prefix");
      workset.parallel = scope;
      workset.body = &body;
      workset.domains.resize(count);
      workset.coordinates.resize(count);
      grouped.push_back(std::move(workset));
    }
    *logicalWorksets = std::move(grouped);
  }
  SmallVector<ParallelWorkset> worksets;
  for (const LogicalWorksetFact &fact : *logicalWorksets) {
    if (!fact.isExact() || !fact.body)
      return function.emitError(
          "canonical workset analysis produced an incomplete execution group");
    ParallelWorkset workset;
    workset.operation = dyn_cast_or_null<intent::ParallelOp>(fact.parallel);
    workset.body = fact.body;
    workset.singleton = fact.singleton;
    workset.coordinateArguments.append(fact.coordinates.begin(),
                                       fact.coordinates.end());
    for (Value domainValue : fact.domains) {
      if (failed(collectIterationAxes(domainValue, workset.axes)))
        return function.emitError(
            "canonical workset axis has no typed domain/subregion bounds");
    }
    if (workset.singleton) {
      workset.launchExtents.push_back(
          expression(function.getContext(), PhysicalExprKind::Constant, 1));
      workset.launchLength = workset.launchExtents.front();
    } else if (failed(finalizeParallelWorkset(workset, function))) {
        return failure();
    }
    worksets.push_back(std::move(workset));
  }

  MLIRContext *context = module.getContext();
  OpBuilder builder(context);
  FailureOr<PhysicalABI> abi = buildPhysicalABI(function, builder);
  if (failed(abi))
    return failure();
  PhysicalExprAttr totalLength = worksets.front().launchLength;
  for (const ParallelWorkset &workset : llvm::drop_begin(worksets))
    totalLength = binaryExpression(context, PhysicalExprKind::Add, totalLength,
                                   workset.launchLength);
  auto capabilityAttr = gpu::CapabilitiesAttr::get(
      context, capabilities.computeUnits, capabilities.sharedMemoryPerUnit,
      capabilities.maxDynamicSharedMemoryPerBlock,
      capabilities.registersPerUnit,
      capabilities.maxThreadsPerBlock, capabilities.computeCapabilityMajor,
      capabilities.computeCapabilityMinor,
      capabilities.singleToDoublePrecisionPerfRatio, capabilities.matrixUnits,
      capabilities.dynamicVectorWidth, capabilities.nativeTupleReductions,
      capabilities.nativeFragmentGather);
  SmallVector<NamedAttribute> functionAttrs{
      builder.getNamedAttr(gpu::kernelAttr, builder.getUnitAttr()),
      builder.getNamedAttr(interfaceAttr, abi->interface),
      builder.getNamedAttr(gpu::parametersAttr, builder.getArrayAttr({})),
      builder.getNamedAttr(gpu::capabilitiesAttr, capabilityAttr),
      builder.getNamedAttr(gpu::programSpaceAttr,
                           builder.getArrayAttr({totalLength})),
      builder.getNamedAttr(gpu::gridRankAttr, builder.getI64IntegerAttr(1)),
      builder.getNamedAttr(gpu::effectOriginsAttr,
                           observableEffectOrigins(function)),
      builder.getNamedAttr(gpu::originAttr, sourceOrigin),
  };
  builder.setInsertionPointAfter(function);
  auto physical = builder.create<func::FuncOp>(
      function.getLoc(), ("__intent_gpu_" + function.getName()).str(),
      FunctionType::get(context, abi->arguments, {}), functionAttrs,
      abi->argumentAttrs);
  Block *entry = physical.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  llvm::DenseMap<StringAttr, Value> parameterValues;
  SmallVector<Value> sourceArguments;
  sourceArguments.reserve(abi->physicalArgumentForSource.size());
  for (std::optional<unsigned> physicalIndex : abi->physicalArgumentForSource)
    sourceArguments.push_back(physicalIndex ? Value(entry->getArgument(*physicalIndex)) : Value{});
  llvm::DenseMap<int64_t, Value> dimensionValues;
  unsigned metadataOffset = abi->interface.getArguments().size();
  for (auto [offset, dimension] : llvm::enumerate(abi->dimensionOrder))
    dimensionValues[dimension] = entry->getArgument(metadataOffset + offset);

  llvm::DenseMap<Value, Value> values;
  for (auto [logical, physicalView] : llvm::zip(
           function.getBody().front().getArguments(), sourceArguments))
    if (physicalView) values[logical] = physicalView;
  for (Operation &operation : function.getBody().front()) {
    if (auto constant = dyn_cast<intent::ConstantOp>(operation)) {
      FailureOr<Value> value = gpu::materializeScalarConstant(
          builder, constant.getLoc(), constant.getValue(),
          constant.getResult().getType());
      if (failed(value))
        return constant.emitOpError(
            "constant cannot be represented by its physical result type");
      values[constant.getResult()] = *value;
      if (Operation *target = value->getDefiningOp())
        if (Attribute node = operation.getAttr("intent.node"))
          target->setAttr(gpu::originAttr, node);
    } else if (auto dim = dyn_cast<intent::DimOp>(operation)) {
      RankedTensorType tensor = viewTensor(dim.getSource());
      if (tensor && dim.getAxis() < static_cast<uint64_t>(tensor.getRank()) &&
          !tensor.isDynamicDim(dim.getAxis())) {
        values[dim.getResult()] = builder.create<arith::ConstantIndexOp>(
            dim.getLoc(), tensor.getDimSize(dim.getAxis()));
        continue;
      }
      auto found = dimensionValues.find(dim.getDimension());
      if (found != dimensionValues.end())
        values[dim.getResult()] = found->second;
    }
  }
  Value pid = builder.create<gpu::ProgramIdOp>(function.getLoc(),
                                               builder.getIndexType(), 0);
  Value zero = builder.create<arith::ConstantIndexOp>(function.getLoc(), 0);
  Value one = builder.create<arith::ConstantIndexOp>(function.getLoc(), 1);
  Value runtimeOffset = zero;
  PhysicalExprAttr launchOffset =
      expression(context, PhysicalExprKind::Constant, 0);
  ScalarRegionLowering rootLowering(builder, values, sourceArguments,
                                    dimensionValues, parameterValues,
                                    canonicalAnalysis, physical);
  auto formWorksetCoordinate = [&](OpBuilder &nested, Location location,
                                   Value source, Value coordinate,
                                   Value step,
                                   unsigned worksetAxis) -> Value {
    auto domainType = dyn_cast<intent::DomainType>(source.getType());
    auto regionType = dyn_cast<intent::RegionType>(source.getType());
    uint64_t sourceId = domainType ? domainType.getOriginId()
                                  : regionType.getSourceId();
    std::optional<int64_t> dimension =
        sourceExtentDimension(source);
    if (!dimension || *dimension <= 0) {
      source.getDefiningOp()->emitOpError(
          "parallel workset coordinate has no logical dimension identity");
      return {};
    }
    auto mapped = nested.create<gpu::WorksetCoordinateOp>(
        location, nested.getIndexType(), coordinate, step,
        sourceId,
        /*sourceAxis=*/0, /*sourceRank=*/1, *dimension);
    mapped->setAttr(gpu::worksetAxisAttr,
                    nested.getI64IntegerAttr(worksetAxis));
    return mapped.getResult();
  };
  bool dispatchLoweringFailed = false;
  for (auto [groupIndex, workset] : llvm::enumerate(worksets)) {
    SmallVector<Value> runtimeExtents;
    SmallVector<Value> starts;
    SmallVector<Value> steps;
    Value runtimeLength = one;
    for (const IterationAxis &axis : workset.axes) {
      Location location = axis.source.getLoc();
      FailureOr<Value> start = rootLowering.lowerIndexValue(axis.start);
      FailureOr<Value> stop = rootLowering.lowerIndexValue(axis.stop);
      FailureOr<Value> step =
          axis.step
              ? rootLowering.lowerIndexValue(axis.step)
              : FailureOr<Value>(one);
      if (failed(start) || failed(stop) || failed(step))
        return axis.source.getDefiningOp()->emitOpError(
            "parallel workset runtime bounds are unavailable");
      Value distance = createBinary(builder, location,
                                    builder.getIndexType(), *stop, *start,
                                    BinaryOperator::Subtract);
      Value adjusted = createBinary(
          builder, location, builder.getIndexType(), distance,
          createBinary(builder, location, builder.getIndexType(), *step,
                       one, BinaryOperator::Subtract),
          BinaryOperator::Add);
      Value extent = createBinary(builder, location,
                                  builder.getIndexType(), adjusted, *step,
                                  BinaryOperator::FloorDivide);
      runtimeExtents.push_back(extent);
      starts.push_back(*start);
      steps.push_back(*step);
      runtimeLength = createBinary(builder, location,
                                   builder.getIndexType(), runtimeLength, extent,
                                   BinaryOperator::Multiply);
    }
    if (workset.singleton)
      runtimeExtents.push_back(one);
    Value segmentEnd = createBinary(builder, function.getLoc(),
                                    builder.getIndexType(), runtimeOffset,
                                    runtimeLength, BinaryOperator::Add);
    auto lowerGroup = [&](OpBuilder &nested, Value linear) -> LogicalResult {
      SmallVector<Attribute> launchExtents(workset.launchExtents.begin(),
                                           workset.launchExtents.end());
      Location worksetLocation = workset.singleton ? function.getLoc()
                                                   : workset.operation.getLoc();
      auto group = gpu::createExecutionGroup(
          nested, worksetLocation, linear, runtimeExtents,
          nested.getArrayAttr(launchExtents), nested.getDenseI64ArrayAttr(
              SmallVector<int64_t>(runtimeExtents.size(), static_cast<int64_t>(
                  workset.singleton ? gpu::CoordinateRole::Unspecified
                                    : gpu::CoordinateRole::Workset))),
          groupIndex, launchOffset, workset.launchLength);
      OpBuilder body = OpBuilder::atBlockBegin(&group.getBody().front());
      auto childValues = rootLowering.mapping();
      Block &sourceBlock = *workset.body;
      for (auto [axis, localCoordinate] :
           llvm::enumerate(group.getCoordinates())) {
        if (workset.singleton)
          break;
        Value scaled = createBinary(body, worksetLocation,
                                    body.getIndexType(), localCoordinate,
                                    steps[axis], BinaryOperator::Multiply);
        Value coordinate = createBinary(body, worksetLocation,
                                        body.getIndexType(), starts[axis],
                                        scaled, BinaryOperator::Add);
        childValues[workset.coordinateArguments[axis]] = formWorksetCoordinate(
            body, worksetLocation, workset.axes[axis].source, coordinate, steps[axis],
            axis);
      }
      ScalarRegionLowering lowering(body, std::move(childValues),
                                    sourceArguments, dimensionValues,
                                    parameterValues, canonicalAnalysis, physical);
      return lowering.lowerWorksetBlock(sourceBlock);
    };
    if (worksets.size() == 1) {
      if (failed(lowerGroup(builder, pid))) return failure();
      runtimeOffset = segmentEnd;
      launchOffset = binaryExpression(context, PhysicalExprKind::Add,
                                      launchOffset, workset.launchLength);
      continue;
    }
    Value afterOffset = createCompare(builder, function.getLoc(),
                                      builder.getI1Type(), pid, runtimeOffset,
                                      ComparePredicate::Ge);
    Value beforeEnd = createCompare(builder, function.getLoc(), builder.getI1Type(),
                                    pid, segmentEnd, ComparePredicate::Lt);
    Value active = createBinary(builder, function.getLoc(), builder.getI1Type(),
                                afterOffset, beforeEnd,
                                BinaryOperator::LogicalAnd);
    Location worksetLocation = workset.singleton ? function.getLoc()
                                                 : workset.operation.getLoc();
    auto dispatch = builder.create<scf::IfOp>(
        worksetLocation, active,
        [&](OpBuilder &nested, Location location) {
          Value local = createBinary(nested, location, nested.getIndexType(), pid,
                                     runtimeOffset, BinaryOperator::Subtract);
          if (failed(lowerGroup(nested, local))) {
            dispatchLoweringFailed = true;
            return;
          }
          nested.create<scf::YieldOp>(location);
        });
    runtimeOffset = segmentEnd;
    launchOffset = binaryExpression(context, PhysicalExprKind::Add, launchOffset,
                                    workset.launchLength);
  }
  if (dispatchLoweringFailed)
    return failure();
  builder.create<func::ReturnOp>(function.getLoc());
  if (failed(gpu::closeValueRelations(physical, gpu::ValueRelationScope::Pointwise)))
    return physical.emitError(
        "initial physical value relations are incomplete");
  if (failed(gpu::closeValueRelations(physical, gpu::ValueRelationScope::Contracts)))
    return physical.emitError(
        "initial physical contract relations are incomplete");
  function.erase();
  module->setAttr("intent_gpu.physical", builder.getUnitAttr());
  return success();
}

} // namespace

LogicalResult lowerCanonicalKIRToGPU(ModuleOp module,
                                     const GPUCapabilities &capabilities) {
  if (failed(verifyKernelModule(module)))
    return failure();
  if (capabilities.computeUnits <= 0 || capabilities.sharedMemoryPerUnit <= 0 ||
      capabilities.maxDynamicSharedMemoryPerBlock <= 0 ||
      capabilities.registersPerUnit <= 0 ||
      capabilities.maxThreadsPerBlock <= 0 ||
      capabilities.computeCapabilityMajor <= 0 ||
      capabilities.computeCapabilityMinor < 0 ||
      capabilities.singleToDoublePrecisionPerfRatio <= 0)
    return module.emitError("selected GPU capabilities are incomplete");
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  if (functions.size() != 1)
    return module.emitError("GPU construction requires exactly one kernel entry");
  if (failed(constructGPUProgram(module, capabilities, functions.front())))
    return failure();
  return gpu::completeGPUProgramConstruction(module);
}

} // namespace intent
