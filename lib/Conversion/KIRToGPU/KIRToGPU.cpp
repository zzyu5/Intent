#include "Intent/Conversion/KIRToGPU/KIRToGPU.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/Intent/IR/IntentAttrs.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
      context, static_cast<uint32_t>(kind), value,
      StringAttr::get(context, symbol), ArrayAttr::get(context, operands));
}

PhysicalExprAttr parameterExpression(MLIRContext *context, StringRef name) {
  return expression(context, PhysicalExprKind::Parameter, 0, name);
}

PhysicalExprAttr dimensionExpression(MLIRContext *context, int64_t dimension) {
  return expression(context, PhysicalExprKind::Dimension, dimension,
                    ("D" + Twine(dimension)).str());
}

PhysicalExprAttr binaryExpression(MLIRContext *context, PhysicalExprKind kind,
                                  PhysicalExprAttr lhs,
                                  PhysicalExprAttr rhs) {
  auto leftKind = static_cast<PhysicalExprKind>(lhs.getKind());
  auto rightKind = static_cast<PhysicalExprKind>(rhs.getKind());
  bool leftConstant = leftKind == PhysicalExprKind::Constant;
  bool rightConstant = rightKind == PhysicalExprKind::Constant;
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

bool dependsOnLoopCarry(Value root) {
  SmallVector<Value> worklist{root};
  llvm::SmallDenseSet<Value> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (auto argument = dyn_cast<BlockArgument>(current)) {
      auto loop = dyn_cast_or_null<scf::ForOp>(
          argument.getOwner()->getParentOp());
      if (loop && argument != loop.getInductionVar())
        return true;
      // scf::ForOp invokes its body builder before the new operation is fully
      // attached, so the parent op is not always queryable here.  Physical
      // function arguments are views/scalars; a fragment/record block argument
      // after the leading induction argument is therefore an in-construction
      // loop carry and already owns the relation that the next state must keep.
      if (argument.getArgNumber() > 0 &&
          isa<gpu::FragmentType, gpu::RecordType>(argument.getType()))
        return true;
      continue;
    }
    if (Operation *definition = current.getDefiningOp())
      worklist.append(definition->getOperands().begin(),
                      definition->getOperands().end());
  }
  return false;
}

FailureOr<Value> projectAccumulatorIdentity(OpBuilder &builder, Location location,
                                            Value identity, Type accumulator) {
  return gpu::projectPhysicalValueToSchema(builder, location, identity,
                                           accumulator);
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
      left.getShape() == right.getShape() &&
      left.getOwner() == right.getOwner() &&
      left.getValidity() == right.getValidity()) {
    bool leftCarry = dependsOnLoopCarry(lhs);
    bool rightCarry = dependsOnLoopCarry(rhs);
    if (leftCarry != rightCarry) {
      Value &projected = leftCarry ? rhs : lhs;
      gpu::FragmentType target = leftCarry ? left : right;
      FailureOr<Value> aligned =
          retargetBroadcast(builder, location, projected, target);
      if (failed(aligned))
        return failure();
      projected = *aligned;
      return success();
    }
  }
  if (leftLogical && rightLogical &&
      dimensionIds(leftLogical) == dimensionIds(rightLogical) &&
      left.getShape() == right.getShape() &&
      left.getOwner() == right.getOwner() &&
      left.getValidity() == right.getValidity()) {
    bool sameDimensions = left.getAxisMaps().size() == right.getAxisMaps().size();
    for (auto [leftMapping, rightMapping] :
         llvm::zip(left.getAxisMaps(), right.getAxisMaps()))
      sameDimensions &= cast<gpu::AxisMapAttr>(leftMapping).getDimensionId() ==
                        cast<gpu::AxisMapAttr>(rightMapping).getDimensionId();
    if (sameDimensions) {
      FailureOr<Value> aligned = retargetBroadcast(builder, location, rhs, left);
      if (failed(aligned))
        return failure();
      rhs = *aligned;
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
                  static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) ||
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
  if (isa_and_nonnull<gpu::BroadcastOp, gpu::SplatOp>(rhs.getDefiningOp())) {
    FailureOr<Value> aligned = retargetBroadcast(builder, location, rhs, left);
    if (failed(aligned))
      return failure();
    rhs = *aligned;
    return success();
  }
  if (isa_and_nonnull<gpu::BroadcastOp, gpu::SplatOp>(lhs.getDefiningOp())) {
    FailureOr<Value> aligned = retargetBroadcast(builder, location, lhs, right);
    if (failed(aligned))
      return failure();
    lhs = *aligned;
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
  std::string name;
};

struct PhysicalABI {
  SmallVector<Type> arguments;
  SmallVector<DictionaryAttr> argumentAttrs;
  SmallVector<unsigned> physicalArgumentForSource;
  SmallVector<unsigned> viewPhysicalArguments;
  llvm::DenseMap<int64_t, MetadataBinding> dimensions;
  SmallVector<int64_t> dimensionOrder;
};

FailureOr<PhysicalABI> buildPhysicalABI(func::FuncOp function,
                                        OpBuilder &builder) {
  PhysicalABI result;
  MLIRContext *context = function.getContext();
  Block &sourceEntry = function.getBody().front();
  auto parameterSchema =
      function->getAttrOfType<ArrayAttr>("intent.parameters");
  if (!parameterSchema || parameterSchema.size() != sourceEntry.getNumArguments())
    return function.emitError("canonical ABI parameter schema is missing");

  for (auto [abi, argument] : llvm::enumerate(sourceEntry.getArguments())) {
    auto parameter = cast<ParameterAttr>(parameterSchema[abi]);
    std::string name = parameter.getName().getValue().str();
    result.physicalArgumentForSource.push_back(result.arguments.size());
    auto logicalView = dyn_cast<ViewType>(argument.getType());
    if (!logicalView) {
      Type physicalType = argument.getType();
      StringRef kind;
      if (parameter.getKind() == 1) {
        if (!isa<IntegerType, IndexType, FloatType>(physicalType))
          return function.emitError(
              "runtime scalar ABI parameter has a non-scalar type");
        kind = "scalar";
      } else if (parameter.getKind() == 2) {
        auto constexprType = dyn_cast<ConstexprType>(physicalType);
        if (!constexprType)
          return function.emitError(
              "constexpr ABI parameter lost its canonical wrapper type");
        physicalType = constexprType.getValueType();
        if (auto enumeration = dyn_cast<EnumType>(physicalType))
          physicalType = builder.getI64Type();
        if (!isa<IntegerType, IndexType, FloatType>(physicalType))
          return function.emitError(
              "constexpr ABI payload is not a scalar provider value");
        kind = "constexpr";
      } else if (parameter.getKind() == 3) {
        if (!isa<IntegerType, IndexType, FloatType>(physicalType))
          return function.emitError(
              "kernel value ABI requires an explicit scalar value");
        kind = "value";
      } else {
        return function.emitError("canonical ABI parameter kind is unsupported");
      }
      result.arguments.push_back(physicalType);
      result.argumentAttrs.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr(gpu::abiKindAttr, builder.getStringAttr(kind)),
          builder.getNamedAttr(gpu::abiNameAttr, builder.getStringAttr(name)),
      }));
      continue;
    }
    auto tensor = dyn_cast<RankedTensorType>(logicalView.getTensor());
    DenseI64ArrayAttr canonicalIds =
        tensor ? dimensionIds(tensor) : DenseI64ArrayAttr();
    if (!tensor ||
        (canonicalIds && canonicalIds.size() != tensor.getRank()))
      return function.emitError("view lost canonical dimension identities");
    SmallVector<int64_t> ids(tensor.getRank(), 0);
    if (canonicalIds)
      llvm::copy(canonicalIds.asArrayRef(), ids.begin());
    for (unsigned axis = 0; axis < static_cast<unsigned>(tensor.getRank()); ++axis)
      if (tensor.isDynamicDim(axis) && ids[axis] <= 0)
        return function.emitError(
            "dynamic view axis lost its canonical dimension identity");
    result.viewPhysicalArguments.push_back(result.arguments.size());
    SmallVector<Attribute> strides;
    SmallVector<Attribute> extents;
    for (unsigned axis = 0; axis < static_cast<unsigned>(tensor.getRank()); ++axis) {
      std::string stride = ("S" + Twine(abi) + "_" + Twine(axis)).str();
      strides.push_back(StringAttr::get(context, stride));
      int64_t dimension = ids[axis];
      extents.push_back(
          tensor.isDynamicDim(axis)
              ? Attribute(dimensionExpression(context, dimension))
              : Attribute(expression(context, PhysicalExprKind::Constant,
                                     tensor.getDimSize(axis))));
      if (dimension > 0 && !result.dimensions.count(dimension))
        result.dimensions.try_emplace(
            dimension,
            MetadataBinding{dimension, static_cast<unsigned>(abi), axis,
                            ("D" + Twine(dimension)).str()});
    }
    auto constraints = logicalView.getConstraints();
    auto layout = gpu::ViewLayoutAttr::get(
        context, ArrayAttr::get(context, extents),
        DenseI64ArrayAttr::get(context, ids), true,
        ArrayAttr::get(context, strides),
        constraints.getAlias(), constraints.getNoalias());
    result.arguments.push_back(gpu::ViewType::get(
        context, tensor.getElementType(), tensor.getRank(), logicalView.getAccess(),
        abi, layout));
    result.argumentAttrs.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr(gpu::abiKindAttr, builder.getStringAttr("view")),
        builder.getNamedAttr(gpu::abiNameAttr, builder.getStringAttr(name)),
    }));
  }

  for (auto &entry : result.dimensions)
    result.dimensionOrder.push_back(entry.first);
  llvm::sort(result.dimensionOrder);
  for (int64_t dimension : result.dimensionOrder) {
    MetadataBinding &binding = result.dimensions.find(dimension)->second;
    result.arguments.push_back(builder.getIndexType());
    result.argumentAttrs.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr(gpu::abiKindAttr,
                             builder.getStringAttr("dimension")),
        builder.getNamedAttr(gpu::abiNameAttr,
                             builder.getStringAttr(binding.name)),
        builder.getNamedAttr(gpu::dimensionAttr,
                             builder.getI64IntegerAttr(dimension)),
        builder.getNamedAttr(gpu::sourceABIAttr,
                             builder.getI64IntegerAttr(binding.sourceABI)),
        builder.getNamedAttr(gpu::sourceAxisAttr,
                             builder.getI64IntegerAttr(binding.sourceAxis)),
    }));
  }
  for (auto [abi, argument] : llvm::enumerate(sourceEntry.getArguments())) {
    auto tensor = viewTensor(argument);
    if (!tensor)
      continue;
    for (unsigned axis = 0; axis < static_cast<unsigned>(tensor.getRank()); ++axis) {
      result.arguments.push_back(builder.getIndexType());
      std::string name = ("S" + Twine(abi) + "_" + Twine(axis)).str();
      result.argumentAttrs.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr(gpu::abiKindAttr,
                               builder.getStringAttr("stride")),
          builder.getNamedAttr(gpu::abiNameAttr, builder.getStringAttr(name)),
          builder.getNamedAttr(gpu::sourceABIAttr,
                               builder.getI64IntegerAttr(abi)),
          builder.getNamedAttr(gpu::sourceAxisAttr,
                               builder.getI64IntegerAttr(axis)),
      }));
    }
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
    auto schema = function->getAttrOfType<ArrayAttr>("intent.parameters");
    if (!schema || argument.getArgNumber() >= schema.size())
      return failure();
    auto parameter = dyn_cast<ParameterAttr>(schema[argument.getArgNumber()]);
    if (!parameter || (parameter.getKind() != 1 && parameter.getKind() != 2))
      return failure();
    Type type = value.getType();
    if (auto constexprType = dyn_cast<ConstexprType>(type))
      type = constexprType.getValueType();
    if (!isa<IntegerType, IndexType>(type))
      return failure();
    return expression(context, PhysicalExprKind::ScalarABI, 0,
                      parameter.getName().getValue());
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
      return dimensionExpression(context, dim.getDimension());
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
              static_cast<uint32_t>(PhysicalExprKind::Constant) &&
          start->getValue() == 0 &&
          step->getKind() ==
              static_cast<uint32_t>(PhysicalExprKind::Constant) &&
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

FailureOr<int64_t> physicalDimensionIdentity(Operation *origin,
                                             int64_t logicalIdentity) {
  if (!origin || logicalIdentity <= 0)
    return failure();
  func::FuncOp function = origin->getParentOfType<func::FuncOp>();
  if (!function)
    return logicalIdentity;
  std::optional<int64_t> physicalIdentity;
  bool conflict = false;
  function.walk([&](intent::SubregionOp subregion) {
    bool definesIdentity = llvm::any_of(
        subregion.getExtentDimensions(), [&](Attribute attribute) {
          return cast<IntegerAttr>(attribute).getInt() == logicalIdentity;
        });
    if (!definesIdentity)
      return;
    std::optional<int64_t> candidate =
        sourceExtentDimension(subregion.getInputs().front());
    if (!candidate)
      return;
    if (physicalIdentity && *physicalIdentity != *candidate)
      conflict = true;
    else
      physicalIdentity = *candidate;
  });
  if (conflict)
    return failure();
  return physicalIdentity ? FailureOr<int64_t>(*physicalIdentity)
                          : FailureOr<int64_t>(logicalIdentity);
}

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
  auto constant = value.getDefiningOp<intent::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
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

FailureOr<PhysicalExprAttr> fragmentExtentForDimension(Operation *origin,
                                                      int64_t dimension) {
  if (!origin || dimension <= 0)
    return failure();
  func::FuncOp function = origin->getParentOfType<func::FuncOp>();
  if (function) {
    bool launchVisible = false;
    for (BlockArgument argument : function.getArguments()) {
      auto tensor = viewTensor(argument);
      auto identities = tensor ? dimensionIds(tensor) : DenseI64ArrayAttr();
      launchVisible |=
          identities && llvm::is_contained(identities.asArrayRef(), dimension);
    }
    if (launchVisible)
      return dimensionExpression(origin->getContext(), dimension);
  }
  gpu::ParameterOp declaration;
  bool ambiguous = false;
  origin->getParentOfType<ModuleOp>().walk([&](gpu::ParameterOp parameter) {
    auto binding = parameter->getAttrOfType<IntegerAttr>(gpu::dimensionAttr);
    if (!binding || binding.getInt() != dimension)
      return;
    if (declaration && declaration != parameter)
      ambiguous = true;
    else
      declaration = parameter;
  });
  return declaration && !ambiguous
             ? FailureOr<PhysicalExprAttr>(parameterExpression(
                   origin->getContext(),
                   declaration.getParameter().getName().getValue()))
             : FailureOr<PhysicalExprAttr>(failure());
}

FailureOr<PhysicalExprAttr> fragmentExtentExpression(RankedTensorType tensor,
                                                     Operation *origin,
                                                     unsigned axis) {
  DenseI64ArrayAttr identities = dimensionIds(tensor);
  if (!identities || axis >= identities.size())
    return failure();
  if (std::optional<int64_t> staticBound =
          subregionStaticExtentBound(origin, identities[axis]))
    return expression(tensor.getContext(), PhysicalExprKind::Constant,
                      *staticBound);
  FailureOr<int64_t> physicalIdentity =
      physicalDimensionIdentity(origin, identities[axis]);
  if (failed(physicalIdentity))
    return failure();
  FailureOr<PhysicalExprAttr> extent =
      fragmentExtentForDimension(origin, *physicalIdentity);
  if (failed(extent))
    return failure();
  return *extent;
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

  SmallVector<bool> reduced(sourceType.getShape().size(), false);
  for (int64_t axis : reducedAxes) {
    if (axis < 0 || axis >= static_cast<int64_t>(reduced.size()) ||
        reduced[axis])
      return failure();
    reduced[axis] = true;
  }
  if (logicalType.getRank() !=
      static_cast<int64_t>(sourceType.getShape().size() - reducedAxes.size()))
    return failure();

  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  unsigned resultAxis = 0;
  for (auto [axis, extent] : llvm::enumerate(sourceType.getShape())) {
    if (reduced[axis])
      continue;
    shape.push_back(extent);
    auto sourceMapping = cast<AxisMapAttr>(sourceType.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(
        logical.getContext(), sourceMapping.getSourceId(),
        sourceMapping.getSourceAxis(), sourceMapping.getDimensionId(), resultAxis,
        sourceMapping.getDerived()));
    ++resultAxis;
  }
  auto prototype = FragmentType::get(
      logical.getContext(), logicalType.getElementType(),
      ArrayAttr::get(logical.getContext(), shape),
      ArrayAttr::get(logical.getContext(), mappings), sourceType.getValidity(),
      sourceType.getOwner());
  return convertDataType(logical, origin, Type(prototype), 1, resultIndex);
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

LogicalResult collectDomainAxes(Value source,
                                SmallVectorImpl<intent::DomainOp> &axes);

struct OrderedIterationAxis {
  Value start;
  Value stop;
  Value step;
  Value coordinatePrototype;
};

LogicalResult collectOrderedIterationAxes(
    Value source, SmallVectorImpl<OrderedIterationAxis> &axes);

class ScalarRegionLowering {
public:
  ScalarRegionLowering(OpBuilder &builder, llvm::DenseMap<Value, Value> values,
                       ArrayRef<Value> views,
                       llvm::DenseMap<int64_t, Value> dimensions,
                       llvm::DenseMap<StringAttr, Value> parameters,
                       unsigned orderedDepth = 0)
      : builder(builder), values(std::move(values)), views(views),
        dimensions(std::move(dimensions)), parameters(std::move(parameters)),
        orderedDepth(orderedDepth) {}

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

  llvm::DenseMap<Value, Value> &mapping() { return values; }
  FailureOr<Value> lowerValue(Value source) { return get(source); }

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
                               dimensions, parameters, orderedDepth);
    FailureOr<SmallVector<Value>> yielded = child.lowerBlock(source.front());
    if (failed(yielded))
      return failure();
    if (!resultTypes.empty()) {
      if (yielded->size() != resultTypes.size())
        return failure();
      for (auto [index, type] : llvm::enumerate(resultTypes)) {
        FailureOr<Value> projected = projectAccumulatorIdentity(
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
    auto kind = static_cast<PhysicalExprKind>(expression.getKind());
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
      auto found = parameters.find(expression.getSymbol());
      return found == parameters.end() ? FailureOr<Value>(failure())
                                       : FailureOr<Value>(found->second);
    }
    Operation *parent = builder.getInsertionBlock()->getParentOp();
    if (!parent)
      return failure();
    func::FuncOp function = dyn_cast<func::FuncOp>(parent);
    if (!function)
      function = parent->getParentOfType<func::FuncOp>();
    if (!function)
      return failure();
    if (kind == PhysicalExprKind::ScalarABI) {
      for (BlockArgument argument : function.getArguments()) {
        auto name = function.getArgAttrOfType<StringAttr>(
            argument.getArgNumber(), gpu::abiNameAttr);
        if (name == expression.getSymbol())
          return Value(argument);
      }
      return failure();
    }
    if (expression.getOperands().size() != 2)
      return failure();
    FailureOr<Value> lhs = physicalExtentValue(
        location, cast<PhysicalExprAttr>(expression.getOperands()[0]));
    FailureOr<Value> rhs = physicalExtentValue(
        location, cast<PhysicalExprAttr>(expression.getOperands()[1]));
    if (failed(lhs) || failed(rhs))
      return failure();
    if (kind == PhysicalExprKind::Add)
      return createBinary(builder, location, builder.getIndexType(), *lhs, *rhs,
                          BinaryOperator::Add);
    if (kind == PhysicalExprKind::Subtract)
      return createBinary(builder, location, builder.getIndexType(), *lhs, *rhs,
                          BinaryOperator::Subtract);
    if (kind == PhysicalExprKind::Multiply)
      return createBinary(builder, location, builder.getIndexType(), *lhs, *rhs,
                          BinaryOperator::Multiply);
    if (kind == PhysicalExprKind::FloorDiv)
      return createBinary(builder, location, builder.getIndexType(), *lhs, *rhs,
                          BinaryOperator::FloorDivide);
    if (kind == PhysicalExprKind::Minimum)
      return createBinary(builder, location, builder.getIndexType(), *lhs, *rhs,
                          BinaryOperator::Minimum);
    if (kind == PhysicalExprKind::Maximum)
      return createBinary(builder, location, builder.getIndexType(), *lhs, *rhs,
                          BinaryOperator::Maximum);
    if (kind == PhysicalExprKind::CeilDiv) {
      Value one = builder.create<arith::ConstantIndexOp>(location, 1);
      Value adjusted = createBinary(
          builder, location, builder.getIndexType(), *lhs,
          createBinary(builder, location, builder.getIndexType(), *rhs, one,
                       BinaryOperator::Subtract),
          BinaryOperator::Add);
      return createBinary(builder, location, builder.getIndexType(), adjusted,
                          *rhs, BinaryOperator::FloorDivide);
    }
    return failure();
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

  void bindExtentDimensions(ArrayAttr identities, Value extent) {
    if (!identities || identities.size() != 1)
      return;
    int64_t identity = cast<IntegerAttr>(identities[0]).getInt();
    if (identity > 0)
      dimensions[identity] = extent;
  }

  FailureOr<SmallVector<Value>> accessCoordinates(Operation *operation) {
    auto relation = operation->getAttrOfType<IndexRelationAttr>("index");
    if (!relation)
      return failure();
    SmallVector<Value> coordinates;
    FailureOr<Value> resource = get(operation->getOperand(0));
    if (failed(resource))
      return failure();
    gpu::FragmentType valuePrototype;
    if (operation->getNumResults() == 0)
      if (auto valueIndex =
              operation->getAttrOfType<IntegerAttr>("value_operand_index")) {
        int64_t index = valueIndex.getInt();
        if (index >= 0 && index < operation->getNumOperands()) {
          FailureOr<Value> value = get(operation->getOperand(index));
          if (succeeded(value))
            valuePrototype = dyn_cast<gpu::FragmentType>((*value).getType());
        }
      }
    auto resultExtent = [&](size_t axis,
                            PhysicalExprAttr fallback)
        -> FailureOr<PhysicalExprAttr> {
      if (valuePrototype) {
        size_t resultRank = relation.getResultDimensions().size();
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
      Value one = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 1);
      Value distance = createBinary(builder, operation->getLoc(),
                                    builder.getIndexType(), stop, start,
                                    BinaryOperator::Subtract);
      Value adjusted = createBinary(
          builder, operation->getLoc(), builder.getIndexType(), distance,
          createBinary(builder, operation->getLoc(), builder.getIndexType(), step,
                       one, BinaryOperator::Subtract),
          BinaryOperator::Add);
      Value extent = createBinary(builder, operation->getLoc(),
                                  builder.getIndexType(), adjusted, step,
                                  BinaryOperator::FloorDivide);
      auto type = fragmentType(operation->getContext(), builder.getIndexType(),
                               {physicalExtent},
                               {{sourceId, sourceAxis, dimensionId, derived}});
      return Value(builder.create<gpu::MakeRangeOp>(
          operation->getLoc(), type, start, extent, step, sourceId, sourceAxis,
          derived));
    };
    unsigned sourceAxis = 0;
    unsigned resultAxis = 0;
    unsigned basicResultAxes = 0;
    for (Attribute attribute : relation.getTerms()) {
      uint32_t kind = cast<IndexTermAttr>(attribute).getKind();
      if (kind == 0 || kind == 1 || kind == 4 || kind == 5)
        ++basicResultAxes;
    }
    if (basicResultAxes > relation.getResultDimensions().size())
      return failure();
    unsigned advancedRank =
        relation.getResultDimensions().size() - basicResultAxes;
    std::optional<unsigned> advancedStart;
    for (Attribute attribute : relation.getTerms()) {
      auto term = cast<IndexTermAttr>(attribute);
      if (term.getKind() == 1) {
        ++resultAxis;
        continue;
      }
      if (term.getKind() == 0) {
        auto view = dyn_cast<gpu::ViewType>((*resource).getType());
        auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
        if ((!view && !fragment) ||
            (view && sourceAxis >= view.getRank()) ||
            (fragment && sourceAxis >= fragment.getShape().size())) {
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
        if (view) {
          stop = builder.create<gpu::DimOp>(operation->getLoc(),
                                            builder.getIndexType(), *resource,
                                            sourceAxis);
          sourceId =
              (static_cast<uint64_t>(view.getAbiIndex()) + 1) * 65536 +
              sourceAxis + 1;
          extent = cast<PhysicalExprAttr>(
              view.getLayout().getExtents()[sourceAxis]);
        } else {
          auto mapping =
              cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[sourceAxis]);
          FailureOr<Value> physicalExtent = physicalExtentValue(
              operation->getLoc(),
              cast<PhysicalExprAttr>(fragment.getShape()[sourceAxis]));
          if (failed(physicalExtent)) {
            operation->emitOpError(
                "fragment full-slice extent is not materialized in the current program: ")
                << fragment.getShape()[sourceAxis];
            return failure();
          }
          stop = *physicalExtent;
          sourceId = mapping.getSourceId();
          logicalSourceAxis = mapping.getSourceAxis();
          derived = mapping.getDerived();
          extent =
              cast<PhysicalExprAttr>(fragment.getShape()[sourceAxis]);
        }
        int64_t dimension = 0;
        if (view) {
          auto dimensions = view.getLayout().getDimensionIds();
          if (sourceAxis >= dimensions.size())
            return failure();
          dimension = dimensions[sourceAxis];
        } else {
          dimension = cast<gpu::AxisMapAttr>(
                          fragment.getAxisMaps()[sourceAxis])
                          .getDimensionId();
        }
        if (dimension <= 0)
          return failure();
        // A full slice of an already-physical fragment consumes that
        // fragment's current extent.  Reconstructing the helper-local logical
        // result dimension here would discard an enclosing region segment.
        if (view) {
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
      if (term.getKind() == 2) {
        coordinates.push_back(builder.create<arith::ConstantIndexOp>(
            operation->getLoc(), term.getStaticValues()[0]));
        ++sourceAxis;
        continue;
      }
      if (term.getKind() == 3) {
        int64_t position = term.getOperandPositions()[0];
        if (position < 0 || position >= operation->getNumOperands())
          return failure();
        FailureOr<Value> coordinate = get(operation->getOperand(position));
        if (failed(coordinate))
          return failure();
        Value physicalCoordinate = *coordinate;
        if (auto fragment = dyn_cast<gpu::FragmentType>(physicalCoordinate.getType())) {
          auto logicalCoordinate =
              dyn_cast<RankedTensorType>(operation->getOperand(position).getType());
          if (logicalCoordinate) {
            if (fragment.getShape().size() > advancedRank)
              return failure();
            if (!advancedStart) {
              advancedStart = resultAxis;
              resultAxis += advancedRank;
            }
          }
          SmallVector<Attribute> mappings;
          for (Attribute attribute : fragment.getAxisMaps()) {
            auto mapping = cast<gpu::AxisMapAttr>(attribute);
            mappings.push_back(gpu::AxisMapAttr::get(
                operation->getContext(), mapping.getSourceId(),
                mapping.getSourceAxis(), mapping.getDimensionId(),
                mappings.size(), mapping.getDerived()));
          }
          auto target = gpu::FragmentType::get(
              operation->getContext(), fragment.getElementType(),
              fragment.getShape(), builder.getArrayAttr(mappings),
              fragment.getValidity(), fragment.getOwner());
          if (target != fragment)
            physicalCoordinate = builder.create<gpu::BroadcastOp>(
                operation->getLoc(), target, physicalCoordinate);
        }
        coordinates.push_back(physicalCoordinate);
        ++sourceAxis;
        continue;
      }
      if (term.getKind() == 4) {
        int64_t position = term.getOperandPositions()[0];
        FailureOr<Value> range = get(operation->getOperand(position));
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
        FailureOr<Value> coordinate = makeRange(
            logicalSourceAxis, start, stop, step, sourceId,
            rangeType.getDimensionId(), rangeType.getDerived(), extent);
        if (failed(coordinate))
          return failure();
        if ((*range).getDefiningOp()->hasAttr(gpu::sourceSubregionAttr))
          coordinate->getDefiningOp()->setAttr(gpu::sourceSubregionAttr,
                                               builder.getUnitAttr());
        coordinates.push_back(*coordinate);
        ++sourceAxis;
        ++resultAxis;
        continue;
      }
      if (term.getKind() == 5) {
        constexpr int64_t absent = std::numeric_limits<int64_t>::min();
        SmallVector<Value> bounds;
        for (unsigned component = 0; component < 3; ++component) {
          int64_t position = term.getOperandPositions()[component];
          int64_t literal = term.getStaticValues()[component];
          if (position >= 0) {
            FailureOr<Value> value = get(operation->getOperand(position));
            if (failed(value))
              return failure();
            bounds.push_back(*value);
          } else if (literal != absent) {
            bounds.push_back(builder.create<arith::ConstantIndexOp>(
                operation->getLoc(), literal));
          } else {
            auto view = dyn_cast<gpu::ViewType>((*resource).getType());
            auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
            if ((!view && !fragment) ||
                (view && sourceAxis >= view.getRank()) ||
                (fragment && sourceAxis >= fragment.getShape().size()))
              return failure();
            if (component == 0) {
              bounds.push_back(builder.create<arith::ConstantIndexOp>(
                  operation->getLoc(), 0));
            } else if (component == 1) {
              if (view) {
                bounds.push_back(builder.create<gpu::DimOp>(
                    operation->getLoc(), builder.getIndexType(), *resource,
                    sourceAxis));
              } else {
                FailureOr<Value> physicalExtent = physicalExtentValue(
                    operation->getLoc(), cast<PhysicalExprAttr>(
                                             fragment.getShape()[sourceAxis]));
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
        auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
        if ((!view && !fragment) ||
            (view && sourceAxis >= view.getRank()) ||
            (fragment && sourceAxis >= fragment.getShape().size()))
          return failure();
        PhysicalExprAttr fallback =
            view ? cast<PhysicalExprAttr>(
                       view.getLayout().getExtents()[sourceAxis])
                 : cast<PhysicalExprAttr>(fragment.getShape()[sourceAxis]);
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
          sourceId =
              (static_cast<uint64_t>(view.getAbiIndex()) + 1) * 65536 +
              sourceAxis + 1;
          auto dimensions = view.getLayout().getDimensionIds();
          if (sourceAxis >= dimensions.size())
            return failure();
          dimension = dimensions[sourceAxis];
        } else {
          auto mapping =
              cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[sourceAxis]);
          sourceId = mapping.getSourceId();
          logicalSourceAxis = mapping.getSourceAxis();
          dimension = mapping.getDimensionId();
          derived = mapping.getDerived();
        }
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
    auto relation = operation->getAttrOfType<IndexRelationAttr>("index");
    if (!relation)
      return failure();
    SmallVector<int64_t> axes;
    int64_t sourceAxis = 0;
    for (Attribute attribute : relation.getTerms()) {
      auto term = cast<IndexTermAttr>(attribute);
      if (term.getKind() == 1)
        continue;
      axes.push_back(sourceAxis++);
    }
    return axes;
  }

  FailureOr<Type> accessResultType(Operation *operation, Type logical,
                                   ArrayRef<Value> coordinates,
                                   std::optional<Type> prototype = std::nullopt) {
    auto relation = operation->getAttrOfType<IndexRelationAttr>("index");
    if (!relation)
      return failure();
    SmallVector<Attribute> liftedShape;
    SmallVector<Attribute> liftedMappings;
    uint64_t liftedOwner = 1;
    if (prototype) {
      auto fragment = dyn_cast<gpu::FragmentType>(*prototype);
      size_t logicalRank = relation.getResultDimensions().size();
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
    for (Attribute attribute : relation.getTerms()) {
      auto term = cast<IndexTermAttr>(attribute);
      if (term.getKind() == 1)
        continue;
      if (coordinateIndex >= coordinates.size())
        return failure();
      Value current = coordinates[coordinateIndex++];
      if (term.getKind() != 3)
        continue;
      int64_t position = term.getOperandPositions()[0];
      if (position < 0 || position >= operation->getNumOperands() ||
          isa<RankedTensorType>(operation->getOperand(position).getType()))
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
      size_t resultRank = relation.getResultDimensions().size();
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
      ArrayRef<int64_t> dimensions = relation.getResultDimensions();
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
    for (Attribute attribute : relation.getTerms()) {
      uint32_t kind = cast<IndexTermAttr>(attribute).getKind();
      if (kind == 0 || kind == 1 || kind == 4 || kind == 5)
        ++basicResultAxes;
    }
    if (basicResultAxes > relation.getResultDimensions().size())
      return failure();
    unsigned advancedRank =
        relation.getResultDimensions().size() - basicResultAxes;
    bool advancedMapped = false;
    auto resultDimension = [&](size_t axis) -> FailureOr<int64_t> {
      ArrayRef<int64_t> dimensions = relation.getResultDimensions();
      if (axis < liftedRank || axis - liftedRank >= dimensions.size())
        return failure();
      return dimensions[axis - liftedRank];
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
      gpu::AxisMapAttr unitMapping;
      gpu::AxisMapAttr nonUnitMapping;
      bool ambiguousUnit = false;
      bool ambiguousNonUnit = false;
      unsigned coordinateIndex = 0;
      for (Attribute attribute : relation.getTerms()) {
        auto term = cast<IndexTermAttr>(attribute);
        if (term.getKind() == 1)
          continue;
        if (coordinateIndex >= coordinates.size())
          return {};
        Value current = coordinates[coordinateIndex++];
        if (term.getKind() != 3)
          continue;
        int64_t position = term.getOperandPositions()[0];
        if (position < 0 || position >= operation->getNumOperands() ||
            !isa<RankedTensorType>(operation->getOperand(position).getType()))
          continue;
        auto source = dyn_cast<gpu::FragmentType>(current.getType());
        if (!source || source.getShape().size() > advancedRank)
          continue;
        unsigned alignedStart = advancedRank - source.getShape().size();
        if (advancedAxis < alignedStart)
          continue;
        unsigned localAxis = advancedAxis - alignedStart;
        auto mapping =
            cast<gpu::AxisMapAttr>(source.getAxisMaps()[localAxis]);
        auto extent = cast<gpu::PhysicalExprAttr>(source.getShape()[localAxis]);
        bool unit = extent.getKind() ==
                        static_cast<uint32_t>(PhysicalExprKind::Constant) &&
                    extent.getValue() == 1;
        auto sameMapping = [](gpu::AxisMapAttr lhs, gpu::AxisMapAttr rhs) {
          return lhs.getSourceId() == rhs.getSourceId() &&
                 lhs.getSourceAxis() == rhs.getSourceAxis() &&
                 lhs.getDerived() == rhs.getDerived();
        };
        if (!unit) {
          if (!nonUnitMapping)
            nonUnitMapping = mapping;
          else if (!sameMapping(nonUnitMapping, mapping))
            ambiguousNonUnit = true;
          continue;
        }
        if (!unitMapping)
          unitMapping = mapping;
        else if (!sameMapping(unitMapping, mapping))
          ambiguousUnit = true;
      }
      if (ambiguousNonUnit)
        return {};
      if (nonUnitMapping)
        return nonUnitMapping;
      return ambiguousUnit ? gpu::AxisMapAttr() : unitMapping;
    };
    for (Attribute attribute : relation.getTerms()) {
      auto term = cast<IndexTermAttr>(attribute);
      if (term.getKind() == 1) {
        FailureOr<int64_t> dimension = resultDimension(resultAxis);
        FailureOr<PhysicalAxisIdentity> identity =
            resultAxisIdentity(operation, /*resultIndex=*/0, resultAxis);
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
      if (term.getKind() == 0 || term.getKind() == 4 ||
          term.getKind() == 5) {
        if (failed(appendCoordinate(current))) {
          operation->emitOpError(
              "access range coordinate cannot map a result axis");
          return failure();
        }
        continue;
      }
      if (term.getKind() == 3 && isa<gpu::FragmentType>(current.getType())) {
        int64_t position = term.getOperandPositions()[0];
        if (position < 0 || position >= operation->getNumOperands())
          return failure();
        if (!isa<RankedTensorType>(operation->getOperand(position).getType()))
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
                  resultAxisIdentity(operation, /*resultIndex=*/0, resultAxis);
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
      if (term.getKind() != 2 && term.getKind() != 3)
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
          static_cast<uint32_t>(PhysicalExprKind::Constant)) {
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
      auto type = gpu::RangeType::get(
          operation->getContext(), logical.getSourceId(), 0,
          sourceRange.getDimensionId(), sourceRange.getDerived());
      auto target =
          builder.create<gpu::RangeOp>(location, type, start, stop, step);
      target->setAttr(gpu::sourceSubregionAttr, builder.getUnitAttr());
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
        auto coordinateType = gpu::FragmentType::get(
            operation->getContext(), builder.getIndexType(),
            builder.getArrayAttr({extent}),
            builder.getArrayAttr({gpu::AxisMapAttr::get(
                operation->getContext(), mapping.getSourceId(),
                mapping.getSourceAxis(), mapping.getDimensionId(), 0,
                mapping.getDerived())}),
            fragment.getValidity(), fragment.getOwner());
        Value coordinate = builder.create<gpu::MakeRangeOp>(
            location, coordinateType, zero, *physicalExtent, one,
            mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDerived());
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
      Value distance = createBinary(builder, location, builder.getIndexType(),
                                    stop, start, BinaryOperator::Subtract);
      Value one = builder.create<arith::ConstantIndexOp>(location, 1);
      Value adjusted = createBinary(
          builder, location, builder.getIndexType(), distance,
          createBinary(builder, location, builder.getIndexType(), step, one,
                       BinaryOperator::Subtract),
          BinaryOperator::Add);
      Value extent = createBinary(builder, location, builder.getIndexType(),
                                  adjusted, step, BinaryOperator::FloorDivide);
      FailureOr<Type> result =
          convertDataType(indices.getResult().getType(), operation);
      if (failed(result) || !isa<gpu::FragmentType>(*result))
        return indices.emitOpError("indices result has no physical fragment type");
      auto rangeType = cast<gpu::RangeType>((*source).getType());
      auto genericType = cast<gpu::FragmentType>(*result);
      auto resultType = gpu::FragmentType::get(
          operation->getContext(), genericType.getElementType(),
          genericType.getShape(),
          builder.getArrayAttr({gpu::AxisMapAttr::get(
              operation->getContext(), rangeType.getSourceId(),
              rangeType.getSourceAxis(), rangeType.getDimensionId(), 0,
              rangeType.getDerived())}),
          genericType.getValidity(), genericType.getOwner());
      auto target = builder.create<gpu::MakeRangeOp>(
          location, resultType, start, extent, step, rangeType.getSourceId(),
          rangeType.getSourceAxis(), rangeType.getDerived());
      if ((*source).getDefiningOp()->hasAttr(gpu::sourceSubregionAttr))
        target->setAttr(gpu::sourceSubregionAttr, builder.getUnitAttr());
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
                  sourceMapping.getSourceAxis(), resultMapping.getDimensionId(),
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
          std::optional<PhysicalExprAttr> launched;
          for (auto [sourceAxis, attribute] :
               llvm::enumerate(source.getAxisMaps())) {
            auto mapping = dyn_cast<gpu::AxisMapAttr>(attribute);
            if (!mapping ||
                mapping.getDimensionId() != extent.getDimension())
              continue;
            auto candidate =
                cast<PhysicalExprAttr>(source.getShape()[sourceAxis]);
            if (launched && *launched != candidate)
              return reshape.emitOpError(
                  "reshape dimension has ambiguous physical extents");
            launched = candidate;
          }
          if (!launched) {
            FailureOr<PhysicalExprAttr> derived = physicalShapeExpression(
                operation->getOperand(operand), operation);
            if (failed(derived))
              return reshape.emitOpError(
                  "reshape dynamic extent has no exact physical authority");
            launched = *derived;
          }
          physical = *launched;
        } else if (extent.getKind() == 2) {
          if (inferredAxis)
            return reshape.emitOpError(
                "reshape has more than one inferred physical extent");
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
      if (inferredAxis) {
        PhysicalExprAttr sourceProduct = expression(
            operation->getContext(), PhysicalExprKind::Constant, 1);
        for (Attribute extent : llvm::drop_begin(source.getShape(), worksetRank))
          sourceProduct = binaryExpression(
              operation->getContext(), PhysicalExprKind::Multiply,
              sourceProduct, cast<PhysicalExprAttr>(extent));
        logicalResultExtents[*inferredAxis] = binaryExpression(
            operation->getContext(), PhysicalExprKind::FloorDiv,
            sourceProduct, knownResultProduct);
      }
      for (auto [axis, extent] : llvm::enumerate(logicalResultExtents)) {
        shape.push_back(extent);
        FailureOr<Attribute> mapping = resultMapping(axis, worksetRank + axis);
        if (failed(mapping))
          return reshape.emitOpError(
              "reshape result axis has no coordinate identity");
        mappings.push_back(*mapping);
      }

      auto result = gpu::FragmentType::get(
          operation->getContext(), source.getElementType(),
          builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
          source.getValidity(), source.getOwner());
      FailureOr<ArrayAttr> reassociation = gpu::inferReshapeReassociation(
          source, result, worksetRank, worksetRank);
      if (failed(reassociation))
        return reshape.emitOpError(
            "canonical reshape relation cannot be decomposed into exact "
            "row-major physical groups");
      for (Attribute attribute : *reassociation) {
        auto group = cast<gpu::ReshapeGroupAttr>(attribute);
        for (int64_t logicalResultAxis :
             group.getResultAxes().asArrayRef()) {
          unsigned resultAxis = worksetRank + logicalResultAxis;
          auto resultExtent =
              cast<PhysicalExprAttr>(result.getShape()[resultAxis]);
          if (resultExtent.getKind() ==
                  static_cast<uint32_t>(PhysicalExprKind::Constant) &&
              resultExtent.getValue() == 1)
            continue;
          std::optional<unsigned> matchedSource;
          for (int64_t logicalSourceAxis :
               group.getSourceAxes().asArrayRef()) {
            unsigned sourceAxis = worksetRank + logicalSourceAxis;
            if (source.getShape()[sourceAxis] != resultExtent)
              continue;
            if (matchedSource) {
              matchedSource.reset();
              break;
            }
            matchedSource = sourceAxis;
          }
          if (!matchedSource)
            continue;
          auto mapping = cast<gpu::AxisMapAttr>(
              source.getAxisMaps()[*matchedSource]);
          mappings[resultAxis] = gpu::AxisMapAttr::get(
              operation->getContext(), mapping.getSourceId(),
              mapping.getSourceAxis(), mapping.getDimensionId(), resultAxis,
              mapping.getDerived());
        }
      }
      result = gpu::FragmentType::get(
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
                                                 unary.getOperatorKind());
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
                                                   binary.getOperatorKind());
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
          FailureOr<Value> aligned =
              retargetBroadcast(builder, location, *condition, target);
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
    if (auto reduce = dyn_cast<intent::ReduceOp>(operation)) {
      SmallVector<Value> inputs;
      for (Value input : reduce.getInputs()) {
        FailureOr<Value> lowered = get(input);
        if (failed(lowered))
          return reduce.emitOpError("reduce physical operand is unavailable");
        inputs.push_back(*lowered);
      }
      SmallVector<int64_t> axes;
      for (Attribute axis : reduce.getAxes())
        axes.push_back(cast<IntegerAttr>(axis).getInt());
      SmallVector<Type> results;
      for (auto [index, logical] : llvm::enumerate(reduce.getResultTypes())) {
        if (index >= reduce.getSourceCount())
          return reduce.emitOpError(
              "reduce result has no corresponding physical source");
        FailureOr<Type> converted = convertReductionResultType(
            logical, operation, inputs[index], axes, index);
        if (failed(converted))
          return reduce.emitOpError("reduce result has no physical schema")
                 << "; logical_result=" << logical
                 << ", physical_source=" << inputs[index].getType()
                 << ", reduced_axes=" << reduce.getAxes();
        results.push_back(*converted);
      }
      for (unsigned index = 0; index < reduce.getIdentityCount(); ++index) {
        unsigned operand = reduce.getSourceCount() + index;
        FailureOr<Value> identity = projectAccumulatorIdentity(
            builder, location, inputs[operand], results[index]);
        if (failed(identity))
          return reduce.emitOpError(
              "reduce identity cannot adopt its physical accumulator schema");
        inputs[operand] = *identity;
      }
      OperationState state(location, gpu::ReduceOp::getOperationName());
      state.addOperands(inputs);
      state.addTypes(results);
      state.addAttribute("axes", builder.getDenseI64ArrayAttr(axes));
      state.addAttribute("source_count",
                         builder.getI64IntegerAttr(reduce.getSourceCount()));
      state.addAttribute("identity_count",
                         builder.getI64IntegerAttr(reduce.getIdentityCount()));
      state.addAttribute("capture_count",
                         builder.getI64IntegerAttr(reduce.getCaptureCount()));
      state.addRegion();
      Operation *raw = builder.create(state);
      auto target = cast<gpu::ReduceOp>(raw);
      SmallVector<Type> arguments(results);
      arguments.append(results);
      for (Value capture : llvm::drop_begin(
               inputs, reduce.getSourceCount() + reduce.getIdentityCount()))
        arguments.push_back(capture.getType());
      if (failed(lowerPureRegion(reduce.getCombine(), target.getCombine(),
                                 arguments, results)))
        return failure();
      mapResults(operation, raw);
      return success();
    }
    if (auto scan = dyn_cast<intent::ScanOp>(operation)) {
      SmallVector<Value> inputs;
      for (Value input : scan.getInputs()) {
        FailureOr<Value> lowered = get(input);
        if (failed(lowered))
          return scan.emitOpError("scan physical operand is unavailable");
        inputs.push_back(*lowered);
      }
      SmallVector<Type> results;
      for (auto [index, logical] : llvm::enumerate(scan.getResultTypes())) {
        if (index >= scan.getSourceCount())
          return scan.emitOpError(
              "scan result has no corresponding physical source");
        std::optional<Type> prototype = inputs[index].getType();
        FailureOr<Type> converted =
            convertDataType(logical, operation, prototype, /*owner=*/1, index);
        if (failed(converted))
          return scan.emitOpError("scan result has no physical schema");
        results.push_back(*converted);
      }
      for (unsigned index = 0; index < scan.getIdentityCount(); ++index) {
        unsigned operand = scan.getSourceCount() + index;
        FailureOr<Value> identity = projectAccumulatorIdentity(
            builder, location, inputs[operand], results[index]);
        if (failed(identity))
          return scan.emitOpError(
              "scan identity cannot adopt its physical accumulator schema");
        inputs[operand] = *identity;
      }
      OperationState state(location, gpu::ScanOp::getOperationName());
      state.addOperands(inputs);
      state.addTypes(results);
      state.addAttribute("axis", builder.getI64IntegerAttr(scan.getAxis()));
      state.addAttribute("inclusive", builder.getBoolAttr(scan.getInclusive()));
      state.addAttribute("reverse", builder.getBoolAttr(scan.getReverse()));
      state.addAttribute("source_count",
                         builder.getI64IntegerAttr(scan.getSourceCount()));
      state.addAttribute("identity_count",
                         builder.getI64IntegerAttr(scan.getIdentityCount()));
      state.addAttribute("capture_count",
                         builder.getI64IntegerAttr(scan.getCaptureCount()));
      state.addRegion();
      Operation *raw = builder.create(state);
      auto target = cast<gpu::ScanOp>(raw);
      SmallVector<Type> arguments(results);
      arguments.append(results);
      for (Value capture : llvm::drop_begin(
               inputs, scan.getSourceCount() + scan.getIdentityCount()))
        arguments.push_back(capture.getType());
      if (failed(lowerPureRegion(scan.getCombine(), target.getCombine(),
                                 arguments, results)))
        return failure();
      mapResults(operation, raw);
      return success();
    }
    if (auto fold = dyn_cast<intent::RegionFoldOp>(operation)) {
      SmallVector<Value> inputs;
      for (Value input : fold.getInputs()) {
        FailureOr<Value> lowered = get(input);
        if (failed(lowered))
          return fold.emitOpError("region-fold physical operand is unavailable");
        inputs.push_back(*lowered);
      }
      SmallVector<Type> results;
      for (auto [index, logical] : llvm::enumerate(fold.getResultTypes())) {
        std::optional<Type> prototype =
            inputs[fold.getSourceCount() + index].getType();
        FailureOr<Type> converted =
            convertDataType(logical, operation, prototype, /*owner=*/1, index);
        if (failed(converted))
          return fold.emitOpError("region-fold result has no physical schema");
        results.push_back(*converted);
      }
      for (unsigned index = 0; index < fold.getIdentityCount(); ++index) {
        unsigned operand = fold.getSourceCount() + index;
        FailureOr<Value> identity = projectAccumulatorIdentity(
            builder, location, inputs[operand], results[index]);
        if (failed(identity))
          return fold.emitOpError(
              "region-fold identity cannot adopt its physical summary schema");
        inputs[operand] = *identity;
      }
      Block &sourceSummary = fold.getSummarize().front();
      int64_t sourceDimension = 0;
      for (unsigned index = 0; index < fold.getSourceCount(); ++index) {
        auto tensor = dyn_cast<RankedTensorType>(fold.getInputs()[index].getType());
        DenseI64ArrayAttr ids = tensor ? dimensionIds(tensor)
                                       : DenseI64ArrayAttr();
        if (!tensor || fold.getAxis() >= static_cast<uint64_t>(tensor.getRank()) ||
            !ids || ids[fold.getAxis()] <= 0 ||
            (sourceDimension != 0 && ids[fold.getAxis()] != sourceDimension))
          return fold.emitOpError(
              "region-fold sources do not share one canonical slice dimension");
        sourceDimension = ids[fold.getAxis()];
      }
      auto node = operation->getAttrOfType<IntegerAttr>("intent.node");
      if (!node || node.getInt() < 0)
        return fold.emitOpError(
            "region-fold has no stable identity for its segment decision");
      std::string name =
          ("SEGMENT_N" + Twine(node.getInt()) + "_D" +
           Twine(sourceDimension))
              .str();
      auto segment = gpu::ParameterAttr::get(
          operation->getContext(), builder.getStringAttr(name),
          static_cast<uint32_t>(gpu::ParameterRole::ScanChunk),
          builder.getDenseI64ArrayAttr(
              {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
               32768, 65536}));
      gpu::ParameterOp declaration;
      operation->getParentOfType<ModuleOp>().walk([&](gpu::ParameterOp parameter) {
            if (declaration ||
                parameter.getParameter().getName() != segment.getName())
              return;
            parameter->setAttr("parameter", segment);
            declaration = parameter;
          });
      if (!declaration)
        declaration = builder.create<gpu::ParameterOp>(
            location, builder.getIndexType(), segment);
      parameters[segment.getName()] = declaration.getResult();
      OperationState state(location, gpu::RegionFoldOp::getOperationName());
      state.addOperands(inputs);
      state.addTypes(results);
      state.addAttribute("axis", builder.getI64IntegerAttr(fold.getAxis()));
      state.addAttribute("source_count",
                         builder.getI64IntegerAttr(fold.getSourceCount()));
      state.addAttribute("identity_count",
                         builder.getI64IntegerAttr(fold.getIdentityCount()));
      state.addAttribute("capture_count",
                         builder.getI64IntegerAttr(fold.getCaptureCount()));
      state.addAttribute("segment", segment);
      state.addRegion();
      state.addRegion();
      Operation *raw = builder.create(state);
      auto target = cast<gpu::RegionFoldOp>(raw);
      SmallVector<Type> summarizeArguments;
      for (unsigned index = 0; index < fold.getSourceCount(); ++index) {
        auto tensor = dyn_cast<RankedTensorType>(
            sourceSummary.getArgument(index).getType());
        auto prototype = dyn_cast<FragmentType>(inputs[index].getType());
        FailureOr<FragmentType> converted =
            tensor && prototype
                ? convertSegmentSliceType(tensor, prototype, fold.getAxis(), name)
                : FailureOr<FragmentType>(failure());
        if (failed(converted))
          return fold.emitOpError(
              "region-fold source slice has no physical fragment schema");
        summarizeArguments.push_back(*converted);
      }
      for (Value capture : llvm::drop_begin(
               inputs, fold.getSourceCount() + fold.getIdentityCount()))
        summarizeArguments.push_back(capture.getType());
      if (failed(lowerPureRegion(fold.getSummarize(), target.getSummarize(),
                                 summarizeArguments, results)))
        return failure();
      SmallVector<Type> combineArguments(results);
      combineArguments.append(results);
      if (failed(lowerPureRegion(fold.getCombine(), target.getCombine(),
                                 combineArguments, results)))
        return failure();
      mapResults(operation, raw);
      return success();
    }
    if (auto scan = dyn_cast<intent::RegionScanOp>(operation)) {
      SmallVector<Value> inputs;
      for (Value input : scan.getInputs()) {
        FailureOr<Value> lowered = get(input);
        if (failed(lowered))
          return scan.emitOpError("region-scan physical operand is unavailable");
        inputs.push_back(*lowered);
      }
      SmallVector<Type> results;
      for (auto [index, logical] : llvm::enumerate(scan.getResultTypes())) {
        std::optional<Type> prototype;
        if (index >= scan.getOutputCount()) {
          unsigned stateIndex = index - scan.getOutputCount();
          unsigned stateOffset =
              scan.getSourceCount() + scan.getIdentityCount();
          if (stateIndex >= scan.getStateCount() ||
              stateOffset + stateIndex >= inputs.size())
            return scan.emitOpError(
                "region-scan final-state result has no matching state operand");
          prototype = inputs[stateOffset + stateIndex].getType();
        }
        FailureOr<Type> converted =
            convertDataType(logical, operation, prototype, /*owner=*/1,
                            index);
        if (failed(converted))
          return scan.emitOpError("region-scan result has no physical schema");
        results.push_back(*converted);
      }
      Block &sourceSummary = scan.getSummarize().front();
      int64_t sourceDimension = 0;
      for (unsigned index = 0; index < scan.getSourceCount(); ++index) {
        auto tensor = dyn_cast<RankedTensorType>(scan.getInputs()[index].getType());
        DenseI64ArrayAttr ids = tensor ? dimensionIds(tensor)
                                       : DenseI64ArrayAttr();
        if (!tensor || scan.getAxis() >= static_cast<uint64_t>(tensor.getRank()) ||
            !ids || ids[scan.getAxis()] <= 0 ||
            (sourceDimension != 0 && ids[scan.getAxis()] != sourceDimension))
          return scan.emitOpError(
              "region-scan sources do not share one canonical slice dimension");
        sourceDimension = ids[scan.getAxis()];
      }
      auto node = operation->getAttrOfType<IntegerAttr>("intent.node");
      if (!node || node.getInt() < 0)
        return scan.emitOpError(
            "region-scan has no stable identity for its segment decision");
      std::string name =
          ("SEGMENT_N" + Twine(node.getInt()) + "_D" +
           Twine(sourceDimension))
              .str();
      auto segment = gpu::ParameterAttr::get(
          operation->getContext(), builder.getStringAttr(name),
          static_cast<uint32_t>(gpu::ParameterRole::ScanChunk),
          builder.getDenseI64ArrayAttr(
              {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
               32768, 65536}));
      gpu::ParameterOp declaration;
      operation->getParentOfType<ModuleOp>().walk([&](gpu::ParameterOp parameter) {
            if (declaration ||
                parameter.getParameter().getName() != segment.getName())
              return;
            parameter->setAttr("parameter", segment);
            declaration = parameter;
          });
      if (!declaration)
        declaration = builder.create<gpu::ParameterOp>(
            location, builder.getIndexType(), segment);
      parameters[segment.getName()] = declaration.getResult();
      OperationState state(location, gpu::RegionScanOp::getOperationName());
      state.addOperands(inputs);
      state.addTypes(results);
      state.addAttribute("axis", builder.getI64IntegerAttr(scan.getAxis()));
      state.addAttribute("source_count",
                         builder.getI64IntegerAttr(scan.getSourceCount()));
      state.addAttribute("identity_count",
                         builder.getI64IntegerAttr(scan.getIdentityCount()));
      state.addAttribute("state_count",
                         builder.getI64IntegerAttr(scan.getStateCount()));
      state.addAttribute("capture_count",
                         builder.getI64IntegerAttr(scan.getCaptureCount()));
      state.addAttribute("output_count",
                         builder.getI64IntegerAttr(scan.getOutputCount()));
      state.addAttribute("segment", segment);
      for (unsigned region = 0; region < 4; ++region)
        state.addRegion();
      Operation *raw = builder.create(state);
      auto target = cast<gpu::RegionScanOp>(raw);
      SmallVector<Type> sliceTypes;
      for (unsigned index = 0; index < scan.getSourceCount(); ++index) {
        auto tensor = dyn_cast<RankedTensorType>(
            sourceSummary.getArgument(index).getType());
        auto prototype = dyn_cast<FragmentType>(inputs[index].getType());
        FailureOr<FragmentType> converted =
            tensor && prototype
                ? convertSegmentSliceType(tensor, prototype, scan.getAxis(), name)
                : FailureOr<FragmentType>(failure());
        if (failed(converted))
          return scan.emitOpError(
              "region-scan source slice has no physical fragment schema");
        sliceTypes.push_back(*converted);
      }
      SmallVector<Type> transitionTypes;
      for (unsigned index = 0; index < scan.getIdentityCount(); ++index)
        transitionTypes.push_back(
            inputs[scan.getSourceCount() + index].getType());
      SmallVector<Type> stateTypes;
      unsigned stateOffset = scan.getSourceCount() + scan.getIdentityCount();
      for (unsigned index = 0; index < scan.getStateCount(); ++index)
        stateTypes.push_back(inputs[stateOffset + index].getType());
      SmallVector<Type> captureTypes;
      unsigned captureOffset = stateOffset + scan.getStateCount();
      for (Value capture : llvm::drop_begin(inputs, captureOffset))
        captureTypes.push_back(capture.getType());
      SmallVector<Type> summarizeArguments(sliceTypes);
      summarizeArguments.append(captureTypes);
      if (failed(lowerPureRegion(scan.getSummarize(), target.getSummarize(),
                                 summarizeArguments, transitionTypes)))
        return failure();
      SmallVector<Type> combineArguments(transitionTypes);
      combineArguments.append(transitionTypes);
      if (failed(lowerPureRegion(scan.getCombine(), target.getCombine(),
                                 combineArguments, transitionTypes)))
        return failure();
      SmallVector<Type> applyArguments(transitionTypes);
      applyArguments.append(stateTypes);
      if (failed(lowerPureRegion(scan.getApply(), target.getApply(),
                                 applyArguments, stateTypes)))
        return failure();
      SmallVector<Type> emitArguments(sliceTypes);
      emitArguments.append(stateTypes);
      emitArguments.append(captureTypes);
      SmallVector<Type> emittedTypes(
          results.begin(), results.begin() + scan.getOutputCount());
      if (failed(lowerPureRegion(scan.getEmit(), target.getEmit(),
                                 emitArguments, emittedTypes)))
        return failure();
      mapResults(operation, raw);
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
      FailureOr<FragmentType> valueType = convertTensorType(tensor, operation);
      if (failed(valueType))
        return buffer.emitOpError("logical buffer shape is not physicalizable");
      FailureOr<PhysicalAxisIdentity> instance = resultAxisIdentity(operation);
      if (failed(instance))
        return buffer.emitOpError(
            "logical buffer has no canonical physical instance identity");
      auto physicalType = gpu::BufferType::get(
          operation->getContext(), tensor.getElementType(), valueType->getShape(),
          /*scope=*/orderedDepth == 0 ? 0 : 1, instance->sourceId,
          /*owner=*/1, buffer.getInitialOperand() ? 0 : 1,
          /*lifetime=*/orderedDepth == 0 ? 0 : 1,
          /*visibility=*/0, /*workspace=*/false);
      Value initial;
      if (buffer.getInitialOperand()) {
        FailureOr<Value> lowered =
            get(buffer.getInputs()[*buffer.getInitialOperand()]);
        if (failed(lowered))
          return buffer.emitOpError("logical buffer initializer is unavailable");
        initial = *lowered;
      }
      auto target =
          builder.create<gpu::BufferOp>(location, physicalType, initial);
      mapResults(operation, target);
      return success();
    }
    if (auto gather = dyn_cast<intent::GatherOp>(operation)) {
      FailureOr<Value> source = get(gather.getInputs().front());
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
      if (gather.getValidOperandIndex()) {
        FailureOr<Value> lowered =
            get(gather.getInputs()[*gather.getValidOperandIndex()]);
        if (failed(lowered))
          return failure();
        valid = *lowered;
      }
      if (gather.getFillOperandIndex()) {
        FailureOr<Value> lowered =
            get(gather.getInputs()[*gather.getFillOperandIndex()]);
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
      auto target = builder.create<gpu::GatherOp>(
          location, *result, *source, *coordinates, valid, fill, *axes);
      mapResults(operation, target);
      return success();
    }
    if (auto load = dyn_cast<intent::ViewLoadOp>(operation)) {
      FailureOr<Value> resource = get(load.getInputs().front());
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
      if (load.getValidOperandIndex()) {
        FailureOr<Value> lowered =
            get(load.getInputs()[*load.getValidOperandIndex()]);
        if (failed(lowered))
          return failure();
        valid = *lowered;
      }
      if (load.getFillOperandIndex()) {
        FailureOr<Value> lowered =
            get(load.getInputs()[*load.getFillOperandIndex()]);
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
      auto target = builder.create<gpu::LoadOp>(
          location, *result, *resource, *coordinates, valid,
          fill, *axes);
      mapResults(operation, target);
      return success();
    }
    if (auto load = dyn_cast<intent::BufferLoadOp>(operation)) {
      FailureOr<Value> resource = get(load.getInputs().front());
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
      auto target = builder.create<gpu::LoadOp>(
          location, *result, *resource, *coordinates, Value(), Value(), *axes);
      mapResults(operation, target);
      return success();
    }
    if (auto store = dyn_cast<intent::ViewStoreOp>(operation)) {
      FailureOr<Value> resource = get(store.getInputs().front());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            store.getInputs()[store.getValueOperandIndex()],
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
      auto target = builder.create<gpu::StoreOp>(
          location, *resource, *coordinates, *value, Value(), *axes);
      if (Attribute node = operation->getAttr("intent.node"))
        target->setAttr(gpu::originAttr, node);
      return success();
    }
    if (auto store = dyn_cast<intent::BufferStoreOp>(operation)) {
      FailureOr<Value> resource = get(store.getInputs().front());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            store.getInputs()[store.getValueOperandIndex()],
                            *coordinates);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value))
        return store.emitOpError("buffer store physical relation is unavailable");
      auto target = builder.create<gpu::StoreOp>(
          location, *resource, *coordinates, *value, Value(), *axes);
      attachOrigin(operation, target);
      return success();
    }
    if (auto store = dyn_cast<intent::ScatterUniqueOp>(operation)) {
      FailureOr<Value> resource = get(store.getInputs().front());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            store.getInputs()[store.getValueOperandIndex()],
                            *coordinates);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value))
        return store.emitOpError("unique scatter is not a scalar physical access");
      auto target = builder.create<gpu::StoreOp>(
          location, *resource, *coordinates, *value, Value(), *axes);
      if (Attribute node = operation->getAttr("intent.node"))
        target->setAttr(gpu::originAttr, node);
      return success();
    }
    if (auto scatter = dyn_cast<intent::ScatterReduceOp>(operation)) {
      FailureOr<Value> resource = get(scatter.getInputs().front());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            scatter.getInputs()[scatter.getValueOperandIndex()],
                            *coordinates);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value))
        return scatter.emitOpError(
            "scatter-reduce physical relation is unavailable");
      OperationState state(location, gpu::ScatterReduceOp::getOperationName());
      state.addOperands(*resource);
      state.addOperands(*coordinates);
      state.addOperands(*value);
      state.addAttribute("source_axes", builder.getDenseI64ArrayAttr(*axes));
      state.addAttribute(
          "sharing", gpu::AtomicSharingDomainAttr::get(
                         operation->getContext(),
                         isa<gpu::ViewType>((*resource).getType())
                             ? gpu::AtomicSharingDomain::KernelInvocation
                             : gpu::AtomicSharingDomain::ProgramInstance));
      state.addAttribute("operandSegmentSizes",
                         builder.getDenseI32ArrayAttr(
                             {1, static_cast<int32_t>(coordinates->size()), 1, 0}));
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
      FailureOr<Value> resource = get(atomic.getInputs().front());
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
      auto target = builder.create<gpu::AtomicLoadOp>(
          location, *result, *resource, *coordinates, Value(),
          atomic.getOrdering(), sharing, *axes);
      mapResults(operation, target);
      return success();
    }
    if (auto atomic = dyn_cast<intent::AtomicStoreOp>(operation)) {
      FailureOr<Value> resource = get(atomic.getInputs().front());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            atomic.getInputs()[atomic.getValueOperand()],
                            *coordinates);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value))
        return atomic.emitOpError("atomic store address/value is unavailable");
      gpu::AtomicSharingDomain sharing =
          isa<gpu::ViewType>((*resource).getType())
              ? gpu::AtomicSharingDomain::KernelInvocation
              : gpu::AtomicSharingDomain::ProgramInstance;
      auto target = builder.create<gpu::AtomicStoreOp>(
          location, *resource, *coordinates, *value, Value(),
          atomic.getOrdering(), sharing, *axes);
      if (Attribute node = operation->getAttr("intent.node"))
        target->setAttr(gpu::originAttr, node);
      return success();
    }
    if (auto atomic = dyn_cast<intent::AtomicRMWOp>(operation)) {
      FailureOr<Value> resource = get(atomic.getInputs().front());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            atomic.getInputs()[atomic.getValueOperand()],
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
      auto target = builder.create<gpu::AtomicRMWOp>(
          location, *result, *resource, *coordinates, *value, Value(),
          atomic.getKind(), atomic.getOrdering(), sharing, *axes);
      mapResults(operation, target);
      return success();
    }
    if (auto atomic = dyn_cast<intent::AtomicCompareExchangeOp>(operation)) {
      FailureOr<Value> resource = get(atomic.getInputs().front());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> expected =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            atomic.getInputs()[atomic.getExpectedOperand()],
                            *coordinates);
      FailureOr<Value> desired =
          failed(coordinates)
              ? FailureOr<Value>(failure())
              : accessValue(operation,
                            atomic.getInputs()[atomic.getDesiredOperand()],
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
      auto target = builder.create<gpu::AtomicCompareExchangeOp>(
          location, *result, *resource, *coordinates, *expected, *desired,
          Value(), atomic.getOrdering(), sharing, *axes);
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
                                   orderedDepth);
        FailureOr<SmallVector<Value>> yielded =
            child.lowerBlock(sourceRegion.front());
        if (failed(yielded) || yielded->size() != resultTypes.size())
          return failure();
        for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
          FailureOr<Value> projected = projectAccumulatorIdentity(
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
    if (auto forOperation = dyn_cast<intent::ForOp>(operation)) {
      if (forOperation.getInputs().empty())
        return forOperation.emitOpError("ordered for lacks its logical domain");
      SmallVector<OrderedIterationAxis> axes;
      if (failed(collectOrderedIterationAxes(forOperation.getInputs().front(),
                                             axes)) ||
          axes.empty())
        return forOperation.emitOpError(
            "ordered for source has no exact domain/subregion iteration relation");
      SmallVector<Value> lowers, uppers, steps;
      for (OrderedIterationAxis axis : axes) {
        FailureOr<Value> lower = get(axis.start);
        FailureOr<Value> upper = get(axis.stop);
        FailureOr<Value> prototype = get(axis.coordinatePrototype);
        if (failed(lower) || failed(upper) || failed(prototype))
          return forOperation.emitOpError(
              "ordered physical loop bounds are unavailable");
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
          return forOperation.emitOpError(
              "ordered physical subregion bounds cannot adopt their source coordinate type");
        Value step;
        if (axis.step) {
          FailureOr<Value> lowered = get(axis.step);
          if (failed(lowered))
            return forOperation.emitOpError(
                "ordered physical loop step is unavailable");
          FailureOr<Value> aligned = alignBound(*lowered);
          if (failed(aligned))
            return forOperation.emitOpError(
                "ordered physical loop step cannot adopt its source coordinate type");
          step = *aligned;
        } else if ((*lower).getType().isIndex()) {
          step = builder.create<arith::ConstantIndexOp>(location, 1);
        } else if (auto integer = dyn_cast<IntegerType>((*lower).getType())) {
          step = builder.create<arith::ConstantOp>(
              location, integer, builder.getIntegerAttr(integer, 1));
        }
        if (!step || (*lower).getType() != (*upper).getType() ||
            (*lower).getType() != step.getType())
          return forOperation.emitOpError(
              "ordered physical loop bounds must share one scalar type");
        lowers.push_back(*lower);
        uppers.push_back(*upper);
        steps.push_back(step);
      }
      SmallVector<Value> initial;
      for (Value input : forOperation.getInputs().drop_front()) {
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
          Block &source = forOperation.getBody().front();
          for (auto [argument, coordinate] :
               llvm::zip(source.getArguments().take_front(axes.size()),
                         coordinates))
            childValues[argument] = coordinate;
          for (auto [argument, carry] : llvm::zip(
                   source.getArguments().drop_front(axes.size()), carries))
            childValues[argument] = carry;
          ScalarRegionLowering child(nested, std::move(childValues), views,
                                     dimensions, parameters, orderedDepth + 1);
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
            [&](OpBuilder &bodyBuilder, Location, Value induction,
                ValueRange innerCarries) {
              SmallVector<Value> nextCoordinates(coordinates);
              nextCoordinates.push_back(induction);
              SmallVector<Value> yielded =
                  lowerAxis(bodyBuilder, axis + 1, std::move(nextCoordinates),
                            innerCarries);
              if (!nestedFailed)
                bodyBuilder.create<scf::YieldOp>(location, yielded);
            });
        auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
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
          FailureOr<Value> aligned = projectAccumulatorIdentity(
              before, location, yielded, initial.getType());
          if (failed(aligned)) {
            nestedFailed = true;
            break;
          }
          yield->setOperand(index, *aligned);
        }
        loopResults.append(loop.getResults().begin(), loop.getResults().end());
        return loopResults;
      };
      SmallVector<Value> resultValues =
          lowerAxis(builder, 0, SmallVector<Value>{}, initial);
      if (nestedFailed)
        return failure();
      for (auto [source, target] :
           llvm::zip(forOperation.getResults(), resultValues))
        values[source] = target;
      return success();
    }
    if (auto whileOperation = dyn_cast<intent::WhileOp>(operation)) {
      SmallVector<Value> initial;
      SmallVector<Type> resultTypes;
      for (Value input : whileOperation.getInputs()) {
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
                 whileOperation.getBefore().front().getArguments(),
                 before->getArguments()))
          childValues[source] = targetArgument;
        builder.setInsertionPointToStart(before);
        ScalarRegionLowering child(builder, std::move(childValues), views,
                                   dimensions, parameters, orderedDepth + 1);
        Block &source = whileOperation.getBefore().front();
        for (Operation &nested : source.without_terminator())
          if (failed(child.lower(&nested)))
            return failure();
        auto condition = cast<intent::ConditionOp>(source.getTerminator());
        FailureOr<Value> predicate = child.get(condition.getInputs().front());
        SmallVector<Value> forwarded;
        for (Value value : condition.getInputs().drop_front()) {
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
                 whileOperation.getAfter().front().getArguments(),
                 after->getArguments()))
          childValues[source] = targetArgument;
        builder.setInsertionPointToStart(after);
        ScalarRegionLowering child(builder, std::move(childValues), views,
                                   dimensions, parameters, orderedDepth + 1);
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
      auto target = builder.create<gpu::AssumeInBoundsOp>(
          location, *index, *resource, assumption.getAxis());
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
  unsigned orderedDepth;
};

struct ParallelWorkset {
  intent::ParallelOp operation;
  Block *body = nullptr;
  bool singleton = false;
  SmallVector<intent::DomainOp> axes;
  SmallVector<BlockArgument> coordinateArguments;
  SmallVector<PhysicalExprAttr> launchExtents;
  PhysicalExprAttr launchLength;
};

LogicalResult collectDomainAxes(Value source,
                                SmallVectorImpl<intent::DomainOp> &axes) {
  if (auto domain = source.getDefiningOp<intent::DomainOp>()) {
    axes.push_back(domain);
    return success();
  }
  auto product = source.getDefiningOp<intent::DomainProductOp>();
  if (!product) {
    emitError(source.getLoc())
        << "parallel GPU workset must have a launch-visible domain product";
    return failure();
  }
  for (Value component : product.getDomains())
    if (failed(collectDomainAxes(component, axes)))
      return failure();
  return success();
}

LogicalResult collectOrderedIterationAxes(
    Value source, SmallVectorImpl<OrderedIterationAxis> &axes) {
  if (auto domain = source.getDefiningOp<intent::DomainOp>()) {
    axes.push_back({domain.getBounds()[0], domain.getBounds()[1],
                    domain.getBounds().size() == 3 ? domain.getBounds()[2]
                                                   : Value(),
                    domain.getBounds()[0]});
    return success();
  }
  if (auto product = source.getDefiningOp<intent::DomainProductOp>()) {
    for (Value component : product.getDomains())
      if (failed(collectOrderedIterationAxes(component, axes)))
        return failure();
    return success();
  }
  auto subregion = source.getDefiningOp<intent::SubregionOp>();
  if (!subregion || subregion.getInputs().empty())
    return failure();
  SmallVector<OrderedIterationAxis> parent;
  if (failed(collectOrderedIterationAxes(subregion.getInputs().front(), parent)) ||
      parent.size() != 1)
    return failure();
  unsigned operand = 1;
  if (subregion.getHasStart())
    parent.front().start = subregion.getInputs()[operand++];
  if (subregion.getHasStop())
    parent.front().stop = subregion.getInputs()[operand++];
  if (operand != subregion.getInputs().size())
    return failure();
  axes.append(parent.begin(), parent.end());
  return success();
}

LogicalResult finalizeParallelWorkset(ParallelWorkset &workset,
                                      func::FuncOp function) {
  MLIRContext *context = workset.operation.getContext();
  for (intent::DomainOp domain : workset.axes) {
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
      return domain.emitOpError(
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

LogicalResult collectParallelWorksets(
    intent::ParallelOp operation, func::FuncOp function,
    ArrayRef<intent::DomainOp> parentAxes,
    ArrayRef<BlockArgument> parentArguments,
    SmallVectorImpl<ParallelWorkset> &worksets) {
  ParallelWorkset current;
  current.operation = operation;
  current.axes.append(parentAxes.begin(), parentAxes.end());
  current.coordinateArguments.append(parentArguments.begin(),
                                     parentArguments.end());
  SmallVector<intent::DomainOp> localAxes;
  if (failed(collectDomainAxes(operation.getSource(), localAxes)) ||
      localAxes.empty())
    return failure();
  current.axes.append(localAxes.begin(), localAxes.end());
  Block &body = operation.getBody().front();
  if (body.getNumArguments() != localAxes.size())
    return operation.emitOpError(
        "parallel body coordinates do not match its domain product");
  current.coordinateArguments.append(body.getArguments().begin(),
                                     body.getArguments().end());

  SmallVector<intent::ParallelOp> children;
  bool hasDirectEffect = false;
  auto classifyWorksetOperation = [&](Operation *candidate) {
    hasDirectEffect |= isCanonicalEffect(candidate);
  };
  for (Operation &nested : body.without_terminator()) {
    if (auto child = dyn_cast<intent::ParallelOp>(nested)) {
      children.push_back(child);
      continue;
    }
    classifyWorksetOperation(&nested);
    nested.walk([&](Operation *candidate) {
      if (candidate != &nested)
        classifyWorksetOperation(candidate);
    });
  }
  if (!children.empty()) {
    if (hasDirectEffect)
      return operation.emitOpError(
          "one parallel region cannot mix direct effects with nested independent worksets");
    for (intent::ParallelOp child : children)
      if (failed(collectParallelWorksets(child, function, current.axes,
                                         current.coordinateArguments, worksets)))
        return failure();
    return success();
  }

  current.body = &body;
  if (failed(finalizeParallelWorkset(current, function)))
    return failure();
  worksets.push_back(std::move(current));
  return success();
}

LogicalResult constructGPUProgram(ModuleOp module,
                                  const GPUCapabilities &capabilities,
                                  func::FuncOp function) {
  SmallVector<ParallelWorkset> worksets;
  for (Operation &operation : function.getBody().front())
    if (auto parallel = dyn_cast<intent::ParallelOp>(operation)) {
      if (failed(collectParallelWorksets(parallel, function, {}, {}, worksets)))
        return failure();
    }
  if (worksets.empty()) {
    ParallelWorkset singleton;
    singleton.body = &function.getBody().front();
    singleton.singleton = true;
    singleton.launchExtents.push_back(
        expression(function.getContext(), PhysicalExprKind::Constant, 1));
    singleton.launchLength = singleton.launchExtents.front();
    worksets.push_back(std::move(singleton));
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
      capabilities.registersPerUnit, capabilities.matrixUnits,
      capabilities.dynamicVectorWidth);
  SmallVector<NamedAttribute> functionAttrs{
      builder.getNamedAttr(gpu::kernelAttr, builder.getUnitAttr()),
      builder.getNamedAttr(gpu::capabilitiesAttr, capabilityAttr),
      builder.getNamedAttr(gpu::programSpaceAttr,
                           builder.getArrayAttr({totalLength})),
      builder.getNamedAttr(gpu::gridRankAttr, builder.getI64IntegerAttr(1)),
      builder.getNamedAttr(gpu::effectOriginsAttr,
                           observableEffectOrigins(function)),
      builder.getNamedAttr(
          gpu::originAttr,
          function->getAttr("intent.source")
              ? function->getAttr("intent.source")
              : builder.getStringAttr(function.getName())),
  };
  builder.setInsertionPointAfter(function);
  auto physical = builder.create<func::FuncOp>(
      function.getLoc(), ("__intent_gpu_" + function.getName()).str(),
      FunctionType::get(context, abi->arguments, {}), functionAttrs,
      abi->argumentAttrs);
  Block *entry = physical.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  llvm::SmallDenseSet<int64_t> launchDimensions(abi->dimensionOrder.begin(),
                                                abi->dimensionOrder.end());
  llvm::SmallDenseSet<int64_t> derivedFragmentDimensions;
  function.walk([&](Operation *operation) {
    auto collect = [&](Type type) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      DenseI64ArrayAttr identities = tensor ? dimensionIds(tensor)
                                            : DenseI64ArrayAttr();
      if (!tensor || !identities)
        return;
      for (auto [axis, identity] : llvm::enumerate(identities.asArrayRef())) {
        if (!tensor.isDynamicDim(axis) || identity <= 0)
          continue;
        FailureOr<int64_t> physicalIdentity =
            physicalDimensionIdentity(operation, identity);
        if (succeeded(physicalIdentity) &&
            !launchDimensions.contains(*physicalIdentity))
          derivedFragmentDimensions.insert(*physicalIdentity);
      }
    };
    for (Type type : operation->getOperandTypes())
      collect(type);
    for (Type type : operation->getResultTypes())
      collect(type);
  });
  llvm::DenseMap<StringAttr, Value> parameterValues;
  SmallVector<int64_t> orderedDerived(derivedFragmentDimensions.begin(),
                                      derivedFragmentDimensions.end());
  llvm::sort(orderedDerived);
  for (int64_t dimension : orderedDerived) {
    std::string name = ("FRAGMENT_D" + Twine(dimension)).str();
    auto parameter = gpu::ParameterAttr::get(
        context, builder.getStringAttr(name),
        static_cast<uint32_t>(gpu::ParameterRole::OwnershipN),
        builder.getDenseI64ArrayAttr(
            {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}));
    auto declaration = builder.create<gpu::ParameterOp>(
        function.getLoc(), builder.getIndexType(), parameter);
    declaration->setAttr(gpu::dimensionAttr,
                         builder.getI64IntegerAttr(dimension));
    parameterValues[parameter.getName()] = declaration.getResult();
  }
  SmallVector<Value> sourceArguments;
  sourceArguments.reserve(abi->physicalArgumentForSource.size());
  for (unsigned physicalIndex : abi->physicalArgumentForSource)
    sourceArguments.push_back(entry->getArgument(physicalIndex));
  llvm::DenseMap<int64_t, Value> dimensionValues;
  unsigned metadataOffset = abi->physicalArgumentForSource.size();
  for (auto [offset, dimension] : llvm::enumerate(abi->dimensionOrder))
    dimensionValues[dimension] = entry->getArgument(metadataOffset + offset);

  llvm::DenseMap<Value, Value> values;
  for (auto [logical, physicalView] : llvm::zip(
           function.getBody().front().getArguments(), sourceArguments))
    values[logical] = physicalView;
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
                                    dimensionValues, parameterValues);
  auto formWorksetCoordinate = [&](OpBuilder &nested, Location location,
                                   intent::DomainOp domain, Value coordinate,
                                   unsigned worksetAxis) -> Value {
    auto domainType = cast<intent::DomainType>(domain.getResult().getType());
    std::optional<int64_t> dimension =
        sourceExtentDimension(domain.getResult());
    if (!dimension || *dimension <= 0) {
      domain.emitOpError(
          "parallel workset coordinate has no logical dimension identity");
      return {};
    }
    auto mapped = nested.create<gpu::WorksetCoordinateOp>(
        location, nested.getIndexType(), coordinate, domainType.getOriginId(),
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
    for (intent::DomainOp domain : workset.axes) {
      FailureOr<Value> start = rootLowering.lowerValue(domain.getBounds()[0]);
      FailureOr<Value> stop = rootLowering.lowerValue(domain.getBounds()[1]);
      FailureOr<Value> step =
          domain.getBounds().size() == 3
              ? rootLowering.lowerValue(domain.getBounds()[2])
              : FailureOr<Value>(one);
      if (failed(start) || failed(stop) || failed(step))
        return domain.emitOpError(
            "parallel workset runtime bounds are unavailable");
      Value distance = createBinary(builder, domain.getLoc(),
                                    builder.getIndexType(), *stop, *start,
                                    BinaryOperator::Subtract);
      Value adjusted = createBinary(
          builder, domain.getLoc(), builder.getIndexType(), distance,
          createBinary(builder, domain.getLoc(), builder.getIndexType(), *step,
                       one, BinaryOperator::Subtract),
          BinaryOperator::Add);
      Value extent = createBinary(builder, domain.getLoc(),
                                  builder.getIndexType(), adjusted, *step,
                                  BinaryOperator::FloorDivide);
      runtimeExtents.push_back(extent);
      starts.push_back(*start);
      steps.push_back(*step);
      runtimeLength = createBinary(builder, domain.getLoc(),
                                   builder.getIndexType(), runtimeLength, extent,
                                   BinaryOperator::Multiply);
    }
    if (workset.singleton)
      runtimeExtents.push_back(one);
    Value segmentEnd = createBinary(builder, function.getLoc(),
                                    builder.getIndexType(), runtimeOffset,
                                    runtimeLength, BinaryOperator::Add);
    if (worksets.size() == 1) {
      SmallVector<Attribute> launchExtents(workset.launchExtents.begin(),
                                           workset.launchExtents.end());
      Location worksetLocation = workset.singleton ? function.getLoc()
                                                   : workset.operation.getLoc();
      auto decoded = builder.create<gpu::DelinearizeOp>(
          worksetLocation,
          SmallVector<Type>(runtimeExtents.size(), builder.getIndexType()), pid,
          runtimeExtents, builder.getArrayAttr(launchExtents));
      decoded->setAttr(gpu::executionGroupAttr,
                       builder.getI64IntegerAttr(groupIndex));
      decoded->setAttr(gpu::segmentOffsetAttr, launchOffset);
      decoded->setAttr(gpu::segmentLengthAttr, workset.launchLength);
      if (!workset.singleton)
        decoded->setAttr(
            gpu::coordinateRolesAttr,
            DenseI64ArrayAttr::get(
                context,
                SmallVector<int64_t>(
                    decoded.getNumResults(),
                    static_cast<int64_t>(gpu::CoordinateRole::Workset))));
      auto childValues = rootLowering.mapping();
      Block &sourceBlock = *workset.body;
      for (auto [axis, localCoordinate] :
           llvm::enumerate(decoded.getCoordinates())) {
        if (workset.singleton)
          break;
        Value scaled = createBinary(builder, worksetLocation,
                                    builder.getIndexType(), localCoordinate,
                                    steps[axis], BinaryOperator::Multiply);
        Value coordinate = createBinary(builder, worksetLocation,
                                        builder.getIndexType(), starts[axis],
                                        scaled, BinaryOperator::Add);
        childValues[workset.coordinateArguments[axis]] = formWorksetCoordinate(
            builder, worksetLocation, workset.axes[axis], coordinate,
            axis);
      }
      ScalarRegionLowering lowering(builder, std::move(childValues),
                                    sourceArguments, dimensionValues,
                                    parameterValues);
      if (failed(lowering.lowerBlock(sourceBlock)))
        return failure();
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
          SmallVector<Attribute> launchExtents(workset.launchExtents.begin(),
                                               workset.launchExtents.end());
          auto decoded = nested.create<gpu::DelinearizeOp>(
              location,
              SmallVector<Type>(runtimeExtents.size(), nested.getIndexType()),
              local, runtimeExtents,
              nested.getArrayAttr(launchExtents));
          if (!workset.singleton)
            decoded->setAttr(
                gpu::coordinateRolesAttr,
                DenseI64ArrayAttr::get(
                    context,
                    SmallVector<int64_t>(
                        decoded.getNumResults(),
                        static_cast<int64_t>(gpu::CoordinateRole::Workset))));
          auto childValues = rootLowering.mapping();
          Block &sourceBlock = *workset.body;
          for (auto [axis, localCoordinate] :
               llvm::enumerate(decoded.getCoordinates())) {
            if (workset.singleton)
              break;
            Value scaled = createBinary(nested, location, nested.getIndexType(),
                                        localCoordinate, steps[axis],
                                        BinaryOperator::Multiply);
            Value coordinate = createBinary(nested, location, nested.getIndexType(),
                                            starts[axis], scaled,
                                            BinaryOperator::Add);
            childValues[workset.coordinateArguments[axis]] =
                formWorksetCoordinate(nested, location, workset.axes[axis],
                                      coordinate, axis);
          }
          ScalarRegionLowering lowering(nested, std::move(childValues),
                                        sourceArguments, dimensionValues,
                                        parameterValues);
          if (failed(lowering.lowerBlock(sourceBlock)))
            dispatchLoweringFailed = true;
        });
    dispatch->setAttr(gpu::executionGroupAttr,
                      builder.getI64IntegerAttr(groupIndex));
    dispatch->setAttr(gpu::segmentOffsetAttr, launchOffset);
    dispatch->setAttr(gpu::segmentLengthAttr, workset.launchLength);
    runtimeOffset = segmentEnd;
    launchOffset = binaryExpression(context, PhysicalExprKind::Add, launchOffset,
                                    workset.launchLength);
  }
  if (dispatchLoweringFailed)
    return failure();
  builder.create<func::ReturnOp>(function.getLoc());
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
      capabilities.registersPerUnit <= 0)
    return module.emitError("selected GPU capabilities are incomplete");
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  if (functions.size() != 1)
    return module.emitError("GPU construction requires exactly one kernel entry");
  if (failed(constructGPUProgram(module, capabilities, functions.front())))
    return failure();
  return gpu::completeGPUProgramConstruction(module);
}

} // namespace intent
