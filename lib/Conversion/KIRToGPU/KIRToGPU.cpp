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

ArrayAttr observableEffectOrigins(func::FuncOp function) {
  SmallVector<Attribute> origins;
  function.walk([&](Operation *operation) {
    if (!isa<ViewStoreOp, ScatterUniqueOp, ScatterReduceOp, AtomicStoreOp,
             AtomicRMWOp, AtomicCompareExchangeOp>(operation))
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
  return expression(context, PhysicalExprKind::Dimension, 0,
                    ("D" + Twine(dimension)).str());
}

PhysicalExprAttr binaryExpression(MLIRContext *context, PhysicalExprKind kind,
                                  PhysicalExprAttr lhs,
                                  PhysicalExprAttr rhs) {
  return expression(context, kind, 0, {}, {lhs, rhs});
}

FragmentType fragmentType(MLIRContext *context, Type element,
                          ArrayRef<PhysicalExprAttr> shape,
                          ArrayRef<std::pair<uint64_t, uint32_t>> axes,
                          uint64_t owner = 1) {
  SmallVector<Attribute> extents(shape.begin(), shape.end());
  SmallVector<Attribute> mappings;
  for (auto [fragmentAxis, mapping] : llvm::enumerate(axes))
    mappings.push_back(AxisMapAttr::get(context, mapping.first, mapping.second,
                                        fragmentAxis));
  return FragmentType::get(context, element, ArrayAttr::get(context, extents),
                           ArrayAttr::get(context, mappings), 1, owner);
}

Value createBinary(OpBuilder &builder, Location location, Type result, Value lhs,
                   Value rhs, uint64_t kind) {
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

FailureOr<Value> projectAccumulatorIdentity(OpBuilder &builder, Location location,
                                            Value identity, Type accumulator) {
  if (identity.getType() == accumulator)
    return identity;
  auto target = dyn_cast<gpu::FragmentType>(accumulator);
  if (!target)
    return failure();
  if (identity.getType() == target.getElementType()) {
    auto splat = builder.create<gpu::SplatOp>(location, target, identity);
    if (Operation *definition = identity.getDefiningOp())
      if (Attribute origin = definition->getAttr(gpu::originAttr))
        splat->setAttr(gpu::originAttr, origin);
    return splat.getResult();
  }
  auto source = dyn_cast<gpu::FragmentType>(identity.getType());
  if (!source || source.getElementType() != target.getElementType())
    return failure();
  return retargetBroadcast(builder, location, identity, target);
}

LogicalResult alignElementwiseOperands(OpBuilder &builder, Location location,
                                       Value &lhs, Value &rhs) {
  auto left = dyn_cast<gpu::FragmentType>(lhs.getType());
  auto right = dyn_cast<gpu::FragmentType>(rhs.getType());
  if (!left || !right || samePhysicalShape(left, right))
    return success();
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
  FailureOr<Value> aligned = retargetBroadcast(builder, location, rhs, left);
  if (failed(aligned))
    return failure();
  rhs = *aligned;
  return success();
}

Value createCompare(OpBuilder &builder, Location location, Type result, Value lhs,
                    Value rhs, uint64_t predicate) {
  return builder.create<gpu::CompareOp>(location, result, lhs, rhs, predicate);
}

FailureOr<TypedAttr> physicalConstant(Attribute value, Type resultType) {
  auto typed = dyn_cast<TypedAttr>(value);
  if (!typed)
    return failure();
  if (typed.getType() == resultType)
    return typed;
  if (isa<IndexType, IntegerType>(resultType)) {
    auto integer = dyn_cast<IntegerAttr>(typed);
    if (!integer)
      return failure();
    return TypedAttr(IntegerAttr::get(resultType, integer.getInt()));
  }
  if (isa<FloatType>(resultType)) {
    auto floating = dyn_cast<FloatAttr>(typed);
    if (!floating)
      return failure();
    return TypedAttr(FloatAttr::get(resultType, floating.getValueAsDouble()));
  }
  return failure();
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
    case 0:
      kind = PhysicalExprKind::Add;
      break;
    case 1:
      kind = PhysicalExprKind::Subtract;
      break;
    case 2:
      kind = PhysicalExprKind::Multiply;
      break;
    case 4:
      kind = PhysicalExprKind::FloorDiv;
      break;
    case 7:
    case 9:
      kind = PhysicalExprKind::Maximum;
      break;
    case 8:
    case 10:
      kind = PhysicalExprKind::Minimum;
      break;
    default:
      return failure();
    }
    return binaryExpression(context, kind, *lhs, *rhs);
  }
  return failure();
}

FailureOr<PhysicalExprAttr> dimensionExtentExpression(RankedTensorType tensor,
                                                      Operation *origin,
                                                      unsigned axis) {
  DenseI64ArrayAttr identities = dimensionIds(tensor);
  if (!identities || axis >= identities.size() || identities[axis] <= 0)
    return failure();
  int64_t identity = identities[axis];
  func::FuncOp function = origin ? origin->getParentOfType<func::FuncOp>()
                                 : func::FuncOp();
  if (!function)
    return dimensionExpression(tensor.getContext(), identity);
  PhysicalExprAttr result;
  bool conflict = false;
  auto merge = [&](PhysicalExprAttr candidate) {
    if (result && result != candidate)
      conflict = true;
    else
      result = candidate;
  };
  function.walk([&](intent::DimOp dim) {
    if (dim.getDimension() != static_cast<uint64_t>(identity))
      return;
    FailureOr<PhysicalExprAttr> candidate =
        launchExpression(dim.getResult(), function);
    if (failed(candidate))
      return;
    merge(*candidate);
  });
  function.walk([&](intent::DomainOp domain) {
    bool ownsIdentity = llvm::any_of(
        domain.getExtentDimensions(), [&](Attribute attribute) {
          return cast<IntegerAttr>(attribute).getInt() == identity;
        });
    if (!ownsIdentity)
      return;
    FailureOr<PhysicalExprAttr> start =
        launchExpression(domain.getBounds()[0], function);
    FailureOr<PhysicalExprAttr> stop =
        launchExpression(domain.getBounds()[1], function);
    FailureOr<PhysicalExprAttr> step =
        domain.getBounds().size() == 3
            ? launchExpression(domain.getBounds()[2], function)
            : FailureOr<PhysicalExprAttr>(expression(
                  tensor.getContext(), PhysicalExprKind::Constant, 1));
    if (failed(start) || failed(stop) || failed(step))
      return;
    if (start->getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        start->getValue() == 0 &&
        step->getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        step->getValue() == 1) {
      merge(*stop);
      return;
    }
    PhysicalExprAttr distance = binaryExpression(
        tensor.getContext(), PhysicalExprKind::Subtract, *stop, *start);
    merge(binaryExpression(tensor.getContext(), PhysicalExprKind::CeilDiv,
                           distance, *step));
  });
  if (conflict)
    return failure();
  return result ? FailureOr<PhysicalExprAttr>(result)
                : FailureOr<PhysicalExprAttr>(
                      dimensionExpression(tensor.getContext(), identity));
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

FailureOr<uint64_t> resultAxisIdentity(Operation *operation,
                                       unsigned resultIndex = 0,
                                       unsigned axis = 0) {
  if (!operation || resultIndex >= operation->getNumResults())
    return failure();
  auto results = operation->getAttrOfType<ArrayAttr>("intent.result_nodes");
  if (!results || resultIndex >= results.size())
    return failure();
  auto value = dyn_cast<IntegerAttr>(results[resultIndex]);
  if (!value || value.getInt() < 0 ||
      static_cast<uint64_t>(value.getInt()) >= (uint64_t{1} << 47) ||
      axis >= 65535)
    return failure();
  return (uint64_t{1} << 63) |
         ((static_cast<uint64_t>(value.getInt()) + 1) << 16) | (axis + 1);
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
  SmallVector<std::pair<uint64_t, uint32_t>> mappings;
  for (unsigned axis = 0; axis < static_cast<unsigned>(tensor.getRank()); ++axis) {
    if (tensor.isDynamicDim(axis)) {
      if (!dimensions || dimensions[axis] <= 0)
        return failure();
      FailureOr<PhysicalExprAttr> extent =
          dimensionExtentExpression(tensor, origin, axis);
      if (failed(extent))
        return failure();
      shape.push_back(*extent);
      mappings.emplace_back(static_cast<uint64_t>(dimensions[axis]), 0);
    } else {
      shape.push_back(expression(context, PhysicalExprKind::Constant,
                                 tensor.getDimSize(axis)));
      FailureOr<uint64_t> identity =
          resultAxisIdentity(origin, resultIndex, axis);
      if (failed(identity))
        return failure();
      mappings.emplace_back(*identity, axis);
    }
  }
  if (prototype && prototype->getShape().size() == shape.size()) {
    bool sameExtents = true;
    for (auto [left, right] : llvm::zip(prototype->getShape(), shape))
      sameExtents &= left == right;
    if (sameExtents) {
      SmallVector<Attribute> extents(shape.begin(), shape.end());
      return FragmentType::get(context, tensor.getElementType(),
                               ArrayAttr::get(context, extents),
                               prototype->getAxisMaps(), prototype->getValidity(),
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
  shape[axis] = parameterExpression(tensor.getContext(), parameter);
  return FragmentType::get(tensor.getContext(), tensor.getElementType(),
                           ArrayAttr::get(tensor.getContext(), shape),
                           prototype.getAxisMaps(), prototype.getValidity(),
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
    SmallVector<Attribute> fields;
    for (Attribute field : record.getFieldTypes()) {
      FailureOr<Type> converted =
          convertDataType(cast<TypeAttr>(field).getValue(), origin,
                          std::nullopt, owner, resultIndex);
      if (failed(converted))
        return failure();
      fields.push_back(TypeAttr::get(*converted));
    }
    return Type(gpu::RecordType::get(type.getContext(), record.getFieldNames(),
                                     ArrayAttr::get(type.getContext(), fields),
                                     owner));
  }
  if (auto tuple = dyn_cast<intent::TupleType>(type)) {
    SmallVector<Attribute> names;
    SmallVector<Attribute> fields;
    for (auto [index, field] : llvm::enumerate(tuple.getComponentTypes())) {
      names.push_back(StringAttr::get(type.getContext(),
                                     ("_" + Twine(index)).str()));
      FailureOr<Type> converted =
          convertDataType(cast<TypeAttr>(field).getValue(), origin,
                          std::nullopt, owner, resultIndex);
      if (failed(converted))
        return failure();
      fields.push_back(TypeAttr::get(*converted));
    }
    return Type(gpu::RecordType::get(
        type.getContext(), ArrayAttr::get(type.getContext(), names),
        ArrayAttr::get(type.getContext(), fields), owner));
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
    auto mapping = cast<AxisMapAttr>(sourceType.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(
        logical.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        resultAxis++));
  }
  auto prototype = FragmentType::get(
      logical.getContext(), logicalType.getElementType(),
      ArrayAttr::get(logical.getContext(), shape),
      ArrayAttr::get(logical.getContext(), mappings), sourceType.getValidity(),
      sourceType.getOwner());
  return convertDataType(logical, origin, Type(prototype), 1, resultIndex);
}

LogicalResult collectDomainAxes(Value source,
                                SmallVectorImpl<intent::DomainOp> &axes);

class ScalarRegionLowering {
public:
  ScalarRegionLowering(OpBuilder &builder, llvm::DenseMap<Value, Value> values,
                       ArrayRef<Value> views,
                       llvm::DenseMap<int64_t, Value> dimensions,
                       unsigned orderedDepth = 0)
      : builder(builder), values(std::move(values)), views(views),
        dimensions(std::move(dimensions)), orderedDepth(orderedDepth) {}

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

  void mapResults(Operation *source, Operation *target) {
    for (auto [from, to] : llvm::zip(source->getResults(), target->getResults()))
      values[from] = to;
    if (Attribute node = source->getAttr("intent.node"))
      target->setAttr(gpu::originAttr, node);
  }

  LogicalResult lowerPureRegion(Region &source, Region &target,
                                ArrayRef<Type> argumentTypes) {
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
                               dimensions, orderedDepth);
    FailureOr<SmallVector<Value>> yielded = child.lowerBlock(source.front());
    if (failed(yielded))
      return failure();
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
                                  start, 1);
    Value adjusted = createBinary(
        builder, location, builder.getIndexType(), distance,
        createBinary(builder, location, builder.getIndexType(), step, one, 1), 0);
    return createBinary(builder, location, builder.getIndexType(), adjusted, step,
                        4);
  }

  FailureOr<Value> physicalExtentValue(Location location,
                                       PhysicalExprAttr expression) {
    auto kind = static_cast<PhysicalExprKind>(expression.getKind());
    if (kind == PhysicalExprKind::Constant)
      return Value(builder.create<arith::ConstantIndexOp>(location,
                                                           expression.getValue()));
    if (kind == PhysicalExprKind::Dimension) {
      StringRef symbol = expression.getSymbol().getValue();
      if (!symbol.consume_front("D"))
        return failure();
      int64_t identity = 0;
      if (symbol.getAsInteger(10, identity))
        return failure();
      auto found = dimensions.find(identity);
      return found == dimensions.end() ? FailureOr<Value>(failure())
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
    if (kind == PhysicalExprKind::Parameter) {
      Value result;
      function.walk([&](gpu::ParameterOp parameter) {
        if (!result && parameter.getParameter().getName() == expression.getSymbol())
          result = parameter.getResult();
      });
      return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
    }
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
      return createBinary(builder, location, builder.getIndexType(), *lhs, *rhs, 0);
    if (kind == PhysicalExprKind::Subtract)
      return createBinary(builder, location, builder.getIndexType(), *lhs, *rhs, 1);
    if (kind == PhysicalExprKind::Multiply)
      return createBinary(builder, location, builder.getIndexType(), *lhs, *rhs, 2);
    if (kind == PhysicalExprKind::FloorDiv)
      return createBinary(builder, location, builder.getIndexType(), *lhs, *rhs, 4);
    if (kind == PhysicalExprKind::Minimum)
      return createBinary(builder, location, builder.getIndexType(), *lhs, *rhs, 8);
    if (kind == PhysicalExprKind::Maximum)
      return createBinary(builder, location, builder.getIndexType(), *lhs, *rhs, 7);
    if (kind == PhysicalExprKind::CeilDiv) {
      Value one = builder.create<arith::ConstantIndexOp>(location, 1);
      Value adjusted = createBinary(
          builder, location, builder.getIndexType(), *lhs,
          createBinary(builder, location, builder.getIndexType(), *rhs, one, 1),
          0);
      return createBinary(builder, location, builder.getIndexType(), adjusted,
                          *rhs, 4);
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
    auto resultDimension = [&](size_t resultAxis) -> FailureOr<int64_t> {
      ArrayRef<int64_t> dimensions = relation.getResultDimensions();
      if (resultAxis >= dimensions.size())
        return failure();
      return dimensions[resultAxis];
    };
    auto resultExtent = [&](size_t axis,
                            PhysicalExprAttr fallback)
        -> FailureOr<PhysicalExprAttr> {
      if (operation->getNumResults() == 0)
        return fallback;
      auto tensor = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
      if (!tensor || axis >= static_cast<size_t>(tensor.getRank()))
        return fallback;
      if (!tensor.isDynamicDim(axis))
        return expression(operation->getContext(), PhysicalExprKind::Constant,
                          tensor.getDimSize(axis));
      FailureOr<PhysicalExprAttr> extent =
          dimensionExtentExpression(tensor, operation, axis);
      return extent;
    };
    auto makeRange = [&](unsigned sourceAxis, Value start, Value stop,
                         Value step, uint64_t sourceId,
                         PhysicalExprAttr physicalExtent) -> Value {
      Value one = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 1);
      Value distance = createBinary(builder, operation->getLoc(),
                                    builder.getIndexType(), stop, start, 1);
      Value adjusted = createBinary(
          builder, operation->getLoc(), builder.getIndexType(), distance,
          createBinary(builder, operation->getLoc(), builder.getIndexType(), step,
                       one, 1),
          0);
      Value extent = createBinary(builder, operation->getLoc(),
                                  builder.getIndexType(), adjusted, step, 4);
      auto type = fragmentType(operation->getContext(), builder.getIndexType(),
                               {physicalExtent}, {{sourceId, sourceAxis}});
      return builder.create<gpu::MakeRangeOp>(
          operation->getLoc(), type, start, extent, step, sourceId, sourceAxis);
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
          extent =
              cast<PhysicalExprAttr>(fragment.getShape()[sourceAxis]);
        }
        FailureOr<int64_t> dimension = resultDimension(resultAxis);
        if (failed(dimension))
          return failure();
        if (*dimension > 0) {
          sourceId = static_cast<uint64_t>(*dimension);
          logicalSourceAxis = 0;
        }
        Value step = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 1);
        coordinates.push_back(makeRange(
            logicalSourceAxis, start, stop, step, sourceId, extent));
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
          if (!logicalCoordinate || fragment.getShape().size() > advancedRank)
            return failure();
          if (!advancedStart) {
            advancedStart = resultAxis;
            resultAxis += advancedRank;
          }
          unsigned alignedStart =
              *advancedStart + advancedRank - fragment.getShape().size();
          SmallVector<Attribute> mappings;
          for (auto [axis, attribute] :
               llvm::enumerate(fragment.getAxisMaps())) {
            auto mapping = cast<gpu::AxisMapAttr>(attribute);
            FailureOr<int64_t> dimension =
                resultDimension(alignedStart + axis);
            if (failed(dimension))
              return failure();
            uint64_t sourceId = mapping.getSourceId();
            uint32_t logicalSourceAxis = mapping.getSourceAxis();
            if (*dimension > 0) {
              sourceId = static_cast<uint64_t>(*dimension);
              logicalSourceAxis = 0;
            }
            mappings.push_back(gpu::AxisMapAttr::get(
                operation->getContext(), sourceId, logicalSourceAxis,
                mappings.size()));
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
        Value start = builder.create<gpu::RangeBoundOp>(
            operation->getLoc(), builder.getIndexType(), *range, 0);
        Value stop = builder.create<gpu::RangeBoundOp>(
            operation->getLoc(), builder.getIndexType(), *range, 1);
        Value step = builder.create<gpu::RangeBoundOp>(
            operation->getLoc(), builder.getIndexType(), *range, 2);
        auto rangeType = cast<gpu::RangeType>((*range).getType());
        FailureOr<int64_t> dimension = resultDimension(resultAxis);
        if (failed(dimension))
          return failure();
        PhysicalExprAttr fallback =
            *dimension > 0
                ? dimensionExpression(operation->getContext(), *dimension)
                : expression(operation->getContext(), PhysicalExprKind::Constant,
                             1);
        FailureOr<PhysicalExprAttr> physicalExtent =
            resultExtent(resultAxis, fallback);
        if (failed(physicalExtent))
          return failure();
        PhysicalExprAttr extent = *physicalExtent;
        uint64_t sourceId = rangeType.getSourceId();
        uint32_t logicalSourceAxis = rangeType.getSourceAxis();
        if (*dimension > 0) {
          sourceId = static_cast<uint64_t>(*dimension);
          logicalSourceAxis = 0;
        }
        coordinates.push_back(makeRange(logicalSourceAxis, start, stop, step,
                                        sourceId, extent));
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
        FailureOr<int64_t> dimension = resultDimension(resultAxis);
        if (failed(dimension))
          return failure();
        PhysicalExprAttr fallback =
            *dimension > 0
                ? dimensionExpression(operation->getContext(), *dimension)
                : view
                      ? cast<PhysicalExprAttr>(
                            view.getLayout().getExtents()[sourceAxis])
                      : cast<PhysicalExprAttr>(fragment.getShape()[sourceAxis]);
        FailureOr<PhysicalExprAttr> physicalExtent =
            resultExtent(resultAxis, fallback);
        if (failed(physicalExtent))
          return failure();
        PhysicalExprAttr extent = *physicalExtent;
        uint64_t sourceId;
        uint32_t logicalSourceAxis = sourceAxis;
        if (view)
          sourceId =
              (static_cast<uint64_t>(view.getAbiIndex()) + 1) * 65536 +
              sourceAxis + 1;
        else {
          auto mapping =
              cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[sourceAxis]);
          sourceId = mapping.getSourceId();
          logicalSourceAxis = mapping.getSourceAxis();
        }
        if (*dimension > 0) {
          sourceId = static_cast<uint64_t>(*dimension);
          logicalSourceAxis = 0;
        }
        coordinates.push_back(makeRange(logicalSourceAxis, bounds[0], bounds[1],
                                        bounds[2], sourceId, extent));
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
                                   ArrayRef<Value> coordinates) {
    FailureOr<Type> converted = convertDataType(logical, operation);
    if (failed(converted)) {
      operation->emitOpError(
          "access result logical type has no physical fragment schema");
      return failure();
    }
    auto fragment = dyn_cast<gpu::FragmentType>(*converted);
    if (!fragment)
      return *converted;
    auto relation = operation->getAttrOfType<IndexRelationAttr>("index");
    if (!relation)
      return failure();
    SmallVector<Attribute> mappings;
    unsigned coordinate = 0;
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
    bool advancedMapped = false;
    auto resultDimension = [&](size_t axis) -> FailureOr<int64_t> {
      ArrayRef<int64_t> dimensions = relation.getResultDimensions();
      if (axis >= dimensions.size())
        return failure();
      return dimensions[axis];
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
        uint64_t sourceId = mapping.getSourceId();
        uint32_t sourceAxis = mapping.getSourceAxis();
        if (*dimension > 0) {
          sourceId = static_cast<uint64_t>(*dimension);
          sourceAxis = 0;
        }
        mappings.push_back(gpu::AxisMapAttr::get(
            operation->getContext(), sourceId, sourceAxis, resultAxis++));
      }
      return success();
    };
    auto advancedAxisMapping = [&](unsigned advancedAxis)
        -> gpu::AxisMapAttr {
      gpu::AxisMapAttr fallback;
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
        if (!unit)
          return mapping;
        if (!fallback)
          fallback = mapping;
      }
      return fallback;
    };
    for (Attribute attribute : relation.getTerms()) {
      auto term = cast<IndexTermAttr>(attribute);
      if (term.getKind() == 1) {
        FailureOr<uint64_t> identity =
            resultAxisIdentity(operation, /*resultIndex=*/0, resultAxis);
        if (failed(identity))
          return failure();
        mappings.push_back(gpu::AxisMapAttr::get(
            operation->getContext(), *identity, 0, resultAxis));
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
        if (position < 0 || position >= operation->getNumOperands() ||
            !isa<RankedTensorType>(operation->getOperand(position).getType()))
          return failure();
        if (!advancedMapped) {
          for (unsigned axis = 0; axis < advancedRank; ++axis) {
            FailureOr<int64_t> dimension = resultDimension(resultAxis);
            if (failed(dimension))
              return failure();
            uint64_t sourceId;
            uint32_t sourceAxis;
            if (*dimension > 0) {
              sourceId = static_cast<uint64_t>(*dimension);
              sourceAxis = 0;
            } else if (auto mapping = advancedAxisMapping(axis)) {
              sourceId = mapping.getSourceId();
              sourceAxis = mapping.getSourceAxis();
            } else {
              FailureOr<uint64_t> identity =
                  resultAxisIdentity(operation, /*resultIndex=*/0, resultAxis);
              if (failed(identity))
                return failure();
              sourceId = *identity;
              sourceAxis = 0;
            }
            mappings.push_back(gpu::AxisMapAttr::get(
                operation->getContext(), sourceId, sourceAxis, resultAxis++));
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
        operation->getContext(), fragment.getElementType(), fragment.getShape(),
        builder.getArrayAttr(mappings), fragment.getValidity(),
        fragment.getOwner()));
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
      FailureOr<TypedAttr> value =
          physicalConstant(constant.getValue(), constant.getResult().getType());
      if (failed(value))
        return constant.emitOpError(
            "constant cannot be represented by its physical result type");
      auto target = builder.create<arith::ConstantOp>(
          location, constant.getResult().getType(), *value);
      mapResults(operation, target);
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
        Value start = builder.create<gpu::RangeBoundOp>(
            location, builder.getIndexType(), *source, 0);
        Value stop = builder.create<gpu::RangeBoundOp>(
            location, builder.getIndexType(), *source, 1);
        Value step = builder.create<gpu::RangeBoundOp>(
            location, builder.getIndexType(), *source, 2);
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
      auto type = gpu::RangeType::get(
          operation->getContext(), domain.getResult().getType().getOriginId(), 0);
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
      Value start = builder.create<gpu::RangeBoundOp>(
          location, builder.getIndexType(), *source, 0);
      Value stop = builder.create<gpu::RangeBoundOp>(
          location, builder.getIndexType(), *source, 1);
      Value step = builder.create<gpu::RangeBoundOp>(
          location, builder.getIndexType(), *source, 2);
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
      auto type = gpu::RangeType::get(operation->getContext(), logical.getSourceId(),
                                      0);
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
      auto target = builder.create<gpu::RangeBoundOp>(
          location, builder.getIndexType(), *source, 1);
      mapResults(operation, target);
      return success();
    }
    if (auto indices = dyn_cast<intent::IndicesOp>(operation)) {
      FailureOr<Value> source = get(indices.getSource());
      if (failed(source) || !isa<gpu::RangeType>((*source).getType()))
        return indices.emitOpError(
            "indices currently requires an explicit physical range source");
      Value start = builder.create<gpu::RangeBoundOp>(
          location, builder.getIndexType(), *source, 0);
      Value stop = builder.create<gpu::RangeBoundOp>(
          location, builder.getIndexType(), *source, 1);
      Value step = builder.create<gpu::RangeBoundOp>(
          location, builder.getIndexType(), *source, 2);
      Value distance = createBinary(builder, location, builder.getIndexType(),
                                    stop, start, 1);
      Value one = builder.create<arith::ConstantIndexOp>(location, 1);
      Value adjusted = createBinary(
          builder, location, builder.getIndexType(), distance,
          createBinary(builder, location, builder.getIndexType(), step, one, 1),
          0);
      Value extent = createBinary(builder, location, builder.getIndexType(),
                                  adjusted, step, 4);
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
              rangeType.getSourceAxis(), 0)}),
          genericType.getValidity(), genericType.getOwner());
      auto target = builder.create<gpu::MakeRangeOp>(
          location, resultType, start, extent, step, rangeType.getSourceId(),
          rangeType.getSourceAxis());
      mapResults(operation, target);
      return success();
    }
    if (auto full = dyn_cast<intent::FullOp>(operation)) {
      FailureOr<Value> fill = get(full.getInputs().front());
      FailureOr<Type> result =
          convertDataType(full.getResult().getType(), operation);
      if (failed(fill) || failed(result) || !isa<gpu::FragmentType>(*result))
        return full.emitOpError("full has no physical fragment realization");
      auto target =
          builder.create<gpu::SplatOp>(location, *result, *fill);
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
      auto target =
          builder.create<gpu::BroadcastOp>(location, *result, *input);
      mapResults(operation, target);
      return success();
    }
    if (auto reshape = dyn_cast<intent::ReshapeOp>(operation)) {
      FailureOr<Value> input = get(reshape.getInputs().front());
      FailureOr<Type> result =
          convertDataType(reshape.getResult().getType(), operation);
      if (failed(input) || failed(result) || !isa<gpu::FragmentType>(*result))
        return reshape.emitOpError(
            "reshape has no physical fragment realization");
      auto target = builder.create<gpu::ReshapeOp>(
          location, *result, *input, reshape.getShape().getAxes());
      mapResults(operation, target);
      return success();
    }
    if (auto transpose = dyn_cast<intent::TransposeOp>(operation)) {
      FailureOr<Value> input = get(transpose.getInput());
      FailureOr<Type> result =
          convertDataType(transpose.getResult().getType(), operation);
      if (failed(input) || failed(result) || !isa<gpu::FragmentType>(*result))
        return transpose.emitOpError(
            "transpose has no physical fragment realization");
      SmallVector<int64_t> permutation;
      for (Attribute axis : transpose.getPermutation())
        permutation.push_back(cast<IntegerAttr>(axis).getInt());
      auto target = builder.create<gpu::TransposeOp>(
          location, *result, *input, permutation);
      mapResults(operation, target);
      return success();
    }
    if (auto join = dyn_cast<intent::JoinOp>(operation)) {
      FailureOr<Value> lhs = get(join.getLhs());
      FailureOr<Value> rhs = get(join.getRhs());
      FailureOr<Type> result =
          convertDataType(join.getResult().getType(), operation);
      if (failed(lhs) || failed(rhs) || failed(result) ||
          !isa<gpu::FragmentType>(*result))
        return join.emitOpError("join has no physical fragment realization");
      auto target = builder.create<gpu::JoinOp>(
          location, *result, *lhs, *rhs,
          cast<gpu::FragmentType>(*result).getShape().size() - 1);
      mapResults(operation, target);
      return success();
    }
    if (auto unary = dyn_cast<intent::UnaryOp>(operation)) {
      FailureOr<Value> input = get(unary.getInput());
      if (failed(input))
        return failure();
      FailureOr<Type> result = convertDataType(
          unary.getResult().getType(), operation, (*input).getType());
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
      if (failed(lhs) || failed(rhs) ||
          failed(alignElementwiseOperands(builder, location, *lhs, *rhs)))
        return binary.emitOpError("binary operands have no physical data schema");
      FailureOr<Type> result = convertDataType(
          binary.getResult().getType(), operation, (*lhs).getType());
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
          failed(alignElementwiseOperands(builder, location, *lhs, *rhs)))
        return compare.emitOpError("comparison operands are unavailable");
      FailureOr<Type> result = convertDataType(
          compare.getResult().getType(), operation, (*lhs).getType());
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
          failed(alignElementwiseOperands(builder, location, *trueValue,
                                          *falseValue)))
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
      FailureOr<Type> result = convertDataType(
          select.getResult().getType(), operation, (*trueValue).getType());
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
                        : convertDataType(cast.getResult().getType(), operation,
                                          (*input).getType());
      if (failed(input) || failed(result))
        return cast.emitOpError("cast input/result is unavailable");
      auto target = builder.create<gpu::CastOp>(location, *result, *input);
      mapResults(operation, target);
      return success();
    }
    if (auto bitcast = dyn_cast<intent::BitcastOp>(operation)) {
      FailureOr<Value> input = get(bitcast.getInput());
      FailureOr<Type> result =
          failed(input) ? FailureOr<Type>(failure())
                        : convertDataType(bitcast.getResult().getType(), operation,
                                          (*input).getType());
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
      FailureOr<Type> result =
          failed(value) ? FailureOr<Type>(failure())
                        : convertDataType(mask.getResult().getType(), operation,
                                          (*value).getType());
      if (failed(value) || failed(predicate) || failed(fill) || failed(result))
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
          return reduce.emitOpError("reduce result has no physical schema");
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
                                 arguments)))
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
                                 arguments)))
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
      FailureOr<uint64_t> segmentIdentity = resultAxisIdentity(operation);
      if (failed(segmentIdentity))
        return fold.emitOpError(
            "region-fold result has no canonical segment identity");
      std::string name =
          ("SEGMENT_" + Twine(*segmentIdentity)).str();
      auto segment = gpu::ParameterAttr::get(
          operation->getContext(), builder.getStringAttr(name),
          static_cast<uint32_t>(gpu::ParameterRole::ScanChunk),
          builder.getDenseI64ArrayAttr({32, 64, 128}));
      builder.create<gpu::ParameterOp>(location, builder.getIndexType(), segment);
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
      Block &sourceSummary = fold.getSummarize().front();
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
                                 summarizeArguments)))
        return failure();
      SmallVector<Type> combineArguments(results);
      combineArguments.append(results);
      if (failed(lowerPureRegion(fold.getCombine(), target.getCombine(),
                                 combineArguments)))
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
        FailureOr<Type> converted =
            convertDataType(logical, operation, std::nullopt, /*owner=*/1,
                            index);
        if (failed(converted))
          return scan.emitOpError("region-scan result has no physical schema");
        results.push_back(*converted);
      }
      FailureOr<uint64_t> segmentIdentity = resultAxisIdentity(operation);
      if (failed(segmentIdentity))
        return scan.emitOpError(
            "region-scan result has no canonical segment identity");
      std::string name =
          ("SEGMENT_" + Twine(*segmentIdentity)).str();
      auto segment = gpu::ParameterAttr::get(
          operation->getContext(), builder.getStringAttr(name),
          static_cast<uint32_t>(gpu::ParameterRole::ScanChunk),
          builder.getDenseI64ArrayAttr({32, 64, 128}));
      builder.create<gpu::ParameterOp>(location, builder.getIndexType(), segment);
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
      Block &sourceSummary = scan.getSummarize().front();
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
                                 summarizeArguments)))
        return failure();
      SmallVector<Type> combineArguments(transitionTypes);
      combineArguments.append(transitionTypes);
      if (failed(lowerPureRegion(scan.getCombine(), target.getCombine(),
                                 combineArguments)))
        return failure();
      SmallVector<Type> applyArguments(transitionTypes);
      applyArguments.append(stateTypes);
      if (failed(lowerPureRegion(scan.getApply(), target.getApply(),
                                 applyArguments)))
        return failure();
      SmallVector<Type> emitArguments(sliceTypes);
      emitArguments.append(stateTypes);
      emitArguments.append(captureTypes);
      if (failed(lowerPureRegion(scan.getEmit(), target.getEmit(),
                                 emitArguments)))
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
      FailureOr<Type> result =
          convertDataType(contract.getResult().getType(), operation);
      if (failed(lhs) || failed(rhs) || failed(result) ||
          !isa<gpu::FragmentType>((*lhs).getType()) ||
          !isa<gpu::FragmentType>((*rhs).getType()) ||
          !isa<gpu::FragmentType>(*result))
        return contract.emitOpError(
            "contract operands/results have no physical fragment schema");
      FailureOr<Value> accumulator =
          zeroAccumulator(cast<gpu::FragmentType>(*result));
      if (failed(accumulator))
        return contract.emitOpError("contract accumulator dtype is unsupported");
      SmallVector<int64_t> lhsReduction, rhsReduction, lhsBatch, rhsBatch;
      axisPairs(contract.getReduce(), lhsReduction, rhsReduction);
      axisPairs(contract.getBatch(), lhsBatch, rhsBatch);
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
      FailureOr<Type> result =
          convertDataType(contract.getResult().getType(), operation);
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
      SmallVector<int64_t> lhsReduction, rhsReduction, lhsBatch, rhsBatch;
      axisPairs(contract.getReduce(), lhsReduction, rhsReduction);
      axisPairs(contract.getBatch(), lhsBatch, rhsBatch);
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
      FailureOr<Type> result =
          convertDataType(contract.getResult().getType(), operation);
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
      SmallVector<int64_t> lhsReduction, rhsReduction, lhsBatch, rhsBatch;
      axisPairs(contract.getReduce(), lhsReduction, rhsReduction);
      axisPairs(contract.getBatch(), lhsBatch, rhsBatch);
      auto format = contract.getFormat();
      auto target = builder.create<gpu::SparseContractOp>(
          location, *result, *compressed, *metadata, *rhs, *logicalExtent,
          *accumulator, format.getKind(), format.getCompressionAxis(),
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
      FailureOr<uint64_t> instance = resultAxisIdentity(operation);
      if (failed(instance))
        return buffer.emitOpError(
            "logical buffer has no canonical physical instance identity");
      auto physicalType = gpu::BufferType::get(
          operation->getContext(), tensor.getElementType(), valueType->getShape(),
          /*scope=*/orderedDepth == 0 ? 0 : 1, *instance,
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
          auto fragment = *operand
                              ? dyn_cast<gpu::FragmentType>((*operand).getType())
                              : gpu::FragmentType();
          if (!fragment || samePhysicalShape(fragment, target))
            continue;
          FailureOr<Value> aligned =
              retargetBroadcast(builder, location, *operand, target);
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
          auto fragment = *operand
                              ? dyn_cast<gpu::FragmentType>((*operand).getType())
                              : gpu::FragmentType();
          if (!fragment || samePhysicalShape(fragment, target))
            continue;
          FailureOr<Value> aligned =
              retargetBroadcast(builder, location, *operand, target);
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
          get(store.getInputs()[store.getValueOperandIndex()]);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value))
        return store.emitOpError("view store is not a scalar physical access");
      auto target = builder.create<gpu::StoreOp>(
          location, *resource, *coordinates, *value, Value(), *axes, 0);
      if (Attribute node = operation->getAttr("intent.node"))
        target->setAttr(gpu::originAttr, node);
      return success();
    }
    if (auto store = dyn_cast<intent::BufferStoreOp>(operation)) {
      FailureOr<Value> resource = get(store.getInputs().front());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          get(store.getInputs()[store.getValueOperandIndex()]);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value))
        return store.emitOpError("buffer store physical relation is unavailable");
      auto target = builder.create<gpu::StoreOp>(
          location, *resource, *coordinates, *value, Value(), *axes, 0);
      attachOrigin(operation, target);
      return success();
    }
    if (auto store = dyn_cast<intent::ScatterUniqueOp>(operation)) {
      FailureOr<Value> resource = get(store.getInputs().front());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          get(store.getInputs()[store.getValueOperandIndex()]);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value))
        return store.emitOpError("unique scatter is not a scalar physical access");
      auto target = builder.create<gpu::StoreOp>(
          location, *resource, *coordinates, *value, Value(), *axes, 0);
      if (Attribute node = operation->getAttr("intent.node"))
        target->setAttr(gpu::originAttr, node);
      return success();
    }
    if (auto scatter = dyn_cast<intent::ScatterReduceOp>(operation)) {
      FailureOr<Value> resource = get(scatter.getInputs().front());
      FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation);
      FailureOr<SmallVector<int64_t>> axes = sourceAxes(operation);
      FailureOr<Value> value =
          get(scatter.getInputs()[scatter.getValueOperandIndex()]);
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
          "sharing",
          builder.getI64IntegerAttr(
              isa<gpu::ViewType>((*resource).getType()) ? 1 : 0));
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
          convertDataType(atomic.getResult().getType(), operation);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(result))
        return atomic.emitOpError("atomic load address/result is unavailable");
      uint64_t sharing = isa<gpu::ViewType>((*resource).getType()) ? 1 : 0;
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
          get(atomic.getInputs()[atomic.getValueOperand()]);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value))
        return atomic.emitOpError("atomic store address/value is unavailable");
      uint64_t sharing = isa<gpu::ViewType>((*resource).getType()) ? 1 : 0;
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
          get(atomic.getInputs()[atomic.getValueOperand()]);
      FailureOr<Type> result =
          convertDataType(atomic.getResult().getType(), operation);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(value) || failed(result))
        return atomic.emitOpError("atomic RMW address/value is unavailable");
      uint64_t sharing = isa<gpu::ViewType>((*resource).getType()) ? 1 : 0;
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
          get(atomic.getInputs()[atomic.getExpectedOperand()]);
      FailureOr<Value> desired =
          get(atomic.getInputs()[atomic.getDesiredOperand()]);
      FailureOr<Type> result =
          convertDataType(atomic.getResult().getType(), operation);
      if (failed(resource) || failed(coordinates) || failed(axes) ||
          failed(expected) || failed(desired) || failed(result))
        return atomic.emitOpError("compare-exchange address/value is unavailable");
      uint64_t sharing = isa<gpu::ViewType>((*resource).getType()) ? 1 : 0;
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
      auto target = builder.create<gpu::MakeRecordOp>(location, *result, fields);
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
      auto target = builder.create<gpu::MakeRecordOp>(location, *result, fields);
      mapResults(operation, target);
      return success();
    }
    if (auto extract = dyn_cast<intent::ExtractOp>(operation)) {
      FailureOr<Value> product = get(extract.getProduct());
      FailureOr<Type> result =
          convertDataType(extract.getResult().getType(), operation);
      if (failed(product) || failed(result))
        return extract.emitOpError("product projection is unavailable");
      auto target = builder.create<gpu::ExtractOp>(location, *result, *product,
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
        ScalarRegionLowering child(nested, values, views, dimensions,
                                   orderedDepth);
        FailureOr<SmallVector<Value>> yielded =
            child.lowerBlock(sourceRegion.front());
        if (failed(yielded))
          return failure();
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
      SmallVector<intent::DomainOp> axes;
      if (failed(collectDomainAxes(forOperation.getInputs().front(), axes)) ||
          axes.empty())
        return forOperation.emitOpError(
            "ordered for source is not an executable domain product");
      SmallVector<Value> lowers, uppers, steps;
      for (intent::DomainOp axis : axes) {
        FailureOr<Value> lower = get(axis.getBounds()[0]);
        FailureOr<Value> upper = get(axis.getBounds()[1]);
        if (failed(lower) || failed(upper))
          return axis.emitOpError("ordered physical loop bounds are unavailable");
        Value step;
        if (axis.getBounds().size() == 3) {
          FailureOr<Value> lowered = get(axis.getBounds()[2]);
          if (failed(lowered))
            return axis.emitOpError(
                "ordered physical loop step is unavailable");
          step = *lowered;
        } else if ((*lower).getType().isIndex()) {
          step = builder.create<arith::ConstantIndexOp>(location, 1);
        } else if (auto integer = dyn_cast<IntegerType>((*lower).getType())) {
          step = builder.create<arith::ConstantOp>(
              location, integer, builder.getIntegerAttr(integer, 1));
        }
        if (!step || (*lower).getType() != (*upper).getType() ||
            (*lower).getType() != step.getType())
          return axis.emitOpError(
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
                                     dimensions, orderedDepth + 1);
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
                                   dimensions, orderedDepth + 1);
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
                                   dimensions, orderedDepth + 1);
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
  for (Operation &nested : body.without_terminator()) {
    if (auto child = dyn_cast<intent::ParallelOp>(nested)) {
      children.push_back(child);
      continue;
    }
    nested.walk([&](Operation *candidate) {
      hasDirectEffect |=
          isa<intent::ViewStoreOp, intent::ScatterUniqueOp,
              intent::ScatterReduceOp, intent::AtomicStoreOp,
              intent::AtomicRMWOp,
              intent::AtomicCompareExchangeOp>(candidate);
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
      FailureOr<TypedAttr> value =
          physicalConstant(constant.getValue(), constant.getResult().getType());
      if (failed(value))
        return constant.emitOpError(
            "constant cannot be represented by its physical result type");
      auto target = builder.create<arith::ConstantOp>(
          constant.getLoc(), constant.getResult().getType(), *value);
      values[constant.getResult()] = target;
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
                                    dimensionValues);
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
                                    builder.getIndexType(), *stop, *start, 1);
      Value adjusted = createBinary(
          builder, domain.getLoc(), builder.getIndexType(), distance,
          createBinary(builder, domain.getLoc(), builder.getIndexType(), *step,
                       one, 1),
          0);
      Value extent = createBinary(builder, domain.getLoc(),
                                  builder.getIndexType(), adjusted, *step, 4);
      runtimeExtents.push_back(extent);
      starts.push_back(*start);
      steps.push_back(*step);
      runtimeLength = createBinary(builder, domain.getLoc(),
                                   builder.getIndexType(), runtimeLength, extent,
                                   2);
    }
    if (workset.singleton)
      runtimeExtents.push_back(one);
    Value segmentEnd = createBinary(builder, function.getLoc(),
                                    builder.getIndexType(), runtimeOffset,
                                    runtimeLength, 0);
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
      auto childValues = rootLowering.mapping();
      Block &sourceBlock = *workset.body;
      for (auto [axis, localCoordinate] :
           llvm::enumerate(decoded.getCoordinates())) {
        if (workset.singleton)
          break;
        Value scaled = createBinary(builder, worksetLocation,
                                    builder.getIndexType(), localCoordinate,
                                    steps[axis], 2);
        Value coordinate = createBinary(builder, worksetLocation,
                                        builder.getIndexType(), starts[axis],
                                        scaled, 0);
        childValues[workset.coordinateArguments[axis]] = coordinate;
      }
      ScalarRegionLowering lowering(builder, std::move(childValues),
                                    sourceArguments, dimensionValues);
      if (failed(lowering.lowerBlock(sourceBlock)))
        return failure();
      runtimeOffset = segmentEnd;
      launchOffset = binaryExpression(context, PhysicalExprKind::Add,
                                      launchOffset, workset.launchLength);
      continue;
    }
    Value afterOffset = createCompare(builder, function.getLoc(),
                                      builder.getI1Type(), pid, runtimeOffset, 5);
    Value beforeEnd = createCompare(builder, function.getLoc(), builder.getI1Type(),
                                    pid, segmentEnd, 2);
    Value active = createBinary(builder, function.getLoc(), builder.getI1Type(),
                                afterOffset, beforeEnd, 11);
    Location worksetLocation = workset.singleton ? function.getLoc()
                                                 : workset.operation.getLoc();
    auto dispatch = builder.create<scf::IfOp>(
        worksetLocation, active,
        [&](OpBuilder &nested, Location location) {
          Value local = createBinary(nested, location, nested.getIndexType(), pid,
                                     runtimeOffset, 1);
          SmallVector<Attribute> launchExtents(workset.launchExtents.begin(),
                                               workset.launchExtents.end());
          auto decoded = nested.create<gpu::DelinearizeOp>(
              location,
              SmallVector<Type>(runtimeExtents.size(), nested.getIndexType()),
              local, runtimeExtents,
              nested.getArrayAttr(launchExtents));
          auto childValues = rootLowering.mapping();
          Block &sourceBlock = *workset.body;
          for (auto [axis, localCoordinate] :
               llvm::enumerate(decoded.getCoordinates())) {
            if (workset.singleton)
              break;
            Value scaled = createBinary(nested, location, nested.getIndexType(),
                                        localCoordinate, steps[axis], 2);
            Value coordinate = createBinary(nested, location, nested.getIndexType(),
                                            starts[axis], scaled, 0);
            childValues[workset.coordinateArguments[axis]] = coordinate;
          }
          ScalarRegionLowering lowering(nested, std::move(childValues),
                                        sourceArguments, dimensionValues);
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
  // Structured construction is atomic: no runtime-sized logical fragment is
  // allowed to escape this boundary.  The shared co-realizers replace the
  // conservative whole-domain nodes with complete parameterized fragments,
  // loops, accesses, validity, accumulators, and effects before the first GPU
  // program is exposed to the rest of the pipeline.
  if (failed(gpu::realizeAccessComposition(module)) ||
      failed(gpu::realizeContractionBlocking(module)) ||
      failed(gpu::realizeReductionBlocking(module)))
    return failure();
  if (failed(gpu::verifyGPUProgram(module)))
    return failure();
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
  return constructGPUProgram(module, capabilities, functions.front());
}

} // namespace intent
