#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace intent::gpu;

#include "Intent/Dialect/GPU/IR/GPUAttrsEnums.cpp.inc"

#define GET_DIALECT_DEF
#include "Intent/Dialect/GPU/IR/GPUDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Intent/Dialect/GPU/IR/GPUAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Intent/Dialect/GPU/IR/GPUTypes.cpp.inc"

namespace intent::gpu {

namespace {

bool isCompileTimePhysicalExpr(PhysicalExprAttr expression) {
  auto kind = static_cast<PhysicalExprKind>(expression.getKind());
  if (kind == PhysicalExprKind::Constant || kind == PhysicalExprKind::Parameter)
    return true;
  if (kind == PhysicalExprKind::Dimension ||
      kind == PhysicalExprKind::ScalarABI)
    return false;
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isCompileTimePhysicalExpr(cast<PhysicalExprAttr>(operand));
  });
}

} // namespace

BroadcastProjection queryAxisProjection(FragmentType source,
                                        FragmentType target) {
  BroadcastProjection result;
  if (source.getOwner() != target.getOwner() ||
      source.getShape().size() > target.getShape().size())
    return result;
  result.targetToSource.resize(target.getShape().size());
  SmallVector<bool> sourceUsed(source.getShape().size(), false);
  const unsigned offset = target.getShape().size() - source.getShape().size();

  auto bindUnique = [&](unsigned sourceIndex,
                        function_ref<bool(AxisMapAttr)> selects) {
    std::optional<unsigned> selected;
    for (auto [targetIndex, mapping] : llvm::enumerate(target.getAxisMaps())) {
      if (result.targetToSource[targetIndex] ||
          !selects(cast<AxisMapAttr>(mapping)))
        continue;
      if (!selected || targetIndex == offset + sourceIndex)
        selected = targetIndex;
    }
    if (!selected)
      return true;
    unsigned matches = llvm::count_if(
        llvm::enumerate(target.getAxisMaps()), [&](auto item) {
          return !result.targetToSource[item.index()] &&
                 selects(cast<AxisMapAttr>(item.value()));
        });
    if (matches > 1 && *selected != offset + sourceIndex) {
      result.state = BroadcastProjectionState::Ambiguous;
      return false;
    }
    result.targetToSource[*selected] = sourceIndex;
    sourceUsed[sourceIndex] = true;
    return true;
  };

  // Broadcast semantics align source axes with the trailing target axes.  Use
  // that explicit occurrence relation before source-identity matching: one
  // logical source axis may legitimately occur more than once in a Cartesian
  // result, and identity-first matching would let the wrong occurrence consume
  // the positional target.  A non-singleton positional pair must still carry
  // either the same immutable source or the same logical dimension.
  for (auto [sourceIndex, mapping] : llvm::enumerate(source.getAxisMaps())) {
    unsigned targetIndex = offset + sourceIndex;
    auto sourceAxis = cast<AxisMapAttr>(mapping);
    auto targetAxis = cast<AxisMapAttr>(target.getAxisMaps()[targetIndex]);
    auto sourceExtent = cast<PhysicalExprAttr>(source.getShape()[sourceIndex]);
    bool singleton =
        sourceExtent.getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        sourceExtent.getValue() == 1;
    bool sameSource =
        sourceAxis.getSourceId() == targetAxis.getSourceId() &&
        sourceAxis.getSourceAxis() == targetAxis.getSourceAxis() &&
        sourceAxis.getDerived() == targetAxis.getDerived();
    if (!singleton && !sameSource &&
        sourceAxis.getDimensionId() != targetAxis.getDimensionId())
      continue;
    result.targetToSource[targetIndex] = sourceIndex;
    sourceUsed[sourceIndex] = true;
  }

  for (auto [sourceIndex, mapping] : llvm::enumerate(source.getAxisMaps())) {
    if (sourceUsed[sourceIndex])
      continue;
    auto sourceAxis = cast<AxisMapAttr>(mapping);
    if (!bindUnique(sourceIndex, [&](AxisMapAttr targetAxis) {
          return sourceAxis.getSourceId() == targetAxis.getSourceId() &&
                 sourceAxis.getSourceAxis() == targetAxis.getSourceAxis() &&
                 sourceAxis.getDerived() == targetAxis.getDerived();
        }))
      return result;
  }
  for (auto [sourceIndex, mapping] : llvm::enumerate(source.getAxisMaps())) {
    if (sourceUsed[sourceIndex])
      continue;
    auto sourceAxis = cast<AxisMapAttr>(mapping);
    auto sourceExtent = cast<PhysicalExprAttr>(source.getShape()[sourceIndex]);
    bool singleton =
        sourceExtent.getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        sourceExtent.getValue() == 1;
    if (singleton)
      continue;
    unsigned sourceOccurrences = llvm::count_if(
        llvm::enumerate(source.getAxisMaps()), [&](auto item) {
          if (sourceUsed[item.index()])
            return false;
          return cast<AxisMapAttr>(item.value()).getDimensionId() ==
                 sourceAxis.getDimensionId();
        });
    unsigned targetOccurrences = llvm::count_if(
        llvm::enumerate(target.getAxisMaps()), [&](auto item) {
          if (result.targetToSource[item.index()])
            return false;
          return cast<AxisMapAttr>(item.value()).getDimensionId() ==
                 sourceAxis.getDimensionId();
        });
    if (sourceOccurrences > 1 || targetOccurrences > 1) {
      result.state = BroadcastProjectionState::Ambiguous;
      return result;
    }
    if (!bindUnique(sourceIndex, [&](AxisMapAttr targetAxis) {
          return sourceAxis.getDimensionId() == targetAxis.getDimensionId();
        }))
      return result;
  }
  // Any remaining axes use the broadcast operation's explicit trailing-axis
  // relation.  Typed positional pairs were consumed first so repeated source
  // occurrences cannot steal one another's target; this final step also keeps
  // physical rematerializations whose producer and consumer carry different
  // derived source identities.
  for (unsigned sourceIndex = 0; sourceIndex < source.getShape().size();
       ++sourceIndex) {
    if (sourceUsed[sourceIndex])
      continue;
    unsigned targetIndex = offset + sourceIndex;
    if (result.targetToSource[targetIndex])
      return result;
    result.targetToSource[targetIndex] = sourceIndex;
    sourceUsed[sourceIndex] = true;
  }

  unsigned previous = 0;
  bool sawSource = false;
  for (std::optional<unsigned> sourceIndex : result.targetToSource) {
    if (!sourceIndex)
      continue;
    if (sawSource && *sourceIndex <= previous)
      return result;
    previous = *sourceIndex;
    sawSource = true;
  }
  result.state = BroadcastProjectionState::Exact;
  return result;
}

BroadcastProjection queryBroadcastProjection(FragmentType source,
                                              FragmentType target) {
  BroadcastProjection result = queryAxisProjection(source, target);
  if (!result.isExact())
    return result;
  for (auto [targetAxis, sourceAxis] :
       llvm::enumerate(result.targetToSource)) {
    if (!sourceAxis)
      continue;
    auto sourceExtent = cast<PhysicalExprAttr>(source.getShape()[*sourceAxis]);
    bool singleton =
        sourceExtent.getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        sourceExtent.getValue() == 1;
    if (!singleton && source.getShape()[*sourceAxis] != target.getShape()[targetAxis]) {
      result.state = BroadcastProjectionState::Unknown;
      return result;
    }
  }
  return result;
}

void IntentGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Intent/Dialect/GPU/IR/GPUAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Intent/Dialect/GPU/IR/GPUTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Intent/Dialect/GPU/IR/GPUOps.cpp.inc"
      >();
}

LogicalResult PhysicalExprAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, uint32_t kind,
    int64_t value, StringAttr symbol, ArrayAttr operands) {
  if (kind > 12 || !symbol || !operands)
    return emitError() << "physical expression has an invalid closed schema";
  for (Attribute operand : operands)
    if (!mlir::isa<PhysicalExprAttr>(operand))
      return emitError() << "physical expression operands must be #intent_gpu.expr";
  if (kind == 0)
    return symbol.empty() && operands.empty()
               ? success()
               : emitError() << "constant physical expression cannot carry a symbol or operands";
  if (kind == 1 || kind == 2 || kind == 3)
    return !symbol.empty() && operands.empty()
               ? success()
               : emitError() << "symbolic physical expression requires one name and no operands";
  if (!symbol.empty() || value != 0)
    return emitError() << "composed physical expression cannot carry a symbol or literal payload";
  unsigned expected = kind == 8 ? 3 : kind == 12 ? 1 : 2;
  if (operands.size() != expected)
    return emitError() << "composed physical expression has the wrong arity";
  if (kind == static_cast<uint32_t>(PhysicalExprKind::CeilDiv) ||
      kind == static_cast<uint32_t>(PhysicalExprKind::FloorDiv)) {
    auto divisor = mlir::cast<PhysicalExprAttr>(operands[1]);
    if (divisor.getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        divisor.getValue() == 0)
      return emitError() << "physical division requires a nonzero divisor";
  }
  return success();
}

LogicalResult ParameterAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, StringAttr name,
    uint32_t role, uint32_t category, uint32_t elementBitWidth,
    DenseI64ArrayAttr candidates) {
  if (!name || name.empty() ||
      role > static_cast<uint32_t>(ParameterRole::FullCoverage) ||
      category > static_cast<uint32_t>(ParameterCategory::RegionContraction) ||
      !candidates || candidates.empty())
    return emitError()
           << "physical parameter requires a name, role, category and candidates";
  auto typedCategory = static_cast<ParameterCategory>(category);
  const bool carriesDataGranularity =
      typedCategory == ParameterCategory::Pointwise ||
      typedCategory == ParameterCategory::Reduction ||
      typedCategory == ParameterCategory::Scan ||
      typedCategory == ParameterCategory::Contraction ||
      typedCategory == ParameterCategory::RegionReduction ||
      typedCategory == ParameterCategory::RegionContraction;
  if (carriesDataGranularity != (elementBitWidth > 0))
    return emitError()
           << "data-granularity parameter categories require an element bit width and non-data categories forbid one";
  llvm::DenseSet<int64_t> unique;
  for (int64_t candidate : candidates.asArrayRef())
    if (candidate <= 0 || !unique.insert(candidate).second)
      return emitError() << "physical parameter candidates must be unique positive integers";
  return success();
}

LogicalResult AxisMapAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, uint64_t sourceId,
    uint32_t sourceAxis, int64_t dimensionId, uint32_t fragmentAxis,
    bool derived) {
  (void)sourceAxis;
  (void)fragmentAxis;
  (void)derived;
  return sourceId != 0 && dimensionId > 0
             ? success()
             : emitError()
                   << "axis mapping requires source and logical-dimension identities";
}

LogicalResult ReshapeGroupAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DenseI64ArrayAttr sourceAxes, DenseI64ArrayAttr resultAxes) {
  if (!sourceAxes || !resultAxes ||
      (sourceAxes.empty() && resultAxes.empty()))
    return emitError()
           << "reshape group must cover at least one source or result axis";
  auto consecutive = [](ArrayRef<int64_t> axes) {
    if (axes.size() < 2)
      return true;
    return llvm::all_of(llvm::seq<size_t>(1, axes.size()), [&](size_t index) {
      return axes[index] == axes[index - 1] + 1;
    });
  };
  if ((!sourceAxes.empty() && sourceAxes[0] < 0) ||
      (!resultAxes.empty() && resultAxes[0] < 0) ||
      !consecutive(sourceAxes.asArrayRef()) ||
      !consecutive(resultAxes.asArrayRef()))
    return emitError()
           << "reshape group axes must be non-negative and consecutive";
  return success();
}

LogicalResult ViewLayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayAttr extents, DenseI64ArrayAttr dimensionIds, bool hasStrides,
    ArrayAttr strides, StringAttr alias, bool noalias) {
  if (!extents || !dimensionIds || !strides || !alias ||
      static_cast<int64_t>(extents.size()) != dimensionIds.size())
    return emitError() << "view layout requires dimensions, strides and alias facts";
  for (Attribute extent : extents)
    if (!mlir::isa<PhysicalExprAttr>(extent))
      return emitError() << "view extents must be typed physical expressions";
  if (!hasStrides && !strides.empty())
    return emitError() << "absent stride schema must use an empty array";
  if (!alias.empty() && noalias)
    return emitError() << "alias and noalias are mutually exclusive";
  for (int64_t dimension : dimensionIds.asArrayRef())
    if (dimension < 0)
      return emitError() << "view dimension identity cannot be negative";
  for (Attribute stride : strides)
    if (!mlir::isa<IntegerAttr, StringAttr>(stride))
      return emitError() << "view stride must be static or an ABI symbol";
  return success();
}

LogicalResult CapabilitiesAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, int64_t computeUnits,
    int64_t sharedMemoryPerUnit, int64_t maxDynamicSharedMemoryPerBlock,
    int64_t registersPerUnit, int64_t maxThreadsPerBlock,
    int64_t computeCapabilityMajor, int64_t computeCapabilityMinor,
    bool matrixUnits, bool dynamicVectorWidth) {
  (void)matrixUnits;
  (void)dynamicVectorWidth;
  if (computeUnits <= 0 || sharedMemoryPerUnit <= 0 ||
      maxDynamicSharedMemoryPerBlock <= 0 || registersPerUnit <= 0 ||
      maxThreadsPerBlock <= 0 || computeCapabilityMajor <= 0 ||
      computeCapabilityMinor < 0)
    return emitError() << "selected GPU capabilities require positive resource limits";
  return success();
}

LogicalResult ViewType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Type elementType, uint32_t rank,
                               uint32_t access, uint32_t abiIndex,
                               uint64_t sourceId,
                               ViewLayoutAttr layout) {
  (void)abiIndex;
  if (!elementType || rank == 0 || access > 2 || sourceId == 0 || !layout)
    return emitError() << "physical view schema is incomplete";
  if (layout.getDimensionIds().size() != rank)
    return emitError() << "physical view dimensions must match its rank";
  if (layout.getExtents().size() != rank)
    return emitError() << "physical view extents must match its rank";
  if (layout.getHasStrides() && layout.getStrides().size() != rank)
    return emitError() << "physical view strides must match its rank";
  return success();
}

LogicalResult FragmentType::verify(
    function_ref<InFlightDiagnostic()> emitError, Type elementType,
    ArrayAttr shape, ArrayAttr axisMaps, uint32_t validity, uint64_t owner) {
  if (!elementType || !shape || !axisMaps || validity > 2 || owner == 0 ||
      shape.empty())
    return emitError() << "physical fragment requires type, shape, mappings and owner";
  if (axisMaps.size() != shape.size())
    return emitError() << "fragment axis mappings must match physical rank";
  llvm::DenseSet<uint32_t> fragmentAxes;
  for (auto [axis, extent, mapping] : llvm::enumerate(shape, axisMaps)) {
    auto physical = mlir::dyn_cast<PhysicalExprAttr>(extent);
    if (!physical)
      return emitError() << "fragment extents must be typed physical expressions";
    if (!isCompileTimePhysicalExpr(physical))
      return emitError()
             << "fragment extents must be constant or physical-parameter expressions";
    auto typed = mlir::dyn_cast<AxisMapAttr>(mapping);
    if (!typed || typed.getFragmentAxis() != axis ||
        !fragmentAxes.insert(typed.getFragmentAxis()).second)
      return emitError() << "fragment axis mapping is incomplete or duplicated";
  }
  return success();
}

LogicalResult RangeType::verify(function_ref<InFlightDiagnostic()> emitError,
                                uint64_t sourceId, uint32_t sourceAxis,
                                int64_t dimensionId, bool derived) {
  (void)sourceAxis;
  (void)derived;
  return sourceId != 0 && dimensionId > 0
             ? success()
             : emitError()
                   << "physical range requires source and dimension provenance";
}

LogicalResult BufferType::verify(
    function_ref<InFlightDiagnostic()> emitError, Type elementType,
    ArrayAttr shape, BufferScopeAttr scope, uint64_t instance, uint64_t owner,
    BufferInitializationAttr initialization, BufferLifetimeAttr lifetime,
    uint32_t visibility, bool workspace) {
  if (!elementType || !shape || shape.empty() || !scope || instance == 0 ||
      owner == 0 || !initialization || !lifetime || visibility > 2)
    return emitError() << "physical buffer schema is incomplete";
  for (Attribute extent : shape)
    if (!mlir::isa<PhysicalExprAttr>(extent))
      return emitError() << "buffer extents must be typed physical expressions";
  if (workspace !=
      (scope.getValue() == BufferScope::InvocationWorkspace))
    return emitError() << "workspace flag and invocation scope disagree";
  bool lifetimeMatches =
      (scope.getValue() == BufferScope::ProgramPrivate &&
       lifetime.getValue() == BufferLifetime::Program) ||
      (scope.getValue() == BufferScope::IterationPrivate &&
       lifetime.getValue() == BufferLifetime::Iteration) ||
      (scope.getValue() == BufferScope::InvocationWorkspace &&
       lifetime.getValue() == BufferLifetime::Invocation);
  if (!lifetimeMatches)
    return emitError()
           << "physical buffer allocation scope and lifetime disagree";
  return success();
}

LogicalResult RecordType::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayAttr fieldNames,
    ArrayAttr fieldTypes, uint64_t owner) {
  if (!fieldNames || !fieldTypes || fieldNames.empty() ||
      fieldNames.size() != fieldTypes.size() || owner == 0)
    return emitError() << "physical record requires named typed components and owner";
  llvm::StringSet<> names;
  for (auto [name, type] : llvm::zip(fieldNames, fieldTypes)) {
    auto fieldName = mlir::dyn_cast<StringAttr>(name);
    auto fieldType = mlir::dyn_cast<TypeAttr>(type);
    if (!fieldName || fieldName.empty() || !fieldType ||
        !names.insert(fieldName.getValue()).second)
      return emitError() << "physical record fields must have unique names and types";
  }
  return success();
}

} // namespace intent::gpu
