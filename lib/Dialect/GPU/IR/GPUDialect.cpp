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
  return operands.size() == expected
             ? success()
             : emitError() << "composed physical expression has the wrong arity";
}

LogicalResult ParameterAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, StringAttr name,
    uint32_t role, DenseI64ArrayAttr candidates) {
  if (!name || name.empty() ||
      role > static_cast<uint32_t>(ParameterRole::ResidentWorkers) ||
      !candidates || candidates.empty())
    return emitError() << "physical parameter requires a name, role and candidates";
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
    int64_t sharedMemoryPerUnit, int64_t registersPerUnit, bool matrixUnits,
    bool dynamicVectorWidth) {
  (void)matrixUnits;
  (void)dynamicVectorWidth;
  if (computeUnits <= 0 || sharedMemoryPerUnit <= 0 || registersPerUnit <= 0)
    return emitError() << "selected GPU capabilities require positive resource limits";
  return success();
}

LogicalResult ViewType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Type elementType, uint32_t rank,
                               uint32_t access, uint32_t abiIndex,
                               ViewLayoutAttr layout) {
  (void)abiIndex;
  if (!elementType || rank == 0 || access > 2 || !layout)
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
    ArrayAttr shape, uint32_t scope, uint64_t instance, uint64_t owner,
    uint32_t initialization, uint32_t lifetime, uint32_t visibility,
    bool workspace) {
  if (!elementType || !shape || shape.empty() || scope > 2 || instance == 0 ||
      owner == 0 || initialization > 2 || lifetime > 2 || visibility > 2)
    return emitError() << "physical buffer schema is incomplete";
  for (Attribute extent : shape)
    if (!mlir::isa<PhysicalExprAttr>(extent))
      return emitError() << "buffer extents must be typed physical expressions";
  if (workspace != (scope == 2))
    return emitError() << "workspace flag and invocation scope disagree";
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
