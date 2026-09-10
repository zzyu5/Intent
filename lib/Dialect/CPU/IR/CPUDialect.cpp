#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace intent::cpu;

#include "Intent/Dialect/CPU/IR/CPUDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Intent/Dialect/CPU/IR/CPUAttrs.cpp.inc"

void IntentCPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Intent/Dialect/CPU/IR/CPUAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Intent/Dialect/CPU/IR/CPUOps.cpp.inc"
      >();
}

LogicalResult ViewArgumentAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, StringAttr name,
    Type element, DenseI64ArrayAttr shape, DenseI64ArrayAttr dimensions,
    uint32_t access, StringAttr alias, bool) {
  if (!name || name.getValue().empty() ||
      (!element.isF32() && !element.isUnsignedInteger(8)) || !shape || !dimensions ||
      shape.size() != dimensions.size() || access > 1 || !alias)
    return error() << "CPU view argument requires a named f32/u8 In/Out view and complete shape identities";
  for (auto [size, dimension] : llvm::zip(shape.asArrayRef(), dimensions.asArrayRef()))
    if ((size < 0 && !ShapedType::isDynamic(size)) || dimension < 0 ||
        (ShapedType::isDynamic(size) && dimension == 0))
      return error() << "CPU view extent or dynamic dimension identity is invalid";
  return success();
}

LogicalResult ScalarArgumentAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, StringAttr name, Type type) {
  if (!name || name.getValue().empty() ||
      (!type.isF32() && !type.isIndex() && !type.isInteger(64)))
    return error() << "CPU scalar argument requires a name and f32/index/i64 type";
  return success();
}

LogicalResult InterfaceAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, ArrayAttr arguments,
    bool contiguousViews, bool disjointOutputs) {
  if (!arguments || arguments.empty() || !contiguousViews || !disjointOutputs)
    return error() << "CPU interface requires contiguous views and disjoint outputs";
  llvm::DenseSet<StringAttr> names;
  for (Attribute argument : arguments) {
    StringAttr name;
    if (auto view = mlir::dyn_cast<ViewArgumentAttr>(argument)) name = view.getName();
    else if (auto scalar = mlir::dyn_cast<ScalarArgumentAttr>(argument)) name = scalar.getName();
    else return error() << "CPU interface contains an untyped argument";
    if (!names.insert(name).second)
      return error() << "CPU argument names must be unique";
  }
  return success();
}

LogicalResult CapabilitiesAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, int64_t vectorBits,
    int64_t workers, int64_t privateBytes) {
  if (vectorBits < 32 || vectorBits % 32 || workers <= 0 || privateBytes <= 0)
    return error() << "CPU capabilities require positive byte-addressable vector, worker and private-storage budgets";
  return success();
}

LogicalResult ConfigurationAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error,
    int64_t grain, int64_t m, int64_t n, int64_t k) {
  if (grain <= 0 || m <= 0 || n <= 0 || k <= 0)
    return error() << "CPU task/cache blocking requires positive extents";
  return success();
}

LogicalResult ImplementationAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, StringAttr name,
    DictionaryAttr parameters) {
  if (!name || name.empty() || !parameters)
    return error() << "CPU implementation requires a registered identity and explicit bindings";
  for (NamedAttribute parameter : parameters) {
    auto integer = mlir::dyn_cast<IntegerAttr>(parameter.getValue());
    if (!integer || integer.getInt() <= 0)
      return error() << "CPU implementation parameter must be a positive integer";
  }
  return success();
}

LogicalResult ReductionOrderAttr::verify(
    llvm::function_ref<InFlightDiagnostic()>, bool) { return success(); }

LogicalResult MicrotileAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, int64_t rows,
    int64_t columns, int64_t width) {
  if (rows <= 0 || columns <= 0 || width <= 0 || (width & (width - 1)) ||
      columns % width)
    return error() << "CPU register tile must contain complete positive issue groups";
  return success();
}
