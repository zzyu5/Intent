#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
using namespace mlir;
using namespace intent::dsa;
#include "Intent/Dialect/DSA/IR/DSADialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Intent/Dialect/DSA/IR/DSAAttrs.cpp.inc"
void IntentDSADialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Intent/Dialect/DSA/IR/DSAAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Intent/Dialect/DSA/IR/DSAOps.cpp.inc"
      >();
}
LogicalResult ViewArgumentAttr::verify(function_ref<InFlightDiagnostic()> error,
    StringAttr name, Type element, DenseI64ArrayAttr shape,
    DenseI64ArrayAttr dimensions, uint32_t access, intent::ViewConstraintsAttr constraints) {
  if (!name || name.empty() || (!element.isF16() && !element.isF32() && !element.isInteger(32) && !element.isInteger(64) && !element.isInteger(1)) ||
      !shape || !dimensions || shape.size() != dimensions.size() || access > 2 || !constraints)
    return error() << "DSA view requires supported numeric storage and a complete typed interface";
  for (auto [extent, dimension] : llvm::zip(shape.asArrayRef(), dimensions.asArrayRef()))
    if ((extent < 0 && !ShapedType::isDynamic(extent)) || dimension < 0 ||
        (ShapedType::isDynamic(extent) && dimension == 0))
      return error() << "invalid DSA extent or dimension identity";
  return success();
}
LogicalResult ScalarArgumentAttr::verify(function_ref<InFlightDiagnostic()> error,
    StringAttr name, Type type) {
  if (!name || name.empty() || (!type.isF32() && !type.isIndex() && !type.isInteger(64) && !type.isInteger(32) && !type.isInteger(1)))
    return error() << "DSA scalar ABI requires a name and a supported scalar type";
  return success();
}
LogicalResult InterfaceAttr::verify(function_ref<InFlightDiagnostic()> error, ArrayAttr arguments) {
  if (!arguments) return error() << "DSA interface requires arguments";
  llvm::DenseSet<StringAttr> names;
  for (Attribute attribute : arguments) {
    StringAttr name;
    if (auto view = mlir::dyn_cast<ViewArgumentAttr>(attribute)) name = view.getName();
    else if (auto scalar = mlir::dyn_cast<ScalarArgumentAttr>(attribute)) name = scalar.getName();
    else return error() << "DSA interface contains an untyped argument";
    if (!names.insert(name).second) return error() << "DSA argument names must be unique";
  }
  return success();
}
LogicalResult ConfigurationAttr::verify(function_ref<InFlightDiagnostic()> error,
    int64_t tile, int64_t m, int64_t n, int64_t k, int64_t tasks, int64_t bytes) {
  if (tile <= 0 || tile % 64 || m <= 0 || n <= 0 || k <= 0 || tasks <= 0 || bytes <= 0)
    return error() << "DSA configuration requires positive blocks/resources and a vector tile divisible by 64";
  return success();
}
