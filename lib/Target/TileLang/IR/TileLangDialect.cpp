#include "Intent/Target/TileLang/IR/TileLangDialect.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"
#include "Intent/Target/TileLang/IR/TileLangTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#define GET_DIALECT_DEF
#include "Intent/Target/TileLang/IR/TileLangDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Intent/Target/TileLang/IR/TileLangTypes.cpp.inc"

namespace intent::tilelang {

void IntentTileLangDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Intent/Target/TileLang/IR/TileLangTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Intent/Target/TileLang/IR/TileLangOps.cpp.inc"
      >();
}

LogicalResult BufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 Type elementType, ArrayAttr shape,
                                 uint32_t space) {
  if (!elementType || !shape || shape.empty() || space > 1)
    return emitError()
           << "TileLang buffer requires element type, non-empty shape and shared/fragment space";
  for (Attribute extent : shape)
    if (!mlir::isa<gpu::PhysicalExprAttr>(extent))
      return emitError() << "TileLang buffer extents must be typed physical expressions";
  return success();
}

} // namespace intent::tilelang
