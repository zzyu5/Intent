#include "Intent/Target/TileLang/IR/TileLangDialect.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"

using namespace mlir;
using namespace intent::tilelang::plan;

#define GET_DIALECT_DEF
#include "Intent/Target/TileLang/IR/TileLangDialect.cpp.inc"

void IntentTileLangDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Intent/Target/TileLang/IR/TileLangOps.cpp.inc"
      >();
}
