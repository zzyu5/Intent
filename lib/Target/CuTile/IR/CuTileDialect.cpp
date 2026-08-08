#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"

using namespace mlir;
using namespace intent::cutile::plan;

#define GET_DIALECT_DEF
#include "Intent/Target/CuTile/IR/CuTileDialect.cpp.inc"

void IntentCuTileDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Intent/Target/CuTile/IR/CuTileOps.cpp.inc"
      >();
}
