#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace intent;

#define GET_DIALECT_DEF
#include "Intent/Dialect/Intent/IR/IntentDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Intent/Dialect/Intent/IR/IntentTypes.cpp.inc"

void IntentDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Intent/Dialect/Intent/IR/IntentTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Intent/Dialect/Intent/IR/IntentOps.cpp.inc"
      >();
}
