#include "Intent/Dialect/Plan/IR/PlanDialect.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"

using namespace mlir;
using namespace intent::plan;

#define GET_DIALECT_DEF
#include "Intent/Dialect/Plan/IR/PlanDialect.cpp.inc"

void IntentPlanDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Intent/Dialect/Plan/IR/PlanOps.cpp.inc"
      >();
}
