#include "Intent/Target/Triton/IR/TritonDialect.h"
#include "Intent/Target/Triton/IR/TritonOps.h"

using namespace mlir;
using namespace intent::triton::plan;

#define GET_DIALECT_DEF
#include "Intent/Target/Triton/IR/TritonDialect.cpp.inc"

void IntentTritonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Intent/Target/Triton/IR/TritonOps.cpp.inc"
      >();
}
