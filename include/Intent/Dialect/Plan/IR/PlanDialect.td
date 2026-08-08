#ifndef INTENT_DIALECT_PLAN_IR_PLANDIALECT_TD
#define INTENT_DIALECT_PLAN_IR_PLANDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentPlan_Dialect : Dialect {
  let name = "intent_plan";
  let cppNamespace = "::intent::plan";
  let summary = "Compiler-owned physical realization of Intent Kernel IR";
  let description = [{
    IntentPlan binds logical Kernel IR nodes to physical extents, ownership,
    storage, layouts, target primitives, boundaries, pipelines and launch.
    It never changes the source-visible algorithm or ABI.
  }];
}

#endif
