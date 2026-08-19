#ifndef INTENT_DIALECT_PLAN_IR_PLANDIALECT_TD
#define INTENT_DIALECT_PLAN_IR_PLANDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentPlan_Dialect : Dialect {
  let name = "intent_plan";
  let cppNamespace = "::intent::plan";
  let summary = "Executable physical programs for Intent Kernel IR";
  let description = [{
    IntentPlan owns the current executable physical program, its selected GPU
    decisions, the legal parameter search space, and the provider program
    produced by target materialization. Once constructed, this program is the
    only executable authority consumed by later lowering stages.
  }];
}

#endif
