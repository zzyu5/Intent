#ifndef INTENT_DIALECT_PLAN_IR_PLANDIALECT_TD
#define INTENT_DIALECT_PLAN_IR_PLANDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentPlan_Dialect : Dialect {
  let name = "intent_plan";
  let cppNamespace = "::intent::plan";
  let summary = "Target-neutral realization envelopes for Intent Kernel IR";
  let description = [{
    IntentPlan separates a resolved realization, which target emission may
    consume, from a search space, which realization may explore. Concrete
    physical choices live in the selected target dialect.
  }];
}

#endif
