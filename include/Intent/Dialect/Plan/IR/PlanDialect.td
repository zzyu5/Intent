#ifndef INTENT_DIALECT_PLAN_IR_PLANDIALECT_TD
#define INTENT_DIALECT_PLAN_IR_PLANDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentPlan_Dialect : Dialect {
  let name = "intent_plan";
  let cppNamespace = "::intent::plan";
  let summary = "Machine realization envelopes for Intent Kernel IR";
  let description = [{
    IntentPlan separates one resolved machine realization from the legal axes
    of a search space. GPU machine decisions are shared here; a surface target
    dialect only projects those decisions and records what it delegates.
  }];
}

#endif
