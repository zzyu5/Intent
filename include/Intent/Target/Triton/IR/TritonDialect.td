#ifndef INTENT_TARGET_TRITON_IR_TRITONDIALECT_TD
#define INTENT_TARGET_TRITON_IR_TRITONDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentTriton_Dialect : Dialect {
  let name = "intent_triton";
  let cppNamespace = "::intent::triton::plan";
  let summary = "Triton projection of a shared GPU realization";
  let description = [{
    IntentTriton records Triton spellings for shared GPU decisions, the
    surface capability cut, and explicit delegation to the Triton compiler.
  }];
}

#endif
