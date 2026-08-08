#ifndef INTENT_TARGET_TRITON_IR_TRITONDIALECT_TD
#define INTENT_TARGET_TRITON_IR_TRITONDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentTriton_Dialect : Dialect {
  let name = "intent_triton";
  let cppNamespace = "::intent::triton::plan";
  let summary = "Resolved Triton mechanisms for an Intent realization";
  let description = [{
    IntentTriton records concrete target choices selected by the Triton
    realizer. These operations are not logical Kernel IR semantics.
  }];
}

#endif
