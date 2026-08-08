#ifndef INTENT_TARGET_CUTILE_IR_CUTILEDIALECT_TD
#define INTENT_TARGET_CUTILE_IR_CUTILEDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentCuTile_Dialect : Dialect {
  let name = "intent_cutile";
  let cppNamespace = "::intent::cutile::plan";
  let summary = "Resolved cuTile mechanisms for an Intent realization";
  let description = [{
    IntentCuTile records concrete cuda.tile scheduling, memory-access,
    primitive and launch choices. These operations are not Kernel IR semantics.
  }];
}

#endif
