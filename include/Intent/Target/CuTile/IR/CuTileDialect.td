#ifndef INTENT_TARGET_CUTILE_IR_CUTILEDIALECT_TD
#define INTENT_TARGET_CUTILE_IR_CUTILEDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentCuTile_Dialect : Dialect {
  let name = "intent_cutile";
  let cppNamespace = "::intent::cutile::plan";
  let summary = "cuTile projection of a shared GPU realization";
  let description = [{
    IntentCuTile records cuda.tile spellings for shared GPU decisions, the
    surface capability cut, and explicit delegation to the cuda.tile compiler.
  }];
}

#endif
