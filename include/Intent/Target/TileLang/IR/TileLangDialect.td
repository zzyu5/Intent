#ifndef INTENT_TARGET_TILELANG_IR_TILELANGDIALECT_TD
#define INTENT_TARGET_TILELANG_IR_TILELANGDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentTileLang_Dialect : Dialect {
  let name = "intent_tilelang";
  let cppNamespace = "::intent::tilelang::plan";
  let summary = "Resolved TileLang mechanisms for an Intent realization";
  let description = [{
    IntentTileLang records concrete TileLang scheduling, explicit buffer spaces,
    layouts, primitive, pipeline and launch choices. These operations are not
    Kernel IR semantics.
  }];
}

#endif
