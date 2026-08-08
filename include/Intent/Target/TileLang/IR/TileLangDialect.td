#ifndef INTENT_TARGET_TILELANG_IR_TILELANGDIALECT_TD
#define INTENT_TARGET_TILELANG_IR_TILELANGDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentTileLang_Dialect : Dialect {
  let name = "intent_tilelang";
  let cppNamespace = "::intent::tilelang::plan";
  let summary = "TileLang projection of a shared GPU realization";
  let description = [{
    IntentTileLang spells shared GPU decisions, including explicit on-chip
    buffer allocation, and records the choices delegated to TileLang.
  }];
}

#endif
