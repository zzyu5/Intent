#ifndef INTENT_TARGET_TILELANG_IR_TILELANGDIALECT_TD
#define INTENT_TARGET_TILELANG_IR_TILELANGDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentTileLang_Dialect : Dialect {
  let name = "intent_tilelang";
  let cppNamespace = "::intent::tilelang";
  let summary = "TileLang-local storage, transfer and native-compute forms";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif
