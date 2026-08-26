#ifndef INTENT_TARGET_TILELANG_IR_TILELANGTYPES_TD
#define INTENT_TARGET_TILELANG_IR_TILELANGTYPES_TD

include "Intent/Target/TileLang/IR/TileLangDialect.td"
include "mlir/IR/AttrTypeBase.td"

def IntentTileLang_BufferType : TypeDef<IntentTileLang_Dialect, "Buffer"> {
  let mnemonic = "buffer";
  let summary = "A selected TileLang shared or fragment allocation";
  let parameters = (ins
    "::mlir::Type":$elementType,
    "::mlir::ArrayAttr":$shape,
    "uint32_t":$space
  );
  let assemblyFormat = "`<` $elementType `,` $shape `,` $space `>`";
  let genVerifyDecl = 1;
}

def IntentTileLang_BufferValue : DialectType<IntentTileLang_Dialect,
    CPred<"::llvm::isa<::intent::tilelang::BufferType>($_self)">,
    "TileLang physical buffer", "::intent::tilelang::BufferType">;

#endif
