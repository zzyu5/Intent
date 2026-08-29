#ifndef INTENT_TARGET_TILELANG_IR_TILELANGATTRS_TD
#define INTENT_TARGET_TILELANG_IR_TILELANGATTRS_TD

include "Intent/Target/TileLang/IR/TileLangDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class IntentTileLang_I32Enum<string name, string description,
                             list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::intent::tilelang";
}

class IntentTileLang_I32EnumAttr<string mnemonic,
                                 IntentTileLang_I32Enum enumInfo>
    : EnumAttr<IntentTileLang_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def IntentTileLang_BufferSpace : IntentTileLang_I32Enum<
    "BufferSpace", "Selected TileLang native allocation space", [
      I32EnumAttrCase<"Shared", 0, "shared">,
      I32EnumAttrCase<"Fragment", 1, "fragment">
    ]>;
def IntentTileLang_BufferSpaceAttr : IntentTileLang_I32EnumAttr<
    "buffer_space", IntentTileLang_BufferSpace>;

#endif
