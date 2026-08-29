set(LLVM_TARGET_DEFINITIONS TileLangDialect.td)
mlir_tablegen(TileLangDialect.h.inc -gen-dialect-decls -dialect=intent_tilelang)
mlir_tablegen(TileLangDialect.cpp.inc -gen-dialect-defs -dialect=intent_tilelang)

set(LLVM_TARGET_DEFINITIONS TileLangAttrs.td)
mlir_tablegen(TileLangAttrsEnums.h.inc -gen-enum-decls)
mlir_tablegen(TileLangAttrsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TileLangAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(TileLangAttrs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS TileLangTypes.td)
mlir_tablegen(TileLangTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TileLangTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS TileLangOps.td)
mlir_tablegen(TileLangOps.h.inc -gen-op-decls)
mlir_tablegen(TileLangOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(IntentTileLangIncGen)
