set(LLVM_TARGET_DEFINITIONS TileLangDialect.td)
mlir_tablegen(TileLangDialect.h.inc -gen-dialect-decls -dialect=intent_tilelang)
mlir_tablegen(TileLangDialect.cpp.inc -gen-dialect-defs -dialect=intent_tilelang)

set(LLVM_TARGET_DEFINITIONS TileLangOps.td)
mlir_tablegen(TileLangOps.h.inc -gen-op-decls)
mlir_tablegen(TileLangOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(IntentTileLangIncGen)
