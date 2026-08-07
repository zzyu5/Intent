set(LLVM_TARGET_DEFINITIONS IntentDialect.td)
mlir_tablegen(IntentDialect.h.inc -gen-dialect-decls -dialect=intent)
mlir_tablegen(IntentDialect.cpp.inc -gen-dialect-defs -dialect=intent)

set(LLVM_TARGET_DEFINITIONS IntentTypes.td)
mlir_tablegen(IntentTypes.h.inc -gen-typedef-decls)
mlir_tablegen(IntentTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS IntentOps.td)
mlir_tablegen(IntentOps.h.inc -gen-op-decls)
mlir_tablegen(IntentOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(IntentIncGen)
