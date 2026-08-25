set(LLVM_TARGET_DEFINITIONS GPUDialect.td)
mlir_tablegen(GPUDialect.h.inc -gen-dialect-decls -dialect=intent_gpu)
mlir_tablegen(GPUDialect.cpp.inc -gen-dialect-defs -dialect=intent_gpu)

set(LLVM_TARGET_DEFINITIONS GPUAttrs.td)
mlir_tablegen(GPUAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(GPUAttrs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS GPUTypes.td)
mlir_tablegen(GPUTypes.h.inc -gen-typedef-decls)
mlir_tablegen(GPUTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS GPUOps.td)
mlir_tablegen(GPUOps.h.inc -gen-op-decls)
mlir_tablegen(GPUOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(IntentGPUIncGen)
