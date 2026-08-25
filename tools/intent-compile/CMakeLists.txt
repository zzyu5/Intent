get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(intent-compile
  intent-compile.cpp
)

llvm_update_compile_flags(intent-compile)
target_link_libraries(intent-compile PRIVATE
  ${dialect_libs}
  MLIRIntentDialect
  MLIRIntentKIRToGPU
  MLIRIntentGPUDialect
  MLIRIntentGPUTransforms
  MLIRIntentTritonTransforms
  MLIRIntentTritonSerialization
  MLIRIntentTransforms
  MLIRParser
  MLIRIR
  MLIRSupport
)

mlir_check_all_link_libraries(intent-compile)
