get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(intent-translate
  intent-translate.cpp
)

llvm_update_compile_flags(intent-translate)
target_link_libraries(intent-translate PRIVATE
  ${dialect_libs}
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTritonTarget
  MLIRIntentTransforms
  MLIRParser
  MLIRIR
  MLIRSupport
)

mlir_check_all_link_libraries(intent-translate)
