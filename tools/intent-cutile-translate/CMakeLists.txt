get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(intent-cutile-translate
  intent-cutile-translate.cpp
)

llvm_update_compile_flags(intent-cutile-translate)
target_link_libraries(intent-cutile-translate PRIVATE
  ${dialect_libs}
  MLIRIntentCuTileDialect
  MLIRIntentCuTileEmission
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTransforms
  MLIRParser
  MLIRIR
  MLIRSupport
)

mlir_check_all_link_libraries(intent-cutile-translate)
