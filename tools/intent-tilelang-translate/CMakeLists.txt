get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(intent-tilelang-translate
  intent-tilelang-translate.cpp
)

llvm_update_compile_flags(intent-tilelang-translate)
target_link_libraries(intent-tilelang-translate PRIVATE
  ${dialect_libs}
  MLIRIntentTileLangDialect
  MLIRIntentTileLangEmission
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTransforms
  MLIRParser
  MLIRIR
  MLIRSupport
)

mlir_check_all_link_libraries(intent-tilelang-translate)
