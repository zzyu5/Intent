get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(intent-tilelang-realize
  intent-tilelang-realize.cpp
)

llvm_update_compile_flags(intent-tilelang-realize)
target_link_libraries(intent-tilelang-realize PRIVATE
  ${dialect_libs}
  MLIRIntentTileLangDialect
  MLIRIntentTileLangRealization
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTransforms
  MLIRParser
  MLIRIR
  MLIRSupport
)

mlir_check_all_link_libraries(intent-tilelang-realize)
