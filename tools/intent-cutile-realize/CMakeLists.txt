get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(intent-cutile-realize
  intent-cutile-realize.cpp
)

llvm_update_compile_flags(intent-cutile-realize)
target_link_libraries(intent-cutile-realize PRIVATE
  ${dialect_libs}
  MLIRIntentCuTileDialect
  MLIRIntentCuTileRealization
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTransforms
  MLIRParser
  MLIRIR
  MLIRSupport
)

mlir_check_all_link_libraries(intent-cutile-realize)
