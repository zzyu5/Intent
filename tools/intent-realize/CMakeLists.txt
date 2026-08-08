get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(intent-realize
  intent-realize.cpp
)

llvm_update_compile_flags(intent-realize)
target_link_libraries(intent-realize PRIVATE
  ${dialect_libs}
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTransforms
  MLIRParser
  MLIRIR
  MLIRSupport
)

mlir_check_all_link_libraries(intent-realize)
