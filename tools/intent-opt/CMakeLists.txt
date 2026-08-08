get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(intent-opt
  intent-opt.cpp
)

llvm_update_compile_flags(intent-opt)
target_link_libraries(intent-opt PRIVATE
  ${dialect_libs}
  MLIRIntentTransforms
  MLIROptLib
)

mlir_check_all_link_libraries(intent-opt)
