get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(intent-compile
  intent-compile.cpp
)

llvm_update_compile_flags(intent-compile)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/profiles")
configure_file("${PROJECT_SOURCE_DIR}/lib/Dialect/CPU/Transforms/TuningProfiles.json"
               "${CMAKE_CURRENT_BINARY_DIR}/profiles/cpu.json" COPYONLY)
configure_file("${PROJECT_SOURCE_DIR}/lib/Dialect/GPU/Transforms/TuningProfiles.json"
               "${CMAKE_CURRENT_BINARY_DIR}/profiles/shared.json" COPYONLY)
foreach(provider Triton CuTile TileLang)
  string(TOLOWER "${provider}" provider_name)
  configure_file("${PROJECT_SOURCE_DIR}/lib/Target/${provider}/Transforms/TuningProfiles.json"
                 "${CMAKE_CURRENT_BINARY_DIR}/profiles/${provider_name}.json" COPYONLY)
endforeach()
target_link_libraries(intent-compile PRIVATE
  ${dialect_libs}
  MLIRIntentDialect
  MLIRIntentKIRToGPU
  MLIRIntentKIRToCPU
  MLIRIntentCPUTransforms
  MLIRIntentCPUDialect
  MLIRIntentMojoSerialization
  MLIRIntentMojoTransforms
  MLIRIntentCuTileDialect
  MLIRIntentCuTileTransforms
  MLIRIntentCuTileSerialization
  MLIRIntentGPUDialect
  MLIRIntentGPUTransforms
  MLIRIntentTileLangDialect
  MLIRIntentTileLangTransforms
  MLIRIntentTileLangSerialization
  MLIRIntentTritonDialect
  MLIRIntentTritonTransforms
  MLIRIntentTritonSerialization
  MLIRIntentTransforms
  MLIRParser
  MLIRIR
  MLIRSupport
)

mlir_check_all_link_libraries(intent-compile)
