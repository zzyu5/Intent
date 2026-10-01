add_llvm_executable(intent-compile
  intent-compile.cpp
)

llvm_update_compile_flags(intent-compile)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/profiles")
configure_file("${PROJECT_SOURCE_DIR}/lib/Dialect/GPU/Transforms/TuningProfiles.json"
               "${CMAKE_CURRENT_BINARY_DIR}/profiles/shared.json" COPYONLY)
foreach(provider Triton CuTile TileLang Mojo Weft)
  string(TOLOWER "${provider}" provider_name)
  configure_file("${PROJECT_SOURCE_DIR}/lib/Target/${provider}/Transforms/TuningProfiles.json"
                 "${CMAKE_CURRENT_BINARY_DIR}/profiles/${provider_name}.json" COPYONLY)
endforeach()
if(SKBUILD)
  set_target_properties(intent-compile PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
else()
  set_target_properties(intent-compile PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()
install(TARGETS intent-compile
  RUNTIME DESTINATION "${INTENT_RUNTIME_INSTALL_DIR}"
  COMPONENT IntentRuntime)
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/profiles/"
  DESTINATION "${INTENT_RUNTIME_INSTALL_DIR}/profiles"
  COMPONENT IntentRuntime
  FILES_MATCHING PATTERN "*.json")
target_link_libraries(intent-compile PRIVATE
  MLIRIntentCompiler
  MLIRParser
)

mlir_check_all_link_libraries(intent-compile)
