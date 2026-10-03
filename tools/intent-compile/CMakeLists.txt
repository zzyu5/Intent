add_llvm_executable(intent-compile
  intent-compile.cpp
)

llvm_update_compile_flags(intent-compile)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/profiles")
configure_file("${PROJECT_SOURCE_DIR}/lib/Dialect/GPU/Transforms/Configuration/TuningProfiles.json"
               "${CMAKE_CURRENT_BINARY_DIR}/profiles/shared.json" COPYONLY)
set(intent_profile_files "${CMAKE_CURRENT_BINARY_DIR}/profiles/shared.json")
foreach(provider Triton CuTile Mojo Weft)
  string(TOLOWER "${provider}" provider_name)
  configure_file("${PROJECT_SOURCE_DIR}/lib/Target/${provider}/Transforms/TuningProfiles.json"
                 "${CMAKE_CURRENT_BINARY_DIR}/profiles/${provider_name}.json" COPYONLY)
  list(APPEND intent_profile_files
       "${CMAKE_CURRENT_BINARY_DIR}/profiles/${provider_name}.json")
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
install(FILES ${intent_profile_files}
  DESTINATION "${INTENT_RUNTIME_INSTALL_DIR}/profiles"
  COMPONENT IntentRuntime)
target_link_libraries(intent-compile PRIVATE
  MLIRIntentCompiler
  MLIRParser
)

mlir_check_all_link_libraries(intent-compile)
