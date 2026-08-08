add_mlir_library(MLIRIntentCuTileRealization
  Driver/Realize.cpp
  Handlers/Operations.cpp
  Policy/Decisions.cpp
  Search/Candidates.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentCuTileDialect
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTargetCommon
  MLIRIntentTransforms
  MLIRSupport
)

target_include_directories(MLIRIntentCuTileRealization PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
