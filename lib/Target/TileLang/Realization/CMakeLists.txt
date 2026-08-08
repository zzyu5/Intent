add_mlir_library(MLIRIntentTileLangRealization
  Driver/Realize.cpp
  Handlers/Operations.cpp
  Policy/Decisions.cpp
  Search/Candidates.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentTileLangDialect
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTargetCommon
  MLIRIntentTransforms
  MLIRSupport
)

target_include_directories(MLIRIntentTileLangRealization PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
