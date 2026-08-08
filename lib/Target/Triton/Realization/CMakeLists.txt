add_mlir_library(MLIRIntentTritonRealization
  Driver/Realize.cpp
  Handlers/Operations.cpp
  Policy/Decisions.cpp
  Proof/Semantics.cpp
  Search/Candidates.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTransforms
  MLIRIntentTritonDialect
  MLIRIntentTargetCommon
  MLIRSupport
)

target_include_directories(MLIRIntentTritonRealization PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
