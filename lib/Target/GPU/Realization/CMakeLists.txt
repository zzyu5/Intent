add_mlir_library(MLIRIntentGPURealization
  Analysis/Operations.cpp
  Driver/Construct.cpp
  Plan/Build.cpp
  Plan/Decisions.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTargetCommon
  MLIRIntentTransforms
  MLIRSupport
)

target_include_directories(MLIRIntentGPURealization PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
