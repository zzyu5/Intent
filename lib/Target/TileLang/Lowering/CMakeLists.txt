add_mlir_library(MLIRIntentTileLangLowering
  Driver/TargetProgram.cpp
  Handlers/Operations.cpp
  Materialization/Program.cpp
  Syntax/Spelling.cpp
  Transforms/ProgramForms.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentGPUTransforms
  MLIRIntentTargetCommon
  MLIRIntentTransforms
  MLIRSupport
)

target_include_directories(MLIRIntentTileLangLowering PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
