add_mlir_library(MLIRIntentTritonLowering
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
  MLIRIntentTransforms
  MLIRIntentTargetCommon
  MLIRSupport
)

target_include_directories(MLIRIntentTritonLowering PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
