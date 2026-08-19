add_mlir_library(MLIRIntentCuTileLowering
  Driver/TargetProgram.cpp
  Handlers/Operations.cpp
  Materialization/Program.cpp
  Syntax/Spelling.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTargetCommon
  MLIRIntentTransforms
  MLIRSupport
)

target_include_directories(MLIRIntentCuTileLowering PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
