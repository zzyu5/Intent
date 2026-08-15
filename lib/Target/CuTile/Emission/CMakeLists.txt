add_mlir_library(MLIRIntentCuTileEmission
  Driver/Translate.cpp
  Handlers/Operations.cpp
  Source/Emitter.cpp
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

target_include_directories(MLIRIntentCuTileEmission PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
