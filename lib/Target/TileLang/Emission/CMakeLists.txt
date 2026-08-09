add_mlir_library(MLIRIntentTileLangEmission
  Driver/Translate.cpp
  Handlers/Operations.cpp
  Source/Emitter.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTargetCommon
  MLIRIntentTransforms
  MLIRSupport
)

target_include_directories(MLIRIntentTileLangEmission PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
