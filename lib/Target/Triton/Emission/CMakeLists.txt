add_mlir_library(MLIRIntentTritonEmission
  Driver/Translate.cpp
  Handlers/Operations.cpp
  Source/Emitter.cpp

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

target_include_directories(MLIRIntentTritonEmission PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
