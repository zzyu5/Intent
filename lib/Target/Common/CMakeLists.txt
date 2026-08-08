add_mlir_library(MLIRIntentTargetCommon
  Analysis/IndexRelation.cpp
  Analysis/Kernel.cpp
  Realization/KernelFacts.cpp
  Realization/Proofs.cpp
  Traversal/OperationRegistry.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)
