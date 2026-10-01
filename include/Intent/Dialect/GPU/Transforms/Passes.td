include "mlir/Pass/PassBase.td"

class IntentGPUTransformPass<string passArgument, string summaryText>
    : Pass<passArgument, "::mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = ["::intent::gpu::IntentGPUDialect",
      "::mlir::arith::ArithDialect", "::mlir::cf::ControlFlowDialect",
      "::mlir::func::FuncDialect", "::mlir::scf::SCFDialect"];
}

def NormalizeStructuredSourcesPass : IntentGPUTransformPass<
    "intent-gpu-normalize-structured-sources",
    "Normalize structured GPU sources">;

def PredicateScalarControlPass : IntentGPUTransformPass<
    "intent-gpu-predicate-scalar-control",
    "Predicate scalar control in the GPU program">;

def PointwiseOwnershipPass : IntentGPUTransformPass<
    "intent-gpu-form-pointwise-ownership",
    "Form pointwise GPU ownership">;

def PointwiseBlockingPass : IntentGPUTransformPass<
    "intent-gpu-form-pointwise-blocking",
    "Form pointwise GPU blocking and its dependent value graph">;

def ScanConsumersPass : IntentGPUTransformPass<
    "intent-gpu-realize-scan-consumers",
    "Realize GPU scan consumer traversals">;

def RetainedValuesPass : IntentGPUTransformPass<
    "intent-gpu-materialize-retained-values",
    "Materialize retained GPU values">;

def OnlineReductionsPass : IntentGPUTransformPass<
    "intent-gpu-co-realize-online-reductions",
    "Co-realize GPU online reductions">;

def RegionFoldsPass : IntentGPUTransformPass<
    "intent-gpu-realize-region-folds",
    "Realize GPU region folds">;

def RegionScansPass : IntentGPUTransformPass<
    "intent-gpu-realize-region-scans",
    "Realize GPU region scans">;

def ReductionsPass : IntentGPUTransformPass<
    "intent-gpu-realize-reductions",
    "Realize GPU reduction programs">;

def ContractionsPass : IntentGPUTransformPass<
    "intent-gpu-realize-contractions",
    "Realize GPU contraction programs">;

def AccessCompositionPass : IntentGPUTransformPass<
    "intent-gpu-compose-realized-accesses",
    "Compose realized GPU accesses and values">;

def PrivateStoresPass : IntentGPUTransformPass<
    "intent-gpu-schedule-private-stores",
    "Schedule private GPU stores">;

def BufferVectorizationPass : IntentGPUTransformPass<
    "intent-gpu-vectorize-buffer-loops",
    "Vectorize GPU buffer loops">;

def BufferPromotionPass : IntentGPUTransformPass<
    "intent-gpu-promote-buffer-values",
    "Promote GPU buffer values">;

def ProgramMappingPass : IntentGPUTransformPass<
    "intent-gpu-refine-program-mapping",
    "Refine GPU program mapping">;

def RangePredicatesPass : IntentGPUTransformPass<
    "intent-gpu-simplify-range-predicates",
    "Remove predicates proven by complete physical tiles">;

def CommonValuesPass : IntentGPUTransformPass<
    "intent-gpu-eliminate-common-values",
    "Eliminate common GPU values and simplify masked coordinates">;

def RealizeSharedProgramPass : IntentGPUTransformPass<
    "intent-gpu-realize-shared-program",
    "Realize the complete shared GPU program including uniform branch dispatch">;

def MaterializeConfigurationsPass : IntentGPUTransformPass<
    "intent-gpu-materialize-configurations",
    "Bind shared configurations from resolved current-module profiles">;

def FuseIndependentTraversalsPass : IntentGPUTransformPass<
    "intent-gpu-fuse-independent-traversals",
    "Fuse realized traversals and verify their closed configurations">;
