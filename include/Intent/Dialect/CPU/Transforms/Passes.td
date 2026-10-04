#ifndef INTENT_DIALECT_CPU_TRANSFORMS_PASSES_TD
#define INTENT_DIALECT_CPU_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

class CPUTransform<string argument, string summaryText> : Pass<argument, "mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = [
    "intent::cpu::IntentCPUDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::math::MathDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::vector::VectorDialect", "mlir::tensor::TensorDialect",
    "mlir::bufferization::BufferizationDialect"
  ];
}

def CPUProviderOption : Option<"provider", "provider", "std::string", "\"\"",
    "Compiled provider whose implementations this pass consumes">;

def CPUConfigureTarget : CPUTransform<"intent-cpu-configure-target",
    "Bind CPU hardware capabilities into the current program"> {
  let description = [{
    Accepts the constructed CPU value program and binds typed hardware
    capabilities on its module. Checks the CPU Values-stage contract; it does
    not select implementations or allocate storage.
  }];
  let options = [
    Option<"vectorBits", "vector-bits", "int64_t", "0", "Hardware vector width in bits">,
    Option<"workers", "workers", "int64_t", "0", "Available CPU workers">,
    Option<"privateBytes", "private-bytes", "int64_t", "262144", "Private storage budget in bytes">,
    Option<"matrixI8I32", "matrix-i8-i32", "bool", "false", "Signed i8 to i32 matrix capability">
  ];
}

def CPUBufferizeValues : CPUTransform<"intent-cpu-bufferize-values",
    "Optimize CPU tensor SSA and materialize owned storage with native bufferization"> {
  let description = [{
    Converts a valid CPU Values-stage program to the Buffers stage using native
    bufferization and ownership. Tensor values become explicit storage while
    structured computations and logical effects remain in the current IR.
  }];
}

def CPURealizeSliceCollectives : CPUTransform<"intent-cpu-realize-slice-collectives",
    "Realize tensor slice reductions and scans as buffer computations"> {
  let description = [{
    Requires and preserves the CPU Buffers contract. Realizes slice collective
    helper schemas and their traversal without selecting implementations.
  }];
}

def CPUNormalizeContractions : CPUTransform<"intent-cpu-normalize-contractions",
    "Normalize contraction source representations and iteration maps"> {
  let description = [{
    Requires a CPU buffer program before implementation selection. Closes
    source representation normalization and contraction canonicalization in
    one pass, producing ordinary current-IR computations for candidate binding.
  }];
}

def CPUFoldUniformComputations : CPUTransform<"intent-cpu-fold-uniform-computations",
    "Fold computations over proven uniform storage values"> {
  let description = [{
    Requires and preserves the CPU Buffers contract. Uses current storage
    effects and uniform-value facts to fold computations and copies; it does
    not require provider options or a selected configuration.
  }];
}

def CPUFuseStructuredComputations : CPUTransform<"intent-cpu-fuse-structured-computations",
    "Fuse compatible structured computation traversals"> {
  let description = [{
    Requires and preserves the CPU Buffers contract. Fuses compatible
    structured computations using their current indexing, storage and effect
    relations. Candidate selection and target expansion remain separate.
  }];
}

def CPUNormalizeReductions : CPUTransform<"intent-cpu-normalize-reductions",
    "Normalize scalar reductions into structured CPU computations"> {
  let description = [{
    Requires a CPU buffer program before implementation selection. Normalizes
    eligible scalar reductions to closed structured computations while keeping
    unsupported reduction forms intact for their existing realization path.
  }];
}

def CPUMaterializeConfigurations : CPUTransform<"intent-cpu-materialize-configurations",
    "Bind complete CPU implementation candidates into the current program"> {
  let description = [{
    Requires one executable CPU buffer function and a registered provider.
    Clones the finite candidate functions and binds complete typed
    configurations and implementations on the current IR. No candidate state
    is retained outside those functions.
  }];
  let options = [CPUProviderOption,
    Option<"defaults", "defaults", "std::string", "\"\"", "Provider default tuning profile file">,
    Option<"overrides", "overrides", "std::string", "\"\"", "Optional tuning profile overrides file">
  ];
}

def CPUGroupRegions : CPUTransform<"intent-cpu-group-regions",
    "Group independent region computations using the current configuration"> {
  let description = [{
    Requires a CPU buffer program with a configuration on each function.
    Groups independent region computations and closes their storage and helper
    schemas. It preserves structured regions for the realization pass.
  }];
}

def CPURealizeRegions : CPUTransform<"intent-cpu-realize-regions",
    "Realize configured region traversals and state transitions"> {
  let description = [{
    Requires configured CPU buffer functions with valid bindings from the
    selected provider. Realizes region traversal and state transitions using
    those bindings, leaving valid buffer computations for subsequent passes.
  }];
  let options = [CPUProviderOption];
}

def CPUFoldContractionInputs : CPUTransform<"intent-cpu-fold-contraction-inputs",
    "Fold legal source conversions into selected contraction implementations"> {
  let description = [{
    Requires the CPU Buffers contract and selected implementation bindings.
    Folds eligible input conversions only when accepted by the registered
    implementation, preserving the current computation's numeric contract.
  }];
  let options = [CPUProviderOption];
}

def CPURealizeHistograms : CPUTransform<"intent-cpu-realize-histograms",
    "Realize histogram updates as CPU storage computations"> {
  let description = [{
    Requires and preserves the CPU Buffers contract. Histograms outside an
    existing parallel scope also require target capabilities and a bound
    configuration: workers and task grain determine their partial groups.
    Materializes partial storage, grouped traversal, initialization and valid
    updates without selecting provider implementations.
  }];
}

def CPUReusePrivateStorage : CPUTransform<"intent-cpu-reuse-private-storage",
    "Reuse private allocations with compatible storage lifetimes"> {
  let description = [{
    Requires and preserves the CPU Buffers contract. Reuses private allocations
    when their current aliases, lifetimes and initialization allow it, and
    removes storage made dead by that rewrite.
  }];
}

def CPUReuseScratchStorage : CPUTransform<"intent-cpu-reuse-scratch-storage",
    "Reuse bounded scratch capacity within its sequential execution owner"> {
  let description = [{
    Requires and preserves the CPU Buffers contract. Reuses nonescaping scratch
    slots with disjoint lifetimes and places storage outside serial traversals.
    Preserves the existing descriptor representation. Initialization and
    accesses retain their original locations; parallel and task allocation
    owners are not crossed.
  }];
}

def CPUPrepareInputs : CPUTransform<"intent-cpu-prepare-inputs",
    "Share prepared input representations required by selected implementations"> {
  let description = [{
    Requires configured CPU buffer functions with selected implementations.
    Forms reusable prepared inputs and grouped quantized input representations
    from those implementations' requirements. Resulting storage and consumers
    are explicit IR, not a deferred preparation plan.
  }];
  let options = [CPUProviderOption];
}

def CPUGroupWorksets : CPUTransform<"intent-cpu-group-worksets",
    "Group structured worksets for selected implementation requirements"> {
  let description = [{
    Requires the CPU Buffers contract and selected implementation bindings.
    Groups independent rows according to the current implementation's workset
    requirements, preserving explicit structured computations.
  }];
  let options = [CPUProviderOption];
}

def CPUBlockComputations : CPUTransform<"intent-cpu-block-computations",
    "Block configured contractions and structured computations"> {
  let description = [{
    Requires configured CPU buffer functions and valid implementation bindings.
    Materializes contraction and structured-computation blocking together,
    consuming selected tile and input-supply requirements in the current IR.
  }];
  let options = [CPUProviderOption];
}

def CPUPartitionTasks : CPUTransform<"intent-cpu-partition-tasks",
    "Partition configured CPU work into independent task bodies"> {
  let description = [{
    Requires configured CPU buffer functions with selected implementations.
    Partitions independent work using current task grain and computation
    relations. Task bodies remain valid with lexical captures until the
    separate capture-isolation pass makes their arguments explicit.
  }];
  let options = [CPUProviderOption];
}

def CPUIsolateTasks : CPUTransform<"intent-cpu-isolate-tasks",
    "Make task captures and their region arguments explicit"> {
  let description = [{
    Requires a CPU buffer program containing formed task bodies. Converts
    lexical captures to explicit operands and corresponding region arguments,
    preserving the buffer contract without selecting a provider.
  }];
}

#endif
