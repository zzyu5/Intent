#ifndef INTENT_DIALECT_CPU_TRANSFORMS_PASSES_TD
#define INTENT_DIALECT_CPU_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

class CPUTransform<string argument, string summaryText> : Pass<argument, "mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = [
    "intent::cpu::IntentCPUDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::math::MathDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::vector::VectorDialect"
  ];
}

def CPUProviderOption : Option<"provider", "provider", "std::string", "\"\"",
    "Compiled provider whose implementations this pass consumes">;

def CPUConfigureTarget : CPUTransform<"intent-cpu-configure-target",
    "Bind CPU hardware capabilities into the current program"> {
  let options = [
    Option<"vectorBits", "vector-bits", "int64_t", "0", "Hardware vector width in bits">,
    Option<"workers", "workers", "int64_t", "0", "Available CPU workers">,
    Option<"privateBytes", "private-bytes", "int64_t", "262144", "Private storage budget in bytes">,
    Option<"matrixI8I32", "matrix-i8-i32", "bool", "false", "Signed i8 to i32 matrix capability">
  ];
}

def CPUNormalizeSource : CPUTransform<"intent-cpu-normalize-source",
    "Normalize source computations before CPU implementation selection">;

def CPUMaterializeConfigurations : CPUTransform<"intent-cpu-materialize-configurations",
    "Bind complete CPU implementation candidates into the current program"> {
  let options = [CPUProviderOption,
    Option<"defaults", "defaults", "std::string", "\"\"", "Provider default tuning profile file">,
    Option<"overrides", "overrides", "std::string", "\"\"", "Optional tuning profile overrides file">
  ];
}

def CPURealizeRegions : CPUTransform<"intent-cpu-realize-regions",
    "Realize configured CPU regions and close their structured computations"> {
  let options = [CPUProviderOption];
}

def CPUFormInputSupply : CPUTransform<"intent-cpu-form-input-supply",
    "Form input representations, reuse storage and block configured CPU worksets"> {
  let options = [CPUProviderOption];
}

def CPUFormTasks : CPUTransform<"intent-cpu-form-tasks",
    "Partition configured CPU work and isolate complete task captures"> {
  let options = [CPUProviderOption];
}

#endif
