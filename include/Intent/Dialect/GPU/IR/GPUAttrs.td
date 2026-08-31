#ifndef INTENT_DIALECT_GPU_IR_GPUATTRS_TD
#define INTENT_DIALECT_GPU_IR_GPUATTRS_TD

include "Intent/Dialect/GPU/IR/GPUDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class IntentGPU_I32Enum<string name, string description,
                        list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::intent::gpu";
}

class IntentGPU_I32EnumAttr<string mnemonic, IntentGPU_I32Enum enumInfo>
    : EnumAttr<IntentGPU_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def IntentGPU_AtomicSharingDomain : IntentGPU_I32Enum<
    "AtomicSharingDomain", "Logical participants in one physical atomic object", [
      I32EnumAttrCase<"ProgramInstance", 0, "program_instance">,
      I32EnumAttrCase<"KernelInvocation", 1, "kernel_invocation">
    ]>;
def IntentGPU_AtomicSharingDomainAttr : IntentGPU_I32EnumAttr<
    "atomic_sharing", IntentGPU_AtomicSharingDomain>;

def IntentGPU_BufferScope : IntentGPU_I32Enum<
    "BufferScope", "Logical allocation scope of a physical mutable buffer", [
      I32EnumAttrCase<"ProgramPrivate", 0, "program_private">,
      I32EnumAttrCase<"IterationPrivate", 1, "iteration_private">,
      I32EnumAttrCase<"InvocationWorkspace", 2, "invocation_workspace">
    ]>;
def IntentGPU_BufferScopeAttr : IntentGPU_I32EnumAttr<
    "buffer_scope", IntentGPU_BufferScope>;

def IntentGPU_BufferInitialization : IntentGPU_I32Enum<
    "BufferInitialization", "Initialization obligation of a physical buffer", [
      I32EnumAttrCase<"FullValue", 0, "full_value">,
      I32EnumAttrCase<"FirstWrite", 1, "first_write">
    ]>;
def IntentGPU_BufferInitializationAttr : IntentGPU_I32EnumAttr<
    "buffer_initialization", IntentGPU_BufferInitialization>;

def IntentGPU_BufferLifetime : IntentGPU_I32Enum<
    "BufferLifetime", "Dynamic lifetime of one physical allocation instance", [
      I32EnumAttrCase<"Program", 0, "program">,
      I32EnumAttrCase<"Iteration", 1, "iteration">,
      I32EnumAttrCase<"Invocation", 2, "invocation">
    ]>;
def IntentGPU_BufferLifetimeAttr : IntentGPU_I32EnumAttr<
    "buffer_lifetime", IntentGPU_BufferLifetime>;

def IntentGPU_PhysicalExprAttr : AttrDef<IntentGPU_Dialect, "PhysicalExpr"> {
  let mnemonic = "expr";
  let summary = "Typed launch or compile-time physical integer expression";
  let parameters = (ins
    "uint32_t":$kind,
    "int64_t":$value,
    "::mlir::StringAttr":$symbol,
    "::mlir::ArrayAttr":$operands
  );
  let assemblyFormat = "`<` $kind `,` $value `,` $symbol `,` $operands `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_ParameterAttr : AttrDef<IntentGPU_Dialect, "Parameter"> {
  let mnemonic = "parameter";
  let summary = "One typed compile-time physical parameter and its legal candidates";
  let parameters = (ins
    "::mlir::StringAttr":$name,
    "uint32_t":$role,
    "uint32_t":$category,
    "uint32_t":$elementBitWidth,
    "::mlir::DenseI64ArrayAttr":$candidates
  );
  let assemblyFormat =
      "`<` $name `,` $role `,` $category `,` $elementBitWidth `,` $candidates `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_AxisMapAttr : AttrDef<IntentGPU_Dialect, "AxisMap"> {
  let mnemonic = "axis_map";
  let summary = "Fragment axis to immutable logical source-axis mapping";
  let parameters = (ins
    "uint64_t":$sourceId,
    "uint32_t":$sourceAxis,
    "int64_t":$dimensionId,
    "uint32_t":$fragmentAxis,
    "bool":$derived
  );
  let assemblyFormat =
      "`<` $sourceId `,` $sourceAxis `,` $dimensionId `,` $fragmentAxis `,` $derived `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_PhysicalSourceAttr : AttrDef<IntentGPU_Dialect, "PhysicalSource"> {
  let mnemonic = "source";
  let summary = "One immutable logical source-axis identity";
  let parameters = (ins
    "uint64_t":$sourceId,
    "uint32_t":$sourceAxis,
    "bool":$derived
  );
  let assemblyFormat = "`<` $sourceId `,` $sourceAxis `,` $derived `>`";
}

def IntentGPU_ReshapeGroupAttr : AttrDef<IntentGPU_Dialect, "ReshapeGroup"> {
  let mnemonic = "reshape_group";
  let summary = "One contiguous row-major physical reshape reassociation group";
  let parameters = (ins
    "::mlir::DenseI64ArrayAttr":$sourceAxes,
    "::mlir::DenseI64ArrayAttr":$resultAxes
  );
  let assemblyFormat = "`<` $sourceAxes `,` $resultAxes `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_ViewLayoutAttr : AttrDef<IntentGPU_Dialect, "ViewLayout"> {
  let mnemonic = "view_layout";
  let summary = "Host-visible physical view shape, stride and alias contract";
  let parameters = (ins
    "::mlir::ArrayAttr":$extents,
    "::mlir::DenseI64ArrayAttr":$dimensionIds,
    "bool":$hasStrides,
    "::mlir::ArrayAttr":$strides,
    "::mlir::StringAttr":$alias,
    "bool":$noalias
  );
  let assemblyFormat =
      "`<` $extents `,` $dimensionIds `,` $hasStrides `,` $strides `,` $alias `,` $noalias `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_CapabilitiesAttr : AttrDef<IntentGPU_Dialect, "Capabilities"> {
  let mnemonic = "capabilities";
  let summary = "Selected GPU hardware resources used by shared legality";
  let parameters = (ins
    "int64_t":$computeUnits,
    "int64_t":$sharedMemoryPerUnit,
    "int64_t":$registersPerUnit,
    "bool":$matrixUnits,
    "bool":$dynamicVectorWidth
  );
  let assemblyFormat =
      "`<` $computeUnits `,` $sharedMemoryPerUnit `,` $registersPerUnit `,` $matrixUnits `,` $dynamicVectorWidth `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_PhysicalExpr : DialectAttr<
    IntentGPU_Dialect,
    CPred<"::llvm::isa<::intent::gpu::PhysicalExprAttr>($_self)">,
    "typed physical expression"> {
  let storageType = "::intent::gpu::PhysicalExprAttr";
  let returnType = "::intent::gpu::PhysicalExprAttr";
  let convertFromStorage = "$_self";
}

def IntentGPU_Parameter : DialectAttr<
    IntentGPU_Dialect,
    CPred<"::llvm::isa<::intent::gpu::ParameterAttr>($_self)">,
    "typed physical parameter"> {
  let storageType = "::intent::gpu::ParameterAttr";
  let returnType = "::intent::gpu::ParameterAttr";
  let convertFromStorage = "$_self";
}

#endif
