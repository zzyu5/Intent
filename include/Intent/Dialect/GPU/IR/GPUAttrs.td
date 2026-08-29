#ifndef INTENT_DIALECT_GPU_IR_GPUATTRS_TD
#define INTENT_DIALECT_GPU_IR_GPUATTRS_TD

include "Intent/Dialect/GPU/IR/GPUDialect.td"
include "mlir/IR/AttrTypeBase.td"

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
  let summary = "One compile-time physical parameter and its legal candidates";
  let parameters = (ins
    "::mlir::StringAttr":$name,
    "uint32_t":$role,
    "::mlir::DenseI64ArrayAttr":$candidates
  );
  let assemblyFormat = "`<` $name `,` $role `,` $candidates `>`";
  let genVerifyDecl = 1;
}

def IntentGPU_AxisMapAttr : AttrDef<IntentGPU_Dialect, "AxisMap"> {
  let mnemonic = "axis_map";
  let summary = "Fragment axis to immutable logical source-axis mapping";
  let parameters = (ins
    "uint64_t":$sourceId,
    "uint32_t":$sourceAxis,
    "int64_t":$dimensionId,
    "uint32_t":$fragmentAxis
  );
  let assemblyFormat =
      "`<` $sourceId `,` $sourceAxis `,` $dimensionId `,` $fragmentAxis `>`";
  let genVerifyDecl = 1;
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
