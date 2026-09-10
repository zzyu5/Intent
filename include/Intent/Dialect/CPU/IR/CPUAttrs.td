#ifndef INTENT_DIALECT_CPU_IR_CPUATTRS_TD
#define INTENT_DIALECT_CPU_IR_CPUATTRS_TD

include "Intent/Dialect/CPU/IR/CPUDialect.td"
include "mlir/IR/AttrTypeBase.td"

class IntentCPU_Attr<string name, string mnemonicName>
    : AttrDef<IntentCPU_Dialect, name> {
  let mnemonic = mnemonicName;
  let genVerifyDecl = 1;
}

def IntentCPU_ViewArgumentAttr : IntentCPU_Attr<"ViewArgument", "view_argument"> {
  let parameters = (ins "::mlir::StringAttr":$name,
    "::mlir::Type":$elementType, "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$dimensions, "uint32_t":$access,
    "::mlir::StringAttr":$alias, "bool":$noalias);
  let assemblyFormat =
    "`<` $name `,` $elementType `,` $shape `,` $dimensions `,` $access `,` $alias `,` $noalias `>`";
}

def IntentCPU_ScalarArgumentAttr : IntentCPU_Attr<"ScalarArgument", "scalar_argument"> {
  let parameters = (ins "::mlir::StringAttr":$name, "::mlir::Type":$type);
  let assemblyFormat = "`<` $name `,` $type `>`";
}

def IntentCPU_InterfaceAttr : IntentCPU_Attr<"Interface", "interface"> {
  let parameters = (ins "::mlir::ArrayAttr":$arguments,
    "bool":$contiguousViews, "bool":$disjointOutputs);
  let assemblyFormat = "`<` $arguments `,` $contiguousViews `,` $disjointOutputs `>`";
}

def IntentCPU_CapabilitiesAttr : IntentCPU_Attr<"Capabilities", "capabilities"> {
  let parameters = (ins "int64_t":$vectorBits, "int64_t":$workers,
    "int64_t":$privateBytes);
  let assemblyFormat = "`<` $vectorBits `,` $workers `,` $privateBytes `>`";
}

def IntentCPU_ConfigurationAttr : IntentCPU_Attr<"Configuration", "configuration"> {
  let parameters = (ins "int64_t":$taskGrain,
    "int64_t":$tileM, "int64_t":$tileN, "int64_t":$tileK);
  let assemblyFormat =
    "`<` $taskGrain `,` $tileM `,` $tileN `,` $tileK `>`";
}

def IntentCPU_ImplementationAttr : IntentCPU_Attr<"Implementation", "implementation"> {
  let parameters = (ins "::mlir::StringAttr":$name,
                        "::mlir::DictionaryAttr":$parameters);
  let assemblyFormat = "`<` $name `,` $parameters `>`";
}

def IntentCPU_ReductionOrderAttr : IntentCPU_Attr<"ReductionOrder", "reduction_order"> {
  let parameters = (ins "bool":$adjacentReassociation);
  let assemblyFormat = "`<` $adjacentReassociation `>`";
}

def IntentCPU_MicrotileAttr : IntentCPU_Attr<"Microtile", "microtile"> {
  let parameters = (ins "int64_t":$rows, "int64_t":$columns,
    "int64_t":$vectorWidth);
  let assemblyFormat = "`<` $rows `,` $columns `,` $vectorWidth `>`";
}

#endif
