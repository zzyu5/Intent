#ifndef INTENT_DIALECT_INTENT_IR_INTENTATTRS_TD
#define INTENT_DIALECT_INTENT_IR_INTENTATTRS_TD

include "Intent/Dialect/Intent/IR/IntentDialect.td"
include "mlir/IR/AttrTypeBase.td"

def Intent_TensorShapeAttr : AttrDef<Intent_Dialect, "TensorShape"> {
  let mnemonic = "tensor_shape";
  let summary = "Stable identities for logical tensor dimensions";
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$dimensions);
  let assemblyFormat = "`<` $dimensions `>`";
  let genVerifyDecl = 1;
}

def Intent_ShapeExprAttr : AttrDef<Intent_Dialect, "ShapeExpr"> {
  let mnemonic = "shape_expr";
  let summary = "One typed logical shape expression";
  let parameters = (ins
    "uint32_t":$kind,
    "int64_t":$dimension,
    "int64_t":$payload
  );
  let assemblyFormat = "`<` $kind `,` $dimension `,` $payload `>`";
  let genVerifyDecl = 1;
}

def Intent_ShapeRelationAttr : AttrDef<Intent_Dialect, "ShapeRelation"> {
  let mnemonic = "shape_relation";
  let summary = "Typed relation from SSA extents to a logical tensor shape";
  let parameters = (ins "::mlir::ArrayAttr":$axes);
  let assemblyFormat = "`<` $axes `>`";
  let genVerifyDecl = 1;
}

def Intent_IndexTermAttr : AttrDef<Intent_Dialect, "IndexTerm"> {
  let mnemonic = "index_term";
  let summary = "One typed coordinate expression in an index relation";
  let parameters = (ins
    "uint32_t":$kind,
    "::mlir::DenseI64ArrayAttr":$operandPositions,
    "::mlir::DenseI64ArrayAttr":$staticValues
  );
  let assemblyFormat =
      "`<` $kind `,` $operandPositions `,` $staticValues `>`";
  let genVerifyDecl = 1;
}

def Intent_IndexRelationAttr : AttrDef<Intent_Dialect, "IndexRelation"> {
  let mnemonic = "index_relation";
  let summary = "Typed source-to-result logical coordinate relation";
  let parameters = (ins
    "uint32_t":$sourceRank,
    "uint32_t":$resultRank,
    "::mlir::DenseI64ArrayAttr":$resultDimensions,
    "::mlir::ArrayAttr":$terms
  );
  let assemblyFormat =
      "`<` $sourceRank `,` $resultRank `,` $resultDimensions `,` $terms `>`";
  let genVerifyDecl = 1;
}

def Intent_ViewConstraintsAttr : AttrDef<Intent_Dialect, "ViewConstraints"> {
  let mnemonic = "view_constraints";
  let summary = "Typed logical stride and alias constraints for an external view";
  let parameters = (ins
    "bool":$hasStrides,
    "::mlir::ArrayAttr":$strides,
    "::mlir::StringAttr":$alias,
    "bool":$noalias
  );
  let assemblyFormat =
      "`<` $hasStrides `,` $strides `,` $alias `,` $noalias `>`";
  let genVerifyDecl = 1;
}

def Intent_ParameterAttr : AttrDef<Intent_Dialect, "Parameter"> {
  let mnemonic = "parameter";
  let summary = "Typed canonical function parameter role";
  let parameters = (ins "::mlir::StringAttr":$name, "uint32_t":$kind);
  let assemblyFormat = "`<` $name `,` $kind `>`";
  let genVerifyDecl = 1;
}

def Intent_FunctionKindAttr : AttrDef<Intent_Dialect, "FunctionKind"> {
  let mnemonic = "function_kind";
  let summary = "Canonical kernel or helper function role";
  let parameters = (ins "uint32_t":$kind);
  let assemblyFormat = "`<` $kind `>`";
  let genVerifyDecl = 1;
}

def Intent_SparseFormatAttr : AttrDef<Intent_Dialect, "SparseFormat"> {
  let mnemonic = "sparse_format";
  let summary = "Closed canonical structured-sparsity format schema";
  let parameters = (ins "uint32_t":$kind, "uint32_t":$compressionAxis);
  let assemblyFormat = "`<` $kind `,` $compressionAxis `>`";
  let genVerifyDecl = 1;
}

def Intent_TensorShape : DialectAttr<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::TensorShapeAttr>($_self)">,
    "typed tensor dimension identities"> {
  let storageType = "::intent::TensorShapeAttr";
  let returnType = "::intent::TensorShapeAttr";
  let convertFromStorage = "$_self";
}
def Intent_ShapeRelation : DialectAttr<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::ShapeRelationAttr>($_self)">,
    "typed logical shape relation"> {
  let storageType = "::intent::ShapeRelationAttr";
  let returnType = "::intent::ShapeRelationAttr";
  let convertFromStorage = "$_self";
}
def Intent_IndexRelation : DialectAttr<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::IndexRelationAttr>($_self)">,
    "typed logical index relation"> {
  let storageType = "::intent::IndexRelationAttr";
  let returnType = "::intent::IndexRelationAttr";
  let convertFromStorage = "$_self";
}
def Intent_SparseFormat : DialectAttr<
    Intent_Dialect,
    CPred<"::llvm::isa<::intent::SparseFormatAttr>($_self)">,
    "typed sparse format schema"> {
  let storageType = "::intent::SparseFormatAttr";
  let returnType = "::intent::SparseFormatAttr";
  let convertFromStorage = "$_self";
}
#endif
