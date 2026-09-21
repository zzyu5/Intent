#ifndef INTENT_DIALECT_DSA_IR_DSAATTRS_TD
#define INTENT_DIALECT_DSA_IR_DSAATTRS_TD
include "Intent/Dialect/DSA/IR/DSADialect.td"
include "Intent/Dialect/Intent/IR/IntentAttrs.td"
include "mlir/IR/AttrTypeBase.td"
def IntentDSA_ViewArgument : AttrDef<IntentDSA_Dialect, "ViewArgument"> {
  let mnemonic = "view_argument";
  let parameters = (ins "::mlir::StringAttr":$name, "::mlir::Type":$element,
    "::mlir::DenseI64ArrayAttr":$shape, "::mlir::DenseI64ArrayAttr":$dimensions,
    "uint32_t":$access, "::intent::ViewConstraintsAttr":$constraints);
  let assemblyFormat = "`<` $name `,` $element `,` $shape `,` $dimensions `,` $access `,` $constraints `>`";
  let genVerifyDecl = 1;
}
def IntentDSA_ScalarArgument : AttrDef<IntentDSA_Dialect, "ScalarArgument"> {
  let mnemonic = "scalar_argument";
  let parameters = (ins "::mlir::StringAttr":$name, "::mlir::Type":$type);
  let assemblyFormat = "`<` $name `,` $type `>`";
  let genVerifyDecl = 1;
}
def IntentDSA_Interface : AttrDef<IntentDSA_Dialect, "Interface"> {
  let mnemonic = "interface";
  let parameters = (ins "::mlir::ArrayAttr":$arguments);
  let assemblyFormat = "`<` $arguments `>`";
  let genVerifyDecl = 1;
}
def IntentDSA_Configuration : AttrDef<IntentDSA_Dialect, "Configuration"> {
  let mnemonic = "configuration";
  let parameters = (ins "int64_t":$tile, "int64_t":$tileM,
    "int64_t":$tileN, "int64_t":$tileK, "int64_t":$tasks,
    "int64_t":$localBytes);
  let assemblyFormat = "`<` $tile `,` $tileM `,` $tileN `,` $tileK `,` $tasks `,` $localBytes `>`";
  let genVerifyDecl = 1;
}
#endif
