#ifndef INTENT_DIALECT_DSA_TRANSFORMS_PASSES_TD
#define INTENT_DIALECT_DSA_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

class IntentDSAPass<string passArgument, string summaryText>
    : Pass<passArgument, "::mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = [
    "::intent::IntentDialect", "::intent::dsa::IntentDSADialect", "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect", "::mlir::math::MathDialect",
    "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect"
  ];
}

def DSACollectiveGatherSupply : IntentDSAPass<
    "intent-dsa-collective-gather-supply",
    "Form collective row supply from current DSA address relations">;
def DSAMatrixSupply : IntentDSAPass<"intent-dsa-matrix-supply",
    "Form resident or shared supply for a complete DSA matrix traversal">;
def DSANormalizeIndices : IntentDSAPass<"intent-dsa-normalize-indices",
    "Normalize exact DSA index programs and canonicalize the result">;

#endif
