#ifndef INTENT_TARGET_BANGC_PASSES_TD
#define INTENT_TARGET_BANGC_PASSES_TD

include "mlir/Pass/PassBase.td"

class IntentBangCPass<string passArgument, string summaryText>
    : Pass<passArgument, "::mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = [
    "::intent::IntentDialect", "::intent::dsa::IntentDSADialect", "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect", "::mlir::math::MathDialect",
    "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect"
  ];
}

def BangCPrepareProgram : IntentBangCPass<"intent-bangc-prepare-program",
    "Verify the shared DSA program and bind the selected BANG C architecture"> {
  let options = [Option<"architecture", "architecture", "std::string", "\"\"",
      "Explicit BANG C architecture whose implementation profile will be used">];
}
def BangCNativeComputations : IntentBangCPass<"intent-bangc-native-computations",
    "Realize BANG C numerical and matrix computations">;
def BangCNativeWorkspace : IntentBangCPass<"intent-bangc-native-workspace",
    "Realize workspace required by selected BANG C computations">;
def BangCNativeImplementations : IntentBangCPass<"intent-bangc-native-implementations",
    "Select BANG C native implementations and matrix layouts">;
def BangCLocalComposition : IntentBangCPass<"intent-bangc-local-composition",
    "Compose local values and accesses in the BANG C program">;
def BangCSupplySynchronization : IntentBangCPass<"intent-bangc-supply-synchronization",
    "Schedule BANG C supply, participants and synchronization">;
def BangCStorageBinding : IntentBangCPass<"intent-bangc-storage-binding",
    "Bind final BANG C physical storage and verify target legality">;

#endif
