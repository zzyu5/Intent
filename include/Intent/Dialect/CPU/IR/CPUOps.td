#ifndef INTENT_DIALECT_CPU_IR_CPUOPS_TD
#define INTENT_DIALECT_CPU_IR_CPUOPS_TD

include "Intent/Dialect/CPU/IR/CPUAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IntentCPU_Op<string name, list<Trait> traits = []>
    : Op<IntentCPU_Dialect, name, traits>;

def IntentCPU_ReduceOp : IntentCPU_Op<"reduce", [SingleBlock,
    MemoryEffects<[MemRead]>]> {
  let summary = "Ordered one-axis reduction with an explicit scalar combine region";
  let arguments = (ins Index:$extent, AnyFloat:$initial,
    Variadic<AnyType>:$inputs, ArrayAttr:$indexingMaps,
    IntentCPU_ReductionOrderAttr:$order);
  let results = (outs AnyFloat:$result);
  let regions = (region AnyRegion:$combine);
  let hasVerifier = 1;
}

def IntentCPU_YieldOp : IntentCPU_Op<"yield", [Pure, Terminator,
    HasParent<"ReduceOp">]> {
  let arguments = (ins AnyFloat:$value);
}

#endif
