#ifndef INTENT_DIALECT_CPU_IR_CPUOPS_TD
#define INTENT_DIALECT_CPU_IR_CPUOPS_TD

include "Intent/Dialect/CPU/IR/CPUAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IntentCPU_Op<string name, list<Trait> traits = []>
    : Op<IntentCPU_Dialect, name, traits>;

def IntentCPU_TasksOp : IntentCPU_Op<"tasks", [SingleBlock,
    IsolatedFromAbove, AutomaticAllocationScope, RecursiveMemoryEffects]> {
  let summary = "Synchronous tasks over [0, count), with explicit captured values";
  let arguments = (ins Index:$count, Variadic<AnyType>:$captures);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def IntentCPU_TaskYieldOp : IntentCPU_Op<"task_yield", [Pure, Terminator,
    HasParent<"TasksOp">]>;

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
