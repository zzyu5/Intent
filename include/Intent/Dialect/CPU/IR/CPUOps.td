#ifndef INTENT_DIALECT_CPU_IR_CPUOPS_TD
#define INTENT_DIALECT_CPU_IR_CPUOPS_TD

include "Intent/Dialect/CPU/IR/CPUAttrs.td"
include "Intent/Dialect/Intent/IR/IntentAttrs.td"
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

def IntentCPU_ScanOp : IntentCPU_Op<"scan", [SingleBlock, IsolatedFromAbove, AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Ordered prefixes with typed scalar or destination-passing slice combines";
  let arguments = (ins Variadic<AnyMemRef>:$sources,
      Variadic<AnyType>:$initials, Variadic<AnyType>:$captures,
      Variadic<AnyMemRef>:$outputs,
      I64Attr:$axis, BoolAttr:$inclusive, BoolAttr:$reverse);
  let regions = (region AnyRegion:$combine);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    bool isDestinationPassing() {
      return !getCombine().empty() && getCombine().front().getNumArguments() ==
          3 * getSources().size() + getCaptures().size();
    }
  }];
}

def IntentCPU_ScanYieldOp : IntentCPU_Op<"scan_yield", [Pure, Terminator,
    HasParent<"ScanOp">]> {
  let arguments = (ins Variadic<AnyType>:$values);
}

def IntentCPU_HistogramOp : IntentCPU_Op<"histogram"> {
  let summary = "Typed active values contributing to logical integer bins";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$values,
      Arg<AnyMemRef, "", [MemRead]>:$valid,
      Arg<AnyMemRef, "", [MemWrite]>:$output, BoolAttr:$unsigned_values);
  let hasVerifier = 1;
}

// Synchronization orders surrounding memory effects, not only the target cell.
class IntentCPU_AtomicOp<string name> : IntentCPU_Op<name,
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;
}

def IntentCPU_AtomicLoadOp : IntentCPU_AtomicOp<"atomic_load"> {
  let arguments = (ins AnyMemRef:$target, Variadic<Index>:$indices,
      Intent_AtomicOrderingAttr:$ordering);
  let results = (outs AnyType:$value);
}

def IntentCPU_AtomicStoreOp : IntentCPU_AtomicOp<"atomic_store"> {
  let arguments = (ins AnyMemRef:$target, AnyType:$value,
      Variadic<Index>:$indices, Intent_AtomicOrderingAttr:$ordering);
}

def IntentCPU_AtomicRMWOp : IntentCPU_AtomicOp<"atomic_rmw"> {
  let arguments = (ins AnyMemRef:$target, AnyType:$value,
      Variadic<Index>:$indices, Intent_AtomicOrderingAttr:$ordering,
      Intent_AtomicRMWKindAttr:$kind, BoolAttr:$unsigned_integer);
  let results = (outs AnyType:$old_value);
}

def IntentCPU_AtomicCompareExchangeOp : IntentCPU_AtomicOp<"atomic_compare_exchange"> {
  let arguments = (ins AnyMemRef:$target, AnyType:$expected, AnyType:$desired,
      Variadic<Index>:$indices, Intent_AtomicOrderingAttr:$ordering);
  let results = (outs AnyType:$old_value, I1:$success);
}

def IntentCPU_QuantizeOp : IntentCPU_Op<"quantize"> {
  let summary = "Quantize complete records into invocation-owned storage";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$input,
      Arg<AnyMemRef, "", [MemWrite]>:$output, Intent_QuantFormatAttr:$format);
  let hasVerifier = 1;
}

def IntentCPU_QuantizedDotOp : IntentCPU_Op<"quantized_dot"> {
  let summary = "Independent closed record dots with explicit output ownership";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$lhs,
      Arg<AnyMemRef, "", [MemRead]>:$rhs,
      Arg<AnyMemRef, "", [MemWrite]>:$output,
      Intent_QuantFormatAttr:$lhs_format, Intent_QuantFormatAttr:$rhs_format);
  let hasVerifier = 1;
}

class IntentCPU_RegionOp<string name> : IntentCPU_Op<name,
    [RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$inputs, I64Attr:$axis,
      I64Attr:$source_count, I64Attr:$identity_count, I64Attr:$capture_count,
      I64Attr:$state_count, I64Attr:$output_count,
      ArrayAttr:$summary_fields, ArrayAttr:$state_fields, DenseI64ArrayAttr:$output_axes,
      OptionalAttr<I64Attr>:$segment_size);
  let hasVerifier = 1;
}

def IntentCPU_RegionFoldOp : IntentCPU_RegionOp<"region_fold"> {
  let summary = "Ordered source slices and destination-passing summary helpers";
  let regions = (region SizedRegion<1>:$summarize, SizedRegion<1>:$combine);
}

def IntentCPU_RegionScanOp : IntentCPU_RegionOp<"region_scan"> {
  let summary = "Region transitions, incoming state, slice outputs and final state";
  let regions = (region SizedRegion<1>:$summarize, SizedRegion<1>:$combine,
      SizedRegion<1>:$apply, SizedRegion<1>:$emit);
}

def IntentCPU_RegionYieldOp : IntentCPU_Op<"region_yield", [Pure, Terminator]>;

#endif
