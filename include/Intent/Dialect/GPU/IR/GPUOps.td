#ifndef INTENT_DIALECT_GPU_IR_GPUOPS_TD
#define INTENT_DIALECT_GPU_IR_GPUOPS_TD

include "Intent/Dialect/GPU/IR/GPUDialect.td"
include "Intent/Dialect/GPU/IR/GPUAttrs.td"
include "Intent/Dialect/GPU/IR/GPUTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IntentGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<IntentGPU_Dialect, mnemonic, traits>;

def IntentGPU_ParameterOp : IntentGPU_Op<"parameter", [Pure]> {
  let arguments = (ins IntentGPU_Parameter:$parameter);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentGPU_PhysicalExprOp : IntentGPU_Op<"physical_expr", [Pure]> {
  let arguments = (ins IntentGPU_PhysicalExpr:$expression);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentGPU_ProgramIdOp : IntentGPU_Op<"program_id", [Pure]> {
  let arguments = (ins I64Attr:$axis);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentGPU_DelinearizeOp : IntentGPU_Op<"delinearize", [Pure]> {
  let summary = "Row-major decoding of one physical program coordinate";
  let arguments = (ins Index:$linear,
                       Variadic<Index>:$extents,
                       ArrayAttr:$launch_extents);
  let results = (outs Variadic<Index>:$coordinates);
  let hasVerifier = 1;
}

def IntentGPU_DimOp : IntentGPU_Op<"dim", [Pure]> {
  let arguments = (ins IntentGPU_ViewValue:$view, I64Attr:$axis);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentGPU_RangeOp : IntentGPU_Op<"range", [Pure]> {
  let arguments = (ins Index:$start, Index:$stop, Index:$step);
  let results = (outs IntentGPU_RangeValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_RangeBoundOp : IntentGPU_Op<"range_bound", [Pure]> {
  let arguments = (ins IntentGPU_RangeValue:$range, I64Attr:$bound);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentGPU_MakeRangeOp : IntentGPU_Op<"make_range", [Pure]> {
  let arguments = (ins Index:$start, Index:$extent, Index:$step,
                       I64Attr:$source_id,
                       I64Attr:$source_axis);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_SplatOp : IntentGPU_Op<"splat", [Pure]> {
  let arguments = (ins IntentGPU_ScalarValue:$value);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_BroadcastOp : IntentGPU_Op<"broadcast", [Pure]> {
  let arguments = (ins IntentGPU_DataValue:$value);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_UnaryOp : IntentGPU_Op<"unary", [Pure]> {
  let arguments = (ins IntentGPU_DataValue:$input, I64Attr:$operator_kind);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_BinaryOp : IntentGPU_Op<"binary", [Pure]> {
  let arguments = (ins IntentGPU_DataValue:$lhs, IntentGPU_DataValue:$rhs,
                       I64Attr:$operator_kind);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_CompareOp : IntentGPU_Op<"compare", [Pure]> {
  let arguments = (ins IntentGPU_DataValue:$lhs, IntentGPU_DataValue:$rhs,
                       I64Attr:$predicate);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_SelectOp : IntentGPU_Op<"select", [Pure]> {
  let arguments = (ins IntentGPU_DataValue:$condition,
                       IntentGPU_DataValue:$true_value,
                       IntentGPU_DataValue:$false_value);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_CastOp : IntentGPU_Op<"cast", [Pure]> {
  let arguments = (ins IntentGPU_DataValue:$value);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_BitcastOp : IntentGPU_Op<"bitcast", [Pure]> {
  let arguments = (ins IntentGPU_DataValue:$value);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_ReshapeOp : IntentGPU_Op<"reshape", [Pure]> {
  let arguments = (ins IntentGPU_DataValue:$value,
                       ArrayAttr:$reassociation);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_TransposeOp : IntentGPU_Op<"transpose", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$value,
                       DenseI64ArrayAttr:$permutation);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_JoinOp : IntentGPU_Op<"join", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$lhs,
                       IntentGPU_FragmentValue:$rhs,
                       I64Attr:$axis);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_MakeRecordOp : IntentGPU_Op<"make_record", [Pure]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$fields);
  let results = (outs IntentGPU_RecordValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_ExtractOp : IntentGPU_Op<"extract", [Pure]> {
  let arguments = (ins IntentGPU_RecordValue:$record, I64Attr:$field);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_LoadOp : IntentGPU_Op<"load", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       Optional<IntentGPU_DataValue>:$valid,
                       Optional<IntentGPU_DataValue>:$fill,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_GatherOp : IntentGPU_Op<"gather", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins IntentGPU_DataValue:$source,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       Optional<IntentGPU_DataValue>:$valid,
                       Optional<IntentGPU_DataValue>:$fill,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_AssumeInBoundsOp : IntentGPU_Op<"assume_in_bounds", [Pure]> {
  let arguments = (ins IntentGPU_DataValue:$index,
                       IntentGPU_ResourceValue:$resource,
                       I64Attr:$axis);
  let hasVerifier = 1;
}

def IntentGPU_StoreOp : IntentGPU_Op<"store", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_DataValue:$value,
                       Optional<IntentGPU_DataValue>:$valid,
                       DenseI64ArrayAttr:$source_axes,
                       I64Attr:$collision);
  let hasVerifier = 1;
}

def IntentGPU_ContractOp : IntentGPU_Op<"contract", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$lhs,
                       IntentGPU_FragmentValue:$rhs,
                       IntentGPU_FragmentValue:$accumulator,
                       DenseI64ArrayAttr:$lhs_reduction_axes,
                       DenseI64ArrayAttr:$rhs_reduction_axes,
                       DenseI64ArrayAttr:$lhs_batch_axes,
                       DenseI64ArrayAttr:$rhs_batch_axes);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_ReduceOp : IntentGPU_Op<"reduce", [Pure]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$inputs,
                       DenseI64ArrayAttr:$axes,
                       I64Attr:$source_count,
                       I64Attr:$identity_count,
                       I64Attr:$capture_count);
  let results = (outs Variadic<IntentGPU_DataValue>:$results);
  let regions = (region SizedRegion<1>:$combine);
  let hasVerifier = 1;
}

def IntentGPU_ScanOp : IntentGPU_Op<"scan", [Pure]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$inputs,
                       I64Attr:$axis, BoolAttr:$inclusive, BoolAttr:$reverse,
                       I64Attr:$source_count, I64Attr:$identity_count,
                       I64Attr:$capture_count);
  let results = (outs Variadic<IntentGPU_DataValue>:$results);
  let regions = (region SizedRegion<1>:$combine);
  let hasVerifier = 1;
}

def IntentGPU_RegionFoldOp : IntentGPU_Op<"region_fold", [Pure]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$inputs,
                       I64Attr:$axis, I64Attr:$source_count,
                       I64Attr:$identity_count, I64Attr:$capture_count,
                       IntentGPU_Parameter:$segment);
  let results = (outs Variadic<IntentGPU_DataValue>:$results);
  let regions = (region SizedRegion<1>:$summarize,
                         SizedRegion<1>:$combine);
  let hasVerifier = 1;
}

def IntentGPU_RegionScanOp : IntentGPU_Op<"region_scan", [Pure]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$inputs,
                       I64Attr:$axis, I64Attr:$source_count,
                       I64Attr:$identity_count, I64Attr:$state_count,
                       I64Attr:$capture_count, I64Attr:$output_count,
                       IntentGPU_Parameter:$segment);
  let results = (outs Variadic<IntentGPU_DataValue>:$results);
  let regions = (region SizedRegion<1>:$summarize,
                         SizedRegion<1>:$combine,
                         SizedRegion<1>:$apply,
                         SizedRegion<1>:$emit);
  let hasVerifier = 1;
}

def IntentGPU_ScaledContractOp : IntentGPU_Op<"scaled_contract", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$lhs,
                       IntentGPU_FragmentValue:$lhs_scale,
                       IntentGPU_FragmentValue:$rhs,
                       IntentGPU_FragmentValue:$rhs_scale,
                       IntentGPU_FragmentValue:$accumulator,
                       DenseI64ArrayAttr:$lhs_reduction_axes,
                       DenseI64ArrayAttr:$rhs_reduction_axes,
                       DenseI64ArrayAttr:$lhs_batch_axes,
                       DenseI64ArrayAttr:$rhs_batch_axes,
                       I64Attr:$lhs_format, I64Attr:$rhs_format,
                       I64Attr:$lhs_group_size, I64Attr:$rhs_group_size);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_SparseContractOp : IntentGPU_Op<"sparse_contract", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$compressed,
                       IntentGPU_DataValue:$metadata,
                       IntentGPU_FragmentValue:$rhs,
                       Index:$logical_extent,
                       IntentGPU_FragmentValue:$accumulator,
                       I64Attr:$format,
                       I64Attr:$compression_axis,
                       DenseI64ArrayAttr:$lhs_reduction_axes,
                       DenseI64ArrayAttr:$rhs_reduction_axes,
                       DenseI64ArrayAttr:$lhs_batch_axes,
                       DenseI64ArrayAttr:$rhs_batch_axes);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_HistogramOp : IntentGPU_Op<"histogram", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$values, Index:$bins,
                       IntentGPU_FragmentValue:$valid);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_ScatterReduceOp : IntentGPU_Op<"scatter_reduce", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_DataValue:$value,
                       Optional<IntentGPU_DataValue>:$valid,
                       DenseI64ArrayAttr:$source_axes,
                       I64Attr:$sharing);
  let regions = (region SizedRegion<1>:$combine);
  let hasVerifier = 1;
}

def IntentGPU_AtomicLoadOp : IntentGPU_Op<"atomic_load", [
    AttrSizedOperandSegments, MemoryEffects<[MemRead]>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       Optional<IntentGPU_DataValue>:$valid,
                       I64Attr:$ordering, I64Attr:$sharing,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_AtomicStoreOp : IntentGPU_Op<"atomic_store", [
    AttrSizedOperandSegments, MemoryEffects<[MemWrite]>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_DataValue:$value,
                       Optional<IntentGPU_DataValue>:$valid,
                       I64Attr:$ordering, I64Attr:$sharing,
                       DenseI64ArrayAttr:$source_axes);
  let hasVerifier = 1;
}

def IntentGPU_AtomicRMWOp : IntentGPU_Op<"atomic_rmw", [
    AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_DataValue:$value,
                       Optional<IntentGPU_DataValue>:$valid,
                       I64Attr:$kind, I64Attr:$ordering, I64Attr:$sharing,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_AtomicCompareExchangeOp : IntentGPU_Op<"atomic_compare_exchange", [
    AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_DataValue:$expected,
                       IntentGPU_DataValue:$desired,
                       Optional<IntentGPU_DataValue>:$valid,
                       I64Attr:$ordering, I64Attr:$sharing,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentGPU_RecordValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_RandomBitsOp : IntentGPU_Op<"random_bits", [Pure]> {
  let arguments = (ins I64:$seed, IntentGPU_DataValue:$counter);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_YieldOp : IntentGPU_Op<"yield", [Terminator]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$values);
}

def IntentGPU_BufferOp : IntentGPU_Op<"buffer", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins Optional<IntentGPU_DataValue>:$initial_value);
  let results = (outs IntentGPU_BufferValue:$result);
  let hasVerifier = 1;
}

#endif
