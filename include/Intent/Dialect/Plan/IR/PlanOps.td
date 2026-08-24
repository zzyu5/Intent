#ifndef INTENT_DIALECT_PLAN_IR_PLANOPS_TD
#define INTENT_DIALECT_PLAN_IR_PLANOPS_TD

include "Intent/Dialect/Plan/IR/PlanDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IntentPlan_Op<string mnemonic, list<Trait> traits = []>
    : Op<IntentPlan_Dialect, mnemonic, traits>;

class IntentPlan_ExecNullaryOneResultOp<string mnemonic,
                                        list<Trait> traits = [Pure]>
    : IntentPlan_Op<mnemonic, traits> {
  let arguments = (ins);
  let results = (outs AnyType:$result);
}

class IntentPlan_ExecUnaryOneResultOp<string mnemonic,
                                      list<Trait> traits = [Pure]>
    : IntentPlan_Op<mnemonic, traits> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}

class IntentPlan_ExecBinaryOneResultOp<string mnemonic,
                                       list<Trait> traits = [Pure]>
    : IntentPlan_Op<mnemonic, traits> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

class IntentPlan_ExecVariadicOneResultOp<string mnemonic,
                                         list<Trait> traits = [Pure]>
    : IntentPlan_Op<mnemonic, traits> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
}

class IntentPlan_ExecVariadicNoResultOp<string mnemonic,
                                        list<Trait> traits = []>
    : IntentPlan_Op<mnemonic, traits> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
}

class IntentPlan_ExecTerminatorOp<string mnemonic>
    : IntentPlan_Op<mnemonic, [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
}

// Executable physical operations.  These mirror the algorithmic arity and
// region contracts of canonical Intent operations while living in the sole
// mutable physical program.  Their attributes retain stable KIR provenance;
// shared and provider passes may refine only physical representation fields.
def IntentPlan_ExecConstantOp
    : IntentPlan_ExecNullaryOneResultOp<"exec_constant">;
def IntentPlan_ExecDimOp
    : IntentPlan_ExecUnaryOneResultOp<"exec_dim">;
def IntentPlan_ExecDomainOp
    : IntentPlan_ExecVariadicOneResultOp<"exec_domain">;
def IntentPlan_ExecDomainProductOp
    : IntentPlan_ExecVariadicOneResultOp<"exec_domain_product">;
def IntentPlan_ExecPartitionOp
    : IntentPlan_ExecVariadicOneResultOp<"exec_partition">;
def IntentPlan_ExecIndicesOp
    : IntentPlan_ExecUnaryOneResultOp<"exec_indices">;
def IntentPlan_ExecRegionEndOp
    : IntentPlan_ExecUnaryOneResultOp<"exec_region_end">;
def IntentPlan_ExecAssumeInBoundsOp
    : IntentPlan_Op<"exec_assume_in_bounds"> {
  let arguments = (ins AnyType:$index, AnyType:$view);
  let results = (outs);
}

def IntentPlan_ExecParallelOp
    : IntentPlan_Op<"exec_parallel", [RecursiveMemoryEffects]> {
  let arguments = (ins AnyType:$source);
  let results = (outs);
  let regions = (region AnyRegion:$body);
}
def IntentPlan_ExecStateStreamOp
    : IntentPlan_Op<"exec_state_stream", [RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
}
def IntentPlan_ExecIfOp
    : IntentPlan_Op<"exec_if", [RecursiveMemoryEffects]> {
  let arguments = (ins AnyType:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<AnyRegion>:$branches);
}
def IntentPlan_ExecForOp
    : IntentPlan_Op<"exec_for", [RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
}
def IntentPlan_ExecWhileOp
    : IntentPlan_Op<"exec_while", [RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$before, AnyRegion:$after);
}
def IntentPlan_ExecYieldOp : IntentPlan_ExecTerminatorOp<"exec_yield">;
def IntentPlan_ExecConditionOp
    : IntentPlan_ExecTerminatorOp<"exec_condition">;

def IntentPlan_ExecViewLoadOp
    : IntentPlan_ExecVariadicOneResultOp<"exec_view_load",
                                                [MemoryEffects<[MemRead]>]>;
def IntentPlan_ExecViewStoreOp
    : IntentPlan_ExecVariadicNoResultOp<"exec_view_store",
                                               [MemoryEffects<[MemWrite]>]>;
def IntentPlan_ExecReshapeOp
    : IntentPlan_ExecUnaryOneResultOp<"exec_reshape">;
def IntentPlan_ExecJoinOp
    : IntentPlan_ExecBinaryOneResultOp<"exec_join">;
def IntentPlan_ExecTransposeOp
    : IntentPlan_ExecUnaryOneResultOp<"exec_transpose">;
def IntentPlan_ExecBroadcastOp
    : IntentPlan_ExecUnaryOneResultOp<"exec_broadcast">;
def IntentPlan_ExecFullOp
    : IntentPlan_ExecUnaryOneResultOp<"exec_full">;
def IntentPlan_ExecZerosOp
    : IntentPlan_ExecNullaryOneResultOp<"exec_zeros">;
def IntentPlan_ExecMakeRecordOp
    : IntentPlan_ExecVariadicOneResultOp<"exec_make_record">;
def IntentPlan_ExecExtractOp
    : IntentPlan_ExecUnaryOneResultOp<"exec_extract">;

def IntentPlan_ExecUnaryOp
    : IntentPlan_ExecUnaryOneResultOp<"exec_unary">;
def IntentPlan_ExecBinaryOp
    : IntentPlan_ExecBinaryOneResultOp<"exec_binary">;
def IntentPlan_ExecCompareOp
    : IntentPlan_ExecBinaryOneResultOp<"exec_compare">;
def IntentPlan_ExecSelectOp : IntentPlan_Op<"exec_select", [Pure]> {
  let arguments = (ins AnyType:$condition, AnyType:$true_value,
                        AnyType:$false_value);
  let results = (outs AnyType:$result);
}
def IntentPlan_ExecCastOp
    : IntentPlan_ExecUnaryOneResultOp<"exec_cast">;
def IntentPlan_ExecBitcastOp
    : IntentPlan_ExecUnaryOneResultOp<"exec_bitcast">;
def IntentPlan_ExecMaskOp : IntentPlan_Op<"exec_mask", [Pure]> {
  let arguments = (ins AnyType:$value, AnyType:$predicate, AnyType:$fill);
  let results = (outs AnyType:$result);
}

def IntentPlan_ExecReduceOp : IntentPlan_Op<"exec_reduce", [Pure]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
}
def IntentPlan_ExecScanOp : IntentPlan_Op<"exec_scan", [Pure]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
}
def IntentPlan_ExecContractOp
    : IntentPlan_ExecBinaryOneResultOp<"exec_contract">;
def IntentPlan_ExecScaledContractOp
    : IntentPlan_Op<"exec_scaled_contract", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, AnyType:$lhs_scale,
                        AnyType:$rhs_scale);
  let results = (outs AnyType:$result);
}
def IntentPlan_ExecSparseContractOp
    : IntentPlan_ExecVariadicOneResultOp<"exec_sparse_contract">;

def IntentPlan_ExecRaggedOp : IntentPlan_Op<"exec_ragged", [Pure]> {
  let arguments = (ins AnyType:$outer, AnyType:$members, AnyType:$offsets,
                        Optional<AnyType>:$indices);
  let results = (outs AnyType:$result);
}
def IntentPlan_ExecRaggedOuterOp
    : IntentPlan_ExecUnaryOneResultOp<"exec_ragged_outer">;
def IntentPlan_ExecRaggedMemberOp
    : IntentPlan_ExecBinaryOneResultOp<"exec_ragged_member">;
def IntentPlan_ExecMembersOp
    : IntentPlan_ExecUnaryOneResultOp<"exec_members">;

def IntentPlan_ExecGatherOp
    : IntentPlan_ExecVariadicOneResultOp<"exec_gather",
                                                [MemoryEffects<[MemRead]>]>;
def IntentPlan_ExecScatterUniqueOp
    : IntentPlan_ExecVariadicNoResultOp<"exec_scatter_unique",
                                               [MemoryEffects<[MemWrite]>]>;
def IntentPlan_ExecScatterReduceOp
    : IntentPlan_ExecVariadicNoResultOp<"exec_scatter_reduce",
                                               [MemoryEffects<[MemWrite]>]>;

def IntentPlan_ExecBufferOp
    : IntentPlan_ExecVariadicOneResultOp<"exec_buffer">;
def IntentPlan_ExecBufferLoadOp
    : IntentPlan_ExecVariadicOneResultOp<"exec_buffer_load",
                                                [MemoryEffects<[MemRead]>]>;
def IntentPlan_ExecBufferStoreOp
    : IntentPlan_ExecVariadicNoResultOp<"exec_buffer_store",
                                               [MemoryEffects<[MemWrite]>]>;
def IntentPlan_ExecAtomicAddOp
    : IntentPlan_ExecVariadicOneResultOp<"exec_atomic_add",
                                                [MemoryEffects<[MemRead, MemWrite]>]>;
def IntentPlan_ExecAtomicCasOp
    : IntentPlan_ExecVariadicOneResultOp<"exec_atomic_cas",
                                                [MemoryEffects<[MemRead, MemWrite]>]>;
def IntentPlan_ExecRandomOp : IntentPlan_Op<"exec_random", [Pure]> {
  let arguments = (ins AnyType:$seed, AnyType:$logical_index);
  let results = (outs AnyType:$result);
}
def IntentPlan_ExecReturnOp
    : IntentPlan_Op<"exec_return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
}

def IntentPlan_YieldOp : IntentPlan_Op<"yield", [Terminator]> {
  let arguments = (ins);
  let results = (outs);
}

def IntentPlan_ProgramOp : IntentPlan_Op<"program", [
    IsolatedFromAbove,
    SymbolTable,
    SingleBlock,
    SingleBlockImplicitTerminator<"::intent::plan::YieldOp">
  ]> {
  let arguments = (ins
    FlatSymbolRefAttr:$entry,
    StrAttr:$target
  );
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def IntentPlan_SearchSpaceOp : IntentPlan_Op<"search_space", [
    SingleBlock,
    SingleBlockImplicitTerminator<"::intent::plan::YieldOp">
  ]> {
  let arguments = (ins
    FlatSymbolRefAttr:$entry,
    StrAttr:$target
  );
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def IntentPlan_TargetProgramOp : IntentPlan_Op<"target_program"> {
  let arguments = (ins
    StrAttr:$provider,
    StrAttr:$source
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_DeviceOp : IntentPlan_Op<"device"> {
  let arguments = (ins I64Attr:$device);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_AxisOp : IntentPlan_Op<"axis"> {
  let arguments = (ins
    I64Attr:$node,
    ArrayAttr:$roles,
    OptionalAttr<I64Attr>:$program_order,
    OptionalAttr<I64Attr>:$worker_axis,
    OptionalAttr<I64Attr>:$fold_order,
    BoolAttr:$reuse_worker,
    OptionalAttr<StrAttr>:$group
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_RangeOp : IntentPlan_Op<"range"> {
  let arguments = (ins
    I64Attr:$axis_node,
    StrAttr:$purpose,
    I64Attr:$level,
    StrAttr:$tile,
    StrAttr:$extent,
    OptionalAttr<I64Attr>:$transfer_node,
    OptionalAttr<I64Attr>:$source_axis,
    OptionalAttr<I64Attr>:$divisor,
    OptionalAttr<I64Attr>:$offset
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_RegionBindingOp : IntentPlan_Op<"region_binding"> {
  let arguments = (ins
    I64Attr:$value,
    I64Attr:$axis_node,
    StrAttr:$purpose,
    I64Attr:$level
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_DomainExtentBindingOp
    : IntentPlan_Op<"domain_extent_binding"> {
  let arguments = (ins
    I64Attr:$axis_node,
    I64Attr:$start_value,
    I64Attr:$stop_value,
    OptionalAttr<I64Attr>:$step_value
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_PartitionBindingOp : IntentPlan_Op<"partition_binding"> {
  let arguments = (ins
    I64Attr:$partition_node,
    I64Attr:$axis_node,
    I64Attr:$count_value,
    I64Attr:$part_argument,
    I64Attr:$region_argument,
    StrAttr:$segment_extent
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_LaunchOp : IntentPlan_Op<"launch"> {
  let arguments = (ins
    OptionalAttr<I64Attr>:$loop_node,
    BoolAttr:$persistent
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_BlockExtentOp : IntentPlan_Op<"block_extent"> {
  let arguments = (ins
    StrAttr:$logical_extent,
    StrAttr:$rounding,
    StrAttr:$fill
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_BufferOp : IntentPlan_Op<"buffer"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$space,
    DenseI64ArrayAttr:$owner_nodes
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_PaddingOp : IntentPlan_Op<"padding"> {
  let arguments = (ins
    I64Attr:$value,
    DenseI64ArrayAttr:$tensor_axes,
    DenseI64ArrayAttr:$domain_nodes,
    StrAttr:$fill
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_TransferOp : IntentPlan_Op<"transfer"> {
  let arguments = (ins
    I64Attr:$node,
    DenseI64ArrayAttr:$domain_nodes,
    DenseI64ArrayAttr:$validity_tensor_axes,
    DenseI64ArrayAttr:$validity_domain_nodes,
    StrAttr:$fill,
    BoolAttr:$consumer_neutralized,
    StrAttr:$materialization,
    StrAttr:$tensor_indexing,
    StrAttr:$coverage_space,
    StrAttr:$result_space
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_ReductionOp : IntentPlan_Op<"reduction"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$result_space
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_ScanOp : IntentPlan_Op<"scan"> {
  let arguments = (ins
    I64Attr:$node,
    I64Attr:$axis_node,
    I64Attr:$tensor_axis,
    StrAttr:$result_space,
    StrAttr:$carry_space,
    StrAttr:$materialization,
    DenseI64ArrayAttr:$owner_nodes,
    DenseI64ArrayAttr:$producers,
    DenseI64ArrayAttr:$materialized_values
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_PointwiseOp : IntentPlan_Op<"pointwise"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$result_space,
    BoolAttr:$nonnegative_operands,
    DenseI64ArrayAttr:$axis_nodes
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_ContractOp : IntentPlan_Op<"contract"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$form,
    OptionalAttr<I64Attr>:$reduction_axis_node,
    OptionalAttr<I64Attr>:$lhs_result_axis_node,
    OptionalAttr<I64Attr>:$rhs_result_axis_node,
    StrAttr:$lhs_space,
    StrAttr:$rhs_space,
    StrAttr:$accumulator_space,
    StrAttr:$accumulator_flow,
    OptionalAttr<I64Attr>:$accumulator_owner_node,
    OptionalAttr<I64Attr>:$accumulator_update_node,
    OptionalAttr<I64Attr>:$accumulator_value,
    OptionalAttr<I64Attr>:$accumulator_input_value,
    OptionalAttr<I64Attr>:$accumulator_conditional_node,
    OptionalAttr<I64Attr>:$accumulator_conditional_result,
    BoolAttr:$producer_replay
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_SparseContractOp : IntentPlan_Op<"sparse_contract"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$format,
    I64Attr:$row_axis_node,
    I64Attr:$column_axis_node,
    I64Attr:$reduction_axis_node,
    StrAttr:$compressed_space,
    StrAttr:$metadata_space,
    StrAttr:$rhs_space,
    StrAttr:$accumulator_space
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_StreamAxisOp : IntentPlan_Op<"stream_axis"> {
  let arguments = (ins
    I64Attr:$stream_node,
    I64Attr:$axis_node
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_StreamBindingOp : IntentPlan_Op<"stream_binding"> {
  let arguments = (ins
    I64Attr:$stream_node,
    I64Attr:$axis_node,
    StrAttr:$purpose,
    I64Attr:$level,
    OptionalAttr<I64Attr>:$stop_value,
    OptionalAttr<I64Attr>:$relation_node,
    OptionalAttr<I64Attr>:$partition_node
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_AutotuneOp : IntentPlan_Op<"autotune"> {
  let arguments = (ins
    ArrayAttr:$key,
    ArrayAttr:$parameters
  );
  let results = (outs);
  let hasVerifier = 1;
}

#endif
