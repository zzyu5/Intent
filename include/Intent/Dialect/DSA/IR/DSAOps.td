#ifndef INTENT_DIALECT_DSA_IR_DSAOPS_TD
#define INTENT_DIALECT_DSA_IR_DSAOPS_TD
include "Intent/Dialect/DSA/IR/DSAAttrs.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
class IntentDSA_Op<string name, list<Trait> traits = []>
    : Op<IntentDSA_Dialect, name, traits> { let hasVerifier = 1; }
def IntentDSA_SynchronizeOp : IntentDSA_Op<"synchronize"> {
  let summary = "Wait for the current work unit's selected streams";
  let arguments = (ins DefaultValuedAttr<BoolAttr, "false">:$local_only);
}
def IntentDSA_TaskIdOp : IntentDSA_Op<"task_id", [Pure]> {
  let results = (outs Index:$result);
}
def IntentDSA_TaskCountOp : IntentDSA_Op<"task_count", [Pure]> {
  let results = (outs Index:$result);
}
def IntentDSA_GroupIdOp : IntentDSA_Op<"group_id", [Pure]> {
  let results = (outs Index:$result);
}
def IntentDSA_GroupCountOp : IntentDSA_Op<"group_count", [Pure]> {
  let results = (outs Index:$result);
}
def IntentDSA_LocalIdOp : IntentDSA_Op<"local_id", [Pure]> {
  let results = (outs Index:$result);
}
def IntentDSA_IsMemoryCoreOp : IntentDSA_Op<"is_memory_core", [Pure]> {
  let results = (outs I1:$result);
}
def IntentDSA_GroupSynchronizeOp : IntentDSA_Op<"group_synchronize"> {
  let summary = "Complete local and shared transfers before all execution-group participants continue";
}
def IntentDSA_StageTileOp : IntentDSA_Op<"stage_tile"> {
  let summary = "Memory participant stages the active external rectangle into group-local storage";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$source,
    Arg<AnyMemRef, "Inactive shared elements remain uninitialized", [MemWrite]>:$output,
    Index:$offset, Index:$row_stride, Index:$column_stride, Index:$rows, Index:$columns);
}
def IntentDSA_StrideOp : IntentDSA_Op<"stride", [Pure]> {
  let arguments = (ins AnyMemRef:$source, I64Attr:$axis);
  let results = (outs Index:$result);
}
def IntentDSA_LoadScalarOp : IntentDSA_Op<"load_scalar"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$source, Index:$offset);
  let results = (outs AnyType:$result);
}
def IntentDSA_StoreScalarOp : IntentDSA_Op<"store_scalar"> {
  let arguments = (ins AnyType:$value, Arg<AnyMemRef, "", [MemWrite]>:$destination, Index:$offset);
}
def IntentDSA_LoadTileOp : IntentDSA_Op<"load_tile"> {
  let summary = "Load a zero-padded local tile; asynchronous external loads complete at the next work-unit synchronization";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$source,
    Arg<AnyMemRef, "", [MemWrite]>:$output, Index:$offset,
    Index:$row_stride, Index:$column_stride, Index:$rows, Index:$columns,
    DefaultValuedAttr<BoolAttr, "false">:$asynchronous);
}
def IntentDSA_GatherPlanOp : IntentDSA_Op<"gather_plan"> {
  let summary = "Prepare reusable contiguous row runs from one offset snapshot";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$row_offsets, Index:$rows,
    Arg<AnyMemRef, "Exact f32 row indices", [MemRead]>:$lane_indices,
    Arg<AnyMemRef, "Run descriptor and construction workspace", [MemWrite]>:$output);
}
def IntentDSA_GatherRowsOp : IntentDSA_Op<"gather_rows"> {
  let summary = "Gather external row intervals into a dense local tile using explicit element offsets";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$source,
    Arg<AnyMemRef, "", [MemRead]>:$row_offsets,
    Arg<AnyMemRef, "", [MemWrite]>:$output,
    Index:$column_stride, Index:$rows, Index:$columns,
    Arg<Optional<AnyMemRef>, "Read-only run descriptor for the same offset snapshot", [MemRead]>:$plan,
    DefaultValuedAttr<BoolAttr, "false">:$asynchronous);
}
def IntentDSA_GroupGatherRowsOp : IntentDSA_Op<"group_gather_rows"> {
  let summary = "Four adjacent work owners collectively supply row intervals through shared storage";
  let description = [{
    All group participants execute this operation in uniform control. Prior
    local transfers complete before publication begins. Compute
    lane zero publishes its row offsets and run plan; the memory participant
    fetches four adjacent column intervals, and lane i receives interval i.
    Construction proves that the four logical source relations differ only by
    i times the local column extent. Transfers complete before return, including
    every lane's consumption of shared storage. Inactive local rows are zero.
  }];
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$source,
    Arg<AnyMemRef, "Lane-zero offset snapshot", [MemRead]>:$row_offsets,
    Arg<AnyMemRef, "Lane-zero run descriptor", [MemRead]>:$plan,
    Arg<AnyMemRef, "", [MemWrite]>:$output,
    Arg<AnyMemRef, "Four adjacent column intervals", [MemRead, MemWrite]>:$shared_data,
    Arg<AnyMemRef, "Published offsets and descriptor", [MemRead, MemWrite]>:$shared_metadata,
    Index:$rows, Index:$lane);
}
def IntentDSA_StoreTileOp : IntentDSA_Op<"store_tile"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$destination, Index:$offset,
    Index:$row_stride, Index:$column_stride, Index:$rows, Index:$columns);
}
def IntentDSA_FillOp : IntentDSA_Op<"fill"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemWrite]>:$output, AnyType:$value);
}
def IntentDSA_IotaOp : IntentDSA_Op<"iota"> {
  let summary = "Materialize exact consecutive local lane indices starting at zero";
  let arguments = (ins Arg<AnyMemRef, "", [MemWrite]>:$output);
}
def IntentDSA_IndexLayoutOp : IntentDSA_Op<"index_layout"> {
  let summary = "Convert exact i64 indices between interleaved words and two dense i32 planes, or splat one index";
  let arguments = (ins Arg<AnyType, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$output);
}
def IntentDSA_BroadcastRowsOp : IntentDSA_Op<"broadcast_rows"> {
  let summary = "Broadcast contiguous row scalars through an explicit transposed staging tile";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$output,
    Arg<AnyMemRef, "", [MemWrite]>:$scratch,
    Index:$offset, Index:$rows, Index:$columns);
}
def IntentDSA_IndexBinaryOp : IntentDSA_Op<"index_binary"> {
  let summary = "Exact modular i64 arithmetic on separate low and high word planes";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyType, "", [MemRead]>:$rhs,
    Arg<AnyMemRef, "", [MemWrite]>:$output,
    Intent_BinaryOperatorAttr:$kind);
}
def IntentDSA_TransposeOp : IntentDSA_Op<"transpose"> {
  let summary = "Transpose the active rectangle between independent local tiles";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$output, Index:$rows, Index:$columns);
}
def IntentDSA_SelectOp : IntentDSA_Op<"select"> {
  let arguments = (ins AnyType:$condition, Arg<AnyMemRef, "", [MemRead]>:$true_value,
    Arg<AnyType, "", [MemRead]>:$false_value, Arg<AnyMemRef, "", [MemWrite]>:$output,
    Arg<Optional<AnyMemRef>, "", [MemWrite]>:$scratch);
}
def IntentDSA_MaskedFillOp : IntentDSA_Op<"masked_fill"> {
  let summary = "Retain f32 input bits under a full-width i32 predicate mask, otherwise use the scalar fill";
  let arguments = (ins Arg<AnyMemRef, "Each predicate is zero or all one bits", [MemRead]>:$mask,
    Arg<AnyMemRef, "", [MemRead]>:$input, F32:$value,
    Arg<AnyMemRef, "", [MemWrite]>:$output);
}
def IntentDSA_UnaryOp : IntentDSA_Op<"unary"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$output,
    Intent_UnaryOperatorAttr:$kind, BoolAttr:$approximate, BoolAttr:$flush_to_zero,
    Arg<Optional<AnyMemRef>, "", [MemWrite]>:$scratch);
}
def IntentDSA_BinaryOp : IntentDSA_Op<"binary"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyType, "", [MemRead]>:$rhs, Arg<AnyMemRef, "", [MemWrite]>:$output,
    Intent_BinaryOperatorAttr:$kind, BoolAttr:$approximate, BoolAttr:$flush_to_zero,
    Arg<Optional<AnyMemRef>, "Workspace for destructive target primitives", [MemWrite]>:$scratch);
}
def IntentDSA_CompareOp : IntentDSA_Op<"compare"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyMemRef, "", [MemRead]>:$rhs, Arg<AnyMemRef, "", [MemWrite]>:$output,
    Intent_ComparePredicateAttr:$predicate,
    Arg<Optional<AnyMemRef>, "Long-tile split-word comparison workspace", [MemWrite]>:$scratch);
}
def IntentDSA_CompareRangeOp : IntentDSA_Op<"compare_range"> {
  let summary = "Compare row coordinates against a consecutive column range with zero-padded inactive rows";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$row_coordinates,
    Arg<AnyMemRef, "", [MemWrite]>:$output, Index:$rows, I64Attr:$base);
}
def IntentDSA_CompareRampOp : IntentDSA_Op<"compare_ramp"> {
  let summary = "Compare implicit unit-step coordinates into i1 predicates or full-width i32 masks";
  let arguments = (ins Index:$row_begin, Arg<AnyMemRef, "", [MemWrite]>:$output,
    Arg<Optional<AnyMemRef>, "Workspace for the i1 representation", [MemWrite]>:$scratch, I64Attr:$base);
}
def IntentDSA_CastOp : IntentDSA_Op<"cast"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$output);
}
def IntentDSA_DivideRNOp : IntentDSA_Op<"divide_rn"> {
  let summary = "Correctly rounded local f32 division with explicit integer correction workspace";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyType, "", [MemRead]>:$rhs,
    Arg<AnyMemRef, "", [MemWrite]>:$output,
    Arg<AnyMemRef, "", [MemWrite]>:$scratch,
    Arg<AnyMemRef, "", [MemRead]>:$lane_indices);
}
def IntentDSA_DivideCastOp : IntentDSA_Op<"divide_cast"> {
  let summary = "Strict f32 division followed by an RN-even f16 cast with explicit local workspace";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyType, "", [MemRead]>:$rhs,
    Arg<AnyMemRef, "", [MemWrite]>:$output,
    Arg<AnyMemRef, "", [MemWrite]>:$quotient,
    Arg<AnyMemRef, "", [MemWrite]>:$bounds,
    Arg<AnyMemRef, "", [MemWrite]>:$accepted,
    Arg<AnyMemRef, "", [MemWrite]>:$narrow_bounds,
    Arg<AnyMemRef, "Precomputed exact f32 physical lane indices", [MemRead]>:$lane_indices);
}
def IntentDSA_ReduceOp : IntentDSA_Op<"reduce"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$output,
    Arg<AnyMemRef, "", [MemWrite]>:$scratch, Index:$count,
    AnyFloat:$identity, Intent_BinaryOperatorAttr:$kind,
    DefaultValuedAttr<I64Attr, "1">:$axis);
}
def IntentDSA_SliceReduceOp : IntentDSA_Op<"slice_reduce", [
    AttrSizedOperandSegments, SingleBlock, IsolatedFromAbove,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Reduce local source members with a closed typed combine region";
  let description = [{
    Sources and outputs are complete contiguous local views retaining their
    logical capacity axes. Counts concatenate the active extents of each source,
    in source and then axis order. All sources share the selected member extents;
    each field retains its own free-axis shape. Initials are scalar fills or
    complete state views. The combine takes two component tuples followed by
    explicit captures and yields one component tuple. Its borrowed inputs are
    read-only; any writes are confined to helper-local scratch. Reduce permits
    associative and commutative combination; empty membership returns initials.
  }];
  let arguments = (ins Variadic<AnyMemRef>:$sources,
    Variadic<AnyType>:$initials, Variadic<AnyType>:$captures,
    Variadic<AnyMemRef>:$outputs, Variadic<Index>:$counts,
    DenseI64ArrayAttr:$axes);
  let regions = (region AnyRegion:$combine);
}
def IntentDSA_ScanOp : IntentDSA_Op<"scan", [
    AttrSizedOperandSegments, SingleBlock, IsolatedFromAbove,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Scan local source members and publish their final component state";
  let description = [{
    Uses the same source, active-count and closed combine schema as slice_reduce.
    Outputs preserve every source capacity axis; finals remove the scan axis.
    The combine preserves member order and may reassociate. Inclusive and reverse
    are observable prefix semantics. Empty membership leaves outputs empty and
    writes initials to finals. Helper shape values are ordinary explicit captures.
  }];
  let arguments = (ins Variadic<AnyMemRef>:$sources,
    Variadic<AnyType>:$initials, Variadic<AnyType>:$captures,
    Variadic<AnyMemRef>:$outputs, Variadic<Index>:$counts,
    Variadic<AnyMemRef>:$finals, I64Attr:$axis, BoolAttr:$inclusive,
    BoolAttr:$reverse);
  let regions = (region AnyRegion:$combine);
}
def IntentDSA_CollectiveYieldOp : IntentDSA_Op<"collective_yield", [
    Pure, Terminator, ReturnLike]> {
  let arguments = (ins Variadic<AnyType>:$values);
}
def IntentDSA_PrepareMatrixOp : IntentDSA_Op<"prepare_matrix", [AttrSizedOperandSegments]> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$output,
    Arg<Optional<AnyMemRef>, "", [MemWrite]>:$scratch,
    Arg<Optional<AnyMemRef>, "", [MemWrite]>:$reshaped,
    DefaultValuedAttr<BoolAttr, "false">:$input_transposed);
}
def IntentDSA_PrepareMatrixViewOp : IntentDSA_Op<"prepare_matrix_view"> {
  let summary = "Prepare a resident matrix filter from an external panel using bounded local slices";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$source,
    Arg<AnyMemRef, "", [MemWrite]>:$output,
    Arg<AnyMemRef, "", [MemWrite]>:$input_slice,
    Arg<AnyMemRef, "", [MemWrite]>:$transposed_slice,
    Index:$offset, Index:$row_stride, Index:$column_stride, Index:$columns);
}
def IntentDSA_MatMulOp : IntentDSA_Op<"matmul"> {
  let summary = "Accumulate a bounded logical MatMul within local tensor storage";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyMemRef, "", [MemRead]>:$rhs,
    Arg<AnyMemRef, "", [MemRead, MemWrite]>:$accumulator,
    Index:$rows, Index:$depth, Index:$columns,
    DefaultValuedAttr<BoolAttr, "false">:$rhs_transposed);
}
def IntentDSA_MatrixTileOp : IntentDSA_Op<"matrix_tile"> {
  let summary = "Compute one selected matrix tile, optionally accumulating into the existing result";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyMemRef, "", [MemRead]>:$rhs,
    Arg<AnyMemRef, "", [MemRead, MemWrite]>:$accumulator,
    DefaultValuedAttr<BoolAttr, "true">:$accumulate);
}
#endif
