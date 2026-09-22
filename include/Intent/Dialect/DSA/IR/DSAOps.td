#ifndef INTENT_DIALECT_DSA_IR_DSAOPS_TD
#define INTENT_DIALECT_DSA_IR_DSAOPS_TD
include "Intent/Dialect/DSA/IR/DSAAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
class IntentDSA_Op<string name, list<Trait> traits = []>
    : Op<IntentDSA_Dialect, name, traits> { let hasVerifier = 1; }
def IntentDSA_SynchronizeOp : IntentDSA_Op<"synchronize"> {
  let summary = "Wait for the current work unit's local transfer and compute operations";
}
def IntentDSA_TaskIdOp : IntentDSA_Op<"task_id", [Pure]> {
  let results = (outs Index:$result);
}
def IntentDSA_TaskCountOp : IntentDSA_Op<"task_count", [Pure]> {
  let results = (outs Index:$result);
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
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$source,
    Arg<AnyMemRef, "", [MemWrite]>:$output, Index:$offset,
    Index:$row_stride, Index:$column_stride, Index:$rows, Index:$columns);
}
def IntentDSA_StoreTileOp : IntentDSA_Op<"store_tile"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$destination, Index:$offset,
    Index:$row_stride, Index:$column_stride, Index:$rows, Index:$columns);
}
def IntentDSA_FillOp : IntentDSA_Op<"fill"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemWrite]>:$output, AnyType:$value);
}
def IntentDSA_SelectOp : IntentDSA_Op<"select"> {
  let arguments = (ins AnyType:$condition, Arg<AnyMemRef, "", [MemRead]>:$true_value,
    Arg<AnyMemRef, "", [MemRead]>:$false_value, Arg<AnyMemRef, "", [MemWrite]>:$output);
}
def IntentDSA_UnaryOp : IntentDSA_Op<"unary"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$output,
    Intent_UnaryOperatorAttr:$kind, BoolAttr:$approximate, BoolAttr:$flush_to_zero);
}
def IntentDSA_BinaryOp : IntentDSA_Op<"binary"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyMemRef, "", [MemRead]>:$rhs, Arg<AnyMemRef, "", [MemWrite]>:$output,
    Intent_BinaryOperatorAttr:$kind, BoolAttr:$approximate, BoolAttr:$flush_to_zero);
}
def IntentDSA_CastOp : IntentDSA_Op<"cast"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$output);
}
def IntentDSA_ReduceOp : IntentDSA_Op<"reduce"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$output,
    Arg<AnyMemRef, "", [MemWrite]>:$scratch, Index:$count,
    AnyFloat:$identity, Intent_BinaryOperatorAttr:$kind);
}
def IntentDSA_PrepareMatrixOp : IntentDSA_Op<"prepare_matrix"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$input,
    Arg<AnyMemRef, "", [MemWrite]>:$output,
    Arg<AnyMemRef, "", [MemWrite]>:$scratch);
}
def IntentDSA_MatMulOp : IntentDSA_Op<"matmul"> {
  let summary = "Accumulate a bounded logical MatMul within local tensor storage";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyMemRef, "", [MemRead]>:$rhs,
    Arg<AnyMemRef, "", [MemRead, MemWrite]>:$accumulator,
    Index:$rows, Index:$depth, Index:$columns);
}
def IntentDSA_MatrixTileOp : IntentDSA_Op<"matrix_tile"> {
  let summary = "Accumulate one selected matrix tile with prepared operands";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyMemRef, "", [MemRead]>:$rhs,
    Arg<AnyMemRef, "", [MemRead, MemWrite]>:$accumulator,
    Arg<AnyMemRef, "", [MemWrite]>:$scratch);
}
#endif
