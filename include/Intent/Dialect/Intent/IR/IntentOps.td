#ifndef INTENT_DIALECT_INTENT_IR_INTENTOPS_TD
#define INTENT_DIALECT_INTENT_IR_INTENTOPS_TD

include "Intent/Dialect/Intent/IR/IntentDialect.td"
include "Intent/Dialect/Intent/IR/IntentAttrs.td"
include "Intent/Dialect/Intent/IR/IntentTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Intent_Op<string mnemonic, list<Trait> traits = []>
    : Op<Intent_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}

class Intent_UnaryDataOp<string mnemonic, list<Trait> traits = [Pure]>
    : Intent_Op<mnemonic, traits> {
  let arguments = (ins Intent_DataValue:$input);
  let results = (outs Intent_DataValue:$result);
}

class Intent_BinaryDataOp<string mnemonic, list<Trait> traits = [Pure]>
    : Intent_Op<mnemonic, traits> {
  let arguments = (ins Intent_DataValue:$lhs, Intent_DataValue:$rhs);
  let results = (outs Intent_DataValue:$result);
}

def Intent_ConstantOp : Intent_Op<"constant", [Pure]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs Intent_ScalarValue:$result);
}

def Intent_DimOp : Intent_Op<"dim", [Pure]> {
  let arguments = (ins Intent_Value:$source, I64Attr:$axis,
                       I64Attr:$dimension);
  let results = (outs Index:$result);
}

def Intent_DomainOp : Intent_Op<"domain", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$bounds,
                       ArrayAttr:$extent_dimensions);
  let results = (outs Intent_DomainValue:$result);
}
def Intent_DomainProductOp : Intent_Op<"domain_product", [Pure]> {
  let arguments = (ins Variadic<Intent_DomainValue>:$domains,
                       ArrayAttr:$extent_dimensions);
  let results = (outs Intent_DomainValue:$result);
}
def Intent_SubregionOp : Intent_Op<"subregion", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs,
                       BoolAttr:$has_start, BoolAttr:$has_stop,
                       ArrayAttr:$extent_dimensions);
  let results = (outs Intent_RegionValue:$result);
}
def Intent_IndicesOp : Intent_Op<"indices", [Pure]> {
  let arguments = (ins Intent_Value:$source, OptionalAttr<I64Attr>:$tensor_axis);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_RegionEndOp : Intent_Op<"region_end", [Pure]> {
  let arguments = (ins Intent_IterationValue:$source);
  let results = (outs Index:$result);
}
def Intent_AssumeInBoundsOp : Intent_Op<"assume_in_bounds", [Pure]> {
  let arguments = (ins Intent_Value:$index, Intent_AddressableValue:$view,
                       I64Attr:$axis);
  let results = (outs);
}

def Intent_ParallelOp : Intent_Op<"parallel", [RecursiveMemoryEffects]> {
  let arguments = (ins Intent_IterationValue:$source);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
}
def Intent_IfOp : Intent_Op<"if", [RecursiveMemoryEffects]> {
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<Intent_Value>:$results);
  let regions = (region SizedRegion<1>:$then_region,
                         SizedRegion<1>:$else_region);
}
def Intent_ForOp : Intent_Op<"for", [RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs);
  let results = (outs Variadic<Intent_Value>:$results);
  let regions = (region SizedRegion<1>:$body);
}
def Intent_WhileOp : Intent_Op<"while", [RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs);
  let results = (outs Variadic<Intent_Value>:$results);
  let regions = (region SizedRegion<1>:$before, SizedRegion<1>:$after);
}
def Intent_YieldOp : Intent_Op<"yield", [Terminator]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs);
  let results = (outs);
}
def Intent_ConditionOp : Intent_Op<"condition", [Terminator]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs);
  let results = (outs);
}

def Intent_ViewLoadOp : Intent_Op<"view_load", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_IndexRelation:$index,
                       OptionalAttr<I64Attr>:$valid_operand_index,
                       OptionalAttr<I64Attr>:$fill_operand_index);
  let results = (outs Intent_DataValue:$result);
}
def Intent_ViewStoreOp : Intent_Op<"view_store", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_IndexRelation:$index,
                       I64Attr:$value_operand_index);
  let results = (outs);
}
def Intent_ReshapeOp : Intent_Op<"reshape", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_ShapeRelation:$shape);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_JoinOp : Intent_BinaryDataOp<"join">;
def Intent_TransposeOp : Intent_UnaryDataOp<"transpose"> {
  let arguments = (ins Intent_DataValue:$input, ArrayAttr:$permutation);
}
def Intent_BroadcastOp : Intent_Op<"broadcast", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_ShapeRelation:$shape);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_FullOp : Intent_Op<"full", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_ShapeRelation:$shape);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_MakeTupleOp : Intent_Op<"make_tuple", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$components);
  let results = (outs Intent_TupleValue:$result);
}
def Intent_MakeRecordOp : Intent_Op<"make_record", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$fields);
  let results = (outs Intent_RecordValue:$result);
}
def Intent_ExtractOp : Intent_Op<"extract", [Pure]> {
  let arguments = (ins AnyTypeOf<[Intent_TupleValue, Intent_RecordValue]>:$product,
                       I64Attr:$field);
  let results = (outs Intent_Value:$result);
}

def Intent_UnaryOp : Intent_UnaryDataOp<"unary"> {
  let arguments = (ins Intent_DataValue:$input,
                       Intent_UnaryOperatorAttr:$operator_kind,
                       DefaultValuedAttr<BoolAttr, "false">:$approximate,
                       DefaultValuedAttr<BoolAttr, "false">:$flush_to_zero);
}
def Intent_BinaryOp : Intent_BinaryDataOp<"binary"> {
  let arguments = (ins Intent_DataValue:$lhs, Intent_DataValue:$rhs,
                       Intent_BinaryOperatorAttr:$operator_kind,
                       DefaultValuedAttr<BoolAttr, "false">:$approximate,
                       DefaultValuedAttr<BoolAttr, "false">:$flush_to_zero);
}
def Intent_CompareOp : Intent_Op<"compare", [Pure]> {
  let arguments = (ins Intent_DataValue:$lhs, Intent_DataValue:$rhs,
                       Intent_ComparePredicateAttr:$predicate);
  let results = (outs Intent_DataValue:$result);
}
def Intent_SelectOp : Intent_Op<"select", [Pure]> {
  let arguments = (ins Intent_DataValue:$condition,
                       Intent_DataValue:$true_value,
                       Intent_DataValue:$false_value);
  let results = (outs Intent_DataValue:$result);
}
def Intent_CastOp : Intent_UnaryDataOp<"cast"> {
  let arguments = (ins Intent_DataValue:$input,
                       OptionalAttr<I64Attr>:$rounding);
}
def Intent_BitcastOp : Intent_UnaryDataOp<"bitcast">;
def Intent_MaskOp : Intent_Op<"mask", [Pure]> {
  let arguments = (ins Intent_DataValue:$value,
                       Intent_DataValue:$predicate,
                       Intent_DataValue:$fill);
  let results = (outs Intent_DataValue:$result);
}

def Intent_ReduceOp : Intent_Op<"reduce", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs,
                       ArrayAttr:$axes,
                       I64Attr:$source_count,
                       I64Attr:$identity_count,
                       I64Attr:$capture_count);
  let results = (outs Variadic<Intent_Value>:$results);
  let regions = (region SizedRegion<1>:$combine);
}
def Intent_ScanOp : Intent_Op<"scan", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs,
                       I64Attr:$axis,
                       BoolAttr:$inclusive,
                       BoolAttr:$reverse,
                       I64Attr:$source_count,
                       I64Attr:$identity_count,
                       I64Attr:$capture_count);
  let results = (outs Variadic<Intent_Value>:$results);
  let regions = (region SizedRegion<1>:$combine);
}
def Intent_RegionFoldOp : Intent_Op<"region_fold", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs,
                       I64Attr:$axis,
                       I64Attr:$source_count,
                       I64Attr:$identity_count,
                       I64Attr:$capture_count);
  let results = (outs Variadic<Intent_Value>:$results);
  let regions = (region SizedRegion<1>:$summarize,
                         SizedRegion<1>:$combine);
}
def Intent_RegionScanOp : Intent_Op<"region_scan", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs,
                       I64Attr:$axis,
                       I64Attr:$source_count,
                       I64Attr:$identity_count,
                       I64Attr:$state_count,
                       I64Attr:$capture_count,
                       I64Attr:$output_count);
  let results = (outs Variadic<Intent_Value>:$results);
  let regions = (region SizedRegion<1>:$summarize,
                         SizedRegion<1>:$combine,
                         SizedRegion<1>:$apply,
                         SizedRegion<1>:$emit);
}

def Intent_ContractOp : Intent_Op<"contract", [Pure]> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
                       ArrayAttr:$reduce, ArrayAttr:$batch);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_QuantizeOp : Intent_Op<"quantize", [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, Intent_QuantFormatAttr:$format);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_QuantizedDotOp : Intent_Op<"quantized_dot", [Pure]> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
                       Intent_QuantFormatAttr:$lhs_format,
                       Intent_QuantFormatAttr:$rhs_format);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_ScaledContractOp : Intent_Op<"scaled_contract", [Pure]> {
  let arguments = (ins AnyRankedTensor:$lhs,
                       AnyRankedTensor:$lhs_scale,
                       AnyRankedTensor:$rhs,
                       AnyRankedTensor:$rhs_scale,
                       ArrayAttr:$reduce,
                       ArrayAttr:$batch,
                       Intent_ScaledFormatAttr:$lhs_format,
                       Intent_ScaledFormatAttr:$rhs_format,
                       I64Attr:$lhs_group_size,
                       I64Attr:$rhs_group_size);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_SparseContractOp : Intent_Op<"sparse_contract", [Pure]> {
  let arguments = (ins AnyRankedTensor:$compressed,
                       Intent_Value:$metadata,
                       AnyRankedTensor:$rhs,
                       Index:$logical_extent,
                       Intent_SparseFormat:$format,
                       ArrayAttr:$reduce,
                       ArrayAttr:$batch);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_HistogramOp : Intent_Op<"histogram", [Pure]> {
  let arguments = (ins AnyRankedTensor:$values,
                       Index:$bins,
                       AnyRankedTensor:$valid,
                       I64Attr:$result_dimension);
  let results = (outs AnyRankedTensor:$result);
}

def Intent_GatherOp : Intent_Op<"gather", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_IndexRelation:$index,
                       OptionalAttr<I64Attr>:$valid_operand_index,
                       OptionalAttr<I64Attr>:$fill_operand_index);
  let results = (outs Intent_DataValue:$result);
}
def Intent_ScatterUniqueOp : Intent_Op<"scatter_unique", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_IndexRelation:$index,
                       I64Attr:$value_operand_index);
  let results = (outs);
}
def Intent_ScatterReduceOp : Intent_Op<"scatter_reduce", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_IndexRelation:$index,
                       I64Attr:$value_operand_index);
  let results = (outs);
  let regions = (region SizedRegion<1>:$combine);
}

def Intent_BufferOp : Intent_Op<"buffer"> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_ShapeRelation:$shape,
                       OptionalAttr<I64Attr>:$initial_operand);
  let results = (outs Intent_BufferValue:$result);
}
def Intent_BufferLoadOp : Intent_Op<"buffer_load", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_IndexRelation:$index);
  let results = (outs Intent_DataValue:$result);
}
def Intent_BufferStoreOp : Intent_Op<"buffer_store", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_IndexRelation:$index,
                       I64Attr:$value_operand_index);
  let results = (outs);
}
def Intent_AtomicLoadOp : Intent_Op<"atomic_load", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_IndexRelation:$index,
                       Intent_AtomicOrderingAttr:$ordering);
  let results = (outs Intent_DataValue:$result);
}
def Intent_AtomicStoreOp : Intent_Op<"atomic_store", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_IndexRelation:$index,
                       I64Attr:$value_operand,
                       Intent_AtomicOrderingAttr:$ordering);
  let results = (outs);
}
def Intent_AtomicRMWOp : Intent_Op<"atomic_rmw", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_IndexRelation:$index,
                       I64Attr:$value_operand,
                       Intent_AtomicOrderingAttr:$ordering,
                       Intent_AtomicRMWKindAttr:$kind);
  let results = (outs Intent_DataValue:$result);
}
def Intent_AtomicCompareExchangeOp : Intent_Op<"atomic_compare_exchange",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_IndexRelation:$index,
                       I64Attr:$expected_operand, I64Attr:$desired_operand,
                       Intent_AtomicOrderingAttr:$ordering);
  let results = (outs Intent_RecordValue:$result);
}
def Intent_RandomBitsOp : Intent_Op<"random_bits", [Pure]> {
  let arguments = (ins UI64:$seed, Intent_Value:$logical_counter);
  let results = (outs Intent_DataValue:$result);
}

def Intent_ReturnOp : Intent_Op<"return", [Terminator, ReturnLike]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs);
  let results = (outs);
}

#endif
