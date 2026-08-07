#ifndef INTENT_DIALECT_INTENT_IR_INTENTOPS_TD
#define INTENT_DIALECT_INTENT_IR_INTENTOPS_TD

include "Intent/Dialect/Intent/IR/IntentDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Intent_Op<string mnemonic, list<Trait> traits = []>
    : Op<Intent_Dialect, mnemonic, traits>;

class Intent_NullaryOneResultOp<string mnemonic, list<Trait> traits = [Pure]>
    : Intent_Op<mnemonic, traits> {
  let arguments = (ins);
  let results = (outs AnyType:$result);
}

class Intent_UnaryOneResultOp<string mnemonic, list<Trait> traits = [Pure]>
    : Intent_Op<mnemonic, traits> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}

class Intent_BinaryOneResultOp<string mnemonic, list<Trait> traits = [Pure]>
    : Intent_Op<mnemonic, traits> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

class Intent_VariadicOneResultOp<string mnemonic, list<Trait> traits = [Pure]>
    : Intent_Op<mnemonic, traits> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
}

class Intent_VariadicNoResultOp<string mnemonic, list<Trait> traits = []>
    : Intent_Op<mnemonic, traits> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
}

class Intent_TerminatorOp<string mnemonic>
    : Intent_Op<mnemonic, [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
}

def Intent_ConstantOp : Intent_NullaryOneResultOp<"constant">;
def Intent_DimOp : Intent_UnaryOneResultOp<"dim">;

def Intent_DomainOp : Intent_VariadicOneResultOp<"domain">;
def Intent_DomainProductOp : Intent_VariadicOneResultOp<"domain_product">;
def Intent_PartitionOp : Intent_VariadicOneResultOp<"partition">;
def Intent_IndicesOp : Intent_UnaryOneResultOp<"indices">;

def Intent_ParallelOp : Intent_Op<"parallel", [RecursiveMemoryEffects]> {
  let arguments = (ins AnyType:$source);
  let results = (outs);
  let regions = (region AnyRegion:$body);
}

def Intent_OrderedOp : Intent_Op<"ordered", [RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
}

def Intent_StateStreamOp : Intent_Op<"state_stream", [RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
}

def Intent_IfOp : Intent_Op<"if", [RecursiveMemoryEffects]> {
  let arguments = (ins AnyType:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<AnyRegion>:$branches);
}

def Intent_ForOp : Intent_Op<"for", [RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
}

def Intent_WhileOp : Intent_Op<"while", [RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$before, AnyRegion:$after);
}

def Intent_YieldOp : Intent_TerminatorOp<"yield">;
def Intent_ConditionOp : Intent_TerminatorOp<"condition">;
def Intent_BreakOp : Intent_TerminatorOp<"break">;
def Intent_ContinueOp : Intent_TerminatorOp<"continue">;

def Intent_ViewLoadOp : Intent_VariadicOneResultOp<"view_load", [MemoryEffects<[MemRead]>]>;
def Intent_ViewStoreOp : Intent_VariadicNoResultOp<"view_store", [MemoryEffects<[MemWrite]>]>;
def Intent_ReshapeOp : Intent_UnaryOneResultOp<"reshape">;
def Intent_TransposeOp : Intent_UnaryOneResultOp<"transpose">;
def Intent_BroadcastOp : Intent_UnaryOneResultOp<"broadcast">;
def Intent_FullOp : Intent_UnaryOneResultOp<"full">;
def Intent_ZerosOp : Intent_NullaryOneResultOp<"zeros">;
def Intent_MakeTupleOp : Intent_VariadicOneResultOp<"make_tuple">;
def Intent_MakeRecordOp : Intent_VariadicOneResultOp<"make_record">;
def Intent_ExtractOp : Intent_UnaryOneResultOp<"extract">;

def Intent_UnaryOp : Intent_UnaryOneResultOp<"unary">;
def Intent_BinaryOp : Intent_BinaryOneResultOp<"binary">;
def Intent_CompareOp : Intent_BinaryOneResultOp<"compare">;
def Intent_SelectOp : Intent_Op<"select", [Pure]> {
  let arguments = (ins AnyType:$condition, AnyType:$true_value, AnyType:$false_value);
  let results = (outs AnyType:$result);
}
def Intent_CastOp : Intent_UnaryOneResultOp<"cast">;
def Intent_MaskOp : Intent_Op<"mask", [Pure]> {
  let arguments = (ins AnyType:$value, AnyType:$predicate, AnyType:$fill);
  let results = (outs AnyType:$result);
}

def Intent_ReduceOp : Intent_BinaryOneResultOp<"reduce">;
def Intent_ScanOp : Intent_BinaryOneResultOp<"scan">;
def Intent_ContractOp : Intent_BinaryOneResultOp<"contract">;

def Intent_RaggedOp : Intent_Op<"ragged", [Pure]> {
  let arguments = (ins AnyType:$outer, AnyType:$offsets, AnyType:$indices);
  let results = (outs AnyType:$result);
}
def Intent_RaggedOuterOp : Intent_UnaryOneResultOp<"ragged_outer">;
def Intent_RaggedMemberOp : Intent_BinaryOneResultOp<"ragged_member">;
def Intent_MembersOp : Intent_UnaryOneResultOp<"members">;

def Intent_GatherOp : Intent_VariadicOneResultOp<"gather", [MemoryEffects<[MemRead]>]>;
def Intent_ScatterUniqueOp : Intent_VariadicNoResultOp<"scatter_unique", [MemoryEffects<[MemWrite]>]>;
def Intent_ScatterReduceOp : Intent_VariadicNoResultOp<"scatter_reduce", [MemoryEffects<[MemWrite]>]>;

def Intent_BufferOp : Intent_VariadicOneResultOp<"buffer">;
def Intent_BufferLoadOp : Intent_VariadicOneResultOp<"buffer_load", [MemoryEffects<[MemRead]>]>;
def Intent_BufferStoreOp : Intent_VariadicNoResultOp<"buffer_store", [MemoryEffects<[MemWrite]>]>;
def Intent_AtomicAddOp : Intent_VariadicNoResultOp<"atomic_add", [MemoryEffects<[MemRead, MemWrite]>]>;
def Intent_AtomicCasOp : Intent_VariadicOneResultOp<"atomic_cas", [MemoryEffects<[MemRead, MemWrite]>]>;
def Intent_FenceOp : Intent_Op<"fence", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins);
  let results = (outs);
}
def Intent_RandomOp : Intent_Op<"random", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins AnyType:$seed, AnyType:$logical_index);
  let results = (outs AnyType:$result);
}

def Intent_CallOp : Intent_Op<"call"> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
}
def Intent_ReturnOp : Intent_Op<"return", [Terminator, ReturnLike]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs);
}

#endif
