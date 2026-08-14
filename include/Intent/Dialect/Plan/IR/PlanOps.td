#ifndef INTENT_DIALECT_PLAN_IR_PLANOPS_TD
#define INTENT_DIALECT_PLAN_IR_PLANOPS_TD

include "Intent/Dialect/Plan/IR/PlanDialect.td"
include "mlir/IR/OpBase.td"

class IntentPlan_Op<string mnemonic, list<Trait> traits = []>
    : Op<IntentPlan_Dialect, mnemonic, traits>;

def IntentPlan_YieldOp : IntentPlan_Op<"yield", [Terminator]> {
  let arguments = (ins);
  let results = (outs);
}

def IntentPlan_RealizationOp : IntentPlan_Op<"realization", [
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
    I64Attr:$lower_offset,
    I64Attr:$upper_offset
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_RegionBindingOp : IntentPlan_Op<"region_binding"> {
  let arguments = (ins
    I64Attr:$argument,
    I64Attr:$axis_node,
    StrAttr:$purpose,
    I64Attr:$level
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_ProgramOp : IntentPlan_Op<"program"> {
  let arguments = (ins
    I64Attr:$loop_node,
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
    StrAttr:$tensor_indexing,
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
    StrAttr:$semantics,
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
    I64Attr:$reuse_operand,
    BoolAttr:$nonnegative_operands,
    DenseI64ArrayAttr:$axis_nodes
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_ContractOp : IntentPlan_Op<"contract"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$lhs_space,
    StrAttr:$rhs_space,
    StrAttr:$accumulator_space
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
    I64Attr:$axis_node,
    StrAttr:$role
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
    OptionalAttr<I64Attr>:$relation_node
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_StageOp : IntentPlan_Op<"stage"> {
  let arguments = (ins
    I64Attr:$node,
    DenseI64ArrayAttr:$inputs,
    DenseI64ArrayAttr:$outputs,
    DenseI64ArrayAttr:$operations,
    DenseI64ArrayAttr:$terminals
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_StageAxisOp : IntentPlan_Op<"stage_axis"> {
  let arguments = (ins
    I64Attr:$stage_node,
    StrAttr:$role,
    OptionalAttr<I64Attr>:$axis_node,
    StrAttr:$extent,
    StrAttr:$tile,
    OptionalAttr<I64Attr>:$worker_axis
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
