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
  let arguments = (ins
    I64Attr:$device,
    I64Attr:$compute_units,
    I64Attr:$shared_memory_per_unit,
    I64Attr:$registers_per_unit,
    BoolAttr:$matrix_units,
    BoolAttr:$dynamic_vector_width
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_AxisOp : IntentPlan_Op<"axis"> {
  let arguments = (ins
    I64Attr:$node,
    I64Attr:$source_axis,
    StrAttr:$role,
    StrAttr:$tile
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_ProgramOp : IntentPlan_Op<"program"> {
  let arguments = (ins
    I64Attr:$loop_node,
    DenseI64ArrayAttr:$worker_axes,
    StrAttr:$traversal,
    StrAttr:$mapping
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_StorageOp : IntentPlan_Op<"storage"> {
  let arguments = (ins I64Attr:$value, StrAttr:$space);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_TransferOp : IntentPlan_Op<"transfer"> {
  let arguments = (ins
    I64Attr:$node,
    DenseI64ArrayAttr:$domain_nodes,
    StrAttr:$access,
    StrAttr:$fill,
    StrAttr:$materialization,
    StrAttr:$result_space,
    BoolAttr:$defer
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_ReductionOp : IntentPlan_Op<"reduction"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$role,
    I64Attr:$axis,
    BoolAttr:$keep_dims,
    StrAttr:$input_space,
    StrAttr:$result_space
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_PointwiseOp : IntentPlan_Op<"pointwise"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$role,
    StrAttr:$result_space,
    I64Attr:$reuse_operand,
    StrAttr:$materialization,
    BoolAttr:$defer
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_ContractOp : IntentPlan_Op<"contract"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$primitive,
    StrAttr:$warp_policy,
    StrAttr:$accumulator_type,
    BoolAttr:$lhs_transpose,
    BoolAttr:$rhs_transpose,
    StrAttr:$lhs_space,
    StrAttr:$rhs_space,
    StrAttr:$accumulator_space
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_StreamOp : IntentPlan_Op<"stream"> {
  let arguments = (ins
    I64Attr:$node,
    I64Attr:$axis_node,
    StrAttr:$tile,
    StrAttr:$order,
    StrAttr:$carry_space,
    BoolAttr:$reuse_initial
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_RaggedOp : IntentPlan_Op<"ragged"> {
  let arguments = (ins
    I64Attr:$node,
    I64Attr:$outer_node,
    I64Attr:$member_node,
    StrAttr:$traversal
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_StageOp : IntentPlan_Op<"stage"> {
  let arguments = (ins
    I64Attr:$ordinal,
    I64Attr:$node,
    DenseI64ArrayAttr:$inputs,
    DenseI64ArrayAttr:$outputs
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentPlan_AtomicOp : IntentPlan_Op<"atomic"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$combine,
    StrAttr:$memory_order,
    StrAttr:$memory_scope
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
