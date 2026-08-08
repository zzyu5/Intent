#ifndef INTENT_TARGET_TILELANG_IR_TILELANGOPS_TD
#define INTENT_TARGET_TILELANG_IR_TILELANGOPS_TD

include "Intent/Target/TileLang/IR/TileLangDialect.td"
include "mlir/IR/OpBase.td"

class IntentTileLang_Op<string mnemonic, list<Trait> traits = []>
    : Op<IntentTileLang_Dialect, mnemonic, traits>;

def IntentTileLang_TargetOp : IntentTileLang_Op<"target"> {
  let arguments = (ins StrAttr:$architecture, I64Attr:$device);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_AxisOp : IntentTileLang_Op<"axis"> {
  let arguments = (ins
    I64Attr:$node,
    I64Attr:$source_axis,
    StrAttr:$role,
    StrAttr:$tile
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_ProgramOp : IntentTileLang_Op<"program"> {
  let arguments = (ins
    I64Attr:$loop_node,
    DenseI64ArrayAttr:$worker_axes,
    StrAttr:$traversal,
    StrAttr:$mapping,
    I64Attr:$group_size
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_StorageOp : IntentTileLang_Op<"storage"> {
  let arguments = (ins I64Attr:$value, StrAttr:$space);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_LayoutOp : IntentTileLang_Op<"layout"> {
  let arguments = (ins
    I64Attr:$value,
    StrAttr:$kind,
    DenseI64ArrayAttr:$order
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_ReductionOp : IntentTileLang_Op<"reduction"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$lowering,
    I64Attr:$axis,
    BoolAttr:$keep_dims,
    StrAttr:$input_space,
    StrAttr:$result_space
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_PointwiseOp : IntentTileLang_Op<"pointwise"> {
  let arguments = (ins I64Attr:$node, StrAttr:$lowering, StrAttr:$space);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_ContractOp : IntentTileLang_Op<"contract"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$lowering,
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

def IntentTileLang_StreamOp : IntentTileLang_Op<"stream"> {
  let arguments = (ins
    I64Attr:$node,
    I64Attr:$axis_node,
    StrAttr:$tile,
    StrAttr:$order,
    StrAttr:$carry_space
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_RaggedOp : IntentTileLang_Op<"ragged"> {
  let arguments = (ins
    I64Attr:$node,
    I64Attr:$outer_node,
    I64Attr:$member_node,
    StrAttr:$traversal
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_StageOp : IntentTileLang_Op<"stage"> {
  let arguments = (ins
    I64Attr:$ordinal,
    I64Attr:$node,
    DenseI64ArrayAttr:$inputs,
    DenseI64ArrayAttr:$outputs
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_AtomicOp : IntentTileLang_Op<"atomic"> {
  let arguments = (ins I64Attr:$node, StrAttr:$lowering);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_BoundaryOp : IntentTileLang_Op<"boundary"> {
  let arguments = (ins
    I64Attr:$node,
    DenseI64ArrayAttr:$domain_nodes,
    StrAttr:$access,
    StrAttr:$padding,
    BoolAttr:$check_bounds
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_PipelineOp : IntentTileLang_Op<"pipeline"> {
  let arguments = (ins
    I64Attr:$loop_node,
    I64Attr:$num_stages,
    BoolAttr:$async_copy
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_LaunchOp : IntentTileLang_Op<"launch"> {
  let arguments = (ins
    I64Attr:$loop_node,
    StrAttr:$grid_policy,
    I64Attr:$threads
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_AutotuneOp : IntentTileLang_Op<"autotune"> {
  let arguments = (ins ArrayAttr:$key);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTileLang_ConfigOp : IntentTileLang_Op<"config"> {
  let arguments = (ins
    DictionaryAttr:$parameters,
    I64Attr:$num_stages,
    I64Attr:$threads
  );
  let results = (outs);
  let hasVerifier = 1;
}

#endif
