#ifndef INTENT_TARGET_CUTILE_IR_CUTILEOPS_TD
#define INTENT_TARGET_CUTILE_IR_CUTILEOPS_TD

include "Intent/Target/CuTile/IR/CuTileDialect.td"
include "mlir/IR/OpBase.td"

class IntentCuTile_Op<string mnemonic, list<Trait> traits = []>
    : Op<IntentCuTile_Dialect, mnemonic, traits>;

def IntentCuTile_TargetOp : IntentCuTile_Op<"target"> {
  let arguments = (ins StrAttr:$architecture, I64Attr:$device);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentCuTile_AxisOp : IntentCuTile_Op<"axis"> {
  let arguments = (ins
    I64Attr:$node,
    I64Attr:$source_axis,
    StrAttr:$role,
    StrAttr:$tile
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentCuTile_ProgramOp : IntentCuTile_Op<"program"> {
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

def IntentCuTile_StorageOp : IntentCuTile_Op<"storage"> {
  let arguments = (ins I64Attr:$value, StrAttr:$space);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentCuTile_LayoutOp : IntentCuTile_Op<"layout"> {
  let arguments = (ins I64Attr:$value, DenseI64ArrayAttr:$order);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentCuTile_ReductionOp : IntentCuTile_Op<"reduction"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$lowering,
    I64Attr:$axis,
    BoolAttr:$keep_dims
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentCuTile_PointwiseOp : IntentCuTile_Op<"pointwise"> {
  let arguments = (ins I64Attr:$node, StrAttr:$lowering);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentCuTile_ContractOp : IntentCuTile_Op<"contract"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$lowering,
    StrAttr:$accumulator_type,
    BoolAttr:$lhs_transpose,
    BoolAttr:$rhs_transpose
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentCuTile_StreamOp : IntentCuTile_Op<"stream"> {
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

def IntentCuTile_BoundaryOp : IntentCuTile_Op<"boundary"> {
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

def IntentCuTile_LaunchOp : IntentCuTile_Op<"launch"> {
  let arguments = (ins
    I64Attr:$loop_node,
    StrAttr:$grid_policy,
    I64Attr:$occupancy
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentCuTile_AutotuneOp : IntentCuTile_Op<"autotune"> {
  let arguments = (ins ArrayAttr:$key);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentCuTile_ConfigOp : IntentCuTile_Op<"config"> {
  let arguments = (ins
    DictionaryAttr:$parameters,
    I64Attr:$num_ctas,
    I64Attr:$occupancy
  );
  let results = (outs);
  let hasVerifier = 1;
}

#endif
