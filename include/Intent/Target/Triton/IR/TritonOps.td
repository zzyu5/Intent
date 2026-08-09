#ifndef INTENT_TARGET_TRITON_IR_TRITONOPS_TD
#define INTENT_TARGET_TRITON_IR_TRITONOPS_TD

include "Intent/Target/Triton/IR/TritonDialect.td"
include "mlir/IR/OpBase.td"

class IntentTriton_Op<string mnemonic, list<Trait> traits = []>
    : Op<IntentTriton_Dialect, mnemonic, traits>;

def IntentTriton_TargetOp : IntentTriton_Op<"target"> {
  let arguments = (ins I64Attr:$device);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_CapabilitiesOp : IntentTriton_Op<"capabilities"> {
  let arguments = (ins
    ArrayAttr:$intent_decided,
    ArrayAttr:$delegated,
    ArrayAttr:$absent
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_AxisOp : IntentTriton_Op<"axis"> {
  let arguments = (ins
    I64Attr:$node,
    I64Attr:$source_axis,
    StrAttr:$role,
    StrAttr:$tile
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_ProgramOp : IntentTriton_Op<"program"> {
  let arguments = (ins
    I64Attr:$loop_node,
    DenseI64ArrayAttr:$worker_axes,
    StrAttr:$ownership,
    ArrayAttr:$traversals
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_StorageOp : IntentTriton_Op<"storage"> {
  let arguments = (ins I64Attr:$value, StrAttr:$space);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_ReductionOp : IntentTriton_Op<"reduction"> {
  let arguments = (ins I64Attr:$node, StrAttr:$lowering, I64Attr:$axis);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_PointwiseOp : IntentTriton_Op<"pointwise"> {
  let arguments = (ins I64Attr:$node, StrAttr:$lowering, BoolAttr:$defer);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_ContractOp : IntentTriton_Op<"contract"> {
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

def IntentTriton_StreamOp : IntentTriton_Op<"stream"> {
  let arguments = (ins
    I64Attr:$node,
    I64Attr:$axis_node,
    OptionalAttr<I64Attr>:$stop_node,
    StrAttr:$tile,
    StrAttr:$order,
    StrAttr:$carry_space
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_RaggedOp : IntentTriton_Op<"ragged"> {
  let arguments = (ins
    I64Attr:$node,
    I64Attr:$outer_node,
    DenseI64ArrayAttr:$member_nodes,
    StrAttr:$traversal
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_StageOp : IntentTriton_Op<"stage"> {
  let arguments = (ins
    I64Attr:$ordinal,
    I64Attr:$node,
    DenseI64ArrayAttr:$inputs,
    DenseI64ArrayAttr:$outputs
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_AtomicOp : IntentTriton_Op<"atomic"> {
  let arguments = (ins I64Attr:$node, StrAttr:$lowering, StrAttr:$scope);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_BoundaryOp : IntentTriton_Op<"boundary"> {
  let arguments = (ins
    I64Attr:$node,
    DenseI64ArrayAttr:$domain_nodes,
    StrAttr:$predicate,
    StrAttr:$load_fill,
    StrAttr:$store_mask,
    BoolAttr:$defer
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_AutotuneOp : IntentTriton_Op<"autotune"> {
  let arguments = (ins ArrayAttr:$key, DictionaryAttr:$parameter_map);
  let results = (outs);
  let hasVerifier = 1;
}

#endif
