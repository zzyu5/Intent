#ifndef INTENT_TARGET_TRITON_IR_TRITONOPS_TD
#define INTENT_TARGET_TRITON_IR_TRITONOPS_TD

include "Intent/Target/Triton/IR/TritonDialect.td"
include "mlir/IR/OpBase.td"

class IntentTriton_Op<string mnemonic, list<Trait> traits = []>
    : Op<IntentTriton_Dialect, mnemonic, traits>;

def IntentTriton_TargetOp : IntentTriton_Op<"target"> {
  let arguments = (ins StrAttr:$architecture, I64Attr:$device, I64Attr:$warp_size);
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
    I64Attr:$worker_axis,
    StrAttr:$traversal,
    StrAttr:$mapping
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_StorageOp : IntentTriton_Op<"storage"> {
  let arguments = (ins I64Attr:$value, StrAttr:$space);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_LayoutOp : IntentTriton_Op<"layout"> {
  let arguments = (ins I64Attr:$value, StrAttr:$kind, DenseI64ArrayAttr:$order);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_ReductionOp : IntentTriton_Op<"reduction"> {
  let arguments = (ins I64Attr:$node, StrAttr:$lowering, I64Attr:$axis);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_PointwiseOp : IntentTriton_Op<"pointwise"> {
  let arguments = (ins I64Attr:$node, StrAttr:$lowering);
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_BoundaryOp : IntentTriton_Op<"boundary"> {
  let arguments = (ins
    I64Attr:$node,
    I64Attr:$domain_node,
    StrAttr:$predicate,
    StrAttr:$load_fill,
    StrAttr:$store_mask
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_PipelineOp : IntentTriton_Op<"pipeline"> {
  let arguments = (ins
    I64Attr:$loop_node,
    I64Attr:$low_stages,
    I64Attr:$high_stages,
    I64Attr:$smem_threshold,
    BoolAttr:$prefetch,
    BoolAttr:$async_copy
  );
  let results = (outs);
  let hasVerifier = 1;
}

def IntentTriton_LaunchOp : IntentTriton_Op<"launch"> {
  let arguments = (ins
    I64Attr:$loop_node,
    StrAttr:$grid_policy,
    I64Attr:$num_warps
  );
  let results = (outs);
  let hasVerifier = 1;
}

#endif
