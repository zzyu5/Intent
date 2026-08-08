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

def IntentPlan_PlanOp : IntentPlan_Op<"plan", [
    SingleBlock,
    SingleBlockImplicitTerminator<"::intent::plan::YieldOp">
  ]> {
  let arguments = (ins
    FlatSymbolRefAttr:$entry,
    StrAttr:$backend,
    StrAttr:$architecture,
    I64Attr:$device,
    I64Attr:$warp_size
  );
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def IntentPlan_ExtentOp : IntentPlan_Op<"extent"> {
  let arguments = (ins
    I64Attr:$node,
    I64Attr:$axis,
    StrAttr:$logical,
    StrAttr:$tile
  );
  let results = (outs);
}

def IntentPlan_OwnershipOp : IntentPlan_Op<"ownership"> {
  let arguments = (ins
    I64Attr:$loop_node,
    StrAttr:$worker,
    I64Attr:$worker_axis,
    StrAttr:$traversal,
    StrAttr:$mapping
  );
  let results = (outs);
}

def IntentPlan_StorageOp : IntentPlan_Op<"storage"> {
  let arguments = (ins
    I64Attr:$value,
    StrAttr:$space,
    StrAttr:$access
  );
  let results = (outs);
}

def IntentPlan_LayoutOp : IntentPlan_Op<"layout"> {
  let arguments = (ins
    I64Attr:$value,
    StrAttr:$kind,
    DenseI64ArrayAttr:$order
  );
  let results = (outs);
}

def IntentPlan_PrimitiveOp : IntentPlan_Op<"primitive"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$kind,
    StrAttr:$operator_name,
    I64Attr:$axis,
    StrAttr:$identity
  );
  let results = (outs);
}

def IntentPlan_BoundaryOp : IntentPlan_Op<"boundary"> {
  let arguments = (ins
    I64Attr:$node,
    StrAttr:$logical,
    StrAttr:$tail,
    StrAttr:$predicate,
    StrAttr:$load_fill
  );
  let results = (outs);
}

def IntentPlan_PipelineOp : IntentPlan_Op<"pipeline"> {
  let arguments = (ins
    I64Attr:$low_stages,
    I64Attr:$high_stages,
    I64Attr:$smem_threshold,
    BoolAttr:$prefetch,
    BoolAttr:$async_copy
  );
  let results = (outs);
}

def IntentPlan_LaunchOp : IntentPlan_Op<"launch"> {
  let arguments = (ins
    I64Attr:$loop_node,
    StrAttr:$grid_policy,
    I64Attr:$num_warps
  );
  let results = (outs);
}

#endif
