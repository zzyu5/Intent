#ifndef INTENT_TARGET_TRITON_IR_TRITONOPS_TD
#define INTENT_TARGET_TRITON_IR_TRITONOPS_TD

include "Intent/Dialect/GPU/IR/GPUTypes.td"
include "Intent/Target/Triton/IR/TritonDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IntentTriton_Op<string mnemonic, list<Trait> traits = []> :
    Op<IntentTriton_Dialect, mnemonic, traits>;

def IntentTriton_BlockLoadOp : IntentTriton_Op<"block_load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$view,
                       Variadic<Index>:$offsets,
                       DenseI64ArrayAttr:$block_axes,
                       DenseI64ArrayAttr:$order,
                       DenseI64ArrayAttr:$boundary_axes,
                       StrAttr:$padding);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentTriton_BlockStoreOp : IntentTriton_Op<"block_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$view,
                       Variadic<Index>:$offsets,
                       IntentGPU_FragmentValue:$value,
                       DenseI64ArrayAttr:$block_axes,
                       DenseI64ArrayAttr:$order,
                       DenseI64ArrayAttr:$boundary_axes);
  let hasVerifier = 1;
}

#endif
