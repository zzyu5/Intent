#ifndef INTENT_TARGET_TRITON_IR_TRITONOPS_TD
#define INTENT_TARGET_TRITON_IR_TRITONOPS_TD

include "Intent/Dialect/GPU/IR/GPUTypes.td"
include "Intent/Target/Triton/IR/TritonDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class IntentTriton_Op<string mnemonic, list<Trait> traits = []> :
    Op<IntentTriton_Dialect, mnemonic, traits>;

def IntentTriton_CtaBarrierOp : IntentTriton_Op<"cta_barrier", [
    MemoryEffects<[MemRead, MemWrite]>
  ]>;

def IntentTriton_TensorDescriptorChoiceOp :
    IntentTriton_Op<"tensor_descriptor_choice", [Pure]> {
  let arguments = (ins Variadic<IntentGPU_ViewValue>:$descriptors,
                       StrAttr:$construction,
                       StrAttr:$selection,
                       StrAttr:$config_parameter,
                       StrAttr:$eligibility_argument);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def IntentTriton_TensorDescriptorAllocatorOp :
    IntentTriton_Op<"tensor_descriptor_allocator"> {
  let arguments = (ins I64Attr:$size_argument,
                       I64Attr:$alignment_argument,
                       I64Attr:$stream_argument,
                       StrAttr:$lifetime,
                       StrAttr:$implementation);
  let hasVerifier = 1;
}

def IntentTriton_TensorDescriptorOp : IntentTriton_Op<"tensor_descriptor", [
    Pure,
    AttrSizedOperandSegments
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$base,
                       Variadic<Index>:$shape,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$block_shape,
                       DenseI64ArrayAttr:$source_block_axes,
                       DenseI64ArrayAttr:$initial_block_shape,
                       StrAttr:$base_layout,
                       StrAttr:$padding,
                       DenseI64ArrayAttr:$aligned_stride_axes,
                       DenseI64ArrayAttr:$unit_stride_axes,
                       BoolAttr:$require_positive_shape,
                       BoolAttr:$require_positive_strides,
                       BoolAttr:$require_power_of_two_block_shape,
                       I64Attr:$alignment,
                       I64Attr:$minimum_contiguous_bytes,
                       I64Attr:$pipeline_block_alignment,
                       I64Attr:$maximum_shape_extent,
                       I64Attr:$maximum_block_elements);
  let results = (outs IntentGPU_ViewValue:$result);
  let hasVerifier = 1;
}

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

def IntentTriton_DescriptorLoadOp : IntentTriton_Op<"descriptor_load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$descriptor,
                       Variadic<Index>:$offsets,
                       DenseI64ArrayAttr:$boundary_axes);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentTriton_DescriptorStoreOp : IntentTriton_Op<"descriptor_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$descriptor,
                       Variadic<Index>:$offsets,
                       IntentGPU_FragmentValue:$value,
                       DenseI64ArrayAttr:$boundary_axes);
  let hasVerifier = 1;
}

def IntentTriton_SplitOp : IntentTriton_Op<"split", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$source);
  let results = (outs IntentGPU_FragmentValue:$low,
                      IntentGPU_FragmentValue:$high);
  let hasVerifier = 1;
}

class IntentTriton_ElementwiseCollective<string mnemonic> :
    IntentTriton_Op<mnemonic, [Pure, IsolatedFromAbove, AttrSizedOperandSegments,
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$sources,
                       Variadic<IntentGPU_DataValue>:$identities,
                       I64Attr:$axis,
                       BoolAttr:$reverse);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region AnyRegion:$combine);
  let assemblyFormat = [{
    `sources` `(` $sources `:` type($sources) `)`
    `identities` `(` ($identities^ `:` type($identities))? `)`
    `axis` $axis `reverse` $reverse
    `combine` $combine `->` `(` type($outputs) `)` attr-dict
  }];
  let hasVerifier = 1;
}

// Triton's native callbacks are scalar, unlike a shared fragment combiner.
def IntentTriton_ReduceOp : IntentTriton_ElementwiseCollective<"reduce">;
def IntentTriton_ScanOp : IntentTriton_ElementwiseCollective<"scan">;

def IntentTriton_MapElementwiseOp : IntentTriton_Op<"map_elementwise", [
    IsolatedFromAbove, RecursiveMemoryEffects
  ]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$inputs);
  let results = (outs IntentGPU_FragmentValue:$result);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

#endif
