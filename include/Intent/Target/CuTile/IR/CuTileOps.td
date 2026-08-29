#ifndef INTENT_TARGET_CUTILE_IR_CUTILEOPS_TD
#define INTENT_TARGET_CUTILE_IR_CUTILEOPS_TD

include "Intent/Dialect/GPU/IR/GPUTypes.td"
include "Intent/Dialect/Intent/IR/IntentAttrs.td"
include "Intent/Target/CuTile/IR/CuTileDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IntentCuTile_Op<string mnemonic, list<Trait> traits = []> :
    Op<IntentCuTile_Dialect, mnemonic, traits>;

def IntentCuTile_TileLoadOp : IntentCuTile_Op<"tile_load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       Variadic<Index>:$tile_indices);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_TileStoreOp : IntentCuTile_Op<"tile_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       Variadic<Index>:$tile_indices,
                       IntentGPU_FragmentValue:$value);
  let hasVerifier = 1;
}

def IntentCuTile_ScalarLoadOp : IntentCuTile_Op<"scalar_load", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       Variadic<Index>:$indices,
                       Optional<IntentGPU_ScalarValue>:$valid,
                       Optional<IntentGPU_ScalarValue>:$fill);
  let results = (outs IntentGPU_ScalarValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_ScalarStoreOp : IntentCuTile_Op<"scalar_store", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       Variadic<Index>:$indices,
                       IntentGPU_ScalarValue:$value,
                       Optional<IntentGPU_ScalarValue>:$valid);
  let hasVerifier = 1;
}

def IntentCuTile_GatherLoadOp : IntentCuTile_Op<"gather_load", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       Optional<IntentGPU_DataValue>:$valid,
                       Optional<IntentGPU_ScalarValue>:$fill,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_ScatterStoreOp : IntentCuTile_Op<"scatter_store", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_FragmentValue:$value,
                       Optional<IntentGPU_DataValue>:$valid,
                       DenseI64ArrayAttr:$source_axes);
  let hasVerifier = 1;
}

def IntentCuTile_AtomicRMWOp : IntentCuTile_Op<"atomic_rmw", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_DataValue:$value,
                       Intent_AtomicRMWKindAttr:$kind,
                       Intent_AtomicOrderingAttr:$ordering,
                       IntentGPU_AtomicSharingDomainAttr:$sharing);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_ExtractScalarOp : IntentCuTile_Op<"extract_scalar", [
    Pure, AttrSizedOperandSegments
  ]> {
  let arguments = (ins IntentGPU_FragmentValue:$source,
                       Variadic<IntentGPU_ScalarValue>:$coordinates,
                       Optional<IntentGPU_ScalarValue>:$valid,
                       Optional<IntentGPU_ScalarValue>:$fill);
  let results = (outs IntentGPU_ScalarValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_MMAOp : IntentCuTile_Op<"mma", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$lhs,
                       IntentGPU_FragmentValue:$rhs,
                       IntentGPU_FragmentValue:$accumulator);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_ScaledMMAOp : IntentCuTile_Op<"scaled_mma", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$lhs,
                       IntentGPU_FragmentValue:$lhs_scale,
                       IntentGPU_FragmentValue:$rhs,
                       IntentGPU_FragmentValue:$rhs_scale,
                       IntentGPU_FragmentValue:$accumulator,
                       Intent_ScaledFormatAttr:$lhs_format,
                       Intent_ScaledFormatAttr:$rhs_format,
                       I64Attr:$lhs_group_size,
                       I64Attr:$rhs_group_size);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_ReduceOp : IntentCuTile_Op<"reduce", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$source,
                       I64Attr:$axis,
                       Intent_BinaryOperatorAttr:$kind);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_ScanOp : IntentCuTile_Op<"scan", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$source,
                       I64Attr:$axis,
                       Intent_BinaryOperatorAttr:$kind,
                       BoolAttr:$reverse);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

#endif
