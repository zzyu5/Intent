#ifndef INTENT_TARGET_TILELANG_IR_TILELANGOPS_TD
#define INTENT_TARGET_TILELANG_IR_TILELANGOPS_TD

include "Intent/Dialect/GPU/IR/GPUTypes.td"
include "Intent/Dialect/Intent/IR/IntentAttrs.td"
include "Intent/Target/TileLang/IR/TileLangDialect.td"
include "Intent/Target/TileLang/IR/TileLangTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IntentTileLang_Op<string mnemonic, list<Trait> traits = []> :
    Op<IntentTileLang_Dialect, mnemonic, traits>;

def IntentTileLang_LaunchConfigOp : IntentTileLang_Op<"launch_config", [Pure]> {
  let arguments = (ins Index:$threads);
  let hasVerifier = 1;
}

def IntentTileLang_PipelineOp : IntentTileLang_Op<"pipeline"> {
  let arguments = (ins Index:$stages);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def IntentTileLang_AllocOp : IntentTileLang_Op<"alloc", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let results = (outs IntentTileLang_BufferValue:$result);
  let hasVerifier = 1;
}

def IntentTileLang_ClearOp : IntentTileLang_Op<"clear", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentTileLang_BufferValue:$buffer);
  let hasVerifier = 1;
}

def IntentTileLang_FillOp : IntentTileLang_Op<"fill", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentTileLang_BufferValue:$buffer,
                       IntentGPU_ScalarValue:$value);
  let hasVerifier = 1;
}

def IntentTileLang_SyncOp : IntentTileLang_Op<"sync"> {
  let hasVerifier = 1;
}

def IntentTileLang_CopyInOp : IntentTileLang_Op<"copy_in", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$source,
                       Variadic<Index>:$offsets,
                       IntentTileLang_BufferValue:$destination,
                       DenseI64ArrayAttr:$source_axes);
  let hasVerifier = 1;
}

def IntentTileLang_CopyOutOp : IntentTileLang_Op<"copy_out", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentTileLang_BufferValue:$source,
                       IntentGPU_ViewValue:$destination,
                       Variadic<Index>:$offsets,
                       DenseI64ArrayAttr:$destination_axes);
  let hasVerifier = 1;
}

def IntentTileLang_CastCopyOutOp : IntentTileLang_Op<"cast_copy_out", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentTileLang_BufferValue:$source,
                       IntentGPU_ViewValue:$destination,
                       Variadic<Index>:$offsets,
                       DenseI64ArrayAttr:$destination_axes);
  let hasVerifier = 1;
}

def IntentTileLang_ParallelOp : IntentTileLang_Op<"parallel"> {
  let arguments = (ins ArrayAttr:$shape);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def IntentTileLang_YieldOp : IntentTileLang_Op<"yield", [Terminator]> {
  let hasVerifier = 1;
}

def IntentTileLang_BufferLoadOp : IntentTileLang_Op<"buffer_load", [Pure]> {
  let arguments = (ins IntentTileLang_BufferValue:$buffer,
                       Variadic<Index>:$indices);
  let results = (outs IntentGPU_ScalarValue:$result);
  let hasVerifier = 1;
}

def IntentTileLang_BufferStoreOp : IntentTileLang_Op<"buffer_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentTileLang_BufferValue:$buffer,
                       Variadic<Index>:$indices,
                       IntentGPU_ScalarValue:$value);
  let hasVerifier = 1;
}

def IntentTileLang_ViewLoadOp : IntentTileLang_Op<"view_load", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$view,
                       Variadic<Index>:$indices,
                       Optional<IntentGPU_ScalarValue>:$valid,
                       Optional<IntentGPU_ScalarValue>:$fill);
  let results = (outs IntentGPU_ScalarValue:$result);
  let hasVerifier = 1;
}

def IntentTileLang_ViewStoreOp : IntentTileLang_Op<"view_store", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$view,
                       Variadic<Index>:$indices,
                       IntentGPU_ScalarValue:$value,
                       Optional<IntentGPU_ScalarValue>:$valid);
  let hasVerifier = 1;
}

def IntentTileLang_ReduceOp : IntentTileLang_Op<"reduce", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentTileLang_BufferValue:$source,
                       IntentTileLang_BufferValue:$destination,
                       I64Attr:$axis,
                       Intent_BinaryOperatorAttr:$kind);
  let hasVerifier = 1;
}

def IntentTileLang_ScanOp : IntentTileLang_Op<"scan", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentTileLang_BufferValue:$source,
                       IntentTileLang_BufferValue:$destination,
                       I64Attr:$axis,
                       Intent_BinaryOperatorAttr:$kind,
                       BoolAttr:$reverse);
  let hasVerifier = 1;
}

def IntentTileLang_GemmOp : IntentTileLang_Op<"gemm", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentTileLang_BufferValue:$lhs,
                       IntentTileLang_BufferValue:$rhs,
                       IntentTileLang_BufferValue:$accumulator,
                       BoolAttr:$transpose_lhs,
                       BoolAttr:$transpose_rhs);
  let hasVerifier = 1;
}

#endif
