#ifndef INTENT_GPU_CONFIGURATIONPARAMETEROPINTERFACE_TD
#define INTENT_GPU_CONFIGURATIONPARAMETEROPINTERFACE_TD

include "mlir/IR/OpBase.td"

def IntentGPU_ConfigurationParameterOpInterface : OpInterface<"ConfigurationParameterOpInterface"> {
  let cppNamespace = "::intent::gpu";
  let description = [{
    An operation-owned finite compile-time configuration domain. Ordinary
    physical extents and provider-local choices retain their distinct result
    types and legality. This interface exposes bindings without treating a
    boolean form choice as a positive physical extent.
  }];
  let methods = [
    InterfaceMethod<"The symbol bound by a candidate row.", "::mlir::StringAttr",
      "getConfigurationName">,
    InterfaceMethod<"The ordered finite domain owned by this declaration.",
      "::llvm::ArrayRef<int64_t>", "getConfigurationCandidates">,
    InterfaceMethod<"The phase at which this declaration must be bound.",
      "::intent::gpu::ConfigurationBindingPhase", "getConfigurationBindingPhase">,
    InterfaceMethod<"The semantic role of this compile-time parameter.",
      "::intent::gpu::ParameterRole", "getConfigurationRole">,
    InterfaceMethod<"The family of this compile-time parameter.",
      "::intent::gpu::ParameterCategory", "getConfigurationCategory">
  ];
}

#endif
