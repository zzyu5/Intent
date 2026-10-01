#include "Intent/Target/Triton/IR/Configuration.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;

namespace intent::triton {

bool isLegalDeviceOption(gpu::ParameterRole role, int64_t value,
                         gpu::CapabilitiesAttr capabilities,
                         bool requiresSingleCTA) {
  if (!capabilities || value <= 0) return false;
  switch (role) {
  case gpu::ParameterRole::ProviderWarps:
    return llvm::isPowerOf2_64(value) &&
           value <= capabilities.getMaxThreadsPerBlock() / 32;
  case gpu::ParameterRole::ProviderStages:
    return value <= std::numeric_limits<int32_t>::max();
  case gpu::ParameterRole::ProviderCTAs:
    return llvm::isPowerOf2_64(value) && value <= 16 &&
           (!requiresSingleCTA || value == 1) &&
           (value == 1 || (capabilities.getComputeCapabilityMajor() >= 9 &&
                          capabilities.getComputeCapabilityMajor() != 12));
  default: return false;
  }
}

FailureOr<ConfigurationSchema> ConfigurationSchema::read(func::FuncOp kernel) {
  ConfigurationSchema result;
  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities)
    return kernel.emitError("Triton configuration requires GPU capabilities"), failure();
  bool requiresSingleCTA = false;
  for (BlockArgument argument : kernel.getArguments()) {
    auto kind = kernel.getArgAttrOfType<StringAttr>(argument.getArgNumber(), gpu::abiKindAttr);
    requiresSingleCTA |= isa<gpu::BufferType>(argument.getType()) ||
                         (kind && kind.getValue() == "workspace");
  }
  kernel.walk([&](CtaBarrierOp) { requiresSingleCTA = true; });
  gpu::ParameterAttr stageDeclaration;
  SmallVector<gpu::ParameterOp> parameters;
  bool valid = true;
  kernel.walk([&](gpu::ParameterOp parameter) {
    auto definition = parameter.getParameter();
    auto role = static_cast<gpu::ParameterRole>(definition.getRole());
    StringAttr *option = nullptr;
    switch (role) {
    case gpu::ParameterRole::ProviderWarps: option = &result.warps; break;
    case gpu::ParameterRole::ProviderStages:
      option = &result.stages;
      stageDeclaration = definition;
      break;
    case gpu::ParameterRole::ProviderCTAs: option = &result.ctas; break;
    default: break;
    }
    bool provider = definition.getCategory() ==
                    static_cast<uint32_t>(gpu::ParameterCategory::Provider);
    if (provider != bool(option)) {
      parameter.emitOpError("Triton program contains a foreign provider parameter");
      valid = false;
      return;
    }
    if (option) {
      if (*option) {
        parameter.emitOpError("duplicates a Triton provider-parameter role");
        valid = false;
        return;
      }
      *option = definition.getName();
    }
    if (option)
      for (int64_t candidate : definition.getCandidates().asArrayRef())
        if (!isLegalDeviceOption(role, candidate, capabilities, requiresSingleCTA)) {
          parameter.emitOpError("declares a Triton option outside the current device/launch contract")
              << "; parameter=" << definition.getName() << "; value=" << candidate;
          valid = false;
          return;
        }
    parameters.push_back(parameter);
  });
  if (!valid) return failure();
  if (!result.warps || !result.stages || !result.ctas)
    return kernel.emitError("Triton provider parameter domains are incomplete"), failure();

  bool stagesInKernel = false;
  auto walked = kernel.walk([&](Operation *operation) {
    if (!operation->hasAttr(loopStagesAttr)) return WalkResult::advance();
    auto binding = operation->getAttrOfType<gpu::ParameterAttr>(loopStagesAttr);
    if (!isa<scf::ForOp>(operation) || !binding || binding != stageDeclaration) {
      operation->emitOpError("loop stages must bind the declared Triton stage parameter");
      return WalkResult::interrupt();
    }
    stagesInKernel = true;
    return WalkResult::advance();
  });
  if (walked.wasInterrupted()) return failure();

  kernel.walk([&](TensorDescriptorChoiceOp choice) {
    result.kernelParameters.push_back(choice.getConfigParameterAttr());
  });
  for (gpu::ParameterOp parameter : parameters) {
    StringAttr name = parameter.getParameter().getName();
    if (name == result.warps || name == result.ctas ||
        (name == result.stages && !stagesInKernel))
      continue;
    result.kernelParameters.push_back(name);
  }
  return result;
}

} // namespace intent::triton
