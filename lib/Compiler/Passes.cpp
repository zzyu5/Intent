#include "Intent/Compiler/Passes.h"
#include "Intent/Compiler/Backend.h"
#include "Intent/Conversion/KIRToCPU/KIRToCPU.h"
#include "Intent/Conversion/KIRToDSA/KIRToDSA.h"
#include "Intent/Conversion/KIRToGPU/KIRToGPU.h"
#include "Intent/Dialect/CPU/IR/CPUDialect.h"
#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

namespace intent::compiler {
#define GEN_PASS_DEF_CONSTRUCTGPU
#define GEN_PASS_DEF_CONSTRUCTCPU
#define GEN_PASS_DEF_CONSTRUCTDSA
#define GEN_PASS_DEF_RESOLVEGPUPROFILES
#include "Intent/Compiler/Passes.h.inc"

namespace {

class ConstructGPUPass : public impl::ConstructGPUBase<ConstructGPUPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    GPUCapabilities capabilities{computeUnits, sharedMemoryPerUnit,
        maxDynamicSharedMemoryPerBlock, registersPerUnit, maxThreadsPerBlock,
        computeCapabilityMajor, computeCapabilityMinor,
        singleToDoublePrecisionPerfRatio, matrixUnits, dynamicVectorWidth,
        nativeTupleReductions, nativeFragmentGather};
    if (failed(lowerCanonicalKIRToGPU(getOperation(), capabilities)))
      signalPassFailure();
  }
};

class ConstructCPUPass : public impl::ConstructCPUBase<ConstructCPUPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    if (failed(lowerCanonicalKIRToCPU(getOperation(), stridedInputs
            ? CPUEntryLayout::StridedInputs : CPUEntryLayout::Contiguous)))
      signalPassFailure();
  }
};

class ConstructDSAPass : public impl::ConstructDSABase<ConstructDSAPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    auto module = getOperation();
    auto configuration = dsa::ConfigurationAttr::getChecked(
        [&]() { return module.emitError(); }, module.getContext(), tile.getValue(), tileM.getValue(),
        tileN.getValue(), tileK.getValue(), regionTile.getValue(), tasks.getValue(), localBytes.getValue());
    auto shapeBindings = parseDSAParameterBindings(module, shapes, true);
    auto strideBindings = parseDSAParameterBindings(module, strides, false);
    if (!configuration || failed(shapeBindings) || failed(strideBindings) ||
        failed(lowerCanonicalKIRToDSA(module, configuration, *shapeBindings, *strideBindings)))
      signalPassFailure();
  }
};

class ResolveGPUProfilesPass : public impl::ResolveGPUProfilesBase<ResolveGPUProfilesPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    auto module = getOperation();
    if (directory.empty()) {
      module.emitError("GPU profile resolution requires a resource directory");
      return signalPassFailure();
    }
    auto sources = gpuProfileSources(directory.getValue());
    auto profiles = gpu::TuningProfiles::read(module.getLoc(), sources, overrides);
    if (failed(profiles)) return signalPassFailure();
    profiles->attach(module);
  }
};

} // namespace

#define GEN_PASS_REGISTRATION
#include "Intent/Compiler/Passes.h.inc"
void registerCompilationPasses() { registerIntentCompilerPasses(); }

} // namespace intent::compiler
