#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Contractions.h"
#include "Intent/Dialect/CPU/Transforms/Implementation.h"
#include "Intent/Dialect/CPU/IR/CPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_DEF_CPUCONFIGURETARGET
#define GEN_PASS_DEF_CPUNORMALIZESOURCE
#define GEN_PASS_DEF_CPUMATERIALIZECONFIGURATIONS
#define GEN_PASS_DEF_CPUREALIZEREGIONS
#define GEN_PASS_DEF_CPUFORMINPUTSUPPLY
#define GEN_PASS_DEF_CPUFORMTASKS
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

OpPassManager normalizationPipeline() {
  OpPassManager manager(ModuleOp::getOperationName());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());
  return manager;
}

LogicalResult finishGroup(ModuleOp module, StringRef name, LogicalResult result) {
  if (failed(result))
    return module.emitError() << "CPU transformation failed: " << name;
  if (failed(verifyCPUProgram(module, false)))
    return module.emitError() << "CPU postcondition failed: " << name;
  return success();
}

FailureOr<Configuration> currentConfiguration(func::FuncOp function) {
  auto binding = function->getAttrOfType<ConfigurationAttr>("intent_cpu.configuration");
  if (!binding)
    return function.emitError("CPU physical transformation requires a bound configuration"), failure();
  return Configuration{binding.getTaskGrain(), binding.getTileM(), binding.getTileN(),
                       binding.getTileK(), binding.getRegionSize(), {}};
}

class ConfigureTargetPass : public impl::CPUConfigureTargetBase<ConfigureTargetPass> {
public:
  using CPUConfigureTargetBase::CPUConfigureTargetBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto capabilities = CapabilitiesAttr::getChecked([&]() { return module.emitError(); },
        module.getContext(), vectorBits.getValue(), workers.getValue(),
        privateBytes.getValue(), matrixI8I32.getValue());
    if (!capabilities) return signalPassFailure();
    module->setAttr("intent_cpu.capabilities", capabilities);
    if (failed(finishGroup(module, getArgument(), success()))) signalPassFailure();
  }
};

class NormalizeSourcePass : public impl::CPUNormalizeSourceBase<NormalizeSourcePass> {
public:
  using CPUNormalizeSourceBase::CPUNormalizeSourceBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto transform = [&]() -> LogicalResult {
      auto cleanup = normalizationPipeline();
      if (failed(runPipeline(cleanup, module))) return failure();
      auto functions = module.getOps<func::FuncOp>();
      if (!llvm::hasSingleElement(functions) || (*functions.begin()).isExternal())
        return module.emitError("CPU source normalization requires one executable source function");
      auto function = *functions.begin();
      if (failed(normalizeContractionSources(function)) || failed(normalizeContractions(function)) ||
          failed(realizeSliceScans(function)) || failed(foldUniformComputations(function)) ||
          failed(fuseStructuredComputations(function))) return failure();
      return runPipeline(cleanup, module);
    };
    if (failed(finishGroup(module, getArgument(), transform()))) signalPassFailure();
  }
};

class MaterializeConfigurationsPass : public impl::CPUMaterializeConfigurationsBase<MaterializeConfigurationsPass> {
public:
  using CPUMaterializeConfigurationsBase::CPUMaterializeConfigurationsBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto implementations = lookupImplementationProvider(module, provider.getValue());
    if (failed(implementations)) return signalPassFailure();
    // The complete entry verifies its bound clones. No candidate portfolio is
    // retained by the pass or reused after subsequent program transformations.
    if (failed(materializeCPUConfigurations(module, **implementations, defaults.getValue(), overrides.getValue()))) {
      module.emitError() << "CPU transformation failed: " << getArgument();
      signalPassFailure();
    }
  }
};

class RealizeRegionsPass : public impl::CPURealizeRegionsBase<RealizeRegionsPass> {
public:
  using CPURealizeRegionsBase::CPURealizeRegionsBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto transform = [&]() -> LogicalResult {
      auto implementations = lookupImplementationProvider(module, provider.getValue());
      if (failed(implementations)) return failure();
      if (failed((**implementations).verifyBindings(module))) return failure();
      for (auto function : module.getOps<func::FuncOp>()) {
        auto config = currentConfiguration(function);
        if (failed(config) || failed(groupRegionComputations(function, *config)) ||
            failed(realizeRegions(function, *config, **implementations))) return failure();
      }
      auto cleanup = normalizationPipeline();
      if (failed(runPipeline(cleanup, module))) return failure();
      for (auto function : module.getOps<func::FuncOp>())
        if (failed(foldContractionInputs(function, **implementations)) ||
            failed(realizeHistograms(function)) || failed(foldUniformComputations(function)) ||
            failed(fuseStructuredComputations(function))) return failure();
      return runPipeline(cleanup, module);
    };
    if (failed(finishGroup(module, getArgument(), transform()))) signalPassFailure();
  }
};

class FormInputSupplyPass : public impl::CPUFormInputSupplyBase<FormInputSupplyPass> {
public:
  using CPUFormInputSupplyBase::CPUFormInputSupplyBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto transform = [&]() -> LogicalResult {
      auto implementations = lookupImplementationProvider(module, provider.getValue());
      if (failed(implementations)) return failure();
      if (failed((**implementations).verifyBindings(module))) return failure();
      for (auto function : module.getOps<func::FuncOp>()) {
        if (failed(reusePrivateStorage(function)) ||
            failed(reusePreparedInputs(function, **implementations)) ||
            failed(groupQuantizedDots(function, **implementations))) return failure();
        auto config = currentConfiguration(function);
        if (failed(config) || failed(groupWorksetComputations(function, **implementations)) ||
            failed(blockContractions(function, *config, **implementations)) ||
            failed(blockStructuredComputations(function, **implementations))) return failure();
      }
      return success();
    };
    if (failed(finishGroup(module, getArgument(), transform()))) signalPassFailure();
  }
};

class FormTasksPass : public impl::CPUFormTasksBase<FormTasksPass> {
public:
  using CPUFormTasksBase::CPUFormTasksBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto transform = [&]() -> LogicalResult {
      auto implementations = lookupImplementationProvider(module, provider.getValue());
      if (failed(implementations)) return failure();
      if (failed((**implementations).verifyBindings(module))) return failure();
      for (auto function : module.getOps<func::FuncOp>()) {
        auto config = currentConfiguration(function);
        if (failed(config) || failed(partitionTasks(function, config->taskGrain, **implementations)))
          return failure();
      }
      auto cleanup = normalizationPipeline();
      if (failed(runPipeline(cleanup, module))) return failure();
      for (auto function : module.getOps<func::FuncOp>())
        if (failed(isolateTasks(function))) return failure();
      return success();
    };
    if (failed(finishGroup(module, getArgument(), transform()))) signalPassFailure();
  }
};

} // namespace

void registerCPUPasses() {
  registerCPUTransformPasses();
  PassPipelineRegistration<CPUCompilationOptions>("intent-cpu-pipeline",
      "Construct configured CPU tasks with the selected compiled provider", buildCPUPipeline);
}

void buildCPUPipeline(OpPassManager &manager, const CPUCompilationOptions &options) {
  CPUConfigureTargetOptions target;
  target.vectorBits = options.vectorBits;
  target.workers = options.workers;
  target.privateBytes = options.privateBytes;
  target.matrixI8I32 = options.matrixI8I32;
  manager.addPass(createCPUConfigureTarget(target));
  manager.addPass(createCPUNormalizeSource());
  CPUMaterializeConfigurationsOptions configurations;
  configurations.provider = options.provider;
  configurations.defaults = options.defaults;
  configurations.overrides = options.overrides;
  manager.addPass(createCPUMaterializeConfigurations(configurations));
  CPURealizeRegionsOptions regions;
  regions.provider = options.provider;
  manager.addPass(createCPURealizeRegions(regions));
  CPUFormInputSupplyOptions supply;
  supply.provider = options.provider;
  manager.addPass(createCPUFormInputSupply(supply));
  CPUFormTasksOptions tasks;
  tasks.provider = options.provider;
  manager.addPass(createCPUFormTasks(tasks));
}

} // namespace intent::cpu
