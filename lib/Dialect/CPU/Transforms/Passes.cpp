#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Transforms/Implementation.h"
#include "Intent/Dialect/CPU/IR/CPUDialect.h"
#include "Intent/Transforms/PassManager.h"
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
namespace {

void programDialects(DialectRegistry &registry) {
  registry.insert<IntentCPUDialect, arith::ArithDialect, func::FuncDialect,
                  linalg::LinalgDialect, math::MathDialect, memref::MemRefDialect,
                  scf::SCFDialect, vector::VectorDialect>();
}

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

class NormalizeSourcePass
    : public PassWrapper<NormalizeSourcePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeSourcePass)
  void getDependentDialects(DialectRegistry &registry) const final { programDialects(registry); }
  StringRef getArgument() const final { return "intent-cpu-normalize-source"; }
  StringRef getDescription() const final {
    return "Normalize source computations before CPU implementation selection";
  }
  void runOnOperation() final {
    auto module = getOperation();
    auto transform = [&]() -> LogicalResult {
      auto cleanup = normalizationPipeline();
      if (failed(runPipeline(cleanup, module))) return failure();
      auto functions = module.getOps<func::FuncOp>();
      if (!llvm::hasSingleElement(functions) || (*functions.begin()).isExternal())
        return module.emitError("CPU source normalization requires one executable source function");
      auto function = *functions.begin();
      if (failed(realizeSliceScans(function)) || failed(foldUniformComputations(function)) ||
          failed(fuseStructuredComputations(function))) return failure();
      return runPipeline(cleanup, module);
    };
    if (failed(finishGroup(module, getArgument(), transform()))) signalPassFailure();
  }
};

class MaterializeConfigurationsPass
    : public PassWrapper<MaterializeConfigurationsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeConfigurationsPass)
  MaterializeConfigurationsPass() = default;
  MaterializeConfigurationsPass(const ImplementationRegistry &implementations,
                                StringRef defaults, StringRef overrides)
      : implementations(&implementations), defaults(defaults.str()), overrides(overrides.str()) {}
  void getDependentDialects(DialectRegistry &registry) const final { programDialects(registry); }
  StringRef getArgument() const final { return "intent-cpu-materialize-configurations"; }
  StringRef getDescription() const final {
    return "Bind complete CPU implementation candidates into the current program";
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (!implementations) {
      module.emitError("CPU configuration materialization requires a provider implementation registry");
      return signalPassFailure();
    }
    // The complete entry verifies its bound clones. No candidate portfolio is
    // retained by the pass or reused after subsequent program transformations.
    if (failed(materializeCPUConfigurations(module, *implementations, defaults, overrides))) {
      module.emitError() << "CPU transformation failed: " << getArgument();
      signalPassFailure();
    }
  }
private:
  const ImplementationRegistry *implementations = nullptr;
  std::string defaults, overrides;
};

class RealizeRegionsPass
    : public PassWrapper<RealizeRegionsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RealizeRegionsPass)
  RealizeRegionsPass() = default;
  explicit RealizeRegionsPass(const ImplementationRegistry &implementations)
      : implementations(&implementations) {}
  void getDependentDialects(DialectRegistry &registry) const final { programDialects(registry); }
  StringRef getArgument() const final { return "intent-cpu-realize-regions"; }
  StringRef getDescription() const final {
    return "Realize configured CPU regions and close newly exposed structured computations";
  }
  void runOnOperation() final {
    auto module = getOperation();
    auto transform = [&]() -> LogicalResult {
      if (!implementations)
        return module.emitError("CPU region realization requires a provider implementation registry");
      for (auto function : module.getOps<func::FuncOp>()) {
        auto config = currentConfiguration(function);
        if (failed(config) || failed(groupRegionComputations(function, *config)) ||
            failed(realizeRegions(function, *config, *implementations))) return failure();
      }
      auto cleanup = normalizationPipeline();
      if (failed(runPipeline(cleanup, module))) return failure();
      for (auto function : module.getOps<func::FuncOp>())
        if (failed(realizeHistograms(function)) || failed(foldUniformComputations(function)) ||
            failed(fuseStructuredComputations(function))) return failure();
      return runPipeline(cleanup, module);
    };
    if (failed(finishGroup(module, getArgument(), transform()))) signalPassFailure();
  }
private:
  const ImplementationRegistry *implementations = nullptr;
};

class FormInputSupplyPass
    : public PassWrapper<FormInputSupplyPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormInputSupplyPass)
  FormInputSupplyPass() = default;
  explicit FormInputSupplyPass(const ImplementationRegistry &implementations)
      : implementations(&implementations) {}
  void getDependentDialects(DialectRegistry &registry) const final { programDialects(registry); }
  StringRef getArgument() const final { return "intent-cpu-form-input-supply"; }
  StringRef getDescription() const final {
    return "Form shared input representations, reuse storage and block configured CPU worksets";
  }
  void runOnOperation() final {
    auto module = getOperation();
    auto transform = [&]() -> LogicalResult {
      if (!implementations)
        return module.emitError("CPU input supply requires a provider implementation registry");
      for (auto function : module.getOps<func::FuncOp>()) {
        if (failed(reusePrivateStorage(function)) ||
            failed(reusePreparedInputs(function, *implementations)) ||
            failed(groupQuantizedDots(function, *implementations))) return failure();
        auto config = currentConfiguration(function);
        if (failed(config) || failed(groupWorksetComputations(function, *implementations)) ||
            failed(blockContractions(function, *config, *implementations)) ||
            failed(blockStructuredComputations(function, *implementations))) return failure();
      }
      return success();
    };
    if (failed(finishGroup(module, getArgument(), transform()))) signalPassFailure();
  }
private:
  const ImplementationRegistry *implementations = nullptr;
};

class FormTasksPass
    : public PassWrapper<FormTasksPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormTasksPass)
  FormTasksPass() = default;
  explicit FormTasksPass(const ImplementationRegistry &implementations)
      : implementations(&implementations) {}
  void getDependentDialects(DialectRegistry &registry) const final { programDialects(registry); }
  StringRef getArgument() const final { return "intent-cpu-form-tasks"; }
  StringRef getDescription() const final {
    return "Partition configured CPU work and isolate complete task captures";
  }
  void runOnOperation() final {
    auto module = getOperation();
    auto transform = [&]() -> LogicalResult {
      if (!implementations)
        return module.emitError("CPU task formation requires a provider implementation registry");
      for (auto function : module.getOps<func::FuncOp>()) {
        auto config = currentConfiguration(function);
        if (failed(config) || failed(partitionTasks(function, config->taskGrain, *implementations)))
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
private:
  const ImplementationRegistry *implementations = nullptr;
};

} // namespace

void registerCPUPasses() {
  PassRegistration<NormalizeSourcePass>();
  PassRegistration<MaterializeConfigurationsPass>();
  PassRegistration<RealizeRegionsPass>();
  PassRegistration<FormInputSupplyPass>();
  PassRegistration<FormTasksPass>();
}

LogicalResult runCPUPasses(ModuleOp module, int64_t vectorBits, int64_t workers,
                          bool matrixI8I32, StringRef defaults, StringRef overrides,
                          const ImplementationRegistry &implementations) {
  auto capabilities = CapabilitiesAttr::getChecked([&]() { return module.emitError(); },
      module.getContext(), vectorBits, workers, int64_t{262144}, matrixI8I32);
  if (!capabilities) return failure();
  module->setAttr("intent_cpu.capabilities", capabilities);
  if (failed(verifyCPUProgram(module, false))) return failure();
  // The registry is immutable and alive for this synchronous manager run.
  // Every execution decision that survives a group is carried by current IR.
  PassManager manager(module.getContext(), ModuleOp::getOperationName());
  manager.addPass(std::make_unique<NormalizeSourcePass>());
  manager.addPass(std::make_unique<MaterializeConfigurationsPass>(implementations, defaults, overrides));
  manager.addPass(std::make_unique<RealizeRegionsPass>(implementations));
  manager.addPass(std::make_unique<FormInputSupplyPass>(implementations));
  manager.addPass(std::make_unique<FormTasksPass>(implementations));
  if (failed(intent::configurePassManager(manager))) return failure();
  return manager.run(module);
}

} // namespace intent::cpu
