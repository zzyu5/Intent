#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Transforms/Implementation.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace mlir;

namespace intent::cpu {
namespace {

FailureOr<llvm::json::Object> readProfiles(Location loc, llvm::StringRef path) {
  auto content = llvm::MemoryBuffer::getFile(path);
  if (!content) {
    emitError(loc, "cannot read CPU tuning profiles: ") << path;
    return failure();
  }
  auto value = llvm::json::parse((*content)->getBuffer());
  if (!value) {
    emitError(loc, "invalid CPU tuning JSON: ") << llvm::toString(value.takeError());
    return failure();
  }
  auto root = value->getAsObject();
  if (!root || root->size() != 1 || !root->getObject("cpu")) {
    emitError(loc, "CPU tuning profiles require only the cpu namespace");
    return failure();
  }
  auto profiles = root->getObject("cpu");
  for (auto &item : *profiles) {
    auto rows = item.second.getAsArray();
    if (!rows || rows->empty()) {
      emitError(loc, "CPU tuning family must contain a non-empty row array");
      return failure();
    }
    for (const llvm::json::Value &entry : *rows) {
      auto candidate = entry.getAsObject();
      if (!candidate || candidate->size() != 2 || !candidate->getObject("local")) {
        emitError(loc, "CPU candidate requires shared and local parameter bindings");
        return failure();
      }
      auto row = candidate->getArray("shared");
      if (!row || row->size() != 4) {
        emitError(loc, "CPU shared binding requires task grain and M/N/K outer blocks");
        return failure();
      }
      for (const llvm::json::Value &column : *row) {
        auto value = column.getAsInteger();
        if (!value || *value <= 0) {
          emitError(loc, "CPU tuning columns must be positive integers");
          return failure();
        }
      }
      for (auto &parameter : *candidate->getObject("local"))
        if (!parameter.second.getAsInteger() || *parameter.second.getAsInteger() <= 0)
          return emitError(loc, "implementation parameter values must be positive integers"), failure();
    }
  }
  return std::move(*profiles);
}

LogicalResult normalize(ModuleOp module) {
  PassManager manager(module.getContext());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());
  return manager.run(module);
}

}

LogicalResult runCPUPasses(ModuleOp module, int64_t vectorBits, int64_t workers,
                          llvm::StringRef defaults, llvm::StringRef overrides,
                          const ImplementationRegistry &implementations) {
  auto capabilities = CapabilitiesAttr::getChecked([&]() { return module.emitError(); },
      module.getContext(), vectorBits, workers, int64_t{262144});
  if (!capabilities) return failure();
  module->setAttr("intent_cpu.capabilities", capabilities);
  if (failed(verifyCPUProgram(module, false)) || failed(normalize(module))) return failure();
  auto profiles = readProfiles(module.getLoc(), defaults);
  if (failed(profiles)) return failure();
  if (!overrides.empty()) {
    auto replacement = readProfiles(module.getLoc(), overrides);
    if (failed(replacement)) return failure();
    for (auto &item : *replacement) (*profiles)[item.first] = std::move(item.second);
  }
  auto original = *module.getOps<func::FuncOp>().begin();
  llvm::StringRef family = implementations.profile(original);
  auto rows = profiles->getArray(family);
  if (!rows || rows->empty()) return module.emitError("CPU candidate family is empty or missing");
  SmallVector<Configuration> configurations;
  bool hasContraction = false;
  original.walk([&](linalg::GenericOp operation) { hasContraction |= isMatrixContraction(operation); });
  for (const llvm::json::Value &value : *rows) {
    auto candidate = value.getAsObject();
    auto row = candidate->getArray("shared");
    Builder builder(module.getContext());
    SmallVector<NamedAttribute> local;
    for (auto &parameter : *candidate->getObject("local"))
      local.push_back(builder.getNamedAttr(parameter.first, builder.getI64IntegerAttr(*parameter.second.getAsInteger())));
    Configuration config{*(*row)[0].getAsInteger(), *(*row)[1].getAsInteger(),
        *(*row)[2].getAsInteger(), *(*row)[3].getAsInteger(), builder.getDictionaryAttr(local)};
    if (!hasContraction && (config.tileM != 1 || config.tileN != 1 || config.tileK != 1))
      return original.emitError("M/N/K block parameters require a matrix contraction consumer; otherwise they must be 1");
    if (implementations.legal(*module.getOps<func::FuncOp>().begin(), capabilities, config))
      configurations.push_back(config);
  }
  if (configurations.empty()) return module.emitError("no legal CPU candidates remain");
  if (failed(fuseStructuredComputations(original)) || failed(normalize(module)) ||
      failed(verifyCPUProgram(module, false))) return failure();
  SmallVector<func::FuncOp> functions;
  for (auto [number, config] : llvm::enumerate(configurations)) {
    auto function = cast<func::FuncOp>(original->clone());
    function.setName(original.getName().str() + "_config_" + std::to_string(number));
    module.push_back(function);
    Builder b(module.getContext());
    function->setAttr("intent_cpu.configuration", ConfigurationAttr::get(module.getContext(),
        config.taskGrain, config.tileM, config.tileN, config.tileK));
    if (failed(implementations.bind(function, capabilities, config))) return failure();
    SmallVector<Attribute> bindings;
    function.walk([&](Operation *operation) {
      if (auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation"))
        if (!llvm::is_contained(bindings, binding)) bindings.push_back(binding);
    });
    function->setAttr("intent_cpu.implementations", b.getArrayAttr(bindings));
    if (llvm::any_of(functions, [&](func::FuncOp previous) {
          return previous->getAttr("intent_cpu.configuration") == function->getAttr("intent_cpu.configuration") &&
              previous->getAttr("intent_cpu.implementations") == function->getAttr("intent_cpu.implementations");
        })) {
      function.erase();
      continue;
    }
    functions.push_back(function);
  }
  original.erase();
  for (auto function : functions) {
    auto binding = function->getAttrOfType<ConfigurationAttr>("intent_cpu.configuration");
    Configuration config{binding.getTaskGrain(), binding.getTileM(), binding.getTileN(),
        binding.getTileK(), {}};
    if (failed(blockContractions(function, config, implementations)) || failed(verifyCPUProgram(module, false)) ||
        failed(partitionTasks(function, config.taskGrain))) return failure();
  }
  if (failed(normalize(module))) return failure();
  for (func::FuncOp function : functions)
    if (failed(isolateTasks(function))) return failure();
  return verifyCPUProgram(module, false);
}

}
