#include "Intent/Dialect/CPU/Transforms/Passes.h"
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
    if (item.first != "vector" && item.first != "contraction") {
      emitError(loc, "unknown CPU tuning family: ") << llvm::StringRef(item.first);
      return failure();
    }
    auto rows = item.second.getAsArray();
    if (!rows || rows->empty()) {
      emitError(loc, "CPU tuning family must contain a non-empty row array");
      return failure();
    }
    SmallVector<SmallVector<int64_t>> seen;
    for (const llvm::json::Value &entry : *rows) {
      auto row = entry.getAsArray();
      if (!row || row->size() != (item.first == "vector" ? 4u : 7u)) {
        emitError(loc, "CPU tuning row has the wrong column count");
        return failure();
      }
      SmallVector<int64_t> values;
      for (const llvm::json::Value &column : *row) {
        auto value = column.getAsInteger();
        if (!value || *value <= 0) {
          emitError(loc, "CPU tuning columns must be positive integers");
          return failure();
        }
        values.push_back(*value);
      }
      if (llvm::is_contained(seen, values)) {
        emitError(loc, "CPU tuning family contains a duplicate row");
        return failure();
      }
      seen.push_back(values);
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
                          llvm::StringRef defaults, llvm::StringRef overrides) {
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
  bool contraction = false;
  module.walk([&](linalg::GenericOp operation) { contraction |= isMatrixContraction(operation); });
  llvm::StringRef family = contraction ? "contraction" : "vector";
  auto rows = profiles->getArray(family);
  if (!rows || rows->empty()) return module.emitError("CPU candidate family is empty or missing");
  SmallVector<Configuration> configurations;
  SmallVector<SmallVector<int64_t>> seen;
  for (const llvm::json::Value &value : *rows) {
    auto row = value.getAsArray();
    if (!row || row->size() != (contraction ? 7u : 4u))
      return module.emitError("CPU candidate has an invalid column count");
    SmallVector<int64_t> columns;
    for (const llvm::json::Value &item : *row) {
      auto number = item.getAsInteger();
      if (!number || *number <= 0) return module.emitError("CPU candidate values must be positive integers");
      columns.push_back(*number);
    }
    if (llvm::is_contained(seen, columns)) return module.emitError("duplicate CPU candidate");
    seen.push_back(columns);
    if ((columns[0] & (columns[0] - 1)) != 0)
      return module.emitError("CPU vector width must be a power of two");
    if (columns[0] > vectorBits / 32) continue;
    Configuration config{columns[0], columns[1], 1, 1, 1, 1, 1, 1, 1};
    if (contraction) {
      config.tileM = columns[2]; config.tileN = columns[3];
      config.tileK = columns[4]; config.microM = columns[5];
      config.microN = columns[6];
      if (config.tileN % config.vectorWidth ||
          config.microM > 8 || config.microN > 4 ||
          config.microM * config.microN > 24 ||
          config.tileK > capabilities.getPrivateBytes() / 4 / config.vectorWidth / config.microN)
        return module.emitError("CPU tile candidate violates vector, microtile or stack-size constraints");
    } else {
      config.registerReplicas = columns[2];
      config.reductionReplicas = columns[3];
      for (int64_t replicas : {config.registerReplicas, config.reductionReplicas})
        if (replicas > 16 || (replicas & (replicas - 1)))
          return module.emitError("CPU vector candidate requires power-of-two register replicas up to sixteen");
    }
    configurations.push_back(config);
  }
  if (configurations.empty()) return module.emitError("no legal CPU candidates remain");
  auto original = *module.getOps<func::FuncOp>().begin();
  if (failed(fuseStructuredComputations(original)) || failed(normalize(module)) ||
      failed(verifyCPUProgram(module, false))) return failure();
  SmallVector<func::FuncOp> functions;
  for (auto [number, config] : llvm::enumerate(configurations)) {
    auto function = cast<func::FuncOp>(original->clone());
    function.setName(original.getName().str() + "_config_" + std::to_string(number));
    module.push_back(function);
    Builder b(module.getContext());
    function->setAttr("intent_cpu.configuration", ConfigurationAttr::get(module.getContext(),
        config.vectorWidth, config.taskGrain, config.tileM, config.tileN,
        config.tileK, config.microM, config.microN, config.registerReplicas,
        config.reductionReplicas));
    functions.push_back(function);
  }
  original.erase();
  for (auto [function, config] : llvm::zip(functions, configurations)) {
    if (failed(blockContractions(function, config)) || failed(verifyCPUProgram(module, false)) ||
        failed(partitionTasks(function, config.taskGrain))) return failure();
  }
  if (failed(normalize(module))) return failure();
  for (func::FuncOp function : functions)
    if (failed(isolateTasks(function))) return failure();
  return verifyCPUProgram(module, false);
}

LogicalResult materializeCPUProgram(ModuleOp module) {
  if (failed(verifyCPUProgram(module, false))) return failure();
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto configuration = function->getAttrOfType<ConfigurationAttr>("intent_cpu.configuration");
    if (!configuration) return function.emitError("CPU materialization requires a complete binding");
    if (failed(materializeRegisterContractions(function)) ||
        failed(materializeStructuredComputations(function)) ||
        failed(fuseIntermediateBuffers(function))) return failure();
  }
  if (failed(normalize(module))) return failure();
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto configuration = function->getAttrOfType<ConfigurationAttr>("intent_cpu.configuration");
    if (failed(fuseReductionTraversals(function)) ||
        failed(vectorizeLoops(function, configuration.getVectorWidth(),
                              configuration.getRegisterReplicas(),
                              configuration.getReductionReplicas()))) return failure();
  }
  if (failed(normalize(module))) return failure();
  return verifyCPUProgram(module, true);
}

}
