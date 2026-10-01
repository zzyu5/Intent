#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/Passes.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Target/Mojo/Serialization/Serializer.h"
#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#ifdef INTENT_HAS_WEFT_CANONICAL
#include "Intent/Target/Weft/IR/WeftDialect.h"
#include "Intent/Target/Weft/Serialization/Serializer.h"
#include "Intent/Target/Weft/Transforms/Passes.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#endif

using namespace mlir;

namespace intent::compiler {

ArrayRef<Backend> cpuBackends() {
  static const Backend adapters[] = {
      {Provider::Mojo, "mojo", true, CPUBackend{true, "mojo.json", mojo::implementations},
       nullptr, mojo::registerMojoPasses,
       [](OpPassManager &manager, const Request &) { mojo::buildMojoPipeline(manager); },
       mojo::serializeProgram},
#ifdef INTENT_HAS_WEFT_CANONICAL
      {Provider::Weft, "weft", true, CPUBackend{false, "weft.json", weft_provider::implementations},
       [](DialectRegistry &registry) {
         registry.insert<weft_provider::IntentWeftDialect, ::weft::kernel::WEFTKernelDialect>();
       },
       weft_provider::registerWeftPasses,
       [](OpPassManager &manager, const Request &) {
         weft_provider::buildWeftPipeline(manager);
       },
       weft_provider::serializeProgram}
#else
      {Provider::Weft, "weft", false, CPUBackend{false, "weft.json", nullptr},
       nullptr, nullptr, nullptr, nullptr}
#endif
  };
  return adapters;
}

void CPUBackend::buildConstruction(OpPassManager &manager, const Request &) const {
  ConstructCPUOptions options;
  options.stridedInputs = stridedInputs;
  manager.addPass(createConstructCPU(options));
}

void CPUBackend::buildShared(OpPassManager &manager, const Request &request, StringRef provider) const {
  cpu::CPUCompilationOptions options;
  options.provider = provider.str();
  llvm::SmallString<256> filename(request.profileDirectory);
  llvm::sys::path::append(filename, profileFilename);
  options.defaults = std::string(filename);
  options.overrides = request.tuningConfig;
  options.vectorBits = request.cpu.vectorBits;
  options.workers = request.cpu.workers;
  options.matrixI8I32 = request.cpu.matrixI8I32;
  cpu::buildCPUPipeline(manager, options);
}

LogicalResult CPUBackend::verifySharedInput(ModuleOp module, const Request &request, StringRef provider) const {
  if (failed(cpu::verifyCPUProgram(module, false))) return failure();
  cpu::CPUCompilationOptions defaults;
  auto expected = cpu::CapabilitiesAttr::getChecked([&] { return module.emitError(); },
      module.getContext(), request.cpu.vectorBits, request.cpu.workers,
      defaults.privateBytes.getValue(), request.cpu.matrixI8I32);
  if (!expected) return failure();
  if (module->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities") != expected)
    return module.emitError("shared CPU capabilities disagree with the requested CPU resources");
  auto registry = cpu::lookupImplementationProvider(module, provider);
  if (failed(registry)) return failure();
  if (failed((**registry).verifyBindings(module))) return failure();
  cpu::InterfaceAttr interface;
  FunctionType functionType;
  for (auto function : module.getOps<func::FuncOp>()) {
    auto current = function->getAttrOfType<cpu::InterfaceAttr>("intent_cpu.interface");
    auto summary = function->getAttrOfType<ArrayAttr>("intent_cpu.implementations");
    if (function.isExternal() || !current || !summary || summary.empty() ||
        !function->getAttrOfType<cpu::ConfigurationAttr>("intent_cpu.configuration") ||
        !function->getAttrOfType<BoolAttr>("intent_cpu.requires_matrix_i8_i32"))
      return function.emitError("shared CPU input requires executable candidates with complete bound configuration and implementation summaries");
    if (current.getContiguousViews() != !stridedInputs)
      return function.emitError("shared CPU entry layout disagrees with the selected provider");
    if (interface && (interface != current || functionType != function.getFunctionType()))
      return function.emitError("shared CPU candidates disagree on their typed public interface");
    interface = current;
    functionType = function.getFunctionType();
    auto status = function.walk([&](Operation *operation) -> WalkResult {
      if (!operation->hasAttr("intent_cpu.implementation")) return WalkResult::advance();
      return succeeded((**registry).lookup(operation)) ? WalkResult::advance() : WalkResult::interrupt();
    });
    if (status.wasInterrupted()) return failure();
  }
  if (!interface) return module.emitError("shared CPU input requires at least one executable candidate");
  return success();
}

} // namespace intent::compiler
