#include "Intent/Target/Weft/Serialization/Serializer.h"
#include "Intent/Target/Weft/IR/Program.h"
#include "Intent/Target/Weft/IR/WeftAttrs.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "TaskABI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
namespace {

llvm::json::Array integers(DenseI64ArrayAttr entries) {
  llvm::json::Array result;
  for (int64_t entry : entries.asArrayRef()) result.push_back(entry);
  return result;
}

FailureOr<llvm::json::Array> serializeParameters(func::FuncOp function) {
  auto interface = function->getAttrOfType<cpu::InterfaceAttr>("intent_cpu.interface");
  llvm::json::Array parameters;
  for (auto [index, parameter] : llvm::enumerate(interface.getArguments())) {
    if (auto view = dyn_cast<cpu::ViewArgumentAttr>(parameter)) {
      Type element = view.getElementType();
      std::string dtype;
      if (element.isF32()) dtype = "f32";
      else if (auto integer = dyn_cast<IntegerType>(element))
        dtype = (integer.isUnsigned() ? "u" : "i") + std::to_string(integer.getWidth());
      else return function.emitError("CPU view has no Weft native dtype"), failure();
      auto alignment = function.getArgAttrOfType<ArgumentAlignmentAttr>(index, argumentAlignmentAttr);
      parameters.push_back(llvm::json::Object{{"name", view.getName().getValue().str()},
          {"kind", "view"}, {"dtype", dtype},
          {"shape", integers(view.getShape())}, {"dimensions", integers(view.getDimensions())},
          {"alignment", alignment.getBytes()}, {"access", view.getAccess()},
          {"alias", view.getAlias().getValue().str()}, {"noalias", view.getNoalias()}});
    } else {
      auto scalar = cast<cpu::ScalarArgumentAttr>(parameter);
      parameters.push_back(llvm::json::Object{{"name", scalar.getName().getValue().str()},
          {"kind", "scalar"}, {"dtype", scalar.getType().isF32() ? "f32" : "i64"}});
    }
  }
  return std::move(parameters);
}

llvm::json::Object serializeCandidate(func::FuncOp function) {
  auto config = function->getAttrOfType<cpu::ConfigurationAttr>("intent_cpu.configuration");
  llvm::json::Array implementations;
  for (Attribute entry : function->getAttrOfType<ArrayAttr>("intent_cpu.implementations")) {
    auto binding = cast<cpu::ImplementationAttr>(entry);
    llvm::json::Object values;
    for (NamedAttribute parameter : binding.getParameters())
      values[parameter.getName().getValue()] = cast<IntegerAttr>(parameter.getValue()).getInt();
    implementations.push_back(llvm::json::Object{{"name", binding.getName().getValue().str()},
                                               {"parameters", std::move(values)}});
  }
  return llvm::json::Object{{"entry", function.getName().str()},
      {"values", llvm::json::Array{config.getTaskGrain(), config.getTileM(), config.getTileN(),
                                   config.getTileK(), config.getRegionSize()}},
      {"requires_matrix_i8_i32", function->getAttrOfType<BoolAttr>("intent_cpu.requires_matrix_i8_i32").getValue()},
      {"implementations", std::move(implementations)}};
}

FailureOr<llvm::json::Array> serializeTasks(ModuleOp program) {
  llvm::json::Array tasks;
  for (Attribute entry : program->getAttrOfType<ArrayAttr>(taskBindingsAttr)) {
    auto binding = cast<TaskBindingAttr>(entry);
    auto hostEntry = cast<func::FuncOp>(SymbolTable::lookupSymbolIn(program, binding.getHostEntry()));
    auto kernel = cast<wk::KernelOp>(SymbolTable::lookupSymbolIn(program, binding.getDeviceKernel()));
    auto abi = serializeTaskABI(kernel);
    if (failed(abi)) return failure();
    tasks.push_back(llvm::json::Object{{"cpu_entry", hostEntry.getName().str()},
        {"task_ordinal", binding.getOrdinal()}, {"weft_entry", kernel.getSymName().str()},
        {"coordinate_argument", binding.getCoordinateArgument()}, {"abi", std::move(*abi)},
        {"argument_count", static_cast<int64_t>(kernel.getBody().front().getNumArguments())}});
  }
  return std::move(tasks);
}

} // namespace

LogicalResult serializeProgram(ModuleOp program, std::string &source, std::string &metadata) {
  if (failed(verifyProgram(program))) return failure();
  auto modules = getProgramModules(program);
  if (failed(modules)) return failure();
  SmallVector<func::FuncOp> candidates;
  for (auto function : modules->host.getOps<func::FuncOp>())
    if (!function.isExternal()) candidates.push_back(function);
  auto parameters = serializeParameters(candidates.front());
  auto tasks = serializeTasks(program);
  if (failed(parameters) || failed(tasks)) return failure();
  llvm::json::Array configurations;
  for (auto function : candidates) configurations.push_back(serializeCandidate(function));
  std::string hostSource;
  if (failed(serializeHostProgram(modules->host, hostSource))) return failure();

  llvm::raw_string_ostream output(source);
  modules->device.print(output, OpPrintingFlags().enableDebugInfo());
  output << '\n';
  auto interface = candidates.front()->getAttrOfType<cpu::InterfaceAttr>("intent_cpu.interface");
  auto capabilities = modules->host->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities");
  llvm::raw_string_ostream metadataStream(metadata);
  metadataStream << llvm::json::Value(llvm::json::Object{{"kind", "weft-generation"},
      {"native", false}, {"host_source", hostSource}, {"tasks", std::move(*tasks)},
      {"parameters", std::move(*parameters)}, {"candidates", std::move(configurations)},
      {"contiguous_views", interface.getContiguousViews()}, {"disjoint_outputs", interface.getDisjointOutputs()},
      {"matrix_i8_i32", capabilities.getMatrixI8I32()}, {"workers", capabilities.getWorkers()}});
  return success();
}

} // namespace intent::weft_provider
