#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/Passes.h"
#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "Intent/Target/BangC/Passes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::compiler {

FailureOr<DictionaryAttr> parseDSAParameterBindings(ModuleOp module,
                                                    StringRef text, bool shapes) {
  auto parsed = llvm::json::parse(text);
  if (!parsed) {
    module.emitError() << "invalid DSA " << (shapes ? "shapes" : "strides")
                       << ": " << llvm::toString(parsed.takeError());
    return failure();
  }
  auto object = parsed->getAsObject();
  if (!object)
    return module.emitError("DSA bindings must be a JSON object"), failure();
  Builder builder(module.getContext());
  NamedAttrList bindings;
  for (auto &[name, value] : *object) {
    auto entries = value.getAsArray();
    if (!entries)
      return module.emitError("DSA parameter binding must be an integer array"), failure();
    SmallVector<int64_t> integers;
    for (auto &entry : *entries) {
      auto integer = entry.getAsInteger();
      if (!integer || (shapes && *integer < -1))
        return module.emitError("invalid DSA bound dimension/stride"), failure();
      integers.push_back(*integer);
    }
    bindings.append(name.str(), builder.getDenseI64ArrayAttr(integers));
  }
  return bindings.getDictionary(module.getContext());
}

namespace {

LogicalResult verifyExistingBindings(func::FuncOp function,
                                      DictionaryAttr bindings, bool shapes) {
  auto interface = getPublicInterface(function);
  for (NamedAttribute binding : bindings) {
    intent::ViewType view;
    for (Attribute attribute : interface.getArguments()) {
      auto parameter = cast<PublicParameterAttr>(attribute);
      if (parameter.getName() == binding.getName()) {
        view = dyn_cast<intent::ViewType>(parameter.getType());
        break;
      }
    }
    if (!view)
      return function.emitError("shared DSA binding does not name an existing public view: ")
             << binding.getName();
    auto values = cast<DenseI64ArrayAttr>(binding.getValue());
    auto tensor = publicViewTensor(view);
    if (values.size() != tensor.getRank())
      return function.emitError("shared DSA binding rank disagrees with the public view: ")
             << binding.getName();
    for (auto [axis, value] : llvm::enumerate(values.asArrayRef())) {
      if (shapes) {
        if (value >= 0 && tensor.getDimSize(axis) != value)
          return function.emitError("shared DSA shape binding would change the existing interface: ")
                 << binding.getName() << " axis " << axis;
      } else {
        auto constraints = view.getConstraints();
        auto fixed = constraints.getHasStrides()
                         ? dyn_cast<IntegerAttr>(constraints.getStrides()[axis])
                         : IntegerAttr();
        if (!fixed || fixed.getInt() != value)
          return function.emitError("shared DSA stride binding would change the existing interface: ")
                 << binding.getName() << " axis " << axis;
      }
    }
  }
  return success();
}

} // namespace

ArrayRef<Backend> dsaBackends() {
  static const Backend adapters[] = {
      {Provider::BangC, "bangc", true, DSABackend{}, nullptr,
       [] { bangc::registerBangCPasses(); bangc::registerBangCPipelines(); },
       [](OpPassManager &manager, const Request &request) {
         bangc::buildBangCPipeline(manager, request.dsa.architecture);
       },
       bangc::serializeProgram}};
  return adapters;
}

void DSABackend::buildConstruction(OpPassManager &manager, const Request &request) const {
  ConstructDSAOptions options;
  options.tile = request.dsa.tile;
  options.tileM = request.dsa.tileM;
  options.tileN = request.dsa.tileN;
  options.tileK = request.dsa.tileK;
  options.regionTile = request.dsa.regionTile;
  options.tasks = request.dsa.tasks;
  options.localBytes = request.dsa.localBytes;
  options.shapes = request.dsa.shapes;
  options.strides = request.dsa.strides;
  manager.addPass(createConstructDSA(options));
}

void DSABackend::buildShared(OpPassManager &manager, const Request &, StringRef) const {
  dsa::buildDSAPipeline(manager);
}

LogicalResult DSABackend::verifySharedInput(ModuleOp module, const Request &request, StringRef) const {
  if (failed(dsa::verifyRealizedProgram(module))) return failure();
  if (module->hasAttr("bangc.architecture"))
    return module.emitError("provider-specific BANG C input cannot be resumed as shared DSA IR");
  if (request.dsa.architecture != "mtp_372")
    return module.emitError("shared DSA input requires the supported mtp_372 implementation profile");
  const auto &options = request.dsa;
  auto expected = dsa::ConfigurationAttr::getChecked(
      [&] { return module.emitError(); }, module.getContext(), options.tile,
      options.tileM, options.tileN, options.tileK, options.regionTile,
      options.tasks, options.localBytes);
  auto shapes = parseDSAParameterBindings(module, options.shapes, true);
  auto strides = parseDSAParameterBindings(module, options.strides, false);
  if (!expected || failed(shapes) || failed(strides))
    return failure();
  for (auto function : module.getOps<func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration") != expected)
      return function.emitError("shared DSA configuration disagrees with the requested target bindings");
    if (failed(verifyExistingBindings(function, *shapes, true)) ||
        failed(verifyExistingBindings(function, *strides, false)))
      return failure();
  }
  return success();
}

} // namespace intent::compiler
