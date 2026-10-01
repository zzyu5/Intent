#include "TaskInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
namespace {

void removeDeadValues(wk::KernelOp kernel) {
  SmallVector<Operation *> operations;
  kernel.walk([&](Operation *operation) {
    // Symbols and domains also bind type-level identities; ordinary SSA use
    // counts do not describe their liveness. Keep structured control intact.
    if (operation->getNumRegions() || isa<wk::SymbolOp>(operation) ||
        llvm::any_of(operation->getResultTypes(), [](Type type) { return isa<wk::DomainType>(type); })) return;
    operations.push_back(operation);
  });
  // Users follow their dominating leaf definitions, including values captured
  // by nested regions. Removing unused reads/casts exposes their slices here.
  for (Operation *operation : llvm::reverse(operations))
    if (isOpTriviallyDead(operation)) operation->erase();
}

FailureOr<llvm::json::Object> taskABI(wk::KernelOp kernel) {
  llvm::json::Array arguments, symbols;
  for (Attribute symbol : kernel.getShapeSymbols())
    symbols.push_back(cast<StringAttr>(symbol).getValue().str());
  for (auto [index, argument] : llvm::enumerate(kernel.getBody().front().getArguments())) {
    auto view = cast<wk::ViewType>(argument.getType());
    auto encoding = cast<wk::EncodingType>(view.getEncoding());
    int64_t bytes, elements, alignment;
    std::string scalar;
    if (encoding.getKind() == "dense") {
      if (encoding.getFamily() == "f32") { scalar = "float"; bytes = 4; }
      else if (encoding.getFamily() == "i32") { scalar = "int32_t"; bytes = 4; }
      else if (encoding.getFamily() == "i64") { scalar = "int64_t"; bytes = 8; }
      else if (encoding.getFamily() == "u8" || encoding.getFamily() == "i8") {
        scalar = encoding.getFamily() == "u8" ? "uint8_t" : "int8_t"; bytes = 1;
      } else return kernel.emitError("CPU task ABI has no native dense element representation"), failure();
      elements = 1; alignment = bytes;
    } else {
      wk::EncodingDeclOp declaration;
      for (auto candidate : kernel->getParentOfType<ModuleOp>().getOps<wk::EncodingDeclOp>())
        if (candidate.getSymName() == encoding.getFamily()) declaration = candidate;
      if (encoding.getKind() != "base" || !declaration)
        return kernel.emitError("CPU task ABI requires a declared base encoding"), failure();
      scalar = "uint8_t"; bytes = declaration.getStorageBits() / 8;
      elements = declaration.getElements(); alignment = declaration.getAlignment();
    }
    StringRef access = cast<StringAttr>(kernel.getArgAccess()[index]).getValue();
    bool writable = access == "write" || access == "readwrite";
    int64_t alias = kernel.getArgAliasSets()[index];
    bool restricted = alias >= 0 && llvm::count(kernel.getArgAliasSets(), alias) == 1;
    llvm::json::Array dimensions;
    for (int64_t extent : view.getShape().asArrayRef())
      dimensions.push_back(extent >= 0 ? std::to_string(extent)
          : cast<StringAttr>(kernel.getShapeSymbols()[-extent - 1]).getValue().str());
    arguments.push_back(llvm::json::Object{
        {"name", cast<StringAttr>(kernel.getArgNames()[index]).getValue().str()},
        {"c_type", (writable ? "" : "const ") + scalar + (restricted ? " *restrict" : " *")},
        {"encoding", encoding.getLayoutIdentity().str()}, {"shape", std::move(dimensions)},
        {"storage_bytes", bytes}, {"record_elements", elements}, {"alignment", alignment},
        {"interleave_rows", 0}, {"alias_set", alias}, {"writable", writable}});
  }
  return llvm::json::Object{{"symbol", kernel.getSymName().str()},
      {"arguments", std::move(arguments)}, {"shape_parameters", std::move(symbols)}};
}

} // namespace

FailureOr<llvm::json::Object> finalizeTaskInterface(
    wk::KernelOp kernel, SmallVectorImpl<unsigned> &argumentPositions) {
  removeDeadValues(kernel);
  Block &body = kernel.getBody().front();
  SmallVector<Attribute> names, accesses;
  SmallVector<int64_t> aliases;
  SmallVector<unsigned> positions;
  for (auto [index, argument] : llvm::enumerate(body.getArguments())) {
    if (argument.use_empty()) continue;
    names.push_back(kernel.getArgNames()[index]);
    accesses.push_back(kernel.getArgAccess()[index]);
    aliases.push_back(kernel.getArgAliasSets()[index]);
    positions.push_back(argumentPositions[index]);
  }
  body.eraseArguments([](BlockArgument argument) { return argument.use_empty(); });
  Builder builder(kernel.getContext());
  kernel.setArgNamesAttr(builder.getArrayAttr(names));
  kernel.setArgAccessAttr(builder.getArrayAttr(accesses));
  kernel.setArgAliasSetsAttr(builder.getDenseI64ArrayAttr(aliases));
  argumentPositions.assign(positions.begin(), positions.end());
  if (failed(verify(kernel))) return failure();
  return taskABI(kernel);
}

} // namespace intent::weft_provider
