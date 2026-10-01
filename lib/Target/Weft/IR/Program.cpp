#include "Intent/Target/Weft/IR/Program.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
namespace {

bool matchesDenseStorage(Type element, StringRef family) {
  if (element.isIndex()) return family == "i64";
  if (element.isF32()) return family == "f32";
  auto integer = dyn_cast<IntegerType>(element);
  if (!integer) return false;
  return family == (Twine(integer.isUnsigned() ? "u" : "i") +
                    Twine(integer.getWidth())).str();
}

LogicalResult verifyTaskSignature(func::FuncOp declaration, wk::KernelOp kernel) {
  Block &body = kernel.getBody().front();
  if (!declaration.getResultTypes().empty() || declaration.getNumArguments() !=
      body.getNumArguments() + kernel.getShapeSymbols().size())
    return declaration.emitError("host task signature disagrees with its Weft arguments and shape parameters");
  for (auto [index, argument] : llvm::enumerate(body.getArguments())) {
    auto view = dyn_cast<wk::ViewType>(argument.getType());
    auto memory = dyn_cast<MemRefType>(declaration.getArgumentTypes()[index]);
    if (!view || !memory || memory.getRank() != static_cast<int64_t>(view.getShape().size()))
      return declaration.emitError("host task storage rank disagrees with its Weft view");
    auto encoding = dyn_cast<wk::EncodingType>(view.getEncoding());
    if (!encoding) return kernel.emitError("task view requires a native encoding");
    SmallVector<int64_t> storageShape(view.getShape().asArrayRef());
    if (encoding.getKind() == "dense") {
      if (!matchesDenseStorage(memory.getElementType(), encoding.getFamily()))
        return declaration.emitError("host task element type disagrees with its Weft dense encoding");
    } else if (encoding.getKind() == "base") {
      auto encodingDeclaration = SymbolTable::lookupSymbolIn(
          kernel->getParentOfType<ModuleOp>(), encoding.getFamily());
      auto record = dyn_cast_or_null<wk::EncodingDeclOp>(encodingDeclaration);
      if (!record || storageShape.empty() || !memory.getElementType().isInteger(8) ||
          storageShape.back() != record.getElements() || record.getStorageBits() % 8)
        return declaration.emitError("host task requires complete byte-addressed Weft storage records");
      storageShape.back() = record.getStorageBits() / 8;
    } else {
      return declaration.emitError("host task encoding has no native storage representation");
    }
    for (auto [logical, storage] : llvm::zip_equal(storageShape, memory.getShape()))
      if ((logical >= 0) != !ShapedType::isDynamic(storage) ||
          (logical >= 0 && logical != storage))
        return declaration.emitError("host task storage shape disagrees with its Weft view");
  }
  for (Type type : declaration.getArgumentTypes().drop_front(body.getNumArguments()))
    if (!type.isIndex())
      return declaration.emitError("Weft shape parameters require host index arguments");
  return success();
}

} // namespace

FailureOr<ProgramModules> getProgramModules(ModuleOp program) {
  ProgramModules modules;
  for (Operation &operation : program.getBody()->getOperations()) {
    auto child = dyn_cast<ModuleOp>(operation);
    if (!child) return program.emitError("Weft program must contain only host and device modules"), failure();
    if (child.getName() == hostModuleName && !modules.host)
      modules.host = child;
    else if (child.getName() == deviceModuleName && !modules.device)
      modules.device = child;
    else
      return child.emitError("unexpected or duplicate Weft program module"), failure();
  }
  if (!modules.host || !modules.device)
    return program.emitError("Weft program requires named host and device modules"), failure();
  return modules;
}

LogicalResult verifyProgram(ModuleOp program) {
  if (failed(mlir::verify(program))) return failure();
  auto modules = getProgramModules(program);
  if (failed(modules) || failed(cpu::verifyCPUProgram(modules->host, true))) return failure();
  if (!modules->host->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities"))
    return modules->host.emitError("Weft host requires CPU capabilities");

  cpu::InterfaceAttr interface;
  FunctionType signature;
  SmallVector<ArgumentAlignmentAttr> publicAlignments;
  llvm::DenseSet<Operation *> entries, declarations;
  for (func::FuncOp function : modules->host.getOps<func::FuncOp>()) {
    if (function.isExternal()) {
      if (!function->hasAttr("cpu.external_runtime"))
        return function.emitError("Weft host declaration requires a task binding");
      declarations.insert(function);
      continue;
    }
    auto current = function->getAttrOfType<cpu::InterfaceAttr>("intent_cpu.interface");
    bool firstEntry = !interface;
    if (firstEntry) {
      interface = current;
      signature = function.getFunctionType();
      publicAlignments.resize(function.getNumArguments());
    }
    if (!current || current != interface || signature != function.getFunctionType())
      return function.emitError("Weft candidates must share one complete public interface");
    auto implementations = function->getAttrOfType<ArrayAttr>("intent_cpu.implementations");
    if (!function->getAttrOfType<cpu::ConfigurationAttr>("intent_cpu.configuration") ||
        !function->getAttrOfType<BoolAttr>("intent_cpu.requires_matrix_i8_i32") ||
        !implementations || implementations.empty() ||
        !llvm::all_of(implementations, [](Attribute binding) { return isa<cpu::ImplementationAttr>(binding); }))
      return function.emitError("Weft candidate requires its complete typed configuration and implementations");
    for (auto [index, field] : llvm::enumerate(current.getArguments())) {
      if (!isa<cpu::ViewArgumentAttr>(field)) continue;
      auto alignment = function.getArgAttrOfType<ArgumentAlignmentAttr>(index, argumentAlignmentAttr);
      if (!alignment)
        return function.emitError("Weft public view requires a typed alignment");
      if (firstEntry) publicAlignments[index] = alignment;
      else if (publicAlignments[index] != alignment)
        return function.emitError("Weft candidates must share public argument alignment requirements");
    }
    entries.insert(function);
  }
  if (entries.empty()) return program.emitError("Weft program requires an executable host candidate");
  auto bindings = program->getAttrOfType<ArrayAttr>(taskBindingsAttr);
  if (!bindings || bindings.empty()) return program.emitError("Weft program requires typed task bindings");
  llvm::DenseSet<Operation *> boundKernels, boundDeclarations, boundEntries;
  llvm::DenseSet<std::pair<Operation *, int64_t>> ordinals;
  for (Attribute attribute : bindings) {
    auto binding = dyn_cast<TaskBindingAttr>(attribute);
    if (!binding) return program.emitError("Weft task binding must be typed");
    auto entry = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(program, binding.getHostEntry()));
    auto declaration = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(program, binding.getHostCallee()));
    auto kernel = dyn_cast_or_null<wk::KernelOp>(SymbolTable::lookupSymbolIn(program, binding.getDeviceKernel()));
    if (!entry || !entries.contains(entry) || !declaration || !declarations.contains(declaration) ||
        !kernel || kernel->getParentOfType<ModuleOp>() != modules->device ||
        declaration.getName() != kernel.getSymName())
      return program.emitError("Weft task binding does not resolve its host entry, declaration and device kernel");
    if (!boundKernels.insert(kernel).second || !boundDeclarations.insert(declaration).second ||
        !ordinals.insert({entry, binding.getOrdinal()}).second)
      return program.emitError("Weft task bindings must have unique kernels, declarations and entry ordinals");
    boundEntries.insert(entry);
    if (binding.getCoordinateArgument() >=
        static_cast<int64_t>(kernel.getBody().front().getNumArguments()))
      return program.emitError("Weft task coordinate slot is outside the device interface");
    if (failed(verifyTaskSignature(declaration, kernel))) return failure();
    if (binding.getCoordinateArgument() == 0) {
      auto coordinate = cast<wk::ViewType>(kernel.getBody().front().getArgument(0).getType());
      auto encoding = cast<wk::EncodingType>(coordinate.getEncoding());
      if (coordinate.getShape().asArrayRef() != ArrayRef<int64_t>{1} ||
          encoding.getKind() != "dense" || encoding.getFamily() != "i64")
        return kernel.emitError("task coordinate requires one boxed i64 argument");
    }
    bool called = false, wrongEntry = false;
    modules->host.walk([&](func::CallOp call) {
      if (call.getCallee() != declaration.getName()) return;
      called = true;
      wrongEntry |= call->getParentOfType<func::FuncOp>() != entry;
    });
    if (!called || wrongEntry)
      return declaration.emitError("Weft task declaration must be called by its bound host entry");
  }
  if (boundDeclarations.size() != declarations.size() || boundEntries.size() != entries.size())
    return program.emitError("Weft host contains an entry or declaration without a task binding");
  for (Operation &operation : modules->device.getBody()->getOperations())
    if (auto kernel = dyn_cast<wk::KernelOp>(operation)) {
      if (!boundKernels.contains(kernel)) return kernel.emitError("Weft device kernel lacks its host binding");
    } else if (!isa<wk::EncodingDeclOp>(operation)) {
      return operation.emitError("operation is not part of the canonical Weft device program");
    }
  return success();
}

} // namespace intent::weft_provider
