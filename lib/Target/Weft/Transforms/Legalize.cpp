#include "Intent/Dialect/CPU/Transforms/Structure/Computations.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Target/Weft/Transforms/Passes.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "Intent/Target/Weft/IR/Program.h"
#include "Intent/Target/Weft/IR/WeftDialect.h"
#include "Intent/Target/Weft/Serialization/HostScalar.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Bufferization.h"
#include "TaskLowering.h"
#include "Quantization.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include <algorithm>
#include <string>

using namespace mlir;
namespace wk = ::weft::kernel;
namespace intent::weft_provider {

LogicalResult legalizeProgram(ModuleOp program) {
  auto registry = cpu::lookupImplementationProvider(program, "weft");
  if (failed(registry)) return failure();
  if (failed(cpu::verifyCPUProgram(program, cpu::CPUProgramStage::Buffers))) return failure();
  if (failed(cpu::lowerOwnership(program))) return failure();
  // Canonical Weft has ordered SCF carries, but no native prefix collective.
  // Form the existing CPU scan traversal before any task axis/storage snapshot.
  SmallVector<cpu::ScanOp> scans;
  program.walk([&](cpu::ScanOp scan) { scans.push_back(scan); });
  for (cpu::ScanOp scan : scans)
  {
    auto function = scan->getParentOfType<func::FuncOp>();
    cpu::StorageAnalysis storage(function);
    DominanceInfo dominance(function);
    SmallVector<std::pair<Value, Value>> initializations;
    for (auto [initial, output] : llvm::zip(scan.getInitials(), scan.getOutputs())) {
      auto allocation = output.getDefiningOp<memref::AllocOp>();
      if (!allocation || !allocation->getParentOfType<cpu::TasksOp>() ||
          initial.getType() != allocation.getType().getElementType()) continue;
      auto aliases = storage.aliases(output);
      auto accesses = storage.accesses(output);
      if (!aliases.complete || !accesses.complete || accesses.ordered) continue;
      bool complete = true;
      for (Value source : scan.getSources()) {
        auto type = dyn_cast<MemRefType>(source.getType());
        if (!type || type.getRank() != allocation.getType().getRank() ||
            !storage.disjoint(source, output)) { complete = false; break; }
        for (unsigned axis = 0; axis < type.getRank(); ++axis)
          complete &= cpu::haveEqualExtents(ValueBoundsConstraintSet::Variable(source, axis),
              ValueBoundsConstraintSet::Variable(output, axis));
      }
      for (Value capture : scan.getCaptures())
        if (isa<MemRefType>(capture.getType())) complete &= storage.disjoint(capture, output);
      for (Value other : scan.getOutputs())
        if (other != output) complete &= storage.disjoint(other, output);
      for (const cpu::StorageEffect &access : accesses.entries) {
        if (access.operation == scan.getOperation() ||
            !isa<MemoryEffects::Read, MemoryEffects::Write>(access.effect.getEffect())) continue;
        if (!access.effect.getValue() ||
            !dominance.properlyDominates(scan.getOperation(), access.operation))
          complete = false;
      }
      for (Operation *user : aliases.users) {
        if (user == scan.getOperation() || cpu::isStorageAliasOperation(user) ||
            isa<memref::DimOp, memref::DeallocOp>(user)) continue;
        auto effects = storage.effects(user);
        if (!effects.complete || effects.ordered || effects.entries.empty()) complete = false;
      }
      if (complete) initializations.emplace_back(initial, output);
    }
    // The complete private prefix is observed only after Scan. Its real seed
    // supplies the physical update owner's otherwise unobserved lanes; it is
    // not a definition for arbitrary uninitialized allocations or partial scans.
    OpBuilder builder(scan);
    for (auto [initial, output] : initializations)
      builder.create<linalg::FillOp>(scan.getLoc(), ValueRange{initial}, ValueRange{output});
    if (failed(cpu::materializeStructuredComputation(scan, 1, {}))) return failure();
  }
  program.getContext()->loadDialect<IntentWeftDialect, wk::WEFTKernelDialect>();
  auto cpuProgram = ModuleOp::create(program.getLoc(), hostModuleName);
  auto output = ModuleOp::create(program.getLoc(), deviceModuleName);
  auto containerName = program.getSymNameAttr();
  auto options = readCompileOptions(program);
  if (failed(options)) return failure();
  cpuProgram->setAttrs(program->getAttrDictionary());
  cpuProgram.setSymName(hostModuleName);
  cpuProgram.getBodyRegion().takeBody(program.getBodyRegion());
  program.getBodyRegion().emplaceBlock();
  program->setAttrs(DictionaryAttr::get(program.getContext()));
  program->setAttr(compileOptionsAttr, *options);
  if (containerName) program.setSymNameAttr(containerName);
  program.getBody()->push_back(cpuProgram);
  program.getBody()->push_back(output);
  bool quantized = false;
  cpuProgram.walk([&](cpu::QuantizedDotOp) { quantized = true; });
  cpuProgram.walk([&](cpu::QuantizeOp) { quantized = true; });
  if (quantized) declareQuantEncodings(output);
  SmallVector<Attribute> taskBindings;
  SmallVector<func::FuncOp> functions(cpuProgram.getOps<func::FuncOp>());
  if (functions.empty()) return cpuProgram.emitError("Weft conversion requires CPU candidates");
  auto interface = getPublicInterface(functions.front());
  cpu::StorageAnalysis rootAnalysis(functions.front());
  llvm::DenseMap<Value, int64_t> alignments;
  auto alignmentStatus = functions.front().walk([&](cpu::QuantizedDotOp op) {
    for (auto [memory, alignment] :
         {std::pair<Value, int64_t>{op.getLhs(), 2}, {op.getRhs(), 4}}) {
      auto origins = rootAnalysis.origins(memory);
      if (!origins.complete) {
        op.emitError("encoded input alignment requires known storage origins");
        return WalkResult::interrupt();
      }
      for (Value origin : origins.values)
        alignments[origin] = std::max(alignments.lookup(origin), alignment);
    }
    return WalkResult::advance();
  });
  if (alignmentStatus.wasInterrupted()) return failure();
  SmallVector<ArgumentAlignmentAttr> publicAlignments(interface.getArguments().size());
  for (auto [index, parameter] : llvm::enumerate(interface.getArguments())) {
    if (auto view = getPublicView(interface, index)) {
      Type element = publicViewTensor(view).getElementType();
      auto storage = denseStorageType(element);
      if (!storage)
        return cpuProgram.emitError("CPU view has no Weft native dtype");
      int64_t alignment = std::max(storage->bytes,
          alignments.lookup(functions.front().getArgument(index)));
      publicAlignments[index] = ArgumentAlignmentAttr::getChecked(
          cpuProgram.getLoc(), program.getContext(), alignment);
      if (!publicAlignments[index]) return failure();
    }
  }
  auto symbol = [&](StringRef module, StringRef name) {
    return SymbolRefAttr::get(program.getContext(), module,
        {FlatSymbolRefAttr::get(program.getContext(), name)});
  };
  for (func::FuncOp function : functions) {
    for (auto [index, alignment] : llvm::enumerate(publicAlignments))
      if (alignment) function.setArgAttr(index, argumentAlignmentAttr, alignment);
    cpu::StorageAnalysis storage(function);
    llvm::DenseMap<Value, intent::QuantFormat> formats;
    bool conflict = false;
    bool unresolved = false;
    auto requireFormat = [&](Value memory, intent::QuantFormat format) {
      auto origins = storage.origins(memory);
      unresolved |= !origins.complete;
      for (Value origin : origins.values) {
        auto [it, inserted] = formats.try_emplace(origin, format);
        if (!inserted && it->second != format) conflict = true;
      }
    };
    function.walk([&](cpu::QuantizeOp op) { requireFormat(op.getOutput(), op.getFormat()); });
    function.walk([&](cpu::QuantizedDotOp op) {
      requireFormat(op.getLhs(), op.getLhsFormat()); requireFormat(op.getRhs(), op.getRhsFormat());
    });
    if (unresolved) return function.emitError("encoded CPU storage has unresolved origins");
    if (conflict) return function.emitError("one CPU storage has incompatible quantized record interpretations");
    TaskLowering lowering(function, output, formats, **registry);
    if (failed(lowering.normalizeComputations())) return failure();
    OpBuilder host(function.getContext());
    host.setInsertionPointToStart(&function.front());
    auto shapeArguments = lowering.shapeArguments(function, host);
    SmallVector<cpu::TasksOp> tasks;
    function.walk([&](cpu::TasksOp operation) { tasks.push_back(operation); });
    if (tasks.empty()) return function.emitError("Weft generation requires an explicit CPU task interface");
    for (auto [ordinal, task] : llvm::enumerate(tasks)) {
      std::string name = function.getName().str() + "_task_" + std::to_string(ordinal);
      SmallVector<unsigned> argumentPositions;
      for (BlockArgument argument : task.getBody().front().getArguments())
        if (!argument.use_empty()) argumentPositions.push_back(argument.getArgNumber());
      if (failed(lowering.lower(task, name, argumentPositions))) return failure();
      host.setInsertionPoint(task);
      auto loc = task.getLoc();
      auto zero = host.create<arith::ConstantIndexOp>(loc, 0);
      auto one = host.create<arith::ConstantIndexOp>(loc, 1);
      auto loop = host.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{task.getCount()}, ValueRange{one});
      host.setInsertionPointToStart(loop.getBody());
      SmallVector<Value> arguments;
      SmallVector<Value> original{loop.getInductionVars()[0]};
      llvm::append_range(original, task.getCaptures());
      for (unsigned position : argumentPositions) {
        Value capture = original[position];
        if (isa<MemRefType>(capture.getType())) arguments.push_back(capture);
        else {
          Type type = capture.getType().isIndex() ? Type(host.getI64Type()) : capture.getType();
          Value box = host.create<memref::AllocaOp>(loc, MemRefType::get({1}, type));
          Value scalar = capture;
          if (capture.getType().isIndex()) scalar = host.create<arith::IndexCastOp>(loc, type, capture);
          host.create<memref::StoreOp>(loc, scalar, box, ValueRange{zero});
          arguments.push_back(box);
        }
      }
      llvm::append_range(arguments, shapeArguments);
      {
        OpBuilder::InsertionGuard guard(host);
        host.setInsertionPointToStart(cpuProgram.getBody());
        auto declaration = host.create<func::FuncOp>(loc, name,
            host.getFunctionType(TypeRange(arguments), {}));
        declaration.setPrivate();
        declaration->setAttr("cpu.external_runtime", host.getUnitAttr());
      }
      auto deviceTask = cast<wk::KernelOp>(SymbolTable::lookupSymbolIn(output, name));
      auto accessModes = taskCallAccesses(deviceTask);
      if (failed(accessModes)) return failure();
      host.create<cpu::InvokeOp>(loc, FlatSymbolRefAttr::get(program.getContext(), name),
                                 arguments, *accessModes);
      taskBindings.push_back(TaskBindingAttr::get(program.getContext(),
          symbol(hostModuleName, function.getName()), symbol(hostModuleName, name),
          symbol(deviceModuleName, name), ordinal,
          llvm::is_contained(argumentPositions, 0u) ? 0 : -1));
      task.erase();
    }
    if (failed(cpu::materializeStructuredComputations(
            function, [](Operation *operation) {
              return cpu::materializeStructuredComputation(operation, 1, {});
            })))
      return failure();
  }
  program->setAttr(taskBindingsAttr, ArrayAttr::get(program.getContext(), taskBindings));
  return verifyProgram(program);
}

}
