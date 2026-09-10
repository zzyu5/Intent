#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/RegionProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>

using namespace mlir;

namespace intent::cpu {

bool isMatrixContraction(linalg::GenericOp operation) {
  if (operation.getInputs().size() != 2 || operation.getOutputs().size() != 1 ||
      operation.getNumResults()) return false;
  AffineExpr m, n, k;
  bindDims(operation.getContext(), m, n, k);
  SmallVector<AffineMap> maps = {
      AffineMap::get(3, 0, {m, k}, operation.getContext()),
      AffineMap::get(3, 0, {k, n}, operation.getContext()),
      AffineMap::get(3, 0, {m, n}, operation.getContext())};
  if (operation.getIndexingMapsArray() != maps ||
      operation.getIteratorTypesArray() != SmallVector<utils::IteratorType>{
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::reduction}) return false;
  Block &body = operation.getRegion().front();
  auto fma = body.getTerminator()->getOperand(0).getDefiningOp<math::FmaOp>();
  return fma && llvm::hasSingleElement(body.without_terminator()) && fma.getA() == body.getArgument(0) &&
      fma.getB() == body.getArgument(1) && fma.getC() == body.getArgument(2);
}

Value PhysicalProgramAnalysis::storageRoot(Value memory) {
  while (true) {
    if (auto view = memory.getDefiningOp<memref::SubViewOp>()) memory = view.getSource();
    else if (auto cast = memory.getDefiningOp<memref::CastOp>()) memory = cast.getSource();
    else if (auto argument = dyn_cast<BlockArgument>(memory)) {
      auto tasks = dyn_cast<TasksOp>(argument.getOwner()->getParentOp());
      if (!tasks || argument.getArgNumber() == 0) return memory;
      memory = tasks.getCaptures()[argument.getArgNumber() - 1];
    }
    else return memory;
  }
}

ViewArgumentAttr PhysicalProgramAnalysis::externalView(Value memory) {
  auto argument = dyn_cast<BlockArgument>(storageRoot(memory));
  if (!argument || argument.getOwner() != &function.front()) return {};
  auto interface = function->getAttrOfType<InterfaceAttr>("intent_cpu.interface");
  return interface ? dyn_cast<ViewArgumentAttr>(interface.getArguments()[argument.getArgNumber()])
                   : ViewArgumentAttr();
}

bool PhysicalProgramAnalysis::isReadOnly(Value memory) {
  auto view = externalView(memory);
  return view && view.getAccess() == 0;
}

bool PhysicalProgramAnalysis::mayReadAt(Value memory, Operation *from, Operation *to) {
  if (isReadOnly(memory)) return true;
  if (from->getBlock() != to->getBlock() || !from->isBeforeInBlock(to)) return false;
  Value root = storageRoot(memory);
  if (!root.getDefiningOp<memref::AllocOp>() && !root.getDefiningOp<memref::AllocaOp>())
    return false;
  for (Operation *operation = from->getNextNode(); operation != to;
       operation = operation->getNextNode()) {
    if (isMemoryEffectFree(operation)) continue;
    auto effects = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!effects) return false;
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    for (auto &effect : instances) {
      if (isa<MemoryEffects::Read>(effect.getEffect())) continue;
      if (!effect.getValue() || storageRoot(effect.getValue()) == root) return false;
    }
  }
  return true;
}

SmallVector<MemoryAccess> PhysicalProgramAnalysis::accesses(Operation *scope) {
  SmallVector<MemoryAccess> result;
  scope->walk([&](Operation *operation) {
    auto add = [&](Value memory, bool read, bool write) {
      if (isa<MemRefType>(memory.getType()))
        result.push_back({operation, memory, read, write});
    };
    if (auto generic = dyn_cast<linalg::LinalgOp>(operation)) {
      for (OpOperand *input : generic.getDpsInputOperands())
        if (generic.payloadUsesValueFromOperand(input)) add(input->get(), true, false);
      for (OpOperand &output : generic.getDpsInitsMutable())
        add(output.get(), generic.payloadUsesValueFromOperand(&output), true);
    } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      for (Value input : reduce.getInputs()) add(input, true, false);
    } else if (isa<RegionFoldOp, RegionScanOp>(operation)) {
      RegionProgram program(operation);
      for (Value input : program.sources()) add(input, true, false);
      for (Value input : program.identities()) add(input, true, false);
      for (Value input : program.initialState()) add(input, true, false);
      for (Value input : program.captures()) add(input, true, false);
      for (Value output : program.outputs()) add(output, false, true);
    } else if (auto quantize = dyn_cast<QuantizeOp>(operation)) {
      add(quantize.getInput(), true, false);
      add(quantize.getOutput(), false, true);
    } else if (auto dot = dyn_cast<QuantizedDotOp>(operation)) {
      add(dot.getLhs(), true, false);
      add(dot.getRhs(), true, false);
    } else if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
      add(copy.getSource(), true, false); add(copy.getTarget(), false, true);
    } else if (auto load = dyn_cast<memref::LoadOp>(operation)) add(load.getMemref(), true, false);
    else if (auto load = dyn_cast<vector::LoadOp>(operation)) add(load.getBase(), true, false);
    else if (auto store = dyn_cast<memref::StoreOp>(operation)) add(store.getMemref(), false, true);
    else if (auto store = dyn_cast<vector::StoreOp>(operation)) add(store.getBase(), false, true);
  });
  return result;
}

SmallVector<AllocationFacts> PhysicalProgramAnalysis::allocations() {
  SmallVector<AllocationFacts> result;
  function.walk([&](Operation *operation) {
    if (!isa<memref::AllocOp, memref::AllocaOp>(operation)) return;
    Value value = operation->getResult(0);
    auto type = cast<MemRefType>(value.getType());
    std::optional<int64_t> bytes;
    int64_t elementBytes = type.getElementType().isIndex() ? 8 : (type.getElementTypeBitWidth() + 7) / 8;
    if (type.hasStaticShape() && type.getNumElements() <= std::numeric_limits<int64_t>::max() / elementBytes)
      bytes = type.getNumElements() * elementBytes;
    Operation *writer = nullptr;
    bool multiple = false;
    for (Operation *user : value.getUsers()) {
      bool writes = false;
      if (auto generic = dyn_cast<linalg::LinalgOp>(user))
        writes = llvm::is_contained(generic.getDpsInits(), value);
      else if (auto store = dyn_cast<memref::StoreOp>(user)) writes = store.getMemref() == value;
      else if (auto copy = dyn_cast<memref::CopyOp>(user)) writes = copy.getTarget() == value;
      else if (auto quantize = dyn_cast<QuantizeOp>(user)) writes = quantize.getOutput() == value;
      else if (!isa<memref::LoadOp, memref::DimOp, memref::DeallocOp, ReduceOp, QuantizedDotOp>(user))
        multiple = true;
      if (writes) {
        if (writer) multiple = true;
        writer = user;
      }
    }
    result.push_back({value, operation->getParentOp(), multiple ? nullptr : writer,
                      bytes, isa<memref::AllocaOp>(operation)});
  });
  return result;
}

LogicalResult PhysicalProgramAnalysis::verify(bool realized) {
  auto interface = function->getAttrOfType<InterfaceAttr>("intent_cpu.interface");
  if (!interface || interface.getArguments().size() != function.getNumArguments() ||
      !interface.getContiguousViews() || !interface.getDisjointOutputs())
    return function.emitError("CPU function requires its complete typed native interface");
  for (auto [argument, field] : llvm::zip(function.getArguments(), interface.getArguments())) {
    if (auto view = dyn_cast<ViewArgumentAttr>(field)) {
      auto type = dyn_cast<MemRefType>(argument.getType());
      if (!type || type.getElementType() != view.getElementType() ||
          type.getShape() != view.getShape().asArrayRef() || !type.getLayout().isIdentity())
        return function.emitError("CPU view type disagrees with its physical ABI");
    } else if (auto scalar = dyn_cast<ScalarArgumentAttr>(field)) {
      if (scalar.getType() != argument.getType())
        return function.emitError("CPU scalar type disagrees with its physical ABI");
    } else return function.emitError("CPU interface contains an unknown argument schema");
  }
  auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>(
      "intent_cpu.capabilities");
  for (auto facts : allocations()) {
    if (facts.stack && capabilities && (!facts.bytes || *facts.bytes > capabilities.getPrivateBytes()))
      return facts.value.getDefiningOp()->emitError("CPU stack allocation exceeds its declared budget");
    if (!facts.stack) {
      memref::DeallocOp deallocation;
      for (Operation *user : facts.value.getUsers())
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
          if (deallocation) return dealloc.emitError("CPU allocation has more than one lifetime end");
          deallocation = dealloc;
        }
      if (!deallocation || deallocation->getBlock() != facts.value.getDefiningOp()->getBlock())
        return facts.value.getDefiningOp()->emitError("CPU heap allocation requires an explicit lexical lifetime end");
      for (Operation *user : facts.value.getUsers()) {
        Operation *ancestor = deallocation->getBlock()->findAncestorOpInBlock(*user);
        if (!ancestor || (ancestor != deallocation && !ancestor->isBeforeInBlock(deallocation)))
          return user->emitError("CPU buffer use escapes its declared lifetime");
      }
    }
  }
  bool invalid = false;
  for (MemoryAccess access : accesses(function)) {
    auto view = externalView(access.memory);
    if (access.write && view && view.getAccess() != 1) {
      access.operation->emitError("CPU write contradicts its input-only ABI");
      invalid = true;
    }
  }
  function.walk([&](Operation *operation) {
    if (realized && (isa<RegionFoldOp, RegionScanOp, ReduceOp, QuantizeOp, QuantizedDotOp>(operation) || operation->getName().getDialectNamespace() == "linalg")) {
      operation->emitError("CPU structured operation has not been materialized for the provider");
      invalid = true;
    }
  });
  return failure(invalid);
}

LogicalResult verifyCPUProgram(ModuleOp module, bool realized) {
  if (failed(mlir::verify(module))) return failure();
  bool invalid = false;
  module.walk([&](Operation *operation) {
    llvm::StringRef dialect = operation->getName().getDialectNamespace();
    if (dialect != "builtin" && dialect != "func" && dialect != "arith" &&
        dialect != "math" && dialect != "memref" && dialect != "scf" &&
        dialect != "vector" && dialect != "linalg" && dialect != "intent_cpu") {
      operation->emitError("operation is outside the current CPU execution family");
      invalid = true;
    }
  });
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal() && function->hasAttr("cpu.external_runtime")) continue;
    if (failed(PhysicalProgramAnalysis(function).verify(realized))) invalid = true;
  }
  return failure(invalid);
}

}
