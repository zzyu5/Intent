#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/Contractions.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
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

std::optional<SmallVector<std::pair<unsigned, unsigned>>> unitReshapeAxes(Operation *operation) {
  if (!operation) return std::nullopt;
  bool expanding = isa<memref::ExpandShapeOp>(operation);
  if (!expanding && !isa<memref::CollapseShapeOp>(operation)) return std::nullopt;
  auto source = cast<MemRefType>(operation->getOperand(0).getType());
  auto result = cast<MemRefType>(operation->getResult(0).getType());
  auto groups = expanding ? cast<memref::ExpandShapeOp>(operation).getReassociationIndices()
                          : cast<memref::CollapseShapeOp>(operation).getReassociationIndices();
  auto wider = expanding ? result : source;
  auto narrower = expanding ? source : result;
  SmallVector<std::pair<unsigned, unsigned>> axes;
  for (auto [axis, group] : llvm::enumerate(groups)) {
    std::optional<unsigned> nonunit;
    for (int64_t member : group)
      if (wider.getDimSize(member) != 1) {
        if (nonunit) return std::nullopt;
        nonunit = member;
      }
    if (!nonunit) {
      if (narrower.getDimSize(axis) != 1) return std::nullopt;
      continue;
    }
    if (wider.getDimSize(*nonunit) != narrower.getDimSize(axis)) return std::nullopt;
    axes.emplace_back(expanding ? axis : *nonunit, expanding ? *nonunit : axis);
  }
  return axes;
}

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
  return queryContractionAxes(operation).has_value();
}

Value PhysicalProgramAnalysis::storageRoot(Value memory) {
  while (true) {
    if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(memory.getDefiningOp())) memory = view.getViewSource();
    else if (auto cast = memory.getDefiningOp<memref::CastOp>()) memory = cast.getSource();
    else if (auto metadata = memory.getDefiningOp<memref::ExtractStridedMetadataOp>()) memory = metadata.getSource();
    else if (auto argument = dyn_cast<BlockArgument>(memory)) {
      auto tasks = dyn_cast<TasksOp>(argument.getOwner()->getParentOp());
      if (!tasks || argument.getArgNumber() == 0) return memory;
      memory = tasks.getCaptures()[argument.getArgNumber() - 1];
    }
    else return memory;
  }
}

intent::ViewType PhysicalProgramAnalysis::externalView(Value memory) {
  auto argument = dyn_cast<BlockArgument>(storageRoot(memory));
  if (!argument || argument.getOwner() != &function.front()) return {};
  auto interface = getPublicInterface(function);
  return interface ? getPublicView(interface, argument.getArgNumber()) : intent::ViewType();
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
    auto effects = getEffectsRecursively(operation);
    if (!effects) return false;
    for (auto &effect : *effects) {
      if (isa<MemoryEffects::Read>(effect.getEffect())) continue;
      if (!effect.getValue() || storageRoot(effect.getValue()) == root) return false;
    }
  }
  return true;
}

SmallVector<MemoryAccess> PhysicalProgramAnalysis::accesses(Operation *scope) {
  SmallVector<MemoryAccess> result;
  scope->walk<WalkOrder::PreOrder>([&](Operation *operation) {
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
    } else if (auto reduce = dyn_cast<SliceReduceOp>(operation)) {
      for (Value input : reduce.getSources()) add(input, true, false);
      for (Value input : reduce.getIdentities()) add(input, true, false);
      for (Value input : reduce.getCaptures()) add(input, true, false);
      for (Value output : reduce.getOutputs()) add(output, false, true);
    } else if (auto scan = dyn_cast<ScanOp>(operation)) {
      for (Value input : scan.getSources()) add(input, true, false);
      for (Value input : scan.getInitials()) add(input, true, false);
      for (Value input : scan.getCaptures()) add(input, true, false);
      for (Value output : scan.getOutputs()) add(output, false, true);
    } else if (auto histogram = dyn_cast<HistogramOp>(operation)) {
      add(histogram.getValues(), true, false); add(histogram.getValid(), true, false);
      add(histogram.getOutput(), false, true);
    } else if (auto atomic = dyn_cast<AtomicLoadOp>(operation)) {
      add(atomic.getTarget(), true, false);
    } else if (auto atomic = dyn_cast<AtomicStoreOp>(operation)) {
      add(atomic.getTarget(), false, true);
    } else if (auto atomic = dyn_cast<AtomicRMWOp>(operation)) {
      add(atomic.getTarget(), true, true);
    } else if (auto atomic = dyn_cast<AtomicCompareExchangeOp>(operation)) {
      add(atomic.getTarget(), true, true);
    } else if (auto update = dyn_cast<memref::GenericAtomicRMWOp>(operation)) {
      add(update.getMemref(), true, true);
    } else if (isa<RegionFoldOp, RegionScanOp>(operation)) {
      auto program = cast<RegionOpInterface>(operation);
      for (Value input : program.getSources()) add(input, true, false);
      for (Value input : program.getIdentities()) add(input, true, false);
      for (Value input : program.getInitialStates()) add(input, true, false);
      for (Value input : program.getCaptures()) add(input, true, false);
      for (Value output : program.getDestinations()) add(output, false, true);
    } else if (auto quantize = dyn_cast<QuantizeOp>(operation)) {
      add(quantize.getInput(), true, false);
      add(quantize.getOutput(), false, true);
    } else if (auto dot = dyn_cast<QuantizedDotOp>(operation)) {
      add(dot.getLhs(), true, false);
      add(dot.getRhs(), true, false);
      add(dot.getOutput(), false, true);
    } else if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
      add(copy.getSource(), true, false); add(copy.getTarget(), false, true);
    } else if (auto load = dyn_cast<memref::LoadOp>(operation)) add(load.getMemref(), true, false);
    else if (auto load = dyn_cast<vector::LoadOp>(operation)) add(load.getBase(), true, false);
    else if (auto store = dyn_cast<memref::StoreOp>(operation)) add(store.getMemref(), false, true);
    else if (auto store = dyn_cast<vector::StoreOp>(operation)) add(store.getBase(), false, true);
    return isa<SliceReduceOp, ScanOp, RegionOpInterface>(operation)
        ? WalkResult::skip() : WalkResult::advance();
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
      else if (auto dot = dyn_cast<QuantizedDotOp>(user)) writes = dot.getOutput() == value;
      else if (auto scan = dyn_cast<ScanOp>(user)) writes = llvm::is_contained(scan.getOutputs(), value);
      else if (auto reduce = dyn_cast<SliceReduceOp>(user)) writes = llvm::is_contained(reduce.getOutputs(), value);
      else if (auto histogram = dyn_cast<HistogramOp>(user)) writes = histogram.getOutput() == value;
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
  auto interface = getPublicInterface(function);
  auto requirements = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  if (failed(verifyPublicInterface(function, interface))) return failure();
  if (interface.getArguments().empty() || interface.getArguments().size() != function.getNumArguments() ||
      !requirements || !requirements.getDisjointOutputs())
    return function.emitError("CPU function requires its complete typed native interface");
  for (auto [argument, field] : llvm::zip(function.getArguments(), interface.getArguments())) {
    auto storageType = [&](Type logical) -> Type {
      if (auto integer = dyn_cast<IntegerType>(logical)) return IntegerType::get(function.getContext(), integer.getWidth());
      return logical;
    };
    Type logical = cast<PublicParameterAttr>(field).getType();
    if (auto view = dyn_cast<intent::ViewType>(logical)) {
      auto tensor = publicViewTensor(view);
      Type element = tensor.getElementType();
      if (!element.isF16() && !element.isBF16() && !element.isF32() && !element.isF64() &&
          !isa<Float8E4M3FNType, Float8E5M2Type>(element) && !isa<IntegerType>(element))
        return function.emitError("CPU view has unsupported storage element type");
      auto type = dyn_cast<MemRefType>(argument.getType());
      if (!type || type.getElementType() != storageType(element) || type.getShape() != tensor.getShape())
        return function.emitError("CPU view type disagrees with its physical ABI");
      SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(type.getStridesAndOffset(strides, offset)) || offset != 0 ||
          ((requirements.getContiguousViews() || view.getAccess() != 0) && !type.getLayout().isIdentity()))
        return function.emitError("CPU view layout disagrees with its physical ABI");
      if (view.getConstraints().getHasStrides())
        for (auto [constraint, stride] : llvm::zip(view.getConstraints().getStrides(), strides)) {
          if (!isa<UnitAttr, IntegerAttr>(constraint))
            return function.emitError("CPU stride constraint must be an integer or unconstrained");
          if (auto fixed = dyn_cast<IntegerAttr>(constraint);
              fixed && !ShapedType::isDynamic(stride) && fixed.getInt() != stride)
            return function.emitError("CPU view layout contradicts its declared stride constraint");
        }
    } else {
      if (!logical.isF32() && !logical.isF64() && !logical.isIndex() && !isa<IntegerType>(logical))
        return function.emitError("CPU scalar requires a supported C ABI numeric type");
      if (storageType(logical) != argument.getType())
        return function.emitError("CPU scalar type disagrees with its physical ABI");
    }
  }
  auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>(
      "intent_cpu.capabilities");
  for (auto facts : allocations()) {
    if (facts.stack && capabilities && (!facts.bytes || *facts.bytes > capabilities.getPrivateBytes()))
      return facts.value.getDefiningOp()->emitError("CPU stack allocation exceeds its declared budget");
    if (!facts.stack) {
      if (!queryStorageLifetime(cast<memref::AllocOp>(facts.value.getDefiningOp())))
        return facts.value.getDefiningOp()->emitError(
            "CPU heap allocation requires one lexical lifetime end covering every known alias use");
    }
  }
  bool invalid = false;
  for (MemoryAccess access : accesses(function)) {
    auto view = externalView(access.memory);
    if (access.write && view && view.getAccess() == 0) {
      access.operation->emitError("CPU write contradicts its input-only ABI");
      invalid = true;
    }
  }
  function.walk([&](Operation *operation) {
    if (realized && (isa<RegionFoldOp, RegionScanOp, ReduceOp, SliceReduceOp, ScanOp, HistogramOp, QuantizeOp, QuantizedDotOp>(operation) || operation->getName().getDialectNamespace() == "linalg")) {
      operation->emitError("CPU structured operation has not been materialized for the provider");
      invalid = true;
    }
  });
  return failure(invalid);
}

bool supportsVectorScan(ScanOp operation) {
  if (operation.isDestinationPassing()) return false;
  auto type = cast<MemRefType>(operation.getSources()[0].getType());
  if (operation.getAxis() + 1 != static_cast<uint64_t>(type.getRank())) return false;
  auto contiguous = [](Value memory) {
    auto type = cast<MemRefType>(memory.getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    return !type.getElementType().isInteger(1) && succeeded(type.getStridesAndOffset(strides, offset)) &&
        (strides.back() == 1 || ShapedType::isDynamic(strides.back()));
  };
  return llvm::all_of(operation.getSources(), contiguous) && llvm::all_of(operation.getOutputs(), contiguous) &&
      llvm::all_of(operation.getCombine().front().without_terminator(), [](Operation &instruction) {
        return isa<arith::ConstantOp>(instruction) ||
            (instruction.hasTrait<OpTrait::Elementwise>() && instruction.getNumResults() == 1);
      });
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
  func::FuncOp first;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal() && function->hasAttr("cpu.external_runtime")) continue;
    if (failed(PhysicalProgramAnalysis(function).verify(realized))) invalid = true;
    if (!first) first = function;
    else if (getPublicInterface(function) != getPublicInterface(first) ||
             function.getFunctionType() != first.getFunctionType() ||
             function->getAttr(entryRequirementsAttr) != first->getAttr(entryRequirementsAttr)) {
      function.emitError("CPU candidates must share one public interface and physical entry ABI");
      invalid = true;
    }
  }
  return failure(invalid);
}

}
