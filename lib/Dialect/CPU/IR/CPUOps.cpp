#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace intent::cpu;

#define GET_OP_CLASSES
#include "Intent/Dialect/CPU/IR/CPUOps.cpp.inc"

LogicalResult TasksOp::verify() {
  Block &body = getBody().front();
  if (body.empty() || body.getNumArguments() != getCaptures().size() + 1 ||
      !body.getArgument(0).getType().isIndex() ||
      !isa<TaskYieldOp>(body.getTerminator()))
    return emitOpError("tasks require an index coordinate, explicit captures and task_yield");
  for (auto [argument, capture] : llvm::zip(body.getArguments().drop_front(), getCaptures()))
    if (argument.getType() != capture.getType())
      return emitOpError("task capture and body argument types must agree");
  llvm::APInt count;
  if (matchPattern(getCount(), m_ConstantInt(&count)) && count.isNegative())
    return emitOpError("task count cannot be negative");
  if (getOperation()->getParentOfType<TasksOp>() || getOperation()->getParentOfType<TaskDispatchOp>())
    return emitOpError("nested task scheduling has not been realized by CPU partitioning");
  return success();
}

LogicalResult TaskDispatchOp::verify() {
  Block &body = getBody().front();
  if (body.empty() || body.getNumArguments() != 1 ||
      !body.getArgument(0).getType().isIndex() || !isa<TaskYieldOp>(body.getTerminator()))
    return emitOpError("task dispatch requires one coordinate and task_yield");
  llvm::APInt count;
  if (matchPattern(getCount(), m_ConstantInt(&count)) && count.isNegative())
    return emitOpError("task dispatch count cannot be negative");
  llvm::APInt workers;
  if (matchPattern(getWorkerCount(), m_ConstantInt(&workers)) && !workers.isStrictlyPositive())
    return emitOpError("task dispatch worker budget must be positive");
  if (getOperation()->getParentOfType<TasksOp>() || getOperation()->getParentOfType<TaskDispatchOp>())
    return emitOpError("task dispatch cannot introduce nested scheduling");
  return success();
}

LogicalResult TaskYieldOp::verify() {
  if (!isa<TasksOp, TaskDispatchOp>(getOperation()->getParentOp()))
    return emitOpError("must terminate tasks or task_dispatch");
  return success();
}

void ReduceOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (OpOperand &operand : getOperation()->getOpOperands())
    if (isa<MemRefType>(operand.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &operand);
}

LogicalResult ReduceOp::verify() {
  if (getInitial().getType() != getResult().getType() || getInputs().empty() ||
      getIndexingMaps().size() != getInputs().size() ||
      !llvm::hasSingleElement(getCombine()))
    return emitOpError("reduction requires one result/identity dtype, input maps and a single combine block");
  Block &block = getCombine().front();
  if (block.empty() || block.getNumArguments() != getInputs().size() + 1 ||
      block.getArgument(0).getType() != getResult().getType())
    return emitOpError("combine arguments must be accumulator followed by input elements");
  for (auto [number, input] : llvm::enumerate(getInputs())) {
    auto memory = dyn_cast<MemRefType>(input.getType());
    Type element = memory ? memory.getElementType() : input.getType();
    auto map = dyn_cast<AffineMapAttr>(getIndexingMaps()[number]);
    if (!map || map.getValue().getNumDims() != 1 || map.getValue().getNumSymbols() ||
        map.getValue().getNumResults() != (memory ? memory.getRank() : 0) ||
        block.getArgument(number + 1).getType() != element)
      return emitOpError("reduction input map or scalar combine type is incomplete");
    for (AffineExpr index : map.getValue().getResults())
      if (!isa<AffineDimExpr>(index) &&
          (!isa<AffineConstantExpr>(index) || cast<AffineConstantExpr>(index).getValue() != 0))
        return emitOpError("reduction maps require the reduction coordinate or a broadcast zero");
  }
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield || yield.getValue().getType() != getResult().getType())
    return emitOpError("combine must yield its accumulator dtype");
  for (Operation &operation : block.without_terminator())
    if (operation.getNumRegions() || !isMemoryEffectFree(&operation))
      return emitOpError("combine must be a closed pure scalar expression");
  return success();
}

void ScanOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  unsigned outputBegin = getOperation()->getNumOperands() - getOutputs().size();
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    if (!isa<MemRefType>(operand.get().getType())) continue;
    if (operand.getOperandNumber() >= outputBegin)
      effects.emplace_back(MemoryEffects::Write::get(), &operand);
    else effects.emplace_back(MemoryEffects::Read::get(), &operand);
  }
}

LogicalResult ScanOp::verify() {
  unsigned count = getSources().size();
  if (!count || getInitials().size() != count || getOutputs().size() != count ||
      !llvm::hasSingleElement(getCombine()))
    return emitOpError("scan requires matching sources, identities and destinations");
  auto first = cast<MemRefType>(getSources()[0].getType());
  if (getAxis() >= static_cast<uint64_t>(first.getRank()))
    return emitOpError("scan axis is outside its source rank");
  if (isDestinationPassing()) {
    SmallVector<Type> states, members;
    for (auto [source, initial, output] : llvm::zip(getSources(), getInitials(), getOutputs())) {
      auto input = cast<MemRefType>(source.getType()), result = cast<MemRefType>(output.getType());
      auto state = dyn_cast<MemRefType>(initial.getType());
      if (!state) state = MemRefType::get({}, initial.getType());
      if (getAxis() >= static_cast<uint64_t>(input.getRank()) ||
          input.getDimSize(getAxis()) != first.getDimSize(getAxis()) ||
          input.getShape() != result.getShape() || input.getElementType() != result.getElementType() ||
          input.getElementType() != state.getElementType())
        return emitOpError("slice scan sources must agree on their member extent and preserve output types");
      SmallVector<int64_t> shape(input.getShape());
      shape.erase(shape.begin() + getAxis());
      if (ArrayRef<int64_t>(shape) != state.getShape())
        return emitOpError("slice scan identities must describe complete source slices");
      states.push_back(MemRefType::get(shape, state.getElementType()));
      members.push_back(MemRefType::get(shape, state.getElementType(),
          StridedLayoutAttr::get(getContext(), ShapedType::kDynamic,
              SmallVector<int64_t>(shape.size(), ShapedType::kDynamic))));
    }
    SmallVector<Type> arguments(states);
    llvm::append_range(arguments, members);
    llvm::append_range(arguments, getCaptures().getTypes());
    llvm::append_range(arguments, states);
    Block &body = getCombine().front();
    auto yield = body.empty() ? ScanYieldOp() : dyn_cast<ScanYieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() || !llvm::equal(body.getArgumentTypes(), arguments))
      return emitOpError("slice scan combine requires incoming states, members, captures and destination slots");
    for (BlockArgument destination : body.getArguments().take_back(count)) {
      bool written = false;
      for (OpOperand &use : destination.getUses()) {
        Operation *user = use.getOwner();
        if (isa<memref::DimOp>(user)) continue;
        if (user->getBlock() != &body)
          return emitOpError("slice scan destinations require unconditional complete writes");
        if (auto copy = dyn_cast<memref::CopyOp>(user)) {
          if (copy.getTarget() != destination || copy.getSource() == destination)
            return emitOpError("slice scan destinations cannot be read before initialization");
        } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
          if (store.getMemref() != destination || cast<MemRefType>(destination.getType()).getRank() || !store.getIndices().empty())
            return emitOpError("slice scan scalar destination stores require rank-zero slots");
        } else if (auto generic = dyn_cast<linalg::LinalgOp>(user)) {
          if (!llvm::is_contained(generic.getDpsInits(), destination) || generic.getNumReductionLoops() ||
              !generic.getIndexingMapsArray()[use.getOperandNumber()].isPermutation() ||
              generic.payloadUsesValueFromOperand(&use))
            return emitOpError("slice scan destination computations must write every element without reading the slot");
        } else return emitOpError("slice scan destination use does not prove a complete write");
        written = true;
      }
      if (!written) return emitOpError("slice scan combine must initialize every destination");
    }
    bool invalid = false;
    getCombine().walk([&](Operation *nested) {
      if (invalid) return;
      for (Value input : nested->getOperands())
        if (!getCombine().isAncestor(input.getParentRegion())) {
          nested->emitOpError("slice scan combine has an implicit capture: ") << input;
          invalid = true;
          return;
        }
      auto effects = getEffectsRecursively(nested);
      if (!effects) {
        nested->emitOpError("slice scan combine requires known effects");
        invalid = true;
        return;
      }
      for (auto &effect : *effects) {
        if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) continue;
        Value root = effect.getValue();
        while (root) {
          if (auto view = root.getDefiningOp<memref::SubViewOp>()) root = view.getSource();
          else if (auto cast = root.getDefiningOp<memref::CastOp>()) root = cast.getSource();
          else break;
        }
        if (auto argument = dyn_cast_or_null<BlockArgument>(root)) {
          invalid |= argument.getOwner() != &body || argument.getArgNumber() < arguments.size() - count ||
              !isa<MemoryEffects::Write>(effect.getEffect());
        } else {
          Operation *owner = root ? root.getDefiningOp() : nullptr;
          invalid |= !isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(owner) ||
              !getCombine().isAncestor(owner->getParentRegion());
        }
        if (invalid) {
          nested->emitOpError("slice scan write/free must target its destinations or local scratch");
          return;
        }
      }
    });
    if (invalid) return emitOpError("slice scan requires explicit read-only inputs and locally owned scratch");
    return success();
  }
  SmallVector<Type> elements;
  for (auto [source, initial, output] : llvm::zip(getSources(), getInitials(), getOutputs())) {
    auto inputType = cast<MemRefType>(source.getType()), outputType = cast<MemRefType>(output.getType());
    if (inputType.getShape() != first.getShape() || outputType.getShape() != inputType.getShape() ||
        inputType.getElementType() != initial.getType() || outputType.getElementType() != initial.getType())
      return emitOpError("scan components must preserve a common shape and their scalar accumulator dtypes");
    elements.push_back(initial.getType());
  }
  SmallVector<Type> arguments(elements);
  llvm::append_range(arguments, elements);
  llvm::append_range(arguments, getCaptures().getTypes());
  Block &body = getCombine().front();
  auto yield = body.empty() ? ScanYieldOp() : dyn_cast<ScanYieldOp>(body.getTerminator());
  if (!yield || !llvm::equal(body.getArgumentTypes(), arguments) || !llvm::equal(yield.getOperandTypes(), elements))
    return emitOpError("scan combine must accept two accumulator tuples and captures, then yield one tuple");
  for (Value capture : getCaptures())
    if (!isa<IntegerType, IndexType, FloatType>(capture.getType()))
      return emitOpError("scalar scan captures must be numeric scalars");
  for (Operation &operation : body.without_terminator())
    if (operation.getNumRegions() || !isMemoryEffectFree(&operation))
      return emitOpError("scalar scan combine must be a closed pure expression");
  return success();
}

LogicalResult HistogramOp::verify() {
  auto values = cast<MemRefType>(getValues().getType());
  auto valid = cast<MemRefType>(getValid().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  if (!isa<IntegerType, IndexType>(values.getElementType()) || values.getElementType().isInteger(1) ||
      values.getShape() != valid.getShape() || !valid.getElementType().isInteger(1) ||
      output.getRank() != 1 || !isa<IntegerType>(output.getElementType()) || output.getElementType().isInteger(1) ||
      (!output.isDynamicDim(0) && output.getDimSize(0) <= 0))
    return emitOpError("histogram requires integer values, a matching bool predicate and a nonempty integer-bin vector");
  return success();
}

namespace {
LogicalResult verifyAtomicAddress(Operation *operation, Value target, ValueRange indices,
                                  TypeRange operands, Type result = {}) {
  auto memory = cast<MemRefType>(target.getType());
  Type element = memory.getElementType();
  if (indices.size() != static_cast<size_t>(memory.getRank()) ||
      !isa<IntegerType, FloatType>(element) || element.isInteger(1) ||
      llvm::any_of(operands, [&](Type type) { return type != element; }) ||
      (result && result != element))
    return operation->emitOpError("atomic operation requires complete coordinates and matching numeric scalar types");
  return success();
}
}

LogicalResult AtomicLoadOp::verify() {
  if (getOrdering() != intent::AtomicOrdering::Relaxed && getOrdering() != intent::AtomicOrdering::Acquire)
    return emitOpError("atomic load requires relaxed or acquire ordering");
  return verifyAtomicAddress(*this, getTarget(), getIndices(), {}, getValue().getType());
}

LogicalResult AtomicStoreOp::verify() {
  if (getOrdering() != intent::AtomicOrdering::Relaxed && getOrdering() != intent::AtomicOrdering::Release)
    return emitOpError("atomic store requires relaxed or release ordering");
  return verifyAtomicAddress(*this, getTarget(), getIndices(), TypeRange{getValue().getType()});
}

LogicalResult AtomicRMWOp::verify() {
  if (getUnsignedInteger() && !isa<IntegerType>(getValue().getType()))
    return emitOpError("unsigned atomic interpretation requires integer storage");
  if (getKind() == intent::AtomicRMWKind::BitwiseAnd || getKind() == intent::AtomicRMWKind::BitwiseOr ||
      getKind() == intent::AtomicRMWKind::BitwiseXor)
    if (!isa<IntegerType>(getValue().getType())) return emitOpError("bitwise atomic RMW requires an integer");
  return verifyAtomicAddress(*this, getTarget(), getIndices(), TypeRange{getValue().getType()}, getOldValue().getType());
}

LogicalResult AtomicCompareExchangeOp::verify() {
  return verifyAtomicAddress(*this, getTarget(), getIndices(),
      TypeRange{getExpected().getType(), getDesired().getType()}, getOldValue().getType());
}

namespace {
bool recordType(Type type, int64_t bytes) {
  auto memory = dyn_cast<MemRefType>(type);
  return memory && memory.getRank() == 2 &&
      (memory.isDynamicDim(1) || memory.getDimSize(1) == bytes) &&
      memory.getElementType().isSignlessInteger(8);
}
}

LogicalResult QuantizeOp::verify() {
  auto input = cast<MemRefType>(getInput().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  if (getFormat() != intent::QuantFormat::Q8K || input.getRank() != 2 ||
      (!input.isDynamicDim(1) && input.getDimSize(1) != 256) ||
      !input.getElementType().isF32() || !recordType(output, 292) ||
      (!input.isDynamicDim(0) && !output.isDynamicDim(0) &&
       input.getDimSize(0) != output.getDimSize(0)))
    return emitOpError("Q8_K preparation requires f32[G,256] and u8[G,292] storage");
  return success();
}

LogicalResult QuantizedDotOp::verify() {
  auto lhs = cast<MemRefType>(getLhs().getType());
  auto rhs = cast<MemRefType>(getRhs().getType());
  auto output = cast<MemRefType>(getOutput().getType());
  if (getLhsFormat() != intent::QuantFormat::Q4K || getRhsFormat() != intent::QuantFormat::Q8K ||
      !recordType(rhs, 292) || (lhs.getRank() != 2 && lhs.getRank() != 3) ||
      !lhs.getElementType().isSignlessInteger(8) ||
      (!lhs.isDynamicDim(lhs.getRank() - 1) && lhs.getShape().back() != 144) ||
      output.getRank() != lhs.getRank() - 2 || !output.getElementType().isF32())
    return emitOpError("quantized dots require Q4_K [G,144] or [R,G,144], shared Q8_K [G,292], and f32 [] or [R] output");
  unsigned recordAxis = lhs.getRank() - 2;
  if (!lhs.isDynamicDim(recordAxis) && !rhs.isDynamicDim(0) && lhs.getDimSize(recordAxis) != rhs.getDimSize(0))
    return emitOpError("quantized dot record extents must agree");
  if (output.getRank() && !lhs.isDynamicDim(0) && !output.isDynamicDim(0) &&
      lhs.getDimSize(0) != output.getDimSize(0))
    return emitOpError("quantized dot output must preserve the independent row extent");
  return success();
}
