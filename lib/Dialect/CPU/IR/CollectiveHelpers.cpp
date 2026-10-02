#include "Intent/Dialect/CPU/IR/CollectiveHelpers.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::cpu {

MemRefType collectiveStateType(Type type) {
  if (auto memory = dyn_cast<MemRefType>(type))
    return MemRefType::get(memory.getShape(), memory.getElementType());
  return MemRefType::get({}, type);
}

MemRefType collectiveMemberType(MemRefType source, ArrayRef<int64_t> axes) {
  SmallVector<int64_t> shape;
  for (int64_t axis = 0; axis < source.getRank(); ++axis)
    if (!llvm::is_contained(axes, axis))
      shape.push_back(source.getDimSize(axis));
  return MemRefType::get(shape, source.getElementType(),
      StridedLayoutAttr::get(source.getContext(), ShapedType::kDynamic,
          SmallVector<int64_t>(shape.size(), ShapedType::kDynamic)));
}

LogicalResult verifyCollectiveHelper(Operation *owner, Region &region,
                                     unsigned destinations) {
  Block &body = region.front();
  for (BlockArgument destination : body.getArguments().take_back(destinations)) {
    bool written = false;
    for (OpOperand &use : destination.getUses()) {
      Operation *user = use.getOwner();
      if (isa<memref::DimOp>(user)) continue;
      if (user->getBlock() != &body)
        return owner->emitOpError("collective destinations require unconditional complete writes");
      if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getTarget() != destination || copy.getSource() == destination)
          return owner->emitOpError("collective destinations cannot be read before initialization");
      } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemref() != destination ||
            cast<MemRefType>(destination.getType()).getRank() || !store.getIndices().empty())
          return owner->emitOpError("collective scalar destination stores require rank-zero slots");
      } else if (auto generic = dyn_cast<linalg::LinalgOp>(user)) {
        if (!llvm::is_contained(generic.getDpsInits(), destination) ||
            generic.getNumReductionLoops() ||
            !generic.getIndexingMapsArray()[use.getOperandNumber()].isPermutation() ||
            generic.payloadUsesValueFromOperand(&use))
          return owner->emitOpError("collective destination computation must define every element");
      } else {
        return owner->emitOpError("collective destination use does not prove a complete write");
      }
      written = true;
    }
    if (!written)
      return owner->emitOpError("collective helper must initialize every destination");
  }
  return verifyCollectiveHelperEffects(owner, region, destinations);
}

LogicalResult verifyCollectiveHelperEffects(Operation *owner, Region &region,
                                            unsigned destinations) {
  Block &body = region.front();
  WalkResult status = region.walk<WalkOrder::PreOrder>([&](Operation *nested) -> WalkResult {
    if (nested->getName().getDialectNamespace() == "intent") {
      nested->emitOpError("CPU collective helper cannot retain canonical operations");
      return WalkResult::interrupt();
    }
    for (Value input : nested->getOperands())
      if (!region.isAncestor(input.getParentRegion())) {
        nested->emitOpError("collective helper has an implicit capture: ") << input;
        return WalkResult::interrupt();
      }
    bool collective = isa<SliceReduceOp, ScanOp, RegionOpInterface>(nested);
    SmallVector<MemoryEffects::EffectInstance> effects;
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(nested)) {
      interface.getEffects(effects);
    } else if (nested->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      return WalkResult::advance();
    } else if (!isMemoryEffectFree(nested)) {
      nested->emitOpError("collective helper requires known effects");
      return WalkResult::interrupt();
    }
    for (auto &effect : effects) {
      if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) continue;
      Value root = effect.getValue();
      while (root) {
        if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(root.getDefiningOp()))
          root = view.getViewSource();
        else if (auto cast = root.getDefiningOp<memref::CastOp>())
          root = cast.getSource();
        else if (auto metadata = root.getDefiningOp<memref::ExtractStridedMetadataOp>())
          root = metadata.getSource();
        else break;
      }
      bool permitted;
      if (auto argument = dyn_cast_or_null<BlockArgument>(root)) {
        permitted = argument.getOwner() == &body &&
            argument.getArgNumber() >= body.getNumArguments() - destinations &&
            isa<MemoryEffects::Write>(effect.getEffect());
      } else {
        Operation *allocation = root ? root.getDefiningOp() : nullptr;
        permitted = isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(allocation) &&
            region.isAncestor(allocation->getParentRegion());
      }
      if (!permitted) {
        nested->emitOpError("collective helper write/free must target its destinations or local scratch");
        return WalkResult::interrupt();
      }
    }
    // The nested collective's verifier owns its formal arguments and scratch.
    // Its interface describes effects on the actual operands at this call site.
    return collective ? WalkResult::skip() : WalkResult::advance();
  });
  return failure(status.wasInterrupted());
}

void SliceReduceOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  unsigned outputBegin = getOperation()->getNumOperands() - getOutputs().size();
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    if (!isa<MemRefType>(operand.get().getType())) continue;
    if (operand.getOperandNumber() >= outputBegin)
      effects.emplace_back(MemoryEffects::Write::get(), &operand);
    else effects.emplace_back(MemoryEffects::Read::get(), &operand);
  }
}

LogicalResult SliceReduceOp::verify() {
  unsigned count = getSources().size();
  if (!count || getIdentities().size() != count || getOutputs().size() != count ||
      getAxes().empty() || !llvm::hasSingleElement(getCombine()))
    return emitOpError("slice reduction requires sources, identities, outputs, axes and one combine block");
  auto first = cast<MemRefType>(getSources().front().getType());
  SmallVector<int64_t> seen;
  for (int64_t axis : getAxes()) {
    if (axis < 0 || axis >= first.getRank() || llvm::is_contained(seen, axis))
      return emitOpError("reduction member axes must be distinct source axes");
    seen.push_back(axis);
  }
  SmallVector<Type> states, members;
  for (auto [source, identity, output] : llvm::zip(getSources(), getIdentities(), getOutputs())) {
    auto input = cast<MemRefType>(source.getType());
    auto destination = cast<MemRefType>(output.getType());
    for (int64_t axis : getAxes())
      if (axis >= input.getRank() || input.getDimSize(axis) != first.getDimSize(axis))
        return emitOpError("reduction sources must share every member extent");
    auto member = collectiveMemberType(input, getAxes());
    auto state = collectiveStateType(identity.getType());
    if (destination.getShape() != member.getShape() ||
        destination.getElementType() != input.getElementType() ||
        state.getElementType() != input.getElementType() ||
        (isa<MemRefType>(identity.getType()) && state.getShape() != member.getShape()))
      return emitOpError("reduction outputs and identities must preserve their complete source slices");
    states.push_back(collectiveStateType(destination));
    members.push_back(member);
  }
  Block &body = getCombine().front();
  auto yield = body.empty() ? SliceReduceYieldOp() : dyn_cast<SliceReduceYieldOp>(body.getTerminator());
  if (!yield) return emitOpError("slice reduction requires slice_reduce_yield");
  SmallVector<Type> arguments;
  if (isDestinationPassing()) {
    arguments = states;
    llvm::append_range(arguments, members);
    llvm::append_range(arguments, getCaptures().getTypes());
    llvm::append_range(arguments, states);
    if (yield.getNumOperands() || !llvm::equal(body.getArgumentTypes(), arguments))
      return emitOpError("slice reduction helper requires states, members, captures and destinations");
    return verifyCollectiveHelper(getOperation(), getCombine(), count);
  }
  for (Value identity : getIdentities()) {
    if (!isa<FloatType, IntegerType, IndexType>(identity.getType()))
      return emitOpError("scalar reduction helper identities must be numeric scalars");
    arguments.push_back(identity.getType());
  }
  llvm::append_range(arguments, getIdentities().getTypes());
  llvm::append_range(arguments, getCaptures().getTypes());
  if (!llvm::equal(body.getArgumentTypes(), arguments) ||
      !llvm::equal(yield.getOperandTypes(), getIdentities().getTypes()))
    return emitOpError("scalar reduction helper must combine two component tuples and captures");
  return verifyCollectiveHelper(getOperation(), getCombine(), 0);
}

} // namespace intent::cpu
