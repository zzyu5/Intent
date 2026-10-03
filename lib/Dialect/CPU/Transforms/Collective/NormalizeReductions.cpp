#include "Intent/Dialect/CPU/Transforms/Collective/Collectives.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool scalarType(Type type) {
  return isa<FloatType, IntegerType, IndexType>(type);
}

void normalize(linalg::GenericOp operation) {
  if (operation.getNumResults() || operation.getOutputs().size() != 1 ||
      operation.getNumLoops() != 1 || operation.getNumReductionLoops() != 1 ||
      operation->hasAttr("intent_cpu.implementation"))
    return;
  auto order = operation->getAttrOfType<ReductionOrderAttr>(
      "intent_cpu.reduction_order");
  Value destination = operation.getOutputs().front();
  auto allocation = destination.getDefiningOp<memref::AllocOp>();
  auto type = dyn_cast<MemRefType>(destination.getType());
  if (!order || !allocation || !type || type.getRank() ||
      !scalarType(type.getElementType()) ||
      allocation->getBlock() != operation->getBlock() ||
      llvm::is_contained(operation.getInputs(), destination))
    return;
  StorageAnalysis storage(operation->getParentOfType<func::FuncOp>());
  auto lifetime = storage.lifetime(allocation);
  if (!lifetime || !lifetime->aliases.complete ||
      lifetime->aliases.values.size() != 1 || !lifetime->contains(operation))
    return;

  auto maps = operation.getIndexingMapsArray();
  if (maps.back().getNumResults())
    return;
  Value extentSource;
  int64_t extentAxis = 0;
  for (auto [input, map] : llvm::zip(operation.getInputs(), maps)) {
    auto memory = dyn_cast<MemRefType>(input.getType());
    if (!scalarType(memory ? memory.getElementType() : input.getType()) ||
        map.getNumDims() != 1 || map.getNumSymbols())
      return;
    for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
      if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
        if (dimension.getPosition() != 0 || !memory)
          return;
        if (!extentSource) {
          extentSource = input;
          extentAxis = axis;
        }
      } else if (auto constant = dyn_cast<AffineConstantExpr>(expression);
                 !constant || constant.getValue() != 0) {
        return;
      }
    }
  }
  if (!extentSource)
    return;
  Block &body = operation.getRegion().front();
  for (Operation &instruction : body.without_terminator())
    if (instruction.getNumRegions() || isa<linalg::IndexOp>(instruction) ||
        !isMemoryEffectFree(&instruction) ||
        !llvm::all_of(instruction.getOperandTypes(), scalarType) ||
        !llvm::all_of(instruction.getResultTypes(), scalarType))
      return;

  // The rank-zero destination is a private value slot. Keep the complete
  // storage proof here: unrelated SSA roots or an unknown alias are not enough
  // to replace its reads by one scalar result.
  Operation *initialization = nullptr;
  Value initial;
  SmallVector<memref::LoadOp> reads;
  for (Operation *user : lifetime->aliases.users) {
    if (user == operation || user == lifetime->end)
      continue;
    if (user->getBlock() != operation->getBlock())
      return;
    if (auto load = dyn_cast<memref::LoadOp>(user)) {
      if (load.getMemref() != destination || !load.getIndices().empty() ||
          !operation->isBeforeInBlock(load))
        return;
      reads.push_back(load);
      continue;
    }
    Value value;
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      if (store.getMemref() == destination && store.getIndices().empty())
        value = store.getValue();
    } else if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (fill.getNumResults() == 0 && fill.getOutputs().size() == 1 &&
          fill.getOutputs().front() == destination)
        value = fill.getInputs().front();
    }
    if (!value || initialization || !user->isBeforeInBlock(operation))
      return;
    initialization = user;
    initial = value;
  }
  if (!initialization || initial.getType() != type.getElementType())
    return;

  OpBuilder builder(operation);
  Location location = operation.getLoc();
  Value extent = builder.create<memref::DimOp>(location, extentSource, extentAxis);
  auto reduction = builder.create<ReduceOp>(location, initial.getType(), extent,
      initial, operation.getInputs(),
      builder.getAffineMapArrayAttr(ArrayRef(maps).drop_back()), order);
  NamedAttrList attributes(llvm::to_vector(operation->getDiscardableAttrs()));
  attributes.erase("intent_cpu.reduction_order");
  reduction->setDiscardableAttrs(attributes.getAttrs());
  Block &target = reduction.getCombine().emplaceBlock();
  target.addArgument(initial.getType(), location);
  for (BlockArgument argument : body.getArguments().drop_back())
    target.addArgument(argument.getType(), location);
  builder.setInsertionPointToStart(&target);
  IRMapping mapping;
  mapping.map(body.getArguments().back(), target.getArgument(0));
  mapping.map(body.getArguments().drop_back(), target.getArguments().drop_front());
  for (Operation &instruction : body.without_terminator())
    builder.clone(instruction, mapping);
  builder.create<YieldOp>(location,
      mapping.lookupOrDefault(body.getTerminator()->getOperand(0)));
  for (memref::LoadOp read : reads) {
    read.getResult().replaceAllUsesWith(reduction.getResult());
    read.erase();
  }
  operation.erase();
  initialization->erase();
  lifetime->end.erase();
  allocation.erase();
}

} // namespace

LogicalResult normalizeScalarReductions(func::FuncOp function) {
  SmallVector<linalg::GenericOp> reductions;
  function.walk([&](linalg::GenericOp operation) {
    if (operation.getNumReductionLoops())
      reductions.push_back(operation);
  });
  for (linalg::GenericOp operation : reductions)
    normalize(operation);
  return success();
}

} // namespace intent::cpu
