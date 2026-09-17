#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool isSingletonInsertion(ReshapeOp reshape) {
  auto source = cast<FragmentType>(reshape.getValue().getType());
  auto target = cast<FragmentType>(reshape.getResult().getType());
  if (source.getShape().size() >= target.getShape().size())
    return false;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getResultAxes().empty() || group.getSourceAxes().size() > 1 ||
        (!group.getSourceAxes().empty() && group.getResultAxes().size() != 1))
      return false;
  }
  auto projection = queryBroadcastProjection(source, target);
  if (!projection.isExact())
    return false;
  unsigned nextSource = 0;
  for (auto [axis, mapped] : llvm::enumerate(projection.targetToSource)) {
    if (mapped) {
      if (*mapped != nextSource++ ||
          source.getShape()[*mapped] != target.getShape()[axis])
        return false;
      continue;
    }
    auto extent = cast<PhysicalExprAttr>(target.getShape()[axis]);
    if (extent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) ||
        extent.getValue() != 1)
      return false;
  }
  return nextSource == source.getShape().size();
}

void eliminateInBlock(Block &block) {
  llvm::DenseMap<OperationName, SmallVector<Operation *>> available;
  for (Operation &operation : llvm::make_early_inc_range(block)) {
    for (Region &region : operation.getRegions())
      for (Block &nested : region)
        eliminateInBlock(nested);
    // Parameter declarations also have symbolic type/attribute users. They are
    // retained and cleaned up by eraseUnusedPhysicalParameters, not SSA DCE.
    if (isa<ParameterOp, DelinearizeOp>(operation) ||
        operation.getNumRegions() != 0 ||
        operation.getNumResults() == 0 || !isMemoryEffectFree(&operation) ||
        !isPhysicalReplayNode(&operation, PhysicalReplayScope::ValueGraph,
                              /*allowAccesses=*/false))
      continue;
    auto &candidates = available[operation.getName()];
    Operation *equivalent = nullptr;
    for (Operation *candidate : candidates)
      if (OperationEquivalence::isEquivalentTo(
              candidate, &operation, OperationEquivalence::exactValueMatch,
              nullptr, OperationEquivalence::IgnoreLocations)) {
        equivalent = candidate;
        break;
      }
    if (!equivalent) {
      candidates.push_back(&operation);
      continue;
    }
    operation.replaceAllUsesWith(equivalent->getResults());
    operation.erase();
  }
}

} // namespace

LogicalResult eliminateCommonValues(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  // Preserve the explicit axis projection for provider analysis and CSE.
  kernel->walk([&](ReshapeOp reshape) {
    if (!isSingletonInsertion(reshape))
      return;
    OpBuilder builder(reshape);
    auto broadcast = builder.create<BroadcastOp>(
        reshape.getLoc(), reshape.getResult().getType(), reshape.getValue());
    broadcast->setDiscardableAttrs(llvm::to_vector(reshape->getDiscardableAttrs()));
    reshape.getResult().replaceAllUsesWith(broadcast.getResult());
    reshape.erase();
  });
  for (Block &block : kernel->getBody())
    eliminateInBlock(block);
  return success();
}

} // namespace intent::gpu
