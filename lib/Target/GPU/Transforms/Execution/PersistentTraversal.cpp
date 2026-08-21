#include "Intent/Target/GPU/Transforms/Passes.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace intent::gpu {
namespace {

class RefinePersistentTraversalPass final
    : public PassWrapper<RefinePersistentTraversalPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RefinePersistentTraversalPass)

  StringRef getArgument() const final {
    return "intent-refine-gpu-persistent-traversal";
  }
  StringRef getDescription() const final {
    return "Refine conservative GPU execution decisions with persistent traversal";
  }

  void runOnOperation() final {
    SmallVector<plan::ProgramOp> programs(
        getOperation().getOps<plan::ProgramOp>());
    if (programs.size() != 1) {
      getOperation().emitError(
          "persistent traversal refinement requires one physical program");
      signalPassFailure();
      return;
    }
    plan::ProgramOp program = programs.front();
    FailureOr<std::unique_ptr<PhysicalProgramAnalysis>> analysis =
        PhysicalProgramAnalysis::compute(program);
    if (failed(analysis) || failed(refine(program, **analysis)))
      signalPassFailure();
  }

private:
  static bool spansMultipleOwnershipTiles(
      plan::RangeOp range, Operation *domain,
      const target::KernelFacts &facts) {
    StringRef tile = range.getTile();
    if (tile == "one" || tile.starts_with("lane_pack"))
      return false;
    if (!tile.consume_front("fixed_"))
      return true;
    int64_t selected = 0;
    if (tile.getAsInteger(10, selected) || selected <= 0)
      return true;
    auto extent = facts.staticDomainExtents.find(domain);
    if (extent != facts.staticDomainExtents.end())
      return extent->second > selected;
    auto bounds = facts.staticDomainBounds.find(domain);
    if (bounds != facts.staticDomainBounds.end())
      return bounds->second.second - bounds->second.first > selected;
    return true;
  }

  static LogicalResult refine(plan::ProgramOp program,
                              PhysicalProgramAnalysis &analysis) {
    auto launches = program.getBody().getOps<plan::LaunchOp>();
    if (!llvm::hasSingleElement(launches))
      return program.emitOpError(
          "persistent traversal refinement requires one launch decision");
    plan::LaunchOp launch = *launches.begin();

    llvm::DenseMap<int64_t, plan::AxisOp> axes;
    for (plan::AxisOp axis : program.getBody().getOps<plan::AxisOp>())
      axes[axis.getNode()] = axis;
    llvm::DenseMap<int64_t, plan::RangeOp> ownership;
    for (plan::RangeOp range : program.getBody().getOps<plan::RangeOp>())
      if (range.getPurpose() == "ownership" && range.getLevel() == 0)
        ownership[range.getAxisNode()] = range;

    const target::KernelFacts &facts = analysis.getFacts();
    bool invalid = false;
    bool persistent = llvm::any_of(facts.contractions, [&](const auto &entry) {
      llvm::DenseSet<int64_t> owned;
      unsigned parallel = 0;
      unsigned multiTile = 0;
      bool ragged = false;
      for (Operation *parent = entry.first->getParentOp(); parent;
           parent = parent->getParentOp()) {
        if (::intent::target::semanticOperationName(*parent) != "intent.parallel" ||
            parent->getNumOperands() != 1)
          continue;
        auto domains = facts.parallelDomains.find(parent);
        if (domains == facts.parallelDomains.end())
          continue;
        for (Operation *domain : domains->second) {
          FailureOr<int64_t> node =
              target::getNodeID(*domain, "persistent traversal axis");
          if (failed(node)) {
            invalid = true;
            return false;
          }
          auto axis = axes.find(*node);
          if (axis == axes.end() || !axis->second.getProgramOrderAttr() ||
              !owned.insert(*node).second)
            continue;
          ragged |= facts.raggedMembers.count(domain);
          ++parallel;
          auto range = ownership.find(*node);
          multiTile += range != ownership.end() &&
                       spansMultipleOwnershipTiles(range->second, domain, facts);
        }
      }
      return !ragged && parallel >= 3 && multiTile >= 2;
    });
    if (invalid)
      return failure();
    if (!persistent)
      return success();

    OpBuilder builder(program.getContext());
    launch->setAttr("persistent", builder.getBoolAttr(true));
    SmallVector<plan::AxisOp> programAxes;
    for (auto &entry : axes)
      if (entry.second.getProgramOrderAttr())
        programAxes.push_back(entry.second);
    llvm::sort(programAxes, [](plan::AxisOp lhs, plan::AxisOp rhs) {
      return lhs.getProgramOrderAttr().getInt() <
             rhs.getProgramOrderAttr().getInt();
    });
    for (auto [fold, axis] : llvm::enumerate(programAxes)) {
      axis->setAttr("worker_axis", builder.getI64IntegerAttr(0));
      axis->setAttr("fold_order", builder.getI64IntegerAttr(fold));
      axis->setAttr("reuse_worker", builder.getBoolAttr(false));
    }
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createRefinePersistentTraversalPass() {
  return std::make_unique<RefinePersistentTraversalPass>();
}

} // namespace intent::gpu
