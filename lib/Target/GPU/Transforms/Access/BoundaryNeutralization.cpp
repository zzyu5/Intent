#include "Intent/Target/GPU/Transforms/Passes.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace intent::gpu {
namespace {

struct PaddingBinding {
  Value value;
  SmallVector<int64_t> axes;
  SmallVector<int64_t> nodes;
  std::string fill;
};


class RefineBoundaryNeutralizationPass final
    : public PassWrapper<RefineBoundaryNeutralizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RefineBoundaryNeutralizationPass)

  StringRef getArgument() const final {
    return "intent-refine-gpu-boundary-neutralization";
  }
  StringRef getDescription() const final {
    return "Refine conservative GPU transfers using consumer neutralization proofs";
  }

  void runOnOperation() final {
    SmallVector<plan::ProgramOp> programs(
        getOperation().getOps<plan::ProgramOp>());
    if (programs.size() != 1) {
      getOperation().emitError(
          "boundary neutralization refinement requires one physical program");
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
  static LogicalResult refine(plan::ProgramOp program,
                              PhysicalProgramAnalysis &analysis) {
    SmallVector<PaddingBinding> paddings;
    target::KernelModel &kernel = analysis.getKernel();
    for (plan::PaddingOp padding :
         program.getBody().getOps<plan::PaddingOp>()) {
      Value value = kernel.values.lookup(padding.getValue());
      if (!value)
        return padding.emitOpError(
            "does not resolve a physical-program value for padding");
      paddings.push_back(PaddingBinding{
          value, SmallVector<int64_t>(padding.getTensorAxes()),
          SmallVector<int64_t>(padding.getDomainNodes()),
          padding.getFill().str()});
    }
    const target::KernelFacts &facts = analysis.getFacts();
    target::MaterializedPaddingQuery materializedPadding =
        [&](Value value, Operation *domain,
            Value ignoredMaterialization) -> std::optional<std::string> {
      if (!value || !domain || value == ignoredMaterialization)
        return std::nullopt;
      Operation *definition = value.getDefiningOp();
      if (definition &&
          ::intent::target::semanticOperationName(*definition) ==
              "intent.view_load")
        return std::nullopt;
      auto node = domain->getAttrOfType<IntegerAttr>("intent.node");
      auto axes = facts.valueAxes.find(value);
      if (!node || axes == facts.valueAxes.end())
        return std::nullopt;
      std::optional<unsigned> tensorAxis;
      for (auto [axisNumber, axis] : llvm::enumerate(axes->second)) {
        if (axis.domain != domain)
          continue;
        if (tensorAxis)
          return std::nullopt;
        tensorAxis = axisNumber;
      }
      if (!tensorAxis)
        return std::nullopt;
      for (const PaddingBinding &padding : paddings) {
        if (padding.value != value)
          continue;
        for (auto [axis, domainNode] :
             llvm::zip(padding.axes, padding.nodes))
          if (axis == static_cast<int64_t>(*tensorAxis) &&
              domainNode == node.getInt())
            return padding.fill;
      }
      return std::nullopt;
    };
    SmallVector<plan::PaddingOp> redundantPaddings;
    for (plan::PaddingOp padding :
         program.getBody().getOps<plan::PaddingOp>()) {
      Value value = kernel.values.lookup(padding.getValue());
      bool redundant = value && !padding.getDomainNodes().empty() &&
                       llvm::all_of(
                           padding.getDomainNodes(), [&](int64_t node) {
                             Operation *domain = kernel.nodes.lookup(node);
                             if (!domain)
                               return false;
                             std::optional<std::string> derived =
                                 target::inferDomainPadding(
                                     value, domain, facts,
                                     materializedPadding, value);
                             return derived && *derived == padding.getFill();
                           });
      if (redundant)
        redundantPaddings.push_back(padding);
    }
    for (plan::PaddingOp padding : redundantPaddings)
      padding.erase();
    OpBuilder builder(program.getContext());
    for (plan::TransferOp transfer :
         program.getBody().getOps<plan::TransferOp>()) {
      Operation *operation = kernel.nodes.lookup(transfer.getNode());
      bool neutralized =
          operation &&
          ::intent::target::semanticOperationName(*operation) ==
              "intent.view_load" &&
          target::proveBoundaryNeutralization(
              *operation, transfer.getFill(), facts, materializedPadding);
      transfer->setAttr("consumer_neutralized",
                        builder.getBoolAttr(neutralized));
    }
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createRefineBoundaryNeutralizationPass() {
  return std::make_unique<RefineBoundaryNeutralizationPass>();
}

} // namespace intent::gpu
