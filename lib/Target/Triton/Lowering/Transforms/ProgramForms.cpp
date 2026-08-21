#include "Intent/Target/Triton/Lowering/Passes.h"

#include "Intent/Target/Common/Lowering/ProgramAnalysis.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace intent::triton::lowering {
namespace {

FailureOr<std::pair<intent::plan::ProgramOp, intent::plan::SearchSpaceOp>>
getProgram(ModuleOp module) {
  SmallVector<intent::plan::ProgramOp> programs(
      module.getOps<intent::plan::ProgramOp>());
  SmallVector<intent::plan::SearchSpaceOp> searchSpaces(
      module.getOps<intent::plan::SearchSpaceOp>());
  if (programs.size() != 1 || searchSpaces.size() > 1) {
    module.emitError(
        "Triton provider realization requires one physical program and at most one search space");
    return failure();
  }
  return std::make_pair(programs.front(),
                        searchSpaces.empty() ? intent::plan::SearchSpaceOp()
                                             : searchSpaces.front());
}

intent::plan::AxisOp firstLane(gpu::PhysicalProgramAnalysis &analysis) {
  SmallVector<intent::plan::AxisOp> lanes;
  for (intent::plan::AxisOp axis :
       analysis.getProgram().getBody().getOps<intent::plan::AxisOp>())
    if (analysis.axisHasRole(axis.getNode(), "lane"))
      lanes.push_back(axis);
  llvm::sort(lanes, [](intent::plan::AxisOp lhs, intent::plan::AxisOp rhs) {
    return lhs.getNode() < rhs.getNode();
  });
  return lanes.empty() ? intent::plan::AxisOp() : lanes.front();
}

bool isRuntimeABIDimension(const target::KernelModel &kernel,
                           StringRef dimension) {
  for (const target::ABIArgument &argument : kernel.abi.arguments) {
    auto shape = argument.metadata.getAs<ArrayAttr>("shape");
    if (!shape)
      continue;
    for (Attribute extent : shape) {
      auto symbol = dyn_cast<StringAttr>(extent);
      uint64_t constant = 0;
      if (symbol && symbol.getValue() == dimension &&
          symbol.getValue().getAsInteger(10, constant))
        return true;
    }
  }
  return false;
}

LogicalResult realizeProgram(intent::plan::ProgramOp program,
                             intent::plan::SearchSpaceOp searchSpace) {
  FailureOr<std::unique_ptr<gpu::PhysicalProgramAnalysis>> analysis =
      gpu::PhysicalProgramAnalysis::compute(program);
  if (failed(analysis))
    return failure();
  intent::plan::LaunchOp launch = (*analysis)->getLaunch();
  if (!launch)
    return program.emitOpError("has no launch decision for Triton realization");
  if (launch->hasAttr(rowLaunchAttr))
    return launch.emitOpError("already has a Triton row-launch decision");

  intent::plan::AxisOp lane = firstLane(**analysis);
  intent::plan::RangeOp laneRange =
      lane ? (*analysis)->getRange(lane.getNode(), "lane")
           : intent::plan::RangeOp();
  bool hasExternalRead = false;
  bool hasAggregation = false;
  bool hasScan = false;
  (*analysis)->getKernel().entry.walk([&](Operation *operation) {
    StringRef name = operation->getName().getStringRef();
    hasExternalRead |= name == "intent.view_load";
    hasScan |= name == "intent.scan";
    hasAggregation |= name == "intent.reduce" || name == "intent.arg_reduce" ||
                      name == "intent.scan" || name == "intent.contract" ||
                      name == "intent.sparse_contract" ||
                      name == "intent.state_stream";
  });
  bool configured =
      !searchSpace && (*analysis)->getStages().empty() &&
      !(*analysis)->hasWorkerReuse() && laneRange &&
      laneRange.getTile().starts_with("row_vector") &&
      isRuntimeABIDimension((*analysis)->getKernel(), laneRange.getExtent()) &&
      (hasExternalRead || !hasAggregation) && !hasScan &&
      !target::lowering::hasNonReplayableEffect(
          (*analysis)->getKernel().entry.getOperation());
  OpBuilder builder(program.getContext());
  launch->setAttr(rowLaunchAttr,
                  builder.getStringAttr(configured ? "configured" : "generic"));
  return success();
}

class RealizeProviderProgramPass final
    : public PassWrapper<RealizeProviderProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RealizeProviderProgramPass)

  StringRef getArgument() const final {
    return "intent-realize-triton-program-forms";
  }
  StringRef getDescription() const final {
    return "Select Triton-local program forms before terminal source translation";
  }
  void runOnOperation() final {
    auto selected = getProgram(getOperation());
    if (failed(selected) ||
        failed(realizeProgram(selected->first, selected->second)))
      signalPassFailure();
  }
};

class VerifyProviderProgramPass final
    : public PassWrapper<VerifyProviderProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyProviderProgramPass)

  StringRef getArgument() const final {
    return "intent-verify-triton-program-forms";
  }
  StringRef getDescription() const final {
    return "Verify the complete Triton provider-program contract";
  }
  void runOnOperation() final {
    auto selected = getProgram(getOperation());
    FailureOr<func::FuncOp> entry =
        succeeded(selected) ? intent::plan::getPhysicalEntry(selected->first)
                            : FailureOr<func::FuncOp>(failure());
    FailureOr<target::KernelModel> kernel =
        succeeded(entry) ? target::analyzeKernel(*entry)
                         : FailureOr<target::KernelModel>(failure());
    if (failed(selected) || failed(entry) || failed(kernel) ||
        failed(verifyProviderProgram(*kernel, selected->first,
                                     selected->second)))
      signalPassFailure();
  }
};

} // namespace

LogicalResult verifyProviderProgram(const target::KernelModel &kernel,
                                    intent::plan::ProgramOp program,
                                    intent::plan::SearchSpaceOp searchSpace) {
  (void)kernel;
  auto launches = program.getBody().getOps<intent::plan::LaunchOp>();
  if (!llvm::hasSingleElement(launches))
    return program.emitOpError(
        "Triton provider program requires one launch form");
  intent::plan::LaunchOp launch = *launches.begin();
  auto form = launch->getAttrOfType<StringAttr>(rowLaunchAttr);
  if (!form || (form.getValue() != "generic" &&
                form.getValue() != "configured"))
    return launch.emitOpError(
        "requires one realized Triton row-launch form");
  if (form.getValue() == "configured") {
    bool hasStages = !program.getBody().getOps<intent::plan::StageOp>().empty();
    bool hasReuse = llvm::any_of(
        program.getBody().getOps<intent::plan::AxisOp>(),
        [](intent::plan::AxisOp axis) { return axis.getReuseWorker(); });
    if (searchSpace || hasStages || hasReuse)
      return launch.emitOpError(
          "configured Triton row launch is incompatible with search, stages, or worker reuse");
  }
  return success();
}

void addProviderPasses(PassManager &manager) {
  manager.addPass(std::make_unique<RealizeProviderProgramPass>());
  manager.addPass(std::make_unique<VerifyProviderProgramPass>());
}

} // namespace intent::triton::lowering
