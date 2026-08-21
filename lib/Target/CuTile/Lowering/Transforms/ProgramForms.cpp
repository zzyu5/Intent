#include "Intent/Target/CuTile/Lowering/Passes.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Lowering/ProgramAnalysis.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace intent::cutile::lowering {
namespace {

FailureOr<std::pair<intent::plan::ProgramOp, intent::plan::SearchSpaceOp>>
getProgram(ModuleOp module) {
  SmallVector<intent::plan::ProgramOp> programs(
      module.getOps<intent::plan::ProgramOp>());
  SmallVector<intent::plan::SearchSpaceOp> searchSpaces(
      module.getOps<intent::plan::SearchSpaceOp>());
  if (programs.size() != 1 || searchSpaces.size() > 1) {
    module.emitError(
        "cuTile provider realization requires one physical program and at most one search space");
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

bool isRaggedBoundAxis(gpu::PhysicalProgramAnalysis &analysis, int64_t node) {
  Operation *domain = analysis.getKernel().nodes.lookup(node);
  if (!domain)
    return false;
  const target::KernelFacts &facts = analysis.getFacts();
  return facts.raggedMembers.count(domain) ||
         facts.raggedOuterRelations.count(domain);
}

bool translatedUnitStride(intent::plan::ProgramOp program,
                          intent::plan::TransferOp transfer) {
  SmallVector<intent::plan::RangeOp> ranges;
  for (intent::plan::RangeOp range :
       program.getBody().getOps<intent::plan::RangeOp>())
    if (range.getPurpose() == "access" && range.getTransferNodeAttr() &&
        range.getTransferNodeAttr().getInt() ==
            static_cast<int64_t>(transfer.getNode()))
      ranges.push_back(range);
  if (transfer.getTensorIndexing() != "structured" || ranges.size() != 1 ||
      ranges.front().getDivisorAttr() || !ranges.front().getOffsetAttr() ||
      ranges.front().getOffsetAttr().getInt() <= 0)
    return false;
  StringRef tile = ranges.front().getTile();
  int64_t width = 0;
  return tile.consume_front("fixed_") &&
         !tile.getAsInteger(10, width) && width > 0 &&
         ranges.front().getOffsetAttr().getInt() % width == 0;
}

bool hasIndexedRagged(const target::KernelFacts &facts) {
  return llvm::any_of(facts.raggedRelations, [](const auto &entry) {
    return static_cast<bool>(entry.second.indices);
  });
}

bool needsGuardedGatherTuning(gpu::PhysicalProgramAnalysis &analysis,
                              intent::plan::SearchSpaceOp searchSpace) {
  if (!searchSpace)
    return false;
  bool indexedRagged = hasIndexedRagged(analysis.getFacts());
  bool staged = !analysis.getStages().empty();
  bool orderedRagged = indexedRagged &&
                       !analysis.getFacts().orderedDomains.empty();
  bool guarded = false;
  analysis.getKernel().entry.walk([&](Operation *operation) {
    FailureOr<std::string> role = target::lowering::pointwiseRole(*operation);
    if (failed(role))
      return;
    guarded |= (staged && *role == "indirect_gather") ||
               (orderedRagged && *role == "members");
  });
  return guarded;
}

LogicalResult appendGatherTuning(intent::plan::SearchSpaceOp searchSpace,
                                 bool enabled) {
  if (!searchSpace)
    return enabled ? searchSpace.emitOpError(
                         "cannot declare cuTile gather spelling without a search space")
                   : success();
  auto declarations = searchSpace.getBody().getOps<intent::plan::AutotuneOp>();
  if (!llvm::hasSingleElement(declarations))
    return searchSpace.emitOpError(
        "cuTile provider realization requires one autotune declaration");
  intent::plan::AutotuneOp autotune = *declarations.begin();
  bool present = llvm::any_of(autotune.getParameters(), [](Attribute value) {
    auto role = dyn_cast<StringAttr>(value);
    return role && role.getValue() == gatherSpellingRole;
  });
  if (present)
    return autotune.emitOpError(
        "already contains a cuTile gather-spelling tuning decision");
  if (!enabled)
    return success();
  SmallVector<Attribute> parameters(autotune.getParameters().begin(),
                                    autotune.getParameters().end());
  parameters.push_back(StringAttr::get(searchSpace.getContext(),
                                       gatherSpellingRole));
  autotune->setAttr("parameters",
                    ArrayAttr::get(searchSpace.getContext(), parameters));
  return success();
}

LogicalResult realizeProgram(intent::plan::ProgramOp program,
                             intent::plan::SearchSpaceOp searchSpace) {
  FailureOr<std::unique_ptr<gpu::PhysicalProgramAnalysis>> analysis =
      gpu::PhysicalProgramAnalysis::compute(program);
  if (failed(analysis))
    return failure();
  OpBuilder builder(program.getContext());
  intent::plan::LaunchOp launch = (*analysis)->getLaunch();
  if (!launch)
    return program.emitOpError("has no launch decision for cuTile realization");
  if (launch->hasAttr(rowOccupancyAttr))
    return launch.emitOpError("already has a cuTile row-occupancy decision");

  intent::plan::AxisOp lane = firstLane(**analysis);
  intent::plan::RangeOp laneRange =
      lane ? (*analysis)->getRange(lane.getNode(), "lane")
           : intent::plan::RangeOp();
  bool tuneRows = !searchSpace && (*analysis)->getStages().empty() &&
                  !(*analysis)->hasWorkerReuse() && !launch.getPersistent() &&
                  laneRange && laneRange.getTile().starts_with("row_vector") &&
                  !target::lowering::hasNonReplayableEffect(
                      (*analysis)->getKernel().entry.getOperation());
  launch->setAttr(rowOccupancyAttr,
                  builder.getStringAttr(tuneRows ? "delegated" : "fixed"));

  for (intent::plan::TransferOp transfer :
       program.getBody().getOps<intent::plan::TransferOp>()) {
    if (transfer->hasAttr(accessAttr) || transfer->hasAttr(boundsAttr))
      return transfer.emitOpError(
          "already has a cuTile transfer-form decision");
    Operation *operation =
        (*analysis)->getKernel().nodes.lookup(transfer.getNode());
    StringRef name = operation ? operation->getName().getStringRef() : StringRef();
    bool load = name == "intent.view_load";
    bool store = name == "intent.view_store" || name == "intent.scatter_unique" ||
                 name == "intent.atomic_add" || name == "intent.atomic_cas";
    bool uniqueStore = name == "intent.scatter_unique";
    if (!load && !store)
      return transfer.emitOpError("does not bind a canonical transfer");
    FailureOr<bool> derivedScalar = target::hasDerivedScalarIndex(*operation);
    if (failed(derivedScalar))
      return failure();
    bool tensorIndexed = transfer.getTensorIndexing() != "none" &&
                         !translatedUnitStride(program, transfer);
    bool vectorized = llvm::any_of(
        transfer.getDomainNodes(), [&](int64_t node) {
          return !(*analysis)->isScalarAxis(node);
        });
    bool raggedBound = llvm::any_of(
        transfer.getDomainNodes(), [&](int64_t node) {
          return isRaggedBoundAxis(**analysis, node);
        });
    bool indirect = !uniqueStore &&
                    (tensorIndexed ||
                     (((*analysis)->hasWorkerReuse() || raggedBound) &&
                      vectorized));
    StringRef access = indirect ? (load ? "gather" : "scatter")
                                : (load ? "load" : "store");
    bool bounds = (*analysis)->hasWorkerReuse() || raggedBound ||
                  (!(*analysis)->getStages().empty() && store) ||
                  *derivedScalar;
    transfer->setAttr(accessAttr, builder.getStringAttr(access));
    transfer->setAttr(boundsAttr, builder.getBoolAttr(bounds));
  }
  return appendGatherTuning(
      searchSpace, needsGuardedGatherTuning(**analysis, searchSpace));
}

class RealizeProviderProgramPass final
    : public PassWrapper<RealizeProviderProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RealizeProviderProgramPass)
  StringRef getArgument() const final {
    return "intent-realize-cutile-program-forms";
  }
  StringRef getDescription() const final {
    return "Select cuTile-local access and tuning forms before terminal translation";
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
    return "intent-verify-cutile-program-forms";
  }
  StringRef getDescription() const final {
    return "Verify the complete cuTile provider-program contract";
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
  auto launches = program.getBody().getOps<intent::plan::LaunchOp>();
  if (!llvm::hasSingleElement(launches))
    return program.emitOpError("cuTile provider program requires one launch form");
  intent::plan::LaunchOp launch = *launches.begin();
  auto rowForm = launch->getAttrOfType<StringAttr>(rowOccupancyAttr);
  if (!rowForm || (rowForm.getValue() != "fixed" &&
                   rowForm.getValue() != "delegated"))
    return launch.emitOpError(
        "requires one realized cuTile row-occupancy form");
  for (intent::plan::TransferOp transfer :
       program.getBody().getOps<intent::plan::TransferOp>()) {
    auto access = transfer->getAttrOfType<StringAttr>(accessAttr);
    auto bounds = transfer->getAttrOfType<BoolAttr>(boundsAttr);
    Operation *operation = kernel.nodes.lookup(transfer.getNode());
    StringRef name = operation ? operation->getName().getStringRef() : StringRef();
    bool load = name == "intent.view_load";
    bool validAccess = access &&
                       ((load && (access.getValue() == "load" ||
                                  access.getValue() == "gather")) ||
                        (!load && (access.getValue() == "store" ||
                                   access.getValue() == "scatter")));
    if (!operation || !bounds || !validAccess)
      return transfer.emitOpError(
          "has no complete legal cuTile transfer-form decision");
  }
  if (searchSpace) {
    auto declarations = searchSpace.getBody().getOps<intent::plan::AutotuneOp>();
    if (!llvm::hasSingleElement(declarations))
      return searchSpace.emitOpError(
          "cuTile provider program requires one autotune declaration");
    intent::plan::AutotuneOp autotune = *declarations.begin();
    unsigned gatherParameters = llvm::count_if(
        autotune.getParameters(), [](Attribute value) {
          auto role = dyn_cast<StringAttr>(value);
          return role && role.getValue() == gatherSpellingRole;
        });
    if (gatherParameters > 1)
      return autotune.emitOpError(
          "contains duplicate cuTile gather-spelling parameters");
  }
  return success();
}

void addProviderPasses(PassManager &manager) {
  manager.addPass(std::make_unique<RealizeProviderProgramPass>());
  manager.addPass(std::make_unique<VerifyProviderProgramPass>());
}

} // namespace intent::cutile::lowering
