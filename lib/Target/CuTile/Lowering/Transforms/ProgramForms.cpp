#include "Intent/Target/CuTile/Lowering/Passes.h"

#include "Syntax/Spelling.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/Common/Lowering/ProgramAnalysis.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"
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

LogicalResult realizePointwiseLaneForm(
    gpu::PhysicalProgramAnalysis &analysis,
    intent::plan::SearchSpaceOp searchSpace) {
  if (!searchSpace)
    return success();
  intent::plan::AxisOp pointwiseAxis =
      analysis.getPurePointwiseProgramLane();
  if (!pointwiseAxis)
    return success();
  unsigned vectorProgramAxes = llvm::count_if(
      analysis.getProgram().getBody().getOps<intent::plan::AxisOp>(),
      [&](intent::plan::AxisOp axis) {
        return axis.getProgramOrderAttr() &&
               !analysis.isScalarAxis(axis.getNode());
      });
  if (vectorProgramAxes != 1)
    return success();

  intent::plan::RangeOp ownership =
      analysis.getRange(pointwiseAxis.getNode(), "ownership");
  if (!ownership || !ownership.getTile().starts_with("program_"))
    return success();
  StringRef pointwiseRole = "pointwise_lane";

  auto declarations = searchSpace.getBody().getOps<intent::plan::AutotuneOp>();
  if (!llvm::hasSingleElement(declarations))
    return searchSpace.emitOpError(
        "cuTile pointwise-lane form requires one autotune declaration");
  intent::plan::AutotuneOp autotune = *declarations.begin();
  OpBuilder builder(searchSpace.getContext());
  SmallVector<Attribute> parameters(autotune.getParameters().begin(),
                                    autotune.getParameters().end());
  unsigned replaced = 0;
  for (Attribute &parameter : parameters) {
    auto role = dyn_cast<StringAttr>(parameter);
    if (role && role.getValue() == ownership.getTile()) {
      parameter = builder.getStringAttr(pointwiseRole);
      ++replaced;
    }
  }
  if (replaced != 1)
    return autotune.emitOpError(
        "does not declare exactly one innermost pointwise-lane parameter");
  ownership->setAttr("tile", builder.getStringAttr(pointwiseRole));
  autotune->setAttr("parameters", builder.getArrayAttr(parameters));
  return success();
}

bool needsGuardedGatherTuning(gpu::PhysicalProgramAnalysis &analysis,
                              intent::plan::SearchSpaceOp searchSpace) {
  if (!searchSpace)
    return false;
  bool indexedRagged = hasIndexedRagged(analysis.getFacts());
  bool orderedRagged = indexedRagged &&
                       !analysis.getFacts().orderedDomains.empty();
  bool guarded = false;
  analysis.getKernel().entry.walk([&](Operation *operation) {
    StringRef name = ::intent::target::semanticOperationName(*operation);
    if (name != "intent.gather" && name != "intent.members")
      return;
    FailureOr<std::string> role = target::lowering::pointwiseRole(*operation);
    if (failed(role))
      return;
    guarded |= orderedRagged && *role == "members";
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
  if (failed(realizePointwiseLaneForm(**analysis, searchSpace)))
    return failure();
  analysis = gpu::PhysicalProgramAnalysis::compute(program);
  if (failed(analysis))
    return failure();
  intent::plan::LaunchOp launch = (*analysis)->getLaunch();
  if (!launch)
    return program.emitOpError("has no launch decision for cuTile realization");
  if (launch->hasAttr(rowOccupancyAttr))
    return launch.emitOpError("already has a cuTile row-occupancy decision");

  intent::plan::AxisOp lane = firstLane(**analysis);
  intent::plan::RangeOp laneRange =
      lane ? (*analysis)->getRange(lane.getNode(), "lane")
           : intent::plan::RangeOp();
  bool tuneRows = !searchSpace && !(*analysis)->hasWorkerReuse() &&
                  !launch.getPersistent() &&
                  laneRange && laneRange.getTile().starts_with("row_vector") &&
                  !target::lowering::hasNonReplayableEffect(
                      (*analysis)->getKernel().entry.getOperation());
  launch->setAttr(rowOccupancyAttr,
                  builder.getStringAttr(tuneRows ? "delegated" : "fixed"));

  target::KernelModel &kernel = (*analysis)->getKernel();
  for (const auto &entry : kernel.raggedRelations) {
    Operation *ragged = entry.second.operation;
    if (!ragged || ragged->hasAttr(raggedRouteAttr))
      return program.emitOpError(
          "has an invalid or duplicate cuTile ragged-route decision");
    FailureOr<StringRef> route =
        target::lowering::classifyRaggedProjection(*ragged);
    if (failed(route))
      return failure();
    ragged->setAttr(raggedRouteAttr, builder.getStringAttr(*route));
  }
  for (intent::plan::ContractOp contract :
       program.getBody().getOps<intent::plan::ContractOp>()) {
    if (contract->hasAttr(contractLoweringAttr) ||
        contract->hasAttr(contractOrientationAttr) ||
        contract->hasAttr(contractBatchedAttr) ||
        contract->hasAttr(scaledContractLayoutAttr))
      return contract.emitOpError("already has a cuTile contraction spelling");
    Operation *operation = kernel.nodes.lookup(contract.getNode());
    FailureOr<target::lowering::ContractionOrientation> orientation =
        operation ? target::lowering::contractionOrientation(*operation)
                  : FailureOr<target::lowering::ContractionOrientation>(failure());
    if (failed(orientation))
      return contract.emitOpError(
          "does not bind canonical contraction orientation");
    if (orientation->batched && contract.getForm() != "direct")
      return contract.emitOpError(
          "cuTile cannot project the selected contraction form and orientation");
    StringRef lowering = contract.getForm() == "scaled_direct"
                             ? syntax::scaledContraction()
                             : syntax::contraction();
    contract->setAttr(contractLoweringAttr, builder.getStringAttr(lowering));
    contract->setAttr(contractOrientationAttr,
                      builder.getStringAttr(
                          target::lowering::contractionOrientationName(
                              *orientation)));
    contract->setAttr(contractBatchedAttr,
                      builder.getBoolAttr(orientation->batched));
    if (contract.getForm() == "scaled_direct") {
      FailureOr<StringRef> layout =
          target::lowering::scaledContractionLayout(*operation);
      if (failed(layout))
        return failure();
      contract->setAttr(scaledContractLayoutAttr,
                        builder.getStringAttr(*layout));
    }
  }
  for (intent::plan::ReductionOp reduction :
       program.getBody().getOps<intent::plan::ReductionOp>()) {
    if (reduction->hasAttr(reductionLoweringAttr) ||
        reduction->hasAttr(reductionAxisAttr))
      return reduction.emitOpError("already has a cuTile reduction spelling");
    Operation *operation = kernel.nodes.lookup(reduction.getNode());
    FailureOr<std::string> role =
        operation ? target::lowering::reductionRole(*operation)
                  : FailureOr<std::string>(failure());
    FailureOr<int64_t> axis =
        operation ? target::lowering::reductionAxis(*operation)
                  : FailureOr<int64_t>(failure());
    if (failed(role) || failed(axis))
      return reduction.emitOpError("does not bind canonical reduction semantics");
    reduction->setAttr(reductionLoweringAttr,
                       builder.getStringAttr(syntax::reduction(*role)));
    reduction->setAttr(reductionAxisAttr, builder.getI64IntegerAttr(*axis));
  }
  for (intent::plan::ScanOp scan :
       program.getBody().getOps<intent::plan::ScanOp>()) {
    if (scan->hasAttr(scanLoweringAttr))
      return scan.emitOpError("already has a cuTile scan spelling");
    Operation *operation = kernel.nodes.lookup(scan.getNode());
    FailureOr<std::string> role =
        operation ? target::lowering::scanRole(*operation)
                  : FailureOr<std::string>(failure());
    if (failed(role))
      return scan.emitOpError("does not bind canonical scan semantics");
    scan->setAttr(scanLoweringAttr,
                  builder.getStringAttr(syntax::scan(*role)));
  }
  for (intent::plan::PointwiseOp pointwise :
       program.getBody().getOps<intent::plan::PointwiseOp>()) {
    if (pointwise->hasAttr(pointwiseLoweringAttr))
      return pointwise.emitOpError("already has a cuTile pointwise spelling");
    Operation *operation = kernel.nodes.lookup(pointwise.getNode());
    FailureOr<std::string> role =
        operation ? target::lowering::pointwiseRole(*operation)
                  : FailureOr<std::string>(failure());
    FailureOr<StringRef> lowering =
        succeeded(role)
            ? syntax::pointwise(pointwise.getOperation(), *role,
                                pointwise.getResultSpace())
            : FailureOr<StringRef>(failure());
    if (failed(lowering))
      return pointwise.emitOpError("does not bind canonical pointwise semantics");
    pointwise->setAttr(pointwiseLoweringAttr,
                       builder.getStringAttr(*lowering));
    if (target::semanticOperationName(*operation) == "intent.gather") {
      FailureOr<std::string> gather =
          target::lowering::classifyGatherProjection(*operation);
      if (failed(gather))
        return failure();
      pointwise->setAttr(gatherFormAttr, builder.getStringAttr(*gather));
    }
  }
  for (intent::plan::StreamBindingOp stream :
       program.getBody().getOps<intent::plan::StreamBindingOp>()) {
    if (stream->hasAttr(streamTileAttr))
      return stream.emitOpError("already has a cuTile stream-tile spelling");
    intent::plan::RangeOp range = (*analysis)->getRange(
        stream.getAxisNode(), stream.getPurpose(), stream.getLevel());
    FailureOr<std::string> tile =
        range ? syntax::tile(stream.getOperation(), range.getTile())
              : FailureOr<std::string>(failure());
    if (failed(tile))
      return stream.emitOpError("does not bind one cuTile stream tile");
    stream->setAttr(streamTileAttr, builder.getStringAttr(*tile));
  }

  llvm::DenseSet<int64_t> partitionedStreams;
  for (intent::plan::StreamBindingOp stream :
       program.getBody().getOps<intent::plan::StreamBindingOp>())
    if (stream.getPartitionNodeAttr())
      partitionedStreams.insert(stream.getStreamNode());

  for (intent::plan::TransferOp transfer :
       program.getBody().getOps<intent::plan::TransferOp>()) {
    if (transfer->hasAttr(accessAttr) || transfer->hasAttr(boundsAttr))
      return transfer.emitOpError(
          "already has a cuTile transfer-form decision");
    Operation *operation =
        (*analysis)->getKernel().nodes.lookup(transfer.getNode());
    StringRef name = operation ? ::intent::target::semanticOperationName(*operation) : StringRef();
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
    bool partitionStream = false;
    for (Operation *parent = operation->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (::intent::target::semanticOperationName(*parent) !=
          "intent.state_stream")
        continue;
      auto node = parent->getAttrOfType<IntegerAttr>("intent.node");
      partitionStream = node && partitionedStreams.contains(node.getInt());
      break;
    }
    bool indirect = !uniqueStore &&
                    (tensorIndexed ||
                     (((*analysis)->hasWorkerReuse() || raggedBound ||
                       partitionStream) &&
                      vectorized));
    StringRef access = indirect ? (load ? "gather" : "scatter")
                                : (load ? "load" : "store");
    bool bounds = (*analysis)->hasWorkerReuse() || raggedBound ||
                  partitionStream || *derivedScalar;
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
  for (const auto &entry : kernel.raggedRelations) {
    Operation *ragged = entry.second.operation;
    auto route = ragged ? ragged->getAttrOfType<StringAttr>(raggedRouteAttr)
                        : StringAttr();
    if (!route || (route.getValue() != "compact" &&
                   route.getValue() != "indexed"))
      return program.emitOpError(
          "has no complete cuTile ragged-route form");
  }
  auto launches = program.getBody().getOps<intent::plan::LaunchOp>();
  if (!llvm::hasSingleElement(launches))
    return program.emitOpError("cuTile provider program requires one launch form");
  intent::plan::LaunchOp launch = *launches.begin();
  auto rowForm = launch->getAttrOfType<StringAttr>(rowOccupancyAttr);
  if (!rowForm || (rowForm.getValue() != "fixed" &&
                   rowForm.getValue() != "delegated"))
    return launch.emitOpError(
        "requires one realized cuTile row-occupancy form");
  for (intent::plan::ContractOp contract :
       program.getBody().getOps<intent::plan::ContractOp>()) {
    auto lowering = contract->getAttrOfType<StringAttr>(contractLoweringAttr);
    auto orientation =
        contract->getAttrOfType<StringAttr>(contractOrientationAttr);
    auto batched = contract->getAttrOfType<BoolAttr>(contractBatchedAttr);
    auto layout = contract->getAttrOfType<StringAttr>(scaledContractLayoutAttr);
    if (!lowering || lowering.getValue().empty() || !orientation || !batched ||
        (orientation.getValue() != "nn" && orientation.getValue() != "nt" &&
         orientation.getValue() != "tn" && orientation.getValue() != "tt") ||
        (contract.getForm() == "scaled_direct" &&
         (!layout || (layout.getValue() != "grouped_rank_two" &&
                      layout.getValue() != "flattened_rank_three"))) ||
        (contract.getForm() != "scaled_direct" && layout))
      return contract.emitOpError(
          "has no complete cuTile contraction provider form");
  }
  for (intent::plan::ReductionOp reduction :
       program.getBody().getOps<intent::plan::ReductionOp>()) {
    auto lowering = reduction->getAttrOfType<StringAttr>(reductionLoweringAttr);
    auto axis = reduction->getAttrOfType<IntegerAttr>(reductionAxisAttr);
    if (!lowering || lowering.getValue().empty() || !axis || axis.getInt() < 0 ||
        !kernel.nodes.lookup(reduction.getNode()))
      return reduction.emitOpError("has no complete cuTile reduction spelling");
  }
  for (intent::plan::ScanOp scan :
       program.getBody().getOps<intent::plan::ScanOp>()) {
    auto lowering = scan->getAttrOfType<StringAttr>(scanLoweringAttr);
    if (!lowering || lowering.getValue().empty() ||
        !kernel.nodes.lookup(scan.getNode()))
      return scan.emitOpError("has no complete cuTile scan spelling");
  }
  for (intent::plan::PointwiseOp pointwise :
       program.getBody().getOps<intent::plan::PointwiseOp>()) {
    auto lowering = pointwise->getAttrOfType<StringAttr>(pointwiseLoweringAttr);
    Operation *operation = kernel.nodes.lookup(pointwise.getNode());
    auto gather = pointwise->getAttrOfType<StringAttr>(gatherFormAttr);
    bool requiresGather =
        operation && target::semanticOperationName(*operation) == "intent.gather";
    if (!lowering || lowering.getValue().empty() || !operation ||
        (requiresGather && (!gather || gather.getValue().empty())))
      return pointwise.emitOpError("has no complete cuTile pointwise spelling");
  }
  for (intent::plan::StreamBindingOp stream :
       program.getBody().getOps<intent::plan::StreamBindingOp>()) {
    auto tile = stream->getAttrOfType<StringAttr>(streamTileAttr);
    if (!tile || tile.getValue().empty())
      return stream.emitOpError("has no complete cuTile stream-tile spelling");
  }
  for (intent::plan::TransferOp transfer :
       program.getBody().getOps<intent::plan::TransferOp>()) {
    auto access = transfer->getAttrOfType<StringAttr>(accessAttr);
    auto bounds = transfer->getAttrOfType<BoolAttr>(boundsAttr);
    Operation *operation = kernel.nodes.lookup(transfer.getNode());
    StringRef name = operation ? ::intent::target::semanticOperationName(*operation) : StringRef();
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
