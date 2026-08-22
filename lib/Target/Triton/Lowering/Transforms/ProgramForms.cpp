#include "Intent/Target/Triton/Lowering/Passes.h"

#include "Syntax/Spelling.h"

#include "Intent/Target/Common/Lowering/ProgramAnalysis.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"
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

LogicalResult realizePointwiseLaneForm(
    gpu::PhysicalProgramAnalysis &analysis,
    intent::plan::SearchSpaceOp searchSpace) {
  if (!searchSpace)
    return success();
  intent::plan::AxisOp pointwiseAxis =
      analysis.getPurePointwiseProgramLane();
  if (!pointwiseAxis)
    return success();
  intent::plan::RangeOp ownership =
      analysis.getRange(pointwiseAxis.getNode(), "ownership");
  if (!ownership || !ownership.getTile().starts_with("program_"))
    return success();
  StringRef ownershipRole = ownership.getTile();
  std::string pointwiseRole = ownershipRole == "program_m"
                                  ? "pointwise_lane"
                              : ownershipRole == "program_n"
                                  ? "pointwise_lane_n"
                                  : "pointwise_lane_" +
                                        ownershipRole.drop_front(8).str();

  auto declarations = searchSpace.getBody().getOps<intent::plan::AutotuneOp>();
  if (!llvm::hasSingleElement(declarations))
    return searchSpace.emitOpError(
        "Triton pointwise-lane form requires one autotune declaration");
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

LogicalResult realizeProgram(intent::plan::ProgramOp program,
                             intent::plan::SearchSpaceOp searchSpace) {
  FailureOr<std::unique_ptr<gpu::PhysicalProgramAnalysis>> analysis =
      gpu::PhysicalProgramAnalysis::compute(program);
  if (failed(analysis))
    return failure();
  if (failed(realizePointwiseLaneForm(**analysis, searchSpace)))
    return failure();
  analysis = gpu::PhysicalProgramAnalysis::compute(program);
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
    StringRef name = ::intent::target::semanticOperationName(*operation);
    hasExternalRead |= name == "intent.view_load";
    hasScan |= name == "intent.scan";
    hasAggregation |= name == "intent.reduce" || name == "intent.arg_reduce" ||
                      name == "intent.scan" || name == "intent.contract" ||
                      name == "intent.sparse_contract" ||
                      name == "intent.state_stream";
  });
  bool configured =
      !searchSpace && !(*analysis)->hasWorkerReuse() && laneRange &&
      laneRange.getTile().starts_with("row_vector") &&
      isRuntimeABIDimension((*analysis)->getKernel(), laneRange.getExtent()) &&
      (hasExternalRead || !hasAggregation) && !hasScan &&
      !target::lowering::hasNonReplayableEffect(
          (*analysis)->getKernel().entry.getOperation());
  OpBuilder builder(program.getContext());
  launch->setAttr(rowLaunchAttr,
                  builder.getStringAttr(configured ? "configured" : "generic"));

  target::KernelModel &kernel = (*analysis)->getKernel();
  for (const auto &entry : kernel.raggedRelations) {
    Operation *ragged = entry.second.operation;
    if (!ragged || ragged->hasAttr(raggedRouteAttr))
      return program.emitOpError(
          "has an invalid or duplicate Triton ragged-route decision");
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
      return contract.emitOpError("already has a Triton contraction spelling");
    Operation *operation = kernel.nodes.lookup(contract.getNode());
    FailureOr<target::lowering::ContractionOrientation> orientation =
        operation ? target::lowering::contractionOrientation(*operation)
                  : FailureOr<target::lowering::ContractionOrientation>(failure());
    if (failed(orientation))
      return contract.emitOpError(
          "does not bind canonical contraction orientation");
    if (orientation->batched && contract.getForm() != "direct")
      return contract.emitOpError(
          "Triton cannot project the selected contraction form and orientation");
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
      return reduction.emitOpError("already has a Triton reduction spelling");
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
      return scan.emitOpError("already has a Triton scan spelling");
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
      return pointwise.emitOpError("already has a Triton pointwise spelling");
    Operation *operation = kernel.nodes.lookup(pointwise.getNode());
    FailureOr<std::string> role =
        operation ? target::lowering::pointwiseRole(*operation)
                  : FailureOr<std::string>(failure());
    FailureOr<StringRef> lowering =
        succeeded(role) ? syntax::pointwise(operation, *role)
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
  for (intent::plan::TransferOp transfer :
       program.getBody().getOps<intent::plan::TransferOp>()) {
    if (transfer->hasAttr(transferAccessAttr))
      return transfer.emitOpError("already has a Triton transfer spelling");
    Operation *operation = kernel.nodes.lookup(transfer.getNode());
    StringRef name = operation ? target::semanticOperationName(*operation)
                               : StringRef();
    bool load = name == "intent.view_load";
    bool store = name == "intent.view_store" ||
                 name == "intent.scatter_unique" ||
                 name == "intent.atomic_add" || name == "intent.atomic_cas";
    if (!load && !store)
      return transfer.emitOpError("does not bind canonical transfer semantics");
    transfer->setAttr(transferAccessAttr,
                      builder.getStringAttr(load ? "load" : "store"));
  }
  for (intent::plan::StreamBindingOp stream :
       program.getBody().getOps<intent::plan::StreamBindingOp>()) {
    if (stream->hasAttr(streamTileAttr))
      return stream.emitOpError("already has a Triton stream-tile spelling");
    intent::plan::RangeOp range = (*analysis)->getRange(
        stream.getAxisNode(), stream.getPurpose(), stream.getLevel());
    FailureOr<std::string> tile =
        range ? syntax::tile(stream.getOperation(), range.getTile())
              : FailureOr<std::string>(failure());
    if (failed(tile))
      return stream.emitOpError("does not bind one Triton stream tile");
    stream->setAttr(streamTileAttr, builder.getStringAttr(*tile));
  }

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
  for (const auto &entry : kernel.raggedRelations) {
    Operation *ragged = entry.second.operation;
    auto route = ragged ? ragged->getAttrOfType<StringAttr>(raggedRouteAttr)
                        : StringAttr();
    if (!route || (route.getValue() != "compact" &&
                   route.getValue() != "indexed"))
      return program.emitOpError(
          "has no complete Triton ragged-route form");
  }
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
    bool hasReuse = llvm::any_of(
        program.getBody().getOps<intent::plan::AxisOp>(),
        [](intent::plan::AxisOp axis) { return axis.getReuseWorker(); });
    if (searchSpace || hasReuse)
      return launch.emitOpError(
          "configured Triton row launch is incompatible with search or worker reuse");
  }
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
          "has no complete Triton contraction provider form");
  }
  for (intent::plan::ReductionOp reduction :
       program.getBody().getOps<intent::plan::ReductionOp>()) {
    auto lowering = reduction->getAttrOfType<StringAttr>(reductionLoweringAttr);
    auto axis = reduction->getAttrOfType<IntegerAttr>(reductionAxisAttr);
    if (!lowering || lowering.getValue().empty() || !axis || axis.getInt() < 0 ||
        !kernel.nodes.lookup(reduction.getNode()))
      return reduction.emitOpError("has no complete Triton reduction spelling");
  }
  for (intent::plan::ScanOp scan :
       program.getBody().getOps<intent::plan::ScanOp>()) {
    auto lowering = scan->getAttrOfType<StringAttr>(scanLoweringAttr);
    if (!lowering || lowering.getValue().empty() ||
        !kernel.nodes.lookup(scan.getNode()))
      return scan.emitOpError("has no complete Triton scan spelling");
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
      return pointwise.emitOpError("has no complete Triton pointwise spelling");
  }
  for (intent::plan::TransferOp transfer :
       program.getBody().getOps<intent::plan::TransferOp>()) {
    auto access = transfer->getAttrOfType<StringAttr>(transferAccessAttr);
    if (!access || (access.getValue() != "load" &&
                    access.getValue() != "store") ||
        !kernel.nodes.lookup(transfer.getNode()))
      return transfer.emitOpError("has no complete Triton transfer spelling");
  }
  for (intent::plan::StreamBindingOp stream :
       program.getBody().getOps<intent::plan::StreamBindingOp>()) {
    auto tile = stream->getAttrOfType<StringAttr>(streamTileAttr);
    if (!tile || tile.getValue().empty())
      return stream.emitOpError("has no complete Triton stream-tile spelling");
  }
  return success();
}

void addProviderPasses(PassManager &manager) {
  manager.addPass(std::make_unique<RealizeProviderProgramPass>());
  manager.addPass(std::make_unique<VerifyProviderProgramPass>());
}

} // namespace intent::triton::lowering
