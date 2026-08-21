#include "Intent/Target/TileLang/Lowering/Passes.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Lowering/ProgramAnalysis.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace intent::tilelang::lowering {
namespace {

FailureOr<std::pair<intent::plan::ProgramOp, intent::plan::SearchSpaceOp>>
getProgram(ModuleOp module) {
  SmallVector<intent::plan::ProgramOp> programs(
      module.getOps<intent::plan::ProgramOp>());
  SmallVector<intent::plan::SearchSpaceOp> searchSpaces(
      module.getOps<intent::plan::SearchSpaceOp>());
  if (programs.size() != 1 || searchSpaces.size() > 1) {
    module.emitError(
        "TileLang provider realization requires one physical program and at most one search space");
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

bool feedsAtomicValue(Operation &operation) {
  if (operation.getNumResults() != 1 ||
      !llvm::hasSingleElement(operation.getResult(0).getUsers()))
    return false;
  Operation *user = *operation.getResult(0).user_begin();
  auto valueIndex =
      user->getAttrOfType<IntegerAttr>("intent.value_operand_index");
  return user->getName().getStringRef() == "intent.atomic_add" && valueIndex &&
         valueIndex.getInt() >= 0 &&
         static_cast<unsigned>(valueIndex.getInt()) < user->getNumOperands() &&
         user->getOperand(valueIndex.getInt()) == operation.getResult(0);
}

bool validityCoveredByPhysicalExtent(intent::plan::ProgramOp program,
                                     intent::plan::TransferOp transfer) {
  if (transfer.getValidityDomainNodes().empty())
    return false;
  llvm::StringSet<> physicalExtents;
  for (intent::plan::BlockExtentOp extent :
       program.getBody().getOps<intent::plan::BlockExtentOp>())
    physicalExtents.insert(extent.getLogicalExtent());
  return llvm::all_of(transfer.getValidityDomainNodes(), [&](int64_t node) {
    return llvm::any_of(program.getBody().getOps<intent::plan::RangeOp>(),
                        [&](intent::plan::RangeOp range) {
                          return static_cast<int64_t>(range.getAxisNode()) ==
                                     node &&
                                 range.getPurpose() != "access" &&
                                 range.getLevel() == 0 &&
                                 physicalExtents.contains(range.getExtent());
                        });
  });
}

bool isRankReducingReductionChain(
    Operation &operation, const llvm::DenseSet<int64_t> &reductionNodes) {
  if (operation.getNumOperands() == 0 || operation.getNumResults() != 1)
    return false;
  Operation *producer = operation.getOperand(0).getDefiningOp();
  auto producerNode =
      producer ? producer->getAttrOfType<IntegerAttr>("intent.node")
               : IntegerAttr();
  if (!producer || !producerNode ||
      !reductionNodes.contains(producerNode.getInt()) ||
      producer->getNumOperands() == 0)
    return false;
  auto source = dyn_cast<RankedTensorType>(producer->getOperand(0).getType());
  auto intermediate =
      dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  return source && intermediate && result && source.getRank() >= 4 &&
         intermediate.getRank() + 1 == source.getRank() &&
         result.getRank() + 1 == intermediate.getRank();
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
    return program.emitOpError("has no launch decision for TileLang realization");
  if (launch->hasAttr(rowLaunchAttr) ||
      launch->hasAttr(equalProgramTilesAttr))
    return launch.emitOpError("already has TileLang provider decisions");

  intent::plan::AxisOp lane = firstLane(**analysis);
  intent::plan::RangeOp laneRange =
      lane ? (*analysis)->getRange(lane.getNode(), "lane")
           : intent::plan::RangeOp();
  bool tuneRows = !searchSpace && (*analysis)->getStages().empty() &&
                  !(*analysis)->hasWorkerReuse() && laneRange &&
                  laneRange.getTile().starts_with("row_vector") &&
                  !target::lowering::hasNonReplayableEffect(
                      (*analysis)->getKernel().entry.getOperation());
  launch->setAttr(rowLaunchAttr,
                  builder.getStringAttr(tuneRows ? "delegated" : "fixed"));

  llvm::DenseSet<int64_t> reductionNodes;
  for (intent::plan::ReductionOp reduction :
       program.getBody().getOps<intent::plan::ReductionOp>())
    reductionNodes.insert(reduction.getNode());
  bool equalTiles = false;
  for (int64_t node : reductionNodes) {
    Operation *operation = (*analysis)->getKernel().nodes.lookup(node);
    equalTiles |= operation &&
                  isRankReducingReductionChain(*operation, reductionNodes);
  }
  launch->setAttr(equalProgramTilesAttr, builder.getBoolAttr(equalTiles));

  if (searchSpace) {
    auto declarations = searchSpace.getBody().getOps<intent::plan::AutotuneOp>();
    if (!llvm::hasSingleElement(declarations))
      return searchSpace.emitOpError(
          "TileLang provider realization requires one autotune declaration");
    intent::plan::AutotuneOp autotune = *declarations.begin();
    if (autotune->hasAttr(gemmWarpPolicyAttr))
      return autotune.emitOpError(
          "already has a TileLang GEMM warp-policy decision");
    bool tuneGemm = !program.getBody().getOps<intent::plan::ContractOp>().empty() ||
                    !program.getBody()
                         .getOps<intent::plan::SparseContractOp>()
                         .empty();
    autotune->setAttr(gemmWarpPolicyAttr, builder.getBoolAttr(tuneGemm));
  }

  for (intent::plan::ContractOp contract :
       program.getBody().getOps<intent::plan::ContractOp>()) {
    if (contract->hasAttr(isolateLhsAttr) || contract->hasAttr(isolateRhsAttr))
      return contract.emitOpError(
          "already has TileLang contraction-operand forms");
    Operation *operation =
        (*analysis)->getKernel().nodes.lookup(contract.getNode());
    if (!operation || operation->getNumOperands() != 2)
      return contract.emitOpError("does not bind canonical contraction operands");
    auto repeatedContractionOperand = [](Value operand) {
      return llvm::count_if(operand.getUsers(), [](Operation *user) {
               return user->getName().getStringRef() == "intent.contract";
             }) > 1;
    };
    contract->setAttr(isolateLhsAttr, builder.getBoolAttr(
                                           repeatedContractionOperand(
                                               operation->getOperand(0))));
    contract->setAttr(isolateRhsAttr, builder.getBoolAttr(
                                           repeatedContractionOperand(
                                               operation->getOperand(1))));
  }

  for (intent::plan::PointwiseOp pointwise :
       program.getBody().getOps<intent::plan::PointwiseOp>()) {
    if (pointwise->hasAttr(pointwiseFormAttr))
      return pointwise.emitOpError(
          "already has a TileLang pointwise-form decision");
    Operation *operation =
        (*analysis)->getKernel().nodes.lookup(pointwise.getNode());
    if (!operation)
      return pointwise.emitOpError("does not bind canonical pointwise semantics");
    pointwise->setAttr(
        pointwiseFormAttr,
        builder.getStringAttr(target::lowering::feedsContraction(*operation)
                                  ? "contract_operand"
                                  : "elementwise"));
  }

  for (intent::plan::TransferOp transfer :
       program.getBody().getOps<intent::plan::TransferOp>()) {
    if (transfer->hasAttr(accessAttr) || transfer->hasAttr(transferAttr) ||
        transfer->hasAttr(boundsAttr) || transfer->hasAttr(deferredAttr))
      return transfer.emitOpError(
          "already has a TileLang transfer-form decision");
    Operation *operation =
        (*analysis)->getKernel().nodes.lookup(transfer.getNode());
    StringRef name = operation ? operation->getName().getStringRef() : StringRef();
    bool load = name == "intent.view_load";
    bool store = name == "intent.view_store" || name == "intent.scatter_unique" ||
                 name == "intent.atomic_add" || name == "intent.atomic_cas";
    if (!load && !store)
      return transfer.emitOpError("does not bind a canonical transfer");
    FailureOr<bool> derivedScalar = target::hasDerivedScalarIndex(*operation);
    if (failed(derivedScalar))
      return failure();
    bool tensorIndirect = transfer.getTensorIndexing() == "data_dependent";
    bool raggedBound = llvm::any_of(
        transfer.getDomainNodes(), [&](int64_t node) {
          return isRaggedBoundAxis(**analysis, node);
        });
    bool plannedValidity = !transfer.getValidityDomainNodes().empty() &&
                           (store || transfer.getFill() != "none");
    bool logicalBounds =
        (raggedBound || plannedValidity) && !transfer.getConsumerNeutralized();
    bool physicalValidity =
        validityCoveredByPhysicalExtent(program, transfer);
    bool packedScalar = llvm::any_of(
        transfer.getDomainNodes(), [&](int64_t node) {
          return (*analysis)->isPackedScalarAxis(node);
        });
    bool elementwise = *derivedScalar || tensorIndirect || logicalBounds;
    transfer->setAttr(
        accessAttr,
        builder.getStringAttr((*analysis)->hasWorkerReuse()
                                  ? (load ? "gather" : "scatter")
                                  : (load ? "load" : "store")));
    transfer->setAttr(transferAttr,
                      builder.getStringAttr(elementwise ? "parallel_elements"
                                                        : "bulk_copy"));
    transfer->setAttr(
        boundsAttr,
        builder.getBoolAttr((*analysis)->hasWorkerReuse() || raggedBound ||
                            (logicalBounds && !physicalValidity) ||
                            (!(*analysis)->getStages().empty() && store) ||
                            *derivedScalar || tensorIndirect || packedScalar));
    transfer->setAttr(
        deferredAttr,
        builder.getBoolAttr(load && (*analysis)->getStages().empty() &&
                            feedsAtomicValue(*operation)));
  }
  return success();
}

class RealizeProviderProgramPass final
    : public PassWrapper<RealizeProviderProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RealizeProviderProgramPass)
  StringRef getArgument() const final {
    return "intent-realize-tilelang-program-forms";
  }
  StringRef getDescription() const final {
    return "Select TileLang-local access, value, and launch forms before terminal translation";
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
    return "intent-verify-tilelang-program-forms";
  }
  StringRef getDescription() const final {
    return "Verify the complete TileLang provider-program contract";
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
  (void)searchSpace;
  auto launches = program.getBody().getOps<intent::plan::LaunchOp>();
  if (!llvm::hasSingleElement(launches))
    return program.emitOpError(
        "TileLang provider program requires one launch form");
  intent::plan::LaunchOp launch = *launches.begin();
  auto row = launch->getAttrOfType<StringAttr>(rowLaunchAttr);
  auto equal = launch->getAttrOfType<BoolAttr>(equalProgramTilesAttr);
  if (!row || (row.getValue() != "fixed" && row.getValue() != "delegated") ||
      !equal)
    return launch.emitOpError(
        "has no complete TileLang launch-form decision");
  if (searchSpace) {
    auto declarations = searchSpace.getBody().getOps<intent::plan::AutotuneOp>();
    if (!llvm::hasSingleElement(declarations) ||
        !(*declarations.begin())->getAttrOfType<BoolAttr>(gemmWarpPolicyAttr))
      return searchSpace.emitOpError(
          "has no TileLang GEMM warp-policy tuning decision");
  }
  for (intent::plan::PointwiseOp pointwise :
       program.getBody().getOps<intent::plan::PointwiseOp>()) {
    auto form = pointwise->getAttrOfType<StringAttr>(pointwiseFormAttr);
    if (!form || (form.getValue() != "elementwise" &&
                  form.getValue() != "contract_operand") ||
        !kernel.nodes.lookup(pointwise.getNode()))
      return pointwise.emitOpError(
          "has no complete TileLang pointwise-form decision");
  }
  for (intent::plan::ContractOp contract :
       program.getBody().getOps<intent::plan::ContractOp>()) {
    auto lhs = contract->getAttrOfType<BoolAttr>(isolateLhsAttr);
    auto rhs = contract->getAttrOfType<BoolAttr>(isolateRhsAttr);
    Operation *operation = kernel.nodes.lookup(contract.getNode());
    if (!lhs || !rhs || !operation || operation->getNumOperands() != 2)
      return contract.emitOpError(
          "has no complete TileLang contraction-operand form");
  }
  for (intent::plan::TransferOp transfer :
       program.getBody().getOps<intent::plan::TransferOp>()) {
    auto access = transfer->getAttrOfType<StringAttr>(accessAttr);
    auto form = transfer->getAttrOfType<StringAttr>(transferAttr);
    auto bounds = transfer->getAttrOfType<BoolAttr>(boundsAttr);
    auto deferred = transfer->getAttrOfType<BoolAttr>(deferredAttr);
    Operation *operation = kernel.nodes.lookup(transfer.getNode());
    StringRef name = operation ? operation->getName().getStringRef() : StringRef();
    bool load = name == "intent.view_load";
    bool validAccess = access &&
                       ((load && (access.getValue() == "load" ||
                                  access.getValue() == "gather")) ||
                        (!load && (access.getValue() == "store" ||
                                   access.getValue() == "scatter")));
    bool validTransfer = form &&
                         (form.getValue() == "bulk_copy" ||
                          form.getValue() == "parallel_elements");
    if (!operation || !validAccess || !validTransfer || !bounds || !deferred)
      return transfer.emitOpError(
          "has no complete legal TileLang transfer-form decision");
  }
  return success();
}

void addProviderPasses(PassManager &manager) {
  manager.addPass(std::make_unique<RealizeProviderProgramPass>());
  manager.addPass(std::make_unique<VerifyProviderProgramPass>());
}

} // namespace intent::tilelang::lowering
