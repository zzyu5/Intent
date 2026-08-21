#include "Intent/Target/GPU/Transforms/Passes.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/ContractReplay.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Pass/Pass.h"

#include <functional>
#include <optional>

using namespace mlir;

namespace intent::gpu {
namespace {

struct AccumulatorFlow {
  Operation *owner = nullptr;
  Operation *update = nullptr;
  Operation *conditional = nullptr;
  unsigned conditionalResult = 0;
  Value previous;
};

FailureOr<std::pair<StringRef, StringRef>>
stagedOperandForms(Operation &operation) {
  if (operation.getNumOperands() != 2)
    return operation.emitOpError(
        "staged contraction requires two physical operands");
  StringRef lhsForm = "workspace";
  if (Operation *lhs = operation.getOperand(0).getDefiningOp();
      lhs && target::semanticOperationName(*lhs) == "intent.gather") {
    FailureOr<SmallVector<target::IndexTerm>> relation =
        target::parseIndexRelation(*lhs);
    if (failed(relation) || relation->size() != 2 ||
        (*relation)[0].kind != "value_index" ||
        (*relation)[0].operands.size() != 1 ||
        !(*relation)[0].operands.front() ||
        (*relation)[1].kind != "full_slice")
      return lhs->emitOpError(
          "has no canonical member-row gather for staged contraction");
    lhsForm = "member_row_gather";
  }
  Operation *rhs = operation.getOperand(1).getDefiningOp();
  auto rhsView = rhs && rhs->getNumOperands() > 0
                     ? dyn_cast<intent::ViewType>(rhs->getOperand(0).getType())
                     : intent::ViewType();
  auto rhsTensor = rhsView
                       ? dyn_cast<RankedTensorType>(rhsView.getTensor())
                       : RankedTensorType();
  if (!rhs || target::semanticOperationName(*rhs) != "intent.view_load" ||
      !rhsTensor || rhsTensor.getRank() != 3)
    return operation.emitOpError(
        "staged contraction requires an expert-selected rank-three weight");
  return std::pair<StringRef, StringRef>{lhsForm, "expert_matrix"};
}

std::optional<AccumulatorFlow> findAccumulatorFlow(Operation &operation) {
  if (target::semanticOperationName(operation) != "intent.contract" ||
      operation.getNumResults() != 1 ||
      !llvm::hasSingleElement(operation.getResult(0).getUsers()))
    return std::nullopt;
  Operation *update = *operation.getResult(0).user_begin();
  auto logical = update->getAttrOfType<StringAttr>("intent.operator");
  if (target::semanticOperationName(*update) != "intent.binary" || !logical ||
      logical.getValue() != "add" || update->getNumOperands() != 2 ||
      update->getNumResults() != 1 ||
      !llvm::hasSingleElement(update->getResult(0).getUsers()) ||
      update->getOperand(0) == update->getOperand(1))
    return std::nullopt;

  Operation *updateYield = *update->getResult(0).user_begin();
  if (target::semanticOperationName(*updateYield) != "intent.yield")
    return std::nullopt;
  Operation *conditional = nullptr;
  Operation *yield = updateYield;
  unsigned conditionalResult = 0;
  Operation *updateOwner = updateYield->getParentOp();
  if (updateOwner && target::semanticOperationName(*updateOwner) == "intent.if") {
    conditional = updateOwner;
    std::optional<unsigned> result;
    for (auto [index, operand] : llvm::enumerate(updateYield->getOperands())) {
      if (operand != update->getResult(0))
        continue;
      if (result)
        return std::nullopt;
      result = index;
    }
    if (!result || *result >= conditional->getNumResults() ||
        !llvm::hasSingleElement(conditional->getResult(*result).getUsers()))
      return std::nullopt;
    conditionalResult = *result;
    yield = *conditional->getResult(*result).user_begin();
    if (target::semanticOperationName(*yield) != "intent.yield")
      return std::nullopt;
  }

  Operation *owner = yield->getParentOp();
  StringRef ownerName = owner ? target::semanticOperationName(*owner) : StringRef();
  if (!owner || (ownerName != "intent.for" && ownerName != "intent.state_stream") ||
      owner->getNumRegions() != 1 || owner->getRegion(0).empty())
    return std::nullopt;
  std::optional<unsigned> carried;
  Value expected = conditional ? conditional->getResult(conditionalResult)
                               : update->getResult(0);
  for (auto [index, operand] : llvm::enumerate(yield->getOperands())) {
    if (operand != expected)
      continue;
    if (carried)
      return std::nullopt;
    carried = index;
  }
  Block &body = owner->getRegion(0).front();
  if (!carried || *carried >= owner->getNumResults() ||
      body.getNumArguments() < owner->getNumResults())
    return std::nullopt;
  unsigned domainCount = body.getNumArguments() - owner->getNumResults();
  Value previous = body.getArgument(domainCount + *carried);
  bool consumesContract = false;
  bool consumesPrevious = false;
  for (Value operand : update->getOperands()) {
    consumesContract |= operand == operation.getResult(0);
    consumesPrevious |= operand == previous;
  }
  if (!consumesContract || !consumesPrevious)
    return std::nullopt;

  if (conditional) {
    Region *updateRegion = updateYield->getParentRegion();
    if (updateRegion != &conditional->getRegion(0) &&
        updateRegion != &conditional->getRegion(1))
      return std::nullopt;
    unsigned updateRegionIndex =
        updateRegion == &conditional->getRegion(0) ? 0 : 1;
    Operation &otherYield =
        conditional->getRegion(1 - updateRegionIndex).front().back();
    if (target::semanticOperationName(otherYield) != "intent.yield" ||
        otherYield.getNumOperands() != conditional->getNumResults() ||
        otherYield.getOperand(conditionalResult) != previous)
      return std::nullopt;
  }
  return AccumulatorFlow{owner, update, conditional, conditionalResult, previous};
}

bool isDirectViewLoad(Value value) {
  Operation *definition = value.getDefiningOp();
  return definition && target::semanticOperationName(*definition) ==
                           "intent.view_load";
}

bool isContractionDerived(Value value) {
  llvm::DenseSet<Value> visited;
  std::function<bool(Value)> visit = [&](Value current) {
    if (!visited.insert(current).second)
      return false;
    Operation *definition = current.getDefiningOp();
    if (!definition)
      return false;
    StringRef name = target::semanticOperationName(*definition);
    if (name == "intent.contract" || name == "intent.scaled_contract")
      return true;
    return llvm::any_of(definition->getOperands(), [&](Value operand) {
      return isa<RankedTensorType>(operand.getType()) && visit(operand);
    });
  };
  return visit(value);
}

Operation *enclosingStream(Operation &operation) {
  for (Operation *parent = operation.getParentOp(); parent;
       parent = parent->getParentOp())
    if (target::semanticOperationName(*parent) == "intent.state_stream")
      return parent;
  return nullptr;
}

Operation *reductionDomain(Operation &operation,
                           const target::KernelFacts &facts) {
  auto found = facts.contractions.find(&operation);
  if (found == facts.contractions.end() ||
      found->second.lhsReductionAxes.size() != 1 ||
      found->second.rhsReductionAxes.size() != 1)
    return nullptr;
  unsigned lhs = found->second.lhsReductionAxes.front();
  unsigned rhs = found->second.rhsReductionAxes.front();
  if (lhs >= found->second.lhsAxes.size() ||
      rhs >= found->second.rhsAxes.size())
    return nullptr;
  Operation *lhsDomain = found->second.lhsAxes[lhs].domain;
  return lhsDomain && lhsDomain == found->second.rhsAxes[rhs].domain
             ? lhsDomain
             : nullptr;
}

bool isStreamedReduction(Operation &operation,
                         const target::KernelFacts &facts) {
  Operation *domain = reductionDomain(operation, facts);
  Operation *stream = enclosingStream(operation);
  if (!domain || !stream || stream->getNumOperands() == 0)
    return false;
  FailureOr<SmallVector<Operation *>> domains =
      target::expandDomainSource(stream->getOperand(0), *stream);
  return succeeded(domains) && llvm::is_contained(*domains, domain);
}

bool hasPhysicalReductionAxis(Operation &operation,
                              PhysicalProgramAnalysis &analysis) {
  Operation *domain = reductionDomain(operation, analysis.getFacts());
  if (!domain)
    return false;
  FailureOr<int64_t> node =
      target::getNodeID(*domain, "contraction reduction realization");
  return succeeded(node) && analysis.axisHasRole(*node, "reduction") &&
         static_cast<bool>(analysis.getRange(*node, "reduction"));
}

FailureOr<int64_t> physicalReductionAxis(Operation &operation,
                                         PhysicalProgramAnalysis &analysis) {
  Operation *domain = reductionDomain(operation, analysis.getFacts());
  FailureOr<int64_t> node =
      domain ? target::getNodeID(*domain, "contraction physical reduction axis")
             : FailureOr<int64_t>(failure());
  if (failed(node) || !analysis.axisHasRole(*node, "reduction") ||
      !analysis.getRange(*node, "reduction"))
    return operation.emitOpError(
        "does not resolve one selected physical reduction axis");
  return *node;
}

struct ReplayChoice {
  target::ContractOperandReplay lhs;
  target::ContractOperandReplay rhs;
};

LogicalResult refineContractions(plan::ProgramOp program) {
  FailureOr<std::unique_ptr<PhysicalProgramAnalysis>> analysis =
      PhysicalProgramAnalysis::compute(program);
  if (failed(analysis))
    return failure();
  target::KernelModel &kernel = (*analysis)->getKernel();
  target::KernelFacts &facts = (*analysis)->getFacts();
  llvm::DenseMap<int64_t, plan::ContractOp> contracts;
  llvm::DenseMap<int64_t, plan::TransferOp> transfers;
  llvm::DenseSet<int64_t> staged;
  for (Operation &operation : program.getBody().front()) {
    if (auto binding = dyn_cast<plan::ContractOp>(operation))
      contracts[binding.getNode()] = binding;
    else if (auto binding = dyn_cast<plan::TransferOp>(operation))
      transfers[binding.getNode()] = binding;
    else if (auto binding = dyn_cast<plan::StageOp>(operation))
      staged.insert(binding.getNode());
  }
  auto isStaged = [&](Operation &operation) {
    auto node = operation.getAttrOfType<IntegerAttr>("intent.node");
    return node && staged.contains(node.getInt());
  };

  llvm::DenseMap<Operation *, ReplayChoice> replays;
  llvm::DenseMap<Operation *, Operation *> replayOwners;
  WalkResult replayWalk = kernel.entry.walk([&](Operation *operation) {
    if (target::semanticOperationName(*operation) != "intent.contract" ||
        operation->getNumOperands() != 2 || isStaged(*operation))
      return WalkResult::advance();
    std::optional<target::ContractOperandReplay> lhs =
        target::analyzeContractOperandReplay(operation->getOperand(0), *operation);
    std::optional<target::ContractOperandReplay> rhs =
        target::analyzeContractOperandReplay(operation->getOperand(1), *operation);
    bool compact = lhs && rhs &&
                   llvm::any_of(llvm::concat<Operation *>(lhs->transfers,
                                                         rhs->transfers),
                                [&](Operation *transfer) {
                                  auto node = transfer->getAttrOfType<IntegerAttr>(
                                      "intent.node");
                                  plan::TransferOp binding =
                                      node ? transfers.lookup(node.getInt())
                                           : plan::TransferOp();
                                  return binding &&
                                         binding.getTensorIndexing() == "compact";
                                });
    if (!lhs || !rhs ||
        (isDirectViewLoad(operation->getOperand(0)) &&
         isDirectViewLoad(operation->getOperand(1))) ||
        !hasPhysicalReductionAxis(*operation, **analysis) ||
        (!compact && isStreamedReduction(*operation, facts)))
      return WalkResult::advance();
    for (Operation *transfer :
         llvm::concat<Operation *>(lhs->transfers, rhs->transfers)) {
      auto existing = replayOwners.find(transfer);
      if (existing != replayOwners.end() && existing->second != operation) {
        transfer->emitOpError(
            "cannot defer one transfer to multiple contractions");
        return WalkResult::interrupt();
      }
      replayOwners[transfer] = operation;
    }
    replays[operation] = ReplayChoice{std::move(*lhs), std::move(*rhs)};
    return WalkResult::advance();
  });
  if (replayWalk.wasInterrupted())
    return failure();

  auto sharedOperand = [&](Value operand, Operation &contract) {
    if (isStaged(contract))
      return true;
    if (!isDirectViewLoad(operand))
      return false;
    if (llvm::all_of(contract.getOperands(), isDirectViewLoad))
      return true;
    return llvm::any_of(contract.getOperands(), [&](Value other) {
      if (other == operand)
        return false;
      if (isContractionDerived(other))
        return true;
      return isStreamedReduction(contract, facts) &&
             target::analyzeContractOperandReplay(other, contract).has_value();
    });
  };

  OpBuilder builder(program.getContext());
  for (auto &entry : facts.contractions) {
    Operation &operation = *entry.first;
    FailureOr<int64_t> node =
        target::getNodeID(operation, "contraction realization");
    plan::ContractOp binding =
        succeeded(node) ? contracts.lookup(*node) : plan::ContractOp();
    if (failed(node) || !binding)
      return operation.emitOpError("has no physical contraction binding");
    bool replayed = replays.contains(&operation);
    auto space = [&](Value operand) -> StringRef {
      if (replayed)
        return isDirectViewLoad(operand) ? StringRef("shared")
                                         : StringRef("private_fragment");
      return sharedOperand(operand, operation) ? StringRef("shared")
                                               : StringRef("private_fragment");
    };
    binding->setAttr("lhs_space", builder.getStringAttr(space(operation.getOperand(0))));
    binding->setAttr("rhs_space", builder.getStringAttr(space(operation.getOperand(1))));
    binding->setAttr("producer_replay", builder.getBoolAttr(replayed));
    for (StringRef attribute : {"accumulator_owner_node",
                                "accumulator_update_node", "accumulator_value",
                                "accumulator_conditional_node",
                                "accumulator_conditional_result"})
      binding->removeAttr(attribute);
    std::optional<AccumulatorFlow> flow = findAccumulatorFlow(operation);
    binding->setAttr("accumulator_flow",
                     builder.getStringAttr(flow ? "loop_carried" : "none"));
    if (flow) {
      FailureOr<int64_t> owner =
          target::getNodeID(*flow->owner, "contraction accumulator owner");
      FailureOr<int64_t> update =
          target::getNodeID(*flow->update, "contraction accumulator update");
      FailureOr<int64_t> value = target::getValueID(
          flow->previous, kernel, operation, "contraction accumulator value");
      if (failed(owner) || failed(update) || failed(value))
        return failure();
      binding->setAttr("accumulator_owner_node", builder.getI64IntegerAttr(*owner));
      binding->setAttr("accumulator_update_node", builder.getI64IntegerAttr(*update));
      binding->setAttr("accumulator_value", builder.getI64IntegerAttr(*value));
      if (flow->conditional) {
        FailureOr<int64_t> conditional = target::getNodeID(
            *flow->conditional, "contraction accumulator conditional");
        if (failed(conditional))
          return failure();
        binding->setAttr("accumulator_conditional_node",
                         builder.getI64IntegerAttr(*conditional));
        binding->setAttr("accumulator_conditional_result",
                         builder.getI64IntegerAttr(flow->conditionalResult));
      }
    }
  }

  for (auto [transferNode, binding] : transfers) {
    Operation *semantic = kernel.nodes.lookup(transferNode);
    if (!semantic)
      return binding.emitOpError("does not bind an executable transfer node");
    Operation &operation = *semantic;
    StringRef name = target::semanticOperationName(operation);
    if (name != "intent.view_load" && name != "intent.view_store" &&
        name != "intent.scatter_unique" && name != "intent.atomic_add" &&
        name != "intent.atomic_cas")
      continue;
    bool load = name == "intent.view_load";
    bool returnedAtomic = name == "intent.atomic_cas" ||
                          (name == "intent.atomic_add" &&
                           operation.getNumResults() == 1);
    bool contractOperand =
        load && operation.getNumResults() == 1 &&
        llvm::any_of(operation.getResult(0).getUsers(), [](Operation *user) {
          StringRef name = target::semanticOperationName(*user);
          return name == "intent.contract" || name == "intent.scaled_contract" ||
                 name == "intent.sparse_contract";
        });
    Operation *soleUser =
        contractOperand && llvm::hasSingleElement(operation.getResult(0).getUsers())
            ? *operation.getResult(0).user_begin()
            : nullptr;
    bool replay = replayOwners.contains(&operation);
    Operation *replayOwner = replayOwners.lookup(&operation);
    bool finalReplay = replayOwner && operation.getNumResults() == 1 &&
                       llvm::is_contained(replayOwner->getOperands(),
                                          operation.getResult(0));
    bool compactSource = replay && binding.getTensorIndexing() == "compact" &&
                         !finalReplay;
    bool shared = (replay && !compactSource) ||
                  (soleUser &&
                   (target::semanticOperationName(*soleUser) ==
                        "intent.sparse_contract" ||
                    sharedOperand(operation.getResult(0), *soleUser)));
    StringRef resultSpace = "none";
    if (load || returnedAtomic)
      resultSpace = shared
                        ? StringRef("shared")
                        : isa<RankedTensorType>(operation.getResult(0).getType())
                              ? StringRef("private_fragment")
                              : StringRef("private_scalar");
    bool defer = replay;
    if (load && shared && soleUser) {
      StringRef consumer = target::semanticOperationName(*soleUser);
      if (consumer == "intent.sparse_contract" || isStaged(*soleUser)) {
        defer = true;
      } else if (consumer == "intent.contract" &&
                 llvm::all_of(soleUser->getOperands(), isDirectViewLoad)) {
        defer = hasPhysicalReductionAxis(*soleUser, **analysis);
      } else if (consumer == "intent.contract" &&
                 isStreamedReduction(*soleUser, facts)) {
        defer = true;
      }
    }
    binding->setAttr("materialization",
                     builder.getStringAttr(defer ? "deferred_to_contract"
                                                : "direct"));
    binding->setAttr("result_space", builder.getStringAttr(resultSpace));
    binding->setAttr(
        "coverage_space",
        builder.getStringAttr(binding.getTensorIndexing() == "compact"
                                  ? (compactSource ? StringRef("shared")
                                                   : resultSpace)
                                  : StringRef("none")));
  }

  for (auto &entry : facts.contractions) {
    Operation &operation = *entry.first;
    FailureOr<int64_t> node =
        target::getNodeID(operation, "contraction form realization");
    plan::ContractOp binding =
        succeeded(node) ? contracts.lookup(*node) : plan::ContractOp();
    if (failed(node) || !binding)
      return operation.emitOpError("has no physical contraction binding");
    for (StringRef attribute : {"reduction_axis_node", "lhs_result_axis_node",
                                "rhs_result_axis_node"})
      binding->removeAttr(attribute);
    binding->setAttr("lhs_form", builder.getStringAttr("none"));
    binding->setAttr("rhs_form", builder.getStringAttr("none"));
    StringRef semantic = target::semanticOperationName(operation);
    if (semantic == "intent.scaled_contract") {
      if (staged.contains(*node) || binding.getProducerReplay())
        return binding.emitOpError(
            "scaled contraction has no legal staged or replay realization");
      binding->setAttr("form", builder.getStringAttr("scaled_direct"));
      continue;
    }
    if (staged.contains(*node)) {
      FailureOr<std::pair<StringRef, StringRef>> forms =
          stagedOperandForms(operation);
      if (failed(forms))
        return failure();
      binding->setAttr("form", builder.getStringAttr("staged"));
      binding->setAttr("lhs_form", builder.getStringAttr(forms->first));
      binding->setAttr("rhs_form", builder.getStringAttr(forms->second));
      continue;
    }
    if (binding.getProducerReplay()) {
      binding->setAttr("form", builder.getStringAttr("replay"));
      FailureOr<int64_t> reduction =
          physicalReductionAxis(operation, **analysis);
      if (failed(reduction))
        return failure();
      binding->setAttr("reduction_axis_node",
                       builder.getI64IntegerAttr(*reduction));
      continue;
    }
    unsigned deferred = llvm::count_if(operation.getOperands(), [&](Value value) {
      Operation *definition = value.getDefiningOp();
      auto transferNode =
          definition
              ? definition->getAttrOfType<IntegerAttr>("intent.node")
              : IntegerAttr();
      plan::TransferOp transfer =
          transferNode ? transfers.lookup(transferNode.getInt())
                       : plan::TransferOp();
      return transfer && transfer.getMaterialization() == "deferred_to_contract";
    });
    StringRef form = deferred == 0 ? StringRef("direct")
                     : deferred == 1 ? StringRef("deferred_one")
                                     : StringRef("deferred_two");
    binding->setAttr("form", builder.getStringAttr(form));
    if (form == "direct")
      continue;
    FailureOr<int64_t> reduction = physicalReductionAxis(operation, **analysis);
    if (failed(reduction))
      return failure();
    binding->setAttr("reduction_axis_node",
                     builder.getI64IntegerAttr(*reduction));
    if (form != "deferred_two")
      continue;
    auto resultAxis = [&](Value operand, StringRef side) -> FailureOr<int64_t> {
      Operation *definition = operand.getDefiningOp();
      auto transferNode =
          definition ? definition->getAttrOfType<IntegerAttr>("intent.node")
                     : IntegerAttr();
      plan::TransferOp transfer =
          transferNode ? transfers.lookup(transferNode.getInt())
                       : plan::TransferOp();
      std::optional<int64_t> selected;
      if (transfer)
        for (int64_t domain : transfer.getDomainNodes()) {
          if (domain == *reduction || (*analysis)->isScalarAxis(domain))
            continue;
          if (selected && *selected != domain)
            return operation.emitOpError()
                   << "has more than one physical " << side << " result axis";
          selected = domain;
        }
      if (!selected)
        return operation.emitOpError()
               << "has no physical " << side << " result axis";
      return *selected;
    };
    FailureOr<int64_t> lhs = resultAxis(operation.getOperand(0), "lhs");
    FailureOr<int64_t> rhs = resultAxis(operation.getOperand(1), "rhs");
    if (failed(lhs) || failed(rhs))
      return failure();
    binding->setAttr("lhs_result_axis_node", builder.getI64IntegerAttr(*lhs));
    binding->setAttr("rhs_result_axis_node", builder.getI64IntegerAttr(*rhs));
  }

  SmallVector<plan::StreamAxisOp> oldStreamAxes(
      program.getBody().getOps<plan::StreamAxisOp>());
  for (plan::StreamAxisOp binding : oldStreamAxes)
    binding.erase();
  builder.setInsertionPoint(program.getBody().front().getTerminator());
  llvm::DenseSet<std::pair<int64_t, int64_t>> streamAxes;
  for (auto &entry : facts.contractions) {
    Operation &operation = *entry.first;
    Operation *stream = enclosingStream(operation);
    if (!stream)
      continue;
    FailureOr<int64_t> streamNode =
        target::getNodeID(*stream, "stream contraction realization");
    if (failed(streamNode))
      return failure();
    for (unsigned lhsAxis : entry.second.lhsReductionAxes) {
      if (lhsAxis >= entry.second.lhsAxes.size())
        return operation.emitOpError("has no logical inner reduction axis");
      Operation *domain = entry.second.lhsAxes[lhsAxis].domain;
      if (!domain)
        continue;
      FailureOr<int64_t> axisNode =
          target::getNodeID(*domain, "stream inner reduction realization");
      if (failed(axisNode))
        return failure();
      if (streamAxes.insert({*streamNode, *axisNode}).second)
        builder.create<plan::StreamAxisOp>(operation.getLoc(),
                                           builder.getI64IntegerAttr(*streamNode),
                                           builder.getI64IntegerAttr(*axisNode));
    }
  }
  return success();
}

class RefineContractionRealizationPass final
    : public PassWrapper<RefineContractionRealizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RefineContractionRealizationPass)
  StringRef getArgument() const final {
    return "intent-refine-gpu-contraction-realization";
  }
  StringRef getDescription() const final {
    return "Refine contraction value placement, replay, and accumulator flow";
  }
  void runOnOperation() final {
    SmallVector<plan::ProgramOp> programs(
        getOperation().getOps<plan::ProgramOp>());
    if (programs.size() != 1 || failed(refineContractions(programs.front())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createRefineContractionRealizationPass() {
  return std::make_unique<RefineContractionRealizationPass>();
}

} // namespace intent::gpu
