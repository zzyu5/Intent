#include "Intent/Target/Common/Realization/ScheduleStructure.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::target {
namespace {

FailureOr<Operation *> ownedDomain(Operation &parallel,
                                   const KernelFacts &facts) {
  Operation *source = parallel.getOperand(0).getDefiningOp();
  if (facts.domainSourceAxes.count(source))
    return source;
  auto partition = facts.partitionDomains.find(source);
  if (partition != facts.partitionDomains.end())
    return partition->second;
  parallel.emitOpError("has no source domain for schedule analysis");
  return failure();
}

SmallVector<Operation *>
orderedDomains(const llvm::DenseSet<Operation *> &domains) {
  SmallVector<Operation *> result(domains.begin(), domains.end());
  llvm::sort(result, [](Operation *lhs, Operation *rhs) {
    return lhs->getAttrOfType<IntegerAttr>("intent.node").getInt() <
           rhs->getAttrOfType<IntegerAttr>("intent.node").getInt();
  });
  return result;
}

bool dependsOn(Value value, Operation &producer,
               llvm::DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  if (value.getDefiningOp() == &producer)
    return true;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return dependsOn(operand, producer, visited);
  });
}

} // namespace

FailureOr<std::string> sourceDimensionSymbol(Operation &domain,
                                             const KernelFacts &facts) {
  Value source = facts.domainSources.lookup(&domain);
  int64_t axis = facts.domainSourceAxes.lookup(&domain);
  auto argument = llvm::find_if(facts.kernel.abi.arguments,
                                [&](const ABIArgument &candidate) {
                                  return candidate.value == source;
                                });
  if (argument == facts.kernel.abi.arguments.end()) {
    domain.emitOpError("does not refer to a canonical ABI argument");
    return failure();
  }
  auto shape = argument->metadata.getAs<ArrayAttr>("shape");
  auto symbol = shape && axis >= 0 && static_cast<size_t>(axis) < shape.size()
                    ? dyn_cast<StringAttr>(shape[axis])
                    : StringAttr();
  if (!symbol || symbol.getValue().empty()) {
    domain.emitOpError("does not expose a symbolic specialization dimension");
    return failure();
  }
  return symbol.getValue().str();
}

FailureOr<ScheduleStructure>
analyzeScheduleStructure(const KernelFacts &facts) {
  if (facts.parallels.empty()) {
    facts.kernel.entry.emitOpError("has no parallel ownership to schedule");
    return failure();
  }

  ScheduleStructure structure;
  llvm::DenseSet<Operation *> programDomainSet;
  for (Operation *parallel : facts.parallels) {
    FailureOr<Operation *> domain = ownedDomain(*parallel, facts);
    if (failed(domain))
      return failure();
    if (programDomainSet.insert(*domain).second)
      structure.programDomains.push_back(*domain);
    Operation *source = parallel->getOperand(0).getDefiningOp();
    if (facts.partitionDomains.count(source))
      structure.tiledProgramDomains.insert(*domain);
  }
  structure.vectorDomains = orderedDomains(facts.vectorDomains);
  structure.contractionDomains = orderedDomains(facts.contractionDomains);
  structure.orderedStreamDomains = orderedDomains(facts.orderedStreamDomains);
  for (const auto &entry : facts.raggedRelations) {
    const RaggedRelationFact &fact = entry.second;
    if (!fact.outerDomain || fact.memberDomains.empty()) {
      entry.first->emitOpError("has incomplete ragged ownership facts");
      return failure();
    }
    structure.raggedOwnerships.push_back(
        RaggedOwnership{entry.first, fact.outerDomain, fact.memberDomains});
  }
  llvm::sort(structure.raggedOwnerships,
             [](const RaggedOwnership &lhs, const RaggedOwnership &rhs) {
               return lhs.relation->getAttrOfType<IntegerAttr>("intent.node")
                          .getInt() <
                      rhs.relation->getAttrOfType<IntegerAttr>("intent.node")
                          .getInt();
             });
  structure.scatterReductions.assign(facts.scatterReductions.begin(),
                                     facts.scatterReductions.end());
  llvm::sort(structure.scatterReductions,
             [](Operation *lhs, Operation *rhs) {
               return lhs->getAttrOfType<IntegerAttr>("intent.node").getInt() <
                      rhs->getAttrOfType<IntegerAttr>("intent.node").getInt();
             });

  llvm::DenseSet<Operation *> classified = programDomainSet;
  classified.insert(facts.vectorDomains.begin(), facts.vectorDomains.end());
  classified.insert(facts.contractionDomains.begin(),
                    facts.contractionDomains.end());
  classified.insert(facts.orderedStreamDomains.begin(),
                    facts.orderedStreamDomains.end());
  for (const auto &binding : facts.boundaryDomains)
    for (Operation *domain : binding.second)
      if (!classified.contains(domain)) {
        domain->emitOpError(
            "has no program, vector, or streamed-reduction mechanism");
        return failure();
      }

  llvm::DenseSet<Operation *> rootCandidates;
  for (const RegionNode &region : facts.kernel.regions.nodes) {
    if (region.operation->getName().getStringRef() != "intent.parallel")
      continue;
    bool hasParallelAncestor = false;
    Operation *ancestor = region.parent;
    while (ancestor) {
      if (ancestor->getName().getStringRef() == "intent.parallel") {
        hasParallelAncestor = true;
        break;
      }
      auto position = facts.kernel.regions.positions.find(ancestor);
      ancestor = position == facts.kernel.regions.positions.end()
                     ? nullptr
                     : facts.kernel.regions.nodes[position->second].parent;
    }
    if (!hasParallelAncestor)
      rootCandidates.insert(region.operation);
  }
  if (rootCandidates.size() != 1) {
    facts.kernel.entry.emitOpError(
        "schedule analysis requires exactly one root parallel ownership region");
    return failure();
  }
  structure.programRoot = *rootCandidates.begin();
  return structure;
}

FailureOr<SmallVector<ContractionStage>>
analyzeContractionPipeline(const KernelFacts &facts) {
  SmallVector<Operation *> contractions;
  contractions.reserve(facts.contractions.size());
  for (const auto &entry : facts.contractions)
    contractions.push_back(entry.first);
  llvm::sort(contractions, [](Operation *lhs, Operation *rhs) {
    return lhs->getAttrOfType<IntegerAttr>("intent.node").getInt() <
           rhs->getAttrOfType<IntegerAttr>("intent.node").getInt();
  });
  if (contractions.empty()) {
    facts.kernel.entry.emitOpError("has no contraction pipeline to realize");
    return failure();
  }

  SmallVector<ContractionStage> stages;
  for (Operation *contraction : contractions)
    stages.push_back(ContractionStage{contraction, {}, {}});
  for (size_t index = 0; index + 1 < contractions.size(); ++index) {
    Operation *producer = contractions[index];
    Operation *consumer = contractions[index + 1];
    SmallVector<Value> dependentOperands;
    for (Value operand : consumer->getOperands()) {
      llvm::DenseSet<Value> visited;
      if (dependsOn(operand, *producer, visited))
        dependentOperands.push_back(operand);
    }
    if (dependentOperands.size() != 1 ||
        !isa<RankedTensorType>(dependentOperands.front().getType())) {
      consumer->emitOpError(
          "does not form one materializable edge from the preceding contraction");
      return failure();
    }
    Value materialized = dependentOperands.front();
    if (!facts.kernel.valueIDs.count(materialized)) {
      consumer->emitOpError(
          "contraction stage edge has no canonical value identity");
      return failure();
    }
    stages[index].outputs.push_back(materialized);
    stages[index + 1].inputs.push_back(materialized);
  }
  return stages;
}

} // namespace intent::target
