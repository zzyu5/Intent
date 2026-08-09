#include "Intent/Target/Common/Realization/SchedulePolicy.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::target {
namespace {

bool contains(ArrayRef<Operation *> operations, Operation *operation) {
  return llvm::is_contained(operations, operation);
}

unsigned terminalWriteCount(const KernelFacts &facts) {
  return llvm::count_if(facts.boundaryDomains, [](const auto &binding) {
    StringRef name = binding.first->getName().getStringRef();
    return name == "intent.view_store" || name == "intent.scatter_unique" ||
           name == "intent.scatter_reduce";
  });
}

void appendUnique(SmallVectorImpl<std::string> &values, StringRef value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value.str());
}

FailureOr<std::string> appendDimensionKey(
    Operation &domain, const KernelFacts &facts,
    SmallVectorImpl<std::string> &keys) {
  FailureOr<std::string> key = sourceDimensionSymbol(domain, facts);
  if (failed(key))
    return failure();
  appendUnique(keys, *key);
  return *key;
}

FailureOr<RaggedOwnership *>
singleRaggedOwnership(ScheduleStructure &structure, const KernelFacts &facts) {
  if (structure.raggedOwnerships.size() != 1) {
    facts.kernel.entry.emitOpError(
        "GPU ragged ownership currently requires one relation");
    return failure();
  }
  return &structure.raggedOwnerships.front();
}

LogicalResult decideOwnership(ScheduleStructure &structure,
                              const KernelFacts &facts,
                              ScheduleDecision &decision) {
  if (!structure.raggedOwnerships.empty()) {
    FailureOr<RaggedOwnership *> ownership =
        singleRaggedOwnership(structure, facts);
    if (failed(ownership))
      return failure();
    SmallVector<Operation *> ownedMembers;
    for (Operation *member : (*ownership)->memberDomains)
      if (contains(structure.programDomains, member))
        ownedMembers.push_back(member);
    if (ownedMembers.size() != 1 || structure.programDomains.size() != 2 ||
        structure.tiledProgramDomains.size() != 1 ||
        !structure.tiledProgramDomains.contains(ownedMembers.front()) ||
        !contains(structure.programDomains, (*ownership)->outerDomain)) {
      return facts.kernel.entry.emitOpError(
          "ragged ownership requires one outer domain and one tiled member domain");
    }
    decision.ownership = "ragged";
    decision.workerAxes = {0, 1};
    decision.raggedRelations.push_back((*ownership)->relation);
    decision.axes.push_back(AxisDecision{
        (*ownership)->outerDomain,
        facts.domainSourceAxes.lookup((*ownership)->outerDomain), "program_0",
        "one"});
    decision.axes.push_back(AxisDecision{
        ownedMembers.front(), facts.domainSourceAxes.lookup(ownedMembers.front()),
        "program_1", structure.orderedStreamDomains.empty() ? "ragged_member"
                                                            : "query"});
    return success();
  }

  if (structure.tiledProgramDomains.empty()) {
    if (structure.programDomains.size() != 1)
      return facts.kernel.entry.emitOpError(
          "row ownership requires one program domain");
    decision.ownership = "row";
    decision.workerAxes = {0};
    Operation *domain = structure.programDomains.front();
    decision.axes.push_back(AxisDecision{
        domain, facts.domainSourceAxes.lookup(domain), "program_0", "one"});
    return success();
  }

  decision.ownership = "tiled";
  decision.workerAxes = structure.programDomains.size() == 3
                            ? SmallVector<int64_t>{0, 1}
                            : SmallVector<int64_t>{0};
  for (auto [index, domain] : llvm::enumerate(structure.programDomains)) {
    if (index >= 3)
      return domain->emitOpError("exceeds the supported GPU program rank");
    bool tiled = structure.tiledProgramDomains.contains(domain);
    std::string tile = "one";
    if (tiled)
      tile = structure.programDomains.size() == 3
                 ? "query"
                 : (index == 0 ? "program_m" : "program_n");
    decision.axes.push_back(AxisDecision{
        domain, facts.domainSourceAxes.lookup(domain),
        "program_" + std::to_string(index), std::move(tile)});
  }
  return success();
}

LogicalResult decideOrderedTraversal(ScheduleStructure &structure,
                                     const KernelFacts &facts,
                                     ScheduleDecision &decision) {
  if (structure.orderedStreamDomains.empty())
    return success();
  if (structure.orderedStreamDomains.size() != 1)
    return facts.kernel.entry.emitOpError(
        "GPU ordered traversal currently requires one streamed domain");

  Operation *streamDomain = structure.orderedStreamDomains.front();
  for (const auto &binding : facts.stateStreams)
    if (binding.second.axisDomain == streamDomain)
      decision.stateStreams.push_back(binding.first);
  llvm::sort(decision.stateStreams, [](Operation *lhs, Operation *rhs) {
    return lhs->getAttrOfType<IntegerAttr>("intent.node").getInt() <
           rhs->getAttrOfType<IntegerAttr>("intent.node").getInt();
  });
  if (decision.stateStreams.empty())
    return facts.kernel.entry.emitOpError(
        "ordered traversal has no state-stream operation");

  if (decision.ownership == "row") {
    if (!structure.tiledProgramDomains.empty() ||
        !structure.contractionDomains.empty())
      return facts.kernel.entry.emitOpError(
          "row ownership cannot realize this ordered traversal");
  } else if (decision.ownership == "tiled") {
    if (structure.programDomains.size() != 3 ||
        structure.tiledProgramDomains.size() != 1 ||
        structure.contractionDomains.size() != 1 ||
        structure.contractionDomains.front() != streamDomain)
      return facts.kernel.entry.emitOpError(
          "tiled ordered traversal requires one query tile and one streamed contraction domain");
  } else if (decision.ownership == "ragged") {
    FailureOr<RaggedOwnership *> ownership =
        singleRaggedOwnership(structure, facts);
    if (failed(ownership))
      return failure();
    if (!contains((*ownership)->memberDomains, streamDomain) ||
        contains(structure.programDomains, streamDomain) ||
        terminalWriteCount(facts) != 1 ||
        structure.contractionDomains.size() != 1 ||
        structure.contractionDomains.front() != streamDomain)
      return facts.kernel.entry.emitOpError(
          "ragged ordered traversal requires distinct owned and streamed member domains with one terminal scatter");
  } else {
    return facts.kernel.entry.emitOpError(
        "ordered traversal has no compatible ownership decision");
  }

  decision.axes.push_back(AxisDecision{
      streamDomain, facts.domainSourceAxes.lookup(streamDomain), "stream_0",
      "stream"});
  if (decision.ownership != "row") {
    auto query = llvm::find_if(decision.axes, [](const AxisDecision &axis) {
      return axis.tile == "query";
    });
    if (query == decision.axes.end() ||
        failed(appendDimensionKey(*query->domain, facts,
                                  decision.autotuneKeys)))
      return failure();
    if (failed(appendDimensionKey(*streamDomain, facts,
                                  decision.autotuneKeys)))
      return failure();
    decision.autotuneParameters = {"query", "stream"};
  } else {
    if (failed(appendDimensionKey(*streamDomain, facts,
                                  decision.autotuneKeys)))
      return failure();
    decision.autotuneParameters = {"stream"};
  }
  decision.traversals.push_back("ordered_stream");
  decision.usesAutotuner = true;
  return success();
}

LogicalResult decideUnorderedTraversal(ScheduleStructure &structure,
                                       const KernelFacts &facts,
                                       ScheduleDecision &decision) {
  if (!structure.orderedStreamDomains.empty())
    return success();

  if (decision.ownership == "ragged") {
    if (structure.scatterWrites.size() != 1)
      return facts.kernel.entry.emitOpError(
          "staged ragged traversal requires one terminal scatter write");
    FailureOr<SmallVector<ContractionStage>> stages =
        analyzeContractionPipeline(facts);
    if (failed(stages))
      return failure();
    decision.stages = std::move(*stages);
    decision.traversals.push_back("staged");
    Operation *memberDomain = nullptr;
    for (const AxisDecision &axis : decision.axes)
      if (axis.role == "program_1")
        memberDomain = axis.domain;
    if (!memberDomain ||
        failed(appendDimensionKey(*memberDomain, facts,
                                  decision.autotuneKeys)))
      return failure();
    llvm::StringSet<> keys;
    for (const std::string &key : decision.autotuneKeys)
      keys.insert(key);
    for (const ContractionStage &stage : decision.stages) {
      const ContractionFact &contract =
          facts.contractions.lookup(stage.contraction);
      for (const LogicalAxis &axis : llvm::concat<const LogicalAxis>(
               contract.lhsAxes, contract.rhsAxes, contract.resultAxes)) {
        if (axis.domain || axis.extent == "1" ||
            StringRef(axis.extent).starts_with("?region_") ||
            !keys.insert(axis.extent).second)
          continue;
        decision.autotuneKeys.push_back(axis.extent);
      }
    }
    decision.autotuneParameters = {"ragged_member", "feature", "reduction"};
    decision.usesAutotuner = true;
    return success();
  }

  if (decision.ownership == "row") {
    if (structure.vectorDomains.size() != 1 ||
        !structure.contractionDomains.empty())
      return facts.kernel.entry.emitOpError(
          "persistent row traversal requires one vector domain");
    Operation *domain = structure.vectorDomains.front();
    decision.axes.push_back(AxisDecision{
        domain, facts.domainSourceAxes.lookup(domain), "lane_0", "row_vector"});
    decision.traversals.push_back("persistent");
    return success();
  }

  if (structure.programDomains.size() != 2 ||
      structure.tiledProgramDomains.size() != 2 ||
      structure.contractionDomains.size() != 1)
    return facts.kernel.entry.emitOpError(
        "grouped traversal requires two tiled program axes and one reduction axis");
  Operation *reduction = structure.contractionDomains.front();
  decision.axes.push_back(AxisDecision{
      reduction, facts.domainSourceAxes.lookup(reduction), "reduction_0",
      "reduction"});
  for (Operation *domain :
       {structure.programDomains[0], structure.programDomains[1], reduction})
    if (failed(appendDimensionKey(*domain, facts, decision.autotuneKeys)))
      return failure();
  decision.traversals.push_back("grouped");
  decision.autotuneParameters = {"program_m", "program_n", "reduction",
                                 "group_m"};
  decision.usesAutotuner = true;
  return success();
}

} // namespace

FailureOr<ScheduleDecision> decideGpuSchedule(const KernelFacts &facts) {
  FailureOr<ScheduleStructure> analyzed = analyzeScheduleStructure(facts);
  if (failed(analyzed))
    return failure();
  ScheduleStructure structure = std::move(*analyzed);
  ScheduleDecision decision;
  decision.programRoot = structure.programRoot;
  if (failed(decideOwnership(structure, facts, decision)) ||
      failed(decideOrderedTraversal(structure, facts, decision)) ||
      failed(decideUnorderedTraversal(structure, facts, decision)))
    return failure();
  if (decision.ownership.empty() || decision.traversals.empty())
    return facts.kernel.entry.emitOpError(
        "GPU realization did not resolve ownership and traversal components");
  return decision;
}

} // namespace intent::target
