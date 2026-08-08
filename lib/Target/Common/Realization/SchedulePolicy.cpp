#include "Intent/Target/Common/Realization/SchedulePolicy.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::target {

FailureOr<ScheduleDecision> decideSchedule(const KernelFacts &facts,
                                           const ScheduleRules &rules) {
  FailureOr<ScheduleStructure> structure = analyzeScheduleStructure(facts);
  if (failed(structure))
    return failure();

  ScheduleDecision decision;
  decision.programRoot = structure->programRoot;
  decision.workerAxes = {0};

  if (!structure->raggedOwnerships.empty()) {
    if (structure->raggedOwnerships.size() != 1 ||
        structure->raggedOwnerships.front().memberDomains.size() != 1 ||
        structure->scatterReductions.size() != 1 ||
        !structure->orderedStreamDomains.empty() ||
        structure->programDomains.size() != 2 ||
        structure->tiledProgramDomains.size() != 1) {
      facts.kernel.entry.emitOpError(
          "ragged staging requires one outer/member ownership, one additive merge, and one tiled member domain");
      return failure();
    }
    const RaggedOwnership &ownership = structure->raggedOwnerships.front();
    FailureOr<SmallVector<ContractionStage>> stages =
        analyzeContractionPipeline(facts);
    if (failed(stages))
      return failure();
    Operation *memberDomain = ownership.memberDomains.front();
    decision.axes.push_back(AxisDecision{
        ownership.outerDomain,
        facts.domainSourceAxes.lookup(ownership.outerDomain), "program_0",
        "one"});
    decision.axes.push_back(
        AxisDecision{memberDomain, facts.domainSourceAxes.lookup(memberDomain),
                     "program_1", rules.raggedMemberTile.str()});
    FailureOr<std::string> memberKey =
        sourceDimensionSymbol(*memberDomain, facts);
    if (failed(memberKey))
      return failure();
    decision.autotuneKeys.push_back(*memberKey);
    llvm::StringSet<> keys;
    keys.insert(*memberKey);
    for (const ContractionStage &stage : *stages) {
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
    decision.workerAxes = {0, 1};
    decision.raggedRelation = ownership.relation;
    decision.stages = std::move(*stages);
    decision.traversal = "expert_major";
    decision.mapping = "ragged_stages";
    decision.usesAutotuner = true;
    return decision;
  }

  if (!structure->orderedStreamDomains.empty()) {
    if (structure->orderedStreamDomains.size() != 1 ||
        structure->programDomains.size() != 3 ||
        structure->tiledProgramDomains.size() != 1 ||
        structure->contractionDomains.size() != 1 ||
        structure->contractionDomains.front() !=
            structure->orderedStreamDomains.front()) {
      facts.kernel.entry.emitOpError(
          "multi-axis streamed scheduling requires three program domains, one tiled program domain, and one ordered contraction domain");
      return failure();
    }
    Operation *streamDomain = structure->orderedStreamDomains.front();
    for (const auto &binding : facts.stateStreams)
      if (binding.second.axisDomain == streamDomain) {
        if (decision.stateStream) {
          facts.kernel.entry.emitOpError(
              "physical scheduling requires one ordered state stream");
          return failure();
        }
        decision.stateStream = binding.first;
      }
    if (!decision.stateStream) {
      facts.kernel.entry.emitOpError(
          "ordered stream domain has no state-stream operation");
      return failure();
    }
    decision.workerAxes = {0, 1};
    for (auto [index, domain] : llvm::enumerate(structure->programDomains)) {
      bool tiled = structure->tiledProgramDomains.contains(domain);
      decision.axes.push_back(AxisDecision{
          domain, facts.domainSourceAxes.lookup(domain),
          "program_" + std::to_string(index),
          tiled ? rules.streamProgramTile.str() : "one"});
      if (tiled) {
        FailureOr<std::string> key = sourceDimensionSymbol(*domain, facts);
        if (failed(key))
          return failure();
        decision.autotuneKeys.push_back(*key);
      }
    }
    decision.axes.push_back(AxisDecision{
        streamDomain, facts.domainSourceAxes.lookup(streamDomain), "stream_0",
        rules.streamTile.str()});
    FailureOr<std::string> streamKey =
        sourceDimensionSymbol(*streamDomain, facts);
    if (failed(streamKey))
      return failure();
    decision.autotuneKeys.push_back(*streamKey);
    decision.traversal = "forward";
    decision.mapping = "multi_axis_stream";
    decision.usesAutotuner = true;
    return decision;
  }

  for (auto [index, domain] : llvm::enumerate(structure->programDomains)) {
    if (index >= rules.programTiles.size() &&
        structure->tiledProgramDomains.contains(domain)) {
      domain->emitOpError("exceeds the supported ")
          << rules.targetName << " tiled program rank";
      return failure();
    }
    decision.axes.push_back(AxisDecision{
        domain, facts.domainSourceAxes.lookup(domain),
        "program_" + std::to_string(index),
        structure->tiledProgramDomains.contains(domain)
            ? rules.programTiles[index].str()
            : "one"});
  }
  for (auto [index, domain] : llvm::enumerate(structure->contractionDomains))
    decision.axes.push_back(AxisDecision{
        domain, facts.domainSourceAxes.lookup(domain),
        "reduction_" + std::to_string(index),
        (index == 0 ? rules.firstReductionTile : rules.extraReductionTile)
            .str()});
  for (Operation *domain : structure->vectorDomains) {
    if (facts.contractionDomains.contains(domain))
      continue;
    unsigned index =
        llvm::count_if(decision.axes, [](const AxisDecision &axis) {
          return StringRef(axis.role).starts_with("lane_");
        });
    decision.axes.push_back(AxisDecision{
        domain, facts.domainSourceAxes.lookup(domain),
        "lane_" + std::to_string(index), rules.vectorTile.str()});
  }

  if (structure->tiledProgramDomains.empty()) {
    if (structure->programDomains.size() != 1 ||
        structure->vectorDomains.size() != 1 ||
        !structure->contractionDomains.empty()) {
      facts.kernel.entry.emitOpError(
          "persistent mapping requires one program and one vector domain");
      return failure();
    }
    decision.traversal = "persistent";
    decision.mapping = rules.persistentMapping.str();
    return decision;
  }

  if (structure->programDomains.size() != 2 ||
      structure->tiledProgramDomains.size() != 2 ||
      structure->contractionDomains.size() != 1) {
    facts.kernel.entry.emitOpError(
        "grouped scheduling requires two tiled program axes and one reduction axis");
    return failure();
  }
  decision.traversal = "grouped";
  decision.mapping = "grouped_2d_tiles";
  for (Operation *domain :
       {structure->programDomains[0], structure->programDomains[1],
        structure->contractionDomains[0]}) {
    FailureOr<std::string> key = sourceDimensionSymbol(*domain, facts);
    if (failed(key))
      return failure();
    decision.autotuneKeys.push_back(*key);
  }
  decision.usesAutotuner = true;
  return decision;
}

} // namespace intent::target
