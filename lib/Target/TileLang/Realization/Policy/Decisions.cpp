#include "Support/Model.h"

#include "Intent/Target/Common/Realization/ScheduleStructure.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::tilelang::realization {

FailureOr<PolicyDecision> decidePolicy(const OperationFacts &facts) {
  const target::KernelFacts &semantics = facts.semantics;
  FailureOr<target::ScheduleStructure> structure =
      target::analyzeScheduleStructure(semantics);
  if (failed(structure))
    return failure();

  PolicyDecision policy;
  policy.programRoot = structure->programRoot;
  policy.stateStream = nullptr;
  policy.raggedRelation = nullptr;
  policy.workerAxes = {0};
  policy.groupSize = 1;
  policy.fixedThreads = 128;
  policy.fixedStages = 1;

  if (!structure->raggedOwnerships.empty()) {
    if (structure->raggedOwnerships.size() != 1 ||
        structure->raggedOwnerships.front().memberDomains.size() != 1 ||
        structure->scatterReductions.size() != 1 ||
        !structure->orderedStreamDomains.empty() ||
        structure->programDomains.size() != 2 ||
        structure->tiledProgramDomains.size() != 1) {
      semantics.kernel.entry.emitOpError(
          "ragged staging requires one outer/member ownership, one additive merge, and one tiled member domain");
      return failure();
    }
    const target::RaggedOwnership &ownership =
        structure->raggedOwnerships.front();
    FailureOr<SmallVector<target::ContractionStage>> stages =
        target::analyzeContractionPipeline(semantics);
    if (failed(stages))
      return failure();
    Operation *memberDomain = ownership.memberDomains.front();
    policy.axes.push_back(AxisDecision{
        ownership.outerDomain,
        semantics.domainSourceAxes.lookup(ownership.outerDomain), "program_0",
        "one"});
    policy.axes.push_back(
        AxisDecision{memberDomain,
                     semantics.domainSourceAxes.lookup(memberDomain),
                     "program_1", "TILE_SIZE_M"});
    FailureOr<std::string> memberKey =
        target::sourceDimensionSymbol(*memberDomain, semantics);
    if (failed(memberKey))
      return failure();
    policy.autotuneKeys.push_back(*memberKey);
    llvm::StringSet<> keys;
    keys.insert(*memberKey);
    for (const target::ContractionStage &stage : *stages) {
      const target::ContractionFact &contract =
          semantics.contractions.lookup(stage.contraction);
      for (const target::LogicalAxis &axis :
           llvm::concat<const target::LogicalAxis>(
               contract.lhsAxes, contract.rhsAxes, contract.resultAxes)) {
        if (axis.domain || axis.extent == "1" ||
            StringRef(axis.extent).starts_with("?region_") ||
            !keys.insert(axis.extent).second)
          continue;
        policy.autotuneKeys.push_back(axis.extent);
      }
    }
    policy.workerAxes = {0, 1};
    policy.raggedRelation = ownership.relation;
    policy.stages = std::move(*stages);
    policy.traversal = "expert_major";
    policy.mapping = "ragged_stages";
    policy.usesAutotuner = true;
    return policy;
  }

  if (!structure->orderedStreamDomains.empty()) {
    if (structure->orderedStreamDomains.size() != 1 ||
        structure->programDomains.size() != 3 ||
        structure->tiledProgramDomains.size() != 1 ||
        structure->contractionDomains.size() != 1 ||
        structure->contractionDomains.front() !=
            structure->orderedStreamDomains.front()) {
      semantics.kernel.entry.emitOpError(
          "multi-axis streamed scheduling requires three program domains, one tiled program domain, and one ordered contraction domain");
      return failure();
    }
    Operation *streamDomain = structure->orderedStreamDomains.front();
    for (const auto &binding : semantics.stateStreams)
      if (binding.second.axisDomain == streamDomain) {
        if (policy.stateStream) {
          semantics.kernel.entry.emitOpError(
              "physical scheduling requires one ordered state stream");
          return failure();
        }
        policy.stateStream = binding.first;
      }
    if (!policy.stateStream) {
      semantics.kernel.entry.emitOpError(
          "ordered stream domain has no state-stream operation");
      return failure();
    }
    policy.workerAxes = {0, 1};
    for (auto [index, domain] : llvm::enumerate(structure->programDomains)) {
      bool tiled = structure->tiledProgramDomains.contains(domain);
      policy.axes.push_back(AxisDecision{
          domain, semantics.domainSourceAxes.lookup(domain),
          "program_" + std::to_string(index), tiled ? "TILE_SIZE_M" : "one"});
      if (tiled) {
        FailureOr<std::string> key =
            target::sourceDimensionSymbol(*domain, semantics);
        if (failed(key))
          return failure();
        policy.autotuneKeys.push_back(*key);
      }
    }
    policy.axes.push_back(AxisDecision{
        streamDomain, semantics.domainSourceAxes.lookup(streamDomain), "stream_0",
        "TILE_SIZE_N"});
    FailureOr<std::string> streamKey =
        target::sourceDimensionSymbol(*streamDomain, semantics);
    if (failed(streamKey))
      return failure();
    policy.autotuneKeys.push_back(*streamKey);
    policy.traversal = "forward";
    policy.mapping = "multi_axis_stream";
    policy.usesAutotuner = true;
    return policy;
  }

  static constexpr StringLiteral programTiles[] = {"TILE_SIZE_M",
                                                    "TILE_SIZE_N"};
  for (auto [index, domain] : llvm::enumerate(structure->programDomains)) {
    if (index >= std::size(programTiles) &&
        structure->tiledProgramDomains.contains(domain)) {
      domain->emitOpError("exceeds the supported TileLang program rank");
      return failure();
    }
    policy.axes.push_back(AxisDecision{
        domain, semantics.domainSourceAxes.lookup(domain),
        "program_" + std::to_string(index),
        structure->tiledProgramDomains.contains(domain) ? programTiles[index].str()
                                                        : "one"});
  }
  for (auto [index, domain] :
       llvm::enumerate(structure->contractionDomains))
    policy.axes.push_back(AxisDecision{
        domain, semantics.domainSourceAxes.lookup(domain),
        "reduction_" + std::to_string(index),
        index == 0 ? "TILE_SIZE_K" : "TILE_SIZE_REDUCTION"});
  for (Operation *domain : structure->vectorDomains) {
    if (semantics.contractionDomains.contains(domain))
      continue;
    unsigned index = llvm::count_if(policy.axes, [](const AxisDecision &axis) {
      return StringRef(axis.role).starts_with("lane_");
    });
    policy.axes.push_back(AxisDecision{
        domain, semantics.domainSourceAxes.lookup(domain),
        "lane_" + std::to_string(index), "TILE_SIZE"});
  }

  if (structure->tiledProgramDomains.empty()) {
    if (structure->programDomains.size() != 1 ||
        structure->vectorDomains.size() != 1 ||
        !structure->contractionDomains.empty()) {
      semantics.kernel.entry.emitOpError(
          "persistent rows require one program and one vector domain");
      return failure();
    }
    policy.traversal = "persistent";
    policy.mapping = "persistent_rows";
    policy.fixedThreads = 128;
    policy.fixedStages = 1;
    policy.usesAutotuner = false;
    return policy;
  }

  if (structure->programDomains.size() != 2 ||
      structure->tiledProgramDomains.size() != 2 ||
      structure->contractionDomains.size() != 1) {
    semantics.kernel.entry.emitOpError(
        "grouped tiles require two program axes and one reduction axis");
    return failure();
  }
  policy.traversal = "grouped";
  policy.mapping = "grouped_2d_tiles";
  policy.groupSize = 1;
  for (Operation *domain :
       {structure->programDomains[0], structure->programDomains[1],
        structure->contractionDomains[0]}) {
    FailureOr<std::string> key =
        target::sourceDimensionSymbol(*domain, semantics);
    if (failed(key))
      return failure();
    policy.autotuneKeys.push_back(*key);
  }
  policy.usesAutotuner = true;
  return policy;
}

} // namespace intent::tilelang::realization
