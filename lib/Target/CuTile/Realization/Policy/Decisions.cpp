#include "Support/Model.h"

#include "Intent/Target/Common/Realization/ScheduleStructure.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::cutile::realization {

FailureOr<PolicyDecision> decidePolicy(const OperationFacts &facts) {
  const target::KernelFacts &semantics = facts.semantics;
  FailureOr<target::ScheduleStructure> structure =
      target::analyzeScheduleStructure(semantics);
  if (failed(structure))
    return failure();

  PolicyDecision policy;
  policy.programRoot = structure->programRoot;
  policy.workerAxes = {0};
  policy.groupSize = 1;
  policy.fixedOccupancy = 0;

  static constexpr StringLiteral programTiles[] = {"TILE_SIZE_M",
                                                    "TILE_SIZE_N"};
  for (auto [index, domain] : llvm::enumerate(structure->programDomains)) {
    if (index >= std::size(programTiles) &&
        structure->tiledProgramDomains.contains(domain)) {
      domain->emitOpError("exceeds the supported cuTile program rank");
      return failure();
    }
    policy.axes.push_back(AxisDecision{
        domain, semantics.domainSourceAxes.lookup(domain),
        "program_" + std::to_string(index),
        structure->tiledProgramDomains.contains(domain) ? programTiles[index].str()
                                                        : "one"});
  }
  for (auto [index, domain] :
       llvm::enumerate(structure->streamedReductionDomains))
    policy.axes.push_back(AxisDecision{
        domain, semantics.domainSourceAxes.lookup(domain),
        "reduction_" + std::to_string(index),
        index == 0 ? "TILE_SIZE_K" : "TILE_SIZE_REDUCTION"});
  for (Operation *domain : structure->vectorDomains) {
    if (semantics.streamedReductionDomains.contains(domain))
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
        !structure->streamedReductionDomains.empty()) {
      semantics.kernel.entry.emitOpError(
          "persistent rows require one program and one vector domain");
      return failure();
    }
    policy.traversal = "persistent";
    policy.mapping = "persistent_rows";
    policy.fixedOccupancy = 4;
    policy.usesAutotuner = false;
    return policy;
  }

  if (structure->programDomains.size() != 2 ||
      structure->tiledProgramDomains.size() != 2 ||
      structure->streamedReductionDomains.size() != 1) {
    semantics.kernel.entry.emitOpError(
        "grouped tiles require two program axes and one reduction axis");
    return failure();
  }
  policy.traversal = "grouped";
  policy.mapping = "grouped_2d_tiles";
  policy.groupSize = 8;
  for (Operation *domain :
       {structure->programDomains[0], structure->programDomains[1],
        structure->streamedReductionDomains[0]}) {
    FailureOr<std::string> key =
        target::sourceDimensionSymbol(*domain, semantics);
    if (failed(key))
      return failure();
    policy.autotuneKeys.push_back(*key);
  }
  policy.usesAutotuner = true;
  return policy;
}

} // namespace intent::cutile::realization
