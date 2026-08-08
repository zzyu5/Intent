#include "Support/Model.h"

#include "Intent/Target/Common/Realization/SchedulePolicy.h"

using namespace mlir;

namespace intent::tilelang::realization {

FailureOr<PolicyDecision> decidePolicy(const OperationFacts &facts) {
  static const llvm::StringRef programTiles[] = {"TILE_SIZE_M",
                                                 "TILE_SIZE_N"};
  const target::ScheduleRules rules{
      "TileLang",
      "TILE_SIZE_M",
      "TILE_SIZE_M",
      "TILE_SIZE_N",
      programTiles,
      "TILE_SIZE_K",
      "TILE_SIZE_REDUCTION",
      "TILE_SIZE",
      "persistent_rows",
  };
  FailureOr<target::ScheduleDecision> schedule =
      target::decideSchedule(facts.semantics, rules);
  if (failed(schedule))
    return failure();
  PolicyDecision policy;
  static_cast<target::ScheduleDecision &>(policy) = std::move(*schedule);
  return policy;
}

} // namespace intent::tilelang::realization
