#include "Support/Model.h"

#include "Intent/Target/Common/Realization/SchedulePolicy.h"

using namespace mlir;

namespace intent::triton::realization {

FailureOr<PolicyDecision> decidePolicy(const OperationFacts &facts) {
  static const llvm::StringRef programTiles[] = {"BLOCK_SIZE_M",
                                                 "BLOCK_SIZE_N"};
  const target::ScheduleRules rules{
      "Triton",
      "BLOCK_SIZE_M",
      "BLOCK_SIZE_Q",
      "BLOCK_SIZE_K",
      programTiles,
      "BLOCK_SIZE_K",
      "BLOCK_SIZE_REDUCTION",
      "next_power_of_two",
      "grid_stride",
  };
  FailureOr<target::ScheduleDecision> schedule =
      target::decideSchedule(facts.semantics, rules);
  if (failed(schedule))
    return failure();
  PolicyDecision policy;
  static_cast<target::ScheduleDecision &>(policy) = std::move(*schedule);
  return policy;
}

} // namespace intent::triton::realization
