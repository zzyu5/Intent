#ifndef INTENT_TARGET_GPU_TRANSFORMS_ANALYSIS_PHYSICALPROGRAM_H
#define INTENT_TARGET_GPU_TRANSFORMS_ANALYSIS_PHYSICALPROGRAM_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "Intent/Target/Common/Realization/KernelFacts.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace intent::gpu {

class PhysicalProgramAnalysis {
public:
  static mlir::FailureOr<std::unique_ptr<PhysicalProgramAnalysis>>
  compute(intent::plan::ProgramOp program);

  intent::target::KernelModel &getKernel() { return *kernel; }
  intent::target::KernelFacts &getFacts() { return *facts; }
  intent::plan::ProgramOp getProgram() const { return program; }
  intent::plan::LaunchOp getLaunch() const { return launch; }
  intent::plan::AxisOp getAxis(int64_t node) const {
    auto found = axes.find(node);
    return found == axes.end() ? intent::plan::AxisOp() : found->second;
  }
  intent::plan::RangeOp getRange(int64_t axisNode, llvm::StringRef purpose,
                                 int64_t level = 0) const;
  llvm::ArrayRef<intent::plan::StageOp> getStages() const { return stages; }
  bool hasWorkerReuse() const;
  bool axisHasRole(int64_t node, llvm::StringRef role) const;
  bool isScalarAxis(int64_t node) const;
  bool isPackedScalarAxis(int64_t node) const;

private:
  explicit PhysicalProgramAnalysis(intent::target::KernelModel model);

  std::unique_ptr<intent::target::KernelModel> kernel;
  std::unique_ptr<intent::target::KernelFacts> facts;
  intent::plan::ProgramOp program;
  intent::plan::LaunchOp launch;
  llvm::DenseMap<int64_t, intent::plan::AxisOp> axes;
  llvm::DenseMap<int64_t, llvm::SmallVector<intent::plan::RangeOp>> ranges;
  llvm::SmallVector<intent::plan::StageOp> stages;
};

} // namespace intent::gpu

#endif
