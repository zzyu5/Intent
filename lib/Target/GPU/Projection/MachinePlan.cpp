#include "Intent/Target/GPU/Projection/MachinePlan.h"

using namespace mlir;

namespace intent::gpu {

FailureOr<MachinePlanIndex>
indexMachinePlan(intent::plan::RealizationOp realization) {
  if (failed(intent::plan::verifyGpuRealization(realization)))
    return failure();
  MachinePlanIndex index;
  for (Operation &operation : realization.getBody().front()) {
    if (auto value = dyn_cast<intent::plan::DeviceOp>(operation))
      index.device = value;
    else if (auto value = dyn_cast<intent::plan::ProgramOp>(operation))
      index.program = value;
    else if (auto value = dyn_cast<intent::plan::AxisOp>(operation))
      index.axes.push_back(value);
    else if (auto value = dyn_cast<intent::plan::StorageOp>(operation))
      index.storage.push_back(value);
    else if (auto value = dyn_cast<intent::plan::PaddingOp>(operation))
      index.paddings.push_back(value);
    else if (auto value = dyn_cast<intent::plan::TransferOp>(operation))
      index.transfers.push_back(value);
    else if (auto value = dyn_cast<intent::plan::ReductionOp>(operation))
      index.reductions.push_back(value);
    else if (auto value = dyn_cast<intent::plan::PointwiseOp>(operation))
      index.pointwise.push_back(value);
    else if (auto value = dyn_cast<intent::plan::ContractOp>(operation))
      index.contracts.push_back(value);
    else if (auto value = dyn_cast<intent::plan::StreamOp>(operation))
      index.streams.push_back(value);
    else if (auto value = dyn_cast<intent::plan::RaggedOp>(operation))
      index.ragged.push_back(value);
    else if (auto value = dyn_cast<intent::plan::StageOp>(operation))
      index.stages.push_back(value);
    else if (auto value = dyn_cast<intent::plan::AtomicOp>(operation))
      index.atomics.push_back(value);
  }
  return index;
}

FailureOr<intent::plan::AutotuneOp>
indexMachineSearch(intent::plan::SearchSpaceOp searchSpace) {
  if (!searchSpace)
    return intent::plan::AutotuneOp();
  if (failed(intent::plan::verifyGpuSearchSpace(searchSpace)))
    return failure();
  return *searchSpace.getBody().front().getOps<intent::plan::AutotuneOp>().begin();
}

StringAttr stringAttr(OpBuilder &builder, llvm::StringRef value) {
  return builder.getStringAttr(value);
}

ArrayAttr stringArrayAttr(OpBuilder &builder,
                          llvm::ArrayRef<llvm::StringRef> values) {
  SmallVector<Attribute> attributes;
  for (StringRef value : values)
    attributes.push_back(builder.getStringAttr(value));
  return builder.getArrayAttr(attributes);
}

} // namespace intent::gpu
