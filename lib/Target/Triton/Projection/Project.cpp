#include "Intent/Target/Triton/Projection/Project.h"

#include "Intent/Target/GPU/Projection/MachinePlan.h"
#include "Intent/Target/Triton/IR/TritonOps.h"

using namespace mlir;

namespace intent::triton {
namespace {

IntegerAttr i64(OpBuilder &builder, int64_t value) {
  return builder.getI64IntegerAttr(value);
}

FailureOr<StringRef> tileSpelling(Operation *operation, StringRef role) {
  if (role == "one")
    return StringRef("one");
  if (role == "row_vector")
    return StringRef("BLOCK_SIZE");
  if (role == "program_m" || role == "ragged_member")
    return StringRef("BLOCK_SIZE_M");
  if (role == "program_n")
    return StringRef("BLOCK_SIZE_N");
  if (role == "reduction")
    return StringRef("BLOCK_SIZE_K");
  if (role == "reduction_extra")
    return StringRef("BLOCK_SIZE_REDUCTION");
  if (role == "query")
    return StringRef("BLOCK_SIZE_Q");
  if (role == "stream")
    return StringRef("BLOCK_SIZE_K");
  operation->emitOpError("has no Triton tile spelling for role ") << role;
  return failure();
}

FailureOr<StringRef> pointwiseSpelling(Operation *operation, StringRef role) {
  if (role == "indices")
    return StringRef("logical_indices");
  if (role == "broadcast")
    return StringRef("alias");
  if (role == "cast")
    return StringRef("tl.cast");
  if (role == "unary_exp")
    return StringRef("tl.exp");
  if (role == "unary_exp2")
    return StringRef("tl.exp2");
  if (role == "unary_log")
    return StringRef("tl.log");
  if (role == "unary_rsqrt")
    return StringRef("tl.rsqrt");
  if (role == "unary_negate")
    return StringRef("python_negate");
  if (role == "binary_add")
    return StringRef("python_add");
  if (role == "binary_subtract")
    return StringRef("python_subtract");
  if (role == "binary_multiply")
    return StringRef("python_multiply");
  if (role == "binary_true_divide")
    return StringRef("python_true_divide");
  if (role == "binary_maximum")
    return StringRef("tl.maximum");
  if (role == "compare_greater_equal")
    return StringRef("python_greater_equal");
  if (role == "mask")
    return StringRef("tl.where");
  if (role == "full")
    return StringRef("tl.full");
  if (role == "zeros")
    return StringRef("tl.zeros");
  if (role == "members")
    return StringRef("tl.members");
  if (role == "expand_dims")
    return StringRef("expand_dims");
  if (role == "indirect_gather")
    return StringRef("tl.indirect_gather");
  operation->emitOpError("has no Triton pointwise spelling for role ") << role;
  return failure();
}

FailureOr<StringRef> parameterSpelling(Operation *operation, StringRef role) {
  if (role == "program_m" || role == "ragged_member")
    return StringRef("BLOCK_SIZE_M");
  if (role == "program_n" || role == "feature")
    return StringRef("BLOCK_SIZE_N");
  if (role == "reduction")
    return StringRef("BLOCK_SIZE_K");
  if (role == "group_m")
    return StringRef("GROUP_SIZE_M");
  if (role == "query")
    return StringRef("BLOCK_SIZE_Q");
  if (role == "stream")
    return StringRef("BLOCK_SIZE_K");
  operation->emitOpError("has no Triton tuner parameter for role ") << role;
  return failure();
}

StringRef ownershipSpelling(StringRef ownership) {
  if (ownership == "row")
    return "program_rows";
  if (ownership == "tiled")
    return "program_tiles";
  if (ownership == "ragged")
    return "program_ragged";
  return {};
}

LogicalResult projectRealization(intent::plan::RealizationOp realization) {
  FailureOr<gpu::MachinePlanIndex> indexed = gpu::indexMachinePlan(realization);
  if (failed(indexed))
    return failure();
  OpBuilder builder(realization.getContext());
  builder.setInsertionPoint(realization.getBody().front().getTerminator());
  Location location = realization.getLoc();
  builder.create<plan::TargetOp>(location,
                                 i64(builder, indexed->device.getDevice()));
  const intent::target::CapabilityProfile &capabilities =
      plan::getCapabilityProfile();
  builder.create<plan::CapabilitiesOp>(
      location,
      gpu::stringArrayAttr(builder, capabilities.intentDecided),
      gpu::stringArrayAttr(builder, capabilities.delegated),
      gpu::stringArrayAttr(builder, capabilities.absent));
  for (intent::plan::AxisOp axis : indexed->axes) {
    FailureOr<StringRef> tile = tileSpelling(axis, axis.getTile());
    if (failed(tile))
      return failure();
    builder.create<plan::AxisOp>(
        axis.getLoc(), axis.getNodeAttr(), axis.getSourceAxisAttr(),
        axis.getRoleAttr(), gpu::stringAttr(builder, *tile));
  }
  StringRef ownership = ownershipSpelling(indexed->program.getOwnership());
  if (ownership.empty())
    return indexed->program.emitOpError(
        "has no Triton ownership projection");
  builder.create<plan::ProgramOp>(
      indexed->program.getLoc(), indexed->program.getLoopNodeAttr(),
      indexed->program.getWorkerAxesAttr(), gpu::stringAttr(builder, ownership),
      indexed->program.getTraversalsAttr());
  for (intent::plan::StorageOp storage : indexed->storage)
    builder.create<plan::StorageOp>(
        storage.getLoc(), storage.getValueAttr(),
        gpu::stringAttr(builder, "global"));
  for (intent::plan::TransferOp transfer : indexed->transfers)
    builder.create<plan::BoundaryOp>(
        transfer.getLoc(), transfer.getNodeAttr(), transfer.getDomainNodesAttr(),
        gpu::stringAttr(builder, transfer.getDomainNodes().empty()
                                    ? "none"
                                    : "index_lt_extent"),
        transfer.getFillAttr(), gpu::stringAttr(builder, "predicate"),
        transfer.getDeferAttr());
  for (intent::plan::ReductionOp reduction : indexed->reductions) {
    StringRef lowering = reduction.getRole() == "reduce_maximum"
                             ? StringRef("tl.max")
                             : reduction.getRole() == "reduce_add"
                                   ? StringRef("tl.sum")
                                   : StringRef();
    if (lowering.empty())
      return reduction.emitOpError("has no Triton reduction projection");
    builder.create<plan::ReductionOp>(
        reduction.getLoc(), reduction.getNodeAttr(),
        gpu::stringAttr(builder, lowering), reduction.getAxisAttr());
  }
  for (intent::plan::PointwiseOp pointwise : indexed->pointwise) {
    FailureOr<StringRef> lowering =
        pointwiseSpelling(pointwise, pointwise.getRole());
    if (failed(lowering))
      return failure();
    builder.create<plan::PointwiseOp>(
        pointwise.getLoc(), pointwise.getNodeAttr(),
        gpu::stringAttr(builder, *lowering), pointwise.getDeferAttr());
  }
  for (intent::plan::ContractOp contract : indexed->contracts)
    builder.create<plan::ContractOp>(
        contract.getLoc(), contract.getNodeAttr(),
        gpu::stringAttr(builder, "tl.dot"), contract.getAccumulatorTypeAttr(),
        contract.getLhsTransposeAttr(), contract.getRhsTransposeAttr());
  for (intent::plan::StreamOp stream : indexed->streams) {
    FailureOr<StringRef> tile =
        tileSpelling(stream, stream.getTile());
    if (failed(tile))
      return failure();
    builder.create<plan::StreamOp>(
        stream.getLoc(), stream.getNodeAttr(), stream.getAxisNodeAttr(),
        stream.getStopNodeAttr(), gpu::stringAttr(builder, *tile),
        stream.getOrderAttr(),
        gpu::stringAttr(builder, "register"));
  }
  for (intent::plan::RaggedOp ragged : indexed->ragged)
    builder.create<plan::RaggedOp>(
        ragged.getLoc(), ragged.getNodeAttr(), ragged.getOuterNodeAttr(),
        ragged.getMemberNodesAttr(), ragged.getTraversalAttr());
  for (intent::plan::StageOp stage : indexed->stages)
    builder.create<plan::StageOp>(
        stage.getLoc(), stage.getOrdinalAttr(), stage.getNodeAttr(),
        stage.getInputsAttr(), stage.getOutputsAttr());
  for (intent::plan::AtomicOp atomic : indexed->atomics)
    builder.create<plan::AtomicOp>(atomic.getLoc(), atomic.getNodeAttr(),
                                   gpu::stringAttr(builder, "tl.atomic_add"),
                                   gpu::stringAttr(builder, "gpu"));
  return plan::verifyTritonRealization(realization);
}

LogicalResult projectSearch(intent::plan::SearchSpaceOp searchSpace) {
  if (!searchSpace)
    return success();
  FailureOr<intent::plan::AutotuneOp> machine =
      gpu::indexMachineSearch(searchSpace);
  if (failed(machine))
    return failure();
  OpBuilder builder(searchSpace.getContext());
  builder.setInsertionPoint(searchSpace.getBody().front().getTerminator());
  SmallVector<NamedAttribute> mappings;
  for (Attribute attribute : machine->getParameters()) {
    StringRef role = cast<StringAttr>(attribute).getValue();
    FailureOr<StringRef> spelling = parameterSpelling(*machine, role);
    if (failed(spelling))
      return failure();
    mappings.push_back(builder.getNamedAttr(*spelling,
                                            gpu::stringAttr(builder, role)));
  }
  builder.create<plan::AutotuneOp>(
      searchSpace.getLoc(), machine->getKeyAttr(),
      builder.getDictionaryAttr(mappings));
  return plan::verifyTritonSearchSpace(searchSpace);
}

} // namespace

LogicalResult projectSurface(ModuleOp module) {
  SmallVector<intent::plan::RealizationOp> realizations(
      module.getOps<intent::plan::RealizationOp>());
  SmallVector<intent::plan::SearchSpaceOp> searches(
      module.getOps<intent::plan::SearchSpaceOp>());
  if (realizations.size() != 1 || searches.size() > 1)
    return module.emitError(
        "Triton projection requires one GPU plan and at most one search space");
  if (failed(projectRealization(realizations.front())))
    return failure();
  return projectSearch(searches.empty() ? intent::plan::SearchSpaceOp()
                                        : searches.front());
}

} // namespace intent::triton
