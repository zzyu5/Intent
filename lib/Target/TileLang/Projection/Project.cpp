#include "Intent/Target/TileLang/Projection/Project.h"

#include "Intent/Target/GPU/Projection/MachinePlan.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::tilelang {
namespace {

IntegerAttr i64(OpBuilder &builder, int64_t value) {
  return builder.getI64IntegerAttr(value);
}

FailureOr<StringRef> tileSpelling(Operation *operation, StringRef role) {
  if (role == "one")
    return StringRef("one");
  if (role == "row_vector")
    return StringRef("TILE_SIZE");
  if (role == "program_m" || role == "ragged_member" || role == "query")
    return StringRef("TILE_SIZE_M");
  if (role == "program_n" || role == "stream")
    return StringRef("TILE_SIZE_N");
  if (role == "reduction")
    return StringRef("TILE_SIZE_K");
  if (role == "reduction_extra")
    return StringRef("TILE_SIZE_REDUCTION");
  operation->emitOpError("has no TileLang tile spelling for role ") << role;
  return failure();
}

StringRef bufferSpace(StringRef space) {
  if (space == "external" || space == "workspace")
    return "global";
  if (space == "shared")
    return "shared";
  if (space == "private_fragment")
    return "fragment";
  if (space == "private_scalar")
    return "local";
  if (space == "none")
    return "none";
  return {};
}

FailureOr<StringRef> pointwiseSpelling(Operation *operation, StringRef role,
                                       StringRef materialization,
                                       bool orderedStream) {
  if (role == "indices")
    return StringRef("logical_indices");
  if (role == "broadcast")
    return StringRef("alias");
  if (role == "cast")
    return materialization == "contract_operand" ? StringRef("T.copy_cast")
                                                   : StringRef("T.cast");
  if (role == "unary_exp")
    return StringRef("T.exp");
  if (role == "unary_exp2")
    return StringRef("T.exp2");
  if (role == "unary_log")
    return StringRef("T.log");
  if (role == "unary_rsqrt")
    return StringRef("T.rsqrt");
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
    return StringRef("T.max");
  if (role == "compare_greater_equal")
    return StringRef("python_greater_equal");
  if (role == "mask")
    return StringRef("T.if_then_else");
  if (role == "full")
    return StringRef("T.fill");
  if (role == "zeros")
    return StringRef("T.clear");
  if (role == "members")
    return StringRef("T.members");
  if (role == "expand_dims")
    return orderedStream ? StringRef("alias_column")
                         : StringRef("expand_dims");
  if (role == "indirect_gather")
    return StringRef("T.indirect_gather");
  operation->emitOpError("has no TileLang pointwise spelling for role ") << role;
  return failure();
}

FailureOr<StringRef> parameterSpelling(Operation *operation, StringRef role) {
  if (role == "program_m" || role == "ragged_member" || role == "query")
    return StringRef("TILE_SIZE_M");
  if (role == "program_n" || role == "feature" || role == "stream")
    return StringRef("TILE_SIZE_N");
  if (role == "reduction")
    return StringRef("TILE_SIZE_K");
  if (role == "group_m")
    return StringRef("GROUP_SIZE_M");
  operation->emitOpError("has no TileLang tuner parameter for role ") << role;
  return failure();
}

bool hasTraversal(intent::plan::ProgramOp program, StringRef expected) {
  return llvm::any_of(program.getTraversals(), [&](Attribute attribute) {
    return cast<StringAttr>(attribute).getValue() == expected;
  });
}

StringRef ownershipSpelling(StringRef ownership) {
  if (ownership == "row")
    return "block_rows";
  if (ownership == "tiled")
    return "block_tiles";
  if (ownership == "ragged")
    return "block_ragged";
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
        "has no TileLang ownership projection");
  builder.create<plan::ProgramOp>(
      indexed->program.getLoc(), indexed->program.getLoopNodeAttr(),
      indexed->program.getWorkerAxesAttr(), gpu::stringAttr(builder, ownership),
      indexed->program.getTraversalsAttr());
  for (intent::plan::StorageOp storage : indexed->storage)
    builder.create<plan::StorageOp>(
        storage.getLoc(), storage.getValueAttr(),
        gpu::stringAttr(builder, bufferSpace(storage.getSpace())));
  for (intent::plan::PaddingOp padding : indexed->paddings)
    builder.create<plan::PaddingOp>(
        padding.getLoc(), padding.getValueAttr(), padding.getTensorAxesAttr(),
        padding.getDomainNodesAttr(), padding.getFillAttr(),
        padding.getMaterializationAttr());
  bool rowStrided = indexed->program.getOwnership() == "row" &&
                    hasTraversal(indexed->program, "persistent");
  bool raggedOrdered = indexed->program.getOwnership() == "ragged" &&
                       hasTraversal(indexed->program, "ordered_stream");
  bool elementwiseTransfer =
      (indexed->program.getOwnership() == "tiled" || raggedOrdered) &&
      hasTraversal(indexed->program, "ordered_stream");
  for (intent::plan::TransferOp transfer : indexed->transfers) {
    bool load = transfer.getAccess() == "load";
    StringRef access = rowStrided ? (load ? "gather" : "scatter")
                                  : (load ? "load" : "store");
    StringRef resultSpace = bufferSpace(transfer.getResultSpace());
    if (resultSpace.empty())
      return transfer.emitOpError("has no TileLang result-space projection");
    builder.create<plan::BoundaryOp>(
        transfer.getLoc(), transfer.getNodeAttr(), transfer.getDomainNodesAttr(),
        gpu::stringAttr(builder, access),
        gpu::stringAttr(builder,
                        elementwiseTransfer ? "parallel_elements" : "bulk_copy"),
        transfer.getFillAttr(),
        builder.getBoolAttr(rowStrided || raggedOrdered ||
                            (hasTraversal(indexed->program, "staged") &&
                             transfer.getAccess() == "store")),
        gpu::stringAttr(builder, resultSpace), transfer.getDeferAttr());
  }
  for (intent::plan::ReductionOp reduction : indexed->reductions) {
    StringRef lowering = reduction.getRole() == "reduce_maximum"
                             ? StringRef("T.reduce_max")
                             : reduction.getRole() == "reduce_add"
                                   ? StringRef("T.reduce_sum")
                                   : StringRef();
    if (lowering.empty())
      return reduction.emitOpError("has no TileLang reduction projection");
    builder.create<plan::ReductionOp>(
        reduction.getLoc(), reduction.getNodeAttr(),
        gpu::stringAttr(builder, lowering), reduction.getAxisAttr(),
        reduction.getKeepDimsAttr(), gpu::stringAttr(builder, "fragment"),
        gpu::stringAttr(builder, "fragment"));
  }
  for (intent::plan::PointwiseOp pointwise : indexed->pointwise) {
    FailureOr<StringRef> lowering = pointwiseSpelling(
        pointwise, pointwise.getRole(), pointwise.getMaterialization(),
        hasTraversal(indexed->program, "ordered_stream"));
    StringRef space = bufferSpace(pointwise.getResultSpace());
    if (failed(lowering) || space.empty())
      return failure();
    builder.create<plan::PointwiseOp>(
        pointwise.getLoc(), pointwise.getNodeAttr(),
        gpu::stringAttr(builder, *lowering), gpu::stringAttr(builder, space),
        pointwise.getReuseOperandAttr(), pointwise.getDeferAttr());
  }
  for (intent::plan::ContractOp contract : indexed->contracts) {
    StringRef lhsSpace = bufferSpace(contract.getLhsSpace());
    StringRef rhsSpace = bufferSpace(contract.getRhsSpace());
    StringRef accumulatorSpace = bufferSpace(contract.getAccumulatorSpace());
    if (lhsSpace.empty() || rhsSpace.empty() || accumulatorSpace.empty())
      return contract.emitOpError("has no TileLang matrix-space projection");
    builder.create<plan::ContractOp>(
        contract.getLoc(), contract.getNodeAttr(),
        gpu::stringAttr(builder, "T.gemm"), contract.getWarpPolicyAttr(),
        contract.getAccumulatorTypeAttr(), contract.getLhsTransposeAttr(),
        contract.getRhsTransposeAttr(), gpu::stringAttr(builder, lhsSpace),
        gpu::stringAttr(builder, rhsSpace),
        gpu::stringAttr(builder, accumulatorSpace));
  }
  for (intent::plan::StreamOp stream : indexed->streams) {
    FailureOr<StringRef> tile =
        tileSpelling(stream, stream.getTile());
    if (failed(tile))
      return failure();
    builder.create<plan::StreamOp>(
        stream.getLoc(), stream.getNodeAttr(), stream.getAxisNodeAttr(),
        stream.getStopNodeAttr(), gpu::stringAttr(builder, *tile),
        stream.getOrderAttr(),
        gpu::stringAttr(builder, "fragment"), stream.getReuseInitialAttr());
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
                                   gpu::stringAttr(builder, "T.atomic_add"));
  return plan::verifyTileLangRealization(realization);
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
  return plan::verifyTileLangSearchSpace(searchSpace);
}

} // namespace

LogicalResult projectSurface(ModuleOp module) {
  SmallVector<intent::plan::RealizationOp> realizations(
      module.getOps<intent::plan::RealizationOp>());
  SmallVector<intent::plan::SearchSpaceOp> searches(
      module.getOps<intent::plan::SearchSpaceOp>());
  if (realizations.size() != 1 || searches.size() > 1)
    return module.emitError(
        "TileLang projection requires one GPU plan and at most one search space");
  if (failed(projectRealization(realizations.front())))
    return failure();
  return projectSearch(searches.empty() ? intent::plan::SearchSpaceOp()
                                        : searches.front());
}

} // namespace intent::tilelang
