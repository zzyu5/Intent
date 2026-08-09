#include "Intent/Target/CuTile/Projection/Project.h"

#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "Intent/Target/GPU/Projection/MachinePlan.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::cutile {
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
  operation->emitOpError("has no cuTile tile spelling for role ") << role;
  return failure();
}

FailureOr<StringRef> pointwiseSpelling(Operation *operation, StringRef role,
                                       StringRef resultSpace,
                                       bool orderedStream) {
  if (role == "broadcast")
    return StringRef("alias");
  if (role == "cast")
    return resultSpace == "private_scalar" ? StringRef("ct.full_cast")
                                            : StringRef("ct.astype");
  if (role == "unary_exp")
    return StringRef("ct.exp");
  if (role == "unary_exp2")
    return StringRef("ct.exp2");
  if (role == "unary_log")
    return StringRef("ct.log");
  if (role == "unary_rsqrt")
    return StringRef("ct.rsqrt");
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
    return StringRef("ct.maximum");
  if (role == "full")
    return StringRef("ct.full");
  if (role == "zeros")
    return StringRef("ct.zeros");
  if (role == "members")
    return StringRef("ct.members");
  if (role == "expand_dims")
    return orderedStream ? StringRef("alias_column")
                         : StringRef("expand_dims");
  if (role == "indirect_gather")
    return StringRef("ct.indirect_gather");
  operation->emitOpError("has no cuTile pointwise spelling for role ") << role;
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
  operation->emitOpError("has no cuTile tuner parameter for role ") << role;
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
        "has no cuTile ownership projection");
  builder.create<plan::ProgramOp>(
      indexed->program.getLoc(), indexed->program.getLoopNodeAttr(),
      indexed->program.getWorkerAxesAttr(), gpu::stringAttr(builder, ownership),
      indexed->program.getTraversalsAttr());
  for (intent::plan::StorageOp storage : indexed->storage)
    builder.create<plan::StorageOp>(storage.getLoc(), storage.getValueAttr(),
                                    gpu::stringAttr(builder, "global"));
  bool rowStrided = indexed->program.getOwnership() == "row" &&
                    hasTraversal(indexed->program, "persistent");
  bool raggedOrdered = indexed->program.getOwnership() == "ragged" &&
                       hasTraversal(indexed->program, "ordered_stream");
  for (intent::plan::TransferOp transfer : indexed->transfers) {
    bool load = transfer.getAccess() == "load";
    bool vectorized = llvm::any_of(
        transfer.getDomainNodes(), [&](int64_t node) {
          auto axis = llvm::find_if(indexed->axes, [&](intent::plan::AxisOp value) {
            return value.getNodeAttr().getInt() == node;
          });
          return axis != indexed->axes.end() && axis->getTile() != "one";
        });
    StringRef access = (rowStrided || raggedOrdered) && vectorized
                           ? (load ? "gather" : "scatter")
                           : (load ? "load" : "store");
    builder.create<plan::BoundaryOp>(
        transfer.getLoc(), transfer.getNodeAttr(), transfer.getDomainNodesAttr(),
        gpu::stringAttr(builder, access), transfer.getFillAttr(),
        builder.getBoolAttr(rowStrided || raggedOrdered ||
                            (hasTraversal(indexed->program, "staged") &&
                             transfer.getAccess() == "store")),
        transfer.getDeferAttr());
  }
  for (intent::plan::ReductionOp reduction : indexed->reductions) {
    StringRef lowering = reduction.getRole() == "reduce_maximum"
                             ? StringRef("ct.max")
                             : reduction.getRole() == "reduce_add"
                                   ? StringRef("ct.sum")
                                   : StringRef();
    if (lowering.empty())
      return reduction.emitOpError("has no cuTile reduction projection");
    builder.create<plan::ReductionOp>(
        reduction.getLoc(), reduction.getNodeAttr(),
        gpu::stringAttr(builder, lowering), reduction.getAxisAttr(),
        reduction.getKeepDimsAttr());
  }
  for (intent::plan::PointwiseOp pointwise : indexed->pointwise) {
    FailureOr<StringRef> lowering =
        pointwiseSpelling(pointwise, pointwise.getRole(),
                          pointwise.getResultSpace(),
                          hasTraversal(indexed->program, "ordered_stream"));
    if (failed(lowering))
      return failure();
    builder.create<plan::PointwiseOp>(
        pointwise.getLoc(), pointwise.getNodeAttr(),
        gpu::stringAttr(builder, *lowering), pointwise.getDeferAttr());
  }
  for (intent::plan::ContractOp contract : indexed->contracts)
    builder.create<plan::ContractOp>(
        contract.getLoc(), contract.getNodeAttr(),
        gpu::stringAttr(builder, "ct.mma"), contract.getAccumulatorTypeAttr(),
        contract.getLhsTransposeAttr(), contract.getRhsTransposeAttr());
  for (intent::plan::StreamOp stream : indexed->streams) {
    FailureOr<StringRef> tile =
        tileSpelling(stream, stream.getTile());
    if (failed(tile))
      return failure();
    builder.create<plan::StreamOp>(
        stream.getLoc(), stream.getNodeAttr(), stream.getAxisNodeAttr(),
        gpu::stringAttr(builder, *tile), stream.getOrderAttr(),
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
    builder.create<plan::AtomicOp>(
        atomic.getLoc(), atomic.getNodeAttr(),
        gpu::stringAttr(builder, "ct.atomic_add"),
        gpu::stringAttr(builder, "relaxed"),
        gpu::stringAttr(builder, "device"));
  return plan::verifyCuTileRealization(realization);
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
  return plan::verifyCuTileSearchSpace(searchSpace);
}

} // namespace

LogicalResult projectSurface(ModuleOp module) {
  SmallVector<intent::plan::RealizationOp> realizations(
      module.getOps<intent::plan::RealizationOp>());
  SmallVector<intent::plan::SearchSpaceOp> searches(
      module.getOps<intent::plan::SearchSpaceOp>());
  if (realizations.size() != 1 || searches.size() > 1)
    return module.emitError(
        "cuTile projection requires one GPU plan and at most one search space");
  if (failed(projectRealization(realizations.front())))
    return failure();
  return projectSearch(searches.empty() ? intent::plan::SearchSpaceOp()
                                        : searches.front());
}

} // namespace intent::cutile
