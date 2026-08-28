#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace intent::gpu {
namespace {

PhysicalExprAttr parameterExpression(MLIRContext *context, StringRef name) {
  return PhysicalExprAttr::get(
      context, static_cast<uint32_t>(PhysicalExprKind::Parameter), 0,
      StringAttr::get(context, name), ArrayAttr::get(context, {}));
}

Value multiply(OpBuilder &builder, Location location, Value lhs, Value rhs) {
  return builder.create<BinaryOp>(location, builder.getIndexType(), lhs, rhs,
                                  /*operator_kind=*/2);
}

} // namespace

LogicalResult refineProgramMapping(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;

  SmallVector<DelinearizeOp> mappings;
  kernel.walk([&](DelinearizeOp mapping) { mappings.push_back(mapping); });
  if (mappings.size() != 1)
    return success();
  DelinearizeOp mapping = mappings.front();
  if (mapping->getBlock() != &kernel.getBody().front())
    return success();

  auto roles =
      mapping->getAttrOfType<DenseI64ArrayAttr>(coordinateRolesAttr);
  if (!roles || static_cast<size_t>(roles.size()) !=
                    mapping.getCoordinates().size())
    return success();
  const int64_t traversalWorker =
      static_cast<int64_t>(CoordinateRole::TraversalWorker);
  SmallVector<unsigned> traversalAxes;
  for (auto [axis, role] : llvm::enumerate(roles.asArrayRef()))
    if (role == traversalWorker)
      traversalAxes.push_back(axis);
  if (traversalAxes.empty())
    return success();

  auto program = mapping.getLinear().getDefiningOp<ProgramIdOp>();
  if (!program || program.getAxis() != 0)
    return mapping.emitOpError(
        "persistent traversal requires one linear program coordinate");
  auto programSpace = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  auto segmentOffset =
      mapping->getAttrOfType<PhysicalExprAttr>(segmentOffsetAttr);
  auto segmentLength =
      mapping->getAttrOfType<PhysicalExprAttr>(segmentLengthAttr);
  if (!programSpace || programSpace.size() != 1 || !segmentOffset ||
      !segmentLength ||
      segmentOffset.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Constant) ||
      segmentOffset.getValue() != 0 || programSpace[0] != segmentLength)
    return mapping.emitOpError(
        "persistent traversal requires one full-program execution segment");

  for (unsigned axis : traversalAxes) {
    auto parameter =
        mapping.getExtents()[axis].getDefiningOp<ParameterOp>();
    if (!parameter ||
        parameter.getParameter().getRole() !=
            static_cast<uint32_t>(ParameterRole::TraversalWorkers))
      return mapping.emitOpError(
          "traversal-worker coordinate lacks its physical parameter");
  }

  auto capabilities =
      kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getComputeUnits() <= 0)
    return kernel.emitError(
        "persistent traversal requires a positive compute-unit capability");
  OpBuilder parameterBuilder(&kernel.getBody().front(),
                             kernel.getBody().front().begin());
  auto residentSchema = ParameterAttr::get(
      module.getContext(),
      parameterBuilder.getStringAttr("RESIDENT_WORKERS"),
      static_cast<uint32_t>(ParameterRole::ResidentWorkers),
      DenseI64ArrayAttr::get(module.getContext(),
                             {capabilities.getComputeUnits()}));
  auto residentWorkers = parameterBuilder.create<ParameterOp>(
      mapping.getLoc(), parameterBuilder.getIndexType(), residentSchema);

  SmallVector<Operation *> taskBody;
  for (Operation *operation = mapping.getOperation();
       operation && !isa<func::ReturnOp>(operation);
       operation = operation->getNextNode())
    taskBody.push_back(operation);
  if (taskBody.empty())
    return mapping.emitOpError("persistent traversal has no physical task body");
  auto terminator =
      dyn_cast<func::ReturnOp>(kernel.getBody().front().getTerminator());
  if (!terminator || terminator.getNumOperands() != 0)
    return kernel.emitError(
        "persistent traversal requires a void physical kernel terminator");

  OpBuilder builder(mapping);
  Value totalTasks = builder.create<arith::ConstantIndexOp>(mapping.getLoc(), 1);
  for (Value extent : mapping.getExtents())
    totalTasks = multiply(builder, mapping.getLoc(), totalTasks, extent);
  auto loop = builder.create<scf::ForOp>(
      mapping.getLoc(), program.getResult(), totalTasks,
      residentWorkers.getResult());
  mapping->setOperand(0, loop.getInductionVar());
  Operation *yield = loop.getBody()->getTerminator();
  for (Operation *operation : taskBody)
    operation->moveBefore(yield);

  PhysicalExprAttr residentExtent = parameterExpression(
      module.getContext(), residentSchema.getName().getValue());
  kernel->setAttr(programSpaceAttr,
                  ArrayAttr::get(module.getContext(), {residentExtent}));
  kernel->setAttr(gridRankAttr,
                  IntegerAttr::get(IntegerType::get(module.getContext(), 64), 1));
  return success();
}

} // namespace intent::gpu
