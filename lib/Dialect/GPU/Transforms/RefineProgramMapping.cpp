#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"

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

LogicalResult realizeGroupedContractionMapping(
    ModuleOp module, func::FuncOp kernel, DelinearizeOp mapping,
    DenseI64ArrayAttr roles) {
  const int64_t contractionM =
      static_cast<int64_t>(CoordinateRole::ContractionM);
  const int64_t contractionN =
      static_cast<int64_t>(CoordinateRole::ContractionN);
  std::optional<unsigned> rowAxis;
  std::optional<unsigned> columnAxis;
  for (auto [axis, role] : llvm::enumerate(roles.asArrayRef())) {
    if (role == contractionM) {
      if (rowAxis)
        return mapping.emitOpError(
            "grouped contraction mapping requires one M coordinate");
      rowAxis = axis;
    }
    if (role == contractionN) {
      if (columnAxis)
        return mapping.emitOpError(
            "grouped contraction mapping requires one N coordinate");
      columnAxis = axis;
    }
  }
  if (!rowAxis || !columnAxis)
    return success();

  OpBuilder parameterBuilder(&kernel.getBody().front(),
                             kernel.getBody().front().begin());
  auto schema = ParameterAttr::get(
      module.getContext(), parameterBuilder.getStringAttr("GROUP_SIZE_M"),
      static_cast<uint32_t>(ParameterRole::TraversalGroup),
      DenseI64ArrayAttr::get(module.getContext(), {1, 2, 4, 8}));
  Value groupSize = parameterBuilder.create<ParameterOp>(
      mapping.getLoc(), parameterBuilder.getIndexType(), schema);

  OpBuilder builder(mapping);
  builder.setInsertionPointAfter(mapping);
  llvm::SmallPtrSet<Operation *, 16> swizzleOperations;
  auto binary = [&](Value lhs, Value rhs, uint64_t kind) {
    auto operation = builder.create<BinaryOp>(mapping.getLoc(),
                                              builder.getIndexType(), lhs, rhs,
                                              kind);
    swizzleOperations.insert(operation.getOperation());
    return operation.getResult();
  };
  Value rowCount = mapping.getExtents()[*rowAxis];
  Value columnCount = mapping.getExtents()[*columnAxis];
  Value row = mapping.getCoordinates()[*rowAxis];
  Value column = mapping.getCoordinates()[*columnAxis];
  Value linear = binary(binary(row, columnCount, 2), column, 0);
  Value programsPerGroup = binary(groupSize, columnCount, 2);
  Value group = binary(linear, programsPerGroup, 4);
  Value firstRow = binary(group, groupSize, 2);
  Value liveRows = binary(rowCount, firstRow, 1);
  Value activeGroupSize = binary(liveRows, groupSize, 10);
  Value groupOffset = binary(linear, programsPerGroup, 5);
  Value groupedRow =
      binary(firstRow, binary(groupOffset, activeGroupSize, 5), 0);
  Value groupedColumn = binary(groupOffset, activeGroupSize, 4);

  auto replaceExternalUses = [&](Value source, Value replacement) {
    for (OpOperand &use : llvm::make_early_inc_range(source.getUses()))
      if (!swizzleOperations.contains(use.getOwner()))
        use.set(replacement);
  };
  replaceExternalUses(row, groupedRow);
  replaceExternalUses(column, groupedColumn);
  return success();
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
  if (failed(realizeGroupedContractionMapping(module, kernel, mapping, roles)))
    return failure();
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
