#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

PhysicalExprAttr parameterExpression(MLIRContext *context, StringRef name) {
  return PhysicalExprAttr::get(
      context, static_cast<uint32_t>(PhysicalExprKind::Parameter), 0,
      StringAttr::get(context, name), ArrayAttr::get(context, {}));
}

void preserveBoundedTileOrigins(func::FuncOp kernel, DelinearizeOp mapping) {
  kernel.walk([&](MakeRangeOp range) {
    if (range->hasAttr(sourceSubregionAttr) || !isUnitStepRange(range))
      return;
    auto start = range.getLogicalStart().getDefiningOp<arith::ConstantOp>();
    auto startValue =
        start ? dyn_cast<IntegerAttr>(start.getValue()) : IntegerAttr();
    auto block = range.getExtent().getDefiningOp<ParameterOp>();
    FailureOr<int64_t> dimension = queryRangeDimension(range);
    if (!startValue || startValue.getInt() != 0 || !block || failed(dimension) ||
        llvm::any_of(block.getParameter().getCandidates().asArrayRef(),
                     [](int64_t value) { return value <= 0; }))
      return;
    for (auto [axis, coordinate] : llvm::enumerate(mapping.getCoordinates())) {
      Attribute launchExtent = mapping.getLaunchExtents()[axis];
      FailureOr<uint64_t> mappedDimension = blockedDimension(launchExtent);
      auto extent = dyn_cast<PhysicalExprAttr>(launchExtent);
      if (failed(mappedDimension) ||
          *mappedDimension != static_cast<uint64_t>(*dimension) ||
          !extent || extent.getOperands().size() != 2)
        continue;
      auto divisor = dyn_cast<PhysicalExprAttr>(extent.getOperands()[1]);
      if (!divisor || divisor.getKind() !=
                          static_cast<uint32_t>(PhysicalExprKind::Parameter) ||
          divisor.getSymbol() != block.getParameter().getName())
        continue;
      for (Operation *user : coordinate.getUsers()) {
        auto product = dyn_cast<BinaryOp>(user);
        if (!product || product.getOperatorKind() != BinaryOperator::Multiply)
          continue;
        bool scaledCoordinate =
            (product.getLhs() == coordinate &&
             product.getRhs() == range.getExtent()) ||
            (product.getRhs() == coordinate &&
             product.getLhs() == range.getExtent());
        if (scaledCoordinate &&
            samePhysicalScalarExpression(range.getStart(), product.getResult())) {
          range->setAttr(programBoundedOriginAttr,
                         UnitAttr::get(kernel.getContext()));
          break;
        }
      }
    }
  });
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

  // The permutation preserves the rectangular tile domain, including its last
  // partial tile. Keep exact origin bounds before replacing coordinate uses.
  preserveBoundedTileOrigins(kernel, mapping);

  OpBuilder parameterBuilder(&kernel.getBody().front(),
                             kernel.getBody().front().begin());
  auto schema = ParameterAttr::get(
      module.getContext(), parameterBuilder.getStringAttr("GROUP_SIZE_M"),
      static_cast<uint32_t>(ParameterRole::TraversalGroup),
      static_cast<uint32_t>(ParameterCategory::Execution),
      /*elementBitWidth=*/0,
      DenseI64ArrayAttr::get(module.getContext(), {1, 2, 4, 8}));
  Value groupSize = parameterBuilder.create<ParameterOp>(
      mapping.getLoc(), parameterBuilder.getIndexType(), schema);

  OpBuilder builder(mapping);
  builder.setInsertionPointAfter(mapping);
  llvm::SmallPtrSet<Operation *, 16> swizzleOperations;
  auto binary = [&](Value lhs, Value rhs, BinaryOperator kind) {
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
  Value group = binary(row, groupSize, BinaryOperator::FloorDivide);
  Value firstRow = binary(group, groupSize, BinaryOperator::Multiply);
  Value liveRows = binary(rowCount, firstRow, BinaryOperator::Subtract);
  Value activeGroupSize =
      binary(liveRows, groupSize, BinaryOperator::MinimumNum);
  Value rowInGroup = binary(row, groupSize, BinaryOperator::Remainder);
  Value groupOffset = binary(
      binary(rowInGroup, columnCount, BinaryOperator::Multiply), column,
      BinaryOperator::Add);
  Value groupedRow =
      binary(firstRow,
             binary(groupOffset, activeGroupSize, BinaryOperator::Remainder),
             BinaryOperator::Add);
  Value groupedColumn =
      binary(groupOffset, activeGroupSize, BinaryOperator::FloorDivide);

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
  const int64_t workset = static_cast<int64_t>(CoordinateRole::Workset);
  const int64_t ownership =
      static_cast<int64_t>(CoordinateRole::PointwiseOwnership);
  const int64_t contractionM =
      static_cast<int64_t>(CoordinateRole::ContractionM);
  const int64_t contractionN =
      static_cast<int64_t>(CoordinateRole::ContractionN);
  SmallVector<unsigned> traversalAxes;
  unsigned outerAxes = 0;
  unsigned contractionMAxes = 0;
  unsigned contractionNAxes = 0;
  bool onlyBatchedContractionRoles = true;
  for (auto [axis, role] : llvm::enumerate(roles.asArrayRef()))
    if (role == traversalWorker) {
      traversalAxes.push_back(axis);
    } else {
      outerAxes += role == workset || role == ownership;
      contractionMAxes += role == contractionM;
      contractionNAxes += role == contractionN;
      onlyBatchedContractionRoles &=
          role == workset || role == ownership ||
          role == contractionM || role == contractionN;
    }
  bool batchedContraction = traversalAxes.empty() &&
                            onlyBatchedContractionRoles && outerAxes > 0 &&
                            contractionMAxes == 1 && contractionNAxes == 1;
  if (traversalAxes.empty() && !batchedContraction)
    return success();

  auto program = mapping.getLinear().getDefiningOp<ProgramIdOp>();
  if (!program || program.getAxis() != 0)
    return mapping.emitOpError(
        "persistent traversal requires one linear program coordinate");
  if (!program.getResult().hasOneUse())
    return success();
  for (Operation &operation : kernel.getBody().front()) {
    if (&operation == mapping.getOperation())
      break;
    if (!isMemoryEffectFree(&operation))
      return success();
  }
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
  int64_t residentCount = capabilities.getComputeUnits();
  // Independent contraction tiles need enough resident programs to occupy two
  // workers per compute unit while the grid-stride loop preserves exact task
  // coverage.  Ordered traversal keeps its existing one-worker policy.
  if (batchedContraction)
    residentCount *= 2;
  OpBuilder parameterBuilder(&kernel.getBody().front(),
                             kernel.getBody().front().begin());
  auto residentSchema = ParameterAttr::get(
      module.getContext(),
      parameterBuilder.getStringAttr("RESIDENT_WORKERS"),
      static_cast<uint32_t>(ParameterRole::ResidentWorkers),
      static_cast<uint32_t>(ParameterCategory::Execution),
      /*elementBitWidth=*/0,
      DenseI64ArrayAttr::get(module.getContext(), {residentCount}));
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
  Value totalTasks = builder.create<PhysicalExprOp>(
      mapping.getLoc(), builder.getIndexType(), segmentLength);
  auto loop = builder.create<scf::ForOp>(
      mapping.getLoc(), program.getResult(), totalTasks,
      residentWorkers.getResult());
  mapping->setOperand(0, loop.getInductionVar());
  Operation *yield = loop.getBody()->getTerminator();
  for (Operation *operation : taskBody)
    operation->moveBefore(yield);

  PhysicalExprAttr residentExtent = parameterExpression(
      module.getContext(), residentSchema.getName().getValue());
  auto boundedResidentExtent = PhysicalExprAttr::get(
      module.getContext(), static_cast<uint32_t>(PhysicalExprKind::Minimum), 0,
      StringAttr::get(module.getContext()),
      ArrayAttr::get(module.getContext(), {segmentLength, residentExtent}));
  kernel->setAttr(programSpaceAttr,
                  ArrayAttr::get(module.getContext(), {boundedResidentExtent}));
  kernel->setAttr(gridRankAttr,
                  IntegerAttr::get(IntegerType::get(module.getContext(), 64), 1));
  return success();
}

} // namespace intent::gpu
