#include "Intent/Target/Triton/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"

using namespace mlir;

namespace intent::triton {

LogicalResult legalizeProgramGrid(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = gpu::getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<gpu::DelinearizeOp> mappings;
  kernel.walk(
      [&](gpu::DelinearizeOp mapping) { mappings.push_back(mapping); });
  if (mappings.size() != 1)
    return success();

  gpu::DelinearizeOp mapping = mappings.front();
  if (mapping.getCoordinates().empty() ||
      mapping.getCoordinates().size() > 3)
    return success();
  auto linearProgram = mapping.getLinear().getDefiningOp<gpu::ProgramIdOp>();
  if (!linearProgram || linearProgram.getAxis() != 0)
    return success();

  const unsigned rank = mapping.getCoordinates().size();
  SmallVector<unsigned> programOrder;
  auto roles =
      mapping->getAttrOfType<DenseI64ArrayAttr>(gpu::coordinateRolesAttr);
  const int64_t workset =
      static_cast<int64_t>(gpu::CoordinateRole::Workset);
  const int64_t pointwiseOwnership =
      static_cast<int64_t>(gpu::CoordinateRole::PointwiseOwnership);
  const int64_t tiledWorkset =
      static_cast<int64_t>(gpu::CoordinateRole::TiledWorkset);
  const int64_t contractionM =
      static_cast<int64_t>(gpu::CoordinateRole::ContractionM);
  const int64_t contractionN =
      static_cast<int64_t>(gpu::CoordinateRole::ContractionN);
  const int64_t traversalWorker =
      static_cast<int64_t>(gpu::CoordinateRole::TraversalWorker);
  const int64_t indirectTraversal =
      static_cast<int64_t>(gpu::CoordinateRole::IndirectTraversal);
  const bool hasRowOwnership =
      roles && llvm::is_contained(roles.asArrayRef(), contractionM);
  const bool hasWorkerTraversal =
      roles && llvm::is_contained(roles.asArrayRef(), traversalWorker);
  const bool hasIndirectTraversal =
      roles && llvm::is_contained(roles.asArrayRef(), indirectTraversal);
  const bool hasTiledWorkset =
      roles && llvm::is_contained(roles.asArrayRef(), tiledWorkset);

  if (hasWorkerTraversal) {
    for (unsigned axis = 0; axis < rank; ++axis)
      if (roles[axis] == traversalWorker)
        programOrder.push_back(axis);
    for (unsigned axis = 0; axis < rank; ++axis)
      if (roles[axis] == contractionN)
        programOrder.push_back(axis);
    for (unsigned axis = 0; axis < rank; ++axis)
      if (!llvm::is_contained(programOrder, axis))
        programOrder.push_back(axis);
  } else if (hasRowOwnership || !roles) {
    for (unsigned axis = 0; axis < rank; ++axis)
      programOrder.push_back(axis);
  } else if (hasIndirectTraversal) {
    for (unsigned axis = 0; axis < rank; ++axis)
      if (roles[axis] == pointwiseOwnership)
        programOrder.push_back(axis);
    for (unsigned axis = 0; axis < rank; ++axis)
      if (roles[axis] == indirectTraversal)
        programOrder.push_back(axis);
    for (unsigned axis = 0; axis < rank; ++axis)
      if (!llvm::is_contained(programOrder, axis))
        programOrder.push_back(axis);
  } else if (hasTiledWorkset) {
    for (unsigned axis = rank; axis > 0; --axis)
      if (roles[axis - 1] == tiledWorkset)
        programOrder.push_back(axis - 1);
    for (unsigned axis = 0; axis < rank; ++axis)
      if (!llvm::is_contained(programOrder, axis))
        programOrder.push_back(axis);
  } else {
    for (unsigned axis = 0; axis < rank; ++axis)
      if (roles[axis] == workset)
        programOrder.push_back(axis);
    for (unsigned axis = 0; axis < rank; ++axis)
      if (roles[axis] == pointwiseOwnership)
        programOrder.push_back(axis);
    for (unsigned axis = 0; axis < rank; ++axis)
      if (!llvm::is_contained(programOrder, axis))
        programOrder.push_back(axis);
  }

  // The shared mapping records which coordinates own fragments and which
  // remain program-internal traversal drivers.  Triton puts pointwise
  // fragment ownership before an unowned driver, preserves the established
  // row-major order when all coordinates own output, and keeps ragged
  // contraction workers in their explicit worker order.  No access or
  // structured operation is rebuilt here.
  SmallVector<Attribute> gridExtents;
  for (unsigned coordinateAxis : programOrder)
    gridExtents.push_back(mapping.getLaunchExtents()[coordinateAxis]);
  kernel->setAttr(gpu::programSpaceAttr,
                  ArrayAttr::get(module.getContext(), gridExtents));
  kernel->setAttr(
      gpu::gridRankAttr,
      IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                       mapping.getCoordinates().size()));
  OpBuilder builder(mapping);
  SmallVector<unsigned> coordinateToProgram(rank);
  for (auto [programAxis, coordinateAxis] : llvm::enumerate(programOrder))
    coordinateToProgram[coordinateAxis] = programAxis;
  SmallVector<Value> coordinates(rank);
  for (unsigned axis = 0; axis < rank; ++axis)
    coordinates[axis] = builder.create<gpu::ProgramIdOp>(
        mapping.getLoc(), builder.getIndexType(), coordinateToProgram[axis]);

  if (hasRowOwnership && roles) {
    std::optional<unsigned> rowAxis;
    std::optional<unsigned> columnAxis;
    for (unsigned axis = 0; axis < rank; ++axis) {
      if (roles[axis] == contractionM)
        rowAxis = axis;
      if (roles[axis] == contractionN)
        columnAxis = axis;
    }
    if (rowAxis && columnAxis) {
      OpBuilder parameterBuilder(&kernel.getBody().front(),
                                 kernel.getBody().front().begin());
      auto schema = gpu::ParameterAttr::get(
          module.getContext(), parameterBuilder.getStringAttr("GROUP_SIZE_M"),
          static_cast<uint32_t>(gpu::ParameterRole::TraversalGroup),
          DenseI64ArrayAttr::get(module.getContext(), {1, 2, 4, 8}));
      Value groupSize = parameterBuilder.create<gpu::ParameterOp>(
          mapping.getLoc(), parameterBuilder.getIndexType(), schema);
      auto binary = [&](Value lhs, Value rhs, uint64_t kind) {
        return builder.create<gpu::BinaryOp>(mapping.getLoc(),
                                             builder.getIndexType(), lhs, rhs,
                                             kind);
      };
      Value rowCount = mapping.getExtents()[*rowAxis];
      Value columnCount = mapping.getExtents()[*columnAxis];
      Value linear = binary(coordinates[*rowAxis],
                            binary(coordinates[*columnAxis], rowCount, 2), 0);
      Value programsPerGroup = binary(groupSize, columnCount, 2);
      Value group = binary(linear, programsPerGroup, 4);
      Value firstRow = binary(group, groupSize, 2);
      Value liveRows = binary(rowCount, firstRow, 1);
      Value activeGroupSize = binary(liveRows, groupSize, 10);
      Value groupOffset = binary(linear, programsPerGroup, 5);
      coordinates[*rowAxis] =
          binary(firstRow, binary(groupOffset, activeGroupSize, 5), 0);
      coordinates[*columnAxis] = binary(groupOffset, activeGroupSize, 4);
    }
  }
  for (auto [coordinate, replacement] :
       llvm::zip(mapping.getCoordinates(), coordinates))
    coordinate.replaceAllUsesWith(replacement);
  mapping.erase();
  if (linearProgram->use_empty())
    linearProgram.erase();
  return success();
}

} // namespace intent::triton
