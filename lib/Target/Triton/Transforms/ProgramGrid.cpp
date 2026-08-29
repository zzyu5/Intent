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
  if (roles && roles.size() != rank)
    return mapping.emitOpError(
        "Triton program-grid legalization requires one role per coordinate");
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

  // Preserve the shared DelinearizeOp as the execution-group carrier.  The
  // provider grid only changes how the original row-major linear program id is
  // obtained: reconstruct that id from the permuted Triton grid coordinates,
  // then let the existing mapping continue to define runtime workset
  // coordinates, segment coverage and all mapping attributes.
  Value linear = coordinates.front();
  for (unsigned axis = 1; axis < rank; ++axis) {
    linear = builder.create<gpu::BinaryOp>(
        mapping.getLoc(), builder.getIndexType(), linear,
        mapping.getExtents()[axis],
        BinaryOperator::Multiply);
    linear = builder.create<gpu::BinaryOp>(
        mapping.getLoc(), builder.getIndexType(), linear, coordinates[axis],
        BinaryOperator::Add);
  }
  mapping->setOperand(0, linear);
  if (linearProgram->use_empty())
    linearProgram.erase();
  return success();
}

} // namespace intent::triton
