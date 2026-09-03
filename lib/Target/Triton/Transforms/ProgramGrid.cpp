#include "Intent/Target/Triton/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "llvm/ADT/SmallPtrSet.h"

#include <optional>

using namespace mlir;

namespace intent::triton {

namespace {

bool dependsOn(Value value, Value root,
               llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (value == root)
    return true;
  Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return false;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return dependsOn(operand, root, visited);
  });
}

FailureOr<SmallVector<unsigned>>
writeEffectProgramOrder(func::FuncOp kernel, gpu::DelinearizeOp mapping) {
  const unsigned rank = mapping.getCoordinates().size();
  SmallVector<std::optional<int64_t>> resourceAxes(rank);
  bool sawStore = false;
  bool ambiguous = false;

  kernel.walk([&](gpu::StoreOp store) {
    sawStore = true;
    if (!isa<gpu::ViewType>(store.getResource().getType())) {
      ambiguous = true;
      return;
    }
    for (unsigned programAxis = 0; programAxis < rank; ++programAxis) {
      std::optional<int64_t> storeAxis;
      for (auto [coordinateIndex, coordinate] :
           llvm::enumerate(store.getCoordinates())) {
        llvm::SmallPtrSet<Operation *, 16> visited;
        if (!dependsOn(coordinate, mapping.getCoordinates()[programAxis],
                       visited))
          continue;
        int64_t sourceAxis = store.getSourceAxes()[coordinateIndex];
        if (storeAxis && *storeAxis != sourceAxis) {
          ambiguous = true;
          return;
        }
        storeAxis = sourceAxis;
      }
      if (!storeAxis) {
        ambiguous = true;
        return;
      }
      if (resourceAxes[programAxis] &&
          *resourceAxes[programAxis] != *storeAxis) {
        ambiguous = true;
        return;
      }
      resourceAxes[programAxis] = *storeAxis;
    }
  });

  bool hasOtherWrites = false;
  kernel.walk([&](Operation *operation) {
    hasOtherWrites |= isa<gpu::ScatterReduceOp, gpu::AtomicStoreOp,
                          gpu::AtomicRMWOp, gpu::AtomicCompareExchangeOp>(
        operation);
  });
  if (!sawStore || ambiguous || hasOtherWrites ||
      llvm::any_of(resourceAxes, [](const std::optional<int64_t> &axis) {
        return !axis.has_value();
      }))
    return failure();

  for (unsigned lhs = 0; lhs < rank; ++lhs)
    for (unsigned rhs = lhs + 1; rhs < rank; ++rhs)
      if (resourceAxes[lhs] == resourceAxes[rhs])
        return failure();

  SmallVector<unsigned> order;
  for (unsigned axis = 0; axis < rank; ++axis)
    order.push_back(axis);
  llvm::stable_sort(order, [&](unsigned lhs, unsigned rhs) {
    return *resourceAxes[lhs] < *resourceAxes[rhs];
  });
  return order;
}

} // namespace

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
  } else if (FailureOr<SmallVector<unsigned>> effectOrder =
                 writeEffectProgramOrder(kernel, mapping);
             succeeded(effectOrder)) {
    programOrder = std::move(*effectOrder);
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

  // Specialized traversal forms retain their typed shared ordering.  For an
  // ordinary workset, use the current write-coordinate graph only when every
  // external effect gives the same unambiguous resource-axis order; otherwise
  // preserve the shared order.  No access or structured operation is rebuilt.
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
