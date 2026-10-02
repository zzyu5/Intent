#include "Intent/Dialect/GPU/Transforms/ProgramInterface.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Storage.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"

using namespace mlir;

namespace intent::gpu {

LogicalResult materializeProgramBuffers(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<BufferOp> buffers;
  kernel.walk([&](BufferOp buffer) { buffers.push_back(buffer); });
  if (buffers.empty())
    return success();
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!space || space.empty())
    return kernel.emitError("mutable buffers require a physical program space");
  bool singleton = llvm::all_of(space, [](Attribute attribute) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    return extent.getKind() ==
               PhysicalExprKind::Constant &&
           extent.getValue() == 1;
  });
  SmallVector<Value> programCoordinates;
  if (!singleton) {
    OpBuilder entry(&kernel.front(), kernel.front().begin());
    for (auto [axis, attribute] : llvm::enumerate(space)) {
      auto extent = cast<PhysicalExprAttr>(attribute);
      auto kind = extent.getKind();
      if (kind != PhysicalExprKind::Constant &&
          kind != PhysicalExprKind::Dimension)
        return kernel.emitError(
            "private buffer workspace requires constant or ABI dimension program extents");
      ProgramIdOp coordinate;
      kernel.walk([&](ProgramIdOp existing) {
        if (existing.getAxis() == axis)
          coordinate = existing;
      });
      if (coordinate) {
        if (coordinate.getOperation() != &kernel.front().front())
          coordinate->moveBefore(&kernel.front(), kernel.front().begin());
      } else {
        entry.setInsertionPointToStart(&kernel.front());
        coordinate = entry.create<ProgramIdOp>(
            kernel.getLoc(), entry.getIndexType(), axis);
      }
      programCoordinates.push_back(coordinate.getResult());
    }
  }
  for (BufferOp buffer : buffers) {
    auto type = buffer.getResult().getType();
    if (type.getScope().getValue() != BufferScope::ProgramPrivate)
      return buffer.emitOpError(
          "mutable buffer requires program-private storage with program lifetime");
    if (buffer.getInitialValue())
      return buffer.emitOpError("buffer initialization must be lowered to explicit writes");
    for (Operation *user : buffer.getResult().getUsers())
      if (!isa<LoadOp, StoreOp, AssumeInBoundsOp>(user))
        return user->emitOpError("mutable buffer supports explicit loads and stores");
    for (Attribute attribute : type.getShape()) {
      auto extent = cast<PhysicalExprAttr>(attribute);
      auto kind = extent.getKind();
      if ((kind != PhysicalExprKind::Constant &&
           kind != PhysicalExprKind::Dimension) || extent.getValue() <= 0)
        return buffer.emitOpError(
            "mutable buffer requires positive constants or ABI dimensions");
    }

    SmallVector<Attribute> shape;
    if (!singleton)
      llvm::append_range(shape, space);
    llvm::append_range(shape, type.getShape());
    // Only storage is allocated at entry; lexical initialization and accesses
    // retain their original control and order, including across iterations.
    Value workspace = createInvocationWorkspace(
        kernel, buffer.getLoc(), type.getElementType(),
        ArrayAttr::get(kernel.getContext(), shape), type.getOwner());
    // Program-private storage stays disjoint when the launch has many programs.
    // The grid coordinates are explicit prefix indices of the existing workspace.
    unsigned prefixRank = programCoordinates.size();
    auto prefixAccess = [&](AccessOpInterface access) {
      SmallVector<Value> coordinates(programCoordinates);
      llvm::append_range(coordinates, access.getAccessCoordinates());
      SmallVector<int64_t> axes;
      for (unsigned axis = 0; axis < prefixRank; ++axis)
        axes.push_back(axis);
      for (int64_t axis : access.getAccessSourceAxes())
        axes.push_back(axis + prefixRank);
      access.getAccessCoordinatesMutable().assign(coordinates);
      access.setAccessSourceAxes(axes);
    };
    for (Operation *user : llvm::make_early_inc_range(buffer.getResult().getUsers())) {
      if (auto access = dyn_cast<AccessOpInterface>(user))
        prefixAccess(access);
      else if (auto bounds = dyn_cast<AssumeInBoundsOp>(user))
        bounds.setAxis(bounds.getAxis() + prefixRank);
    }
    buffer.getResult().replaceAllUsesWith(workspace);
    OpBuilder builder(buffer);
    for (auto [axis, coordinate] : llvm::enumerate(programCoordinates))
      builder.create<AssumeInBoundsOp>(buffer.getLoc(), coordinate,
                                           workspace, axis);
    buffer.erase();
  }
  eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult lowerInvocationWorkspaces(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<BlockArgument> workspaces;
  for (BlockArgument argument : kernel.getArguments()) {
    if (isa<BufferType>(argument.getType()))
      workspaces.push_back(argument);
  }
  if (workspaces.empty())
    return success();
  uint64_t nextSource = nextPhysicalAxisIdentities(kernel).first;
  OpBuilder builder(kernel.getContext());
  for (BlockArgument workspace : workspaces) {
    auto buffer = cast<BufferType>(workspace.getType());
    if (!buffer.isInvocationWorkspace() ||
        buffer.getInitialization().getValue() !=
            BufferInitialization::FirstWrite)
      return kernel.emitError(
          "requires an invocation workspace with explicit first writes");

    for (Operation *user : workspace.getUsers()) {
      if (isa<DimOp, AssumeInBoundsOp>(user))
        continue;
      if (!isa<LoadOp, StoreOp>(user))
        return user->emitOpError(
            "workspace supports explicit loads and stores");
    }
    SmallVector<int64_t> dimensions;
    SmallVector<Attribute> strides;
    for (auto [axis, attribute] : llvm::enumerate(buffer.getShape())) {
      auto extent = cast<PhysicalExprAttr>(attribute);
      auto kind = extent.getKind();
      if (kind != PhysicalExprKind::Constant &&
          kind != PhysicalExprKind::Dimension)
        return kernel.emitError(
            "workspace shape requires constant or ABI dimension extents");
      dimensions.push_back(kind == PhysicalExprKind::Dimension
                               ? extent.getValue()
                               : 0);
      auto binding = ArgumentBindingAttr::get(kernel.getContext(),
          nextArgumentReference(kernel), ArgumentKind::Stride, IntegerAttr{},
          getArgumentReference(workspace), builder.getI64IntegerAttr(axis), IntegerAttr{});
      auto argument = appendArgument(kernel, builder.getIndexType(), binding);
      if (failed(argument)) return failure();
      strides.push_back(queryArgumentExpression(*argument));
    }
    auto layout = ViewLayoutAttr::get(
        kernel.getContext(), buffer.getShape(),
        builder.getDenseI64ArrayAttr(dimensions),
        builder.getArrayAttr(strides));
    if (failed(setArgumentType(workspace, ViewType::get(
        kernel.getContext(), buffer.getElementType(),
        /*access=*/2, nextSource++, layout)))) return failure();
  }
  return success();
}

} // namespace intent::gpu
