#include "Intent/Target/Triton/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::triton {

LogicalResult materializeProgramBuffers(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = gpu::getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<gpu::BufferOp> buffers;
  kernel.walk([&](gpu::BufferOp buffer) { buffers.push_back(buffer); });
  if (buffers.empty())
    return success();
  auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
  if (!space || space.empty())
    return kernel.emitError("Triton mutable buffers require a physical program space");
  bool singleton = llvm::all_of(space, [](Attribute attribute) {
    auto extent = cast<gpu::PhysicalExprAttr>(attribute);
    return extent.getKind() ==
               static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) &&
           extent.getValue() == 1;
  });
  SmallVector<Value> programCoordinates;
  if (!singleton) {
    OpBuilder entry(&kernel.front(), kernel.front().begin());
    for (auto [axis, attribute] : llvm::enumerate(space)) {
      auto extent = cast<gpu::PhysicalExprAttr>(attribute);
      auto kind = static_cast<gpu::PhysicalExprKind>(extent.getKind());
      if (kind != gpu::PhysicalExprKind::Constant &&
          kind != gpu::PhysicalExprKind::Dimension)
        return kernel.emitError(
            "Triton private buffer workspace requires constant or ABI dimension program extents");
      gpu::ProgramIdOp coordinate;
      kernel.walk([&](gpu::ProgramIdOp existing) {
        if (existing.getAxis() == axis)
          coordinate = existing;
      });
      if (coordinate) {
        if (coordinate.getOperation() != &kernel.front().front())
          coordinate->moveBefore(&kernel.front(), kernel.front().begin());
      } else {
        entry.setInsertionPointToStart(&kernel.front());
        coordinate = entry.create<gpu::ProgramIdOp>(
            kernel.getLoc(), entry.getIndexType(), axis);
      }
      programCoordinates.push_back(coordinate.getResult());
    }
  }
  for (gpu::BufferOp buffer : buffers) {
    auto type = buffer.getResult().getType();
    if (type.getScope().getValue() != gpu::BufferScope::ProgramPrivate ||
        type.getLifetime().getValue() != gpu::BufferLifetime::Program ||
        buffer->getBlock() != &kernel.front())
      return buffer.emitOpError("Triton mutable buffer requires an entry program allocation");
    if (buffer.getInitialValue())
      return buffer.emitOpError("buffer initialization must be lowered to explicit writes");
    for (Operation *user : buffer.getResult().getUsers())
      if (!isa<gpu::LoadOp, gpu::StoreOp, gpu::AssumeInBoundsOp>(user))
        return user->emitOpError("Triton mutable buffer supports explicit loads and stores");
    for (Attribute attribute : type.getShape()) {
      auto extent = cast<gpu::PhysicalExprAttr>(attribute);
      auto kind = static_cast<gpu::PhysicalExprKind>(extent.getKind());
      if ((kind != gpu::PhysicalExprKind::Constant &&
           kind != gpu::PhysicalExprKind::Dimension) || extent.getValue() <= 0)
        return buffer.emitOpError(
            "Triton mutable buffer requires positive constants or ABI dimensions");
    }

    SmallVector<Attribute> shape;
    if (!singleton)
      llvm::append_range(shape, space);
    llvm::append_range(shape, type.getShape());
    Value workspace = gpu::createInvocationWorkspace(
        kernel, buffer.getLoc(), type.getElementType(),
        ArrayAttr::get(kernel.getContext(), shape), type.getOwner());
    // Program-private storage stays disjoint when the launch has many programs.
    // The grid coordinates are explicit prefix indices of the existing workspace.
    unsigned prefixRank = programCoordinates.size();
    auto prefixAccess = [&](auto access) {
      SmallVector<Value> coordinates(programCoordinates);
      llvm::append_range(coordinates, access.getCoordinates());
      SmallVector<int64_t> axes;
      for (unsigned axis = 0; axis < prefixRank; ++axis)
        axes.push_back(axis);
      for (int64_t axis : access.getSourceAxes())
        axes.push_back(axis + prefixRank);
      access.getCoordinatesMutable().assign(coordinates);
      access.setSourceAxes(axes);
    };
    for (Operation *user : llvm::make_early_inc_range(buffer.getResult().getUsers())) {
      if (auto load = dyn_cast<gpu::LoadOp>(user))
        prefixAccess(load);
      else if (auto store = dyn_cast<gpu::StoreOp>(user))
        prefixAccess(store);
      else if (auto bounds = dyn_cast<gpu::AssumeInBoundsOp>(user))
        bounds.setAxis(bounds.getAxis() + prefixRank);
    }
    buffer.getResult().replaceAllUsesWith(workspace);
    OpBuilder builder(buffer);
    for (auto [axis, coordinate] : llvm::enumerate(programCoordinates))
      builder.create<gpu::AssumeInBoundsOp>(buffer.getLoc(), coordinate,
                                           workspace, axis);
    buffer.erase();
  }
  gpu::eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult lowerInvocationWorkspaces(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = gpu::getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<BlockArgument> workspaces;
  llvm::StringSet<> argumentNames;
  for (BlockArgument argument : kernel.getArguments()) {
    argumentNames.insert(kernel.getArgAttrDict(argument.getArgNumber())
                             .getAs<StringAttr>(gpu::abiNameAttr).getValue());
    if (isa<gpu::BufferType>(argument.getType()))
      workspaces.push_back(argument);
  }
  if (workspaces.empty())
    return success();
  uint64_t nextSource = gpu::nextPhysicalAxisIdentities(kernel).first;
  OpBuilder builder(kernel.getContext());
  for (BlockArgument workspace : workspaces) {
    auto buffer = cast<gpu::BufferType>(workspace.getType());
    if (!buffer.getWorkspace() ||
        buffer.getScope().getValue() != gpu::BufferScope::InvocationWorkspace ||
        buffer.getInitialization().getValue() !=
            gpu::BufferInitialization::FirstWrite ||
        buffer.getLifetime().getValue() != gpu::BufferLifetime::Invocation)
      return kernel.emitError(
          "Triton requires an invocation workspace with explicit first writes");

    for (Operation *user : workspace.getUsers()) {
      if (isa<gpu::DimOp, gpu::AssumeInBoundsOp>(user))
        continue;
      if (!isa<gpu::LoadOp, gpu::StoreOp>(user))
        return user->emitOpError(
            "Triton workspace supports explicit loads and stores");
    }
    unsigned argumentIndex = workspace.getArgNumber();
    SmallVector<int64_t> dimensions;
    SmallVector<Attribute> strides;
    uint32_t abi = argumentIndex;
    for (auto [axis, attribute] : llvm::enumerate(buffer.getShape())) {
      auto extent = cast<gpu::PhysicalExprAttr>(attribute);
      auto kind = static_cast<gpu::PhysicalExprKind>(extent.getKind());
      if (kind != gpu::PhysicalExprKind::Constant &&
          kind != gpu::PhysicalExprKind::Dimension)
        return kernel.emitError(
            "Triton workspace shape requires constant or ABI dimension extents");
      dimensions.push_back(kind == gpu::PhysicalExprKind::Dimension
                               ? extent.getValue()
                               : 0);
      std::string name = ("WS" + Twine(abi) + "_" + Twine(axis)).str();
      while (!argumentNames.insert(name).second)
        name += "_";
      strides.push_back(builder.getStringAttr(name));
      kernel.insertArgument(
          kernel.getNumArguments(), builder.getIndexType(),
          builder.getDictionaryAttr({
              builder.getNamedAttr(gpu::abiKindAttr,
                                   builder.getStringAttr("stride")),
              builder.getNamedAttr(gpu::abiNameAttr, builder.getStringAttr(name)),
              builder.getNamedAttr(gpu::sourceABIAttr,
                                   builder.getI64IntegerAttr(abi)),
              builder.getNamedAttr(gpu::sourceAxisAttr,
                                   builder.getI64IntegerAttr(axis)),
          }),
          kernel.getLoc());
    }
    auto layout = gpu::ViewLayoutAttr::get(
        kernel.getContext(), buffer.getShape(),
        builder.getDenseI64ArrayAttr(dimensions), true,
        builder.getArrayAttr(strides), builder.getStringAttr(""), true);
    workspace.setType(gpu::ViewType::get(
        kernel.getContext(), buffer.getElementType(), buffer.getShape().size(),
        /*access=*/2, abi, nextSource++, layout));
  }
  kernel.setType(FunctionType::get(kernel.getContext(),
                                   kernel.front().getArgumentTypes(),
                                   kernel.getResultTypes()));
  return success();
}

} // namespace intent::triton
