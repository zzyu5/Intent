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
  if (!space || space.empty() || !llvm::all_of(space, [](Attribute attribute) {
        auto extent = cast<gpu::PhysicalExprAttr>(attribute);
        return extent.getKind() == static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) &&
               extent.getValue() == 1;
      }))
    return kernel.emitError("Triton mutable buffers require one physical program");
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

    Value workspace = gpu::createInvocationWorkspace(
        kernel, buffer.getLoc(), type.getElementType(), type.getShape(),
        type.getOwner());
    buffer.getResult().replaceAllUsesWith(workspace);
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
