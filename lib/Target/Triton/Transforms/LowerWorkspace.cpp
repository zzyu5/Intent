#include "Intent/Target/Triton/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::triton {

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
  DominanceInfo dominance(kernel);
  OpBuilder builder(kernel.getContext());
  llvm::DenseSet<Operation *> synchronizedWriters;
  for (BlockArgument workspace : workspaces) {
    auto buffer = cast<gpu::BufferType>(workspace.getType());
    if (!buffer.getWorkspace() ||
        buffer.getScope().getValue() != gpu::BufferScope::InvocationWorkspace ||
        buffer.getInitialization().getValue() !=
            gpu::BufferInitialization::FirstWrite ||
        buffer.getLifetime().getValue() != gpu::BufferLifetime::Invocation)
      return kernel.emitError(
          "Triton requires an invocation workspace with explicit first writes");

    auto anchorIn = [](Operation *operation, Block *block) {
      while (operation && operation->getBlock() != block)
        operation = operation->getParentOp();
      return operation;
    };
    Block *scope = nullptr;
    for (Operation *user : workspace.getUsers()) {
      if (isa<gpu::DimOp, gpu::AssumeInBoundsOp>(user))
        continue;
      if (!scope)
        scope = user->getBlock();
      while (scope && !anchorIn(user, scope))
        scope = scope->getParentOp()->getBlock();
    }
    if (!scope)
      return kernel.emitError("Triton workspace has no common access scope");
    Operation *writer = nullptr;
    SmallVector<Operation *> readers;
    for (Operation *user : workspace.getUsers()) {
      if (isa<gpu::DimOp, gpu::AssumeInBoundsOp>(user))
        continue;
      if (!isa<gpu::LoadOp, gpu::StoreOp>(user))
        return user->emitOpError(
            "Triton workspace supports explicit loads and stores");
      Operation *anchor = anchorIn(user, scope);
      if (isa<gpu::StoreOp>(user)) {
        if (writer && writer != anchor)
          return user->emitOpError(
              "Triton workspace requires one completed initialization region");
        writer = anchor;
      } else {
        readers.push_back(anchor);
      }
    }
    if (!writer || llvm::any_of(readers, [&](Operation *reader) {
          return writer == reader || !dominance.properlyDominates(writer, reader);
        }))
      return kernel.emitError(
          "Triton workspace reads must follow its completed initialization region");
    if (synchronizedWriters.insert(writer).second) {
      builder.setInsertionPointAfter(writer);
      builder.create<CtaBarrierOp>(writer->getLoc());
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
