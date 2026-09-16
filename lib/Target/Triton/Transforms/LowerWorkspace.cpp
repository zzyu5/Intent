#include "Intent/Target/Triton/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

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
  auto [source, dimension] = gpu::nextPhysicalAxisIdentities(kernel);
  for (gpu::BufferOp buffer : buffers) {
    auto type = buffer.getResult().getType();
    if (type.getScope().getValue() != gpu::BufferScope::ProgramPrivate ||
        type.getLifetime().getValue() != gpu::BufferLifetime::Program ||
        buffer->getBlock() != &kernel.front())
      return buffer.emitOpError("Triton mutable buffer requires an entry program allocation");
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
      if (buffer.getInitialValue() && kind != gpu::PhysicalExprKind::Constant)
        return buffer.emitOpError(
            "Triton mutable buffer initialization requires a static shape");
    }

    OpBuilder builder(buffer);
    Value initial = buffer.getInitialValue();
    Value workspace = gpu::createInvocationWorkspace(
        kernel, buffer.getLoc(), type.getElementType(), type.getShape(),
        type.getOwner());
    if (initial) {
      auto payload = dyn_cast<gpu::FragmentType>(initial.getType());
      if (!payload) {
        SmallVector<Attribute> maps;
        for (unsigned axis = 0; axis < type.getShape().size(); ++axis)
          maps.push_back(gpu::AxisMapAttr::get(kernel.getContext(), source, axis,
                                              dimension++, axis, false));
        ++source;
        payload = gpu::FragmentType::get(kernel.getContext(), type.getElementType(),
            type.getShape(), builder.getArrayAttr(maps), 1, type.getOwner());
      }
      if (auto splat = initial.getDefiningOp<gpu::SplatOp>())
        initial = splat.getValue();
      if (auto broadcast = initial.getDefiningOp<gpu::BroadcastOp>();
          broadcast && !isa<gpu::FragmentType, gpu::RecordType>(broadcast.getValue().getType()))
        initial = broadcast.getValue();
      SmallVector<Value> coordinates, predicates;
      SmallVector<int64_t> axes;
      SmallVector<Attribute> shape;
      for (auto [axis, attribute] : llvm::enumerate(type.getShape())) {
        int64_t count = cast<gpu::PhysicalExprAttr>(attribute).getValue();
        uint64_t padded = llvm::PowerOf2Ceil(static_cast<uint64_t>(count));
        if (padded > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
          return buffer.emitOpError("Triton buffer initialization extent exceeds index range");
        auto extent = gpu::PhysicalExprAttr::get(kernel.getContext(),
            static_cast<uint32_t>(gpu::PhysicalExprKind::Constant), padded,
            builder.getStringAttr(""), builder.getArrayAttr({}));
        shape.push_back(extent);
        auto mapping = cast<gpu::AxisMapAttr>(payload.getAxisMaps()[axis]);
        auto ordinal = gpu::AxisMapAttr::get(kernel.getContext(),
            mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDimensionId(),
            0, mapping.getDerived());
        auto rangeType = gpu::FragmentType::get(kernel.getContext(), builder.getIndexType(),
            builder.getArrayAttr({extent}), builder.getArrayAttr({ordinal}),
            payload.getValidity(), payload.getOwner());
        Value zero = builder.create<arith::ConstantIndexOp>(buffer.getLoc(), 0);
        Value one = builder.create<arith::ConstantIndexOp>(buffer.getLoc(), 1);
        Value size = builder.create<arith::ConstantIndexOp>(buffer.getLoc(), count);
        Value width = builder.create<arith::ConstantIndexOp>(buffer.getLoc(), padded);
        Value range = builder.create<gpu::MakeRangeOp>(buffer.getLoc(), rangeType,
            zero, width, one, zero, size, mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDerived());
        coordinates.push_back(range);
        axes.push_back(axis);
        if (padded != static_cast<uint64_t>(count)) {
          Value end = builder.create<gpu::BroadcastOp>(buffer.getLoc(), rangeType, size);
          auto boolean = gpu::FragmentType::get(kernel.getContext(), builder.getI1Type(),
              rangeType.getShape(), rangeType.getAxisMaps(),
              rangeType.getValidity(), rangeType.getOwner());
          predicates.push_back(builder.create<gpu::CompareOp>(buffer.getLoc(),
              boolean, range, end, ComparePredicate::Lt));
        }
      }
      auto boolean = gpu::FragmentType::get(kernel.getContext(), builder.getI1Type(),
          builder.getArrayAttr(shape), payload.getAxisMaps(),
          payload.getValidity(), payload.getOwner());
      Value valid;
      for (Value predicate : predicates) {
        auto projected = gpu::materializeBroadcastToFragment(
            builder, buffer.getLoc(), predicate, boolean);
        if (failed(projected))
          return buffer.emitOpError("Triton buffer initialization predicate has no exact projection");
        valid = valid ? Value(builder.create<gpu::BinaryOp>(buffer.getLoc(), boolean,
                    valid, *projected, BinaryOperator::LogicalAnd)) : *projected;
      }
      auto valueType = gpu::FragmentType::get(kernel.getContext(), type.getElementType(),
          builder.getArrayAttr(shape), payload.getAxisMaps(),
          payload.getValidity(), payload.getOwner());
      if (!isa<gpu::FragmentType>(initial.getType()))
        initial = builder.create<gpu::BroadcastOp>(buffer.getLoc(), valueType, initial);
      else if (initial.getType() != valueType)
        return buffer.emitOpError("Triton buffer initializer requires complete physical padding");
      builder.create<gpu::StoreOp>(buffer.getLoc(), workspace,
                                   coordinates, initial, valid, axes);
    }
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
