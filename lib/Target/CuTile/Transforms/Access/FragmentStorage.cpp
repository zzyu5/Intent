#include "FragmentStorage.h"
#include "Accesses.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Workspace.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"
#include <limits>
#include <numeric>

using namespace mlir;

namespace intent::cutile {
namespace {

Value ordinal(OpBuilder &builder, Location location,
              gpu::PhysicalExprAttr extent, gpu::AxisMapAttr mapping,
              uint64_t owner) {
  auto type = gpu::FragmentType::get(builder.getContext(), builder.getIndexType(),
      builder.getArrayAttr({extent}), builder.getArrayAttr({gpu::AxisMapAttr::get(
          builder.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), 0, mapping.getDerived())}), 1, owner);
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value width = builder.create<gpu::PhysicalExprOp>(location, builder.getIndexType(), extent);
  return builder.create<gpu::MakeRangeOp>(location, type, zero, width, one, zero,
      width, mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived());
}

} // namespace

LogicalResult materializeFragmentStorage(func::FuncOp kernel) {
  llvm::MapVector<Value, SmallVector<gpu::GatherOp>> readers;
  kernel.walk([&](gpu::GatherOp gather) {
    if (requiresFragmentStorage(gather)) readers[gather.getSource()].push_back(gather);
  });
  IRMapping replacements;
  for (auto &[original, gathers] : readers) {
    Value source = replacements.lookupOrDefault(original);
    auto type = cast<gpu::FragmentType>(source.getType());
    Type storageElement = type.getElementType().isIndex()
        ? Type(IntegerType::get(kernel.getContext(), 64)) : type.getElementType();
    uint64_t elements = 1;
    for (Attribute attribute : type.getShape()) {
      auto bounds = gpu::queryPositiveExtentBounds(cast<gpu::PhysicalExprAttr>(attribute), kernel);
      if (!bounds || bounds->second <= 0 ||
          elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                         static_cast<uint64_t>(bounds->second))
        return gathers.front().emitOpError("fragment storage requires finite representable physical extents");
      elements *= bounds->second;
    }
    unsigned bytes = (storageElement.getIntOrFloatBitWidth() + 7) / 8;
    if (elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / bytes)
      return gathers.front().emitOpError("fragment storage byte capacity is not representable");
    OpBuilder builder(kernel.getContext());
    Operation *anchor;
    if (Operation *definition = source.getDefiningOp()) {
      builder.setInsertionPointAfter(definition);
      anchor = definition;
    } else {
      auto argument = cast<BlockArgument>(source);
      builder.setInsertionPointToStart(argument.getOwner());
      anchor = &argument.getOwner()->front();
    }
    if (!gpu::isProgramAllocationContext(anchor, kernel))
      return gathers.front().emitOpError("fragment storage cannot escape an isolated helper");
    Location location = source.getLoc();
    auto buffer = gpu::createProgramBuffer(builder, location, storageElement,
                                           type.getShape(), type.getOwner());
    auto [sourceId, dimension] = gpu::nextPhysicalAxisIdentities(kernel);
    SmallVector<Attribute> mappings, groups;
    SmallVector<Value> coordinates;
    SmallVector<int64_t> axes(type.getShape().size());
    std::iota(axes.begin(), axes.end(), 0);
    for (int64_t axis : axes) {
      auto mapping = gpu::AxisMapAttr::get(kernel.getContext(), sourceId, axis,
                                          dimension++, axis, true);
      mappings.push_back(mapping);
      groups.push_back(gpu::ReshapeGroupAttr::get(kernel.getContext(),
          builder.getDenseI64ArrayAttr({axis}), builder.getDenseI64ArrayAttr({axis})));
      coordinates.push_back(ordinal(builder, location,
          cast<gpu::PhysicalExprAttr>(type.getShape()[axis]), mapping, type.getOwner()));
    }
    // Store the already-computed physical lanes, including their existing fill.
    // Fresh axes denote positions in this private snapshot, not a new logical
    // extent or permission to reread/replay the original producer.
    auto storedType = gpu::FragmentType::get(kernel.getContext(), type.getElementType(),
        type.getShape(), builder.getArrayAttr(mappings), type.getValidity(), type.getOwner());
    Value stored = builder.create<gpu::ReshapeOp>(location, storedType, source,
                                                 builder.getArrayAttr(groups));
    if (storageElement != type.getElementType()) {
      storedType = gpu::FragmentType::get(kernel.getContext(), storageElement,
          type.getShape(), builder.getArrayAttr(mappings), type.getValidity(), type.getOwner());
      stored = builder.create<gpu::CastOp>(location, storedType, stored);
    }
    builder.create<gpu::StoreOp>(location, buffer, coordinates, stored, Value(), axes);
    for (gpu::GatherOp gather : gathers) {
      OpBuilder read(gather);
      SmallVector<Value> indices(type.getShape().size());
      for (auto [coordinate, axis] : llvm::zip(gather.getCoordinates(), gather.getSourceAxes()))
        indices[axis] = coordinate;
      for (int64_t axis : axes)
        if (!indices[axis])
          indices[axis] = ordinal(read, gather.getLoc(),
              cast<gpu::PhysicalExprAttr>(type.getShape()[axis]),
              cast<gpu::AxisMapAttr>(type.getAxisMaps()[axis]), type.getOwner());
      Type loadedType = gather.getResult().getType();
      if (storageElement != type.getElementType()) {
        if (auto fragment = dyn_cast<gpu::FragmentType>(loadedType))
          loadedType = gpu::FragmentType::get(kernel.getContext(), storageElement,
              fragment.getShape(), fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
        else loadedType = storageElement;
      }
      Value fill = gather.getFill();
      if (fill && storageElement != type.getElementType()) {
        Type fillType = storageElement;
        if (auto fragment = dyn_cast<gpu::FragmentType>(fill.getType()))
          fillType = gpu::FragmentType::get(kernel.getContext(), storageElement,
              fragment.getShape(), fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
        fill = read.create<gpu::CastOp>(gather.getLoc(), fillType, fill);
      }
      auto loaded = read.create<gpu::LoadOp>(gather.getLoc(), loadedType,
          buffer, indices, gather.getValid(), fill, axes);
      if (Attribute origin = gather->getAttr(gpu::originAttr)) loaded->setAttr(gpu::originAttr, origin);
      Value result = loaded;
      if (storageElement != type.getElementType())
        result = read.create<gpu::CastOp>(gather.getLoc(), gather.getResult().getType(), result);
      replacements.map(gather.getResult(), result);
      gather.getResult().replaceAllUsesWith(result);
      gather.erase();
    }
  }
  return success();
}

} // namespace intent::cutile
