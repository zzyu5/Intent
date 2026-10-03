#include "Gathers.h"
#include "Intent/Target/Triton/Analysis/Configuration.h"
#include "llvm/ADT/DenseSet.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include <algorithm>
#include <limits>
#include <optional>


using namespace mlir;

namespace intent::triton::detail {

LogicalResult materializeOversizedGathers(func::FuncOp kernel) {
  auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
  if (!space || space.size() != 1)
    return success();
  auto configurations = gpu::ParameterSpace::read(kernel);
  if (failed(configurations)) return failure();
  auto tuples = configurations->configurations(gpu::ConfigurationStage::Shared);
  if (failed(tuples)) return failure();
  int64_t maximumPrograms = 0;
  for (DictionaryAttr tuple : *tuples) {
    NamedAttrList bindings;
    Builder attributes(kernel.getContext());
    for (gpu::ParameterAttr schema : configurations->extentDeclarations()) {
      if (schema.getCandidates().size() == 1)
        bindings.set(schema.getName(), attributes.getI64IntegerAttr(schema.getCandidates()[0]));
    }
    for (NamedAttribute entry : tuple)
      bindings.set(entry.getName(), entry.getValue());
    auto count = evaluateCompileTimeExpression(
        cast<gpu::PhysicalExprAttr>(space[0]), bindings.getDictionary(kernel.getContext()));
    if (!count || *count <= 0)
      return success();
    maximumPrograms = std::max(maximumPrograms, *count);
  }
  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  llvm::MapVector<Value, SmallVector<gpu::GatherOp>> readers;
  kernel.walk([&](gpu::GatherOp gather) {
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    if (!source || gather.getCoordinates().empty() ||
        belongsToSplitGatherPair(gather) ||
        !isa<FloatType, IntegerType>(source.getElementType()))
      return;
    int64_t bytes = (source.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    for (Attribute attribute : source.getShape()) {
      auto extent = cast<gpu::PhysicalExprAttr>(attribute);
      if (extent.getKind() != gpu::PhysicalExprKind::Constant ||
          extent.getValue() <= 0 ||
          bytes > std::numeric_limits<int64_t>::max() / extent.getValue())
        return;
      bytes *= extent.getValue();
    }
    if (bytes <= capabilities.getMaxDynamicSharedMemoryPerBlock())
      return;
    SmallVector<Value> selected(source.getShape().size());
    for (auto [coordinate, axis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes()))
      selected[axis] = coordinate;
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis) {
      Type coordinateType;
      if (selected[axis]) {
        coordinateType = selected[axis].getType();
      } else {
        if (!result)
          return;
        auto mapping = cast<gpu::AxisMapAttr>(source.getAxisMaps()[axis]);
        auto projection = gpu::queryFragmentAxis(result, gpu::sourceAxisIdentity(mapping));
        if (!projection.isExact() ||
            projection.dimensionId != mapping.getDimensionId() ||
            result.getShape()[projection.fragmentAxis] != source.getShape()[axis])
          return;
        auto ordinal = gpu::AxisMapAttr::get(kernel.getContext(),
            mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDimensionId(),
            0, mapping.getDerived());
        coordinateType = gpu::FragmentType::get(kernel.getContext(),
            IndexType::get(kernel.getContext()), ArrayAttr::get(kernel.getContext(),
                {source.getShape()[axis]}), ArrayAttr::get(kernel.getContext(), {ordinal}),
            source.getValidity(), source.getOwner());
      }
      if (auto coordinate = dyn_cast<gpu::FragmentType>(coordinateType)) {
        if (!result)
          return;
        auto projected = gpu::FragmentType::get(kernel.getContext(),
            coordinate.getElementType(), result.getShape(), result.getAxisMaps(),
            result.getValidity(), result.getOwner());
        if (!gpu::queryBroadcastProjection(coordinate, projected).isExact())
          return;
      }
    }
    readers[gather.getSource()].push_back(gather);
  });
  if (readers.empty())
    return success();
  OpBuilder entry(&kernel.front(), kernel.front().begin());
  gpu::ProgramIdOp programId;
  kernel.walk([&](gpu::ProgramIdOp operation) {
    if (operation.getAxis() == 0)
      programId = operation;
  });
  if (programId) {
    if (programId->getBlock() != &kernel.front() ||
        programId.getOperation() != &kernel.front().front())
      programId->moveBefore(&kernel.front(), kernel.front().begin());
  } else {
    programId = entry.create<gpu::ProgramIdOp>(kernel.getLoc(), entry.getIndexType(), 0);
  }
  Value program = programId.getResult();
  entry.setInsertionPointAfter(programId);
  auto prefix = gpu::PhysicalExprAttr::get(kernel.getContext(),
      gpu::PhysicalExprKind::Constant, maximumPrograms,
      entry.getStringAttr(""), entry.getArrayAttr({}));
  for (auto &readerGroup : readers) {
    auto &gathers = readerGroup.second;
    Value source = gathers.front().getSource();
    auto payload = cast<gpu::FragmentType>(source.getType());
    SmallVector<Attribute> shape{prefix};
    llvm::append_range(shape, payload.getShape());
    Value workspace = gpu::createInvocationWorkspace(
        kernel, source.getLoc(), payload.getElementType(),
        entry.getArrayAttr(shape), payload.getOwner());
    // The maximum is evaluated over every shared tuple; the private prefix
    // remains valid while Triton chooses its local configuration.
    entry.create<gpu::AssumeInBoundsOp>(source.getLoc(), program, workspace, 0);
    OpBuilder builder(kernel.getContext());
    if (Operation *definition = source.getDefiningOp())
      builder.setInsertionPointAfter(definition);
    else
      builder.setInsertionPointToStart(cast<BlockArgument>(source).getOwner());
    SmallVector<Value> coordinates{program};
    SmallVector<int64_t> sourceAxes{0};
    SmallVector<Value> ordinals;
    for (auto [axis, attribute] : llvm::enumerate(payload.getShape())) {
      auto mapping = cast<gpu::AxisMapAttr>(payload.getAxisMaps()[axis]);
      auto ordinalMap = gpu::AxisMapAttr::get(kernel.getContext(),
          mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDimensionId(),
          0, mapping.getDerived());
      auto type = gpu::FragmentType::get(kernel.getContext(), builder.getIndexType(),
          builder.getArrayAttr({attribute}), builder.getArrayAttr({ordinalMap}),
          payload.getValidity(), payload.getOwner());
      Value zero = builder.create<arith::ConstantIndexOp>(source.getLoc(), 0);
      Value one = builder.create<arith::ConstantIndexOp>(source.getLoc(), 1);
      Value size = builder.create<gpu::PhysicalExprOp>(source.getLoc(),
          builder.getIndexType(), cast<gpu::PhysicalExprAttr>(attribute));
      Value ordinal = builder.create<gpu::MakeRangeOp>(source.getLoc(), type,
          zero, size, one, zero, size, mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDerived());
      ordinals.push_back(ordinal);
      coordinates.push_back(ordinal);
      sourceAxes.push_back(axis + 1);
    }
    builder.create<gpu::StoreOp>(source.getLoc(), workspace, coordinates, source,
                                 Value(), sourceAxes);
    for (gpu::GatherOp gather : gathers) {
      builder.setInsertionPoint(gather);
      auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
      SmallVector<Value> selected(payload.getShape().size());
      for (auto [coordinate, axis] :
           llvm::zip(gather.getCoordinates(), gather.getSourceAxes()))
        selected[axis] = coordinate;
      SmallVector<Value> access{program};
      for (unsigned axis = 0; axis < payload.getShape().size(); ++axis) {
        Value coordinate = selected[axis] ? selected[axis] : ordinals[axis];
        if (result) {
          auto indices = gpu::FragmentType::get(kernel.getContext(),
              gpu::uniformElementType(coordinate.getType()), result.getShape(), result.getAxisMaps(),
              result.getValidity(), result.getOwner());
          if (coordinate.getType() != indices)
            coordinate = builder.create<gpu::BroadcastOp>(gather.getLoc(), indices, coordinate);
        }
        access.push_back(coordinate);
      }
      auto load = builder.create<gpu::LoadOp>(gather.getLoc(), gather.getResult().getType(), workspace,
          access, gather.getValid(), gather.getFill(), sourceAxes);
      if (Attribute origin = gather->getAttr(gpu::originAttr))
        load->setAttr(gpu::originAttr, origin);
      gather.getResult().replaceAllUsesWith(load.getResult());
      gather.erase();
    }
  }
  gpu::eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace intent::triton::detail
