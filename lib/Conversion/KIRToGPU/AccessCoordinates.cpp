#include "Construction.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::kir_to_gpu {

FailureOr<unsigned> ScalarRegionLowering::physicalResourceAxis(
    Type logicalResource,
    Type physicalResource,
    unsigned logicalAxis) {
  RankedTensorType logicalTensor;
  if (auto view = dyn_cast<intent::ViewType>(logicalResource))
    logicalTensor = dyn_cast<RankedTensorType>(view.getTensor());
  else if (auto buffer = dyn_cast<intent::BufferType>(logicalResource))
    logicalTensor = dyn_cast<RankedTensorType>(buffer.getTensor());
  else
    logicalTensor = dyn_cast<RankedTensorType>(logicalResource);
  if (!logicalTensor || logicalAxis >= static_cast<unsigned>(logicalTensor.getRank()))
    return failure();

  unsigned physicalRank = 0;
  if (auto view = dyn_cast<gpu::ViewType>(physicalResource))
    physicalRank = view.getRank();
  else if (auto buffer = dyn_cast<gpu::BufferType>(physicalResource))
    physicalRank = buffer.getShape().size();
  else if (auto fragment = dyn_cast<gpu::FragmentType>(physicalResource))
    physicalRank = fragment.getShape().size();
  else
    return failure();
  unsigned logicalRank = logicalTensor.getRank();
  if (physicalRank < logicalRank)
    return failure();
  if (!isa<gpu::FragmentType>(physicalResource) && physicalRank != logicalRank)
    return failure();
  return physicalRank - logicalRank + logicalAxis;
}

FailureOr<Value> ScalarRegionLowering::resourceExtent(
    Location location,
    Value resource,
    unsigned axis) {
  PhysicalExprAttr extent;
  if (auto view = dyn_cast<gpu::ViewType>(resource.getType())) {
    if (axis >= view.getRank())
      return failure();
    extent = cast<PhysicalExprAttr>(view.getLayout().getExtents()[axis]);
    if (extent.getKind() ==
        PhysicalExprKind::Constant)
      return physicalExtentValue(location, extent);
    return Value(builder.create<gpu::DimOp>(location, builder.getIndexType(),
                                            resource, axis));
  } else if (auto buffer = dyn_cast<gpu::BufferType>(resource.getType())) {
    if (axis >= buffer.getShape().size())
      return failure();
    extent = cast<PhysicalExprAttr>(buffer.getShape()[axis]);
  } else if (auto fragment = dyn_cast<gpu::FragmentType>(resource.getType())) {
    if (axis >= fragment.getShape().size())
      return failure();
    gpu::PhysicalProgramAnalysis analysis(physicalKernel);
    if (analysis.axisRealization(resource, axis).constructionScalarSeed) {
      auto mapping = cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[axis]);
      auto dimension = dimensions.find(mapping.getDimensionId());
      if (dimension != dimensions.end())
        return dimension->second;
      FailureOr<gpu::MakeRangeOp> range =
          gpu::queryExactLogicalRange(analysis.axisRanges(resource, axis));
      if (failed(range))
        return failure();
      return rangeExtent(location, range->getLogicalStart(),
                         range->getLogicalStop(), range->getStep());
    }
    extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
  } else {
    return failure();
  }
  return physicalExtentValue(location, extent);
}

FailureOr<SmallVector<Value>> ScalarRegionLowering::accessCoordinates(
    Operation *operation) {
  auto relation = canonicalAnalysis.indexRelation(operation);
  if (failed(relation))
    return failure();
  SmallVector<Value> coordinates;
  FailureOr<Value> resource = get(relation->source);
  if (failed(resource))
    return failure();
  Value stored = cast<IndexedAccessOpInterface>(operation).getStoredValue();
  gpu::FragmentType valuePrototype;
  if (isa<ViewStoreOp, BufferStoreOp, ScatterUniqueOp, ScatterReduceOp>(operation)) {
    FailureOr<Value> value = get(stored);
    if (succeeded(value))
      valuePrototype = dyn_cast<gpu::FragmentType>((*value).getType());
  }
  auto resultExtent = [&](size_t axis,
                          PhysicalExprAttr fallback)
      -> FailureOr<PhysicalExprAttr> {
    if (valuePrototype) {
      size_t resultRank = relation->resultDimensionIdentities.size();
      if (valuePrototype.getShape().size() < resultRank)
        return failure();
      // A scalar logical index may already have been promoted to a physical
      // fragment.  Those ownership axes lead the value fragment but are not
      // part of the logical access result.  Relation result axis zero must
      // therefore address the first axis after that exact lifted prefix.
      size_t liftedRank = valuePrototype.getShape().size() - resultRank;
      size_t prototypeAxis = liftedRank + axis;
      if (prototypeAxis < valuePrototype.getShape().size())
        return cast<PhysicalExprAttr>(
            valuePrototype.getShape()[prototypeAxis]);
    }
    if (operation->getNumResults() == 0)
      return fallback;
    auto tensor = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
    if (!tensor || axis >= static_cast<size_t>(tensor.getRank()))
      return fallback;
    if (!tensor.isDynamicDim(axis))
      return expression(operation->getContext(), PhysicalExprKind::Constant,
                        tensor.getDimSize(axis));
    FailureOr<PhysicalExprAttr> extent =
        fragmentExtentExpression(canonicalAnalysis, tensor, operation, axis);
    return extent;
  };
  auto makeRange = [&](unsigned sourceAxis, Value start, Value stop,
                       Value step, uint64_t sourceId,
                       int64_t dimensionId,
                       bool derived,
                       PhysicalExprAttr physicalExtent) -> FailureOr<Value> {
    FailureOr<Value> physicalStart = asIndex(operation->getLoc(), start);
    FailureOr<Value> physicalStop = asIndex(operation->getLoc(), stop);
    FailureOr<Value> physicalStep = asIndex(operation->getLoc(), step);
    if (failed(physicalStart) || failed(physicalStop) || failed(physicalStep))
      return failure();
    start = *physicalStart;
    stop = *physicalStop;
    step = *physicalStep;
    FailureOr<Value> extent =
        physicalExtentValue(operation->getLoc(), physicalExtent);
    if (failed(extent))
      return failure();
    auto type = fragmentType(operation->getContext(), builder.getIndexType(),
                             {physicalExtent},
                             {{sourceId, sourceAxis, dimensionId, derived}});
    return Value(builder.create<gpu::MakeRangeOp>(
        operation->getLoc(), type, start, *extent, step, start, stop, sourceId,
        sourceAxis, derived));
  };
  for (const IndexTermFact &term : relation->terms) {
    if (!term.sourceAxis)
      continue;
    unsigned sourceAxis = *term.sourceAxis;
    unsigned resultAxis = term.resultAxes.empty() ? 0 : term.resultAxes.front();
    FailureOr<unsigned> physicalSourceAxis = physicalResourceAxis(
        relation->source.getType(), (*resource).getType(), sourceAxis);
    if (failed(physicalSourceAxis))
      return failure();
    if (term.kind == 0) {
      auto view = dyn_cast<gpu::ViewType>((*resource).getType());
      auto buffer = dyn_cast<gpu::BufferType>((*resource).getType());
      auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
      if ((!view && !buffer && !fragment) ||
          (view && *physicalSourceAxis >= view.getRank()) ||
          (buffer && *physicalSourceAxis >= buffer.getShape().size()) ||
          (fragment &&
           *physicalSourceAxis >= fragment.getShape().size())) {
        operation->emitOpError(
            "full-slice term has no matching physical source axis");
        return failure();
      }
      Value start = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 0);
      Value stop;
      uint64_t sourceId;
      PhysicalExprAttr extent;
      uint32_t logicalSourceAxis = sourceAxis;
      bool derived = false;
      if (view || buffer) {
        // A buffer is a canonical value result, using the same derived
        // identity convention as resultAxisIdentity.
        sourceId = view ? view.getSourceId() : buffer.getInstance() + 1;
        derived = static_cast<bool>(buffer);
        extent = cast<PhysicalExprAttr>(
            view ? view.getLayout().getExtents()[*physicalSourceAxis]
                 : buffer.getShape()[*physicalSourceAxis]);
        FailureOr<Value> physicalStop =
            physicalExtentValue(operation->getLoc(), extent);
        if (failed(physicalStop)) {
          operation->emitOpError(
              "resource full-slice extent is not materialized in the current program: ")
              << extent;
          return failure();
        }
        stop = *physicalStop;
      } else {
        auto mapping =
            cast<gpu::AxisMapAttr>(
                fragment.getAxisMaps()[*physicalSourceAxis]);
        FailureOr<Value> physicalExtent = physicalExtentValue(
            operation->getLoc(),
            cast<PhysicalExprAttr>(
                fragment.getShape()[*physicalSourceAxis]));
        if (failed(physicalExtent)) {
          operation->emitOpError(
              "fragment full-slice extent is not materialized in the current program: ")
              << fragment.getShape()[*physicalSourceAxis];
          return failure();
        }
        stop = *physicalExtent;
        sourceId = mapping.getSourceId();
        logicalSourceAxis = mapping.getSourceAxis();
        derived = mapping.getDerived();
        extent =
            cast<PhysicalExprAttr>(
                fragment.getShape()[*physicalSourceAxis]);
      }
      int64_t dimension = 0;
      if (view) {
        auto dimensions = view.getLayout().getDimensionIds();
        if (*physicalSourceAxis >= dimensions.size())
          return failure();
        dimension = dimensions[*physicalSourceAxis];
      } else if (buffer) {
        dimension = relation->resultDimensionIdentities[resultAxis];
      } else {
        dimension = cast<gpu::AxisMapAttr>(
                        fragment.getAxisMaps()[*physicalSourceAxis])
                        .getDimensionId();
      }
      if (dimension <= 0)
        return failure();
      // A full slice of an already-physical fragment consumes that
      // fragment's current extent.  Reconstructing the helper-local logical
      // result dimension here would discard an enclosing region segment.
      if (view || buffer) {
        FailureOr<PhysicalExprAttr> resultPhysicalExtent =
            resultExtent(resultAxis, extent);
        if (failed(resultPhysicalExtent))
          return failure();
        extent = *resultPhysicalExtent;
      }
      Value step = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 1);
      FailureOr<Value> coordinate = makeRange(
          logicalSourceAxis, start, stop, step, sourceId, dimension, derived,
          extent);
      if (failed(coordinate))
        return failure();
      coordinates.push_back(*coordinate);
      continue;
    }
    if (term.kind == 2) {
      int64_t literal = *term.staticValues[0];
      Value coordinate = builder.create<arith::ConstantIndexOp>(
          operation->getLoc(), literal);
      if (literal < 0) {
        FailureOr<Value> extent =
            resourceExtent(operation->getLoc(), *resource,
                           *physicalSourceAxis);
        if (failed(extent))
          return failure();
        coordinate = createBinary(builder, operation->getLoc(),
                                  builder.getIndexType(), *extent, coordinate,
                                  BinaryOperator::Add);
      }
      coordinates.push_back(coordinate);
      continue;
    }
    if (term.kind == 3) {
      Value index = term.operands[0];
      FailureOr<Value> coordinate = get(index);
      if (failed(coordinate))
        return failure();
      Value physicalCoordinate = *coordinate;
      if (auto fragment = dyn_cast<gpu::FragmentType>(physicalCoordinate.getType())) {
        auto logicalCoordinate =
            dyn_cast<RankedTensorType>(index.getType());
        if (logicalCoordinate) {
          if (fragment.getShape().size() < term.indexAxes.size())
            return failure();
        }
        unsigned prefix = logicalCoordinate
            ? fragment.getShape().size() - term.indexAxes.size() : 0;
        SmallVector<Attribute> mappings;
        for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps())) {
          auto mapping = cast<gpu::AxisMapAttr>(attribute);
          int64_t dimension = mapping.getDimensionId();
          if (logicalCoordinate && axis >= prefix)
            dimension = relation->resultDimensionIdentities[
                term.indexAxes[axis - prefix]];
          mappings.push_back(gpu::AxisMapAttr::get(
              operation->getContext(), mapping.getSourceId(),
              mapping.getSourceAxis(), dimension,
              mappings.size(), mapping.getDerived()));
        }
        auto target = gpu::FragmentType::get(
            operation->getContext(), fragment.getElementType(),
            fragment.getShape(), builder.getArrayAttr(mappings),
            fragment.getValidity(), fragment.getOwner());
        if (target != fragment) {
          auto reassociation = gpu::inferReshapeReassociation(fragment, target);
          if (failed(reassociation))
            return failure();
          physicalCoordinate = builder.create<gpu::ReshapeOp>(
              operation->getLoc(), target, physicalCoordinate, *reassociation);
        }
      }
      FailureOr<Value> logicalIndex =
          asLogicalIndex(operation->getLoc(), physicalCoordinate);
      if (failed(logicalIndex))
        return failure();
      coordinates.push_back(*logicalIndex);
      continue;
    }
    if (term.kind == 4) {
      FailureOr<Value> range = get(term.operands[0]);
      if (failed(range) || !isa<gpu::RangeType>((*range).getType()))
        return failure();
      Value start = rangeBound(operation->getLoc(), *range, 0);
      Value stop = rangeBound(operation->getLoc(), *range, 1);
      Value step = rangeBound(operation->getLoc(), *range, 2);
      auto rangeType = cast<gpu::RangeType>((*range).getType());
      PhysicalExprAttr fallback = expression(
          operation->getContext(), PhysicalExprKind::Constant, 1);
      FailureOr<PhysicalExprAttr> physicalExtent =
          resultExtent(resultAxis, fallback);
      if (failed(physicalExtent))
        return failure();
      PhysicalExprAttr extent = *physicalExtent;
      uint64_t sourceId = rangeType.getSourceId();
      uint32_t logicalSourceAxis = rangeType.getSourceAxis();
      bool derived = rangeType.getDerived();
      auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
      auto logical =
          dyn_cast<RankedTensorType>(relation->source.getType());
      const bool fragmentIndex = static_cast<bool>(fragment);
      unsigned logicalAxis = sourceAxis;
      unsigned fragmentAxis = *physicalSourceAxis;
      if (!fragment && valuePrototype) {
        auto valueType = dyn_cast<RankedTensorType>(stored.getType());
        if (valueType &&
            valueType.getRank() == relation->resultDimensionIdentities.size()) {
          fragment = valuePrototype;
          logical = valueType;
          logicalAxis = resultAxis;
          fragmentAxis =
              fragment.getShape().size() - valueType.getRank() + resultAxis;
        }
      }
      if (fragment && logical && integerConstant(start) == 0 &&
          integerConstant(step) == 1) {
        DenseI64ArrayAttr identities = dimensionIds(logical);
        auto mapping = cast<gpu::AxisMapAttr>(
            fragment.getAxisMaps()[fragmentAxis]);
        bool coversSource = false;
        if (logical.isDynamicDim(logicalAxis)) {
          if (identities) {
            auto binding = dimensions.find(identities[logicalAxis]);
            coversSource =
                binding != dimensions.end() && stop == binding->second;
          }
        } else {
          coversSource = integerConstant(stop) == logical.getDimSize(logicalAxis);
        }
        if (coversSource && identities &&
            mapping.getDimensionId() == identities[logicalAxis] &&
            rangeType.getDimensionId() == identities[logicalAxis]) {
          // Full logical indexing preserves the value's axis relation. Reads
          // use local fragment coordinates; writes retain logical bounds so
          // ownership advances their coordinates with the producing tile.
          sourceId = mapping.getSourceId();
          logicalSourceAxis = mapping.getSourceAxis();
          derived = mapping.getDerived();
          extent = cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
          if (fragmentIndex) {
            FailureOr<Value> fragmentStop =
                physicalExtentValue(operation->getLoc(), extent);
            if (failed(fragmentStop))
              return failure();
            stop = *fragmentStop;
          }
        }
      }
      FailureOr<Value> coordinate = makeRange(
          logicalSourceAxis, start, stop, step, sourceId,
          rangeType.getDimensionId(), derived, extent);
      if (failed(coordinate))
        return failure();
      for (StringRef name : {gpu::sourceSubregionAttr,
                             gpu::sourceSubregionBoundAttr})
        if (Attribute value = (*range).getDefiningOp()->getAttr(name))
          coordinate->getDefiningOp()->setAttr(name, value);
      coordinates.push_back(*coordinate);
      continue;
    }
    if (term.kind == 5) {
      SmallVector<Value> bounds;
      for (unsigned component = 0; component < 3; ++component) {
        Value dynamic = term.operands[component];
        std::optional<int64_t> literal = term.staticValues[component];
        if (dynamic) {
          FailureOr<Value> value = get(dynamic);
          if (failed(value))
            return failure();
          bounds.push_back(*value);
        } else if (literal) {
          bounds.push_back(builder.create<arith::ConstantIndexOp>(
              operation->getLoc(), *literal));
        } else {
          auto view = dyn_cast<gpu::ViewType>((*resource).getType());
          auto buffer = dyn_cast<gpu::BufferType>((*resource).getType());
          auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
          if ((!view && !buffer && !fragment) ||
              (view && *physicalSourceAxis >= view.getRank()) ||
              (buffer && *physicalSourceAxis >= buffer.getShape().size()) ||
              (fragment &&
               *physicalSourceAxis >= fragment.getShape().size()))
            return failure();
          if (component == 0) {
            bounds.push_back(builder.create<arith::ConstantIndexOp>(
                operation->getLoc(), 0));
          } else if (component == 1) {
            if (view || buffer) {
              FailureOr<Value> physicalExtent = physicalExtentValue(
                  operation->getLoc(),
                  cast<PhysicalExprAttr>(
                      view ? view.getLayout().getExtents()[*physicalSourceAxis]
                           : buffer.getShape()[*physicalSourceAxis]));
              if (failed(physicalExtent))
                return failure();
              bounds.push_back(*physicalExtent);
            } else {
              FailureOr<Value> physicalExtent = physicalExtentValue(
                  operation->getLoc(), cast<PhysicalExprAttr>(
                                           fragment.getShape()[*physicalSourceAxis]));
              if (failed(physicalExtent))
                return failure();
              bounds.push_back(*physicalExtent);
            }
          } else {
            bounds.push_back(builder.create<arith::ConstantIndexOp>(
                operation->getLoc(), 1));
          }
        }
      }
      auto view = dyn_cast<gpu::ViewType>((*resource).getType());
      auto buffer = dyn_cast<gpu::BufferType>((*resource).getType());
      auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
      if ((!view && !buffer && !fragment) ||
          (view && *physicalSourceAxis >= view.getRank()) ||
          (buffer && *physicalSourceAxis >= buffer.getShape().size()) ||
          (fragment &&
           *physicalSourceAxis >= fragment.getShape().size()))
        return failure();
      PhysicalExprAttr fallback = cast<PhysicalExprAttr>(
          view     ? view.getLayout().getExtents()[*physicalSourceAxis]
          : buffer ? buffer.getShape()[*physicalSourceAxis]
                   : fragment.getShape()[*physicalSourceAxis]);
      FailureOr<PhysicalExprAttr> physicalExtent =
          resultExtent(resultAxis, fallback);
      if (failed(physicalExtent))
        return failure();
      PhysicalExprAttr extent = *physicalExtent;
      uint64_t sourceId;
      uint32_t logicalSourceAxis = sourceAxis;
      int64_t dimension = 0;
      bool derived = false;
      if (view) {
        sourceId = view.getSourceId();
        auto dimensions = view.getLayout().getDimensionIds();
        if (*physicalSourceAxis >= dimensions.size())
          return failure();
        dimension = dimensions[*physicalSourceAxis];
      } else if (buffer) {
        sourceId = buffer.getInstance() + 1;
        derived = true;
      } else {
        auto mapping =
            cast<gpu::AxisMapAttr>(
                fragment.getAxisMaps()[*physicalSourceAxis]);
        sourceId = mapping.getSourceId();
        logicalSourceAxis = mapping.getSourceAxis();
        dimension = mapping.getDimensionId();
        derived = mapping.getDerived();
      }
      dimension = relation->resultDimensionIdentities[resultAxis];
      if (dimension <= 0)
        return failure();
      FailureOr<Value> coordinate = makeRange(
          logicalSourceAxis, bounds[0], bounds[1], bounds[2], sourceId,
          dimension, derived, extent);
      if (failed(coordinate))
        return failure();
      coordinates.push_back(*coordinate);
      continue;
    }
    return failure();
  }
  return coordinates;
}

FailureOr<SmallVector<int64_t>> ScalarRegionLowering::sourceAxes(
    Operation *operation) {
  auto relation = canonicalAnalysis.indexRelation(operation);
  if (failed(relation))
    return failure();
  FailureOr<Value> resource = get(relation->source);
  if (failed(resource))
    return failure();
  SmallVector<int64_t> axes;
  for (const IndexTermFact &term : relation->terms) {
    if (!term.sourceAxis)
      continue;
    FailureOr<unsigned> physicalSourceAxis = physicalResourceAxis(
        relation->source.getType(), (*resource).getType(), *term.sourceAxis);
    if (failed(physicalSourceAxis))
      return failure();
    axes.push_back(*physicalSourceAxis);
  }
  return axes;
}

} // namespace intent::kir_to_gpu
