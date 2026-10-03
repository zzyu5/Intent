#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Worklist.h"

using namespace mlir;

namespace intent::gpu::value_relations {

WalkResult alignPointwiseValue(Operation *operation,
                               RelationWorklist &changes) {
  auto kernel = operation->getParentOfType<func::FuncOp>();
  auto isParameterExtent = [](Attribute attribute) {
    auto extent = dyn_cast<PhysicalExprAttr>(attribute);
    return extent && extent.getKind() == PhysicalExprKind::Parameter;
  };
  auto sameSchema = [](FragmentType lhs, FragmentType rhs) {
    return lhs.getShape() == rhs.getShape() &&
           lhs.getAxisMaps() == rhs.getAxisMaps() &&
           lhs.getValidity() == rhs.getValidity() &&
           lhs.getOwner() == rhs.getOwner();
  };
  auto align = [&](Operation *operation, unsigned operandIndex,
                   FragmentType targetShape) -> LogicalResult {
    Value value = operation->getOperand(operandIndex);
    auto source = dyn_cast<FragmentType>(value.getType());
    if (source && sameSchema(source, targetShape))
      return success();
    auto target = FragmentType::get(
        kernel.getContext(), source ? source.getElementType() : value.getType(),
        targetShape.getShape(), targetShape.getAxisMaps(),
        targetShape.getValidity(), targetShape.getOwner());
    OpBuilder builder(operation);
    builder.setListener(&changes);
    FailureOr<Value> replacement = projectPhysicalValueToSchema(
        builder, operation->getLoc(), value, target, changes.typeChanged());
    if (failed(replacement))
      return failure();
    operation->setOperand(operandIndex, *replacement);
    return success();
  };

  if (auto transpose = dyn_cast<TransposeOp>(operation)) {
    auto relations = queryFragmentOperandRelations(transpose);
    if (failed(relations)) {
      transpose.emitOpError("transpose has no physical operand relation");
      return WalkResult::interrupt();
    }
    auto result =
        transportFragmentResultType(*relations, transpose->getOperandTypes(),
                                    transpose.getResult().getType());
    if (failed(result)) {
      transpose.emitOpError(
          "transpose cannot transport its physical operand schema");
      return WalkResult::interrupt();
    }
    changes.setType(transpose.getResult(), *result);
    return WalkResult::advance();
  }
  if (auto broadcast = dyn_cast<BroadcastOp>(operation)) {
    auto target = dyn_cast<FragmentType>(broadcast.getResult().getType());
    auto source = dyn_cast<FragmentType>(broadcast.getValue().getType());
    if (!target || !source)
      return WalkResult::advance();
    auto relations = queryFragmentOperandRelations(broadcast);
    if (failed(relations) || !relations->front().hasCompatibleExtents()) {
      OpBuilder builder(broadcast);
      builder.setListener(&changes);
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, broadcast.getLoc(), broadcast.getValue(), target,
          changes.typeChanged());
      if (succeeded(projected)) {
        broadcast->setOperand(0, *projected);
        source = cast<FragmentType>(projected->getType());
      }
    }
    relations = queryFragmentOperandRelations(broadcast);
    if (failed(relations)) {
      broadcast.emitOpError(
          "broadcast has no unique physical source projection")
          << "; input=" << source << "; result=" << target;
      return WalkResult::interrupt();
    }
    // A non-singleton BroadcastOp is an explicit extent-preserving value
    // relation even when its input and result use distinct canonical
    // occurrence identities.  Pointwise ownership may first reach only one
    // side of that relation (for example a RegionFold summary schema).  Close
    // the already-selected parameter extent across the typed projection
    // before asking the verifier to observe the intermediate program.  A true
    // singleton broadcast remains an expansion and never acquires the
    // consumer's extent.
    for (const FragmentAxisGroup &group : relations->front().groups) {
      if (group.sourceAxes.empty())
        continue;
      unsigned sourceAxis = group.sourceAxes.front();
      unsigned targetAxis = group.resultAxes.front();
      if (source.getShape()[sourceAxis] == target.getShape()[targetAxis])
        continue;
      auto sourceExtent = cast<PhysicalExprAttr>(source.getShape()[sourceAxis]);
      bool singleton = sourceExtent.getKind() == PhysicalExprKind::Constant &&
                       sourceExtent.getValue() == 1;
      if (singleton)
        continue;
      auto targetExtent = cast<PhysicalExprAttr>(target.getShape()[targetAxis]);
      bool targetSingleton =
          targetExtent.getKind() == PhysicalExprKind::Constant &&
          targetExtent.getValue() == 1;
      if (targetSingleton) {
        PhysicalProgramAnalysis analysis(kernel);
        auto sourceMap = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
        auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
        bool derivedOccurrence =
            targetMap.getDerived() && sourceMap.getDimensionId() > 0 &&
            sourceMap.getDimensionId() == targetMap.getDimensionId();
        auto realization =
            analysis.axisRealization(broadcast.getResult(), targetAxis);
        auto ranges = analysis.programRanges(sourceAxisIdentity(targetMap));
        bool selectedProgramExtent =
            !realization.hasExtentAuthority() && ranges.isExact() &&
            !ranges.roots.empty() && sourceMap.getDimensionId() > 0 &&
            sourceMap.getDimensionId() == targetMap.getDimensionId() &&
            queryFragmentAxis(target, sourceAxisIdentity(targetMap),
                              targetMap.getDimensionId())
                .isExact() &&
            analysis.lockstepRanges(ranges.roots).isExact() &&
            llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
              auto dimension = queryRangeDimension(range);
              return analysis.isProgramOwnedRange(range) &&
                     succeeded(dimension) &&
                     *dimension == targetMap.getDimensionId() &&
                     queryLaunchExpression(range.getExtent()) == sourceExtent;
            });
        if (realization.constructionScalarSeed || derivedOccurrence ||
            selectedProgramExtent) {
          if (failed(retargetSourceExtent(
                  broadcast.getResult(), sourceAxisIdentity(targetMap),
                  sourceExtent, std::nullopt, changes.typeChanged(), &changes)))
            return WalkResult::interrupt();
          target = cast<FragmentType>(broadcast.getResult().getType());
          continue;
        }
        broadcast.emitOpError(
            "broadcast cannot contract a non-singleton physical axis")
            << "; input=" << source << "; result=" << target;
        return WalkResult::interrupt();
      }
      bool sourceParameter = isParameterExtent(source.getShape()[sourceAxis]);
      bool targetParameter = isParameterExtent(targetExtent);
      if (!sourceParameter && !targetParameter) {
        PhysicalProgramAnalysis analysis(kernel);
        auto input = analysis.axisRealization(broadcast.getValue(), sourceAxis);
        auto output =
            analysis.axisRealization(broadcast.getResult(), targetAxis);
        if (input.hasExtentAuthority() && !input.constructionScalarSeed &&
            !output.hasExtentAuthority()) {
          auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
          if (failed(retargetSourceExtent(
                  broadcast.getResult(), sourceAxisIdentity(targetMap),
                  sourceExtent, std::nullopt, changes.typeChanged(), &changes)))
            return WalkResult::interrupt();
          target = cast<FragmentType>(broadcast.getResult().getType());
          continue;
        }
      }
      if (sourceParameter == targetParameter) {
        broadcast.emitOpError(
            "broadcast has conflicting non-singleton physical extents")
            << "; input=" << source << "; result=" << target;
        return WalkResult::interrupt();
      }
      auto sourceMap = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
      auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
      if (sourceMap.getDimensionId() <= 0 || targetMap.getDimensionId() <= 0) {
        broadcast.emitOpError(
            "broadcast extent refinement has no logical occurrence authority");
        return WalkResult::interrupt();
      }
      if (targetParameter) {
        if (failed(retargetSourceExtent(
                broadcast.getValue(), sourceAxisIdentity(sourceMap),
                cast<PhysicalExprAttr>(target.getShape()[targetAxis]),
                std::nullopt, changes.typeChanged(), &changes)))
          return WalkResult::interrupt();
      } else {
        if (failed(retargetSourceExtent(
                broadcast.getResult(), sourceAxisIdentity(targetMap),
                sourceExtent, std::nullopt, changes.typeChanged(), &changes)))
          return WalkResult::interrupt();
      }
      source = cast<FragmentType>(broadcast.getValue().getType());
      target = cast<FragmentType>(broadcast.getResult().getType());
    }
    FailureOr<FragmentType> refined =
        queryValueSchema(kernel, target, ValueRange{broadcast.getValue()});
    if (failed(refined)) {
      broadcast.emitOpError(
          "broadcast result has no unique physical source projection");
      return WalkResult::interrupt();
    }
    // Broadcast has two independent typed relations: its input supplies the
    // physical extents, while the result type supplies the logical occurrence
    // seen by consumers.  Adopting the input's AxisMap would erase an explicit
    // output/index relation (for example a reshaped value stored into a view)
    // and force a later access pass to reconstruct it.
    changes.setType(
        broadcast.getResult(),
        FragmentType::get(kernel.getContext(), target.getElementType(),
                          (*refined).getShape(), target.getAxisMaps(),
                          target.getValidity(), target.getOwner()));
    return WalkResult::advance();
  }
  FragmentType target;
  if (operation->getNumResults() == 1)
    target = dyn_cast<FragmentType>(operation->getResult(0).getType());
  if (!isa<FragmentOpInterface>(operation) || isa<SplatOp>(operation))
    return WalkResult::advance();
  if (!target && operation->getNumResults() == 1 &&
      isa<IntegerType, FloatType, IndexType>(
          operation->getResult(0).getType())) {
    FragmentType prototype;
    for (Value operand : operation->getOperands())
      if ((prototype = dyn_cast<FragmentType>(operand.getType())))
        break;
    if (prototype) {
      target = FragmentType::get(kernel.getContext(),
                                 operation->getResult(0).getType(),
                                 prototype.getShape(), prototype.getAxisMaps(),
                                 prototype.getValidity(), prototype.getOwner());
      changes.setType(operation->getResult(0), target);
    }
  }
  if (!target)
    return WalkResult::advance();
  // Sole-source operations acquire their operand's complete lane schema.
  // Multi-operand operations first resolve extent authorities below: their
  // still-unprojected uniform inputs need not yet have the result's rank.
  if (isa<UnaryOp, CastOp, BitcastOp>(operation) &&
      isa<FragmentType>(operation->getOperand(0).getType())) {
    auto relations = queryFragmentOperandRelations(operation);
    if (failed(relations)) {
      operation->emitOpError(
          "pointwise operation has no physical operand relation");
      return WalkResult::interrupt();
    }
    auto result = transportFragmentResultType(
        *relations, operation->getOperandTypes(), target);
    if (failed(result)) {
      operation->emitOpError(
          "pointwise operation cannot transport its physical operand schema");
      return WalkResult::interrupt();
    }
    changes.setType(operation->getResult(0), *result);
    return WalkResult::advance();
  }
  FailureOr<FragmentType> refined =
      queryValueSchema(kernel, target, operation->getOperands());
  if (failed(refined)) {
    InFlightDiagnostic diagnostic = operation->emitOpError(
        "pointwise result has no unique physical operand projection");
    diagnostic << "; result=" << target;
    for (Value operand : operation->getOperands())
      diagnostic << "; operand=" << operand.getType();
    return WalkResult::interrupt();
  }
  for (auto [axis, mapping] : llvm::enumerate((*refined).getAxisMaps())) {
    if (axis >= target.getShape().size() ||
        target.getShape()[axis] == (*refined).getShape()[axis])
      continue;
    int64_t dimension = cast<AxisMapAttr>(mapping).getDimensionId();
    if (dimension <= 0) {
      operation->emitOpError(
          "pointwise result refinement has no logical dimension authority");
      return WalkResult::interrupt();
    }
    if (failed(retargetSourceExtent(
            operation->getResult(0),
            sourceAxisIdentity(cast<AxisMapAttr>(target.getAxisMaps()[axis])),
            cast<PhysicalExprAttr>((*refined).getShape()[axis]),
            cast<AxisMapAttr>(target.getAxisMaps()[axis]).getDimensionId(),
            changes.typeChanged(), &changes)))
      return WalkResult::interrupt();
  }
  target = *refined;
  changes.setType(operation->getResult(0), target);
  for (unsigned index = 0; index < operation->getNumOperands(); ++index)
    if (failed(align(operation, index, target))) {
      operation->emitOpError(
          "pointwise operand cannot adopt the result relation")
          << "; operand_index=" << index
          << "; operand=" << operation->getOperand(index).getType()
          << "; result=" << target;
      return WalkResult::interrupt();
    }
  return WalkResult::advance();
}

LogicalResult refreshReshapeRelation(ReshapeOp reshape,
                                     RelationWorklist &changes) {
  // The operation owns its row-major groups. Refinement transports extents
  // through that existing relation; it does not infer a new reassociation.
  auto relations = queryFragmentOperandRelations(reshape);
  if (failed(relations))
    return reshape.emitOpError("reshape has no physical operand relation");
  auto result = transportFragmentResultType(
      *relations, reshape->getOperandTypes(), reshape.getResult().getType());
  if (failed(result))
    return reshape.emitOpError(
        "reshape cannot transport its physical operand schema");
  changes.setType(reshape.getResult(), *result);
  return reshape.verify();
}

} // namespace intent::gpu::value_relations
