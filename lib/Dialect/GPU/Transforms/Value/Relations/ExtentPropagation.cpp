#include "Intent/Analysis/ControlFlow.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

using AxisSelector = llvm::function_ref<bool(AxisMapAttr)>;

FragmentType replaceExtent(FragmentType source, AxisSelector selects,
                           ArrayRef<Attribute> previousExtents,
                           PhysicalExprAttr extent) {
  SmallVector<Attribute> shape(source.getShape().begin(),
                               source.getShape().end());
  bool changed = false;
  for (auto [axis, attribute] : llvm::enumerate(source.getAxisMaps())) {
    if (!selects(cast<AxisMapAttr>(attribute)) ||
        !llvm::is_contained(previousExtents, source.getShape()[axis]))
      continue;
    shape[axis] = extent;
    changed = true;
  }
  if (!changed)
    return source;
  return FragmentType::get(source.getContext(), source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           source.getAxisMaps(), source.getValidity(),
                           source.getOwner());
}

Type replaceExtent(Type source, AxisSelector selects,
                   ArrayRef<Attribute> previousExtents,
                   PhysicalExprAttr extent) {
  if (auto fragment = dyn_cast<FragmentType>(source))
    return replaceExtent(fragment, selects, previousExtents, extent);
  auto record = dyn_cast<RecordType>(source);
  if (!record)
    return source;
  SmallVector<Attribute> fields;
  bool changed = false;
  for (Attribute attribute : record.getFieldTypes()) {
    Type field = cast<TypeAttr>(attribute).getValue();
    Type replacement = replaceExtent(field, selects, previousExtents, extent);
    fields.push_back(TypeAttr::get(replacement));
    changed |= replacement != field;
  }
  if (!changed)
    return source;
  return RecordType::get(source.getContext(), record.getFieldNames(),
                         ArrayAttr::get(source.getContext(), fields),
                         record.getOwner());
}

bool carriesExtent(Type type, AxisSelector selects,
                   ArrayRef<Attribute> extents) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps()))
      if (selects(cast<AxisMapAttr>(attribute)) &&
          llvm::is_contained(extents, fragment.getShape()[axis]))
        return true;
    return false;
  }
  auto record = dyn_cast<RecordType>(type);
  if (!record)
    return false;
  return llvm::any_of(record.getFieldTypes(), [&](Attribute attribute) {
    return carriesExtent(cast<TypeAttr>(attribute).getValue(), selects,
                         extents);
  });
}

bool carriesSelectedAxis(Type type, AxisSelector selects) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
      return selects(cast<AxisMapAttr>(attribute));
    });
  auto record = dyn_cast<RecordType>(type);
  return record &&
         llvm::any_of(record.getFieldTypes(), [&](Attribute attribute) {
           return carriesSelectedAxis(cast<TypeAttr>(attribute).getValue(),
                                      selects);
         });
}

void collectSelectedExtents(Type type, AxisSelector selects,
                            SmallVectorImpl<Attribute> &extents) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps()))
      if (selects(cast<AxisMapAttr>(attribute)) &&
          !llvm::is_contained(extents, fragment.getShape()[axis]))
        extents.push_back(fragment.getShape()[axis]);
    return;
  }
  if (auto record = dyn_cast<RecordType>(type))
    for (Attribute attribute : record.getFieldTypes())
      collectSelectedExtents(cast<TypeAttr>(attribute).getValue(), selects,
                             extents);
}

bool preservesIntroducedUnitAxis(Value value, AxisSelector selects) {
  auto reshape = value.getDefiningOp<ReshapeOp>();
  auto result = dyn_cast<FragmentType>(value.getType());
  if (!result)
    return false;
  std::optional<unsigned> axis;
  for (Attribute attribute : result.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (!selects(mapping))
      continue;
    if (axis)
      return false;
    axis = mapping.getFragmentAxis();
  }
  if (!axis)
    return false;
  auto extent = cast<PhysicalExprAttr>(result.getShape()[*axis]);
  bool unit =
      extent.getKind() == PhysicalExprKind::Constant && extent.getValue() == 1;
  if (!unit)
    return false;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (kernel && PhysicalProgramAnalysis(kernel)
                    .axisRealization(value, *axis)
                    .constructionScalarSeed)
    return false;
  if (!reshape) {
    bool expanded = false;
    for (Operation *user : value.getUsers()) {
      auto broadcast = dyn_cast<BroadcastOp>(user);
      if (!broadcast || broadcast.getValue() != value) {
        if (llvm::any_of(user->getResultTypes(), [&](Type type) {
              return carriesSelectedAxis(type, selects);
            }))
          return false;
        continue;
      }
      auto target = dyn_cast<FragmentType>(broadcast.getResult().getType());
      auto relations = queryFragmentOperandRelations(broadcast);
      if (!target || failed(relations))
        return false;
      std::optional<unsigned> targetAxis;
      for (const auto &group : relations->front().groups)
        if (group.sourceAxes.size() == 1 && group.sourceAxes.front() == *axis)
          targetAxis = group.resultAxes.front();
      if (!targetAxis)
        return false;
      expanded |= target.getShape()[*targetAxis] != result.getShape()[*axis];
    }
    return expanded;
  }
  return false;
}

bool selectsSegmentAxis(Type type, uint64_t axis, AxisSelector selects) {
  auto fragment = dyn_cast<FragmentType>(type);
  return fragment && axis < fragment.getAxisMaps().size() &&
         selects(cast<AxisMapAttr>(fragment.getAxisMaps()[axis]));
}

enum class StructuredBoundary { Parent, Child, Result };

void appendStructuredRelations(StructuredOpInterface operation, Value value,
                               SmallVectorImpl<Value> &worklist,
                               AxisSelector selects,
                               StructuredBoundary boundary,
                               OpOperand *use = nullptr) {
  using K = StructuredRelationKind;
  auto relations = operation.getValueRelations();
  auto crossesSelectedAxis = [&](const StructuredValueRelation &relation) {
    Type type = relation.kind == K::SourceSlice ? relation.to.getType()
                                                : relation.from.getType();
    return llvm::any_of(relation.axes, [&](int64_t axis) {
      return selectsSegmentAxis(type, axis, selects);
    });
  };
  auto append = [&](Value related) {
    if (related != value && !llvm::is_contained(worklist, related))
      worklist.push_back(related);
  };
  bool regionOperation =
      operation.getStructuredKind() == StructuredOpKind::RegionFold ||
      operation.getStructuredKind() == StructuredOpKind::RegionScan;
  if (regionOperation) {
    for (const StructuredSchemaGroup &group :
         queryStructuredSchemaGroups(operation)) {
      bool connected =
          boundary == StructuredBoundary::Child
              ? use == group.producer || llvm::is_contained(group.yields, use)
          : boundary == StructuredBoundary::Parent
              ? llvm::is_contained(group.arguments, value)
              : llvm::is_contained(group.results, value);
      if (!connected)
        continue;
      // A seed may initialize several independent components or have other
      // numeric users. Only the component's operand slot is projected during
      // aggregate closure; never use the seed SSA value to join their schemas.
      append(group.producer->get());
      for (BlockArgument argument : group.arguments)
        append(argument);
      for (Value result : group.results)
        append(result);
      for (OpOperand *yield : group.yields)
        append(yield->get());
    }
  }
  if (boundary == StructuredBoundary::Child) {
    if (Region *emit = operation.getEmitRegion();
        emit && use && use->getOwner() == emit->front().getTerminator()) {
      bool memberAxis =
          llvm::any_of(operation.getEmitSources(), [&](Value source) {
            return llvm::any_of(
                operation.getIterationAxes(), [&](int64_t axis) {
                  return selectsSegmentAxis(source.getType(), axis, selects);
                });
          });
      // Emit yields are positional; identical SSA values at two yield slots do
      // not merge their outputs. The segmented member extent stays local to the
      // helper and must not replace the full output's member extent.
      if (!memberAxis)
        append(operation.getEmittedResults()[use->getOperandNumber()]);
    }
    for (const auto &relation : relations)
      if (relation.from == value &&
          (relation.kind == K::Capture ||
           (!regionOperation && (relation.kind == K::SameSchema ||
                                 relation.kind == K::Accumulator)) ||
           (relation.kind == K::SourceSlice && !crossesSelectedAxis(relation))))
        append(relation.to);
    return;
  }
  if (boundary == StructuredBoundary::Parent) {
    if (!regionOperation)
      return;
    for (const auto &relation : relations) {
      if (relation.to != value ||
          (relation.kind != K::Capture &&
           (relation.kind != K::SourceSlice || crossesSelectedAxis(relation))))
        continue;
      append(relation.from);
    }
    return;
  }
  if (operation.getStructuredKind() == StructuredOpKind::Scan)
    return;
  for (const auto &relation : relations) {
    if (relation.to != value)
      continue;
    if (relation.kind == K::Reduction) {
      if (!crossesSelectedAxis(relation))
        append(relation.from);
    } else if (relation.kind == K::Emission) {
      append(relation.from);
    } else if (relation.kind == K::SameSchema ||
               relation.kind == K::Accumulator) {
      if (regionOperation)
        continue;
      append(relation.from);
      if (operation.getStructuredKind() == StructuredOpKind::Reduce) {
        unsigned component = cast<OpResult>(value).getResultNumber();
        append(operation.getCombineLhs()[component]);
        append(operation.getCombineRhs()[component]);
        append(operation.getCombineYields()[component]);
      }
    }
  }
}

LogicalResult retargetExtent(Value root, AxisSelector selects,
                             PhysicalExprAttr extent,
                             bool followLogicalDimension,
                             ValueTypeChangeCallback changed,
                             OpBuilder::Listener *listener) {
  SmallVector<Attribute> previousExtents;
  collectSelectedExtents(root.getType(), selects, previousExtents);
  if (previousExtents.empty())
    return success();
  SmallVector<Attribute> connectedExtents(previousExtents.begin(),
                                          previousExtents.end());
  if (!llvm::is_contained(connectedExtents, Attribute(extent)))
    connectedExtents.push_back(extent);
  SmallVector<Value> worklist{root};
  llvm::SetVector<Value> boundaries;
  llvm::DenseMap<Value, Type> visitedTypes;
  SmallVector<std::pair<Value, AxisMapAttr>> valueAliases;
  auto isSegmentSourceSlice = [&](Value value) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument)
      return false;
    Operation *parent = argument.getOwner()->getParentOp();
    if (auto fold = dyn_cast_or_null<RegionFoldOp>(parent))
      return argument.getOwner()->getParent() == &fold.getSummarize() &&
             argument.getArgNumber() < fold.getSources().size() &&
             selectsSegmentAxis(argument.getType(), fold.getAxis(), selects);
    if (auto scan = dyn_cast_or_null<RegionScanOp>(parent))
      return (argument.getOwner()->getParent() == &scan.getSummarize() ||
              argument.getOwner()->getParent() == &scan.getEmit()) &&
             argument.getArgNumber() < scan.getSources().size() &&
             selectsSegmentAxis(argument.getType(), scan.getAxis(), selects);
    return false;
  };
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    auto previousFragment = dyn_cast<FragmentType>(value.getType());
    auto [visited, inserted] = visitedTypes.try_emplace(value, value.getType());
    if (!inserted && visited->second == value.getType())
      continue;
    visited->second = value.getType();
    // A structured segment slice has one exact extent authority: the segment
    // parameter owned by its region_fold/region_scan operation.  Retargeting
    // stops only for that segmented axis; every non-segment axis must preserve
    // the source schema across the helper boundary.
    if (isSegmentSourceSlice(value))
      continue;
    // Each make_range is an independent physical traversal authority.  A
    // source-axis extent selected for one range may flow through its users,
    // but must not cross a shared consumer and rewrite a different range that
    // happens to carry the same logical provenance.
    if (value != root && value.getDefiningOp<MakeRangeOp>())
      continue;
    // A reduction or other structured operation can consume an axis and
    // produce a scalar/record that no longer carries it.  The extent decision
    // ends there; following the scalar into a later broadcast would conflate a
    // new traversal with the one being retargeted.
    if (!(followLogicalDimension
              ? carriesSelectedAxis(value.getType(), selects)
              : carriesExtent(value.getType(), selects, connectedExtents)))
      continue;
    // The root is the value whose physical extent this decision owns.  A
    // construction-time singleton on that root is only a conservative seed;
    // it cannot veto the decision and leave a make_range type out of sync with
    // its extent operand.  The guard applies only while propagating through
    // downstream value flow, where a genuinely introduced unit axis must stay
    // scalar across an expanding broadcast.
    if (value == root || !preservesIntroducedUnitAxis(value, selects)) {
      SmallVector<Attribute> replaceableExtents(previousExtents.begin(),
                                                previousExtents.end());
      if (followLogicalDimension) {
        replaceableExtents.clear();
        collectSelectedExtents(value.getType(), selects, replaceableExtents);
      }
      auto replaceableAxis = [&](AxisMapAttr mapping) {
        return selects(mapping) &&
               !isIntroducedReshapeUnitAxis(value, mapping.getFragmentAxis());
      };
      Type replacement = replaceExtent(value.getType(), replaceableAxis,
                                       replaceableExtents, extent);
      if (replacement != value.getType()) {
        setPhysicalValueType(value, replacement, changed);
        // A make_range owns both the fragment schema and the SSA extent used
        // to materialize that schema.  Retarget them from the same physical
        // decision; leaving the operand behind creates two executable
        // authorities for one traversal.
        if (auto range = value.getDefiningOp<MakeRangeOp>()) {
          OpBuilder builder(range);
          builder.setListener(listener);
          Value physicalExtent;
          if (extent.getKind() == PhysicalExprKind::Constant)
            physicalExtent = builder.create<arith::ConstantIndexOp>(
                range.getLoc(), extent.getValue());
          else
            physicalExtent = builder.create<PhysicalExprOp>(
                range.getLoc(), builder.getIndexType(), extent);
          range->setOperand(1, physicalExtent);
        }
      }
    }
    // Product fields and structured helper arguments are part of the same
    // physical value flow even though MLIR does not connect them with ordinary
    // result uses.  A blocking decision for one provenance axis must cross
    // those boundaries; otherwise an operation can retain two physical
    // schemas for one summary/carry value.
    if (isa_and_nonnull<MakeRecordOp, ExtractOp>(value.getDefiningOp()))
      boundaries.insert(value);
    if (Operation *producer = value.getDefiningOp();
        isa_and_nonnull<FragmentOpInterface>(producer)) {
      auto relations = queryFragmentOperandRelations(producer);
      if (succeeded(relations))
        for (const auto &relation : *relations)
          if (relation.preservesSourceSchema)
            worklist.push_back(producer->getOperand(relation.operandNumber));
    }
    if (auto argument = dyn_cast<BlockArgument>(value))
      if (auto structured = dyn_cast<StructuredOpInterface>(
              argument.getOwner()->getParentOp()))
        appendStructuredRelations(structured, argument, worklist, selects,
                                  StructuredBoundary::Parent);
    Operation *boundaryOwner = value.getDefiningOp();
    if (!boundaryOwner)
      boundaryOwner = cast<BlockArgument>(value).getOwner()->getParentOp();
    bool controlTarget = isa<RegionBranchOpInterface>(boundaryOwner);
    if (auto argument = dyn_cast<BlockArgument>(value))
      controlTarget |= !argument.getOwner()->isEntryBlock();
    if (controlTarget) {
      auto incoming = queryControlFlowIncoming(value);
      if (!incoming.complete)
        return boundaryOwner->emitOpError(
            "cannot retarget a produced or unknown control schema slot");
      boundaries.insert(value);
      for (const ControlFlowEdge &edge : incoming.edges) {
        if (!edge.operand)
          return failure();
        auto outgoing = queryControlFlowOutgoing(*edge.operand);
        if (!outgoing.complete)
          return failure();
        for (const ControlFlowEdge &successor : outgoing.edges)
          worklist.push_back(successor.target);
      }
    }
    if (auto structured =
            dyn_cast_or_null<StructuredOpInterface>(value.getDefiningOp()))
      appendStructuredRelations(structured, value, worklist, selects,
                                StructuredBoundary::Result);
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto buffer = dyn_cast<BufferOp>(user);
          buffer && buffer.getInitialValue() == value && previousFragment) {
        auto storage = buffer.getResult().getType();
        auto initial = dyn_cast<FragmentType>(value.getType());
        if (initial && storage.getShape() == previousFragment.getShape() &&
            storage.getOwner() == previousFragment.getOwner() &&
            initial.getOwner() == storage.getOwner()) {
          bool padding = llvm::all_of(
              llvm::zip(previousFragment.getShape(), initial.getShape()),
              [](auto dimensions) {
                auto [before, after] = dimensions;
                if (before == after)
                  return true;
                auto oldExtent = cast<PhysicalExprAttr>(before);
                auto newExtent = cast<PhysicalExprAttr>(after);
                return oldExtent.getKind() == PhysicalExprKind::Constant &&
                       newExtent.getKind() == PhysicalExprKind::Constant &&
                       oldExtent.getValue() > 0 &&
                       static_cast<uint64_t>(newExtent.getValue()) ==
                           llvm::PowerOf2Ceil(
                               static_cast<uint64_t>(oldExtent.getValue()));
              });
          if (padding) {
            setPhysicalValueType(
                buffer.getResult(),
                BufferType::get(storage.getContext(), storage.getElementType(),
                                initial.getShape(), storage.getScope(),
                                storage.getInstance(), storage.getOwner(),
                                storage.getInitialization(),
                                storage.getVisibility()),
                changed);
          }
        }
      }
      if (previousFragment && isa<BroadcastOp, ReshapeOp>(user)) {
        Value result = user->getResult(0);
        auto target = dyn_cast<FragmentType>(result.getType());
        if (target) {
          // Query the operation with its pre-rewrite operand schema. Expansion
          // and one-to-one reassociation cannot be recovered from the mutated
          // source's extent alone.
          auto relations = queryFragmentOperandRelations(
              user, TypeRange{previousFragment}, target);
          bool broadcast = isa<BroadcastOp>(user);
          if (succeeded(relations) &&
              (!broadcast || relations->front().hasCompatibleExtents())) {
            for (const FragmentAxisGroup &group : relations->front().groups) {
              if (group.sourceAxes.size() != 1 ||
                  group.resultAxes.size() != 1 ||
                  (!broadcast &&
                   group.kind != FragmentAxisRelationKind::Reassociation))
                continue;
              unsigned sourceAxis = group.sourceAxes.front();
              unsigned targetAxis = group.resultAxes.front();
              auto inputMap =
                  cast<AxisMapAttr>(previousFragment.getAxisMaps()[sourceAxis]);
              auto targetMap =
                  cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
              auto oldExtent = cast<PhysicalExprAttr>(
                  previousFragment.getShape()[sourceAxis]);
              if (!selects(inputMap) || selects(targetMap) ||
                  oldExtent != target.getShape()[targetAxis] ||
                  target.getShape()[targetAxis] == extent)
                continue;
              if (broadcast) {
                bool unit = oldExtent.getKind() == PhysicalExprKind::Constant &&
                            oldExtent.getValue() <= 1;
                bool sameLogicalExtent =
                    inputMap.getDimensionId() > 0 &&
                    inputMap.getDimensionId() == targetMap.getDimensionId();
                if ((unit && !sameLogicalExtent) ||
                    isIntroducedReshapeUnitAxis(value, sourceAxis))
                  continue;
              }
              std::pair<Value, AxisMapAttr> alias{result, targetMap};
              if (!llvm::is_contained(valueAliases, alias))
                valueAliases.push_back(alias);
            }
          }
        }
      }
      if (isa<RegionFoldOp, RegionScanOp>(user)) {
        appendStructuredRelations(cast<StructuredOpInterface>(user), value,
                                  worklist, selects, StructuredBoundary::Child,
                                  &use);
        continue;
      }
      if (isa<RegionBranchOpInterface, RegionBranchTerminatorOpInterface,
              BranchOpInterface>(user)) {
        auto outgoing = queryControlFlowOutgoing(use);
        if (!outgoing.complete)
          return user->emitOpError(
              "cannot retarget an unknown control successor slot");
        for (const ControlFlowEdge &edge : outgoing.edges)
          worklist.push_back(edge.target);
        continue;
      }
      if (isa<YieldOp>(user))
        if (auto structured =
                dyn_cast<StructuredOpInterface>(user->getParentOp()))
          appendStructuredRelations(structured, value, worklist, selects,
                                    StructuredBoundary::Child, &use);
      if (isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
              ContractOp, ScaledContractOp, SparseContractOp, ReduceOp, ScanOp,
              RandomBitsOp, ScatterReduceOp, AtomicStoreOp, AtomicRMWOp,
              AtomicCompareExchangeOp>(user)) {
        for (Region &region : user->getRegions())
          for (Block &block : region)
            for (BlockArgument argument : block.getArguments())
              if (!isSegmentSourceSlice(argument))
                worklist.push_back(argument);
      }
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }
  for (Value boundary : boundaries)
    if (failed(projectSchemaBoundary(boundary, changed, listener)))
      return failure();
  for (auto [value, axis] : valueAliases) {
    auto type = cast<FragmentType>(value.getType());
    if (type.getShape()[axis.getFragmentAxis()] == extent)
      continue;
    if (failed(retargetExtent(
            value,
            [=](AxisMapAttr mapping) {
              return sourceAxisIdentity(mapping) == sourceAxisIdentity(axis) &&
                     mapping.getDimensionId() == axis.getDimensionId();
            },
            extent, /*followLogicalDimension=*/false, changed, listener)))
      return failure();
  }
  return success();
}

} // namespace

LogicalResult retargetSourceExtent(Value root, PhysicalSourceAxis source,
                                   PhysicalExprAttr extent,
                                   std::optional<int64_t> dimension,
                                   ValueTypeChangeCallback changed,
                                   OpBuilder::Listener *listener) {
  return retargetExtent(
      root,
      [=](AxisMapAttr mapping) {
        return mapping.getSourceId() == source.sourceId &&
               mapping.getSourceAxis() == source.sourceAxis &&
               mapping.getDerived() == source.derived &&
               (!dimension || mapping.getDimensionId() == *dimension);
      },
      extent, /*followLogicalDimension=*/false, changed, listener);
}

LogicalResult retargetDimensionExtent(Value root, int64_t dimensionId,
                                      PhysicalExprAttr extent,
                                      ValueTypeChangeCallback changed,
                                      OpBuilder::Listener *listener) {
  if (dimensionId <= 0)
    return success();
  return retargetExtent(
      root,
      [=](AxisMapAttr mapping) {
        return mapping.getDimensionId() == dimensionId;
      },
      extent, /*followLogicalDimension=*/true, changed, listener);
}

} // namespace intent::gpu
