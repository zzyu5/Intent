#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"
#include "Intent/Analysis/ControlFlow.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/ValueMaterialization.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include <algorithm>
#include <functional>
#include <optional>
#include <array>
#include <deque>
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace intent::gpu {

namespace {

// One queue owns relation invalidation while a transformation closes its IR.
// Rules never select a new ownership policy: they propagate an extent already
// selected by a range, operand, structured boundary or exact projection.
class RelationWorklist : public RewriterBase::Listener {
public:
  enum Rule : unsigned {
    Captures, ReductionResult, ReductionIdentity, Aggregate, AccessResult,
    Pointwise, ReductionYield, AccessValue, Reshape, ContractOperands,
    ContractAccumulator, RuleCount
  };

  RelationWorklist(func::FuncOp kernel, ValueRelationScope scope)
      : kernel(kernel), scope(scope), callback([this](Value value, Type previous) {
          typeChanged(value, previous);
        }) {}

  ValueTypeChangeCallback typeChanged() { return callback; }

  void setType(Value value, Type type) {
    setPhysicalValueType(value, type, callback);
  }

  void notifyOperationInserted(Operation *operation,
                               OpBuilder::InsertPoint) override {
    operation->walk<WalkOrder::PreOrder>([&](Operation *nested) {
      live.insert(nested);
      enqueue(nested);
    });
  }

  void notifyOperationErased(Operation *operation) override {
    // Removing a use can change projection/authority queries on its producers.
    for (Value operand : operation->getOperands()) affected(operand);
    enqueue(operation->getParentOp());
    operation->walk([&](Operation *nested) {
      live.erase(nested);
      for (auto &pending : queued) pending.erase(nested);
      for (Value result : nested->getResults()) transitions.erase(result);
      for (Region &region : nested->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            transitions.erase(argument);
    });
  }

  void notifyOperationReplaced(Operation *operation,
                               ValueRange replacements) override {
    // Rewriter notifications precede RAUW, so the original result users are
    // still available here. They will observe the new edges when dequeued.
    for (Value result : operation->getResults()) affected(result);
    for (Value value : replacements) affected(value);
  }

  void notifyOperationModified(Operation *operation) override {
    enqueue(operation);
    for (Value operand : operation->getOperands()) affected(operand);
    for (Value result : operation->getResults()) affected(result);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments()) affected(argument);
  }

  LogicalResult run();

private:
  bool enabled(Rule rule) const {
    switch (scope) {
    case ValueRelationScope::Complete: return true;
    case ValueRelationScope::Pointwise:
      return rule == Pointwise || rule == Reshape ||
             rule == ReductionResult || rule == ReductionIdentity;
    case ValueRelationScope::Contracts:
      return rule == ContractOperands || rule == ContractAccumulator;
    case ValueRelationScope::AccessResults: return rule == AccessResult;
    case ValueRelationScope::ReductionInputs:
      return rule == ReductionResult || rule == ReductionIdentity;
    }
    llvm_unreachable("unknown value relation scope");
  }

  void push(Operation *operation, Rule rule) {
    if (enabled(rule) && queued[rule].insert(operation).second)
      worklists[priority(rule)].emplace_back(operation, rule);
  }

  static unsigned priority(Rule rule) {
    // Access results, reshapes and elementwise values share the same SSA queue:
    // a reshape must run between its producer and its consumers, not in a
    // separate whole-program sweep after those consumers have been visited.
    if (rule == AccessResult || rule == Pointwise || rule == Reshape)
      return AccessResult;
    return rule;
  }

  void enqueue(Operation *operation) {
    if (!operation || !live.contains(operation)) return;
    if (isa<RegionFoldOp, RegionScanOp>(operation)) push(operation, Captures);
    if (isa<ReduceOp, ScanOp>(operation)) {
      push(operation, ReductionResult);
      push(operation, ReductionIdentity);
      push(operation, ReductionYield);
    }
    if (hasSchemaBoundary(operation))
      push(operation, Aggregate);
    if (auto access = dyn_cast<AccessOpInterface>(operation)) {
      AccessKind kind = access.getAccessKind();
      if (kind == AccessKind::Load || kind == AccessKind::Gather)
        push(operation, AccessResult);
      if (kind == AccessKind::Load || kind == AccessKind::Gather ||
          kind == AccessKind::Store)
        push(operation, AccessValue);
    }
    if (isa<FragmentOpInterface>(operation) &&
        !isa<SplatOp, ReshapeOp>(operation))
      push(operation, Pointwise);
    if (isa<ReshapeOp>(operation)) push(operation, Reshape);
    if (isa<ContractOp>(operation)) push(operation, ContractOperands);
    if (isa<ContractOp, ScaledContractOp, SparseContractOp>(operation))
      push(operation, ContractAccumulator);
    // A yield/capture change affects its structured owner even though those
    // schema edges are not ordinary SSA result uses.
    Operation *parent = operation->getParentOp();
    if (parent && parent != kernel) enqueue(parent);
  }

  void affected(Value value) {
    if (auto argument = dyn_cast<BlockArgument>(value))
      enqueue(argument.getOwner()->getParentOp());
    else
      enqueue(value.getDefiningOp());
    for (Operation *user : value.getUsers()) enqueue(user);
  }

  void typeChanged(Value value, Type previous) {
    if (previous == value.getType()) return;
    auto transition = std::make_pair(previous, value.getType());
    auto &history = transitions[value];
    // Detect an actual repeated conflicting refinement, rather than silently
    // stopping after an arbitrary number of iterations.
    if (llvm::is_contained(history, transition)) {
      if (!conflict) {
        Operation *owner = value.getDefiningOp();
        if (!owner) owner = cast<BlockArgument>(value).getOwner()->getParentOp();
        owner->emitOpError("physical relation closure repeated a conflicting type refinement")
            << "; previous=" << previous << "; selected=" << value.getType();
      }
      conflict = true;
    } else {
      history.push_back(transition);
    }
    affected(value);
  }

  func::FuncOp kernel;
  ValueRelationScope scope;
  std::function<void(Value, Type)> callback;
  std::array<std::deque<std::pair<Operation *, Rule>>, RuleCount> worklists;
  std::array<DenseSet<Operation *>, RuleCount> queued;
  DenseSet<Operation *> live;
  DenseMap<Value, SmallVector<std::pair<Type, Type>>> transitions;
  bool conflict = false;
};

LogicalResult alignContractAccumulator(Operation *operation, RelationWorklist &changes) {
  auto align = [&](Operation *owner, OpOperand &accumulatorOperand,
                   Value result) -> LogicalResult {
    Value accumulator = accumulatorOperand.get();
    if (accumulator.getType() == result.getType())
      return success();
    auto source = dyn_cast<FragmentType>(accumulator.getType());
    auto target = dyn_cast<FragmentType>(result.getType());
    if (!source || !target || source.getElementType() != target.getElementType() ||
        source.getOwner() != target.getOwner())
      return owner->emitOpError(
          "pointwise ownership cannot preserve the contract accumulator relation");
    if (isLiteralZeroProjection(accumulator)) {
      OpBuilder builder(owner);
      builder.setListener(&changes);
      auto projected = projectPhysicalValueToSchema(
          builder, owner->getLoc(), accumulator, target, changes.typeChanged());
      if (failed(projected))
        return owner->emitOpError("contract zero accumulator cannot adopt its result schema");
      accumulatorOperand.set(*projected);
      return success();
    }
    if (source.getAxisMaps() != target.getAxisMaps())
      return owner->emitOpError(
          "pointwise ownership cannot preserve the contract accumulator relation");
    auto isUnit = [](Attribute attribute) {
      auto expression = cast<PhysicalExprAttr>(attribute);
      return expression.getKind() ==
                 PhysicalExprKind::Constant &&
             expression.getValue() == 1;
    };
    SmallVector<Attribute> shape(target.getShape().begin(),
                                 target.getShape().end());
    for (unsigned axis = 0; axis < shape.size(); ++axis) {
      if (source.getShape()[axis] == target.getShape()[axis])
        continue;
      bool sourceUnit = isUnit(source.getShape()[axis]);
      bool targetUnit = isUnit(target.getShape()[axis]);
      if (sourceUnit == targetUnit)
        return owner->emitOpError(
            "pointwise ownership found two non-equivalent contract extents");
      if (targetUnit)
        shape[axis] = source.getShape()[axis];
    }
    auto aligned = FragmentType::get(
        target.getContext(), target.getElementType(),
        ArrayAttr::get(target.getContext(), shape), target.getAxisMaps(),
        target.getValidity(), target.getOwner());
    for (auto [axis, mapping] : llvm::enumerate(aligned.getAxisMaps())) {
      if (source.getShape()[axis] == aligned.getShape()[axis] &&
          target.getShape()[axis] == aligned.getShape()[axis])
        continue;
      int64_t dimension = cast<AxisMapAttr>(mapping).getDimensionId();
      if (dimension <= 0)
        return owner->emitOpError(
            "contract accumulator alignment has no dimension authority");
      if (failed(retargetDimensionExtent(
          result, dimension,
          cast<PhysicalExprAttr>(aligned.getShape()[axis]), changes.typeChanged(), &changes)))
        return failure();
      if (failed(retargetDimensionExtent(
          accumulator, dimension,
          cast<PhysicalExprAttr>(aligned.getShape()[axis]), changes.typeChanged(), &changes)))
        return failure();
    }
    changes.setType(accumulator, aligned);
    changes.setType(result, aligned);
    return success();
  };
      OpOperand *accumulator;
      Value output;
      if (auto contract = dyn_cast<ContractOp>(operation)) {
        accumulator = &contract.getAccumulatorMutable();
        output = contract.getResult();
      } else if (auto contract = dyn_cast<ScaledContractOp>(operation)) {
        accumulator = &contract.getAccumulatorMutable();
        output = contract.getResult();
      } else if (auto contract = dyn_cast<SparseContractOp>(operation)) {
        accumulator = &contract.getAccumulatorMutable();
        output = contract.getResult();
      } else {
        return success();
      }
      return align(operation, *accumulator, output);

}

WalkResult alignContractOperands(Operation *operation, RelationWorklist &changes) {
  auto contract = dyn_cast<ContractOp>(operation);
  if (!contract) return WalkResult::advance();
  auto kernel = operation->getParentOfType<func::FuncOp>();
  auto isUnit = [](Attribute attribute) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    return extent.getKind() ==
               PhysicalExprKind::Constant &&
           extent.getValue() == 1;
  };
  auto isUniformBatch = [&](Value value, unsigned axis) {
    auto broadcast = value.getDefiningOp<BroadcastOp>();
    auto source = broadcast
                      ? dyn_cast<FragmentType>(broadcast.getValue().getType())
                      : FragmentType();
    auto target = cast<FragmentType>(value.getType());
    if (!source)
      return false;
    auto relations = queryFragmentOperandRelations(broadcast);
    auto mapping = cast<AxisMapAttr>(target.getAxisMaps()[axis]);
    if (failed(relations) || !relations->front().hasCompatibleExtents())
      return false;
    const auto *group = relations->front().groupForResultAxis(axis);
    if (!group || !group->sourceAxes.empty() ||
        queryFragmentAxes(target, sourceAxisIdentity(mapping)).size() != 1)
      return false;
    auto ranges = PhysicalProgramAnalysis(kernel).axisRanges(value, axis);
    return ranges.isExact() && ranges.roots.empty() && ranges.blockers.empty();
  };
  auto batchExtentAuthority = [&](Value value, AxisMapAttr mapping) {
    Value authority = value;
    while (Operation *operation = authority.getDefiningOp()) {
      if (!isa<ReshapeOp, TransposeOp>(operation))
        break;
      Value source = operation->getOperand(0);
      if (!queryFragmentAxis(source.getType(), sourceAxisIdentity(mapping),
                             mapping.getDimensionId()).isExact())
        return value;
      authority = source;
    }
    auto load = authority.getDefiningOp<LoadOp>();
    if (!load)
      return value;
    PhysicalProgramAnalysis analysis(kernel);
    for (Value dependency : load->getOperands()) {
      if (dependency == load.getResource() ||
          !isa<FragmentType>(dependency.getType()))
        continue;
      if (queryFragmentAxes(dependency.getType(),
                            sourceAxisIdentity(mapping)).empty())
        continue;
      auto axis = queryFragmentAxis(dependency.getType(),
                                    sourceAxisIdentity(mapping),
                                    mapping.getDimensionId());
      if (!axis.isExact())
        return value;
      auto ranges = analysis.axisRanges(dependency, axis.fragmentAxis);
      if (!ranges.isExact() || !ranges.roots.empty() || !ranges.blockers.empty())
        return value;
    }
    return authority;
  };

    auto alignPairs = [&](Value lhs, Value rhs, ArrayRef<int64_t> lhsAxes,
                          ArrayRef<int64_t> rhsAxes, bool batch) -> LogicalResult {
      if (lhsAxes.size() != rhsAxes.size())
        return failure();
      for (auto [lhsAxis, rhsAxis] : llvm::zip(lhsAxes, rhsAxes)) {
        auto lhsType = cast<FragmentType>(lhs.getType());
        auto rhsType = cast<FragmentType>(rhs.getType());
        if (lhsAxis < 0 || rhsAxis < 0 ||
            lhsAxis >= static_cast<int64_t>(lhsType.getShape().size()) ||
            rhsAxis >= static_cast<int64_t>(rhsType.getShape().size()))
          return failure();
        Attribute lhsExtent = lhsType.getShape()[lhsAxis];
        Attribute rhsExtent = rhsType.getShape()[rhsAxis];
        if (lhsExtent == rhsExtent)
          continue;
        bool lhsUnit = isUnit(lhsExtent);
        bool rhsUnit = isUnit(rhsExtent);
        bool rebindLhs = lhsUnit;
        bool lhsUniform = batch && isUniformBatch(lhs, lhsAxis);
        bool rhsUniform = batch && isUniformBatch(rhs, rhsAxis);
        if (lhsUnit == rhsUnit && lhsUniform == rhsUniform)
          return contract.emitOpError(
                     "ordinary contract paired axes have conflicting physical extents")
                 << "; lhs_axis=" << lhsAxis << "; lhs_extent=" << lhsExtent
                 << "; rhs_axis=" << rhsAxis << "; rhs_extent=" << rhsExtent;
        if (lhsUnit == rhsUnit)
          rebindLhs = lhsUniform;
        if (rebindLhs) {
          auto mapping = cast<AxisMapAttr>(lhsType.getAxisMaps()[lhsAxis]);
          Value authority = batch ? batchExtentAuthority(lhs, mapping) : lhs;
          if (failed(retargetSourceExtent(authority, sourceAxisIdentity(mapping),
                               cast<PhysicalExprAttr>(rhsExtent),
                               mapping.getDimensionId(), changes.typeChanged(), &changes)))
            return failure();
        } else {
          auto mapping = cast<AxisMapAttr>(rhsType.getAxisMaps()[rhsAxis]);
          Value authority = batch ? batchExtentAuthority(rhs, mapping) : rhs;
          if (failed(retargetSourceExtent(authority, sourceAxisIdentity(mapping),
                               cast<PhysicalExprAttr>(lhsExtent),
                               mapping.getDimensionId(), changes.typeChanged(), &changes)))
            return failure();
        }
      }
      return success();
    };
    if (failed(alignPairs(contract.getLhs(), contract.getRhs(),
                          contract.getLhsReductionAxes(),
                          contract.getRhsReductionAxes(), false)) ||
        failed(alignPairs(contract.getLhs(), contract.getRhs(),
                          contract.getLhsBatchAxes(),
                          contract.getRhsBatchAxes(), true)))
      return WalkResult::interrupt();
    return WalkResult::advance();

}

using AxisSelector = llvm::function_ref<bool(AxisMapAttr)>;

FragmentType replaceExtent(FragmentType source, AxisSelector selects,
                           ArrayRef<Attribute> previousExtents,
                           PhysicalExprAttr extent) {
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
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
  return FragmentType::get(
      source.getContext(), source.getElementType(),
      ArrayAttr::get(source.getContext(), shape), source.getAxisMaps(),
      source.getValidity(), source.getOwner());
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
    return carriesExtent(cast<TypeAttr>(attribute).getValue(), selects, extents);
  });
}

bool carriesSelectedAxis(Type type, AxisSelector selects) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
      return selects(cast<AxisMapAttr>(attribute));
    });
  auto record = dyn_cast<RecordType>(type);
  return record && llvm::any_of(record.getFieldTypes(), [&](Attribute attribute) {
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
  bool unit = extent.getKind() ==
                  PhysicalExprKind::Constant &&
              extent.getValue() == 1;
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
                               AxisSelector selects, StructuredBoundary boundary,
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
  bool regionOperation = operation.getStructuredKind() == StructuredOpKind::RegionFold ||
                         operation.getStructuredKind() == StructuredOpKind::RegionScan;
  if (regionOperation) {
    for (const StructuredSchemaGroup &group : queryStructuredSchemaGroups(operation)) {
      bool connected = boundary == StructuredBoundary::Child
          ? use == group.producer || llvm::is_contained(group.yields, use)
          : boundary == StructuredBoundary::Parent
              ? llvm::is_contained(group.arguments, value)
              : llvm::is_contained(group.results, value);
      if (!connected) continue;
      // A seed may initialize several independent components or have other
      // numeric users. Only the component's operand slot is projected during
      // aggregate closure; never use the seed SSA value to join their schemas.
      append(group.producer->get());
      for (BlockArgument argument : group.arguments) append(argument);
      for (Value result : group.results) append(result);
      for (OpOperand *yield : group.yields) append(yield->get());
    }
  }
  if (boundary == StructuredBoundary::Child) {
    if (Region *emit = operation.getEmitRegion();
        emit && use && use->getOwner() == emit->front().getTerminator()) {
      bool memberAxis = llvm::any_of(operation.getEmitSources(), [&](Value source) {
        return llvm::any_of(operation.getIterationAxes(), [&](int64_t axis) {
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
           (!regionOperation && (relation.kind == K::SameSchema || relation.kind == K::Accumulator)) ||
           (relation.kind == K::SourceSlice && !crossesSelectedAxis(relation))))
        append(relation.to);
    return;
  }
  if (boundary == StructuredBoundary::Parent) {
    if (!regionOperation) return;
    for (const auto &relation : relations) {
      if (relation.to != value ||
          (relation.kind != K::Capture &&
           (relation.kind != K::SourceSlice || crossesSelectedAxis(relation))))
        continue;
      append(relation.from);
    }
    return;
  }
  if (operation.getStructuredKind() == StructuredOpKind::Scan) return;
  for (const auto &relation : relations) {
    if (relation.to != value) continue;
    if (relation.kind == K::Reduction) {
      if (!crossesSelectedAxis(relation)) append(relation.from);
    } else if (relation.kind == K::Emission) {
      append(relation.from);
    } else if (relation.kind == K::SameSchema || relation.kind == K::Accumulator) {
      if (regionOperation) continue;
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

} // namespace

static WalkResult alignReductionResultRelation(Operation *operation, RelationWorklist &changes) {
    auto kernel = operation->getParentOfType<func::FuncOp>();
    SmallVector<Value> sources;
    SmallVector<Value> results;
    llvm::SmallDenseSet<int64_t> reducedAxes;
    bool scan = false;
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      llvm::append_range(sources, reduce.getSources());
      results.append(reduce.getResults().begin(), reduce.getResults().end());
      reducedAxes.insert(reduce.getAxes().begin(), reduce.getAxes().end());
    } else if (auto currentScan = dyn_cast<ScanOp>(operation)) {
      llvm::append_range(sources, currentScan.getSources());
      results.append(currentScan.getResults().begin(),
                     currentScan.getResults().end());
      scan = true;
    } else {
      return WalkResult::advance();
    }
    if (sources.size() != results.size())
      return WalkResult::interrupt();
    for (auto [index, source] : llvm::enumerate(sources)) {
      auto sourceType = dyn_cast<FragmentType>(source.getType());
      if (!sourceType || llvm::all_of(sources, [&](Value other) {
            auto type = dyn_cast<FragmentType>(other.getType());
            return type && type.getShape() == sourceType.getShape();
          }))
        continue;
      SmallVector<Value> related;
      llvm::copy_if(sources, std::back_inserter(related), [&](Value other) {
        auto type = dyn_cast<FragmentType>(other.getType());
        return type && type.getShape().size() == sourceType.getShape().size() &&
               queryAxisProjection(type, sourceType).isExact();
      });
      FailureOr<FragmentType> refined =
          queryValueSchema(kernel, sourceType, related);
      if (failed(refined)) {
        operation->emitOpError(
            "tuple reduction sources have no common physical extent relation");
        return WalkResult::interrupt();
      }
      auto target = FragmentType::get(
          kernel.getContext(), sourceType.getElementType(), (*refined).getShape(),
          sourceType.getAxisMaps(), sourceType.getValidity(), sourceType.getOwner());
      OpBuilder builder(operation);
      builder.setListener(&changes);
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, operation->getLoc(), source, target, changes.typeChanged());
      if (failed(projected)) {
        operation->emitOpError(
            "tuple reduction source cannot adopt its physical extent relation");
        return WalkResult::interrupt();
      }
      operation->setOperand(index, *projected);
      sources[index] = *projected;
    }
    for (auto [source, currentResult] : llvm::zip_equal(sources, results)) {
      if (scan) {
        changes.setType(currentResult, source.getType());
        continue;
      }
      Type element = currentResult.getType();
      if (auto fragment = dyn_cast<FragmentType>(element)) element = fragment.getElementType();
      SmallVector<int64_t> axes(reducedAxes.begin(), reducedAxes.end());
      auto type = inferCollectiveResultType(source.getType(), axes, element);
      if (failed(type)) {
        operation->emitOpError("physical reduction has no valid source/result axis schema");
        return WalkResult::interrupt();
      }
      changes.setType(currentResult, *type);
    }
    return WalkResult::advance();
}


static WalkResult alignReductionIdentityRelation(Operation *operation, RelationWorklist &changes) {
  auto structured = cast<StructuredOpInterface>(operation);
  auto identities = isa<ReduceOp>(operation) ? cast<ReduceOp>(operation).getIdentities()
                                            : cast<ScanOp>(operation).getIdentities();
  OpBuilder builder(operation);
  builder.setListener(&changes);
  for (auto [index, identity] : llvm::enumerate(identities)) {
    Type target = operation->getResult(index).getType();
    auto projected = projectPhysicalValueToSchema(builder, operation->getLoc(), identity,
                                                   target, changes.typeChanged());
    if (failed(projected)) {
      operation->emitOpError("physical reduction identity cannot adopt its result relation");
      return WalkResult::interrupt();
    }
    operation->setOperand(identities.getBeginOperandIndex() + index, *projected);
    changes.setType(structured.getCombineLhs()[index], target);
    changes.setType(structured.getCombineRhs()[index], target);
  }
  return WalkResult::advance();
}


WalkResult alignReductionYield(Operation *operation, RelationWorklist &changes) {
    Region *combine = nullptr;
    ValueRange results;
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      combine = &reduce.getCombine();
      results = reduce.getResults();
    } else if (auto scan = dyn_cast<ScanOp>(operation)) {
      combine = &scan.getCombine();
      results = scan.getResults();
    } else {
      return WalkResult::advance();
    }
    if (!combine || !llvm::hasSingleElement(*combine))
      return WalkResult::interrupt();
    auto yield = dyn_cast<YieldOp>(combine->front().getTerminator());
    if (!yield || yield.getValues().size() != results.size())
      return WalkResult::interrupt();
    OpBuilder builder(yield);
      builder.setListener(&changes);
    for (auto [index, target] : llvm::enumerate(results)) {
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, operation->getLoc(), yield.getValues()[index],
          target.getType(), changes.typeChanged());
      if (failed(projected)) {
        operation->emitOpError(
            "physical reduction yield cannot adopt its result relation")
            << "; result_index=" << index
            << "; yield_type=" << yield.getValues()[index].getType()
            << "; result_type=" << target.getType();
        return WalkResult::interrupt();
      }
      yield->setOperand(index, *projected);
    }
    return WalkResult::advance();

}

WalkResult alignAccessResult(Operation *operation, RelationWorklist &changes) {
  auto kernel = operation->getParentOfType<func::FuncOp>();
    auto access = cast<AccessOpInterface>(operation);
    ValueRange coordinates = access.getAccessCoordinates();
    Value result = access.getAccessResult();
    auto current = dyn_cast<FragmentType>(result.getType());
    if (!current)
      return WalkResult::advance();
    FailureOr<FragmentType> refined =
        queryAccessResultSchema(kernel, current, coordinates);
    if (failed(refined)) {
      InFlightDiagnostic diagnostic = operation->emitOpError(
          "access result has no unique physical coordinate projection");
      diagnostic << "; result=" << current;
      for (Value coordinate : coordinates)
        diagnostic << "; coordinate=" << coordinate.getType();
      return WalkResult::interrupt();
    }
    for (auto [axis, mapping] : llvm::enumerate((*refined).getAxisMaps())) {
      if (axis >= current.getShape().size() ||
          current.getShape()[axis] == (*refined).getShape()[axis])
        continue;
      if (failed(retargetSourceExtent(
          result, sourceAxisIdentity(cast<AxisMapAttr>(mapping)),
          cast<PhysicalExprAttr>((*refined).getShape()[axis]),
          cast<AxisMapAttr>(mapping).getDimensionId(), changes.typeChanged(), &changes)))
        return WalkResult::interrupt();
    }
    changes.setType(result, *refined);
    return WalkResult::advance();

}

static LogicalResult refreshReshapeRelation(ReshapeOp reshape, RelationWorklist &changes);

WalkResult alignPointwiseValue(Operation *operation, RelationWorklist &changes) {
  auto kernel = operation->getParentOfType<func::FuncOp>();
  auto isParameterExtent = [](Attribute attribute) {
    auto extent = dyn_cast<PhysicalExprAttr>(attribute);
    return extent &&
           extent.getKind() ==
               PhysicalExprKind::Parameter;
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
        targetShape.getShape(), targetShape.getAxisMaps(), targetShape.getValidity(),
        targetShape.getOwner());
    OpBuilder builder(operation);
      builder.setListener(&changes);
    FailureOr<Value> replacement =
        projectPhysicalValueToSchema(builder, operation->getLoc(), value, target, changes.typeChanged());
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
      auto result = transportFragmentResultType(
          *relations, transpose->getOperandTypes(), transpose.getResult().getType());
      if (failed(result)) {
        transpose.emitOpError("transpose cannot transport its physical operand schema");
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
            builder, broadcast.getLoc(), broadcast.getValue(), target, changes.typeChanged());
        if (succeeded(projected)) {
          broadcast->setOperand(0, *projected);
          source = cast<FragmentType>(projected->getType());
        }
      }
      relations = queryFragmentOperandRelations(broadcast);
      if (failed(relations)) {
        broadcast.emitOpError("broadcast has no unique physical source projection")
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
        auto sourceExtent =
            cast<PhysicalExprAttr>(source.getShape()[sourceAxis]);
        bool singleton =
            sourceExtent.getKind() ==
                PhysicalExprKind::Constant &&
            sourceExtent.getValue() == 1;
        if (singleton)
          continue;
        auto targetExtent =
            cast<PhysicalExprAttr>(target.getShape()[targetAxis]);
        bool targetSingleton =
            targetExtent.getKind() ==
                PhysicalExprKind::Constant &&
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
                                targetMap.getDimensionId()).isExact() &&
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
            if (failed(retargetSourceExtent(broadcast.getResult(),
                                 sourceAxisIdentity(targetMap), sourceExtent, std::nullopt, changes.typeChanged(), &changes)))
              return WalkResult::interrupt();
            target = cast<FragmentType>(broadcast.getResult().getType());
            continue;
          }
          broadcast.emitOpError(
              "broadcast cannot contract a non-singleton physical axis")
              << "; input=" << source << "; result=" << target;
          return WalkResult::interrupt();
        }
        bool sourceParameter =
            isParameterExtent(source.getShape()[sourceAxis]);
        bool targetParameter = isParameterExtent(targetExtent);
        if (!sourceParameter && !targetParameter) {
          PhysicalProgramAnalysis analysis(kernel);
          auto input = analysis.axisRealization(broadcast.getValue(), sourceAxis);
          auto output = analysis.axisRealization(broadcast.getResult(), targetAxis);
          if (input.hasExtentAuthority() && !input.constructionScalarSeed &&
              !output.hasExtentAuthority()) {
            auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
            if (failed(retargetSourceExtent(broadcast.getResult(),
                                 sourceAxisIdentity(targetMap), sourceExtent, std::nullopt, changes.typeChanged(), &changes)))
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
        auto sourceMap =
            cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
        auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
        if (sourceMap.getDimensionId() <= 0 || targetMap.getDimensionId() <= 0) {
          broadcast.emitOpError(
              "broadcast extent refinement has no logical occurrence authority");
          return WalkResult::interrupt();
        }
        if (targetParameter) {
          if (failed(retargetSourceExtent(
              broadcast.getValue(), sourceAxisIdentity(sourceMap),
              cast<PhysicalExprAttr>(target.getShape()[targetAxis]), std::nullopt, changes.typeChanged(), &changes)))
            return WalkResult::interrupt();
        } else {
          if (failed(retargetSourceExtent(broadcast.getResult(),
                               sourceAxisIdentity(targetMap), sourceExtent, std::nullopt, changes.typeChanged(), &changes)))
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
      changes.setType(broadcast.getResult(), FragmentType::get(
          kernel.getContext(), target.getElementType(), (*refined).getShape(),
          target.getAxisMaps(), target.getValidity(), target.getOwner()));
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
        target = FragmentType::get(
            kernel.getContext(), operation->getResult(0).getType(),
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
        operation->emitOpError("pointwise operation has no physical operand relation");
        return WalkResult::interrupt();
      }
      auto result = transportFragmentResultType(
          *relations, operation->getOperandTypes(), target);
      if (failed(result)) {
        operation->emitOpError("pointwise operation cannot transport its physical operand schema");
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
          cast<AxisMapAttr>(target.getAxisMaps()[axis]).getDimensionId(), changes.typeChanged(), &changes)))
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

LogicalResult alignAccessValue(Operation *operation, RelationWorklist &changes) {
  auto kernel = operation->getParentOfType<func::FuncOp>();
  auto predicateType = [&](FragmentType value) {
    return FragmentType::get(
        kernel.getContext(), IntegerType::get(kernel.getContext(), 1),
        value.getShape(), value.getAxisMaps(), value.getValidity(),
        value.getOwner());
  };
  auto project = [&](OpBuilder &builder, Location location, Value value,
                     Type target) -> FailureOr<Value> {
    if (!value)
      return failure();
    return projectPhysicalValueToSchema(builder, location, value, target, changes.typeChanged());
  };
  auto alignCoordinates = [&](AccessOpInterface access, Type valueType) -> LogicalResult {
    auto payload = dyn_cast<FragmentType>(valueType);
    if (!payload)
      return success();
    PhysicalProgramAnalysis analysis(kernel);
    for (auto [slot, coordinate] : llvm::enumerate(access.getAccessCoordinates())) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      if (!type)
        continue;
      SmallVector<Attribute> shape(type.getShape().begin(), type.getShape().end());
      bool changed = false;
      for (auto [axis, attribute] : llvm::enumerate(type.getAxisMaps())) {
        auto mapping = cast<AxisMapAttr>(attribute);
        auto projection = queryFragmentAxis(payload, sourceAxisIdentity(mapping));
        if (!projection.isExact() || projection.dimensionId != mapping.getDimensionId() ||
            shape[axis] == payload.getShape()[projection.fragmentAxis])
          continue;
        PhysicalRangeFact ranges = analysis.axisRanges(coordinate, axis);
        if (!ranges.isExact() || !ranges.roots.empty())
          continue;
        shape[axis] = payload.getShape()[projection.fragmentAxis];
        changed = true;
      }
      auto permutation = queryAxisPermutation(type, payload);
      bool permuted = permutation &&
          llvm::any_of(llvm::enumerate(*permutation), [](auto item) {
            return item.value() != static_cast<int64_t>(item.index());
          });
      if (!changed && !permuted)
        continue;
      auto target = FragmentType::get(
          kernel.getContext(), type.getElementType(),
          permuted ? payload.getShape()
                   : ArrayAttr::get(kernel.getContext(), shape),
          permuted ? payload.getAxisMaps() : type.getAxisMaps(),
          type.getValidity(), type.getOwner());
      OpBuilder builder(access);
      builder.setListener(&changes);
      FailureOr<Value> aligned = project(builder, access.getLoc(), coordinate, target);
      if (failed(aligned))
        return access.emitOpError("cannot align coordinate with its access schema")
               << "; coordinate=" << coordinate;
      access.getAccessCoordinatesMutable().slice(slot, 1).assign(*aligned);
    }
    return success();
  };

  auto access = cast<AccessOpInterface>(operation);
  auto alignRead = [&]() -> LogicalResult {
    if (failed(alignCoordinates(access, access.getAccessValueType())))
      return failure();
    if (!access.getAccessValidity() && !access.getAccessFill())
      return success();
    if (!access.getAccessValidity())
      return access.emitOpError("fill has no validity authority");
    OpBuilder builder(access);
      builder.setListener(&changes);
    Type valueType = access.getAccessValueType();
    auto fragment = dyn_cast<FragmentType>(valueType);
    if (access.getAccessKind() == AccessKind::Gather && !fragment) {
      if (access.getAccessFill())
        return success();
      return access.emitOpError("scalar gather validity and fill must remain paired");
    }
    Type validType = fragment ? Type(predicateType(fragment)) : builder.getI1Type();
    auto valid = project(builder, access.getLoc(), access.getAccessValidity(), validType);
    if (failed(valid))
      return access.emitOpError("cannot align read validity with its value schema");
    auto fill = access.getAccessFill()
                    ? project(builder, access.getLoc(), access.getAccessFill(), valueType)
                    : materializeZeroValue(builder, access.getLoc(), valueType);
    if (failed(fill))
      return access.emitOpError("cannot align read fill with its value schema");
    // Keep the access identity, coordinates, effects and all attributes intact;
    // only its already-established validity/fill relation changes here.
    return access.updateAccessOperands(access.getAccessCoordinates(),
                                       access.getAccessPayloads(), *valid, *fill);
  };
  if (access.getAccessKind() == AccessKind::Load ||
      access.getAccessKind() == AccessKind::Gather)
    return alignRead();
  auto store = dyn_cast<StoreOp>(operation);
  if (!store) return success();
    if (failed(alignCoordinates(access, store.getValue().getType())))
      return failure();
    if (!store.getValid())
      return success();
    auto currentType = dyn_cast<FragmentType>(store.getValue().getType());
    if (!currentType) {
      auto validity = dyn_cast<FragmentType>(store.getValid().getType());
      if (!validity || !llvm::all_of(validity.getShape(), [](Attribute extent) {
            return constantPhysicalExpression(cast<PhysicalExprAttr>(extent)) == 1;
          }))
        return success();
      // Ownership can retain a one-lane predicate for a scalar write.  Adopt
      // that schema without widening the write into additional lanes.
      currentType = FragmentType::get(
          kernel.getContext(), store.getValue().getType(), validity.getShape(),
          validity.getAxisMaps(), validity.getValidity(), validity.getOwner());
      OpBuilder builder(store);
      builder.setListener(&changes);
      FailureOr<Value> value = project(builder, store.getLoc(), store.getValue(),
                                       currentType);
      if (failed(value))
        return store.emitOpError("cannot align scalar store with its one-lane validity");
      store.getValueMutable().assign(*value);
    }
    // A value whose extent is already selected by an exact range or verified
    // reshape is the physical data authority for the write.  Retarget the
    // address relation to that extent before reconciling schemas.  This keeps
    // extents and coordinate provenance separate: the value does not acquire
    // the view's source identity, and the address does not overwrite a verified
    // row-major reshape decision.
    PhysicalProgramAnalysis analysis(kernel);
    SmallVector<Value> fragmentCoordinates;
    for (Value coordinate : store.getCoordinates())
      if (isa<FragmentType>(coordinate.getType()))
        fragmentCoordinates.push_back(coordinate);
    bool cartesian = fragmentCoordinates.size() == currentType.getShape().size() &&
        llvm::all_of(fragmentCoordinates, [](Value coordinate) {
          return cast<FragmentType>(coordinate.getType()).getShape().size() == 1;
        });
    for (unsigned axis = 0; axis < currentType.getShape().size(); ++axis) {
      PhysicalAxisRealizationFact realization =
          analysis.axisRealization(store.getValue(), axis);
      if (!realization.hasExtentAuthority())
        continue;
      auto mapping = cast<AxisMapAttr>(currentType.getAxisMaps()[axis]);
      auto extent = cast<PhysicalExprAttr>(currentType.getShape()[axis]);
      auto positional = cartesian
          ? queryFragmentAxis(fragmentCoordinates[axis].getType(), sourceAxisIdentity(mapping))
          : PhysicalAxisProjection{};
      for (Value coordinate : store.getCoordinates()) {
        if (positional.isExact() &&
            positional.dimensionId == mapping.getDimensionId() &&
            coordinate != fragmentCoordinates[axis])
          continue;
        auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
        if (!coordinateType)
          continue;
        PhysicalAxisProjection projection =
            queryFragmentAxis(coordinateType, sourceAxisIdentity(mapping));
        if (!projection.isExact() ||
            projection.dimensionId != mapping.getDimensionId() ||
            coordinateType.getShape()[projection.fragmentAxis] == extent)
          continue;
        if (failed(retargetSourceExtent(coordinate, projection.source, extent,
                             projection.dimensionId, changes.typeChanged(), &changes)))
          return failure();
      }
    }
    currentType = cast<FragmentType>(store.getValue().getType());
    OpBuilder builder(store);
      builder.setListener(&changes);
    FailureOr<FragmentType> valueType =
        queryAccessResultSchema(kernel, currentType, store.getCoordinates());
    if (failed(valueType))
      return store.emitOpError(
          "store value has no unique physical coordinate projection");
    for (auto [axis, mapping] : llvm::enumerate((*valueType).getAxisMaps())) {
      if (axis >= currentType.getShape().size() ||
          currentType.getShape()[axis] == (*valueType).getShape()[axis])
        continue;
      int64_t dimension = cast<AxisMapAttr>(mapping).getDimensionId();
      if (dimension <= 0)
        return store.emitOpError(
            "store coordinate refinement has no logical dimension authority");
      if (failed(retargetSourceExtent(
          store.getValue(), sourceAxisIdentity(cast<AxisMapAttr>(mapping)),
          cast<PhysicalExprAttr>((*valueType).getShape()[axis]), dimension, changes.typeChanged(), &changes)))
        return failure();
    }
    FailureOr<Value> value = project(builder, store.getLoc(), store.getValue(),
                                     *valueType);
    if (failed(value)) {
      InFlightDiagnostic diagnostic = store.emitOpError(
          "cannot align store value with its coordinate schema");
      diagnostic << "; value=" << store.getValue().getType()
                 << "; coordinate_schema=" << *valueType;
      for (Value coordinate : store.getCoordinates())
        diagnostic << "; coordinate=" << coordinate.getType();
      return failure();
    }
    FailureOr<Value> valid = project(
        builder, store.getLoc(), store.getValid(), predicateType(*valueType));
    if (failed(valid)) {
      InFlightDiagnostic diagnostic = store.emitOpError(
          "cannot align store validity with its value schema");
      diagnostic << "; value=" << *valueType
                 << "; validity=" << store.getValid().getType();
      if (Operation *producer = store.getValue().getDefiningOp())
        diagnostic << "; value_producer=" << producer->getName();
      return failure();
    }
    // Preserve the access identity and every effect/attribute while closing
    // only its payload and predicate schemas.
    store.getValueMutable().assign(*value);
    store.getValidMutable().assign(*valid);
  return success();
}

static LogicalResult refreshReshapeRelation(ReshapeOp reshape, RelationWorklist &changes) {
  // The operation owns its row-major groups. Refinement transports extents
  // through that existing relation; it does not infer a new reassociation.
  auto relations = queryFragmentOperandRelations(reshape);
  if (failed(relations))
    return reshape.emitOpError("reshape has no physical operand relation");
  auto result = transportFragmentResultType(
      *relations, reshape->getOperandTypes(), reshape.getResult().getType());
  if (failed(result))
    return reshape.emitOpError("reshape cannot transport its physical operand schema");
  changes.setType(reshape.getResult(), *result);
  return reshape.verify();
}



static LogicalResult retargetExtent(Value root, AxisSelector selects,
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
               !isIntroducedReshapeUnitAxis(value,
                                            mapping.getFragmentAxis());
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
          if (extent.getKind() ==
              PhysicalExprKind::Constant)
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
      if (auto structured = dyn_cast<StructuredOpInterface>(argument.getOwner()->getParentOp()))
        appendStructuredRelations(structured, argument, worklist, selects, StructuredBoundary::Parent);
    Operation *boundaryOwner = value.getDefiningOp();
    if (!boundaryOwner) boundaryOwner = cast<BlockArgument>(value).getOwner()->getParentOp();
    bool controlTarget = isa<RegionBranchOpInterface>(boundaryOwner);
    if (auto argument = dyn_cast<BlockArgument>(value))
      controlTarget |= !argument.getOwner()->isEntryBlock();
    if (controlTarget) {
      auto incoming = queryControlFlowIncoming(value);
      if (!incoming.complete)
        return boundaryOwner->emitOpError("cannot retarget a produced or unknown control schema slot");
      boundaries.insert(value);
      for (const ControlFlowEdge &edge : incoming.edges) {
        if (!edge.operand) return failure();
        auto outgoing = queryControlFlowOutgoing(*edge.operand);
        if (!outgoing.complete) return failure();
        for (const ControlFlowEdge &successor : outgoing.edges)
          worklist.push_back(successor.target);
      }
    }
    if (auto structured = dyn_cast_or_null<StructuredOpInterface>(value.getDefiningOp()))
      appendStructuredRelations(structured, value, worklist, selects, StructuredBoundary::Result);
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
                return oldExtent.getKind() ==
                           PhysicalExprKind::Constant &&
                       newExtent.getKind() ==
                           PhysicalExprKind::Constant &&
                       oldExtent.getValue() > 0 &&
                       static_cast<uint64_t>(newExtent.getValue()) ==
                           llvm::PowerOf2Ceil(static_cast<uint64_t>(oldExtent.getValue()));
              });
          if (padding) {
            setPhysicalValueType(buffer.getResult(), BufferType::get(
                storage.getContext(), storage.getElementType(), initial.getShape(),
                storage.getScope(), storage.getInstance(), storage.getOwner(),
                storage.getInitialization(), storage.getVisibility()), changed);
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
              if (group.sourceAxes.size() != 1 || group.resultAxes.size() != 1 ||
                  (!broadcast && group.kind != FragmentAxisRelationKind::Reassociation))
                continue;
              unsigned sourceAxis = group.sourceAxes.front();
              unsigned targetAxis = group.resultAxes.front();
              auto inputMap = cast<AxisMapAttr>(previousFragment.getAxisMaps()[sourceAxis]);
              auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
              auto oldExtent = cast<PhysicalExprAttr>(previousFragment.getShape()[sourceAxis]);
              if (!selects(inputMap) || selects(targetMap) ||
                  oldExtent != target.getShape()[targetAxis] ||
                  target.getShape()[targetAxis] == extent)
                continue;
              if (broadcast) {
                bool unit = oldExtent.getKind() == PhysicalExprKind::Constant &&
                            oldExtent.getValue() <= 1;
                bool sameLogicalExtent = inputMap.getDimensionId() > 0 &&
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
        appendStructuredRelations(cast<StructuredOpInterface>(user), value, worklist,
                                  selects, StructuredBoundary::Child, &use);
        continue;
      }
      if (isa<RegionBranchOpInterface, RegionBranchTerminatorOpInterface,
              BranchOpInterface>(user)) {
        auto outgoing = queryControlFlowOutgoing(use);
        if (!outgoing.complete)
          return user->emitOpError("cannot retarget an unknown control successor slot");
        for (const ControlFlowEdge &edge : outgoing.edges)
          worklist.push_back(edge.target);
        continue;
      }
      if (isa<YieldOp>(user))
        if (auto structured = dyn_cast<StructuredOpInterface>(user->getParentOp()))
          appendStructuredRelations(structured, value, worklist, selects,
                                    StructuredBoundary::Child, &use);
      if (isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
              ContractOp, ScaledContractOp, SparseContractOp, ReduceOp,
              ScanOp, RandomBitsOp,
              ScatterReduceOp, AtomicStoreOp, AtomicRMWOp,
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
    if (failed(projectSchemaBoundary(boundary, changed, listener))) return failure();
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
  return retargetExtent(root,
                 [=](AxisMapAttr mapping) {
                   return mapping.getDimensionId() == dimensionId;
                 },
                 extent, /*followLogicalDimension=*/true, changed, listener);
}

WalkResult alignStructuredCaptures(Operation *operation, RelationWorklist &changes) {
  auto sinkUniformCapture = [&](OpOperand &operand, ArrayRef<BlockArgument> arguments) {
    Operation *owner = operand.getOwner();
    Value capture = operand.get();
    auto fragment = dyn_cast<FragmentType>(capture.getType());
    if (!fragment)
      return;
    Value scalar = capture;
    while (isa<FragmentType>(scalar.getType())) {
      UniformExpression expression = describeUniformValue(scalar);
      if (expression.kind != UniformKind::Forward ||
          expression.operands.size() != 1)
        break;
      scalar = expression.operands.front();
    }
    if (scalar.getType() != fragment.getElementType())
      return;
    // Capture the uniform seed and construct its fragment inside the helper.
    // This keeps the region closed while allowing each use to adopt its own
    // physical extent through the ordinary splat projection rule.
    for (BlockArgument argument : arguments) {
      auto type = cast<FragmentType>(argument.getType());
      changes.setType(argument, scalar.getType());
      OpBuilder builder(argument.getOwner(), argument.getOwner()->begin());
      builder.setListener(&changes);
      auto splat = builder.create<SplatOp>(owner->getLoc(), type, argument);
      argument.replaceAllUsesExcept(splat.getResult(), splat.getOperation());
    }
    operand.set(scalar);
  };
  auto align = [&](Operation *owner, Value capture,
                   BlockArgument argument) -> LogicalResult {
    auto authority = dyn_cast<FragmentType>(capture.getType());
    auto target = dyn_cast<FragmentType>(argument.getType());
    if (!authority || !target)
      return capture.getType() == argument.getType()
                 ? success()
                 : owner->emitOpError(
                       "physical structured capture lost its parent schema");
    if (authority.getElementType() != target.getElementType() ||
        authority.getShape().size() != target.getShape().size() ||
        authority.getAxisMaps() != target.getAxisMaps() ||
        authority.getOwner() != target.getOwner())
      return owner->emitOpError(
          "physical structured capture changed its coordinate relation");
    for (auto [axis, attribute] : llvm::enumerate(authority.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      if (mapping.getDimensionId() <= 0)
        continue;
      if (failed(retargetDimensionExtent(argument, mapping.getDimensionId(),
                              cast<PhysicalExprAttr>(authority.getShape()[axis]), changes.typeChanged(), &changes)))
        return failure();
    }
    return capture.getType() == argument.getType()
               ? success()
               : owner->emitOpError(
                     "physical structured capture extent is inconsistent");
  };

  auto structured = cast<StructuredOpInterface>(operation);
  auto captures = isa<RegionFoldOp>(operation) ? cast<RegionFoldOp>(operation).getCaptures()
                                               : cast<RegionScanOp>(operation).getCaptures();
  for (unsigned index = 0; index < captures.size(); ++index) {
    SmallVector<BlockArgument, 2> arguments{
        cast<BlockArgument>(structured.getSummarizeCaptures()[index])};
    if (structured.getEmitRegion())
      arguments.push_back(cast<BlockArgument>(structured.getEmitCaptures()[index]));
    OpOperand &operand = operation->getOpOperand(captures.getBeginOperandIndex() + index);
    sinkUniformCapture(operand, arguments);
    for (BlockArgument argument : arguments)
      if (failed(align(operation, operand.get(), argument)))
        return WalkResult::interrupt();
  }
  return WalkResult::advance();
}

void eraseDeadPhysicalValues(func::FuncOp kernel) {
  bool changed = false;
  do {
    changed = false;
    kernel.walk<WalkOrder::PostOrder>([&](Operation *operation) {
      if (!operation->getBlock() ||
          operation == kernel.getOperation() || !operation->getNumResults() ||
          !llvm::all_of(operation->getResults(),
                        [](Value value) { return value.use_empty(); }))
        return;
      bool unusedReadOnlyLoop = isa<scf::ForOp>(operation) &&
          !operation->walk([](Operation *nested) {
            if (isa<scf::WhileOp>(nested))
              return WalkResult::interrupt();
            return isa<scf::ForOp, scf::IfOp, LoadOp, GatherOp>(nested) ||
                           isMemoryEffectFree(nested)
                       ? WalkResult::advance() : WalkResult::interrupt();
          }).wasInterrupted();
      if (isMemoryEffectFree(operation) || isa<LoadOp, GatherOp>(operation) ||
          unusedReadOnlyLoop) {
        operation->erase();
        changed = true;
      }
    });
  } while (changed);

}


LogicalResult RelationWorklist::run() {
  kernel.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    live.insert(operation);
    enqueue(operation);
  });
  while (true) {
    unsigned selected = 0;
    while (selected != RuleCount && worklists[selected].empty()) ++selected;
    if (selected == RuleCount) return success();
    auto [operation, rule] = worklists[selected].front();
    worklists[selected].pop_front();
    if (!queued[rule].erase(operation) || !live.contains(operation)) continue;

    // Rules can replace operands on an owner or its region terminators. Track
    // these edges in addition to explicit type notifications, so a new
    // projection or a reverse refinement requeues every affected relation.
    SmallVector<std::pair<Operation *, SmallVector<Value>>> boundaries;
    auto remember = [&](Operation *boundary) {
      boundaries.emplace_back(boundary, llvm::to_vector(boundary->getOperands()));
    };
    remember(operation);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        if (!block.empty()) remember(block.getTerminator());

    LogicalResult status = success();
    auto checked = [](WalkResult result) { return failure(result.wasInterrupted()); };
    switch (rule) {
    case Captures: status = checked(alignStructuredCaptures(operation, *this)); break;
    case ReductionResult: status = checked(alignReductionResultRelation(operation, *this)); break;
    case ReductionIdentity: status = checked(alignReductionIdentityRelation(operation, *this)); break;
    case Aggregate: status = closeSchemaBoundary(operation, typeChanged(), this); break;
    case AccessResult: status = checked(alignAccessResult(operation, *this)); break;
    case Pointwise: status = checked(alignPointwiseValue(operation, *this)); break;
    case ReductionYield: status = checked(alignReductionYield(operation, *this)); break;
    case AccessValue: status = alignAccessValue(operation, *this); break;
    case Reshape: status = refreshReshapeRelation(cast<ReshapeOp>(operation), *this); break;
    case ContractOperands: status = checked(alignContractOperands(operation, *this)); break;
    case ContractAccumulator: status = alignContractAccumulator(operation, *this); break;
    case RuleCount: llvm_unreachable("invalid relation rule");
    }
    if (failed(status))
      return operation->emitOpError("failed to close its physical value relation");
    if (conflict) return failure();
    for (auto &[boundary, before] : boundaries) {
      if (!live.contains(boundary) || llvm::equal(before, boundary->getOperands())) continue;
      enqueue(boundary);
      for (Value value : before) affected(value);
      for (Value value : boundary->getOperands()) affected(value);
    }
  }
}

LogicalResult closeValueRelations(func::FuncOp kernel, ValueRelationScope scope) {
  return RelationWorklist(kernel, scope).run();
}

} // namespace intent::gpu
