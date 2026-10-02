#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/ValueMaterialization.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
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
    Type previous = value.getType();
    if (previous == type) return;
    value.setType(type);
    typeChanged(value, previous);
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
    if (isa<MakeRecordOp, ExtractOp, scf::IfOp, scf::ForOp, RegionFoldOp>(operation))
      push(operation, Aggregate);
    if (auto access = dyn_cast<AccessOpInterface>(operation)) {
      AccessKind kind = access.getAccessKind();
      if (kind == AccessKind::Load || kind == AccessKind::Gather)
        push(operation, AccessResult);
      if (kind == AccessKind::Load || kind == AccessKind::Gather ||
          kind == AccessKind::Store)
        push(operation, AccessValue);
    }
    if (isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
            BroadcastOp, TransposeOp>(operation)) push(operation, Pointwise);
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
      retargetDimensionExtent(
          result, dimension,
          cast<PhysicalExprAttr>(aligned.getShape()[axis]), changes.typeChanged());
      retargetDimensionExtent(
          accumulator, dimension,
          cast<PhysicalExprAttr>(aligned.getShape()[axis]), changes.typeChanged());
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
    auto projection = queryBroadcastProjection(source, target);
    auto mapping = cast<AxisMapAttr>(target.getAxisMaps()[axis]);
    if (!projection.isExact() || projection.targetToSource[axis] ||
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
          retargetSourceExtent(authority, sourceAxisIdentity(mapping),
                               cast<PhysicalExprAttr>(rhsExtent),
                               mapping.getDimensionId(), changes.typeChanged());
        } else {
          auto mapping = cast<AxisMapAttr>(rhsType.getAxisMaps()[rhsAxis]);
          Value authority = batch ? batchExtentAuthority(rhs, mapping) : rhs;
          retargetSourceExtent(authority, sourceAxisIdentity(mapping),
                               cast<PhysicalExprAttr>(lhsExtent),
                               mapping.getDimensionId(), changes.typeChanged());
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
      BroadcastProjection projection =
          target ? queryAxisProjection(result, target) : BroadcastProjection();
      if (!target || !projection.isExact())
        return false;
      std::optional<unsigned> targetAxis;
      for (auto [candidate, sourceAxis] :
           llvm::enumerate(projection.targetToSource))
        if (sourceAxis && *sourceAxis == *axis)
          targetAxis = candidate;
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
                               AxisSelector selects, StructuredBoundary boundary) {
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
  if (boundary == StructuredBoundary::Child) {
    for (const auto &relation : relations)
      if (relation.from == value &&
          (relation.kind == K::Capture || (relation.kind == K::SameSchema || relation.kind == K::Accumulator) ||
           (relation.kind == K::SourceSlice && !crossesSelectedAxis(relation))))
        append(relation.to);
    return;
  }
  if (boundary == StructuredBoundary::Parent) {
    if (operation.getStructuredKind() != StructuredOpKind::RegionFold &&
        operation.getStructuredKind() != StructuredOpKind::RegionScan)
      return;
    for (const auto &relation : relations) {
      if (relation.to != value ||
          (relation.kind != K::Capture && relation.kind != K::SameSchema &&
           (relation.kind != K::SourceSlice || crossesSelectedAxis(relation))))
        continue;
      append(relation.from);
      if (relation.kind != K::SameSchema) continue;
      // Locate the formal slot, not the identity SSA value: multiple
      // components may intentionally use the same identity constant.
      auto appendPairedResult = [&](ValueRange arguments, ValueRange results) {
        for (auto [argument, result] : llvm::zip_equal(arguments, results))
          if (argument == value) append(result);
      };
      if (operation.getStructuredKind() == StructuredOpKind::RegionFold) {
        appendPairedResult(operation.getCombineLhs(), operation->getResults());
        appendPairedResult(operation.getCombineRhs(), operation->getResults());
      } else {
        appendPairedResult(operation.getApplyStates(), operation.getFinalStates());
        appendPairedResult(operation.getEmitStates(), operation.getFinalStates());
      }
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
      append(relation.from);
      if (operation.getStructuredKind() == StructuredOpKind::Reduce ||
          operation.getStructuredKind() == StructuredOpKind::RegionFold) {
        unsigned component = cast<OpResult>(value).getResultNumber();
        append(operation.getCombineLhs()[component]);
        append(operation.getCombineRhs()[component]);
        append(operation.getCombineYields()[component]);
        if (operation.getSummarizeRegion())
          append(operation.getSummarizeYields()[component]);
      } else {
        auto states = operation.getFinalStates();
        auto found = llvm::find(states, value);
        if (found != states.end()) {
          unsigned component = std::distance(states.begin(), found);
          append(operation.getApplyStates()[component]);
          append(operation.getEmitStates()[component]);
        }
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
      retargetSourceExtent(
          result, sourceAxisIdentity(cast<AxisMapAttr>(mapping)),
          cast<PhysicalExprAttr>((*refined).getShape()[axis]),
          cast<AxisMapAttr>(mapping).getDimensionId(), changes.typeChanged());
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
      auto source = cast<FragmentType>(transpose.getValue().getType());
      auto target = cast<FragmentType>(transpose.getResult().getType());
      SmallVector<Attribute> shape;
      for (int64_t input : transpose.getPermutation())
        shape.push_back(source.getShape()[input]);
      changes.setType(transpose.getResult(), FragmentType::get(
          kernel.getContext(), target.getElementType(),
          ArrayAttr::get(kernel.getContext(), shape), target.getAxisMaps(),
          target.getValidity(), target.getOwner()));
      return WalkResult::advance();
    }
    if (auto broadcast = dyn_cast<BroadcastOp>(operation)) {
      auto target = dyn_cast<FragmentType>(broadcast.getResult().getType());
      auto source = dyn_cast<FragmentType>(broadcast.getValue().getType());
      if (!target || !source)
        return WalkResult::advance();
      if (!queryBroadcastProjection(source, target).isExact()) {
        OpBuilder builder(broadcast);
      builder.setListener(&changes);
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, broadcast.getLoc(), broadcast.getValue(), target, changes.typeChanged());
        if (succeeded(projected)) {
          broadcast->setOperand(0, *projected);
          source = cast<FragmentType>(projected->getType());
        }
      }
      BroadcastProjection projection = queryAxisProjection(source, target);
      if (!projection.isExact()) {
        broadcast.emitOpError(
            projection.state == BroadcastProjectionState::Ambiguous
                ? "broadcast has an ambiguous physical source projection"
                : "broadcast has no physical source projection")
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
      for (auto [targetAxis, sourceAxis] :
           llvm::enumerate(projection.targetToSource)) {
        if (!sourceAxis ||
            source.getShape()[*sourceAxis] == target.getShape()[targetAxis])
          continue;
        auto sourceExtent =
            cast<PhysicalExprAttr>(source.getShape()[*sourceAxis]);
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
          auto sourceMap = cast<AxisMapAttr>(source.getAxisMaps()[*sourceAxis]);
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
            retargetSourceExtent(broadcast.getResult(),
                                 sourceAxisIdentity(targetMap), sourceExtent, std::nullopt, changes.typeChanged());
            target = cast<FragmentType>(broadcast.getResult().getType());
            continue;
          }
          broadcast.emitOpError(
              "broadcast cannot contract a non-singleton physical axis")
              << "; input=" << source << "; result=" << target;
          return WalkResult::interrupt();
        }
        bool sourceParameter =
            isParameterExtent(source.getShape()[*sourceAxis]);
        bool targetParameter = isParameterExtent(targetExtent);
        if (!sourceParameter && !targetParameter) {
          PhysicalProgramAnalysis analysis(kernel);
          auto input = analysis.axisRealization(broadcast.getValue(), *sourceAxis);
          auto output = analysis.axisRealization(broadcast.getResult(), targetAxis);
          if (input.hasExtentAuthority() && !input.constructionScalarSeed &&
              !output.hasExtentAuthority()) {
            auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
            retargetSourceExtent(broadcast.getResult(),
                                 sourceAxisIdentity(targetMap), sourceExtent, std::nullopt, changes.typeChanged());
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
            cast<AxisMapAttr>(source.getAxisMaps()[*sourceAxis]);
        auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
        if (sourceMap.getDimensionId() <= 0 || targetMap.getDimensionId() <= 0) {
          broadcast.emitOpError(
              "broadcast extent refinement has no logical occurrence authority");
          return WalkResult::interrupt();
        }
        if (targetParameter)
          retargetSourceExtent(
              broadcast.getValue(), sourceAxisIdentity(sourceMap),
              cast<PhysicalExprAttr>(target.getShape()[targetAxis]), std::nullopt, changes.typeChanged());
        else
          retargetSourceExtent(broadcast.getResult(),
                               sourceAxisIdentity(targetMap), sourceExtent, std::nullopt, changes.typeChanged());
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
    if (!isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(
            operation))
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
    if (isa<UnaryOp, CastOp, BitcastOp>(operation) &&
        operation->getNumOperands() == 1) {
      auto source = dyn_cast<FragmentType>(operation->getOperand(0).getType());
      if (source) {
        changes.setType(operation->getResult(0), FragmentType::get(
            kernel.getContext(), target.getElementType(), source.getShape(),
            source.getAxisMaps(), source.getValidity(), source.getOwner()));
        return WalkResult::advance();
      }
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
      retargetSourceExtent(
          operation->getResult(0),
          sourceAxisIdentity(cast<AxisMapAttr>(target.getAxisMaps()[axis])),
          cast<PhysicalExprAttr>((*refined).getShape()[axis]),
          cast<AxisMapAttr>(target.getAxisMaps()[axis]).getDimensionId(), changes.typeChanged());
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
        retargetSourceExtent(coordinate, projection.source, extent,
                             projection.dimensionId, changes.typeChanged());
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
      retargetSourceExtent(
          store.getValue(), sourceAxisIdentity(cast<AxisMapAttr>(mapping)),
          cast<PhysicalExprAttr>((*valueType).getShape()[axis]), dimension, changes.typeChanged());
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
    auto kernel = reshape->getParentOfType<func::FuncOp>();
    // Reassociation is the typed row-major relation selected from canonical
    // KIR.  Refinement passes may change physical extents, but they may not
    // rediscover and replace that relation from the new shapes: doing so turns
    // physical shape coincidence into a second algorithm authority.  Validate
    // the preserved carrier here; the mutating pass that changed an extent is
    // responsible for updating both sides of each existing group.
    auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
    auto target = dyn_cast<FragmentType>(reshape.getResult().getType());
    if (!source || !target) {
      reshape.emitOpError(
          "reshape relation requires physical fragment operands and result");
      return failure();
    }
    unsigned logicalSourceRank = 0;
    unsigned logicalResultRank = 0;
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = dyn_cast<ReshapeGroupAttr>(attribute);
      if (!group) {
        reshape.emitOpError("reshape relation contains an untyped group");
        return failure();
      }
      if (!group.getSourceAxes().empty())
        logicalSourceRank = std::max(
            logicalSourceRank,
            static_cast<unsigned>(
                group.getSourceAxes().asArrayRef().back() + 1));
      if (!group.getResultAxes().empty())
        logicalResultRank = std::max(
            logicalResultRank,
            static_cast<unsigned>(
                group.getResultAxes().asArrayRef().back() + 1));
    }
    if (logicalSourceRank > source.getShape().size() ||
        logicalResultRank > target.getShape().size()) {
      reshape.emitOpError(
          "reshape relation rank exceeds the current physical fragments");
      return failure();
    }
    unsigned sourcePrefix = source.getShape().size() - logicalSourceRank;
    unsigned resultPrefix = target.getShape().size() - logicalResultRank;
    SmallVector<Attribute> resultShape(target.getShape().begin(),
                                       target.getShape().end());
    if (sourcePrefix == resultPrefix &&
        llvm::equal(source.getAxisMaps().getValue().take_front(sourcePrefix),
                    target.getAxisMaps().getValue().take_front(resultPrefix)))
      llvm::copy(source.getShape().getValue().take_front(sourcePrefix),
                 resultShape.begin());
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = cast<ReshapeGroupAttr>(attribute);
      ArrayRef<int64_t> sourceAxes = group.getSourceAxes().asArrayRef();
      ArrayRef<int64_t> resultAxes = group.getResultAxes().asArrayRef();
      if (resultAxes.size() == 1) {
        resultShape[resultPrefix + resultAxes.front()] = productExtent(
            kernel.getContext(), source.getShape().getValue(), sourceAxes,
            sourcePrefix);
        continue;
      }
      if (sourceAxes.size() != 1 || resultAxes.empty())
        continue;
      SmallVector<int64_t> nonUnitAxes;
      for (int64_t axis : resultAxes) {
        auto extent = cast<PhysicalExprAttr>(
            resultShape[resultPrefix + axis]);
        if (extent.getKind() !=
                PhysicalExprKind::Constant ||
            extent.getValue() != 1)
          nonUnitAxes.push_back(axis);
      }
      if (nonUnitAxes.size() == 1)
        resultShape[resultPrefix + nonUnitAxes.front()] =
            cast<PhysicalExprAttr>(
                source.getShape()[sourcePrefix + sourceAxes.front()]);
    }
    changes.setType(reshape.getResult(), FragmentType::get(
        kernel.getContext(), target.getElementType(),
        ArrayAttr::get(kernel.getContext(), resultShape), target.getAxisMaps(),
        target.getValidity(), target.getOwner()));
    return reshape.verify();
}



WalkResult alignAggregateValue(Operation *operation, RelationWorklist &changes) {
  auto kernel = operation->getParentOfType<func::FuncOp>();
  if (auto record = dyn_cast<MakeRecordOp>(operation)) {
    RecordType current = record.getResult().getType();
    SmallVector<Attribute> fields;
    fields.reserve(record.getFields().size());
    for (Value field : record.getFields())
      fields.push_back(TypeAttr::get(field.getType()));
    changes.setType(record.getResult(), RecordType::get(
        kernel.getContext(), current.getFieldNames(),
        ArrayAttr::get(kernel.getContext(), fields), current.getOwner()));
    return WalkResult::advance();
  }
  std::function<FailureOr<Type>(Type, Type, Type)> joinTypes =
      [&](Type current, Type lhs, Type rhs) -> FailureOr<Type> {
    if (auto lhsFragment = dyn_cast<FragmentType>(lhs)) {
      auto rhsFragment = dyn_cast<FragmentType>(rhs);
      auto currentFragment = dyn_cast<FragmentType>(current);
      if (!rhsFragment || !currentFragment ||
          lhsFragment.getElementType() != rhsFragment.getElementType() ||
          lhsFragment.getElementType() != currentFragment.getElementType() ||
          lhsFragment.getShape().size() != rhsFragment.getShape().size() ||
          lhsFragment.getShape().size() != currentFragment.getShape().size() ||
          lhsFragment.getAxisMaps() != rhsFragment.getAxisMaps() ||
          lhsFragment.getValidity() != rhsFragment.getValidity() ||
          lhsFragment.getOwner() != rhsFragment.getOwner())
        return failure();
      SmallVector<Attribute> shape;
      for (auto [left, right] :
           llvm::zip(lhsFragment.getShape(), rhsFragment.getShape())) {
        if (left == right) {
          shape.push_back(left);
          continue;
        }
        auto leftExtent = cast<PhysicalExprAttr>(left);
        auto rightExtent = cast<PhysicalExprAttr>(right);
        bool leftUnit =
            leftExtent.getKind() ==
                PhysicalExprKind::Constant &&
            leftExtent.getValue() == 1;
        bool rightUnit =
            rightExtent.getKind() ==
                PhysicalExprKind::Constant &&
            rightExtent.getValue() == 1;
        if (leftUnit == rightUnit)
          return failure();
        shape.push_back(leftUnit ? right : left);
      }
      return Type(FragmentType::get(
          kernel.getContext(), lhsFragment.getElementType(),
          ArrayAttr::get(kernel.getContext(), shape), lhsFragment.getAxisMaps(),
          lhsFragment.getValidity(), lhsFragment.getOwner()));
    }
    auto lhsRecord = dyn_cast<RecordType>(lhs);
    auto rhsRecord = dyn_cast<RecordType>(rhs);
    auto currentRecord = dyn_cast<RecordType>(current);
    if (lhsRecord || rhsRecord || currentRecord) {
      if (!lhsRecord || !rhsRecord || !currentRecord ||
          lhsRecord.getFieldNames() != rhsRecord.getFieldNames() ||
          lhsRecord.getFieldNames() != currentRecord.getFieldNames() ||
          lhsRecord.getFieldTypes().size() != rhsRecord.getFieldTypes().size() ||
          lhsRecord.getFieldTypes().size() !=
              currentRecord.getFieldTypes().size() ||
          lhsRecord.getOwner() != rhsRecord.getOwner())
        return failure();
      SmallVector<Attribute> fields;
      for (auto [base, left, right] :
           llvm::zip(currentRecord.getFieldTypes(), lhsRecord.getFieldTypes(),
                     rhsRecord.getFieldTypes())) {
        FailureOr<Type> joined = joinTypes(
            cast<TypeAttr>(base).getValue(), cast<TypeAttr>(left).getValue(),
            cast<TypeAttr>(right).getValue());
        if (failed(joined))
          return failure();
        fields.push_back(TypeAttr::get(*joined));
      }
      return Type(RecordType::get(
          kernel.getContext(), lhsRecord.getFieldNames(),
          ArrayAttr::get(kernel.getContext(), fields), lhsRecord.getOwner()));
    }
    return current == lhs && lhs == rhs ? FailureOr<Type>(current)
                                        : FailureOr<Type>(failure());
  };
  if (auto branch = dyn_cast<scf::IfOp>(operation)) {
    if (branch.getNumResults() == 0)
      return WalkResult::advance();
    auto thenYield = dyn_cast<scf::YieldOp>(branch.thenBlock()->getTerminator());
    auto elseYield = dyn_cast<scf::YieldOp>(branch.elseBlock()->getTerminator());
    if (!thenYield || !elseYield ||
        thenYield.getResults().size() != branch.getNumResults() ||
        elseYield.getResults().size() != branch.getNumResults())
      return WalkResult::interrupt();
    for (unsigned index = 0; index < branch.getNumResults(); ++index) {
      Value leftValue = thenYield.getResults()[index];
      Value rightValue = elseYield.getResults()[index];
      Type leftType = leftValue.getType(), rightType = rightValue.getType();
      UniformValueAnalysis uniform(describeUniformValue);
      bool leftUniform = static_cast<bool>(uniform.evaluate(leftValue));
      bool rightUniform = static_cast<bool>(uniform.evaluate(rightValue));
      if (leftUniform != rightUniform) {
        Type &uniformType = leftUniform ? leftType : rightType;
        if (auto fragment = dyn_cast<FragmentType>(uniformType)) {
          // A uniform branch adopts the other branch's selected physical
          // extents, while retaining its axis, dtype and ownership obligations.
          auto unit = PhysicalExprAttr::get(
              kernel.getContext(), PhysicalExprKind::Constant,
              1, StringAttr::get(kernel.getContext(), ""),
              ArrayAttr::get(kernel.getContext(), {}));
          SmallVector<Attribute> units(fragment.getShape().size(), unit);
          uniformType = FragmentType::get(
              kernel.getContext(), fragment.getElementType(),
              ArrayAttr::get(kernel.getContext(), units), fragment.getAxisMaps(),
              fragment.getValidity(), fragment.getOwner());
        }
      }
      FailureOr<Type> target = joinTypes(
          branch.getResult(index).getType(), leftType, rightType);
      if (failed(target)) {
        branch.emitOpError(
            "control-flow branches have no unique physical result relation")
            << "; result_index=" << index
            << "; then=" << thenYield.getResults()[index].getType()
            << "; else=" << elseYield.getResults()[index].getType();
        return WalkResult::interrupt();
      }
      OpBuilder thenBuilder(thenYield);
      thenBuilder.setListener(&changes);
      FailureOr<Value> projectedThen = projectPhysicalValueToSchema(
          thenBuilder, branch.getLoc(), thenYield.getResults()[index], *target, changes.typeChanged());
      OpBuilder elseBuilder(elseYield);
      elseBuilder.setListener(&changes);
      FailureOr<Value> projectedElse = projectPhysicalValueToSchema(
          elseBuilder, branch.getLoc(), elseYield.getResults()[index], *target, changes.typeChanged());
      if (failed(projectedThen) || failed(projectedElse)) {
        branch.emitOpError(
            "control-flow branch cannot adopt its joined physical relation")
            << "; result_index=" << index << "; target=" << *target;
        return WalkResult::interrupt();
      }
      thenYield->setOperand(index, *projectedThen);
      elseYield->setOperand(index, *projectedElse);
      changes.setType(branch.getResult(index), *target);
    }
    return WalkResult::advance();
  }
  if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
    if (!llvm::hasSingleElement(fold.getSummarize()) ||
        !llvm::hasSingleElement(fold.getCombine()))
      return WalkResult::interrupt();
    auto summarizeYield =
        dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
    if (!summarizeYield ||
        summarizeYield.getValues().size() != fold.getIdentities().size())
      return WalkResult::interrupt();
    auto structured = cast<StructuredOpInterface>(fold.getOperation());
    if (failed(verifyStructuredArity(structured))) return WalkResult::interrupt();
    OpBuilder builder(fold);
      builder.setListener(&changes);
    for (unsigned index = 0; index < fold.getIdentities().size(); ++index) {
      Type target = summarizeYield.getValues()[index].getType();
      SmallVector<std::pair<int64_t, PhysicalExprAttr>> dimensions;
      std::function<LogicalResult(Type)> collectDimensions =
          [&](Type type) -> LogicalResult {
        if (auto fragment = dyn_cast<FragmentType>(type)) {
          for (auto [axis, mapping] :
               llvm::enumerate(fragment.getAxisMaps())) {
            int64_t dimension = cast<AxisMapAttr>(mapping).getDimensionId();
            if (dimension <= 0)
              continue;
            auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
            auto found = llvm::find_if(dimensions, [&](const auto &entry) {
              return entry.first == dimension;
            });
            if (found != dimensions.end()) {
              if (found->second != extent)
                return failure();
              continue;
            }
            dimensions.emplace_back(dimension, extent);
          }
          return success();
        }
        if (auto record = dyn_cast<RecordType>(type))
          for (Attribute field : record.getFieldTypes())
            if (failed(collectDimensions(cast<TypeAttr>(field).getValue())))
              return failure();
        return success();
      };
      if (failed(collectDimensions(target))) {
        fold.emitOpError(
            "region-fold summary has conflicting physical dimension extents")
            << "; summary_index=" << index << "; summary=" << target;
        return WalkResult::interrupt();
      }
      for (auto [dimension, extent] : dimensions)
        retargetDimensionExtent(fold.getResult(index), dimension, extent, changes.typeChanged());
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, fold.getLoc(), fold.getIdentities()[index], target, changes.typeChanged());
      if (failed(projected)) {
        fold.emitOpError(
            "region-fold identity cannot adopt its summary relation");
        return WalkResult::interrupt();
      }
      fold.getIdentitiesMutable().slice(index, 1).assign(*projected);
      changes.setType(fold.getResult(index), target);
      changes.setType(structured.getCombineLhs()[index], target);
      changes.setType(structured.getCombineRhs()[index], target);
    }
    return WalkResult::advance();
  }
  if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield || yield.getResults().size() != loop.getInitArgs().size())
      return WalkResult::interrupt();
    for (unsigned index = 0; index < loop.getInitArgs().size(); ++index) {
      Value init = loop.getInitArgs()[index];
      Value yielded = yield.getResults()[index];
      // The loop-body update is the executable relation selected by the
      // transformation that produced it.  The init value is an identity/seed
      // at the structural boundary and must be projected to that relation; the
      // boundary must not rank shapes or independently choose a competing one.
      Type target = yielded.getType();
      OpBuilder initBuilder(loop);
      initBuilder.setListener(&changes);
      FailureOr<Value> projectedInit = projectPhysicalValueToSchema(
          initBuilder, loop.getLoc(), init, target, changes.typeChanged());
      OpBuilder yieldBuilder(yield);
      yieldBuilder.setListener(&changes);
      FailureOr<Value> projectedYield = projectPhysicalValueToSchema(
          yieldBuilder, loop.getLoc(), yielded, target, changes.typeChanged());
      if (failed(projectedInit) || failed(projectedYield)) {
        loop.emitOpError(
            "loop-carried value cannot adopt its unique physical relation")
            << "; init=" << init.getType() << "; yield=" << yielded.getType()
            << "; target=" << target;
        return WalkResult::interrupt();
      }
      loop.getInitArgsMutable()[index].assign(*projectedInit);
      yield->setOperand(index, *projectedYield);
      changes.setType(loop.getRegionIterArgs()[index], target);
      changes.setType(loop.getResult(index), target);
    }
    return WalkResult::advance();
  }
  // Structured arguments/results are the record-schema authority.  Refresh
  // projections only after those schemas have been aligned; doing this before
  // the region owner leaves combine-body fields one refinement behind.
  if (auto extract = dyn_cast<ExtractOp>(operation)) {
    RecordType record = extract.getRecord().getType();
    if (extract.getField() < record.getFieldTypes().size())
      changes.setType(extract.getResult(),
          cast<TypeAttr>(record.getFieldTypes()[extract.getField()]).getValue());
  }
  return WalkResult::advance();
}

static void retargetExtent(Value root, AxisSelector selects,
                           PhysicalExprAttr extent,
                           bool followLogicalDimension,
                           ValueTypeChangeCallback changed) {
  SmallVector<Attribute> previousExtents;
  collectSelectedExtents(root.getType(), selects, previousExtents);
  if (previousExtents.empty())
    return;
  SmallVector<Attribute> connectedExtents(previousExtents.begin(),
                                          previousExtents.end());
  if (!llvm::is_contained(connectedExtents, Attribute(extent)))
    connectedExtents.push_back(extent);
  SmallVector<Value> worklist{root};
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
        Type previous = value.getType();
        value.setType(replacement);
        if (changed) changed(value, previous);
        // A make_range owns both the fragment schema and the SSA extent used
        // to materialize that schema.  Retarget them from the same physical
        // decision; leaving the operand behind creates two executable
        // authorities for one traversal.
        if (auto range = value.getDefiningOp<MakeRangeOp>()) {
          OpBuilder builder(range);
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
    if (auto record = value.getDefiningOp<MakeRecordOp>())
      worklist.append(record.getFields().begin(), record.getFields().end());
    if (auto extract = value.getDefiningOp<ExtractOp>())
      worklist.push_back(extract.getRecord());
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (auto structured = dyn_cast<StructuredOpInterface>(argument.getOwner()->getParentOp()))
        appendStructuredRelations(structured, argument, worklist, selects, StructuredBoundary::Parent);
      auto loop = dyn_cast_or_null<scf::ForOp>(
          argument.getOwner()->getParentOp());
      if (loop)
        for (auto [index, iterArgument] :
             llvm::enumerate(loop.getRegionIterArgs()))
          if (argument == iterArgument) {
            worklist.push_back(loop.getInitArgs()[index]);
            worklist.push_back(loop.getResult(index));
          }
      if (auto whileLoop = dyn_cast<scf::WhileOp>(argument.getOwner()->getParentOp())) {
        unsigned index = argument.getArgNumber();
        if (argument.getOwner() == &whileLoop.getBefore().front()) {
          worklist.push_back(whileLoop.getInits()[index]);
          worklist.push_back(whileLoop.getAfter().front().getTerminator()->getOperand(index));
        } else {
          auto condition = cast<scf::ConditionOp>(whileLoop.getBefore().front().getTerminator());
          worklist.push_back(condition.getArgs()[index]);
          worklist.push_back(whileLoop.getResult(index));
        }
      }
    }
    if (auto loop = value.getDefiningOp<scf::ForOp>())
      for (auto [index, result] : llvm::enumerate(loop.getResults()))
        if (value == result) {
          worklist.push_back(loop.getInitArgs()[index]);
          worklist.push_back(loop.getRegionIterArgs()[index]);
          worklist.push_back(loop.getBody()->getTerminator()->getOperand(index));
        }
    if (auto loop = value.getDefiningOp<scf::WhileOp>()) {
      unsigned index = cast<OpResult>(value).getResultNumber();
      auto condition = cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
      worklist.push_back(condition.getArgs()[index]);
      worklist.push_back(loop.getAfterArguments()[index]);
    }
    if (auto branch = value.getDefiningOp<scf::IfOp>())
      for (auto [index, result] : llvm::enumerate(branch.getResults())) {
        if (value != result)
          continue;
        auto thenYield = cast<scf::YieldOp>(branch.thenBlock()->getTerminator());
        auto elseYield = cast<scf::YieldOp>(branch.elseBlock()->getTerminator());
        worklist.push_back(thenYield.getResults()[index]);
        worklist.push_back(elseYield.getResults()[index]);
      }
    if (auto structured = dyn_cast_or_null<StructuredOpInterface>(value.getDefiningOp()))
      appendStructuredRelations(structured, value, worklist, selects, StructuredBoundary::Result);
    for (Operation *user : value.getUsers()) {
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
            Type previous = buffer.getResult().getType();
            buffer.getResult().setType(BufferType::get(
                storage.getContext(), storage.getElementType(), initial.getShape(),
                storage.getScope(), storage.getInstance(), storage.getOwner(),
                storage.getInitialization(), storage.getVisibility()));
            if (changed && previous != buffer.getResult().getType())
              changed(buffer.getResult(), previous);
          }
        }
      }
      if (auto broadcast = dyn_cast<BroadcastOp>(user)) {
        auto target = dyn_cast<FragmentType>(broadcast.getResult().getType());
        if (previousFragment && target) {
          BroadcastProjection projection =
              queryBroadcastProjection(previousFragment, target);
          if (projection.isExact())
            for (auto [targetAxis, sourceAxis] :
                 llvm::enumerate(projection.targetToSource)) {
              if (!sourceAxis)
                continue;
              auto inputMap =
                  cast<AxisMapAttr>(previousFragment.getAxisMaps()[*sourceAxis]);
              auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
              auto oldExtent =
                  cast<PhysicalExprAttr>(previousFragment.getShape()[*sourceAxis]);
              bool unit = oldExtent.getKind() ==
                              PhysicalExprKind::Constant &&
                          oldExtent.getValue() <= 1;
              bool sameLogicalExtent = inputMap.getDimensionId() > 0 &&
                  inputMap.getDimensionId() == targetMap.getDimensionId();
              if (selects(inputMap) && !selects(targetMap) &&
                  (!unit || sameLogicalExtent) &&
                  !isIntroducedReshapeUnitAxis(value, *sourceAxis) &&
                  oldExtent == target.getShape()[targetAxis] &&
                  target.getShape()[targetAxis] != extent) {
                std::pair<Value, AxisMapAttr> alias{broadcast.getResult(),
                                                    targetMap};
                if (!llvm::is_contained(valueAliases, alias))
                  valueAliases.push_back(alias);
              }
            }
        }
      }
      if (auto reshape = dyn_cast<ReshapeOp>(user)) {
        auto target = dyn_cast<FragmentType>(reshape.getResult().getType());
        if (previousFragment && target) {
          unsigned sourceRank = 0, resultRank = 0;
          for (Attribute attribute : reshape.getReassociation()) {
            auto group = cast<ReshapeGroupAttr>(attribute);
            sourceRank += group.getSourceAxes().size();
            resultRank += group.getResultAxes().size();
          }
          unsigned sourcePrefix = previousFragment.getShape().size() - sourceRank;
          unsigned resultPrefix = target.getShape().size() - resultRank;
          for (Attribute attribute : reshape.getReassociation()) {
            auto group = cast<ReshapeGroupAttr>(attribute);
            if (group.getSourceAxes().size() != 1 ||
                group.getResultAxes().size() != 1)
              continue;
            unsigned sourceAxis = sourcePrefix + group.getSourceAxes()[0];
            unsigned targetAxis = resultPrefix + group.getResultAxes()[0];
            auto inputMap =
                cast<AxisMapAttr>(previousFragment.getAxisMaps()[sourceAxis]);
            auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
            if (!selects(inputMap) || selects(targetMap) ||
                previousFragment.getShape()[sourceAxis] != target.getShape()[targetAxis] ||
                target.getShape()[targetAxis] == extent)
              continue;
            std::pair<Value, AxisMapAttr> alias{reshape.getResult(), targetMap};
            if (!llvm::is_contained(valueAliases, alias))
              valueAliases.push_back(alias);
          }
        }
      }
      if (isa<RegionFoldOp, RegionScanOp>(user)) {
        appendStructuredRelations(cast<StructuredOpInterface>(user), value, worklist,
                                  selects, StructuredBoundary::Child);
        for (Value result : user->getResults())
          worklist.push_back(result);
        continue;
      }
      if (auto loop = dyn_cast<scf::ForOp>(user))
        for (auto [index, init] : llvm::enumerate(loop.getInitArgs()))
          if (value == init) {
            worklist.push_back(loop.getRegionIterArgs()[index]);
            worklist.push_back(loop.getResult(index));
          }
      if (auto loop = dyn_cast<scf::WhileOp>(user))
        for (auto [index, init] : llvm::enumerate(loop.getInits()))
          if (value == init)
            worklist.push_back(loop.getBeforeArguments()[index]);
      if (auto condition = dyn_cast<scf::ConditionOp>(user)) {
        auto loop = cast<scf::WhileOp>(condition->getParentOp());
        for (auto [index, argument] : llvm::enumerate(condition.getArgs()))
          if (value == argument) {
            worklist.push_back(loop.getAfterArguments()[index]);
            worklist.push_back(loop.getResult(index));
          }
      }
      if (auto yield = dyn_cast<scf::YieldOp>(user)) {
        if (auto loop = dyn_cast<scf::WhileOp>(yield->getParentOp()))
          for (auto [index, yielded] : llvm::enumerate(yield.getOperands()))
            if (value == yielded) {
              worklist.push_back(loop.getInits()[index]);
              worklist.push_back(loop.getBeforeArguments()[index]);
            }
        if (auto loop = dyn_cast_or_null<scf::ForOp>(yield->getParentOp()))
          for (auto [index, yielded] : llvm::enumerate(yield.getOperands()))
            if (value == yielded) {
              worklist.push_back(loop.getInitArgs()[index]);
              worklist.push_back(loop.getRegionIterArgs()[index]);
              worklist.push_back(loop.getResult(index));
            }
        if (auto branch = dyn_cast_or_null<scf::IfOp>(yield->getParentOp()))
          for (auto [index, yielded] : llvm::enumerate(yield.getOperands()))
            if (value == yielded)
              worklist.push_back(branch.getResult(index));
      }
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
  for (auto [value, axis] : valueAliases) {
    auto type = cast<FragmentType>(value.getType());
    if (type.getShape()[axis.getFragmentAxis()] == extent)
      continue;
    retargetExtent(
        value,
        [=](AxisMapAttr mapping) {
          return sourceAxisIdentity(mapping) == sourceAxisIdentity(axis) &&
                 mapping.getDimensionId() == axis.getDimensionId();
        },
        extent, /*followLogicalDimension=*/false, changed);
  }
}

void retargetSourceExtent(Value root, PhysicalSourceAxis source,
                          PhysicalExprAttr extent,
                          std::optional<int64_t> dimension,
                          ValueTypeChangeCallback changed) {
  retargetExtent(
      root,
      [=](AxisMapAttr mapping) {
        return mapping.getSourceId() == source.sourceId &&
               mapping.getSourceAxis() == source.sourceAxis &&
               mapping.getDerived() == source.derived &&
               (!dimension || mapping.getDimensionId() == *dimension);
      },
      extent, /*followLogicalDimension=*/false, changed);
}

void retargetDimensionExtent(Value root, int64_t dimensionId,
                             PhysicalExprAttr extent,
                             ValueTypeChangeCallback changed) {
  if (dimensionId <= 0)
    return;
  retargetExtent(root,
                 [=](AxisMapAttr mapping) {
                   return mapping.getDimensionId() == dimensionId;
                 },
                 extent, /*followLogicalDimension=*/true, changed);
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
      retargetDimensionExtent(argument, mapping.getDimensionId(),
                              cast<PhysicalExprAttr>(authority.getShape()[axis]), changes.typeChanged());
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
    case Aggregate: status = checked(alignAggregateValue(operation, *this)); break;
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
