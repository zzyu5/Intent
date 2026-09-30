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

using namespace mlir;

namespace intent::gpu {

namespace {

LogicalResult alignContractAccumulatorTypes(func::FuncOp kernel) {
  auto align = [](Operation *owner, OpOperand &accumulatorOperand, Value result,
                  bool &changed) -> LogicalResult {
    Value accumulator = accumulatorOperand.get();
    if (accumulator.getType() == result.getType())
      return success();
    changed = true;
    auto source = dyn_cast<FragmentType>(accumulator.getType());
    auto target = dyn_cast<FragmentType>(result.getType());
    if (!source || !target || source.getElementType() != target.getElementType() ||
        source.getOwner() != target.getOwner())
      return owner->emitOpError(
          "pointwise ownership cannot preserve the contract accumulator relation");
    if (isLiteralZeroProjection(accumulator)) {
      OpBuilder builder(owner);
      auto projected = projectPhysicalValueToSchema(
          builder, owner->getLoc(), accumulator, target);
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
                 static_cast<uint32_t>(PhysicalExprKind::Constant) &&
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
          cast<PhysicalExprAttr>(aligned.getShape()[axis]));
      retargetDimensionExtent(
          accumulator, dimension,
          cast<PhysicalExprAttr>(aligned.getShape()[axis]));
    }
    accumulator.setType(aligned);
    result.setType(aligned);
    return success();
  };
  bool changed;
  do {
    changed = false;
    WalkResult result = kernel.walk([&](Operation *operation) {
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
        return WalkResult::advance();
      }
      return failed(align(operation, *accumulator, output, changed))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
  } while (changed);
  return success();
}

LogicalResult alignOrdinaryContractOperandTypes(func::FuncOp kernel) {
  auto isUnit = [](Attribute attribute) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    return extent.getKind() ==
               static_cast<uint32_t>(PhysicalExprKind::Constant) &&
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
  WalkResult result = kernel.walk([&](ContractOp contract) {
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
                               mapping.getDimensionId());
        } else {
          auto mapping = cast<AxisMapAttr>(rhsType.getAxisMaps()[rhsAxis]);
          Value authority = batch ? batchExtentAuthority(rhs, mapping) : rhs;
          retargetSourceExtent(authority, sourceAxisIdentity(mapping),
                               cast<PhysicalExprAttr>(lhsExtent),
                               mapping.getDimensionId());
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
  });
  return result.wasInterrupted() ? failure() : success();
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
                  static_cast<uint32_t>(PhysicalExprKind::Constant) &&
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

void appendStructuredParentRelations(BlockArgument argument,
                                     SmallVectorImpl<Value> &worklist,
                                     AxisSelector selects) {
  Block *block = argument.getOwner();
  Region *region = block->getParent();
  Operation *parent = region ? region->getParentOp() : nullptr;
  unsigned index = argument.getArgNumber();
  if (auto fold = dyn_cast_or_null<RegionFoldOp>(parent)) {
    unsigned sources = fold.getSourceCount();
    unsigned identities = fold.getIdentityCount();
    if (region == &fold.getSummarize()) {
      if (index < sources) {
        if (!selectsSegmentAxis(argument.getType(), fold.getAxis(), selects))
          worklist.push_back(fold.getInputs()[index]);
        return;
      }
      unsigned capture = index - sources;
      worklist.push_back(fold.getInputs()[sources + identities + capture]);
      return;
    }
    if (region == &fold.getCombine()) {
      unsigned component = index % identities;
      worklist.push_back(fold.getInputs()[sources + component]);
      worklist.push_back(fold.getResults()[component]);
    }
    return;
  }
  auto scan = dyn_cast_or_null<RegionScanOp>(parent);
  if (!scan)
    return;
  unsigned sources = scan.getSourceCount();
  unsigned identities = scan.getIdentityCount();
  unsigned states = scan.getStateCount();
  unsigned outputs = scan.getOutputCount();
  unsigned stateOffset = sources + identities;
  unsigned captureOffset = stateOffset + states;
  if (region == &scan.getSummarize()) {
    if (index < sources) {
      if (!selectsSegmentAxis(argument.getType(), scan.getAxis(), selects))
        worklist.push_back(scan.getInputs()[index]);
      return;
    }
    worklist.push_back(scan.getInputs()[captureOffset + index - sources]);
    return;
  }
  if (region == &scan.getCombine()) {
    worklist.push_back(scan.getInputs()[sources + index % identities]);
    return;
  }
  if (region == &scan.getApply()) {
    if (index < identities) {
      worklist.push_back(scan.getInputs()[sources + index]);
      return;
    }
    unsigned state = index - identities;
    worklist.push_back(scan.getInputs()[stateOffset + state]);
    worklist.push_back(scan.getResults()[outputs + state]);
    return;
  }
  if (region != &scan.getEmit())
    return;
  if (index < sources) {
    if (!selectsSegmentAxis(argument.getType(), scan.getAxis(), selects))
      worklist.push_back(scan.getInputs()[index]);
    return;
  }
  if (index < sources + states) {
    unsigned state = index - sources;
    worklist.push_back(scan.getInputs()[stateOffset + state]);
    worklist.push_back(scan.getResults()[outputs + state]);
    return;
  }
  worklist.push_back(
      scan.getInputs()[captureOffset + index - sources - states]);
}

void appendStructuredChildRelations(Operation *user, Value value,
                                    SmallVectorImpl<Value> &worklist,
                                    AxisSelector selects) {
  if (auto fold = dyn_cast<RegionFoldOp>(user)) {
    unsigned sources = fold.getSourceCount();
    unsigned identities = fold.getIdentityCount();
    for (auto [index, operand] : llvm::enumerate(fold.getInputs())) {
      if (operand != value)
        continue;
      if (index < sources) {
        BlockArgument slice =
            fold.getSummarize().front().getArgument(index);
        if (!selectsSegmentAxis(slice.getType(), fold.getAxis(), selects))
          worklist.push_back(slice);
        continue;
      }
      if (index < sources + identities) {
        unsigned component = index - sources;
        worklist.push_back(fold.getResults()[component]);
        Block &combine = fold.getCombine().front();
        worklist.push_back(combine.getArgument(component));
        worklist.push_back(combine.getArgument(identities + component));
        continue;
      }
      Block &summarize = fold.getSummarize().front();
      unsigned capture = index - sources - identities;
      worklist.push_back(summarize.getArgument(sources + capture));
    }
    return;
  }
  auto scan = dyn_cast<RegionScanOp>(user);
  if (!scan)
    return;
  unsigned sources = scan.getSourceCount();
  unsigned identities = scan.getIdentityCount();
  unsigned states = scan.getStateCount();
  unsigned outputs = scan.getOutputCount();
  unsigned stateOffset = sources + identities;
  unsigned captureOffset = stateOffset + states;
  for (auto [index, operand] : llvm::enumerate(scan.getInputs())) {
    if (operand != value)
      continue;
    if (index < sources) {
      BlockArgument summarize =
          scan.getSummarize().front().getArgument(index);
      BlockArgument emit = scan.getEmit().front().getArgument(index);
      if (!selectsSegmentAxis(summarize.getType(), scan.getAxis(), selects)) {
        worklist.push_back(summarize);
        worklist.push_back(emit);
      }
      continue;
    }
    if (index < stateOffset) {
      unsigned component = index - sources;
      Block &combine = scan.getCombine().front();
      Block &apply = scan.getApply().front();
      worklist.push_back(combine.getArgument(component));
      worklist.push_back(combine.getArgument(identities + component));
      worklist.push_back(apply.getArgument(component));
      continue;
    }
    if (index < captureOffset) {
      unsigned state = index - stateOffset;
      worklist.push_back(scan.getResults()[outputs + state]);
      worklist.push_back(
          scan.getApply().front().getArgument(identities + state));
      worklist.push_back(scan.getEmit().front().getArgument(sources + state));
      continue;
    }
    unsigned capture = index - captureOffset;
    worklist.push_back(
        scan.getSummarize().front().getArgument(sources + capture));
    worklist.push_back(
        scan.getEmit().front().getArgument(sources + states + capture));
  }
}

void appendStructuredResultRelations(Value value,
                                     SmallVectorImpl<Value> &worklist,
                                     AxisSelector selects) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return;
  unsigned index = result.getResultNumber();
  if (auto reduce = dyn_cast<ReduceOp>(result.getOwner())) {
    unsigned sources = reduce.getSourceCount();
    unsigned identities = reduce.getIdentityCount();
    Value source = reduce.getInputs()[index];
    auto fragment = dyn_cast<FragmentType>(source.getType());
    // Only retained axes connect a reduction result back to its source.
    if (fragment && llvm::none_of(reduce.getAxes(), [&](int64_t axis) {
          return selects(cast<AxisMapAttr>(fragment.getAxisMaps()[axis]));
        }))
      worklist.push_back(source);
    worklist.push_back(reduce.getInputs()[sources + index]);
    worklist.push_back(reduce.getCombine().front().getArgument(index));
    worklist.push_back(
        reduce.getCombine().front().getArgument(identities + index));
    worklist.push_back(cast<YieldOp>(reduce.getCombine().front().getTerminator())
                           .getValues()[index]);
    return;
  }
  if (auto fold = dyn_cast<RegionFoldOp>(result.getOwner())) {
    unsigned sources = fold.getSourceCount();
    unsigned identities = fold.getIdentityCount();
    worklist.push_back(fold.getInputs()[sources + index]);
    worklist.push_back(fold.getCombine().front().getArgument(index));
    worklist.push_back(
        fold.getCombine().front().getArgument(identities + index));
    worklist.push_back(
        cast<YieldOp>(fold.getSummarize().front().getTerminator())
            .getValues()[index]);
    worklist.push_back(cast<YieldOp>(fold.getCombine().front().getTerminator())
                           .getValues()[index]);
    return;
  }
  auto scan = dyn_cast<RegionScanOp>(result.getOwner());
  if (!scan)
    return;
  unsigned outputs = scan.getOutputCount();
  if (index < outputs) {
    worklist.push_back(
        cast<YieldOp>(scan.getEmit().front().getTerminator()).getValues()[index]);
    return;
  }
  unsigned state = index - outputs;
  unsigned sources = scan.getSourceCount();
  unsigned identities = scan.getIdentityCount();
  unsigned stateOffset = sources + identities;
  worklist.push_back(scan.getInputs()[stateOffset + state]);
  worklist.push_back(
      scan.getApply().front().getArgument(identities + state));
  worklist.push_back(scan.getEmit().front().getArgument(sources + state));
  worklist.push_back(
      cast<YieldOp>(scan.getApply().front().getTerminator()).getValues()[state]);
}


} // namespace

static WalkResult alignReductionResultRelation(Operation *operation) {
    auto kernel = operation->getParentOfType<func::FuncOp>();
    SmallVector<Value> sources;
    SmallVector<Value> results;
    llvm::SmallDenseSet<int64_t> reducedAxes;
    bool scan = false;
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      sources.append(reduce.getInputs().begin(),
                     reduce.getInputs().begin() + reduce.getSourceCount());
      results.append(reduce.getResults().begin(), reduce.getResults().end());
      reducedAxes.insert(reduce.getAxes().begin(), reduce.getAxes().end());
    } else if (auto currentScan = dyn_cast<ScanOp>(operation)) {
      sources.append(currentScan.getInputs().begin(),
                     currentScan.getInputs().begin() +
                         currentScan.getSourceCount());
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
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, operation->getLoc(), source, target);
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
        currentResult.setType(source.getType());
        continue;
      }
      auto sourceType = dyn_cast<FragmentType>(source.getType());
      if (!sourceType) {
        operation->emitOpError(
            "physical reduction source has no fragment result relation")
            << "; source=" << source.getType();
        return WalkResult::interrupt();
      }
      SmallVector<Attribute> shape;
      SmallVector<Attribute> mappings;
      for (auto [axis, extent] : llvm::enumerate(sourceType.getShape())) {
        if (reducedAxes.contains(static_cast<int64_t>(axis)))
          continue;
        shape.push_back(extent);
        auto mapping = cast<AxisMapAttr>(sourceType.getAxisMaps()[axis]);
        mappings.push_back(AxisMapAttr::get(
            kernel.getContext(), mapping.getSourceId(),
            mapping.getSourceAxis(), mapping.getDimensionId(),
            static_cast<uint32_t>(mappings.size()), mapping.getDerived()));
      }
      Type element = currentResult.getType();
      if (auto current = dyn_cast<FragmentType>(element))
        element = current.getElementType();
      if (shape.empty()) {
        currentResult.setType(element);
        continue;
      }
      currentResult.setType(FragmentType::get(
          kernel.getContext(), element,
          ArrayAttr::get(kernel.getContext(), shape),
          ArrayAttr::get(kernel.getContext(), mappings),
          sourceType.getValidity(), sourceType.getOwner()));
    }
    return WalkResult::advance();
}

LogicalResult alignReductionResultRelations(func::FuncOp kernel) {
  WalkResult result = kernel.walk(alignReductionResultRelation);
  return result.wasInterrupted() ? failure() : success();
}

static WalkResult alignReductionIdentityRelation(Operation *operation) {
    auto align = [&](ValueRange inputs, ValueRange results, Region &combine,
                     uint64_t sourceCount,
                     uint64_t identityCount) -> LogicalResult {
      if (identityCount != results.size())
        return failure();
      OpBuilder builder(operation);
      for (unsigned index = 0; index < identityCount; ++index) {
        unsigned operand = sourceCount + index;
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, operation->getLoc(), inputs[operand],
            results[index].getType());
        if (failed(projected))
          return operation->emitOpError(
              "physical reduction identity cannot adopt its result relation");
        operation->setOperand(operand, *projected);
      }
      if (!llvm::hasSingleElement(combine) ||
          combine.front().getNumArguments() < 2 * identityCount)
        return operation->emitOpError(
            "physical reduction helper has no complete accumulator schema");
      for (unsigned index = 0; index < identityCount; ++index) {
        Type target = results[index].getType();
        combine.front().getArgument(index).setType(target);
        combine.front().getArgument(identityCount + index).setType(target);
      }
      return success();
    };
    if (auto reduce = dyn_cast<ReduceOp>(operation))
      return failed(align(reduce.getInputs(), reduce.getResults(),
                          reduce.getCombine(),
                          reduce.getSourceCount(), reduce.getIdentityCount()))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    if (auto scan = dyn_cast<ScanOp>(operation))
      return failed(align(scan.getInputs(), scan.getResults(), scan.getCombine(),
                          scan.getSourceCount(), scan.getIdentityCount()))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    return WalkResult::advance();
}

LogicalResult alignReductionIdentityRelations(func::FuncOp kernel) {
  WalkResult result = kernel.walk(alignReductionIdentityRelation);
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult alignReductionYieldRelations(func::FuncOp kernel) {
  WalkResult result = kernel.walk([&](Operation *operation) {
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
    for (auto [index, target] : llvm::enumerate(results)) {
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, operation->getLoc(), yield.getValues()[index],
          target.getType());
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
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult alignAccessResultRelations(func::FuncOp kernel) {
  WalkResult result = kernel.walk([&](Operation *operation) {
    ValueRange coordinates;
    Value result;
    if (auto load = dyn_cast<LoadOp>(operation)) {
      coordinates = load.getCoordinates();
      result = load.getResult();
    } else if (auto gather = dyn_cast<GatherOp>(operation)) {
      coordinates = gather.getCoordinates();
      result = gather.getResult();
    } else {
      return WalkResult::advance();
    }
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
          cast<AxisMapAttr>(mapping).getDimensionId());
    }
    result.setType(*refined);
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

static LogicalResult refreshReshapeRelation(ReshapeOp reshape);

LogicalResult alignPointwiseValueRelations(func::FuncOp kernel) {
  auto isParameterExtent = [](Attribute attribute) {
    auto extent = dyn_cast<PhysicalExprAttr>(attribute);
    return extent &&
           extent.getKind() ==
               static_cast<uint32_t>(PhysicalExprKind::Parameter);
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
    FailureOr<Value> replacement =
        projectPhysicalValueToSchema(builder, operation->getLoc(), value, target);
    if (failed(replacement))
      return failure();
    operation->setOperand(operandIndex, *replacement);
    return success();
  };

  WalkResult result = kernel.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (isa<ReduceOp, ScanOp>(operation)) {
      if (alignReductionResultRelation(operation).wasInterrupted() ||
          alignReductionIdentityRelation(operation).wasInterrupted())
        return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (auto reshape = dyn_cast<ReshapeOp>(operation))
      return failed(refreshReshapeRelation(reshape)) ? WalkResult::interrupt()
                                                     : WalkResult::advance();
    if (auto transpose = dyn_cast<TransposeOp>(operation)) {
      auto source = cast<FragmentType>(transpose.getValue().getType());
      auto target = cast<FragmentType>(transpose.getResult().getType());
      SmallVector<Attribute> shape;
      for (int64_t input : transpose.getPermutation())
        shape.push_back(source.getShape()[input]);
      transpose.getResult().setType(FragmentType::get(
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
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, broadcast.getLoc(), broadcast.getValue(), target);
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
                static_cast<uint32_t>(PhysicalExprKind::Constant) &&
            sourceExtent.getValue() == 1;
        if (singleton)
          continue;
        auto targetExtent =
            cast<PhysicalExprAttr>(target.getShape()[targetAxis]);
        bool targetSingleton =
            targetExtent.getKind() ==
                static_cast<uint32_t>(PhysicalExprKind::Constant) &&
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
                                 sourceAxisIdentity(targetMap), sourceExtent);
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
                                 sourceAxisIdentity(targetMap), sourceExtent);
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
              cast<PhysicalExprAttr>(target.getShape()[targetAxis]));
        else
          retargetSourceExtent(broadcast.getResult(),
                               sourceAxisIdentity(targetMap), sourceExtent);
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
      broadcast.getResult().setType(FragmentType::get(
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
        operation->getResult(0).setType(target);
      }
    }
    if (!target)
      return WalkResult::advance();
    if (isa<UnaryOp, CastOp, BitcastOp>(operation) &&
        operation->getNumOperands() == 1) {
      auto source = dyn_cast<FragmentType>(operation->getOperand(0).getType());
      if (source) {
        operation->getResult(0).setType(FragmentType::get(
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
          cast<AxisMapAttr>(target.getAxisMaps()[axis]).getDimensionId());
    }
    target = *refined;
    operation->getResult(0).setType(target);
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
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult alignAccessValueRelations(func::FuncOp kernel) {
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
    return projectPhysicalValueToSchema(builder, location, value, target);
  };
  auto alignCoordinates = [&](auto access, Type valueType) -> LogicalResult {
    auto payload = dyn_cast<FragmentType>(valueType);
    if (!payload)
      return success();
    PhysicalProgramAnalysis analysis(kernel);
    for (auto [slot, coordinate] : llvm::enumerate(access.getCoordinates())) {
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
      FailureOr<Value> aligned = project(builder, access.getLoc(), coordinate, target);
      if (failed(aligned))
        return access.emitOpError("cannot align coordinate with its access schema")
               << "; coordinate=" << coordinate;
      access.getCoordinatesMutable().slice(slot, 1).assign(*aligned);
    }
    return success();
  };

  SmallVector<LoadOp> loads;
  SmallVector<GatherOp> gathers;
  SmallVector<StoreOp> stores;
  kernel.walk([&](LoadOp load) { loads.push_back(load); });
  kernel.walk([&](GatherOp gather) { gathers.push_back(gather); });
  kernel.walk([&](StoreOp store) { stores.push_back(store); });

  auto alignRead = [&](auto read, StringRef kind) -> LogicalResult {
    if (failed(alignCoordinates(read, read.getResult().getType())))
      return failure();
    if (!read.getValid() && !read.getFill())
      return success();
    if (!read.getValid())
      return read.emitOpError() << kind << " fill has no validity authority";
    OpBuilder builder(read);
    Type valueType = read.getResult().getType();
    auto fragment = dyn_cast<FragmentType>(valueType);
    if (isa<GatherOp>(read.getOperation()) && !fragment) {
      if (read.getFill())
        return success();
      return read.emitOpError("scalar gather validity and fill must remain paired");
    }
    Type validType = fragment ? Type(predicateType(fragment)) : builder.getI1Type();
    auto valid = project(builder, read.getLoc(), read.getValid(), validType);
    if (failed(valid))
      return read.emitOpError() << "cannot align " << kind
                               << " validity with its value schema";
    auto fill = read.getFill()
                    ? project(builder, read.getLoc(), read.getFill(), valueType)
                    : materializeZeroValue(builder, read.getLoc(), valueType);
    if (failed(fill))
      return read.emitOpError() << "cannot align " << kind
                               << " fill with its value schema";
    // Keep the access identity, coordinates, effects and all attributes intact;
    // only its already-established validity/fill relation changes here.
    read.getValidMutable().assign(ValueRange{*valid});
    read.getFillMutable().assign(ValueRange{*fill});
    return success();
  };
  for (LoadOp load : loads)
    if (failed(alignRead(load, "load")))
      return failure();
  for (GatherOp gather : gathers)
    if (failed(alignRead(gather, "gather")))
      return failure();

  for (StoreOp store : stores) {
    if (failed(alignCoordinates(store, store.getValue().getType())))
      return failure();
    if (!store.getValid())
      continue;
    auto currentType = dyn_cast<FragmentType>(store.getValue().getType());
    if (!currentType) {
      auto validity = dyn_cast<FragmentType>(store.getValid().getType());
      if (!validity || !llvm::all_of(validity.getShape(), [](Attribute extent) {
            return constantPhysicalExpression(cast<PhysicalExprAttr>(extent)) == 1;
          }))
        continue;
      // Ownership can retain a one-lane predicate for a scalar write.  Adopt
      // that schema without widening the write into additional lanes.
      currentType = FragmentType::get(
          kernel.getContext(), store.getValue().getType(), validity.getShape(),
          validity.getAxisMaps(), validity.getValidity(), validity.getOwner());
      OpBuilder builder(store);
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
                             projection.dimensionId);
      }
    }
    currentType = cast<FragmentType>(store.getValue().getType());
    OpBuilder builder(store);
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
          cast<PhysicalExprAttr>((*valueType).getShape()[axis]), dimension);
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
    if (*value == store.getValue() && *valid == store.getValid())
      continue;
    auto replacement = builder.create<StoreOp>(
        store.getLoc(), store.getResource(), store.getCoordinates(),
        *value, *valid, store.getSourceAxes());
    if (Attribute origin = store->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    store.erase();
  }
  return success();
}

static LogicalResult refreshReshapeRelation(ReshapeOp reshape) {
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
                static_cast<uint32_t>(PhysicalExprKind::Constant) ||
            extent.getValue() != 1)
          nonUnitAxes.push_back(axis);
      }
      if (nonUnitAxes.size() == 1)
        resultShape[resultPrefix + nonUnitAxes.front()] =
            cast<PhysicalExprAttr>(
                source.getShape()[sourcePrefix + sourceAxes.front()]);
    }
    reshape.getResult().setType(FragmentType::get(
        kernel.getContext(), target.getElementType(),
        ArrayAttr::get(kernel.getContext(), resultShape), target.getAxisMaps(),
        target.getValidity(), target.getOwner()));
    return reshape.verify();
}

LogicalResult refreshReshapeRelations(func::FuncOp kernel) {
  WalkResult result = kernel.walk([&](ReshapeOp reshape) {
    return failed(refreshReshapeRelation(reshape)) ? WalkResult::interrupt()
                                                  : WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult closeValueAccessRelations(func::FuncOp kernel) {
  if (failed(alignAccessResultRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignContractValueRelations(kernel)))
    return failure();
  return success();
}

LogicalResult alignAggregateValueRelations(func::FuncOp kernel) {
  kernel.walk([&](MakeRecordOp record) {
    RecordType current = record.getResult().getType();
    SmallVector<Attribute> fields;
    fields.reserve(record.getFields().size());
    for (Value field : record.getFields())
      fields.push_back(TypeAttr::get(field.getType()));
    record.getResult().setType(RecordType::get(
        kernel.getContext(), current.getFieldNames(),
        ArrayAttr::get(kernel.getContext(), fields), current.getOwner()));
  });
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
                static_cast<uint32_t>(PhysicalExprKind::Constant) &&
            leftExtent.getValue() == 1;
        bool rightUnit =
            rightExtent.getKind() ==
                static_cast<uint32_t>(PhysicalExprKind::Constant) &&
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
  WalkResult branches = kernel.walk([&](scf::IfOp branch) {
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
              kernel.getContext(), static_cast<uint32_t>(PhysicalExprKind::Constant),
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
      FailureOr<Value> projectedThen = projectPhysicalValueToSchema(
          thenBuilder, branch.getLoc(), thenYield.getResults()[index], *target);
      OpBuilder elseBuilder(elseYield);
      FailureOr<Value> projectedElse = projectPhysicalValueToSchema(
          elseBuilder, branch.getLoc(), elseYield.getResults()[index], *target);
      if (failed(projectedThen) || failed(projectedElse)) {
        branch.emitOpError(
            "control-flow branch cannot adopt its joined physical relation")
            << "; result_index=" << index << "; target=" << *target;
        return WalkResult::interrupt();
      }
      thenYield->setOperand(index, *projectedThen);
      elseYield->setOperand(index, *projectedElse);
      branch.getResult(index).setType(*target);
    }
    return WalkResult::advance();
  });
  if (branches.wasInterrupted())
    return failure();
  WalkResult result = kernel.walk([&](RegionFoldOp fold) {
    if (!llvm::hasSingleElement(fold.getSummarize()) ||
        !llvm::hasSingleElement(fold.getCombine()))
      return WalkResult::interrupt();
    auto summarizeYield =
        dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
    if (!summarizeYield ||
        summarizeYield.getValues().size() != fold.getIdentityCount())
      return WalkResult::interrupt();
    Block &combine = fold.getCombine().front();
    if (combine.getNumArguments() != 2 * fold.getIdentityCount())
      return WalkResult::interrupt();
    OpBuilder builder(fold);
    for (unsigned index = 0; index < fold.getIdentityCount(); ++index) {
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
        retargetDimensionExtent(fold.getResult(index), dimension, extent);
      unsigned identity = fold.getSourceCount() + index;
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, fold.getLoc(), fold.getInputs()[identity], target);
      if (failed(projected)) {
        fold.emitOpError(
            "region-fold identity cannot adopt its summary relation");
        return WalkResult::interrupt();
      }
      fold->setOperand(identity, *projected);
      fold.getResult(index).setType(target);
      combine.getArgument(index).setType(target);
      combine.getArgument(fold.getIdentityCount() + index).setType(target);
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  WalkResult loops = kernel.walk([&](scf::ForOp loop) {
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
      FailureOr<Value> projectedInit = projectPhysicalValueToSchema(
          initBuilder, loop.getLoc(), init, target);
      OpBuilder yieldBuilder(yield);
      FailureOr<Value> projectedYield = projectPhysicalValueToSchema(
          yieldBuilder, loop.getLoc(), yielded, target);
      if (failed(projectedInit) || failed(projectedYield)) {
        loop.emitOpError(
            "loop-carried value cannot adopt its unique physical relation")
            << "; init=" << init.getType() << "; yield=" << yielded.getType()
            << "; target=" << target;
        return WalkResult::interrupt();
      }
      loop.getInitArgsMutable()[index].assign(*projectedInit);
      yield->setOperand(index, *projectedYield);
      loop.getRegionIterArgs()[index].setType(target);
      loop.getResult(index).setType(target);
    }
    return WalkResult::advance();
  });
  if (loops.wasInterrupted())
    return failure();
  // Structured arguments/results are the record-schema authority.  Refresh
  // projections only after those schemas have been aligned; doing this before
  // the region owner leaves combine-body fields one refinement behind.
  kernel.walk([&](ExtractOp extract) {
    RecordType record = extract.getRecord().getType();
    if (extract.getField() < record.getFieldTypes().size())
      extract.getResult().setType(
          cast<TypeAttr>(record.getFieldTypes()[extract.getField()]).getValue());
  });
  return success();
}

static void retargetExtent(Value root, AxisSelector selects,
                           PhysicalExprAttr extent,
                           bool followLogicalDimension) {
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
             argument.getArgNumber() < fold.getSourceCount() &&
             selectsSegmentAxis(argument.getType(), fold.getAxis(), selects);
    if (auto scan = dyn_cast_or_null<RegionScanOp>(parent))
      return (argument.getOwner()->getParent() == &scan.getSummarize() ||
              argument.getOwner()->getParent() == &scan.getEmit()) &&
             argument.getArgNumber() < scan.getSourceCount() &&
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
        value.setType(replacement);
        // A make_range owns both the fragment schema and the SSA extent used
        // to materialize that schema.  Retarget them from the same physical
        // decision; leaving the operand behind creates two executable
        // authorities for one traversal.
        if (auto range = value.getDefiningOp<MakeRangeOp>()) {
          OpBuilder builder(range);
          Value physicalExtent;
          if (extent.getKind() ==
              static_cast<uint32_t>(PhysicalExprKind::Constant))
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
      appendStructuredParentRelations(argument, worklist, selects);
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
    appendStructuredResultRelations(value, worklist, selects);
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
                           static_cast<uint32_t>(PhysicalExprKind::Constant) &&
                       newExtent.getKind() ==
                           static_cast<uint32_t>(PhysicalExprKind::Constant) &&
                       oldExtent.getValue() > 0 &&
                       static_cast<uint64_t>(newExtent.getValue()) ==
                           llvm::PowerOf2Ceil(static_cast<uint64_t>(oldExtent.getValue()));
              });
          if (padding)
            buffer.getResult().setType(BufferType::get(
                storage.getContext(), storage.getElementType(), initial.getShape(),
                storage.getScope(), storage.getInstance(), storage.getOwner(),
                storage.getInitialization(), storage.getLifetime(),
                storage.getVisibility(), storage.getWorkspace()));
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
                              static_cast<uint32_t>(PhysicalExprKind::Constant) &&
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
        appendStructuredChildRelations(user, value, worklist, selects);
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
        extent, /*followLogicalDimension=*/false);
  }
}

void retargetSourceExtent(Value root, PhysicalSourceAxis source,
                          PhysicalExprAttr extent,
                          std::optional<int64_t> dimension) {
  retargetExtent(
      root,
      [=](AxisMapAttr mapping) {
        return mapping.getSourceId() == source.sourceId &&
               mapping.getSourceAxis() == source.sourceAxis &&
               mapping.getDerived() == source.derived &&
               (!dimension || mapping.getDimensionId() == *dimension);
      },
      extent, /*followLogicalDimension=*/false);
}

void retargetDimensionExtent(Value root, int64_t dimensionId,
                             PhysicalExprAttr extent) {
  if (dimensionId <= 0)
    return;
  retargetExtent(root,
                 [=](AxisMapAttr mapping) {
                   return mapping.getDimensionId() == dimensionId;
                 },
                 extent, /*followLogicalDimension=*/true);
}

LogicalResult alignStructuredCaptureRelations(func::FuncOp kernel) {
  auto sinkUniformCapture = [&](Operation *owner, unsigned operand,
                                ArrayRef<BlockArgument> arguments) {
    Value capture = owner->getOperand(operand);
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
      argument.setType(scalar.getType());
      OpBuilder builder(argument.getOwner(), argument.getOwner()->begin());
      auto splat = builder.create<SplatOp>(owner->getLoc(), type, argument);
      argument.replaceAllUsesExcept(splat.getResult(), splat.getOperation());
    }
    owner->setOperand(operand, scalar);
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
                              cast<PhysicalExprAttr>(authority.getShape()[axis]));
    }
    return capture.getType() == argument.getType()
               ? success()
               : owner->emitOpError(
                     "physical structured capture extent is inconsistent");
  };

  WalkResult result = kernel.walk([&](Operation *operation) {
    if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
      unsigned sourceCount = fold.getSourceCount();
      unsigned captureOffset = sourceCount + fold.getIdentityCount();
      for (unsigned index = 0; index < fold.getCaptureCount(); ++index) {
        sinkUniformCapture(
            operation, captureOffset + index,
            {fold.getSummarize().front().getArgument(sourceCount + index)});
        if (failed(align(operation, fold.getInputs()[captureOffset + index],
                         fold.getSummarize().front().getArgument(sourceCount +
                                                                 index))))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    auto scan = dyn_cast<RegionScanOp>(operation);
    if (!scan)
      return WalkResult::advance();
    unsigned sourceCount = scan.getSourceCount();
    unsigned stateCount = scan.getStateCount();
    unsigned captureOffset =
        sourceCount + scan.getIdentityCount() + stateCount;
    for (unsigned index = 0; index < scan.getCaptureCount(); ++index) {
      sinkUniformCapture(
          operation, captureOffset + index,
          {scan.getSummarize().front().getArgument(sourceCount + index),
           scan.getEmit().front().getArgument(sourceCount + stateCount + index)});
      Value capture = scan.getInputs()[captureOffset + index];
      if (failed(align(operation, capture,
                       scan.getSummarize().front().getArgument(sourceCount +
                                                               index))) ||
          failed(align(operation, capture,
                       scan.getEmit().front().getArgument(sourceCount +
                                                          stateCount + index))))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

void eraseDeadPhysicalValues(func::FuncOp kernel) {
  bool changed = false;
  do {
    changed = false;
    kernel.walk<WalkOrder::PostOrder>([&](Operation *operation) {
      if (!operation->getBlock() || isa<DelinearizeOp, ParameterOp>(operation) ||
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

LogicalResult alignContractValueRelations(func::FuncOp kernel) {
  if (failed(alignOrdinaryContractOperandTypes(kernel)) ||
      failed(alignContractAccumulatorTypes(kernel)))
    return failure();
  return success();
}

} // namespace intent::gpu
