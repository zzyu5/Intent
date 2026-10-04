#include "ReductionRealization.h"
#include "ReductionValues.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/Helpers.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

using namespace mlir;

namespace intent::gpu::reduction {
namespace {

// Scalarization compares the actual tuple callback, including field coupling
// and numeric attributes, without requiring its intermediate free-axis shapes
// to be identical. It does not grant reassociation to a scan or ordered loop.
bool scalarCombine(Region &source, Region &result) {
  SmallVector<Type> arguments, results;
  for (Type type : source.front().getArgumentTypes())
    arguments.push_back(scalarCallbackType(type));
  for (Type type : source.front().getTerminator()->getOperandTypes())
    results.push_back(scalarCallbackType(type));
  std::string reason;
  if (failed(cloneLaneWiseHelper(source, result, arguments, results, reason)))
    return false;
  result.walk([](Operation *operation) { operation->removeAttr(originAttr); });
  return true;
}

bool sameContract(ReduceOp inner, ReduceOp outer, Region &outerCombine) {
  if (inner.getAxes().size() != 1 || outer.getAxes().size() != 1 ||
      inner.getSources().size() != outer.getSources().size() ||
      !llvm::equal(inner.getCaptures(), outer.getCaptures()) ||
      llvm::any_of(inner.getCaptures(), [](Value value) {
        return !value.getType().isIntOrIndexOrFloat();
      }))
    return false;
  for (auto [a, b] : llvm::zip(inner.getIdentities(), outer.getIdentities()))
    if (!sameScalarValue(a, b))
      return false;
  Region innerCombine;
  return scalarCombine(inner.getCombine(), innerCombine) &&
         OperationEquivalence::isRegionEquivalentTo(
             &innerCombine, &outerCombine,
             OperationEquivalence::IgnoreLocations);
}

// This normalization consumes complete physical fragments. It deliberately
// does not infer that a padded member is neutral from its cardinality or shape.
// Incomplete/unknown logical ranges remain on the existing realization path.
bool completeAxis(Value source, unsigned axis, PhysicalProgramAnalysis &analysis,
                  IndexRelations &relations) {
  auto type = cast<FragmentType>(source.getType());
  auto extent = constantPhysicalExpression(
      cast<PhysicalExprAttr>(type.getShape()[axis]));
  if (!extent || *extent <= 0)
    return false;
  auto realization = analysis.axisRealization(source, axis);
  auto ranges = analysis.axisRanges(source, axis);
  if (!realization.isExact() || !realization.physicalized ||
      realization.constructionScalarSeed || !ranges.isExact() ||
      !ranges.blockers.empty() || ranges.roots.empty())
    return false;
  return llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
    return isUnitStepRange(range) &&
           range.getResult().getType().getShape()[0] == type.getShape()[axis] &&
           samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()) &&
           relations.atMost(range.getLogicalStart(), range.getLogicalStop()) &&
           constantLogicalRangeCardinality(range) == extent;
  });
}

bool collapseChain(ReduceOp outer, func::FuncOp kernel) {
  if (outer.getAxes().size() != 1 || outer.getSources().empty())
    return false;
  Region outerCombine;
  if (!scalarCombine(outer.getCombine(), outerCombine))
    return false;

  SmallVector<ReduceOp> chain{outer};
  while (true) {
    ReduceOp consumer = chain.back();
    auto inner = consumer.getSources().front().getDefiningOp<ReduceOp>();
    if (!inner || inner->getBlock() != outer->getBlock() ||
        !inner->isBeforeInBlock(consumer) ||
        !llvm::equal(inner.getResults(), consumer.getSources()) ||
        llvm::any_of(inner.getResults(), [](Value value) { return !value.hasOneUse(); }) ||
        !sameContract(inner, outer, outerCombine))
      break;
    chain.push_back(inner);
  }
  if (chain.size() < 2)
    return false;

  ReduceOp first = chain.back();
  auto original = dyn_cast<FragmentType>(first.getSources().front().getType());
  if (!original)
    return false;
  SmallVector<unsigned> remaining(original.getShape().size());
  std::iota(remaining.begin(), remaining.end(), 0);
  SmallVector<unsigned> reduced;
  for (ReduceOp reduce : llvm::reverse(chain)) {
    unsigned axis = reduce.getAxes().front();
    if (axis >= remaining.size())
      return false;
    reduced.push_back(remaining[axis]);
    remaining.erase(remaining.begin() + axis);
  }
  llvm::sort(reduced);
  for (unsigned i = 1; i < reduced.size(); ++i)
    if (reduced[i] != reduced.front() + i)
      return false;

  PhysicalProgramAnalysis analysis(kernel);
  IndexRelations relations;
  for (auto [source, result] : llvm::zip(first.getSources(), outer.getResults())) {
    auto type = dyn_cast<FragmentType>(source.getType());
    if (!type || type.getShape() != original.getShape() ||
        type.getAxisMaps() != original.getAxisMaps() ||
        type.getValidity() != original.getValidity() ||
        type.getOwner() != original.getOwner() ||
        type.getElementType() != dataElementType(result.getType()) ||
        llvm::any_of(reduced, [&](unsigned axis) {
          return !completeAxis(source, axis, analysis, relations);
        }))
      return false;
  }
  int64_t capacity = 1;
  for (unsigned axis : reduced) {
    auto extent = constantPhysicalExpression(
        cast<PhysicalExprAttr>(original.getShape()[axis]));
    if (!extent || llvm::MulOverflow(capacity, *extent, capacity))
      return false;
  }

  MLIRContext *context = kernel.getContext();
  auto [sourceId, dimensionId] = nextPhysicalAxisIdentities(kernel);
  unsigned mergedAxis = reduced.front();
  SmallVector<Attribute> shape, mappings, groups;
  for (unsigned axis = 0; axis < original.getShape().size(); ++axis) {
    unsigned targetAxis = shape.size();
    if (axis == mergedAxis) {
      shape.push_back(PhysicalExprAttr::get(context, PhysicalExprKind::Constant,
          capacity, StringAttr::get(context), ArrayAttr::get(context, {})));
      mappings.push_back(AxisMapAttr::get(context, sourceId, 0, dimensionId,
                                         targetAxis, true));
      SmallVector<int64_t> axes(reduced.begin(), reduced.end());
      groups.push_back(ReshapeGroupAttr::get(context,
          DenseI64ArrayAttr::get(context, axes),
          DenseI64ArrayAttr::get(context, {static_cast<int64_t>(targetAxis)})));
      axis = reduced.back();
      continue;
    }
    shape.push_back(original.getShape()[axis]);
    auto mapping = cast<AxisMapAttr>(original.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(context, mapping.getSourceId(),
        mapping.getSourceAxis(), mapping.getDimensionId(), targetAxis,
        mapping.getDerived()));
    groups.push_back(ReshapeGroupAttr::get(context,
        DenseI64ArrayAttr::get(context, {static_cast<int64_t>(axis)}),
        DenseI64ArrayAttr::get(context, {static_cast<int64_t>(targetAxis)})));
  }
  SmallVector<FragmentType> types;
  for (auto [source, result] : llvm::zip(first.getSources(), outer.getResults())) {
    auto type = cast<FragmentType>(source.getType());
    auto collapsed = FragmentType::get(context, type.getElementType(),
        ArrayAttr::get(context, shape), ArrayAttr::get(context, mappings),
        type.getValidity(), type.getOwner());
    auto expected = inferCollectiveResultType(collapsed,
        ArrayRef<int64_t>{static_cast<int64_t>(mergedAxis)},
        dataElementType(result.getType()));
    if (failed(expected) || *expected != result.getType())
      return false;
    types.push_back(collapsed);
  }

  // No source computation or memory access moves: only the two native trees
  // and their intermediate values are replaced, at the final reduction point.
  OpBuilder builder(outer);
  IRMapping mapping;
  for (auto [source, oldInput, type] :
       llvm::zip(first.getSources(), outer.getSources(), types)) {
    Value value = builder.create<ReshapeOp>(outer.getLoc(), type, source,
                                          ArrayAttr::get(context, groups));
    mapping.map(oldInput, value);
  }
  auto replacement = cast<ReduceOp>(builder.clone(*outer, mapping));
  replacement.setAxesAttr(builder.getDenseI64ArrayAttr(
      {static_cast<int64_t>(mergedAxis)}));
  for (auto [before, after] : llvm::zip(outer.getResults(), replacement.getResults()))
    before.replaceAllUsesWith(after);
  for (ReduceOp reduce : chain)
    reduce.erase();
  return true;
}

} // namespace

bool normalizeCompletedReductions(func::FuncOp kernel) {
  bool changed = false;
  while (true) {
    SmallVector<ReduceOp> reductions;
    kernel.walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
    bool rewritten = false;
    for (ReduceOp reduce : llvm::reverse(reductions))
      if (collapseChain(reduce, kernel)) {
        rewritten = changed = true;
        break;
      }
    if (!rewritten)
      return changed;
  }
}

} // namespace intent::gpu::reduction
