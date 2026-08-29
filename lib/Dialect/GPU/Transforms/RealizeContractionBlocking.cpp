#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <functional>
#include <tuple>

using namespace mlir;

namespace intent::gpu {
namespace {

PhysicalExprAttr expression(MLIRContext *context, PhysicalExprKind kind,
                            int64_t value = 0, StringRef symbol = {},
                            ArrayRef<Attribute> operands = {}) {
  return PhysicalExprAttr::get(
      context, static_cast<uint32_t>(kind), value,
      StringAttr::get(context, symbol), ArrayAttr::get(context, operands));
}

PhysicalExprAttr parameterExpression(MLIRContext *context, StringRef name) {
  return expression(context, PhysicalExprKind::Parameter, 0, name);
}

PhysicalExprAttr binaryExpression(MLIRContext *context, PhysicalExprKind kind,
                                  PhysicalExprAttr lhs,
                                  PhysicalExprAttr rhs) {
  return expression(context, kind, 0, {}, {lhs, rhs});
}

bool isCompileTimeExtent(PhysicalExprAttr expression) {
  auto kind = static_cast<PhysicalExprKind>(expression.getKind());
  if (kind == PhysicalExprKind::Constant || kind == PhysicalExprKind::Parameter)
    return true;
  if (kind == PhysicalExprKind::Dimension ||
      kind == PhysicalExprKind::ScalarABI)
    return false;
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isCompileTimeExtent(cast<PhysicalExprAttr>(operand));
  });
}

bool hasRuntimeRange(Value value);

bool isFullCoverageExtent(Operation *origin, Attribute attribute) {
  auto extent = dyn_cast<PhysicalExprAttr>(attribute);
  if (!origin || !extent ||
      extent.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Parameter))
    return false;
  func::FuncOp kernel = origin->getParentOfType<func::FuncOp>();
  if (!kernel)
    return false;
  bool fullCoverage = false;
  kernel.walk([&](ParameterOp parameter) {
    if (parameter.getParameter().getName() == extent.getSymbol() &&
        parameter->hasAttr(coverageDimensionAttr))
      fullCoverage = true;
  });
  return fullCoverage;
}

bool hasFragmentSchema(ContractOp contract) {
  return isa<FragmentType>(contract.getLhs().getType()) &&
         isa<FragmentType>(contract.getRhs().getType()) &&
         isa<FragmentType>(contract.getResult().getType());
}

bool hasFragmentSchema(ScaledContractOp contract) {
  return isa<FragmentType>(contract.getLhs().getType()) &&
         isa<FragmentType>(contract.getLhsScale().getType()) &&
         isa<FragmentType>(contract.getRhs().getType()) &&
         isa<FragmentType>(contract.getRhsScale().getType()) &&
         isa<FragmentType>(contract.getAccumulator().getType()) &&
         isa<FragmentType>(contract.getResult().getType());
}

bool requiresPhysicalRealization(ContractOp contract) {
  for (FragmentType type : {contract.getLhs().getType(),
                            contract.getRhs().getType(),
                            contract.getResult().getType()})
    if (llvm::any_of(type.getShape(), [&](Attribute extent) {
          return !isCompileTimeExtent(cast<PhysicalExprAttr>(extent)) ||
                 isFullCoverageExtent(contract, extent);
        }))
      return true;
  return hasRuntimeRange(contract.getLhs()) || hasRuntimeRange(contract.getRhs());
}

bool requiresPhysicalRealization(ScaledContractOp contract) {
  for (FragmentType type : {
           contract.getLhs().getType(), contract.getLhsScale().getType(),
           contract.getRhs().getType(), contract.getRhsScale().getType(),
           contract.getAccumulator().getType(), contract.getResult().getType()})
    if (llvm::any_of(type.getShape(), [&](Attribute extent) {
          return !isCompileTimeExtent(cast<PhysicalExprAttr>(extent)) ||
                 isFullCoverageExtent(contract, extent);
        }))
      return true;
  return hasRuntimeRange(contract.getLhs()) ||
         hasRuntimeRange(contract.getLhsScale()) ||
         hasRuntimeRange(contract.getRhs()) ||
         hasRuntimeRange(contract.getRhsScale());
}

Value binary(OpBuilder &builder, Location location, Type result, Value lhs,
             Value rhs, BinaryOperator kind) {
  return builder.create<BinaryOp>(location, result, lhs, rhs, kind);
}

Value compare(OpBuilder &builder, Location location, Type result, Value lhs,
              Value rhs, ComparePredicate predicate) {
  return builder.create<CompareOp>(location, result, lhs, rhs, predicate);
}

void inheritRangeAuthority(Value derived, MakeRangeOp source) {
  Operation *operation = derived.getDefiningOp();
  if (Attribute value = source->getAttr(sourceSubregionAttr))
    operation->setAttr(sourceSubregionAttr, value);
}

FailureOr<unsigned> uniqueFreeAxis(FragmentType fragment,
                                   ArrayRef<int64_t> reduction,
                                   ArrayRef<int64_t> batch) {
  std::optional<unsigned> result;
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    if (llvm::is_contained(reduction, static_cast<int64_t>(axis)) ||
        llvm::is_contained(batch, static_cast<int64_t>(axis)))
      continue;
    if (result)
      return failure();
    result = axis;
  }
  return result ? FailureOr<unsigned>(*result) : FailureOr<unsigned>(failure());
}

MakeRangeOp sourceRange(Value value) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return {};
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.sourceRanges(value);
  FailureOr<MakeRangeOp> range = queryExactLogicalRange(fact);
  return succeeded(range) ? *range : MakeRangeOp();
}

FragmentType replaceSourceExtent(FragmentType source, PhysicalSourceAxis logical,
                                 PhysicalExprAttr extent) {
  PhysicalAxisProjection axis = queryFragmentAxis(source, logical);
  if (!axis.isExact())
    return source;
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  shape[axis.fragmentAxis] = extent;
  return FragmentType::get(source.getContext(), source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           source.getAxisMaps(), source.getValidity(),
                           source.getOwner());
}

bool containsSource(Value value, PhysicalSourceAxis source) {
  return queryFragmentAxis(value.getType(), source).isExact();
}

LogicalResult collectProducerRanges(
    Value value, PhysicalSourceAxis source,
    SmallVectorImpl<MakeRangeOp> &ranges);

bool hasExplicitPairedReductionRanges(ContractOp contract) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1)
    return false;
  auto lhsType = dyn_cast<FragmentType>(contract.getLhs().getType());
  auto rhsType = dyn_cast<FragmentType>(contract.getRhs().getType());
  if (!lhsType || !rhsType)
    return false;
  FailureOr<AxisMapAttr> lhsMap =
      queryAxisMap(lhsType, contract.getLhsReductionAxes().front());
  FailureOr<AxisMapAttr> rhsMap =
      queryAxisMap(rhsType, contract.getRhsReductionAxes().front());
  if (failed(lhsMap) || failed(rhsMap))
    return false;
  auto hasRanges = [&](Value value, AxisMapAttr mapping) {
    SmallVector<MakeRangeOp> ranges;
    if (failed(collectProducerRanges(
            value,
            PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis()},
            ranges)) ||
        ranges.empty())
      return false;
    return llvm::all_of(ranges, [&](MakeRangeOp range) {
      return range.getSourceId() == mapping.getSourceId() &&
             range.getSourceAxis() == mapping.getSourceAxis();
    });
  };
  return hasRanges(contract.getLhs(), *lhsMap) &&
         hasRanges(contract.getRhs(), *rhsMap);
}

LogicalResult collectProducerRanges(
    Value value, PhysicalSourceAxis source,
    SmallVectorImpl<MakeRangeOp> &ranges) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.sourceRanges(value, source);
  if (fact.state == PhysicalFactState::Unknown)
    return failure();
  for (MakeRangeOp range : fact.roots)
    if (!llvm::is_contained(ranges, range))
      ranges.push_back(range);
  return ranges.empty() ? failure() : success();
}

FailureOr<MakeRangeOp> producerRange(Value value, PhysicalSourceAxis source) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalRangeFact fact =
      PhysicalProgramAnalysis(kernel).sourceRanges(value, source);
  return queryExactLogicalRange(fact);
}

FailureOr<Value> replaySourceValue(OpBuilder &builder, Location location,
                                   Value value, PhysicalSourceAxis source,
                                   PhysicalExprAttr blockedExtent,
                                   MakeRangeOp root, Value replacement,
                                   IRMapping &mapping) {
  if (value == root.getResult())
    return replacement;
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  if (!containsSource(value, source))
    return value;
  Operation *producer = value.getDefiningOp();
  if (!producer || isa<MakeRangeOp>(producer))
    return failure();
  if (!isPhysicalReplayNode(producer, PhysicalReplayScope::Coordinate,
                            /*allowAccesses=*/true))
    return failure();
  if (producer->getNumRegions() != 0 || producer->getNumResults() != 1)
    return failure();
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replayed = replaySourceValue(
        builder, location, operand, source, blockedExtent, root, replacement,
        mapping);
    if (failed(replayed))
      return failure();
    if (*replayed != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replayed);
  }
  auto originalResultType = dyn_cast<FragmentType>(producer->getResult(0).getType());
  if (!originalResultType)
    return failure();
  FragmentType targetType =
      replaceSourceExtent(originalResultType, source, blockedExtent);
  if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
    auto inputType = dyn_cast<FragmentType>(reshape.getValue().getType());
    if (!inputType || !queryFragmentAxis(inputType, source).isExact()) {
      Value input = mapping.lookupOrDefault(reshape.getValue());
      auto broadcast = builder.create<BroadcastOp>(location, targetType, input);
      if (!mapping.lookupOrNull(value))
        mapping.map(value, broadcast.getResult());
      return broadcast.getResult();
    }
  }
  Operation *clone = builder.clone(*producer, mapping);
  auto resultType = dyn_cast<FragmentType>(clone->getResult(0).getType());
  if (!resultType)
    return failure();
  clone->getResult(0).setType(targetType);
  if (!mapping.lookupOrNull(value))
    mapping.map(value, clone->getResult(0));
  return clone->getResult(0);
}

Value strippedBroadcast(Value value) {
  while (auto broadcast = value.getDefiningOp<BroadcastOp>())
    value = broadcast.getValue();
  return value;
}

bool isIntegerConstant(Value value, int64_t expected) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  return integer && integer.getInt() == expected;
}

bool isTailPredicate(Value value,
                     ArrayRef<std::pair<MakeRangeOp, Value>> ranges) {
  if (!value)
    return true;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  return kernel && PhysicalProgramAnalysis(kernel).isTailPredicate(value, ranges);
}

bool hasRuntimeRange(Value value) {
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment)
    return false;
  SmallVector<PhysicalSourceAxis> sources;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    PhysicalSourceAxis source{mapping.getSourceId(), mapping.getSourceAxis()};
    if (!llvm::is_contained(sources, source))
      sources.push_back(source);
  }
  for (PhysicalSourceAxis source : sources) {
    SmallVector<MakeRangeOp> ranges;
    if (failed(collectProducerRanges(value, source, ranges)))
      continue;
    for (MakeRangeOp range : ranges) {
      Value extent = range.getExtent();
      if (!extent.getDefiningOp<arith::ConstantOp>() &&
          !extent.getDefiningOp<ParameterOp>() &&
          !extent.getDefiningOp<PhysicalExprOp>())
        return true;
    }
  }
  return false;
}

FailureOr<ParameterOp> parameterForExtent(func::FuncOp kernel,
                                          PhysicalExprAttr extent) {
  if (extent.getKind() !=
      static_cast<uint32_t>(PhysicalExprKind::Parameter))
    return failure();
  ParameterOp result;
  kernel.walk([&](ParameterOp parameter) {
    if (!result && parameter.getParameter().getName() == extent.getSymbol())
      result = parameter;
  });
  return result ? FailureOr<ParameterOp>(result)
                : FailureOr<ParameterOp>(failure());
}

LogicalResult markNativeCoverage(func::FuncOp kernel, Value source,
                                 ArrayRef<int64_t> axes) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment)
    return failure();
  for (int64_t axis : axes) {
    if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
      return failure();
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    FailureOr<ParameterOp> parameter = parameterForExtent(kernel, extent);
    if (failed(parameter))
      continue;
    uint32_t role = parameter->getParameter().getRole();
    if (role == static_cast<uint32_t>(ParameterRole::ScanChunk) ||
        role == static_cast<uint32_t>(ParameterRole::Reduction))
      continue;
    PhysicalParameterBinding binding = queryParameterBinding(*parameter);
    if (!binding.isExact() || !binding.dimension)
      return parameter->emitOpError(
          "native contraction coverage parameter has no logical dimension");
    uint64_t dimension = *binding.dimension;
    bool launchVisible = false;
    for (BlockArgument argument : kernel.getArguments()) {
      DictionaryAttr attributes = kernel.getArgAttrDict(argument.getArgNumber());
      auto kind = attributes.getAs<StringAttr>(abiKindAttr);
      auto identity = attributes.getAs<IntegerAttr>(dimensionAttr);
      launchVisible |= kind && kind.getValue() == "dimension" && identity &&
                       identity.getInt() == static_cast<int64_t>(dimension);
    }
    if (!launchVisible)
      return parameter->emitOpError(
          "native contraction cannot fully cover a data-dependent dimension");
    static constexpr int64_t fullCoverageCandidates[] = {
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536};
    ParameterAttr schema = parameter->getParameter();
    (*parameter)->setAttr(
        "parameter",
        ParameterAttr::get(
            kernel.getContext(), schema.getName(), schema.getRole(),
            DenseI64ArrayAttr::get(kernel.getContext(), fullCoverageCandidates)));
    if (auto current =
            (*parameter)->getAttrOfType<IntegerAttr>(coverageDimensionAttr)) {
      if (current.getInt() != static_cast<int64_t>(dimension))
        return parameter->emitOpError(
            "one physical parameter covers multiple logical dimensions");
    } else {
      (*parameter)->setAttr(
          coverageDimensionAttr,
          IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), dimension));
    }
    if (failed(bindFullCoverageDimension(kernel, dimension,
                                         parameter->getResult())))
      return parameter->emitOpError(
          "native contraction coverage could not bind its exact logical dimension");

    // A logical subregion keeps its dynamic end, but a provider-native
    // contraction still needs a compile-time fragment extent.  The coverage
    // parameter covers the parent logical dimension; the existing subregion
    // predicate remains the authority for active members in the final tile.
    // Retarget only the exact provenance carried by this operand axis.
    PhysicalExprAttr parameterExtent = parameterExpression(
        kernel.getContext(), parameter->getParameter().getName().getValue());
    auto sourceMapping = cast<AxisMapAttr>(
        fragment.getAxisMaps()[static_cast<unsigned>(axis)]);
    PhysicalSourceAxis source{sourceMapping.getSourceId(),
                              sourceMapping.getSourceAxis()};
    SmallVector<MakeRangeOp> subregions;
    kernel.walk([&](MakeRangeOp range) {
      FailureOr<int64_t> sourceDimension = queryRangeDimension(range);
      if (range.getSourceId() == source.sourceId &&
          range.getSourceAxis() == source.sourceAxis &&
          range->hasAttr(sourceSubregionAttr) && succeeded(sourceDimension) &&
          *sourceDimension == static_cast<int64_t>(dimension))
        subregions.push_back(range);
    });
    for (MakeRangeOp range : subregions) {
      if (failed(resolveLogicalRangeEnd(kernel, range)))
        return range.emitOpError(
            "native contraction subregion has no exact logical tail bound");
      retargetSourceExtent(range.getResult(), source, parameterExtent);
      range->setOperand(1, parameter->getResult());
    }
  }
  return success();
}

LogicalResult markNativeCoverage(func::FuncOp kernel, ContractOp contract) {
  if (failed(markNativeCoverage(kernel, contract.getLhs(),
                                contract.getLhsReductionAxes())) ||
      failed(markNativeCoverage(kernel, contract.getRhs(),
                                contract.getRhsReductionAxes())))
    return contract.emitOpError(
        "native contraction reduction coverage is not representable");
  return success();
}

LogicalResult markNativeCoverage(func::FuncOp kernel,
                                 ScaledContractOp contract) {
  if (failed(markNativeCoverage(kernel, contract.getLhs(),
                                contract.getLhsReductionAxes())) ||
      failed(markNativeCoverage(kernel, contract.getRhs(),
                                contract.getRhsReductionAxes())))
    return contract.emitOpError(
        "native scaled contraction reduction coverage is not representable");
  return success();
}

FragmentType transposeLastTwo(FragmentType source) {
  const unsigned rank = source.getShape().size();
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  std::swap(shape[rank - 2], shape[rank - 1]);
  SmallVector<Attribute> mappings;
  mappings.reserve(rank);
  for (unsigned resultAxis = 0; resultAxis < rank; ++resultAxis) {
    unsigned sourceAxis = resultAxis;
    if (resultAxis == rank - 2)
      sourceAxis = rank - 1;
    else if (resultAxis == rank - 1)
      sourceAxis = rank - 2;
    auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    mappings.push_back(AxisMapAttr::get(
        source.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), resultAxis));
  }
  return FragmentType::get(
      source.getContext(), source.getElementType(),
      ArrayAttr::get(source.getContext(), shape),
      ArrayAttr::get(source.getContext(), mappings), source.getValidity(),
      source.getOwner());
}

SmallVector<int64_t> remapAxes(ArrayRef<int64_t> axes,
                               ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> inverse(permutation.size());
  for (auto [resultAxis, sourceAxis] : llvm::enumerate(permutation))
    inverse[static_cast<unsigned>(sourceAxis)] = resultAxis;
  SmallVector<int64_t> result;
  result.reserve(axes.size());
  for (int64_t axis : axes)
    result.push_back(inverse[static_cast<unsigned>(axis)]);
  return result;
}

LogicalResult normalizeMatrixContractForms(func::FuncOp kernel) {
  SmallVector<ContractOp> contracts;
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  for (ContractOp contract : contracts) {
    if (contract.getLhsReductionAxes().size() != 1 ||
        contract.getRhsReductionAxes().size() != 1)
      continue;
    auto lhs = contract.getLhs().getType();
    auto rhs = contract.getRhs().getType();
    unsigned lhsRank = lhs.getShape().size();
    unsigned rhsRank = rhs.getShape().size();
    if (lhsRank < 2 || rhsRank < 2)
      continue;
    OpBuilder builder(contract);
    if (contract.getLhsReductionAxes().front() ==
        static_cast<int64_t>(lhsRank - 2)) {
      SmallVector<int64_t> permutation;
      for (unsigned axis = 0; axis < lhsRank; ++axis)
        permutation.push_back(axis);
      std::swap(permutation[lhsRank - 2], permutation[lhsRank - 1]);
      Value transposed = builder.create<TransposeOp>(
          contract.getLoc(), transposeLastTwo(lhs), contract.getLhs(),
          permutation);
      contract->setOperand(0, transposed);
      contract->setAttr("lhs_reduction_axes",
                        builder.getDenseI64ArrayAttr(remapAxes(
                            contract.getLhsReductionAxes(), permutation)));
      contract->setAttr("lhs_batch_axes",
                        builder.getDenseI64ArrayAttr(
                            remapAxes(contract.getLhsBatchAxes(), permutation)));
    }
    if (contract.getRhsReductionAxes().front() ==
        static_cast<int64_t>(rhsRank - 1)) {
      SmallVector<int64_t> permutation;
      for (unsigned axis = 0; axis < rhsRank; ++axis)
        permutation.push_back(axis);
      std::swap(permutation[rhsRank - 2], permutation[rhsRank - 1]);
      Value transposed = builder.create<TransposeOp>(
          contract.getLoc(), transposeLastTwo(rhs), contract.getRhs(),
          permutation);
      contract->setOperand(1, transposed);
      contract->setAttr("rhs_reduction_axes",
                        builder.getDenseI64ArrayAttr(remapAxes(
                            contract.getRhsReductionAxes(), permutation)));
      contract->setAttr("rhs_batch_axes",
                        builder.getDenseI64ArrayAttr(
                            remapAxes(contract.getRhsBatchAxes(), permutation)));
    }
  }
  return success();
}

FragmentType fragmentType(MLIRContext *context, Type element,
                          ArrayRef<PhysicalExprAttr> shape,
                          ArrayRef<AxisMapAttr> sourceMappings,
                          uint64_t owner) {
  SmallVector<Attribute> extents(shape.begin(), shape.end());
  SmallVector<Attribute> mappings;
  for (auto [axis, source] : llvm::enumerate(sourceMappings))
    mappings.push_back(AxisMapAttr::get(context, source.getSourceId(),
                                        source.getSourceAxis(),
                                        source.getDimensionId(), axis));
  return FragmentType::get(context, element, ArrayAttr::get(context, extents),
                           ArrayAttr::get(context, mappings), 1, owner);
}

FailureOr<Value> scalarSource(Value value) {
  if (!isa<FragmentType>(value.getType()))
    return value;
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    if (!isa<FragmentType>(broadcast.getValue().getType()))
      return broadcast.getValue();
  if (auto splat = value.getDefiningOp<SplatOp>())
    return splat.getValue();
  return failure();
}

bool isZeroScalar(Value value) {
  FailureOr<Value> scalar = scalarSource(value);
  if (failed(scalar))
    return false;
  auto constant = (*scalar).getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
    return floating.getValue().isZero();
  return false;
}

Value broadcast(OpBuilder &builder, Location location, FragmentType result,
                Value value) {
  return builder.create<BroadcastOp>(location, result, value);
}

FailureOr<Value> retargetPredicate(OpBuilder &builder, Location location,
                                   Value original, FragmentType result) {
  if (!original)
    return failure();
  FailureOr<Value> scalar = scalarSource(original);
  if (failed(scalar) || !(*scalar).getType().isInteger(1))
    return failure();
  return Value(builder.create<BroadcastOp>(location, result, *scalar));
}

FailureOr<Value> retargetFill(OpBuilder &builder, Location location,
                              Value original, FragmentType result) {
  if (original) {
    FailureOr<Value> scalar = scalarSource(original);
    if (failed(scalar) || (*scalar).getType() != result.getElementType())
      return failure();
    return Value(builder.create<SplatOp>(location, result, *scalar));
  }
  Value zero;
  if (isa<FloatType>(result.getElementType()))
    zero = builder.create<arith::ConstantOp>(
        location, result.getElementType(),
        builder.getFloatAttr(result.getElementType(), 0.0));
  else if (auto integer = dyn_cast<IntegerType>(result.getElementType()))
    if (integer.isSignless()) {
      zero = builder.create<arith::ConstantOp>(
          location, integer, builder.getIntegerAttr(integer, 0));
    } else {
      auto signless = IntegerType::get(builder.getContext(), integer.getWidth());
      Value raw = builder.create<arith::ConstantOp>(
          location, signless, builder.getIntegerAttr(signless, 0));
      zero = builder.create<CastOp>(location, integer, raw);
    }
  else
    return failure();
  return Value(builder.create<SplatOp>(location, result, zero));
}

struct StorePath {
  SmallVector<CastOp> casts;
  StoreOp store;
};

bool collectStorePaths(Value value, SmallVector<CastOp> casts,
                       SmallVectorImpl<StorePath> &paths,
                       llvm::SmallPtrSetImpl<Operation *> &visited) {
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;
    if (auto cast = dyn_cast<CastOp>(user)) {
      SmallVector<CastOp> next(casts);
      next.push_back(cast);
      if (!collectStorePaths(cast.getResult(), std::move(next), paths, visited))
        return false;
      continue;
    }
    auto store = dyn_cast<StoreOp>(user);
    if (!store || store.getValue() != value || store.getCollision() != 0)
      return false;
    paths.push_back({std::move(casts), store});
  }
  return !paths.empty();
}

LoadOp matrixOperandLoad(Value value) {
  if (auto load = value.getDefiningOp<LoadOp>())
    return load;
  auto transpose = value.getDefiningOp<TransposeOp>();
  if (!transpose)
    return {};
  ArrayRef<int64_t> permutation = transpose.getPermutation();
  if (permutation.size() < 2)
    return {};
  for (unsigned axis = 0; axis + 2 < permutation.size(); ++axis)
    if (permutation[axis] != static_cast<int64_t>(axis))
      return {};
  unsigned penultimate = permutation.size() - 2;
  unsigned last = permutation.size() - 1;
  if (permutation[penultimate] != static_cast<int64_t>(last) ||
      permutation[last] != static_cast<int64_t>(penultimate))
    return {};
  return transpose.getValue().getDefiningOp<LoadOp>();
}

bool hasRangeContractForm(ContractOp contract) {
  auto lhsLoad = matrixOperandLoad(contract.getLhs());
  auto rhsLoad = matrixOperandLoad(contract.getRhs());
  if (!lhsLoad || !rhsLoad || contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1 ||
      !contract.getLhsBatchAxes().empty() ||
      !contract.getRhsBatchAxes().empty())
    return false;
  FragmentType lhs = contract.getLhs().getType();
  FragmentType rhs = contract.getRhs().getType();
  FailureOr<unsigned> lhsFree = uniqueFreeAxis(
      lhs, contract.getLhsReductionAxes(), contract.getLhsBatchAxes());
  FailureOr<unsigned> rhsFree = uniqueFreeAxis(
      rhs, contract.getRhsReductionAxes(), contract.getRhsBatchAxes());
  if (failed(lhsFree) || failed(rhsFree))
    return false;
  SmallVector<std::pair<LoadOp, AxisMapAttr>> axes;
  for (auto [load, fragment, axis] :
       {std::tuple<LoadOp, FragmentType, unsigned>{lhsLoad, lhs, *lhsFree},
        std::tuple<LoadOp, FragmentType, unsigned>{
            lhsLoad, lhs,
            static_cast<unsigned>(contract.getLhsReductionAxes().front())},
        std::tuple<LoadOp, FragmentType, unsigned>{
            rhsLoad, rhs,
            static_cast<unsigned>(contract.getRhsReductionAxes().front())},
        std::tuple<LoadOp, FragmentType, unsigned>{rhsLoad, rhs, *rhsFree}}) {
    FailureOr<AxisMapAttr> mapping = queryAxisMap(fragment, axis);
    if (failed(mapping))
      return false;
    axes.emplace_back(load, *mapping);
  }
  for (auto [load, mapping] : axes) {
    FailureOr<unsigned> coordinate = queryCoordinatePosition(
        load.getCoordinates(),
        PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis()});
    if (failed(coordinate))
      return false;
    Value value = load.getCoordinates()[*coordinate];
    if (!sourceRange(value) &&
        failed(producerRange(
            value, PhysicalSourceAxis{mapping.getSourceId(),
                                      mapping.getSourceAxis()})))
      return false;
  }
  SmallVector<StorePath> paths;
  llvm::SmallPtrSet<Operation *, 8> visited;
  return collectStorePaths(contract.getResult(), {}, paths, visited);
}

bool hasSelectedFreeAxes(ContractOp contract) {
  return llvm::all_of(contract.getResult().getType().getShape(),
                      [](Attribute extent) {
                        return isCompileTimeExtent(
                            cast<PhysicalExprAttr>(extent));
                      });
}

ParameterOp getOrCreateParameter(func::FuncOp kernel, StringRef name,
                                 ParameterRole role,
                                 ArrayRef<int64_t> candidates) {
  ParameterOp existing;
  kernel.walk([&](ParameterOp parameter) {
    if (parameter.getParameter().getName().getValue() == name)
      existing = parameter;
  });
  if (existing) {
    ParameterAttr schema = existing.getParameter();
    auto expectedCandidates =
        DenseI64ArrayAttr::get(kernel.getContext(), candidates);
    if (schema.getRole() != static_cast<uint32_t>(role) ||
        schema.getCandidates() != expectedCandidates) {
      existing.emitOpError(
          "physical parameter name is reused with a different role or candidate domain");
      return ParameterOp();
    }
    return existing;
  }
  OpBuilder builder(&kernel.getBody().front(), kernel.getBody().front().begin());
  auto schema = ParameterAttr::get(
      kernel.getContext(), builder.getStringAttr(name),
      static_cast<uint32_t>(role),
      DenseI64ArrayAttr::get(kernel.getContext(), candidates));
  return builder.create<ParameterOp>(kernel.getLoc(), builder.getIndexType(),
                                     schema);
}

FragmentType eraseFragmentAxis(FragmentType source, unsigned erasedAxis) {
  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  for (auto [axis, extent] : llvm::enumerate(source.getShape())) {
    if (axis == erasedAxis)
      continue;
    shape.push_back(extent);
    auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(
        source.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), mappings.size()));
  }
  return FragmentType::get(source.getContext(), source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           ArrayAttr::get(source.getContext(), mappings),
                           source.getValidity(), source.getOwner());
}

FragmentType retargetFragmentToCoordinateRanges(
    FragmentType source, ArrayRef<Value> coordinates) {
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  bool changed = false;
  for (auto [axis, attribute] : llvm::enumerate(source.getAxisMaps())) {
    auto mapping = cast<AxisMapAttr>(attribute);
    FailureOr<unsigned> coordinate = queryCoordinatePosition(
        coordinates,
        PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis()});
    if (failed(coordinate))
      continue;
    MakeRangeOp range = sourceRange(coordinates[*coordinate]);
    if (!range) {
      FailureOr<MakeRangeOp> root =
          producerRange(coordinates[*coordinate],
                        PhysicalSourceAxis{mapping.getSourceId(),
                                           mapping.getSourceAxis()});
      if (succeeded(root))
        range = *root;
    }
    if (!range)
      continue;
    auto rangeType = dyn_cast<FragmentType>(range.getResult().getType());
    if (!rangeType || rangeType.getShape().size() != 1)
      continue;
    shape[axis] = rangeType.getShape()[0];
    changed = true;
  }
  if (!changed)
    return source;
  return FragmentType::get(source.getContext(), source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           source.getAxisMaps(), source.getValidity(),
                           source.getOwner());
}

FragmentType fragmentElementType(FragmentType source, Type element) {
  return FragmentType::get(source.getContext(), element, source.getShape(),
                           source.getAxisMaps(), source.getValidity(),
                           source.getOwner());
}

SmallVector<int64_t> eraseAxis(ArrayRef<int64_t> axes, unsigned erasedAxis) {
  SmallVector<int64_t> result;
  result.reserve(axes.size());
  for (int64_t axis : axes) {
    if (axis == static_cast<int64_t>(erasedAxis))
      continue;
    result.push_back(axis > static_cast<int64_t>(erasedAxis) ? axis - 1 : axis);
  }
  return result;
}

FailureOr<Value> projectBroadcast(OpBuilder &builder, Location location,
                                  Value value, FragmentType target) {
  if (!value)
    return failure();
  if (value.getType() == target)
    return value;
  if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
    Type element = broadcast.getValue().getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    if (element != target.getElementType())
      return failure();
    return Value(
        builder.create<BroadcastOp>(location, target, broadcast.getValue()));
  }
  if (auto splat = value.getDefiningOp<SplatOp>())
    return Value(builder.create<SplatOp>(location, target, splat.getValue()));
  auto retarget = [&](Value operand) -> FailureOr<FragmentType> {
    auto fragment = dyn_cast<FragmentType>(operand.getType());
    if (!fragment)
      return failure();
    return fragmentElementType(target, fragment.getElementType());
  };
  if (auto binary = value.getDefiningOp<BinaryOp>()) {
    FailureOr<FragmentType> lhsType = retarget(binary.getLhs());
    FailureOr<FragmentType> rhsType = retarget(binary.getRhs());
    if (failed(lhsType) || failed(rhsType))
      return failure();
    FailureOr<Value> lhs =
        projectBroadcast(builder, location, binary.getLhs(), *lhsType);
    FailureOr<Value> rhs =
        projectBroadcast(builder, location, binary.getRhs(), *rhsType);
    if (failed(lhs) || failed(rhs))
      return failure();
    return Value(builder.create<BinaryOp>(location, target, *lhs, *rhs,
                                          binary.getOperatorKind()));
  }
  if (auto compare = value.getDefiningOp<CompareOp>()) {
    FailureOr<FragmentType> lhsType = retarget(compare.getLhs());
    FailureOr<FragmentType> rhsType = retarget(compare.getRhs());
    if (failed(lhsType) || failed(rhsType))
      return failure();
    FailureOr<Value> lhs =
        projectBroadcast(builder, location, compare.getLhs(), *lhsType);
    FailureOr<Value> rhs =
        projectBroadcast(builder, location, compare.getRhs(), *rhsType);
    if (failed(lhs) || failed(rhs))
      return failure();
    return Value(builder.create<CompareOp>(location, target, *lhs, *rhs,
                                           compare.getPredicate()));
  }
  if (auto select = value.getDefiningOp<SelectOp>()) {
    FailureOr<FragmentType> conditionType = retarget(select.getCondition());
    FailureOr<FragmentType> trueType = retarget(select.getTrueValue());
    FailureOr<FragmentType> falseType = retarget(select.getFalseValue());
    if (failed(conditionType) || failed(trueType) || failed(falseType))
      return failure();
    FailureOr<Value> condition = projectBroadcast(
        builder, location, select.getCondition(), *conditionType);
    FailureOr<Value> trueValue = projectBroadcast(
        builder, location, select.getTrueValue(), *trueType);
    FailureOr<Value> falseValue = projectBroadcast(
        builder, location, select.getFalseValue(), *falseType);
    if (failed(condition) || failed(trueValue) || failed(falseValue))
      return failure();
    return Value(builder.create<SelectOp>(location, target, *condition,
                                          *trueValue, *falseValue));
  }
  if (auto cast = value.getDefiningOp<CastOp>()) {
    FailureOr<FragmentType> sourceType = retarget(cast.getValue());
    if (failed(sourceType))
      return failure();
    FailureOr<Value> source =
        projectBroadcast(builder, location, cast.getValue(), *sourceType);
    if (failed(source))
      return failure();
    return Value(builder.create<CastOp>(location, target, *source));
  }
  return failure();
}

FailureOr<Value> projectPredicateForScalarAxis(
    OpBuilder &builder, Location location, Value value, MakeRangeOp root,
    Value coordinate, FragmentType target) {
  if (!value)
    return Value();
  if (!containsSource(
          value,
          PhysicalSourceAxis{root.getSourceId(), root.getSourceAxis()})) {
    if (value.getType() == target)
      return value;
    Type element = value.getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    if (element != target.getElementType()) {
      return failure();
    }
    return Value(builder.create<BroadcastOp>(location, target, value));
  }
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    return projectPredicateForScalarAxis(builder, location,
                                         broadcast.getValue(), root,
                                         coordinate, target);
  if (auto splat = value.getDefiningOp<SplatOp>())
    return projectPredicateForScalarAxis(builder, location, splat.getValue(),
                                         root, coordinate, target);
  if (auto conjunction = value.getDefiningOp<BinaryOp>()) {
    if (conjunction.getOperatorKind() != BinaryOperator::LogicalAnd)
      return failure();
    FailureOr<Value> lhs = projectPredicateForScalarAxis(
        builder, location, conjunction.getLhs(), root, coordinate, target);
    FailureOr<Value> rhs = projectPredicateForScalarAxis(
        builder, location, conjunction.getRhs(), root, coordinate, target);
    if (failed(lhs) || failed(rhs))
      return failure();
    return Value(builder.create<BinaryOp>(location, target, *lhs, *rhs,
                                          BinaryOperator::LogicalAnd));
  }
  if (auto comparison = value.getDefiningOp<CompareOp>()) {
    Value lhs = strippedBroadcast(comparison.getLhs());
    Value rhs = strippedBroadcast(comparison.getRhs());
    if (comparison.getPredicate() == ComparePredicate::Lt &&
        lhs == root.getResult()) {
      FailureOr<Value> end = resolveLogicalRangeEnd(
          root->getParentOfType<func::FuncOp>(), root);
      if (failed(end) || rhs != *end)
        return failure();
      Value scalar = builder.create<CompareOp>(
          location, builder.getI1Type(), coordinate, *end,
          comparison.getPredicate());
      return Value(builder.create<SplatOp>(location, target, scalar));
    }
  }
  return projectBroadcast(builder, location, value, target);
}

LogicalResult decomposeMultiReductionContract(ContractOp contract) {
  if (!contract->getBlock() || contract.getLhsReductionAxes().size() <= 1)
    return success();
  if (contract.getLhsReductionAxes().size() !=
          contract.getRhsReductionAxes().size() ||
      !contract.getLhsBatchAxes().empty() ||
      !contract.getRhsBatchAxes().empty())
    return contract.emitOpError(
        "multi-pair contraction decomposition requires paired reductions and no batch axes");
  auto lhsLoad = contract.getLhs().getDefiningOp<LoadOp>();
  auto rhsLoad = contract.getRhs().getDefiningOp<LoadOp>();
  if (!lhsLoad || !rhsLoad)
    return contract.emitOpError(
        "multi-pair contraction decomposition requires direct loads");
  SmallVector<int64_t> lhsReductions(contract.getLhsReductionAxes());
  SmallVector<int64_t> rhsReductions(contract.getRhsReductionAxes());
  SmallVector<int64_t> lhsBatch(contract.getLhsBatchAxes());
  SmallVector<int64_t> rhsBatch(contract.getRhsBatchAxes());
  SmallVector<Value> lhsCoordinates(lhsLoad.getCoordinates());
  SmallVector<Value> rhsCoordinates(rhsLoad.getCoordinates());
  FragmentType lhsType = contract.getLhs().getType();
  FragmentType rhsType = contract.getRhs().getType();
  Location location = contract.getLoc();
  bool failedBody = false;
  std::string failureReason;

  std::function<FailureOr<Value>(
      OpBuilder &, FragmentType, FragmentType, SmallVector<Value>,
      SmallVector<Value>, Value, Value, SmallVector<int64_t>,
      SmallVector<int64_t>, SmallVector<int64_t>, SmallVector<int64_t>, Value)>
      build;
  build = [&](OpBuilder &builder, FragmentType currentLhsType,
              FragmentType currentRhsType,
              SmallVector<Value> currentLhsCoordinates,
              SmallVector<Value> currentRhsCoordinates,
              Value currentLhsValid, Value currentRhsValid,
              SmallVector<int64_t> currentLhsReductions,
              SmallVector<int64_t> currentRhsReductions,
              SmallVector<int64_t> currentLhsBatch,
              SmallVector<int64_t> currentRhsBatch,
              Value accumulator) -> FailureOr<Value> {
    currentLhsType = retargetFragmentToCoordinateRanges(
        currentLhsType, currentLhsCoordinates);
    currentRhsType = retargetFragmentToCoordinateRanges(
        currentRhsType, currentRhsCoordinates);
    if (currentLhsReductions.size() == 1) {
      auto lhsPredicateType =
          fragmentElementType(currentLhsType, builder.getI1Type());
      auto rhsPredicateType =
          fragmentElementType(currentRhsType, builder.getI1Type());
      Value lhsValid;
      Value rhsValid;
      if (currentLhsValid) {
        FailureOr<Value> projected = projectBroadcast(
            builder, location, currentLhsValid, lhsPredicateType);
        if (failed(projected)) {
          failureReason =
              "lhs validity cannot be projected after erasing a reduction axis";
          return failure();
        }
        lhsValid = *projected;
      }
      if (currentRhsValid) {
        FailureOr<Value> projected = projectBroadcast(
            builder, location, currentRhsValid, rhsPredicateType);
        if (failed(projected)) {
          failureReason =
              "rhs validity cannot be projected after erasing a reduction axis";
          return failure();
        }
        rhsValid = *projected;
      }
      FailureOr<Value> lhsFill =
          lhsValid ? retargetFill(builder, location, lhsLoad.getFill(),
                                  currentLhsType)
                   : FailureOr<Value>(Value());
      FailureOr<Value> rhsFill =
          rhsValid ? retargetFill(builder, location, rhsLoad.getFill(),
                                  currentRhsType)
                   : FailureOr<Value>(Value());
      if (failed(lhsFill) || failed(rhsFill)) {
        failureReason =
            "invalid-value fill cannot be retargeted after erasing a reduction axis";
        return failure();
      }
      auto lhs = builder.create<LoadOp>(
          location, currentLhsType, lhsLoad.getResource(), currentLhsCoordinates,
          lhsValid, *lhsFill, lhsLoad.getSourceAxes());
      auto rhs = builder.create<LoadOp>(
          location, currentRhsType, rhsLoad.getResource(), currentRhsCoordinates,
          rhsValid, *rhsFill, rhsLoad.getSourceAxes());
      auto product = builder.create<ContractOp>(
          location, contract.getResult().getType(), lhs, rhs, accumulator,
          currentLhsReductions, currentRhsReductions, currentLhsBatch,
          currentRhsBatch);
      if (Attribute origin = contract->getAttr(originAttr))
        product->setAttr(originAttr, origin);
      return product.getResult();
    }

    unsigned lhsAxis = currentLhsReductions.front();
    unsigned rhsAxis = currentRhsReductions.front();
    FailureOr<AxisMapAttr> lhsMapping = queryAxisMap(currentLhsType, lhsAxis);
    FailureOr<AxisMapAttr> rhsMapping = queryAxisMap(currentRhsType, rhsAxis);
    if (failed(lhsMapping) || failed(rhsMapping)) {
      failureReason = "a reduction pair has no unique physical axis mapping";
      return failure();
    }
    PhysicalSourceAxis lhsSource{lhsMapping->getSourceId(),
                                 lhsMapping->getSourceAxis()};
    PhysicalSourceAxis rhsSource{rhsMapping->getSourceId(),
                                 rhsMapping->getSourceAxis()};
    if (lhsMapping->getDimensionId() <= 0 ||
        lhsMapping->getDimensionId() != rhsMapping->getDimensionId()) {
      failureReason = "a reduction pair does not share physical source provenance";
      return failure();
    }
    FailureOr<unsigned> lhsCoordinate =
        queryCoordinatePosition(currentLhsCoordinates, lhsSource);
    FailureOr<unsigned> rhsCoordinate =
        queryCoordinatePosition(currentRhsCoordinates, rhsSource);
    if (failed(lhsCoordinate) || failed(rhsCoordinate)) {
      failureReason = "a reduction pair has no unique access coordinate";
      return failure();
    }
    MakeRangeOp lhsRange = sourceRange(currentLhsCoordinates[*lhsCoordinate]);
    MakeRangeOp rhsRange = sourceRange(currentRhsCoordinates[*rhsCoordinate]);
    if (!lhsRange || !rhsRange) {
      failureReason = "a reduction pair coordinate has no physical range root";
      return failure();
    }
    FailureOr<int64_t> lhsDimension = queryRangeDimension(lhsRange);
    FailureOr<int64_t> rhsDimension = queryRangeDimension(rhsRange);
    if (failed(lhsDimension) || failed(rhsDimension) ||
        *lhsDimension != *rhsDimension) {
      failureReason = "a reduction pair range root provenance differs";
      return failure();
    }
    FailureOr<Value> lhsStep = scalarSource(lhsRange.getStep());
    FailureOr<Value> rhsStep = scalarSource(rhsRange.getStep());
    if (failed(lhsStep) || failed(rhsStep)) {
      failureReason = "a reduction pair has no scalar physical step";
      return failure();
    }
    if (*lhsStep != *rhsStep) {
      failureReason = "a reduction pair uses independently materialized steps";
      return failure();
    }
    Value stop = binary(builder, location, builder.getIndexType(),
                        lhsRange.getStart(), lhsRange.getExtent(),
                        BinaryOperator::Add);

    FragmentType nestedLhsType = eraseFragmentAxis(currentLhsType, lhsAxis);
    FragmentType nestedRhsType = eraseFragmentAxis(currentRhsType, rhsAxis);
    auto nestedLhsPredicateType =
        fragmentElementType(nestedLhsType, builder.getI1Type());
    auto nestedRhsPredicateType =
        fragmentElementType(nestedRhsType, builder.getI1Type());
    SmallVector<int64_t> nestedLhsReductions(
        currentLhsReductions.begin() + 1, currentLhsReductions.end());
    SmallVector<int64_t> nestedRhsReductions(
        currentRhsReductions.begin() + 1, currentRhsReductions.end());
    for (int64_t &axis : nestedLhsReductions)
      if (axis > static_cast<int64_t>(lhsAxis))
        --axis;
    for (int64_t &axis : nestedRhsReductions)
      if (axis > static_cast<int64_t>(rhsAxis))
        --axis;
    SmallVector<int64_t> nestedLhsBatch =
        eraseAxis(currentLhsBatch, lhsAxis);
    SmallVector<int64_t> nestedRhsBatch =
        eraseAxis(currentRhsBatch, rhsAxis);

    auto loop = builder.create<scf::ForOp>(
        location, lhsRange.getStart(), stop, *lhsStep, ValueRange{accumulator},
        [&](OpBuilder &nested, Location nestedLocation, Value coordinate,
            ValueRange carries) {
          SmallVector<Value> nestedLhsCoordinates(currentLhsCoordinates);
          SmallVector<Value> nestedRhsCoordinates(currentRhsCoordinates);
          nestedLhsCoordinates[*lhsCoordinate] = coordinate;
          nestedRhsCoordinates[*rhsCoordinate] = coordinate;
          FailureOr<Value> nestedLhsValid = projectPredicateForScalarAxis(
              nested, nestedLocation, currentLhsValid, lhsRange, coordinate,
              nestedLhsPredicateType);
          FailureOr<Value> nestedRhsValid = projectPredicateForScalarAxis(
              nested, nestedLocation, currentRhsValid, rhsRange, coordinate,
              nestedRhsPredicateType);
          if (failed(nestedLhsValid) || failed(nestedRhsValid)) {
            failureReason =
                "tail validity could not be projected while scalarizing a reduction pair";
            failedBody = true;
            return;
          }
          FailureOr<Value> product = build(
              nested, nestedLhsType, nestedRhsType,
              std::move(nestedLhsCoordinates), std::move(nestedRhsCoordinates),
              *nestedLhsValid, *nestedRhsValid, nestedLhsReductions,
              nestedRhsReductions, nestedLhsBatch, nestedRhsBatch,
              carries.front());
          if (failed(product)) {
            failedBody = true;
            if (failureReason.empty())
              failureReason = "nested reduction-pair decomposition failed";
            return;
          }
          nested.create<scf::YieldOp>(nestedLocation, *product);
        });
    if (failedBody) {
      loop.erase();
      return failure();
    }
    return loop.getResult(0);
  };

  OpBuilder builder(contract);
  FailureOr<Value> replacement = build(
      builder, lhsType, rhsType, std::move(lhsCoordinates),
      std::move(rhsCoordinates), lhsLoad.getValid(), rhsLoad.getValid(),
      std::move(lhsReductions), std::move(rhsReductions), std::move(lhsBatch),
      std::move(rhsBatch), contract.getAccumulator());
  if (failed(replacement))
    return contract.emitOpError(
               "multi-pair contraction could not be decomposed into provider-native contractions: ")
           << failureReason;
  contract.getResult().replaceAllUsesWith(*replacement);
  contract.erase();
  if (lhsLoad.getResult().use_empty())
    lhsLoad.erase();
  if (rhsLoad.getResult().use_empty())
    rhsLoad.erase();
  return success();
}

/// Block only the reduction traversal of a contraction whose free axes have
/// already been materialized by pointwise ownership.  This form is used when
/// several contractions feed one pure output expression: each term keeps its
/// own K loop while the shared M/N tile and final store remain in the ordinary
/// value graph.
LogicalResult realizeReductionTraversal(ContractOp contract,
                                        func::FuncOp kernel) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1 ||
      !contract.getLhsBatchAxes().empty() ||
      !contract.getRhsBatchAxes().empty())
    return contract.emitOpError(
        "reduction-only contraction blocking requires one reduction pair and no batch axes");

  FragmentType lhsType = contract.getLhs().getType();
  FragmentType rhsType = contract.getRhs().getType();
  FragmentType resultType = contract.getResult().getType();
  if (llvm::any_of(resultType.getShape(), [](Attribute extent) {
        return !isCompileTimeExtent(cast<PhysicalExprAttr>(extent));
      }))
    return contract.emitOpError(
        "reduction-only contraction blocking requires already-selected free-axis fragments");

  unsigned lhsReduction = contract.getLhsReductionAxes().front();
  unsigned rhsReduction = contract.getRhsReductionAxes().front();
  FailureOr<AxisMapAttr> lhsMap = queryAxisMap(lhsType, lhsReduction);
  FailureOr<AxisMapAttr> rhsMap = queryAxisMap(rhsType, rhsReduction);
  if (failed(lhsMap) || failed(rhsMap))
    return contract.emitOpError(
        "reduction-only contraction blocking lost paired reduction provenance");
  auto rangesFor = [&](Value value, AxisMapAttr mapping,
                       SmallVectorImpl<MakeRangeOp> &ranges) {
    if (failed(collectProducerRanges(
            value,
            PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis()},
            ranges)) ||
        ranges.empty())
      return failure();
    return llvm::all_of(ranges, [&](MakeRangeOp range) {
             return range.getSourceId() == mapping.getSourceId() &&
                    range.getSourceAxis() == mapping.getSourceAxis();
           })
               ? success()
               : failure();
  };
  SmallVector<MakeRangeOp> lhsRanges;
  SmallVector<MakeRangeOp> rhsRanges;
  if (failed(rangesFor(contract.getLhs(), *lhsMap, lhsRanges)) ||
      failed(rangesFor(contract.getRhs(), *rhsMap, rhsRanges)))
    return contract.emitOpError(
        "reduction-only contraction blocking requires explicit paired ranges");
  MakeRangeOp lhsRange = lhsRanges.front();
  MakeRangeOp rhsRange = rhsRanges.front();
  FailureOr<Value> logicalEnd = resolveLogicalRangeEnd(kernel, lhsRange);
  FailureOr<Value> lhsStep = scalarSource(lhsRange.getStep());
  FailureOr<Value> rhsStep = scalarSource(rhsRange.getStep());
  if (failed(logicalEnd) || failed(lhsStep) || failed(rhsStep) ||
      !isIntegerConstant(*lhsStep, 1) || !isIntegerConstant(*rhsStep, 1))
    return contract.emitOpError(
        "reduction-only contraction blocking requires a unit-step range with an exact logical end");

  std::string suffix =
      ("_" + Twine(lhsMap->getSourceId()) + "_" +
       Twine(rhsMap->getSourceId()))
          .str();
  ParameterOp blockK = getOrCreateParameter(
      kernel, "BLOCK_K" + suffix, ParameterRole::Reduction, {32, 64, 128});
  if (!blockK)
    return failure();
  FailureOr<int64_t> lhsDimension = queryRangeDimension(lhsRange);
  FailureOr<int64_t> rhsDimension = queryRangeDimension(rhsRange);
  if (succeeded(lhsDimension) && succeeded(rhsDimension) &&
      *lhsDimension == *rhsDimension)
    blockK->setAttr(
        dimensionAttr,
        IntegerAttr::get(IntegerType::get(kernel.getContext(), 64),
                         *lhsDimension));
  MLIRContext *context = kernel.getContext();
  PhysicalExprAttr unitK = parameterExpression(
      context, blockK.getParameter().getName().getValue());
  auto lhsIndexType = fragmentType(
      context, IndexType::get(context), {unitK}, {*lhsMap}, lhsType.getOwner());
  auto rhsIndexType = fragmentType(
      context, IndexType::get(context), {unitK}, {*rhsMap}, rhsType.getOwner());

  Location location = contract.getLoc();
  OpBuilder builder(contract);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  bool bodyFailed = false;
  auto loop = builder.create<scf::ForOp>(
      location, lhsRange.getStart(), *logicalEnd, blockK.getResult(),
      ValueRange{contract.getAccumulator()},
      [&](OpBuilder &nested, Location nestedLocation, Value kStart,
          ValueRange carries) {
        Value lhsK = nested.create<MakeRangeOp>(
            nestedLocation, lhsIndexType, kStart, blockK.getResult(), one,
            lhsMap->getSourceId(), lhsMap->getSourceAxis());
        inheritRangeAuthority(lhsK, lhsRange);
        Value rhsOffset = binary(
            nested, nestedLocation, nested.getIndexType(), kStart,
            lhsRange.getStart(), BinaryOperator::Subtract);
        Value rhsStart = binary(
            nested, nestedLocation, nested.getIndexType(),
            rhsRange.getStart(), rhsOffset, BinaryOperator::Add);
        Value rhsK = nested.create<MakeRangeOp>(
            nestedLocation, rhsIndexType, rhsStart, blockK.getResult(), one,
            rhsMap->getSourceId(), rhsMap->getSourceAxis());
        inheritRangeAuthority(rhsK, rhsRange);
        IRMapping lhsReplay;
        IRMapping rhsReplay;
        for (MakeRangeOp range : lhsRanges)
          lhsReplay.map(range.getResult(), lhsK);
        for (MakeRangeOp range : rhsRanges)
          rhsReplay.map(range.getResult(), rhsK);
        FailureOr<Value> lhs = replaySourceValue(
            nested, nestedLocation, contract.getLhs(),
            PhysicalSourceAxis{lhsMap->getSourceId(), lhsMap->getSourceAxis()},
            unitK, lhsRange, lhsK, lhsReplay);
        FailureOr<Value> rhs = replaySourceValue(
            nested, nestedLocation, contract.getRhs(),
            PhysicalSourceAxis{rhsMap->getSourceId(), rhsMap->getSourceAxis()},
            unitK, rhsRange, rhsK, rhsReplay);
        if (failed(lhs) || failed(rhs)) {
          bodyFailed = true;
          return;
        }
        auto product = nested.create<ContractOp>(
            nestedLocation, resultType, *lhs, *rhs, carries.front(),
            contract.getLhsReductionAxes(), contract.getRhsReductionAxes(),
            contract.getLhsBatchAxes(), contract.getRhsBatchAxes());
        if (Attribute origin = contract->getAttr(originAttr))
          product->setAttr(originAttr, origin);
        nested.create<scf::YieldOp>(nestedLocation, product.getResult());
      });
  if (bodyFailed) {
    loop.erase();
    return contract.emitOpError(
        "reduction-only contraction blocking could not replay its load graph");
  }

  contract.getResult().replaceAllUsesWith(loop.getResult(0));
  contract.erase();
  return success();
}

LogicalResult realizeContract(ContractOp contract, func::FuncOp kernel) {
  if (!contract->getBlock())
    return success();
  const bool required = requiresPhysicalRealization(contract);
  auto unhandled = [&](const Twine &reason) -> LogicalResult {
    return required ? contract.emitOpError()
                          << "cannot form a complete physical contraction: "
                          << reason
                    : success();
  };
  auto lhsLoad = matrixOperandLoad(contract.getLhs());
  auto rhsLoad = matrixOperandLoad(contract.getRhs());
  if (!lhsLoad || !rhsLoad || contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1 ||
      !contract.getLhsBatchAxes().empty() ||
      !contract.getRhsBatchAxes().empty())
    return unhandled("requires direct loads, one reduction pair, and no batch axes");

  auto lhsType = contract.getLhs().getType();
  auto rhsType = contract.getRhs().getType();
  FailureOr<unsigned> lhsFree = uniqueFreeAxis(
      lhsType, contract.getLhsReductionAxes(), contract.getLhsBatchAxes());
  FailureOr<unsigned> rhsFree = uniqueFreeAxis(
      rhsType, contract.getRhsReductionAxes(), contract.getRhsBatchAxes());
  if (failed(lhsFree) || failed(rhsFree))
    return unhandled("each operand must have one physical free axis");
  unsigned lhsReduction = contract.getLhsReductionAxes().front();
  unsigned rhsReduction = contract.getRhsReductionAxes().front();
  FailureOr<AxisMapAttr> rowMap = queryAxisMap(lhsType, *lhsFree);
  FailureOr<AxisMapAttr> lhsReductionMap = queryAxisMap(lhsType, lhsReduction);
  FailureOr<AxisMapAttr> rhsReductionMap = queryAxisMap(rhsType, rhsReduction);
  FailureOr<AxisMapAttr> columnMap = queryAxisMap(rhsType, *rhsFree);
  if (failed(rowMap) || failed(lhsReductionMap) || failed(rhsReductionMap) ||
      failed(columnMap) ||
      lhsReductionMap->getDimensionId() <= 0 ||
      lhsReductionMap->getDimensionId() !=
          rhsReductionMap->getDimensionId())
    return unhandled("paired reduction coordinates lack one shared provenance");

  FailureOr<unsigned> lhsRowCoordinate = queryCoordinatePosition(
      lhsLoad.getCoordinates(),
      PhysicalSourceAxis{rowMap->getSourceId(), rowMap->getSourceAxis()});
  FailureOr<unsigned> lhsReductionCoordinate = queryCoordinatePosition(
      lhsLoad.getCoordinates(),
      PhysicalSourceAxis{lhsReductionMap->getSourceId(),
                         lhsReductionMap->getSourceAxis()});
  FailureOr<unsigned> rhsReductionCoordinate = queryCoordinatePosition(
      rhsLoad.getCoordinates(),
      PhysicalSourceAxis{rhsReductionMap->getSourceId(),
                         rhsReductionMap->getSourceAxis()});
  FailureOr<unsigned> rhsColumnCoordinate = queryCoordinatePosition(
      rhsLoad.getCoordinates(),
      PhysicalSourceAxis{columnMap->getSourceId(), columnMap->getSourceAxis()});
  if (failed(lhsRowCoordinate) || failed(lhsReductionCoordinate) ||
      failed(rhsReductionCoordinate) || failed(rhsColumnCoordinate))
    return unhandled("load coordinates do not cover all free/reduction axes");
  Value originalLhsRowCoordinate = lhsLoad.getCoordinates()[*lhsRowCoordinate];
  auto rowRange = sourceRange(originalLhsRowCoordinate);
  const bool indirectRow = !rowRange;
  if (!rowRange) {
    FailureOr<MakeRangeOp> discovered =
        producerRange(originalLhsRowCoordinate,
                      PhysicalSourceAxis{rowMap->getSourceId(),
                                         rowMap->getSourceAxis()});
    if (succeeded(discovered))
      rowRange = *discovered;
  }
  auto lhsReductionRange =
      sourceRange(lhsLoad.getCoordinates()[*lhsReductionCoordinate]);
  auto rhsReductionRange =
      sourceRange(rhsLoad.getCoordinates()[*rhsReductionCoordinate]);
  auto columnRange = sourceRange(rhsLoad.getCoordinates()[*rhsColumnCoordinate]);
  FailureOr<int64_t> lhsReductionDimension =
      lhsReductionRange ? queryRangeDimension(lhsReductionRange)
                        : FailureOr<int64_t>(failure());
  FailureOr<int64_t> rhsReductionDimension =
      rhsReductionRange ? queryRangeDimension(rhsReductionRange)
                        : FailureOr<int64_t>(failure());
  if (!rowRange || !lhsReductionRange || !rhsReductionRange || !columnRange ||
      failed(lhsReductionDimension) || failed(rhsReductionDimension) ||
      *lhsReductionDimension != *rhsReductionDimension)
    return unhandled("physical coordinates are not explicit compatible ranges");
  FailureOr<Value> rowLogicalEnd = resolveLogicalRangeEnd(kernel, rowRange);
  FailureOr<Value> reductionLogicalEnd =
      resolveLogicalRangeEnd(kernel, lhsReductionRange);
  FailureOr<Value> columnLogicalEnd =
      resolveLogicalRangeEnd(kernel, columnRange);
  if (failed(rowLogicalEnd) || failed(reductionLogicalEnd) ||
      failed(columnLogicalEnd))
    return unhandled("physical ranges have ambiguous logical tail bounds");
  auto rowBound = rowRange.getStart().getDefiningOp<RangeBoundOp>();
  auto rowSourceRange =
      rowBound ? rowBound.getRange().getDefiningOp<RangeOp>() : RangeOp();
  const bool runtimeRowTraversal =
      indirectRow || rowRange->hasAttr(sourceSubregionAttr) ||
      (rowSourceRange && rowSourceRange->hasAttr(sourceSubregionAttr));

  if (!isUnitStepRange(rowRange) ||
      !isUnitStepRange(lhsReductionRange) ||
      !isUnitStepRange(columnRange))
    return unhandled("blocking currently requires unit-step source ranges");

  SmallVector<StorePath> paths;
  llvm::SmallPtrSet<Operation *, 8> visited;
  if (!collectStorePaths(contract.getResult(), {}, paths, visited))
    return unhandled("result does not have a complete unique-store path");
  SmallVector<std::pair<MakeRangeOp, Value>> lhsTailRanges = {
      {rowRange, *rowLogicalEnd},
      {lhsReductionRange, *reductionLogicalEnd},
  };
  SmallVector<std::pair<MakeRangeOp, Value>> outputTailRanges = {
      {rowRange, *rowLogicalEnd},
      {columnRange, *columnLogicalEnd},
  };
  SmallVector<std::pair<MakeRangeOp, Value>> rhsTailRanges = {
      {rhsReductionRange, *reductionLogicalEnd},
      {columnRange, *columnLogicalEnd},
  };
  if (lhsLoad.getFill() && !isZeroScalar(lhsLoad.getFill()))
    return unhandled("lhs invalid fill is not the contraction zero");
  if (rhsLoad.getFill() && !isZeroScalar(rhsLoad.getFill()))
    return unhandled("rhs invalid fill is not the contraction zero");
  FailureOr<Value> initialAccumulator = scalarSource(contract.getAccumulator());
  if (failed(initialAccumulator) ||
      (*initialAccumulator).getType() !=
          contract.getResult().getType().getElementType())
    return unhandled("accumulator is not an explicit scalarizable value");

  SmallVector<AssumeInBoundsOp> rowAssumptions;
  if (indirectRow)
    kernel.walk([&](AssumeInBoundsOp assumption) {
      if (!containsSource(
              assumption.getIndex(),
              PhysicalSourceAxis{rowMap->getSourceId(),
                                 rowMap->getSourceAxis()}))
        return;
      FailureOr<MakeRangeOp> root =
          producerRange(assumption.getIndex(),
                        PhysicalSourceAxis{rowMap->getSourceId(),
                                           rowMap->getSourceAxis()});
      if (succeeded(root) && sameLogicalRange(*root, rowRange))
        rowAssumptions.push_back(assumption);
    });

  DelinearizeOp mapping;
  for (Operation &operation : *contract->getBlock()) {
    auto candidate = dyn_cast<DelinearizeOp>(operation);
    if (!candidate || !candidate.getLinear().getDefiningOp<ProgramIdOp>())
      continue;
    if (candidate->isBeforeInBlock(contract))
      mapping = candidate;
  }
  if (!mapping)
    return unhandled("contract is not dominated by the current program mapping");
  auto programSpace = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  auto segmentOffset =
      mapping->getAttrOfType<PhysicalExprAttr>(segmentOffsetAttr);
  auto segmentLength =
      mapping->getAttrOfType<PhysicalExprAttr>(segmentLengthAttr);
  if (!programSpace || programSpace.size() != 1 || !segmentOffset ||
      !segmentLength ||
      segmentOffset.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Constant) ||
      segmentOffset.getValue() != 0 || programSpace[0] != segmentLength) {
    std::string details;
    llvm::raw_string_ostream stream(details);
    stream << "current rule requires one full-program execution segment"
           << "; program_space=" << programSpace
           << ", segment_offset=" << segmentOffset
           << ", segment_length=" << segmentLength;
    return unhandled(stream.str());
  }

  unsigned rowResourceAxis = lhsLoad.getSourceAxes()[*lhsRowCoordinate];
  unsigned columnResourceAxis = rhsLoad.getSourceAxes()[*rhsColumnCoordinate];
  MLIRContext *context = kernel.getContext();
  Location location = contract.getLoc();
  std::string suffix =
      ("_" + Twine(rowMap->getSourceId()) + "_" +
       Twine(columnMap->getSourceId()))
          .str();
  ParameterOp blockM = getOrCreateParameter(
      kernel, "BLOCK_M" + suffix, ParameterRole::OwnershipM,
      {32, 64, 128, 256});
  ParameterOp blockN = getOrCreateParameter(
      kernel, "BLOCK_N" + suffix, ParameterRole::OwnershipN,
      {32, 64, 128, 256});
  ParameterOp blockK = getOrCreateParameter(
      kernel, "BLOCK_K" + suffix, ParameterRole::Reduction, {32, 64, 128});
  if (!blockM || !blockN || !blockK)
    return failure();
  for (auto [parameter, range] :
       {std::pair<ParameterOp, MakeRangeOp>{blockM, rowRange},
        std::pair<ParameterOp, MakeRangeOp>{blockN, columnRange},
        std::pair<ParameterOp, MakeRangeOp>{blockK, lhsReductionRange}})
    if (FailureOr<int64_t> dimension = queryRangeDimension(range);
        succeeded(dimension))
      parameter->setAttr(
          dimensionAttr,
          IntegerAttr::get(IntegerType::get(kernel.getContext(), 64),
                           *dimension));
  ParameterOp rowWorkers;
  if (runtimeRowTraversal)
    rowWorkers = getOrCreateParameter(
        kernel, "ROW_WORKERS" + suffix, ParameterRole::TraversalWorkers,
        {1, 2, 4, 8});
  if (runtimeRowTraversal && !rowWorkers)
    return failure();
  PhysicalExprAttr unitM =
      parameterExpression(context, blockM.getParameter().getName().getValue());
  PhysicalExprAttr unitN =
      parameterExpression(context, blockN.getParameter().getName().getValue());
  PhysicalExprAttr unitK =
      parameterExpression(context, blockK.getParameter().getName().getValue());
  PhysicalExprAttr unitRowWorkers;
  if (rowWorkers)
    unitRowWorkers = parameterExpression(
        context, rowWorkers.getParameter().getName().getValue());

  OpBuilder mapBuilder(mapping);
  Value rowExtent = mapBuilder.create<DimOp>(
      location, mapBuilder.getIndexType(), lhsLoad.getResource(), rowResourceAxis);
  Value columnExtent = mapBuilder.create<DimOp>(
      location, mapBuilder.getIndexType(), rhsLoad.getResource(),
      columnResourceAxis);
  auto ceilDiv = [&](Value extent, Value divisor) {
    Value one = mapBuilder.create<arith::ConstantIndexOp>(location, 1);
    Value adjusted = binary(
        mapBuilder, location, mapBuilder.getIndexType(), extent,
        binary(mapBuilder, location, mapBuilder.getIndexType(), divisor, one,
               BinaryOperator::Subtract),
        BinaryOperator::Add);
    return binary(mapBuilder, location, mapBuilder.getIndexType(), adjusted,
                  divisor, BinaryOperator::FloorDivide);
  };
  Value rowTiles = ceilDiv(rowExtent, blockM.getResult());
  Value columnTiles = ceilDiv(columnExtent, blockN.getResult());
  SmallVector<Value> mappingExtents(mapping.getExtents());
  if (runtimeRowTraversal)
    mappingExtents.push_back(rowWorkers.getResult());
  else
    mappingExtents.push_back(rowTiles);
  mappingExtents.push_back(columnTiles);
  SmallVector<Attribute> launchExtents(mapping.getLaunchExtents().begin(),
                                       mapping.getLaunchExtents().end());
  auto rowExpression = cast<PhysicalExprAttr>(
      cast<ViewType>(lhsLoad.getResource().getType())
          .getLayout()
          .getExtents()[rowResourceAxis]);
  auto columnExpression = cast<PhysicalExprAttr>(
      cast<ViewType>(rhsLoad.getResource().getType())
          .getLayout()
          .getExtents()[columnResourceAxis]);
  if (runtimeRowTraversal)
    launchExtents.push_back(unitRowWorkers);
  else
    launchExtents.push_back(binaryExpression(context, PhysicalExprKind::CeilDiv,
                                              rowExpression, unitM));
  launchExtents.push_back(binaryExpression(context, PhysicalExprKind::CeilDiv,
                                            columnExpression, unitN));
  SmallVector<Type> mappingTypes(mapping.getResultTypes());
  mappingTypes.push_back(mapBuilder.getIndexType());
  mappingTypes.push_back(mapBuilder.getIndexType());
  auto expandedMapping = mapBuilder.create<DelinearizeOp>(
      location, mappingTypes, mapping.getLinear(), mappingExtents,
      mapBuilder.getArrayAttr(launchExtents));
  SmallVector<int64_t> coordinateRoles(expandedMapping.getNumResults(), -1);
  if (auto existing =
          mapping->getAttrOfType<DenseI64ArrayAttr>(coordinateRolesAttr))
    if (existing.size() == mapping.getNumResults())
      llvm::copy(existing.asArrayRef(), coordinateRoles.begin());
  coordinateRoles[mapping.getNumResults()] = static_cast<int64_t>(
      indirectRow ? CoordinateRole::IndirectTraversal
                  : runtimeRowTraversal ? CoordinateRole::TraversalWorker
                                        : CoordinateRole::ContractionM);
  coordinateRoles[mapping.getNumResults() + 1] =
      static_cast<int64_t>(CoordinateRole::ContractionN);
  expandedMapping->setAttr(
      coordinateRolesAttr,
      DenseI64ArrayAttr::get(context, coordinateRoles));
  for (StringRef attribute : {executionGroupAttr, segmentOffsetAttr})
    if (Attribute value = mapping->getAttr(attribute))
      expandedMapping->setAttr(attribute, value);
  segmentLength = cast<PhysicalExprAttr>(launchExtents.front());
  for (Attribute extent : llvm::drop_begin(launchExtents))
    segmentLength = binaryExpression(context, PhysicalExprKind::Multiply,
                                     segmentLength,
                                     cast<PhysicalExprAttr>(extent));
  expandedMapping->setAttr(segmentLengthAttr, segmentLength);
  for (auto [oldCoordinate, newCoordinate] : llvm::zip(
           mapping.getCoordinates(),
           expandedMapping.getCoordinates().take_front(mapping.getNumResults())))
    oldCoordinate.replaceAllUsesWith(newCoordinate);
  Value rowTile;
  Value rowWorker;
  unsigned physicalAxis = mapping.getNumResults();
  if (runtimeRowTraversal)
    rowWorker = expandedMapping.getCoordinates()[physicalAxis++];
  else
    rowTile = expandedMapping.getCoordinates()[physicalAxis++];
  Value columnTile = expandedMapping.getCoordinates()[physicalAxis];
  mapping.erase();
  kernel->setAttr(programSpaceAttr,
                  ArrayAttr::get(context, {segmentLength}));
  kernel->setAttr(gridRankAttr,
                  IntegerAttr::get(IntegerType::get(context, 64), 1));

  OpBuilder builder(contract);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value rowStop = *rowLogicalEnd;
  Value columnStart = binary(
      builder, location, builder.getIndexType(), columnRange.getStart(),
      binary(builder, location, builder.getIndexType(), columnTile,
             blockN.getResult(), BinaryOperator::Multiply),
      BinaryOperator::Add);
  Value columnStop = *columnLogicalEnd;
  Value reductionStop = *reductionLogicalEnd;

  FragmentType rowIndexType = fragmentType(
      context, builder.getIndexType(), {unitM}, {*rowMap}, lhsType.getOwner());
  FragmentType columnIndexType = fragmentType(
      context, builder.getIndexType(), {unitN}, {*columnMap}, rhsType.getOwner());
  FragmentType reductionIndexType = fragmentType(
      context, builder.getIndexType(), {unitK}, {*lhsReductionMap},
      lhsType.getOwner());
  FragmentType rowPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM}, {*rowMap}, lhsType.getOwner());
  FragmentType columnPredicateType = fragmentType(
      context, builder.getI1Type(), {unitN}, {*columnMap}, rhsType.getOwner());
  FragmentType reductionPredicateType = fragmentType(
      context, builder.getI1Type(), {unitK}, {*lhsReductionMap},
      lhsType.getOwner());
  FragmentType blockedLhsType = fragmentType(
      context, lhsType.getElementType(), {unitM, unitK},
      {*rowMap, *lhsReductionMap}, lhsType.getOwner());
  FragmentType blockedRhsType = fragmentType(
      context, rhsType.getElementType(), {unitK, unitN},
      {*rhsReductionMap, *columnMap}, rhsType.getOwner());
  FragmentType blockedResultType = fragmentType(
      context, contract.getResult().getType().getElementType(), {unitM, unitN},
      {*rowMap, *columnMap}, contract.getResult().getType().getOwner());
  FragmentType lhsPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM, unitK},
      {*rowMap, *lhsReductionMap}, lhsType.getOwner());
  FragmentType rhsPredicateType = fragmentType(
      context, builder.getI1Type(), {unitK, unitN},
      {*rhsReductionMap, *columnMap}, rhsType.getOwner());
  FragmentType outputPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM, unitN},
      {*rowMap, *columnMap}, contract.getResult().getType().getOwner());

  Value columns = builder.create<MakeRangeOp>(
      location, columnIndexType, columnStart, blockN.getResult(), one,
      columnMap->getSourceId(), columnMap->getSourceAxis());
  inheritRangeAuthority(columns, columnRange);
  Value columnEnd = broadcast(builder, location, columnIndexType, columnStop);
  Value columnValid =
      compare(builder, location, columnPredicateType, columns, columnEnd,
              ComparePredicate::Lt);
  auto emitRowBlock = [&](OpBuilder &rowBuilder,
                          Value rowStart) -> LogicalResult {
    Value rows = rowBuilder.create<MakeRangeOp>(
        location, rowIndexType, rowStart, blockM.getResult(), one,
        rowMap->getSourceId(), rowMap->getSourceAxis());
    inheritRangeAuthority(rows, rowRange);
    Value rowEnd = broadcast(rowBuilder, location, rowIndexType, rowStop);
    Value rowValid =
        compare(rowBuilder, location, rowPredicateType, rows, rowEnd,
                ComparePredicate::Lt);
    Value blockedLhsRowCoordinate = rows;
    SmallVector<SmallVector<Value>> replayedStoreCoordinates;
    if (indirectRow) {
      IRMapping replay;
      replay.map(rowRange.getResult(), rows);
      FailureOr<Value> rowCoordinate = replaySourceValue(
          rowBuilder, location, originalLhsRowCoordinate,
          PhysicalSourceAxis{rowMap->getSourceId(), rowMap->getSourceAxis()},
          unitM, rowRange, rows, replay);
      if (failed(rowCoordinate))
        return contract.emitOpError(
            "blocked contraction could not replay its row coordinate graph");
      blockedLhsRowCoordinate = *rowCoordinate;
      for (AssumeInBoundsOp assumption : rowAssumptions) {
        FailureOr<Value> index = replaySourceValue(
            rowBuilder, location, assumption.getIndex(),
            PhysicalSourceAxis{rowMap->getSourceId(), rowMap->getSourceAxis()},
            unitM, rowRange, rows, replay);
        if (failed(index))
          return assumption.emitOpError(
              "blocked contraction could not replay an in-bounds assertion");
        auto replacement = rowBuilder.create<AssumeInBoundsOp>(
            location, *index, assumption.getResource(), assumption.getAxis());
        if (Attribute origin = assumption->getAttr(originAttr))
          replacement->setAttr(originAttr, origin);
      }
      for (StorePath &path : paths) {
        SmallVector<Value> coordinates;
        for (Value coordinate : path.store.getCoordinates()) {
          FailureOr<Value> replayed = replaySourceValue(
              rowBuilder, location, coordinate,
              PhysicalSourceAxis{rowMap->getSourceId(),
                                 rowMap->getSourceAxis()},
              unitM, rowRange, rows, replay);
          if (failed(replayed))
            return path.store.emitOpError(
                "blocked contraction could not replay an output coordinate graph");
          coordinates.push_back(*replayed);
        }
        replayedStoreCoordinates.push_back(std::move(coordinates));
      }
    }
    Value accumulator = rowBuilder.create<SplatOp>(
        location, blockedResultType, *initialAccumulator);

    std::string loopBodyFailure;
    auto loop = rowBuilder.create<scf::ForOp>(
        location, lhsReductionRange.getStart(), reductionStop,
        blockK.getResult(), ValueRange{accumulator},
        [&](OpBuilder &nested, Location nestedLocation, Value kStart,
            ValueRange carries) {
          Value reductions = nested.create<MakeRangeOp>(
              nestedLocation, reductionIndexType, kStart, blockK.getResult(), one,
              lhsReductionMap->getSourceId(), lhsReductionMap->getSourceAxis());
          inheritRangeAuthority(reductions, lhsReductionRange);
          Value reductionEnd =
              broadcast(nested, nestedLocation, reductionIndexType, reductionStop);
          Value reductionValid = compare(nested, nestedLocation,
                                         reductionPredicateType, reductions,
                                         reductionEnd, ComparePredicate::Lt);
          Value lhsRows =
              broadcast(nested, nestedLocation, lhsPredicateType, rowValid);
          Value lhsReductions = broadcast(nested, nestedLocation,
                                          lhsPredicateType, reductionValid);
          Value lhsValid = binary(nested, nestedLocation, lhsPredicateType,
                                  lhsRows, lhsReductions,
                                  BinaryOperator::LogicalAnd);
          Value rhsReductions = broadcast(nested, nestedLocation,
                                          rhsPredicateType, reductionValid);
          Value rhsColumns =
              broadcast(nested, nestedLocation, rhsPredicateType, columnValid);
          Value rhsValid = binary(nested, nestedLocation, rhsPredicateType,
                                  rhsReductions, rhsColumns,
                                  BinaryOperator::LogicalAnd);
          if (lhsLoad.getValid() && !indirectRow &&
              !isTailPredicate(lhsLoad.getValid(), lhsTailRanges)) {
            FailureOr<Value> original = retargetPredicate(
                nested, nestedLocation, lhsLoad.getValid(), lhsPredicateType);
            if (failed(original)) {
              loopBodyFailure = "lhs residual validity could not be retargeted";
              return;
            }
            lhsValid = binary(nested, nestedLocation, lhsPredicateType, lhsValid,
                              *original, BinaryOperator::LogicalAnd);
          }
          if (rhsLoad.getValid() &&
              !isTailPredicate(rhsLoad.getValid(), rhsTailRanges)) {
            FailureOr<Value> original = retargetPredicate(
                nested, nestedLocation, rhsLoad.getValid(), rhsPredicateType);
            if (failed(original)) {
              loopBodyFailure = "rhs residual validity could not be retargeted";
              return;
            }
            rhsValid = binary(nested, nestedLocation, rhsPredicateType, rhsValid,
                              *original, BinaryOperator::LogicalAnd);
          }
          SmallVector<Value> lhsCoordinates(lhsLoad.getCoordinates());
          lhsCoordinates[*lhsRowCoordinate] = blockedLhsRowCoordinate;
          lhsCoordinates[*lhsReductionCoordinate] = reductions;
          SmallVector<Value> rhsCoordinates(rhsLoad.getCoordinates());
          rhsCoordinates[*rhsReductionCoordinate] = reductions;
          rhsCoordinates[*rhsColumnCoordinate] = columns;
          FailureOr<Value> lhsFill = retargetFill(
              nested, nestedLocation, lhsLoad.getFill(), blockedLhsType);
          FailureOr<Value> rhsFill = retargetFill(
              nested, nestedLocation, rhsLoad.getFill(), blockedRhsType);
          if (failed(lhsFill) || failed(rhsFill)) {
            loopBodyFailure = failed(lhsFill) ? "lhs fill could not be retargeted"
                                              : "rhs fill could not be retargeted";
            return;
          }
          Value lhs = nested.create<LoadOp>(
              nestedLocation, blockedLhsType, lhsLoad.getResource(),
              lhsCoordinates, lhsValid, *lhsFill, lhsLoad.getSourceAxes());
          Value rhs = nested.create<LoadOp>(
              nestedLocation, blockedRhsType, rhsLoad.getResource(),
              rhsCoordinates, rhsValid, *rhsFill, rhsLoad.getSourceAxes());
          Value product = nested.create<ContractOp>(
              nestedLocation, blockedResultType, lhs, rhs, carries.front(),
              ArrayRef<int64_t>{1}, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{},
              ArrayRef<int64_t>{});
          nested.create<scf::YieldOp>(nestedLocation, product);
        });
    if (!loopBodyFailure.empty()) {
      loop.erase();
      return contract.emitOpError(
                 "blocked contraction could not materialize its loop body: ")
             << loopBodyFailure;
    }

    Value outputRows =
        broadcast(rowBuilder, location, outputPredicateType, rowValid);
    Value outputColumns =
        broadcast(rowBuilder, location, outputPredicateType, columnValid);
    Value outputValid = binary(rowBuilder, location, outputPredicateType,
                               outputRows, outputColumns,
                               BinaryOperator::LogicalAnd);
    for (auto [pathIndex, path] : llvm::enumerate(paths)) {
      Value output = loop.getResult(0);
      for (CastOp conversion : path.casts) {
        auto original = llvm::cast<FragmentType>(conversion.getResult().getType());
        FragmentType converted = fragmentType(
            context, original.getElementType(), {unitM, unitN},
            {*rowMap, *columnMap}, original.getOwner());
        output = rowBuilder.create<CastOp>(location, converted, output);
      }
      SmallVector<Value> coordinates =
          indirectRow ? replayedStoreCoordinates[pathIndex]
                      : SmallVector<Value>(path.store.getCoordinates());
      FailureOr<unsigned> storeColumn = queryCoordinatePosition(
          path.store.getCoordinates(),
          PhysicalSourceAxis{columnMap->getSourceId(),
                             columnMap->getSourceAxis()});
      FailureOr<unsigned> storeRow;
      if (!indirectRow)
        storeRow = queryCoordinatePosition(
            path.store.getCoordinates(),
            PhysicalSourceAxis{rowMap->getSourceId(), rowMap->getSourceAxis()});
      if ((!indirectRow && failed(storeRow)) || failed(storeColumn))
        return path.store.emitOpError(
            "blocked contract output lost its logical source coordinates");
      if (!indirectRow)
        coordinates[*storeRow] = rows;
      coordinates[*storeColumn] = columns;
      Value valid = outputValid;
      if (path.store.getValid() && !indirectRow &&
          !isTailPredicate(path.store.getValid(), outputTailRanges)) {
        FailureOr<Value> original = retargetPredicate(
            rowBuilder, location, path.store.getValid(), outputPredicateType);
        if (failed(original))
          return path.store.emitOpError(
              "blocked contract output has non-scalar residual validity");
        valid = binary(rowBuilder, location, outputPredicateType, valid,
                       *original, BinaryOperator::LogicalAnd);
      }
      auto replacement = rowBuilder.create<StoreOp>(
          location, path.store.getResource(), coordinates, output, valid,
          path.store.getSourceAxes(), path.store.getCollision());
      if (Attribute origin = path.store->getAttr(originAttr))
        replacement->setAttr(originAttr, origin);
    }
    return success();
  };

  if (runtimeRowTraversal) {
    bool rowBodyFailed = false;
    Value rowStart = binary(
        builder, location, builder.getIndexType(), rowRange.getStart(),
        binary(builder, location, builder.getIndexType(), rowWorker,
               blockM.getResult(), BinaryOperator::Multiply),
        BinaryOperator::Add);
    Value rowStep = binary(builder, location, builder.getIndexType(),
                           blockM.getResult(), rowWorkers.getResult(),
                           BinaryOperator::Multiply);
    auto rowLoop = builder.create<scf::ForOp>(
        location, rowStart, rowStop, rowStep, ValueRange{},
        [&](OpBuilder &nested, Location nestedLocation, Value rowStart,
            ValueRange) {
          if (failed(emitRowBlock(nested, rowStart))) {
            rowBodyFailed = true;
            return;
          }
          nested.create<scf::YieldOp>(nestedLocation);
        });
    if (rowBodyFailed) {
      rowLoop.erase();
      return failure();
    }
  } else {
    Value rowStart = binary(
        builder, location, builder.getIndexType(), rowRange.getStart(),
        binary(builder, location, builder.getIndexType(), rowTile,
               blockM.getResult(), BinaryOperator::Multiply),
        BinaryOperator::Add);
    if (failed(emitRowBlock(builder, rowStart)))
      return failure();
  }

  for (StorePath &path : paths)
    path.store.erase();
  for (StorePath &path : paths)
    for (CastOp conversion : llvm::reverse(path.casts))
      if (conversion->getBlock() && conversion.getResult().use_empty())
        conversion.erase();
  if (contract->getBlock() && contract.getResult().use_empty())
    contract.erase();
  if (lhsLoad->getBlock() && lhsLoad.getResult().use_empty())
    lhsLoad.erase();
  if (rhsLoad->getBlock() && rhsLoad.getResult().use_empty())
    rhsLoad.erase();
  for (AssumeInBoundsOp assumption : rowAssumptions)
    if (assumption->getBlock())
      assumption.erase();
  return success();
}

LogicalResult realizeScaledContract(ScaledContractOp contract,
                                    func::FuncOp kernel) {
  if (!contract->getBlock())
    return success();
  if (!requiresPhysicalRealization(contract))
    return success();
  auto lhsLoad = contract.getLhs().getDefiningOp<LoadOp>();
  auto lhsScaleLoad = contract.getLhsScale().getDefiningOp<LoadOp>();
  auto rhsLoad = contract.getRhs().getDefiningOp<LoadOp>();
  auto rhsScaleLoad = contract.getRhsScale().getDefiningOp<LoadOp>();
  auto lhsType = contract.getLhs().getType();
  auto lhsScaleType = contract.getLhsScale().getType();
  auto rhsType = contract.getRhs().getType();
  auto rhsScaleType = contract.getRhsScale().getType();
  auto resultType = contract.getResult().getType();
  auto reject = [&](const Twine &reason) -> LogicalResult {
    return contract.emitOpError()
           << "cannot form a complete block-scaled contraction: " << reason;
  };
  ArrayRef<int64_t> lhsReduction = contract.getLhsReductionAxes();
  ArrayRef<int64_t> rhsReduction = contract.getRhsReductionAxes();
  if (!lhsLoad || !lhsScaleLoad || !rhsLoad || !rhsScaleLoad ||
      lhsType.getShape().size() != 3 || lhsScaleType.getShape().size() != 2 ||
      rhsType.getShape().size() != 3 || rhsScaleType.getShape().size() != 2 ||
      resultType.getShape().size() != 2 || lhsReduction.size() != 2 ||
      rhsReduction.size() != 2 || !contract.getLhsBatchAxes().empty() ||
      !contract.getRhsBatchAxes().empty())
    return reject("requires direct rank-3 data/rank-2 scale loads, two reduction pairs, one free axis per operand, and no batch axes");
  if (contract.getLhsGroupSize() <= 0 ||
      contract.getLhsGroupSize() != contract.getRhsGroupSize())
    return reject("requires one equal positive scale-group size");
  FailureOr<unsigned> lhsFree = uniqueFreeAxis(
      lhsType, lhsReduction, contract.getLhsBatchAxes());
  FailureOr<unsigned> rhsFree = uniqueFreeAxis(
      rhsType, rhsReduction, contract.getRhsBatchAxes());
  if (failed(lhsFree) || failed(rhsFree))
    return reject("data operands do not have one unique free axis");
  std::optional<unsigned> innerPair;
  for (unsigned pair = 0; pair < lhsReduction.size(); ++pair) {
    auto lhsExtent = cast<PhysicalExprAttr>(
        lhsType.getShape()[lhsReduction[pair]]);
    auto rhsExtent = cast<PhysicalExprAttr>(
        rhsType.getShape()[rhsReduction[pair]]);
    bool groupExtent =
        lhsExtent.getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        rhsExtent == lhsExtent &&
        static_cast<uint64_t>(lhsExtent.getValue()) ==
            contract.getLhsGroupSize();
    if (!groupExtent)
      continue;
    if (innerPair)
      return reject("scale group relation is ambiguous across reduction pairs");
    innerPair = pair;
  }
  if (!innerPair)
    return reject("the innermost physical reduction extent must equal the scale group size");
  unsigned blockPair = *innerPair == 0 ? 1 : 0;
  unsigned lhsBlockAxis = lhsReduction[blockPair];
  unsigned lhsInnerAxis = lhsReduction[*innerPair];
  unsigned rhsBlockAxis = rhsReduction[blockPair];
  unsigned rhsInnerAxis = rhsReduction[*innerPair];
  auto innerExtent =
      cast<PhysicalExprAttr>(lhsType.getShape()[lhsInnerAxis]);

  FailureOr<AxisMapAttr> rowMap = queryAxisMap(lhsType, *lhsFree);
  FailureOr<AxisMapAttr> lhsBlockMap = queryAxisMap(lhsType, lhsBlockAxis);
  FailureOr<AxisMapAttr> lhsInnerMap = queryAxisMap(lhsType, lhsInnerAxis);
  FailureOr<AxisMapAttr> rhsBlockMap = queryAxisMap(rhsType, rhsBlockAxis);
  FailureOr<AxisMapAttr> rhsInnerMap = queryAxisMap(rhsType, rhsInnerAxis);
  FailureOr<AxisMapAttr> columnMap = queryAxisMap(rhsType, *rhsFree);
  auto scaleMapFor = [&](FragmentType scale,
                         AxisMapAttr data) -> FailureOr<AxisMapAttr> {
    PhysicalAxisProjection projection = queryFragmentAxis(
        scale, PhysicalSourceAxis{data.getSourceId(), data.getSourceAxis()});
    return projection.isExact() ? queryAxisMap(scale, projection.fragmentAxis)
                                : FailureOr<AxisMapAttr>(failure());
  };
  FailureOr<AxisMapAttr> lhsScaleRowMap =
      succeeded(rowMap) ? scaleMapFor(lhsScaleType, *rowMap)
                        : FailureOr<AxisMapAttr>(failure());
  FailureOr<AxisMapAttr> lhsScaleBlockMap =
      succeeded(lhsBlockMap) ? scaleMapFor(lhsScaleType, *lhsBlockMap)
                             : FailureOr<AxisMapAttr>(failure());
  FailureOr<AxisMapAttr> rhsScaleBlockMap =
      succeeded(rhsBlockMap) ? scaleMapFor(rhsScaleType, *rhsBlockMap)
                             : FailureOr<AxisMapAttr>(failure());
  FailureOr<AxisMapAttr> rhsScaleColumnMap =
      succeeded(columnMap) ? scaleMapFor(rhsScaleType, *columnMap)
                           : FailureOr<AxisMapAttr>(failure());
  if (failed(rowMap) || failed(lhsBlockMap) || failed(lhsInnerMap) ||
      failed(rhsBlockMap) || failed(rhsInnerMap) || failed(columnMap) ||
      failed(lhsScaleRowMap) || failed(lhsScaleBlockMap) ||
      failed(rhsScaleBlockMap) || failed(rhsScaleColumnMap))
    return reject("data and scale operands do not preserve the paired coordinate provenance");
  auto sameSource = [](AxisMapAttr lhs, AxisMapAttr rhs) {
    return PhysicalSourceAxis{lhs.getSourceId(), lhs.getSourceAxis()} ==
           PhysicalSourceAxis{rhs.getSourceId(), rhs.getSourceAxis()};
  };
  auto sameDimension = [](AxisMapAttr lhs, AxisMapAttr rhs) {
    return lhs.getDimensionId() > 0 &&
           lhs.getDimensionId() == rhs.getDimensionId();
  };
  if (!sameSource(*rowMap, *lhsScaleRowMap) ||
      !sameDimension(*lhsBlockMap, *rhsBlockMap) ||
      !sameSource(*lhsBlockMap, *lhsScaleBlockMap) ||
      !sameDimension(*lhsBlockMap, *rhsScaleBlockMap) ||
      !sameDimension(*lhsInnerMap, *rhsInnerMap) ||
      !sameSource(*columnMap, *rhsScaleColumnMap))
    return reject("data and scale operands do not preserve the paired coordinate provenance");

  auto rangeFor = [&](LoadOp load,
                      AxisMapAttr source) -> FailureOr<MakeRangeOp> {
    FailureOr<unsigned> coordinate = queryCoordinatePosition(
        load.getCoordinates(),
        PhysicalSourceAxis{source.getSourceId(), source.getSourceAxis()});
    if (failed(coordinate))
      return failure();
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact fact = analysis.sourceRanges(
        load.getCoordinates()[*coordinate],
        PhysicalSourceAxis{source.getSourceId(), source.getSourceAxis()});
    return queryExactLogicalRange(fact);
  };
  FailureOr<MakeRangeOp> rowRange = rangeFor(lhsLoad, *rowMap);
  FailureOr<MakeRangeOp> blockRange =
      rangeFor(lhsLoad, *lhsBlockMap);
  FailureOr<MakeRangeOp> innerRange =
      rangeFor(lhsLoad, *lhsInnerMap);
  FailureOr<MakeRangeOp> columnRange =
      rangeFor(rhsLoad, *columnMap);
  FailureOr<MakeRangeOp> lhsScaleRowRange =
      rangeFor(lhsScaleLoad, *lhsScaleRowMap);
  FailureOr<MakeRangeOp> lhsScaleBlockRange =
      rangeFor(lhsScaleLoad, *lhsScaleBlockMap);
  FailureOr<MakeRangeOp> rhsBlockRange =
      rangeFor(rhsLoad, *rhsBlockMap);
  FailureOr<MakeRangeOp> rhsInnerRange =
      rangeFor(rhsLoad, *rhsInnerMap);
  FailureOr<MakeRangeOp> rhsScaleBlockRange =
      rangeFor(rhsScaleLoad, *rhsScaleBlockMap);
  FailureOr<MakeRangeOp> rhsScaleColumnRange =
      rangeFor(rhsScaleLoad, *rhsScaleColumnMap);
  if (failed(rowRange) || failed(blockRange) || failed(innerRange) ||
      failed(columnRange) || failed(lhsScaleRowRange) ||
      failed(lhsScaleBlockRange) || failed(rhsBlockRange) ||
      failed(rhsInnerRange) || failed(rhsScaleBlockRange) ||
      failed(rhsScaleColumnRange))
    return reject("physical data coordinates are not explicit source ranges");
  if (!isUnitStepRange(*rowRange) || !isUnitStepRange(*blockRange) ||
      !isUnitStepRange(*innerRange) || !isUnitStepRange(*columnRange))
    return reject("blocking currently requires unit-step source ranges");
  SmallVector<StorePath> paths;
  llvm::SmallPtrSet<Operation *, 8> visited;
  if (!collectStorePaths(contract.getResult(), {}, paths, visited))
    return reject("result does not have a complete unique-store path");
  FailureOr<Value> rowLogicalEnd = resolveLogicalRangeEnd(kernel, *rowRange);
  FailureOr<Value> blockLogicalEnd =
      resolveLogicalRangeEnd(kernel, *blockRange);
  FailureOr<Value> innerLogicalEnd =
      resolveLogicalRangeEnd(kernel, *innerRange);
  FailureOr<Value> columnLogicalEnd =
      resolveLogicalRangeEnd(kernel, *columnRange);
  FailureOr<Value> lhsScaleRowLogicalEnd =
      resolveLogicalRangeEnd(kernel, *lhsScaleRowRange);
  FailureOr<Value> lhsScaleBlockLogicalEnd =
      resolveLogicalRangeEnd(kernel, *lhsScaleBlockRange);
  FailureOr<Value> rhsBlockLogicalEnd =
      resolveLogicalRangeEnd(kernel, *rhsBlockRange);
  FailureOr<Value> rhsInnerLogicalEnd =
      resolveLogicalRangeEnd(kernel, *rhsInnerRange);
  FailureOr<Value> rhsScaleBlockLogicalEnd =
      resolveLogicalRangeEnd(kernel, *rhsScaleBlockRange);
  FailureOr<Value> rhsScaleColumnLogicalEnd =
      resolveLogicalRangeEnd(kernel, *rhsScaleColumnRange);
  if (failed(rowLogicalEnd) || failed(blockLogicalEnd) ||
      failed(innerLogicalEnd) || failed(columnLogicalEnd) ||
      failed(lhsScaleRowLogicalEnd) || failed(lhsScaleBlockLogicalEnd) ||
      failed(rhsBlockLogicalEnd) || failed(rhsInnerLogicalEnd) ||
      failed(rhsScaleBlockLogicalEnd) || failed(rhsScaleColumnLogicalEnd))
    return reject("physical ranges have ambiguous logical tail bounds");
  SmallVector<std::pair<MakeRangeOp, Value>> lhsTailRanges = {
      {*rowRange, *rowLogicalEnd},
      {*blockRange, *blockLogicalEnd},
      {*innerRange, *innerLogicalEnd},
  };
  SmallVector<std::pair<MakeRangeOp, Value>> lhsScaleTailRanges = {
      {*lhsScaleRowRange, *lhsScaleRowLogicalEnd},
      {*lhsScaleBlockRange, *lhsScaleBlockLogicalEnd},
  };
  SmallVector<std::pair<MakeRangeOp, Value>> rhsTailRanges = {
      {*rhsBlockRange, *rhsBlockLogicalEnd},
      {*rhsInnerRange, *rhsInnerLogicalEnd},
      {*columnRange, *columnLogicalEnd},
  };
  SmallVector<std::pair<MakeRangeOp, Value>> rhsScaleTailRanges = {
      {*rhsScaleBlockRange, *rhsScaleBlockLogicalEnd},
      {*rhsScaleColumnRange, *rhsScaleColumnLogicalEnd},
  };
  SmallVector<std::pair<MakeRangeOp, Value>> outputTailRanges = {
      {*rowRange, *rowLogicalEnd},
      {*columnRange, *columnLogicalEnd},
  };
  auto validIsTail = [&](Value valid,
                         ArrayRef<std::pair<MakeRangeOp, Value>> ranges) {
    return !valid || succeeded(scalarSource(valid)) ||
           isTailPredicate(valid, ranges);
  };
  if (!validIsTail(lhsLoad.getValid(), lhsTailRanges) ||
      !validIsTail(lhsScaleLoad.getValid(), lhsScaleTailRanges) ||
      !validIsTail(rhsLoad.getValid(), rhsTailRanges) ||
      !validIsTail(rhsScaleLoad.getValid(), rhsScaleTailRanges))
    return reject("data/scale loads contain non-tail residual validity");
  for (StorePath &path : paths)
    if (!validIsTail(path.store.getValid(), outputTailRanges))
      return reject("result store contains non-tail residual validity");
  if ((lhsLoad.getFill() && !isZeroScalar(lhsLoad.getFill())) ||
      (rhsLoad.getFill() && !isZeroScalar(rhsLoad.getFill())))
    return reject("invalid data fill is not the contraction zero");
  FailureOr<Value> initialAccumulator = scalarSource(contract.getAccumulator());
  if (failed(initialAccumulator) ||
      (*initialAccumulator).getType() != resultType.getElementType())
    return reject("accumulator is not an explicit scalarizable value");

  DelinearizeOp mapping;
  for (Operation &operation : *contract->getBlock()) {
    auto candidate = dyn_cast<DelinearizeOp>(operation);
    if (candidate && candidate.getLinear().getDefiningOp<ProgramIdOp>() &&
        candidate->isBeforeInBlock(contract))
      mapping = candidate;
  }
  auto programSpace = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  auto segmentOffset =
      mapping ? mapping->getAttrOfType<PhysicalExprAttr>(segmentOffsetAttr)
              : PhysicalExprAttr();
  auto segmentLength =
      mapping ? mapping->getAttrOfType<PhysicalExprAttr>(segmentLengthAttr)
              : PhysicalExprAttr();
  if (!mapping || !programSpace || programSpace.size() != 1 || !segmentOffset ||
      !segmentLength ||
      segmentOffset.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Constant) ||
      segmentOffset.getValue() != 0 || programSpace[0] != segmentLength)
    return reject("requires one complete current program segment");

  auto coordinateFor = [](ValueRange coordinates, AxisMapAttr mapping) {
    return queryCoordinatePosition(
        coordinates,
        PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis()});
  };
  FailureOr<unsigned> lhsRowCoordinate =
      coordinateFor(lhsLoad.getCoordinates(), *rowMap);
  FailureOr<unsigned> lhsBlockCoordinate =
      coordinateFor(lhsLoad.getCoordinates(), *lhsBlockMap);
  FailureOr<unsigned> lhsInnerCoordinate =
      coordinateFor(lhsLoad.getCoordinates(), *lhsInnerMap);
  FailureOr<unsigned> lhsScaleRowCoordinate =
      coordinateFor(lhsScaleLoad.getCoordinates(), *lhsScaleRowMap);
  FailureOr<unsigned> lhsScaleBlockCoordinate =
      coordinateFor(lhsScaleLoad.getCoordinates(), *lhsScaleBlockMap);
  FailureOr<unsigned> rhsBlockCoordinate =
      coordinateFor(rhsLoad.getCoordinates(), *rhsBlockMap);
  FailureOr<unsigned> rhsInnerCoordinate =
      coordinateFor(rhsLoad.getCoordinates(), *rhsInnerMap);
  FailureOr<unsigned> rhsColumnCoordinate =
      coordinateFor(rhsLoad.getCoordinates(), *columnMap);
  FailureOr<unsigned> rhsScaleBlockCoordinate =
      coordinateFor(rhsScaleLoad.getCoordinates(), *rhsScaleBlockMap);
  FailureOr<unsigned> rhsScaleColumnCoordinate =
      coordinateFor(rhsScaleLoad.getCoordinates(), *rhsScaleColumnMap);
  if (failed(lhsRowCoordinate) || failed(lhsBlockCoordinate) ||
      failed(lhsInnerCoordinate) || failed(lhsScaleRowCoordinate) ||
      failed(lhsScaleBlockCoordinate) || failed(rhsBlockCoordinate) ||
      failed(rhsInnerCoordinate) || failed(rhsColumnCoordinate) ||
      failed(rhsScaleBlockCoordinate) || failed(rhsScaleColumnCoordinate))
    return reject("load coordinates do not cover all data and scale axes");

  unsigned rowResourceAxis = lhsLoad.getSourceAxes()[*lhsRowCoordinate];
  unsigned columnResourceAxis = rhsLoad.getSourceAxes()[*rhsColumnCoordinate];
  MLIRContext *context = kernel.getContext();
  Location location = contract.getLoc();
  std::string suffix =
      ("_" + Twine(rowMap->getSourceId()) + "_" +
       Twine(columnMap->getSourceId()))
          .str();
  ParameterOp blockM = getOrCreateParameter(
      kernel, "BLOCK_M" + suffix, ParameterRole::OwnershipM, {64, 128});
  ParameterOp blockN = getOrCreateParameter(
      kernel, "BLOCK_N" + suffix, ParameterRole::OwnershipN, {64, 128});
  ParameterOp blockK = getOrCreateParameter(
      kernel, "BLOCK_K_GROUPS" + suffix, ParameterRole::Reduction, {2, 4, 8});
  if (!blockM || !blockN || !blockK)
    return failure();
  PhysicalExprAttr unitM = parameterExpression(
      context, blockM.getParameter().getName().getValue());
  PhysicalExprAttr unitN = parameterExpression(
      context, blockN.getParameter().getName().getValue());
  PhysicalExprAttr unitK = parameterExpression(
      context, blockK.getParameter().getName().getValue());

  OpBuilder mapBuilder(mapping);
  Value rowExtent = mapBuilder.create<DimOp>(
      location, mapBuilder.getIndexType(), lhsLoad.getResource(), rowResourceAxis);
  Value columnExtent = mapBuilder.create<DimOp>(
      location, mapBuilder.getIndexType(), rhsLoad.getResource(),
      columnResourceAxis);
  auto ceilDiv = [&](Value extent, Value divisor) {
    Value one = mapBuilder.create<arith::ConstantIndexOp>(location, 1);
    Value adjusted = binary(
        mapBuilder, location, mapBuilder.getIndexType(), extent,
        binary(mapBuilder, location, mapBuilder.getIndexType(), divisor, one,
               BinaryOperator::Subtract),
        BinaryOperator::Add);
    return binary(mapBuilder, location, mapBuilder.getIndexType(), adjusted,
                  divisor, BinaryOperator::FloorDivide);
  };
  Value rowTiles = ceilDiv(rowExtent, blockM.getResult());
  Value columnTiles = ceilDiv(columnExtent, blockN.getResult());
  SmallVector<Value> mappingExtents(mapping.getExtents());
  mappingExtents.push_back(rowTiles);
  mappingExtents.push_back(columnTiles);
  SmallVector<Attribute> launchExtents(mapping.getLaunchExtents().begin(),
                                       mapping.getLaunchExtents().end());
  auto rowExpression = cast<PhysicalExprAttr>(
      cast<ViewType>(lhsLoad.getResource().getType())
          .getLayout()
          .getExtents()[rowResourceAxis]);
  auto columnExpression = cast<PhysicalExprAttr>(
      cast<ViewType>(rhsLoad.getResource().getType())
          .getLayout()
          .getExtents()[columnResourceAxis]);
  launchExtents.push_back(binaryExpression(context, PhysicalExprKind::CeilDiv,
                                            rowExpression, unitM));
  launchExtents.push_back(binaryExpression(context, PhysicalExprKind::CeilDiv,
                                            columnExpression, unitN));
  SmallVector<Type> mappingTypes(mapping.getResultTypes());
  mappingTypes.push_back(mapBuilder.getIndexType());
  mappingTypes.push_back(mapBuilder.getIndexType());
  auto expandedMapping = mapBuilder.create<DelinearizeOp>(
      location, mappingTypes, mapping.getLinear(), mappingExtents,
      mapBuilder.getArrayAttr(launchExtents));
  SmallVector<int64_t> coordinateRoles(expandedMapping.getNumResults(), -1);
  if (auto existing =
          mapping->getAttrOfType<DenseI64ArrayAttr>(coordinateRolesAttr))
    if (existing.size() == mapping.getNumResults())
      llvm::copy(existing.asArrayRef(), coordinateRoles.begin());
  coordinateRoles[mapping.getNumResults()] =
      static_cast<int64_t>(CoordinateRole::ContractionM);
  coordinateRoles[mapping.getNumResults() + 1] =
      static_cast<int64_t>(CoordinateRole::ContractionN);
  expandedMapping->setAttr(
      coordinateRolesAttr,
      DenseI64ArrayAttr::get(context, coordinateRoles));
  for (StringRef attribute : {executionGroupAttr, segmentOffsetAttr})
    if (Attribute value = mapping->getAttr(attribute))
      expandedMapping->setAttr(attribute, value);
  segmentLength = cast<PhysicalExprAttr>(launchExtents.front());
  for (Attribute extent : llvm::drop_begin(launchExtents))
    segmentLength = binaryExpression(context, PhysicalExprKind::Multiply,
                                     segmentLength,
                                     cast<PhysicalExprAttr>(extent));
  expandedMapping->setAttr(segmentLengthAttr, segmentLength);
  for (auto [oldCoordinate, newCoordinate] : llvm::zip(
           mapping.getCoordinates(),
           expandedMapping.getCoordinates().take_front(mapping.getNumResults())))
    oldCoordinate.replaceAllUsesWith(newCoordinate);
  unsigned physicalAxis = mapping.getNumResults();
  Value rowTile = expandedMapping.getCoordinates()[physicalAxis++];
  Value columnTile = expandedMapping.getCoordinates()[physicalAxis];
  mapping.erase();
  kernel->setAttr(programSpaceAttr,
                  ArrayAttr::get(context, {segmentLength}));

  OpBuilder builder(contract);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value rowStart = binary(
      builder, location, builder.getIndexType(), rowRange->getStart(),
      binary(builder, location, builder.getIndexType(), rowTile,
             blockM.getResult(), BinaryOperator::Multiply),
      BinaryOperator::Add);
  Value rowStop = binary(builder, location, builder.getIndexType(),
                         rowRange->getStart(), rowRange->getExtent(),
                         BinaryOperator::Add);
  Value columnStart = binary(
      builder, location, builder.getIndexType(), columnRange->getStart(),
      binary(builder, location, builder.getIndexType(), columnTile,
             blockN.getResult(), BinaryOperator::Multiply),
      BinaryOperator::Add);
  Value columnStop = binary(builder, location, builder.getIndexType(),
                            columnRange->getStart(), columnRange->getExtent(),
                            BinaryOperator::Add);
  Value blockStop = binary(builder, location, builder.getIndexType(),
                           blockRange->getStart(), blockRange->getExtent(),
                           BinaryOperator::Add);

  FragmentType rowIndexType = fragmentType(
      context, builder.getIndexType(), {unitM}, {*rowMap}, lhsType.getOwner());
  FragmentType columnIndexType = fragmentType(
      context, builder.getIndexType(), {unitN}, {*columnMap}, rhsType.getOwner());
  FragmentType blockIndexType = fragmentType(
      context, builder.getIndexType(), {unitK}, {*lhsBlockMap}, lhsType.getOwner());
  FragmentType rowPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM}, {*rowMap}, lhsType.getOwner());
  FragmentType columnPredicateType = fragmentType(
      context, builder.getI1Type(), {unitN}, {*columnMap}, rhsType.getOwner());
  FragmentType blockPredicateType = fragmentType(
      context, builder.getI1Type(), {unitK}, {*lhsBlockMap}, lhsType.getOwner());
  PhysicalExprAttr unitInner = innerExtent;
  FragmentType blockedLhsType = fragmentType(
      context, lhsType.getElementType(), {unitM, unitK, unitInner},
      {*rowMap, *lhsBlockMap, *lhsInnerMap}, lhsType.getOwner());
  FragmentType blockedLhsScaleType = fragmentType(
      context, lhsScaleType.getElementType(), {unitM, unitK},
      {*lhsScaleRowMap, *lhsScaleBlockMap}, lhsScaleType.getOwner());
  FragmentType blockedRhsType = fragmentType(
      context, rhsType.getElementType(), {unitK, unitInner, unitN},
      {*rhsBlockMap, *rhsInnerMap, *columnMap}, rhsType.getOwner());
  FragmentType blockedRhsScaleType = fragmentType(
      context, rhsScaleType.getElementType(), {unitK, unitN},
      {*rhsScaleBlockMap, *rhsScaleColumnMap}, rhsScaleType.getOwner());
  FragmentType blockedResultType = fragmentType(
      context, resultType.getElementType(), {unitM, unitN},
      {*rowMap, *columnMap}, resultType.getOwner());
  FragmentType lhsPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM, unitK, unitInner},
      {*rowMap, *lhsBlockMap, *lhsInnerMap}, lhsType.getOwner());
  FragmentType lhsScalePredicateType = fragmentType(
      context, builder.getI1Type(), {unitM, unitK},
      {*lhsScaleRowMap, *lhsScaleBlockMap}, lhsScaleType.getOwner());
  FragmentType rhsPredicateType = fragmentType(
      context, builder.getI1Type(), {unitK, unitInner, unitN},
      {*rhsBlockMap, *rhsInnerMap, *columnMap}, rhsType.getOwner());
  FragmentType rhsScalePredicateType = fragmentType(
      context, builder.getI1Type(), {unitK, unitN},
      {*rhsScaleBlockMap, *rhsScaleColumnMap}, rhsScaleType.getOwner());
  FragmentType outputPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM, unitN},
      {*rowMap, *columnMap}, resultType.getOwner());

  Value rows = builder.create<MakeRangeOp>(
      location, rowIndexType, rowStart, blockM.getResult(), one,
      rowMap->getSourceId(), rowMap->getSourceAxis());
  inheritRangeAuthority(rows, *rowRange);
  Value columns = builder.create<MakeRangeOp>(
      location, columnIndexType, columnStart, blockN.getResult(), one,
      columnMap->getSourceId(), columnMap->getSourceAxis());
  inheritRangeAuthority(columns, *columnRange);
  Value rowValid = compare(
      builder, location, rowPredicateType, rows,
      broadcast(builder, location, rowIndexType, rowStop), ComparePredicate::Lt);
  Value columnValid = compare(
      builder, location, columnPredicateType, columns,
      broadcast(builder, location, columnIndexType, columnStop),
      ComparePredicate::Lt);
  Value accumulator = builder.create<SplatOp>(
      location, blockedResultType, *initialAccumulator);
  bool loopBodyFailed = false;
  auto loop = builder.create<scf::ForOp>(
      location, blockRange->getStart(), blockStop, blockK.getResult(),
      ValueRange{accumulator},
      [&](OpBuilder &nested, Location nestedLocation, Value blockStart,
          ValueRange carries) {
        Value blocks = nested.create<MakeRangeOp>(
            nestedLocation, blockIndexType, blockStart, blockK.getResult(), one,
            lhsBlockMap->getSourceId(), lhsBlockMap->getSourceAxis());
        inheritRangeAuthority(blocks, *blockRange);
        Value blockValid = compare(
            nested, nestedLocation, blockPredicateType, blocks,
            broadcast(nested, nestedLocation, blockIndexType, blockStop),
            ComparePredicate::Lt);
        Value lhsRows = broadcast(nested, nestedLocation, lhsPredicateType,
                                  rowValid);
        Value lhsBlocks = broadcast(nested, nestedLocation, lhsPredicateType,
                                    blockValid);
        Value lhsValid = binary(nested, nestedLocation, lhsPredicateType,
                                lhsRows, lhsBlocks,
                                BinaryOperator::LogicalAnd);
        Value lhsScaleRows = broadcast(
            nested, nestedLocation, lhsScalePredicateType, rowValid);
        Value lhsScaleBlocks = broadcast(
            nested, nestedLocation, lhsScalePredicateType, blockValid);
        Value lhsScaleValid = binary(
            nested, nestedLocation, lhsScalePredicateType, lhsScaleRows,
            lhsScaleBlocks, BinaryOperator::LogicalAnd);
        Value rhsBlocks = broadcast(nested, nestedLocation, rhsPredicateType,
                                    blockValid);
        Value rhsColumns = broadcast(nested, nestedLocation, rhsPredicateType,
                                     columnValid);
        Value rhsValid = binary(nested, nestedLocation, rhsPredicateType,
                                rhsBlocks, rhsColumns,
                                BinaryOperator::LogicalAnd);
        Value rhsScaleBlocks = broadcast(
            nested, nestedLocation, rhsScalePredicateType, blockValid);
        Value rhsScaleColumns = broadcast(
            nested, nestedLocation, rhsScalePredicateType, columnValid);
        Value rhsScaleValid = binary(
            nested, nestedLocation, rhsScalePredicateType, rhsScaleBlocks,
            rhsScaleColumns, BinaryOperator::LogicalAnd);

        SmallVector<Value> lhsCoordinates(lhsLoad.getCoordinates());
        lhsCoordinates[*lhsRowCoordinate] = rows;
        lhsCoordinates[*lhsBlockCoordinate] = blocks;
        SmallVector<Value> lhsScaleCoordinates(lhsScaleLoad.getCoordinates());
        lhsScaleCoordinates[*lhsScaleRowCoordinate] = rows;
        lhsScaleCoordinates[*lhsScaleBlockCoordinate] = blocks;
        SmallVector<Value> rhsCoordinates(rhsLoad.getCoordinates());
        rhsCoordinates[*rhsBlockCoordinate] = blocks;
        rhsCoordinates[*rhsColumnCoordinate] = columns;
        SmallVector<Value> rhsScaleCoordinates(rhsScaleLoad.getCoordinates());
        rhsScaleCoordinates[*rhsScaleBlockCoordinate] = blocks;
        IRMapping rhsScaleReplay;
        rhsScaleReplay.map(rhsScaleColumnRange->getResult(), columns);
        FailureOr<Value> rhsScaleColumn = replaySourceValue(
            nested, nestedLocation,
            rhsScaleLoad.getCoordinates()[*rhsScaleColumnCoordinate],
            PhysicalSourceAxis{columnMap->getSourceId(),
                               columnMap->getSourceAxis()},
            unitN, *rhsScaleColumnRange, columns,
            rhsScaleReplay);
        if (failed(rhsScaleColumn)) {
          loopBodyFailed = true;
          return;
        }
        rhsScaleCoordinates[*rhsScaleColumnCoordinate] = *rhsScaleColumn;
        FailureOr<Value> lhsFill = retargetFill(
            nested, nestedLocation, lhsLoad.getFill(), blockedLhsType);
        FailureOr<Value> lhsScaleFill = retargetFill(
            nested, nestedLocation, lhsScaleLoad.getFill(),
            blockedLhsScaleType);
        FailureOr<Value> rhsFill = retargetFill(
            nested, nestedLocation, rhsLoad.getFill(), blockedRhsType);
        FailureOr<Value> rhsScaleFill = retargetFill(
            nested, nestedLocation, rhsScaleLoad.getFill(),
            blockedRhsScaleType);
        if (failed(lhsFill) || failed(lhsScaleFill) || failed(rhsFill) ||
            failed(rhsScaleFill)) {
          loopBodyFailed = true;
          return;
        }
        Value lhs = nested.create<LoadOp>(
            nestedLocation, blockedLhsType, lhsLoad.getResource(),
            lhsCoordinates, lhsValid, *lhsFill, lhsLoad.getSourceAxes());
        Value lhsScale = nested.create<LoadOp>(
            nestedLocation, blockedLhsScaleType, lhsScaleLoad.getResource(),
            lhsScaleCoordinates, lhsScaleValid, *lhsScaleFill,
            lhsScaleLoad.getSourceAxes());
        Value rhs = nested.create<LoadOp>(
            nestedLocation, blockedRhsType, rhsLoad.getResource(),
            rhsCoordinates, rhsValid, *rhsFill, rhsLoad.getSourceAxes());
        Value rhsScale = nested.create<LoadOp>(
            nestedLocation, blockedRhsScaleType, rhsScaleLoad.getResource(),
            rhsScaleCoordinates, rhsScaleValid, *rhsScaleFill,
            rhsScaleLoad.getSourceAxes());
        auto product = nested.create<ScaledContractOp>(
            nestedLocation, blockedResultType, lhs, lhsScale, rhs, rhsScale,
            carries.front(), ArrayRef<int64_t>{1, 2},
            ArrayRef<int64_t>{0, 1}, ArrayRef<int64_t>{},
            ArrayRef<int64_t>{}, contract.getLhsFormat(),
            contract.getRhsFormat(), contract.getLhsGroupSize(),
            contract.getRhsGroupSize());
        if (Attribute origin = contract->getAttr(originAttr))
          product->setAttr(originAttr, origin);
        nested.create<scf::YieldOp>(nestedLocation, product.getResult());
      });
  if (loopBodyFailed) {
    loop.erase();
    return reject("blocked loop body could not be materialized");
  }

  Value outputRows = broadcast(builder, location, outputPredicateType, rowValid);
  Value outputColumns =
      broadcast(builder, location, outputPredicateType, columnValid);
  Value outputValid = binary(builder, location, outputPredicateType, outputRows,
                             outputColumns, BinaryOperator::LogicalAnd);
  for (StorePath &path : paths) {
    Value output = loop.getResult(0);
    for (CastOp conversion : path.casts) {
      auto original = cast<FragmentType>(conversion.getResult().getType());
      FragmentType converted = fragmentType(
          context, original.getElementType(), {unitM, unitN},
          {*rowMap, *columnMap}, original.getOwner());
      output = builder.create<CastOp>(location, converted, output);
    }
    SmallVector<Value> coordinates(path.store.getCoordinates());
    FailureOr<unsigned> storeRow = queryCoordinatePosition(
        path.store.getCoordinates(),
        PhysicalSourceAxis{rowMap->getSourceId(), rowMap->getSourceAxis()});
    FailureOr<unsigned> storeColumn = queryCoordinatePosition(
        path.store.getCoordinates(),
        PhysicalSourceAxis{columnMap->getSourceId(), columnMap->getSourceAxis()});
    if (failed(storeRow) || failed(storeColumn))
      return path.store.emitOpError(
          "blocked scaled-contract output lost source coordinates");
    coordinates[*storeRow] = rows;
    coordinates[*storeColumn] = columns;
    auto replacement = builder.create<StoreOp>(
        location, path.store.getResource(), coordinates, output, outputValid,
        path.store.getSourceAxes(), path.store.getCollision());
    if (Attribute origin = path.store->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
  }

  for (StorePath &path : paths)
    path.store.erase();
  for (StorePath &path : paths)
    for (CastOp conversion : llvm::reverse(path.casts))
      if (conversion->getBlock() && conversion.getResult().use_empty())
        conversion.erase();
  if (contract->getBlock() && contract.getResult().use_empty())
    contract.erase();
  for (LoadOp load : {lhsLoad, lhsScaleLoad, rhsLoad, rhsScaleLoad})
    if (load->getBlock() && load.getResult().use_empty())
      load.erase();
  return success();
}

} // namespace

LogicalResult realizeContractionBlocking(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<ContractOp> contracts;
  SmallVector<ScaledContractOp> scaledContracts;
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  for (ContractOp contract : contracts)
    if (contract.getLhsReductionAxes().size() > 1 &&
        failed(decomposeMultiReductionContract(contract)))
      return failure();
  contracts.clear();
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  if (failed(normalizeMatrixContractForms(kernel)))
    return failure();
  contracts.clear();
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  kernel.walk(
      [&](ScaledContractOp contract) { scaledContracts.push_back(contract); });
  for (ContractOp contract : contracts)
    if (!hasFragmentSchema(contract))
      return contract.emitOpError(
          "shared contraction blocking requires fragment operands and result");
  for (ScaledContractOp contract : scaledContracts)
    if (!hasFragmentSchema(contract))
      return contract.emitOpError(
          "shared scaled-contraction blocking requires fragment operands, accumulator, and result");
  for (ContractOp contract : contracts)
    if (requiresPhysicalRealization(contract)) {
      const bool rangeSingleReduction = hasRangeContractForm(contract);
      if (!rangeSingleReduction &&
          contract.getLhsReductionAxes().size() == 1 &&
          contract.getRhsReductionAxes().size() == 1 &&
          hasSelectedFreeAxes(contract) &&
          hasExplicitPairedReductionRanges(contract)) {
        if (failed(realizeReductionTraversal(contract, kernel)))
          return failure();
      } else if (!rangeSingleReduction &&
                 contract.getLhsReductionAxes().size() == 1 &&
                 contract.getRhsReductionAxes().size() == 1) {
        if (failed(markNativeCoverage(kernel, contract)))
          return failure();
      } else if (failed(realizeContract(contract, kernel))) {
        return failure();
      }
    } else if (failed(markNativeCoverage(kernel, contract))) {
      return failure();
    }
  for (ScaledContractOp contract : scaledContracts)
    if (requiresPhysicalRealization(contract)) {
      if (failed(realizeScaledContract(contract, kernel)))
        return failure();
    } else if (failed(markNativeCoverage(kernel, contract))) {
      return failure();
    }
  eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace intent::gpu
