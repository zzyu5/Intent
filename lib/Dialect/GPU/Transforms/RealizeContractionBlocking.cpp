#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
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

bool isOwnershipExtent(Operation *origin, Attribute attribute) {
  auto extent = dyn_cast<PhysicalExprAttr>(attribute);
  if (!origin || !extent ||
      extent.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Parameter))
    return false;
  func::FuncOp kernel = origin->getParentOfType<func::FuncOp>();
  if (!kernel)
    return false;
  bool ownership = false;
  kernel.walk([&](ParameterOp parameter) {
    if (parameter.getParameter().getName() != extent.getSymbol())
      return;
    auto role = static_cast<ParameterRole>(parameter.getParameter().getRole());
    ownership |= role == ParameterRole::OwnershipM ||
                 role == ParameterRole::OwnershipN;
  });
  return ownership;
}

bool reductionUsesOwnershipExtent(Operation *origin, Value operand,
                                  ArrayRef<int64_t> reductionAxes) {
  auto fragment = dyn_cast<FragmentType>(operand.getType());
  if (!fragment)
    return false;
  return llvm::any_of(reductionAxes, [&](int64_t axis) {
    return axis >= 0 && axis < static_cast<int64_t>(fragment.getShape().size()) &&
           isOwnershipExtent(origin, fragment.getShape()[axis]);
  });
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

bool hasFragmentSchema(SparseContractOp contract) {
  if (!isa<FragmentType>(contract.getCompressed().getType()) ||
      !isa<FragmentType>(contract.getRhs().getType()) ||
      !isa<FragmentType>(contract.getAccumulator().getType()) ||
      !isa<FragmentType>(contract.getResult().getType()))
    return false;
  Type metadata = contract.getMetadata().getType();
  if (isa<FragmentType>(metadata))
    return true;
  auto record = dyn_cast<RecordType>(metadata);
  return record && llvm::all_of(record.getFieldTypes(), [](Attribute field) {
           return isa<FragmentType>(cast<TypeAttr>(field).getValue());
         });
}

bool hasUnrealizedPhysicalAxis(Value value) {
  auto fragment = dyn_cast<FragmentType>(value.getType());
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!fragment || !kernel)
    return false;
  PhysicalProgramAnalysis analysis(kernel);
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    PhysicalAxisRealizationFact fact = analysis.axisRealization(value, axis);
    if (fact.constructionScalarSeed ||
        (fact.isExact() && !fact.physicalized))
      return true;
  }
  return false;
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
  if (reductionUsesOwnershipExtent(contract, contract.getLhs(),
                                   contract.getLhsReductionAxes()) ||
      reductionUsesOwnershipExtent(contract, contract.getRhs(),
                                   contract.getRhsReductionAxes()))
    return true;
  return hasUnrealizedPhysicalAxis(contract.getLhs()) ||
         hasUnrealizedPhysicalAxis(contract.getRhs());
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
  if (reductionUsesOwnershipExtent(contract, contract.getLhs(),
                                   contract.getLhsReductionAxes()) ||
      reductionUsesOwnershipExtent(contract, contract.getRhs(),
                                   contract.getRhsReductionAxes()))
    return true;
  return hasUnrealizedPhysicalAxis(contract.getLhs()) ||
         hasUnrealizedPhysicalAxis(contract.getLhsScale()) ||
         hasUnrealizedPhysicalAxis(contract.getRhs()) ||
         hasUnrealizedPhysicalAxis(contract.getRhsScale());
}

Value binary(OpBuilder &builder, Location location, Type result, Value lhs,
             Value rhs, BinaryOperator kind) {
  return builder.create<BinaryOp>(location, result, lhs, rhs, kind);
}

Value compare(OpBuilder &builder, Location location, Type result, Value lhs,
              Value rhs, ComparePredicate predicate) {
  auto comparison =
      builder.create<CompareOp>(location, result, lhs, rhs, predicate);
  comparison->setAttr(physicalTailAttr, builder.getUnitAttr());
  return comparison.getResult();
}

void inheritRangeAuthority(Value derived, MakeRangeOp source) {
  Operation *operation = derived.getDefiningOp();
  for (StringRef name : {sourceSubregionAttr, sourceSubregionBoundAttr})
    if (Attribute value = source->getAttr(name))
      operation->setAttr(name, value);
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

bool containsSource(Value value, PhysicalSourceAxis source) {
  return queryFragmentAxis(value.getType(), source).isExact();
}

FailureOr<unsigned> mappingAxisForScalar(Value value, DelinearizeOp mapping) {
  auto roles =
      mapping->getAttrOfType<DenseI64ArrayAttr>(coordinateRolesAttr);
  if (!roles || roles.size() != mapping.getNumResults())
    return failure();
  const int64_t pointwiseOwnership =
      static_cast<int64_t>(CoordinateRole::PointwiseOwnership);
  llvm::SmallDenseSet<unsigned, 2> axes;
  llvm::SmallPtrSet<Operation *, 16> visited;
  std::function<void(Value)> collect = [&](Value current) {
    for (auto [axis, coordinate] : llvm::enumerate(mapping.getCoordinates()))
      if (current == coordinate) {
        if (roles[axis] == pointwiseOwnership)
          axes.insert(axis);
        return;
      }
    Operation *producer = current.getDefiningOp();
    if (!producer || producer->getNumRegions() != 0 ||
        !visited.insert(producer).second)
      return;
    for (Value operand : producer->getOperands())
      collect(operand);
  };
  collect(value);
  return axes.size() == 1 ? FailureOr<unsigned>(*axes.begin())
                          : FailureOr<unsigned>(failure());
}

LogicalResult refineOwnershipParameter(func::FuncOp kernel, MakeRangeOp range,
                                       FailureOr<unsigned> mappingAxis,
                                       ParameterOp replacement) {
  if (failed(mappingAxis))
    return success();
  FailureOr<ParameterOp> previous = queryBlockingParameter(kernel, range);
  if (failed(previous))
    return range.emitOpError(
        "pointwise ownership axis has no typed blocking parameter to refine");
  if (*previous == replacement)
    return success();
  ParameterAttr previousSchema = previous->getParameter();
  auto previousRole = static_cast<ParameterRole>(previousSchema.getRole());
  if (previousSchema.getCategory() !=
          static_cast<uint32_t>(ParameterCategory::Pointwise) ||
      (previousRole != ParameterRole::OwnershipM &&
       previousRole != ParameterRole::OwnershipN))
    return previous->emitOpError(
        "contraction can only refine a provisional pointwise ownership parameter");
  for (StringRef attribute : {dimensionAttr, parameterSourceAttr,
                              coverageDimensionAttr}) {
    Attribute inherited = (*previous)->getAttr(attribute);
    Attribute current = replacement->getAttr(attribute);
    if (inherited && current && inherited != current)
      return replacement.emitOpError(
          "contraction ownership refinement has conflicting parameter bindings");
    if (inherited && !current)
      replacement->setAttr(attribute, inherited);
  }
  return replacePhysicalParameter(kernel, *previous, replacement);
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
            sourceAxisIdentity(mapping),
            ranges)) ||
        ranges.empty())
      return false;
    return llvm::all_of(ranges, [&](MakeRangeOp range) {
      return sourceAxisIdentity(range) == sourceAxisIdentity(mapping);
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

FailureOr<Value> replaySourceValueImpl(OpBuilder &builder, Location location,
                                       Value value,
                                       PhysicalExprAttr blockedExtent,
                                       ArrayRef<MakeRangeOp> roots,
                                       Value replacement,
                                       IRMapping &mapping) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  if (auto range = value.getDefiningOp<MakeRangeOp>())
    if (llvm::any_of(roots, [&](MakeRangeOp root) {
          return range == root || sameLogicalRange(range, root);
        }))
      return replacement;
  auto originalResultType = dyn_cast<FragmentType>(value.getType());
  if (!originalResultType)
    return value;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalRangeAxisFact selected =
      PhysicalProgramAnalysis(kernel).rangeAxes(value, roots);
  if (!selected.isExact()) {
    InFlightDiagnostic diagnostic =
        value.getDefiningOp()
            ? value.getDefiningOp()->emitOpError(
                  "selected range roots have no exact fragment-axis projection")
            : kernel.emitError(
                  "selected range roots have no exact fragment-axis projection");
    for (Operation *blocker : selected.blockers)
      diagnostic << "; blocker=" << blocker->getName();
    diagnostic << "; value_type=" << value.getType();
    if (auto broadcast = value.getDefiningOp<BroadcastOp>())
      diagnostic << "; broadcast_input_type=" << broadcast.getValue().getType();
    return failure();
  }
  if (selected.fragmentAxes.empty())
    return value;
  Operation *producer = value.getDefiningOp();
  if (!producer || isa<MakeRangeOp>(producer)) {
    if (producer)
      producer->emitOpError(
          "selected range root was not bound to its blocked replacement");
    return failure();
  }
  if (!isPhysicalReplayNode(producer, PhysicalReplayScope::ValueGraph,
                            /*allowAccesses=*/true)) {
    producer->emitOpError(
        "selected range value is not a replayable physical value node");
    return failure();
  }
  if (producer->getNumRegions() != 0 || producer->getNumResults() != 1) {
    producer->emitOpError(
        "selected range value does not have a single replayable result");
    return failure();
  }
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replayed = replaySourceValueImpl(
        builder, location, operand, blockedExtent, roots, replacement, mapping);
    if (failed(replayed)) {
      producer->emitOpError(
          "selected range value has an operand that cannot be replayed");
      return failure();
    }
    if (*replayed != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replayed);
  }
  SmallVector<Attribute> targetShape(originalResultType.getShape().begin(),
                                     originalResultType.getShape().end());
  for (unsigned axis : selected.fragmentAxes)
    targetShape[axis] = blockedExtent;
  FragmentType targetType = FragmentType::get(
      originalResultType.getContext(), originalResultType.getElementType(),
      ArrayAttr::get(originalResultType.getContext(), targetShape),
      originalResultType.getAxisMaps(), originalResultType.getValidity(),
      originalResultType.getOwner());
  if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
    PhysicalRangeAxisFact input =
        PhysicalProgramAnalysis(kernel).rangeAxes(reshape.getValue(), roots);
    if (!input.isExact()) {
      reshape.emitOpError(
          "reshape input has no exact selected-range axis projection");
      return failure();
    }
    if (input.fragmentAxes.empty()) {
      Value input = mapping.lookupOrDefault(reshape.getValue());
      auto broadcast = builder.create<BroadcastOp>(location, targetType, input);
      if (!mapping.lookupOrNull(value))
        mapping.map(value, broadcast.getResult());
      return broadcast.getResult();
    }
  }
  IRMapping cloneMapping(mapping);
  Value accumulator;
  if (auto contract = dyn_cast<ContractOp>(producer))
    accumulator = contract.getAccumulator();
  else if (auto contract = dyn_cast<ScaledContractOp>(producer))
    accumulator = contract.getAccumulator();
  else if (auto contract = dyn_cast<SparseContractOp>(producer))
    accumulator = contract.getAccumulator();
  if (accumulator) {
    Value current = mapping.lookupOrDefault(accumulator);
    if (current.getType() != targetType) {
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, location, current, targetType);
      if (failed(projected)) {
        producer->emitOpError(
            "replayed structured value accumulator cannot adopt its result schema");
        return failure();
      }
      cloneMapping.map(accumulator, *projected);
    }
  }
  if (isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(producer)) {
    for (Value operand : producer->getOperands()) {
      Value current = cloneMapping.lookupOrDefault(operand);
      auto fragment = dyn_cast<FragmentType>(current.getType());
      if (!fragment)
        continue;
      auto operandTarget = FragmentType::get(
          targetType.getContext(), fragment.getElementType(),
          targetType.getShape(), targetType.getAxisMaps(),
          targetType.getValidity(), targetType.getOwner());
      if (current.getType() == operandTarget)
        continue;
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, location, current, operandTarget);
      if (failed(projected)) {
        producer->emitOpError(
            "replayed pointwise operand cannot adopt the selected range relation")
            << "; operand=" << current.getType()
            << "; target=" << operandTarget;
        return failure();
      }
      cloneMapping.map(operand, *projected);
    }
  }
  Operation *clone = builder.clone(*producer, cloneMapping);
  auto resultType = dyn_cast<FragmentType>(clone->getResult(0).getType());
  if (!resultType) {
    clone->emitOpError("replayed value did not preserve a fragment result");
    return failure();
  }
  clone->getResult(0).setType(targetType);
  if (!mapping.lookupOrNull(value))
    mapping.map(value, clone->getResult(0));
  return clone->getResult(0);
}

FailureOr<Value> replaySourceValue(OpBuilder &builder, Location location,
                                   Value value,
                                   PhysicalExprAttr blockedExtent,
                                   ArrayRef<MakeRangeOp> roots,
                                   Value replacement,
                                   IRMapping &mapping) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel || roots.empty())
    return failure();
  PhysicalSourceAxis source = sourceAxisIdentity(roots.front());
  FailureOr<int64_t> dimension = queryRangeDimension(roots.front());
  if (failed(dimension) ||
      !llvm::all_of(roots, [&](MakeRangeOp root) {
        FailureOr<int64_t> current = queryRangeDimension(root);
        return sourceAxisIdentity(root) == source && succeeded(current) &&
               *current == *dimension;
      }))
    return failure();
  PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
      value, source, PhysicalReplayScope::ValueGraph,
      /*allowAccesses=*/true, /*insertionAnchor=*/nullptr, *dimension);
  if (!replay.isReplayable()) {
    InFlightDiagnostic diagnostic =
        value.getDefiningOp()
            ? value.getDefiningOp()->emitOpError(
                  "contraction operand has no exact coordinate replay fact")
            : kernel.emitError(
                  "contraction operand has no exact coordinate replay fact");
    for (Operation *blocker : replay.blockers)
      diagnostic << "; blocker=" << blocker->getName();
    return failure();
  }
  FailureOr<Value> result = replaySourceValueImpl(
      builder, location, value, blockedExtent, roots, replacement, mapping);
  if (failed(result))
    (value.getDefiningOp() ? value.getDefiningOp() : kernel.getOperation())
        ->emitError("coordinate replay could not rebuild the current value graph");
  return result;
}

FailureOr<Value> buildRangeTailPredicate(OpBuilder &builder, Location location,
                                         Value range,
                                         MakeRangeOp authority) {
  auto rangeType = dyn_cast<FragmentType>(range.getType());
  if (!rangeType || rangeType.getShape().size() != 1)
    return failure();
  Value logicalStop = builder.create<SplatOp>(
      location, rangeType, authority.getLogicalStop());
  auto predicateType = FragmentType::get(
      rangeType.getContext(), builder.getI1Type(), rangeType.getShape(),
      rangeType.getAxisMaps(), rangeType.getValidity(), rangeType.getOwner());
  return builder
      .create<CompareOp>(location, predicateType, range, logicalStop,
                         ComparePredicate::Lt)
      .getResult();
}

LogicalResult appendTailValidity(Location location, Value source,
                                 Value tailPredicate,
                                 IRMapping &mapping) {
  SmallVector<Value> pending{source};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    Operation *producer = value.getDefiningOp();
    if (!producer)
      continue;
    if (auto originalLoad = dyn_cast<LoadOp>(producer)) {
      Value mapped = mapping.lookupOrNull(originalLoad.getResult());
      auto load = mapped ? mapped.getDefiningOp<LoadOp>() : LoadOp();
      if (!load)
        continue;
      auto resultType = dyn_cast<FragmentType>(load.getResult().getType());
      if (!resultType)
        return load.emitOpError(
            "blocked sparse tail requires a fragment load result");
      auto predicateType = FragmentType::get(
          resultType.getContext(), IntegerType::get(resultType.getContext(), 1),
          resultType.getShape(),
          resultType.getAxisMaps(), resultType.getValidity(),
          resultType.getOwner());
      OpBuilder validityBuilder(load);
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          validityBuilder, location, tailPredicate, predicateType);
      if (failed(projected))
        return load.emitOpError(
            "blocked sparse tail cannot project to its load coordinates");
      Value valid = *projected;
      if (load.getValid()) {
        Value existing = load.getValid();
        if (existing.getType() != predicateType) {
          FailureOr<Value> projectedExisting = projectPhysicalValueToSchema(
              validityBuilder, location, existing, predicateType);
          if (failed(projectedExisting))
            return load.emitOpError(
                "blocked sparse tail cannot preserve existing load validity");
          existing = *projectedExisting;
        }
        valid = validityBuilder.create<BinaryOp>(
            location, predicateType, existing, valid,
            BinaryOperator::LogicalAnd);
      }
      load.getValidMutable().assign(ValueRange{valid});
      if (!load.getFill()) {
        Type elementType = resultType.getElementType();
        Value zero = validityBuilder.create<arith::ConstantOp>(
            location, elementType, validityBuilder.getZeroAttr(elementType));
        Value fill =
            validityBuilder.create<SplatOp>(location, resultType, zero);
        load.getFillMutable().assign(ValueRange{fill});
      }
      continue;
    }
    pending.append(producer->operand_begin(), producer->operand_end());
  }
  return success();
}

FailureOr<Value> replaySourceValue(OpBuilder &builder, Location location,
                                   Value value, PhysicalSourceAxis source,
                                   PhysicalExprAttr blockedExtent,
                                   MakeRangeOp root, Value replacement,
                                   IRMapping &mapping) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  FailureOr<int64_t> dimension = queryRangeDimension(root);
  if (!kernel || !(sourceAxisIdentity(root) == source) || failed(dimension))
    return failure();
  PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
      value, source, PhysicalReplayScope::ValueGraph,
      /*allowAccesses=*/true, /*insertionAnchor=*/nullptr, *dimension);
  if (!replay.isReplayable()) {
    InFlightDiagnostic diagnostic =
        value.getDefiningOp()
            ? value.getDefiningOp()->emitOpError(
                  "contraction operand has no exact source-scoped replay fact")
            : kernel.emitError(
                  "contraction operand has no exact source-scoped replay fact");
    for (Operation *blocker : replay.blockers)
      diagnostic << "; blocker=" << blocker->getName();
    return failure();
  }
  return replaySourceValueImpl(builder, location, value, blockedExtent,
                               ArrayRef<MakeRangeOp>(root), replacement,
                               mapping);
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
            schema.getCategory(), schema.getElementBitWidth(),
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
      if (sourceAxisIdentity(range) == source &&
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
        mapping.getDimensionId(), resultAxis, mapping.getDerived()));
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
                                        source.getDimensionId(), axis,
                                        source.getDerived()));
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
    if (!store || store.getValue() != value)
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

FailureOr<unsigned> accessCoordinatePosition(LoadOp load,
                                             AxisMapAttr mapping) {
  if (FailureOr<unsigned> direct = queryCoordinatePosition(
          load.getCoordinates(), sourceAxisIdentity(mapping));
      succeeded(direct))
    return direct;
  auto kernel = load->getParentOfType<func::FuncOp>();
  if (kernel) {
    PhysicalProgramAnalysis analysis(kernel);
    std::optional<unsigned> replayed;
    for (auto [position, coordinate] :
         llvm::enumerate(load.getCoordinates())) {
      PhysicalRangeFact fact =
          analysis.sourceRanges(coordinate, sourceAxisIdentity(mapping));
      if (failed(queryExactLogicalRange(fact)))
        continue;
      if (replayed)
        return failure();
      replayed = position;
    }
    if (replayed)
      return *replayed;
  }
  auto result = dyn_cast<FragmentType>(load.getResult().getType());
  if (result && result.getShape().size() == load.getCoordinates().size() &&
      llvm::all_of(load.getCoordinates(), [](Value coordinate) {
        auto fragment = dyn_cast<FragmentType>(coordinate.getType());
        return fragment && fragment.getShape().size() == 1;
      }) &&
      mapping.getFragmentAxis() < load.getCoordinates().size())
    return mapping.getFragmentAxis();
  auto view = dyn_cast<ViewType>(load.getResource().getType());
  if (!view || mapping.getDimensionId() <= 0)
    return failure();
  std::optional<unsigned> resourceAxis;
  for (auto [axis, dimension] :
       llvm::enumerate(view.getLayout().getDimensionIds().asArrayRef())) {
    if (dimension != mapping.getDimensionId())
      continue;
    if (resourceAxis)
      return failure();
    resourceAxis = axis;
  }
  if (!resourceAxis)
    return failure();
  std::optional<unsigned> coordinate;
  for (auto [position, axis] : llvm::enumerate(load.getSourceAxes())) {
    if (axis != *resourceAxis)
      continue;
    if (coordinate)
      return failure();
    coordinate = position;
  }
  return coordinate ? FailureOr<unsigned>(*coordinate)
                    : FailureOr<unsigned>(failure());
}

FailureOr<unsigned> directRankOneAccessPosition(ValueRange coordinates,
                                                Type valueType,
                                                unsigned valueAxis) {
  auto fragment = dyn_cast<FragmentType>(valueType);
  if (!fragment || fragment.getShape().size() != coordinates.size() ||
      valueAxis >= coordinates.size() ||
      !llvm::all_of(coordinates, [](Value coordinate) {
        auto type = dyn_cast<FragmentType>(coordinate.getType());
        return type && type.getShape().size() == 1;
      }))
    return failure();
  return valueAxis;
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
    FailureOr<unsigned> coordinate = accessCoordinatePosition(load, mapping);
    if (failed(coordinate))
      return false;
    Value value = load.getCoordinates()[*coordinate];
    if (!sourceRange(value) &&
        failed(producerRange(
            value, sourceAxisIdentity(mapping))))
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

bool freeAxesNeedRealization(ContractOp contract, func::FuncOp kernel) {
  return PhysicalProgramAnalysis(kernel)
      .contractFreeAxes(contract.getOperation())
      .needsRealization();
}

bool hasCompleteStorePath(ContractOp contract) {
  SmallVector<StorePath> paths;
  llvm::SmallPtrSet<Operation *, 8> visited;
  return collectStorePaths(contract.getResult(), {}, paths, visited);
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
        mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
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
        sourceAxisIdentity(mapping));
    if (failed(coordinate))
      continue;
    MakeRangeOp range = sourceRange(coordinates[*coordinate]);
    if (!range) {
      FailureOr<MakeRangeOp> root =
          producerRange(coordinates[*coordinate],
                        sourceAxisIdentity(mapping));
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
    Value coordinate, FragmentType target,
    PhysicalProgramAnalysis &analysis) {
  if (!value)
    return Value();
  FailureOr<int64_t> sourceDimension = queryRangeDimension(root);
  PhysicalReductionDependencyFact dependency = analysis.reductionDependency(
      value, sourceAxisIdentity(root),
      succeeded(sourceDimension)
          ? std::optional<int64_t>(*sourceDimension)
          : std::nullopt);
  if (!dependency.isExact()) {
    InFlightDiagnostic diagnostic = root.emitOpError(
        "validity predicate has no exact source-range dependency");
    for (Operation *blocker : dependency.blockers)
      diagnostic << "; blocker=" << blocker->getName();
    return failure();
  }
  if (!dependency.depends) {
    if (value.getType() == target)
      return value;
    return projectBroadcast(builder, location, value, target);
  }
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    return projectPredicateForScalarAxis(builder, location,
                                         broadcast.getValue(), root,
                                         coordinate, target, analysis);
  if (auto splat = value.getDefiningOp<SplatOp>())
    return projectPredicateForScalarAxis(builder, location, splat.getValue(),
                                         root, coordinate, target, analysis);
  if (auto conjunction = value.getDefiningOp<BinaryOp>()) {
    if (conjunction.getOperatorKind() != BinaryOperator::LogicalAnd)
      return failure();
    FailureOr<Value> lhs = projectPredicateForScalarAxis(
        builder, location, conjunction.getLhs(), root, coordinate, target,
        analysis);
    FailureOr<Value> rhs = projectPredicateForScalarAxis(
        builder, location, conjunction.getRhs(), root, coordinate, target,
        analysis);
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
      Value scalar = builder.create<CompareOp>(
          location, builder.getI1Type(), coordinate, rhs,
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
  PhysicalProgramAnalysis predicateAnalysis(
      contract->getParentOfType<func::FuncOp>());

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
    PhysicalLockstepTraversalFact lockstep =
        predicateAnalysis.lockstepRanges({lhsRange, rhsRange});
    if (!lockstep.isExact()) {
      failureReason =
          "a reduction pair does not have one lockstep physical traversal";
      return failure();
    }
    FailureOr<Value> logicalEnd = resolveLogicalRangeEnd(
        contract->getParentOfType<func::FuncOp>(), lockstep.authority);
    if (failed(logicalEnd)) {
      failureReason = "a reduction pair has no exact logical end";
      return failure();
    }

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
        location, lockstep.authority.getStart(), *logicalEnd, *lhsStep,
        ValueRange{accumulator},
        [&](OpBuilder &nested, Location nestedLocation, Value coordinate,
            ValueRange carries) {
          SmallVector<Value> nestedLhsCoordinates(currentLhsCoordinates);
          SmallVector<Value> nestedRhsCoordinates(currentRhsCoordinates);
          nestedLhsCoordinates[*lhsCoordinate] = coordinate;
          nestedRhsCoordinates[*rhsCoordinate] = coordinate;
          FailureOr<Value> nestedLhsValid = projectPredicateForScalarAxis(
              nested, nestedLocation, currentLhsValid, lhsRange, coordinate,
              nestedLhsPredicateType, predicateAnalysis);
          FailureOr<Value> nestedRhsValid = projectPredicateForScalarAxis(
              nested, nestedLocation, currentRhsValid, rhsRange, coordinate,
              nestedRhsPredicateType, predicateAnalysis);
          if (failed(nestedLhsValid) || failed(nestedRhsValid)) {
            failureReason = failed(nestedLhsValid)
                                ? "lhs tail validity could not be projected while scalarizing a reduction pair"
                                : "rhs tail validity could not be projected while scalarizing a reduction pair";
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

scf::ForOp enclosingRegionContractionSegment(Operation *operation) {
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop)
      continue;
    auto segment = loop.getStep().getDefiningOp<ParameterOp>();
    if (!segment ||
        segment.getParameter().getRole() !=
            static_cast<uint32_t>(ParameterRole::ScanChunk) ||
        segment.getParameter().getCategory() !=
            static_cast<uint32_t>(ParameterCategory::RegionContraction))
      continue;
    return loop;
  }
  return {};
}

FailureOr<bool> realizeSegmentNativeReduction(ContractOp contract,
                                              func::FuncOp kernel) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1)
    return false;
  scf::ForOp segmentLoop =
      enclosingRegionContractionSegment(contract.getOperation());
  if (!segmentLoop)
    return false;
  ParameterOp segment = segmentLoop.getStep().getDefiningOp<ParameterOp>();
  StringAttr segmentName = segment.getParameter().getName();
  auto usesSegment = [&](Value operand, int64_t axis) {
    auto fragment = dyn_cast<FragmentType>(operand.getType());
    if (!fragment || axis < 0 ||
        axis >= static_cast<int64_t>(fragment.getShape().size()))
      return false;
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    return extent.getKind() ==
               static_cast<uint32_t>(PhysicalExprKind::Parameter) &&
           extent.getSymbol() == segmentName;
  };
  if (!usesSegment(contract.getLhs(),
                   contract.getLhsReductionAxes().front()) ||
      !usesSegment(contract.getRhs(),
                   contract.getRhsReductionAxes().front()))
    return false;
  if (failed(markNativeCoverage(kernel, contract)))
    return failure();
  return true;
}

FailureOr<bool> realizeStructuredNativeReduction(ContractOp contract,
                                                 func::FuncOp kernel) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1)
    return false;
  LoadOp lhsLoad = matrixOperandLoad(contract.getLhs());
  LoadOp rhsLoad = matrixOperandLoad(contract.getRhs());
  if (!lhsLoad || !rhsLoad)
    return false;
  scf::ForOp segmentLoop =
      enclosingRegionContractionSegment(contract.getOperation());
  if (!segmentLoop)
    return false;

  DominanceInfo dominance(kernel);
  bool lhsInvariant = dominance.dominates(lhsLoad.getOperation(),
                                          segmentLoop.getOperation());
  bool rhsInvariant = dominance.dominates(rhsLoad.getOperation(),
                                          segmentLoop.getOperation());
  if (lhsInvariant == rhsInvariant)
    return false;

  Value invariant = lhsInvariant ? contract.getLhs() : contract.getRhs();
  unsigned reductionAxis = static_cast<unsigned>(
      lhsInvariant ? contract.getLhsReductionAxes().front()
                   : contract.getRhsReductionAxes().front());
  if (failed(realizeFullCoverageDimension(kernel, invariant, reductionAxis)))
    return contract.emitOpError(
        "structured contraction invariant has no exact full-coverage reduction realization");
  if (failed(markNativeCoverage(kernel, contract)))
    return failure();
  return true;
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
  PhysicalProgramAnalysis physicalAnalysis(kernel);
  auto rangesFor = [&](Value value, unsigned fragmentAxis, AxisMapAttr mapping,
                       SmallVectorImpl<MakeRangeOp> &ranges) {
    PhysicalRangeFact fact =
        physicalAnalysis.axisRanges(value, fragmentAxis);
    if (failed(queryExactLogicalRange(fact)) &&
        queryFragmentAxes(value.getType(), sourceAxisIdentity(mapping)).size() ==
            1)
      fact = physicalAnalysis.sourceRanges(value, sourceAxisIdentity(mapping));
    if (failed(queryExactLogicalRange(fact)) || !fact.unitStep)
      return failure();
    ranges.assign(fact.roots.begin(), fact.roots.end());
    return llvm::all_of(ranges, [&](MakeRangeOp range) {
             return sourceAxisIdentity(range) == sourceAxisIdentity(mapping);
           })
               ? success()
               : failure();
  };
  SmallVector<MakeRangeOp> lhsRanges;
  SmallVector<MakeRangeOp> rhsRanges;
  if (failed(rangesFor(contract.getLhs(), lhsReduction, *lhsMap, lhsRanges)) ||
      failed(rangesFor(contract.getRhs(), rhsReduction, *rhsMap, rhsRanges))) {
    PhysicalRangeFact lhsFact =
        physicalAnalysis.axisRanges(contract.getLhs(), lhsReduction);
    PhysicalRangeFact rhsFact =
        physicalAnalysis.axisRanges(contract.getRhs(), rhsReduction);
    InFlightDiagnostic diagnostic = contract.emitOpError(
        "reduction-only contraction blocking requires explicit paired ranges");
    diagnostic << "; lhs_state=" << static_cast<unsigned>(lhsFact.state)
               << ", lhs_roots=" << lhsFact.roots.size()
               << ", lhs_blockers=" << lhsFact.blockers.size()
               << ", rhs_state=" << static_cast<unsigned>(rhsFact.state)
               << ", rhs_roots=" << rhsFact.roots.size()
               << ", rhs_blockers=" << rhsFact.blockers.size()
               << ", lhs_type=" << contract.getLhs().getType()
               << ", rhs_type=" << contract.getRhs().getType();
    for (Operation *blocker : lhsFact.blockers)
      diagnostic << ", lhs_blocker=" << blocker->getName();
    for (Operation *blocker : rhsFact.blockers)
      diagnostic << ", rhs_blocker=" << blocker->getName();
    return failure();
  }
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
  ParameterOp blockK = getOrCreatePhysicalParameter(
      kernel, "BLOCK_K" + suffix, ParameterRole::Reduction,
      ParameterCategory::Contraction,
      lhsType.getElementType().getIntOrFloatBitWidth(), {32, 64, 128});
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
      location, lhsRange.getLogicalStart(), *logicalEnd, blockK.getResult(),
      ValueRange{contract.getAccumulator()},
      [&](OpBuilder &nested, Location nestedLocation, Value kStart,
          ValueRange carries) {
        Value lhsK = nested.create<MakeRangeOp>(
            nestedLocation, lhsIndexType, kStart, blockK.getResult(), one,
            lhsRange.getLogicalStart(), lhsRange.getLogicalStop(),
            lhsMap->getSourceId(), lhsMap->getSourceAxis(),
            lhsMap->getDerived());
        inheritRangeAuthority(lhsK, lhsRange);
        Value rhsOffset = binary(
            nested, nestedLocation, nested.getIndexType(), kStart,
            lhsRange.getLogicalStart(), BinaryOperator::Subtract);
        Value rhsStart = binary(
            nested, nestedLocation, nested.getIndexType(),
            rhsRange.getLogicalStart(), rhsOffset, BinaryOperator::Add);
        Value rhsK = nested.create<MakeRangeOp>(
            nestedLocation, rhsIndexType, rhsStart, blockK.getResult(), one,
            rhsRange.getLogicalStart(), rhsRange.getLogicalStop(),
            rhsMap->getSourceId(), rhsMap->getSourceAxis(),
            rhsMap->getDerived());
        inheritRangeAuthority(rhsK, rhsRange);
        IRMapping lhsReplay;
        IRMapping rhsReplay;
        for (MakeRangeOp range : lhsRanges)
          lhsReplay.map(range.getResult(), lhsK);
        for (MakeRangeOp range : rhsRanges)
          rhsReplay.map(range.getResult(), rhsK);
        FailureOr<Value> lhs = replaySourceValue(
            nested, nestedLocation, contract.getLhs(), unitK, lhsRanges, lhsK,
            lhsReplay);
        FailureOr<Value> rhs = replaySourceValue(
            nested, nestedLocation, contract.getRhs(), unitK, rhsRanges, rhsK,
            rhsReplay);
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

LogicalResult realizeSparseReductionTraversal(SparseContractOp contract,
                                              func::FuncOp kernel) {
  if (contract.getLhsReductionAxes() != ArrayRef<int64_t>{1} ||
      contract.getRhsReductionAxes() != ArrayRef<int64_t>{0} ||
      !contract.getLhsBatchAxes().empty() ||
      !contract.getRhsBatchAxes().empty() ||
      contract.getFormat().getCompressionAxis() != 1)
    return contract.emitOpError(
        "shared sparse blocking requires rank-two [M,KC] x [K,N] physical axes");
  auto compressedType = dyn_cast<FragmentType>(contract.getCompressed().getType());
  auto rhsType = dyn_cast<FragmentType>(contract.getRhs().getType());
  auto resultType = dyn_cast<FragmentType>(contract.getResult().getType());
  if (!compressedType || !rhsType || !resultType ||
      compressedType.getShape().size() != 2 || rhsType.getShape().size() != 2 ||
      resultType.getShape().size() != 2 ||
      llvm::any_of(resultType.getShape(), [](Attribute extent) {
        return !isCompileTimeExtent(cast<PhysicalExprAttr>(extent));
      }))
    return contract.emitOpError(
        "shared sparse blocking requires selected rank-two free-axis fragments");

  FailureOr<AxisMapAttr> compressedMap = queryAxisMap(compressedType, 1);
  FailureOr<AxisMapAttr> denseMap = queryAxisMap(rhsType, 0);
  if (failed(compressedMap) || failed(denseMap))
    return contract.emitOpError(
        "shared sparse blocking lost compressed/dense reduction provenance");

  SmallVector<Value> metadataComponents;
  RecordType metadataRecordType;
  MakeRecordOp metadataRecord;
  if (auto component = dyn_cast<FragmentType>(contract.getMetadata().getType())) {
    (void)component;
    metadataComponents.push_back(contract.getMetadata());
  } else {
    metadataRecordType = dyn_cast<RecordType>(contract.getMetadata().getType());
    metadataRecord = contract.getMetadata().getDefiningOp<MakeRecordOp>();
    if (!metadataRecordType || !metadataRecord ||
        metadataRecord.getFields().size() !=
            metadataRecordType.getFieldTypes().size())
      return contract.emitOpError(
          "shared sparse blocking requires materialized metadata components");
    metadataComponents.append(metadataRecord.getFields().begin(),
                              metadataRecord.getFields().end());
  }
  if (metadataComponents.empty())
    return contract.emitOpError(
        "shared sparse blocking requires at least one metadata component");
  auto metadataComponentType =
      dyn_cast<FragmentType>(metadataComponents.front().getType());
  if (!metadataComponentType || metadataComponentType.getShape().size() != 2)
    return contract.emitOpError(
        "shared sparse blocking requires rank-two metadata fragments");
  FailureOr<AxisMapAttr> metadataMap = queryAxisMap(metadataComponentType, 1);
  if (failed(metadataMap))
    return contract.emitOpError(
        "shared sparse blocking lost metadata-group provenance");
  for (Value component : llvm::drop_begin(metadataComponents)) {
    auto type = dyn_cast<FragmentType>(component.getType());
    FailureOr<AxisMapAttr> mapping =
        type && type.getShape().size() == 2
            ? queryAxisMap(type, 1)
            : FailureOr<AxisMapAttr>(failure());
    if (!type || failed(mapping) ||
        !(sourceAxisIdentity(*mapping) == sourceAxisIdentity(*metadataMap)))
      return contract.emitOpError(
          "shared sparse metadata components do not traverse one group relation");
  }

  PhysicalProgramAnalysis physicalAnalysis(kernel);
  auto rangesFor = [&](Value value, unsigned fragmentAxis, AxisMapAttr mapping,
                       SmallVectorImpl<MakeRangeOp> &ranges) {
    PhysicalRangeFact fact = physicalAnalysis.axisRanges(value, fragmentAxis);
    if (failed(queryExactLogicalRange(fact)) &&
        queryFragmentAxes(value.getType(), sourceAxisIdentity(mapping)).size() ==
            1)
      fact = physicalAnalysis.sourceRanges(value, sourceAxisIdentity(mapping));
    if (failed(queryExactLogicalRange(fact)) || !fact.unitStep)
      return failure();
    ranges.assign(fact.roots.begin(), fact.roots.end());
    return llvm::all_of(ranges, [&](MakeRangeOp range) {
             return sourceAxisIdentity(range) == sourceAxisIdentity(mapping);
           })
               ? success()
               : failure();
  };
  SmallVector<MakeRangeOp> compressedRanges;
  SmallVector<MakeRangeOp> denseRanges;
  SmallVector<MakeRangeOp> metadataRanges;
  if (failed(rangesFor(contract.getCompressed(), 1, *compressedMap,
                       compressedRanges)) ||
      failed(rangesFor(contract.getRhs(), 0, *denseMap, denseRanges)))
    return contract.emitOpError(
        "shared sparse blocking requires explicit compressed and dense ranges");
  for (Value component : metadataComponents) {
    SmallVector<MakeRangeOp> componentRanges;
    if (failed(rangesFor(component, 1, *metadataMap, componentRanges)))
      return contract.emitOpError(
          "shared sparse blocking requires an explicit metadata-group range");
    for (MakeRangeOp range : componentRanges)
      if (!llvm::is_contained(metadataRanges, range))
        metadataRanges.push_back(range);
  }

  MakeRangeOp compressedRange = compressedRanges.front();
  MakeRangeOp denseRange = denseRanges.front();
  MakeRangeOp metadataRange = metadataRanges.front();
  if (llvm::any_of(metadataRanges, [&](MakeRangeOp range) {
        return !sameLogicalRange(range, metadataRange);
      }))
    return contract.emitOpError(
        "shared sparse metadata components have different logical group ranges");
  FailureOr<Value> denseLogicalEnd = resolveLogicalRangeEnd(kernel, denseRange);
  FailureOr<Value> compressedStep = scalarSource(compressedRange.getStep());
  FailureOr<Value> denseStep = scalarSource(denseRange.getStep());
  FailureOr<Value> metadataStep = scalarSource(metadataRange.getStep());
  if (failed(denseLogicalEnd) || failed(compressedStep) || failed(denseStep) ||
      failed(metadataStep) || !isIntegerConstant(*compressedStep, 1) ||
      !isIntegerConstant(*denseStep, 1) ||
      !isIntegerConstant(*metadataStep, 1))
    return contract.emitOpError(
        "shared sparse blocking requires unit-step ranges with an exact dense K end");

  MLIRContext *context = kernel.getContext();
  const int64_t groupSize = contract.getFormat().getKind() == 0 ? 2 : 4;
  std::string suffix =
      ("_sparse_" + Twine(compressedMap->getSourceId()) + "_" +
       Twine(denseMap->getSourceId()))
          .str();
  ParameterOp blockK = getOrCreatePhysicalParameter(
      kernel, "BLOCK_K" + suffix, ParameterRole::Reduction,
      ParameterCategory::Contraction,
      std::max(compressedType.getElementType().getIntOrFloatBitWidth(),
               rhsType.getElementType().getIntOrFloatBitWidth()),
      {32, 64, 128});
  if (!blockK)
    return failure();
  if (FailureOr<int64_t> dimension = queryRangeDimension(denseRange);
      succeeded(dimension))
    blockK->setAttr(
        dimensionAttr,
        IntegerAttr::get(IntegerType::get(context, 64), *dimension));

  PhysicalExprAttr unitDenseK = parameterExpression(
      context, blockK.getParameter().getName().getValue());
  PhysicalExprAttr two = expression(context, PhysicalExprKind::Constant, 2);
  PhysicalExprAttr group =
      expression(context, PhysicalExprKind::Constant, groupSize);
  PhysicalExprAttr unitCompressedK = binaryExpression(
      context, PhysicalExprKind::FloorDiv, unitDenseK, two);
  PhysicalExprAttr unitMetadataK = binaryExpression(
      context, PhysicalExprKind::FloorDiv, unitDenseK, group);
  FragmentType compressedIndexType = fragmentType(
      context, IndexType::get(context), {unitCompressedK}, {*compressedMap},
      compressedType.getOwner());
  FragmentType denseIndexType = fragmentType(
      context, IndexType::get(context), {unitDenseK}, {*denseMap},
      rhsType.getOwner());
  FragmentType metadataIndexType = fragmentType(
      context, IndexType::get(context), {unitMetadataK}, {*metadataMap},
      metadataComponentType.getOwner());

  Location location = contract.getLoc();
  OpBuilder builder(contract);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value twoValue = builder.create<arith::ConstantIndexOp>(location, 2);
  Value groupValue =
      builder.create<arith::ConstantIndexOp>(location, groupSize);
  Value compressedExtent = builder.create<PhysicalExprOp>(
      location, builder.getIndexType(), unitCompressedK);
  Value metadataExtent = builder.create<PhysicalExprOp>(
      location, builder.getIndexType(), unitMetadataK);
  bool bodyFailed = false;
  auto loop = builder.create<scf::ForOp>(
      location, denseRange.getLogicalStart(), *denseLogicalEnd,
      blockK.getResult(), ValueRange{contract.getAccumulator()},
      [&](OpBuilder &nested, Location nestedLocation, Value denseStart,
          ValueRange carries) {
        Value denseOffset = binary(
            nested, nestedLocation, nested.getIndexType(), denseStart,
            denseRange.getLogicalStart(), BinaryOperator::Subtract);
        Value compressedStart = binary(
            nested, nestedLocation, nested.getIndexType(),
            compressedRange.getLogicalStart(),
            binary(nested, nestedLocation, nested.getIndexType(), denseOffset,
                   twoValue, BinaryOperator::FloorDivide),
            BinaryOperator::Add);
        Value metadataStart = binary(
            nested, nestedLocation, nested.getIndexType(),
            metadataRange.getLogicalStart(),
            binary(nested, nestedLocation, nested.getIndexType(), denseOffset,
                   groupValue, BinaryOperator::FloorDivide),
            BinaryOperator::Add);
        Value compressedK = nested.create<MakeRangeOp>(
            nestedLocation, compressedIndexType, compressedStart,
            compressedExtent, one, compressedRange.getLogicalStart(),
            compressedRange.getLogicalStop(), compressedMap->getSourceId(),
            compressedMap->getSourceAxis(), compressedMap->getDerived());
        inheritRangeAuthority(compressedK, compressedRange);
        Value denseK = nested.create<MakeRangeOp>(
            nestedLocation, denseIndexType, denseStart, blockK.getResult(), one,
            denseRange.getLogicalStart(), denseRange.getLogicalStop(),
            denseMap->getSourceId(), denseMap->getSourceAxis(),
            denseMap->getDerived());
        inheritRangeAuthority(denseK, denseRange);
        Value metadataK = nested.create<MakeRangeOp>(
            nestedLocation, metadataIndexType, metadataStart, metadataExtent,
            one, metadataRange.getLogicalStart(), metadataRange.getLogicalStop(),
            metadataMap->getSourceId(), metadataMap->getSourceAxis(),
            metadataMap->getDerived());
        inheritRangeAuthority(metadataK, metadataRange);

        FailureOr<Value> compressedTail = buildRangeTailPredicate(
            nested, nestedLocation, compressedK, compressedRange);
        FailureOr<Value> denseTail = buildRangeTailPredicate(
            nested, nestedLocation, denseK, denseRange);
        FailureOr<Value> metadataTail = buildRangeTailPredicate(
            nested, nestedLocation, metadataK, metadataRange);
        if (failed(compressedTail) || failed(denseTail) ||
            failed(metadataTail)) {
          bodyFailed = true;
          return;
        }

        IRMapping compressedReplay;
        for (MakeRangeOp range : compressedRanges)
          compressedReplay.map(range.getResult(), compressedK);
        IRMapping denseReplay;
        for (MakeRangeOp range : denseRanges)
          denseReplay.map(range.getResult(), denseK);
        IRMapping metadataReplay;
        for (MakeRangeOp range : metadataRanges)
          metadataReplay.map(range.getResult(), metadataK);
        FailureOr<Value> compressed = replaySourceValue(
            nested, nestedLocation, contract.getCompressed(), unitCompressedK,
            compressedRanges, compressedK, compressedReplay);
        FailureOr<Value> rhs = replaySourceValue(
            nested, nestedLocation, contract.getRhs(), unitDenseK, denseRanges,
            denseK, denseReplay);
        SmallVector<Value> replayedMetadata;
        for (Value component : metadataComponents) {
          FailureOr<Value> replayed = replaySourceValue(
              nested, nestedLocation, component, unitMetadataK, metadataRanges,
              metadataK, metadataReplay);
          if (failed(replayed)) {
            bodyFailed = true;
            return;
          }
          replayedMetadata.push_back(*replayed);
        }
        if (failed(compressed) || failed(rhs)) {
          bodyFailed = true;
          return;
        }
        if (failed(appendTailValidity(nestedLocation, contract.getCompressed(),
                                      *compressedTail,
                                      compressedReplay)) ||
            failed(appendTailValidity(nestedLocation, contract.getRhs(),
                                      *denseTail, denseReplay)) ||
            failed(appendTailValidity(nestedLocation, contract.getMetadata(),
                                      *metadataTail,
                                      metadataReplay))) {
          bodyFailed = true;
          return;
        }
        Value metadata = replayedMetadata.front();
        if (metadataRecordType) {
          SmallVector<Attribute> fieldTypes;
          for (Value component : replayedMetadata)
            fieldTypes.push_back(TypeAttr::get(component.getType()));
          auto blockedRecordType = RecordType::get(
              context, metadataRecordType.getFieldNames(),
              ArrayAttr::get(context, fieldTypes), metadataRecordType.getOwner());
          metadata = nested.create<MakeRecordOp>(nestedLocation,
                                                 blockedRecordType,
                                                 replayedMetadata);
        }
        auto product = nested.create<SparseContractOp>(
            nestedLocation, resultType, *compressed, metadata, *rhs,
            contract.getLogicalExtent(), carries.front(), contract.getFormat(),
            contract.getLhsReductionAxes(), contract.getRhsReductionAxes(),
            contract.getLhsBatchAxes(), contract.getRhsBatchAxes());
        if (Attribute origin = contract->getAttr(originAttr))
          product->setAttr(originAttr, origin);
        nested.create<scf::YieldOp>(nestedLocation, product.getResult());
      });
  if (bodyFailed) {
    loop.erase();
    return contract.emitOpError(
        "shared sparse blocking could not replay its compressed/metadata/dense graphs");
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
    if (!required)
      return success();
    return contract.emitOpError()
           << "cannot form a complete physical contraction: " << reason
           << "; lhs=" << contract.getLhs().getType()
           << "; rhs=" << contract.getRhs().getType()
           << "; result=" << contract.getResult().getType()
           << "; lhs_reduction=" << contract.getLhsReductionAxes()
           << "; rhs_reduction=" << contract.getRhsReductionAxes()
           << "; lhs_batch=" << contract.getLhsBatchAxes()
           << "; rhs_batch=" << contract.getRhsBatchAxes()
           << "; free_axes_need_realization="
           << freeAxesNeedRealization(contract, kernel)
           << "; selected_free_axes=" << hasSelectedFreeAxes(contract);
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

  FailureOr<unsigned> lhsRowCoordinate =
      accessCoordinatePosition(lhsLoad, *rowMap);
  FailureOr<unsigned> lhsReductionCoordinate =
      accessCoordinatePosition(lhsLoad, *lhsReductionMap);
  FailureOr<unsigned> rhsReductionCoordinate =
      accessCoordinatePosition(rhsLoad, *rhsReductionMap);
  FailureOr<unsigned> rhsColumnCoordinate =
      accessCoordinatePosition(rhsLoad, *columnMap);
  if (failed(lhsRowCoordinate) || failed(lhsReductionCoordinate) ||
      failed(rhsReductionCoordinate) || failed(rhsColumnCoordinate)) {
    std::string details;
    llvm::raw_string_ostream stream(details);
    stream << "load coordinates do not cover all free/reduction axes"
           << "; lhs_row=" << succeeded(lhsRowCoordinate)
           << ", lhs_reduction=" << succeeded(lhsReductionCoordinate)
           << ", rhs_reduction=" << succeeded(rhsReductionCoordinate)
           << ", rhs_column=" << succeeded(rhsColumnCoordinate)
           << ", row_map=" << *rowMap << ", lhs_coordinate_types=[";
    llvm::interleaveComma(lhsLoad.getCoordinates(), stream,
                          [&](Value coordinate) { stream << coordinate.getType(); });
    stream << "]";
    return unhandled(stream.str());
  }
  Value originalLhsRowCoordinate = lhsLoad.getCoordinates()[*lhsRowCoordinate];
  auto rowRange = sourceRange(originalLhsRowCoordinate);
  const bool indirectRow = !rowRange;
  if (!rowRange) {
    FailureOr<MakeRangeOp> discovered =
        producerRange(originalLhsRowCoordinate,
                      sourceAxisIdentity(*rowMap));
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
  const bool persistentRowTraversal = runtimeRowTraversal && !indirectRow;
  const ParameterCategory contractionCategory =
      persistentRowTraversal ? ParameterCategory::PersistentContraction
                             : ParameterCategory::Contraction;

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
              sourceAxisIdentity(*rowMap)))
        return;
      FailureOr<MakeRangeOp> root =
          producerRange(assumption.getIndex(),
                        sourceAxisIdentity(*rowMap));
      if (succeeded(root) && sameLogicalRange(*root, rowRange))
        rowAssumptions.push_back(assumption);
    });

  DelinearizeOp mapping;
  DominanceInfo dominance(kernel);
  kernel.walk([&](DelinearizeOp candidate) {
    if (!candidate.getLinear().getDefiningOp<ProgramIdOp>() ||
        !dominance.dominates(candidate.getOperation(), contract.getOperation()))
      return;
    if (mapping &&
        dominance.dominates(mapping.getOperation(), candidate.getOperation()))
      mapping = candidate;
    else if (!mapping)
      mapping = candidate;
  });
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
  ParameterOp blockM = getOrCreatePhysicalParameter(
      kernel, "BLOCK_M" + suffix, ParameterRole::OwnershipM,
      contractionCategory,
      lhsType.getElementType().getIntOrFloatBitWidth(),
      {32, 64, 128, 256});
  ParameterOp blockN = getOrCreatePhysicalParameter(
      kernel, "BLOCK_N" + suffix, ParameterRole::OwnershipN,
      contractionCategory,
      rhsType.getElementType().getIntOrFloatBitWidth(),
      {32, 64, 128, 256});
  ParameterOp blockK = getOrCreatePhysicalParameter(
      kernel, "BLOCK_K" + suffix, ParameterRole::Reduction,
      contractionCategory,
      std::max(lhsType.getElementType().getIntOrFloatBitWidth(),
               rhsType.getElementType().getIntOrFloatBitWidth()),
      {32, 64, 128});
  if (!blockM || !blockN || !blockK)
    return failure();
  FailureOr<unsigned> existingRowAxis =
      mappingAxisForScalar(rowRange.getStart(), mapping);
  FailureOr<unsigned> existingColumnAxis =
      mappingAxisForScalar(columnRange.getStart(), mapping);
  if (succeeded(existingRowAxis) && succeeded(existingColumnAxis) &&
      *existingRowAxis == *existingColumnAxis)
    return unhandled(
        "row and column ownership share one mapping coordinate that cannot be refined independently");
  if (failed(refineOwnershipParameter(kernel, rowRange, existingRowAxis,
                                      blockM)) ||
      failed(refineOwnershipParameter(kernel, columnRange, existingColumnAxis,
                                      blockN)))
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
  if (runtimeRowTraversal) {
    SmallVector<int64_t, 5> rowWorkerCandidates = {1, 2, 4, 8};
    if (persistentRowTraversal)
      rowWorkerCandidates.push_back(16);
    rowWorkers = getOrCreatePhysicalParameter(
        kernel, "ROW_WORKERS" + suffix, ParameterRole::TraversalWorkers,
        contractionCategory,
        lhsType.getElementType().getIntOrFloatBitWidth(),
        rowWorkerCandidates);
  }
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
  SmallVector<Type> mappingTypes(mapping.getResultTypes());
  SmallVector<int64_t> coordinateRoles(mapping.getNumResults(), -1);
  if (auto existing =
          mapping->getAttrOfType<DenseI64ArrayAttr>(coordinateRolesAttr))
    if (existing.size() == mapping.getNumResults())
      llvm::copy(existing.asArrayRef(), coordinateRoles.begin());
  auto bindMappingAxis = [&](FailureOr<unsigned> existing, Value extent,
                             PhysicalExprAttr launchExtent,
                             CoordinateRole role) {
    unsigned axis;
    if (succeeded(existing)) {
      axis = *existing;
      mappingExtents[axis] = extent;
      launchExtents[axis] = launchExtent;
    } else {
      axis = mappingExtents.size();
      mappingExtents.push_back(extent);
      launchExtents.push_back(launchExtent);
      mappingTypes.push_back(mapBuilder.getIndexType());
      coordinateRoles.push_back(-1);
    }
    coordinateRoles[axis] = static_cast<int64_t>(role);
    return axis;
  };
  unsigned rowMappingAxis = bindMappingAxis(
      existingRowAxis, runtimeRowTraversal ? rowWorkers.getResult() : rowTiles,
      runtimeRowTraversal
          ? unitRowWorkers
          : binaryExpression(context, PhysicalExprKind::CeilDiv, rowExpression,
                             unitM),
      indirectRow ? CoordinateRole::IndirectTraversal
                  : runtimeRowTraversal ? CoordinateRole::TraversalWorker
                                        : CoordinateRole::ContractionM);
  unsigned columnMappingAxis = bindMappingAxis(
      existingColumnAxis, columnTiles,
      binaryExpression(context, PhysicalExprKind::CeilDiv, columnExpression,
                       unitN),
      CoordinateRole::ContractionN);
  auto expandedMapping = mapBuilder.create<DelinearizeOp>(
      location, mappingTypes, mapping.getLinear(), mappingExtents,
      mapBuilder.getArrayAttr(launchExtents));
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
  Value rowTile = runtimeRowTraversal
                      ? Value()
                      : expandedMapping.getCoordinates()[rowMappingAxis];
  Value rowWorker = runtimeRowTraversal
                        ? expandedMapping.getCoordinates()[rowMappingAxis]
                        : Value();
  Value columnTile = expandedMapping.getCoordinates()[columnMappingAxis];
  mapping.erase();
  kernel->setAttr(programSpaceAttr,
                  ArrayAttr::get(context, {segmentLength}));
  kernel->setAttr(gridRankAttr,
                  IntegerAttr::get(IntegerType::get(context, 64), 1));

  OpBuilder builder(contract);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value rowStop = *rowLogicalEnd;
  Value columnStart = columnRange.getStart();
  if (failed(existingColumnAxis))
    columnStart = binary(
        builder, location, builder.getIndexType(), columnStart,
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
      columnRange.getLogicalStart(), columnRange.getLogicalStop(),
      columnMap->getSourceId(), columnMap->getSourceAxis(),
      columnMap->getDerived());
  inheritRangeAuthority(columns, columnRange);
  Value columnEnd = broadcast(builder, location, columnIndexType, columnStop);
  Value columnValid =
      compare(builder, location, columnPredicateType, columns, columnEnd,
              ComparePredicate::Lt);
  auto emitRowBlock = [&](OpBuilder &rowBuilder,
                          Value rowStart) -> LogicalResult {
    Value rows = rowBuilder.create<MakeRangeOp>(
        location, rowIndexType, rowStart, blockM.getResult(), one,
        rowRange.getLogicalStart(), rowRange.getLogicalStop(),
        rowMap->getSourceId(), rowMap->getSourceAxis(), rowMap->getDerived());
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
          sourceAxisIdentity(*rowMap),
          unitM, rowRange, rows, replay);
      if (failed(rowCoordinate))
        return contract.emitOpError(
            "blocked contraction could not replay its row coordinate graph");
      blockedLhsRowCoordinate = *rowCoordinate;
      for (AssumeInBoundsOp assumption : rowAssumptions) {
        FailureOr<Value> index = replaySourceValue(
            rowBuilder, location, assumption.getIndex(),
            sourceAxisIdentity(*rowMap),
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
              sourceAxisIdentity(*rowMap),
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
              lhsReductionRange.getLogicalStart(),
              lhsReductionRange.getLogicalStop(),
              lhsReductionMap->getSourceId(), lhsReductionMap->getSourceAxis(),
              lhsReductionMap->getDerived());
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
          FailureOr<Value> retargetedLhs = materializeRetargetedValidity(
              nested, nestedLocation, lhsLoad.getValid(), lhsTailRanges,
              lhsValid, lhsPredicateType);
          FailureOr<Value> retargetedRhs = materializeRetargetedValidity(
              nested, nestedLocation, rhsLoad.getValid(), rhsTailRanges,
              rhsValid, rhsPredicateType);
          if (failed(retargetedLhs) || failed(retargetedRhs)) {
            loopBodyFailure = failed(retargetedLhs)
                                  ? "lhs residual validity could not be retargeted"
                                  : "rhs residual validity could not be retargeted";
            return;
          }
          lhsValid = *retargetedLhs;
          rhsValid = *retargetedRhs;
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
          sourceAxisIdentity(*columnMap));
      if (failed(storeColumn))
        storeColumn = directRankOneAccessPosition(
            path.store.getCoordinates(), path.store.getValue().getType(), 1);
      FailureOr<unsigned> storeRow;
      if (!indirectRow) {
        storeRow = queryCoordinatePosition(
            path.store.getCoordinates(),
            sourceAxisIdentity(*rowMap));
        if (failed(storeRow))
          storeRow = directRankOneAccessPosition(
              path.store.getCoordinates(), path.store.getValue().getType(), 0);
      }
      if ((!indirectRow && failed(storeRow)) || failed(storeColumn))
        return path.store.emitOpError(
            "blocked contract output lost its logical source coordinates");
      if (!indirectRow)
        coordinates[*storeRow] = rows;
      coordinates[*storeColumn] = columns;
      FailureOr<Value> valid = materializeRetargetedValidity(
          rowBuilder, location, path.store.getValid(), outputTailRanges,
          outputValid, outputPredicateType);
      if (failed(valid))
        return path.store.emitOpError(
            "blocked contract output has non-scalar residual validity");
      auto replacement = rowBuilder.create<StoreOp>(
          location, path.store.getResource(), coordinates, output, *valid,
          path.store.getSourceAxes());
      if (Attribute origin = path.store->getAttr(originAttr))
        replacement->setAttr(originAttr, origin);
    }
    return success();
  };

  if (runtimeRowTraversal) {
    bool rowBodyFailed = false;
    Value rowStart = rowRange.getStart();
    if (failed(existingRowAxis))
      rowStart = binary(
          builder, location, builder.getIndexType(), rowStart,
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
    Value rowStart = rowRange.getStart();
    if (failed(existingRowAxis))
      rowStart = binary(
          builder, location, builder.getIndexType(), rowStart,
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
  PhysicalProgramAnalysis physicalAnalysis(kernel);
  auto sourceLoad = [&](Value value) -> LoadOp {
    PhysicalReplayFact fact = physicalAnalysis.replayability(
        value, std::nullopt, PhysicalReplayScope::ValueGraph,
        /*allowAccesses=*/true);
    return fact.isReplayable() && fact.accesses.size() == 1
               ? dyn_cast<LoadOp>(fact.accesses.front())
               : LoadOp();
  };
  auto lhsLoad = sourceLoad(contract.getLhs());
  auto lhsScaleLoad = sourceLoad(contract.getLhsScale());
  auto rhsLoad = sourceLoad(contract.getRhs());
  auto rhsScaleLoad = sourceLoad(contract.getRhsScale());
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
  bool fixedAxes = lhsReduction.size() == 2 && rhsReduction.size() == 2 &&
                   lhsReduction[0] == 1 && lhsReduction[1] == 2 &&
                   rhsReduction[0] == 0 && rhsReduction[1] == 1 &&
                   contract.getLhsBatchAxes().empty() &&
                   contract.getRhsBatchAxes().empty();
  if (!lhsLoad || !lhsScaleLoad || !rhsLoad || !rhsScaleLoad ||
      lhsType.getShape().size() != 3 || lhsScaleType.getShape().size() != 2 ||
      rhsType.getShape().size() != 3 || rhsScaleType.getShape().size() != 2 ||
      resultType.getShape().size() != 2 || !fixedAxes)
    return reject("requires one replayable source load for every operand and the closed [M,G,C]/[M,G] x [G,C,N]/[N,G] schema");
  if (contract.getLhsGroupSize() <= 0 ||
      contract.getLhsGroupSize() != contract.getRhsGroupSize())
    return reject("requires one equal positive scale-group size");
  constexpr unsigned lhsFree = 0;
  constexpr unsigned lhsBlockAxis = 1;
  constexpr unsigned lhsInnerAxis = 2;
  constexpr unsigned rhsBlockAxis = 0;
  constexpr unsigned rhsInnerAxis = 1;
  constexpr unsigned rhsFree = 2;
  auto lhsInnerExtent =
      cast<PhysicalExprAttr>(lhsType.getShape()[lhsInnerAxis]);
  auto rhsInnerExtent =
      cast<PhysicalExprAttr>(rhsType.getShape()[rhsInnerAxis]);

  FailureOr<AxisMapAttr> rowMap = queryAxisMap(lhsType, lhsFree);
  FailureOr<AxisMapAttr> lhsBlockMap = queryAxisMap(lhsType, lhsBlockAxis);
  FailureOr<AxisMapAttr> lhsInnerMap = queryAxisMap(lhsType, lhsInnerAxis);
  FailureOr<AxisMapAttr> rhsBlockMap = queryAxisMap(rhsType, rhsBlockAxis);
  FailureOr<AxisMapAttr> rhsInnerMap = queryAxisMap(rhsType, rhsInnerAxis);
  FailureOr<AxisMapAttr> columnMap = queryAxisMap(rhsType, rhsFree);
  auto scaleMapFor = [&](FragmentType scale,
                         AxisMapAttr data) -> FailureOr<AxisMapAttr> {
    PhysicalAxisProjection projection = queryFragmentAxis(
        scale, sourceAxisIdentity(data));
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
    return sourceAxisIdentity(lhs) == sourceAxisIdentity(rhs);
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
        sourceAxisIdentity(source));
    if (failed(coordinate))
      return failure();
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact fact = analysis.sourceRanges(
        load.getCoordinates()[*coordinate],
        sourceAxisIdentity(source));
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
        sourceAxisIdentity(mapping));
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
  ParameterOp blockM = getOrCreatePhysicalParameter(
      kernel, "BLOCK_M" + suffix, ParameterRole::OwnershipM,
      ParameterCategory::Contraction,
      lhsType.getElementType().getIntOrFloatBitWidth(), {64, 128});
  ParameterOp blockN = getOrCreatePhysicalParameter(
      kernel, "BLOCK_N" + suffix, ParameterRole::OwnershipN,
      ParameterCategory::Contraction,
      rhsType.getElementType().getIntOrFloatBitWidth(), {64, 128});
  ParameterOp blockK = getOrCreatePhysicalParameter(
      kernel, "BLOCK_K_GROUPS" + suffix, ParameterRole::Reduction,
      ParameterCategory::Contraction,
      std::max(lhsType.getElementType().getIntOrFloatBitWidth(),
               rhsType.getElementType().getIntOrFloatBitWidth()),
      {2, 4, 8});
  if (!blockM || !blockN || !blockK)
    return failure();
  FailureOr<unsigned> existingRowAxis =
      mappingAxisForScalar(rowRange->getStart(), mapping);
  FailureOr<unsigned> existingColumnAxis =
      mappingAxisForScalar(columnRange->getStart(), mapping);
  if (succeeded(existingRowAxis) && succeeded(existingColumnAxis) &&
      *existingRowAxis == *existingColumnAxis)
    return reject(
        "row and column ownership share one mapping coordinate that cannot be refined independently");
  if (failed(refineOwnershipParameter(kernel, *rowRange, existingRowAxis,
                                      blockM)) ||
      failed(refineOwnershipParameter(kernel, *columnRange, existingColumnAxis,
                                      blockN)))
    return failure();
  for (auto [parameter, range] :
       {std::pair<ParameterOp, MakeRangeOp>{blockM, *rowRange},
        std::pair<ParameterOp, MakeRangeOp>{blockN, *columnRange},
        std::pair<ParameterOp, MakeRangeOp>{blockK, *blockRange}})
    if (FailureOr<int64_t> dimension = queryRangeDimension(range);
        succeeded(dimension))
      parameter->setAttr(
          dimensionAttr,
          IntegerAttr::get(IntegerType::get(kernel.getContext(), 64),
                           *dimension));
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
  SmallVector<Type> mappingTypes(mapping.getResultTypes());
  SmallVector<int64_t> coordinateRoles(mapping.getNumResults(), -1);
  if (auto existing =
          mapping->getAttrOfType<DenseI64ArrayAttr>(coordinateRolesAttr))
    if (existing.size() == mapping.getNumResults())
      llvm::copy(existing.asArrayRef(), coordinateRoles.begin());
  auto bindMappingAxis = [&](FailureOr<unsigned> existing, Value extent,
                             PhysicalExprAttr launchExtent,
                             CoordinateRole role) {
    unsigned axis;
    if (succeeded(existing)) {
      axis = *existing;
      mappingExtents[axis] = extent;
      launchExtents[axis] = launchExtent;
    } else {
      axis = mappingExtents.size();
      mappingExtents.push_back(extent);
      launchExtents.push_back(launchExtent);
      mappingTypes.push_back(mapBuilder.getIndexType());
      coordinateRoles.push_back(-1);
    }
    coordinateRoles[axis] = static_cast<int64_t>(role);
    return axis;
  };
  unsigned rowMappingAxis = bindMappingAxis(
      existingRowAxis, rowTiles,
      binaryExpression(context, PhysicalExprKind::CeilDiv, rowExpression,
                       unitM),
      CoordinateRole::ContractionM);
  unsigned columnMappingAxis = bindMappingAxis(
      existingColumnAxis, columnTiles,
      binaryExpression(context, PhysicalExprKind::CeilDiv, columnExpression,
                       unitN),
      CoordinateRole::ContractionN);
  auto expandedMapping = mapBuilder.create<DelinearizeOp>(
      location, mappingTypes, mapping.getLinear(), mappingExtents,
      mapBuilder.getArrayAttr(launchExtents));
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
  Value rowTile = expandedMapping.getCoordinates()[rowMappingAxis];
  Value columnTile = expandedMapping.getCoordinates()[columnMappingAxis];
  mapping.erase();
  kernel->setAttr(programSpaceAttr,
                  ArrayAttr::get(context, {segmentLength}));

  OpBuilder builder(contract);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value rowStart = rowRange->getStart();
  if (failed(existingRowAxis))
    rowStart = binary(
        builder, location, builder.getIndexType(), rowStart,
        binary(builder, location, builder.getIndexType(), rowTile,
               blockM.getResult(), BinaryOperator::Multiply),
        BinaryOperator::Add);
  Value rowStop = binary(builder, location, builder.getIndexType(),
                         rowRange->getStart(), rowRange->getExtent(),
                         BinaryOperator::Add);
  Value columnStart = columnRange->getStart();
  if (failed(existingColumnAxis))
    columnStart = binary(
        builder, location, builder.getIndexType(), columnStart,
        binary(builder, location, builder.getIndexType(), columnTile,
               blockN.getResult(), BinaryOperator::Multiply),
        BinaryOperator::Add);
  Value columnStop = binary(builder, location, builder.getIndexType(),
                            columnRange->getStart(), columnRange->getExtent(),
                            BinaryOperator::Add);
  Value blockStop = *blockLogicalEnd;

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
  FragmentType blockedLhsType = fragmentType(
      context, lhsType.getElementType(), {unitM, unitK, lhsInnerExtent},
      {*rowMap, *lhsBlockMap, *lhsInnerMap}, lhsType.getOwner());
  FragmentType blockedLhsScaleType = fragmentType(
      context, lhsScaleType.getElementType(), {unitM, unitK},
      {*lhsScaleRowMap, *lhsScaleBlockMap}, lhsScaleType.getOwner());
  FragmentType blockedRhsType = fragmentType(
      context, rhsType.getElementType(), {unitK, rhsInnerExtent, unitN},
      {*rhsBlockMap, *rhsInnerMap, *columnMap}, rhsType.getOwner());
  FragmentType blockedRhsScaleType = fragmentType(
      context, rhsScaleType.getElementType(), {unitN, unitK},
      {*rhsScaleColumnMap, *rhsScaleBlockMap}, rhsScaleType.getOwner());
  FragmentType blockedResultType = fragmentType(
      context, resultType.getElementType(), {unitM, unitN},
      {*rowMap, *columnMap}, resultType.getOwner());
  FragmentType lhsPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM, unitK, lhsInnerExtent},
      {*rowMap, *lhsBlockMap, *lhsInnerMap}, lhsType.getOwner());
  FragmentType lhsScalePredicateType = fragmentType(
      context, builder.getI1Type(), {unitM, unitK},
      {*lhsScaleRowMap, *lhsScaleBlockMap}, lhsScaleType.getOwner());
  FragmentType rhsPredicateType = fragmentType(
      context, builder.getI1Type(), {unitK, rhsInnerExtent, unitN},
      {*rhsBlockMap, *rhsInnerMap, *columnMap}, rhsType.getOwner());
  FragmentType rhsScalePredicateType = fragmentType(
      context, builder.getI1Type(), {unitN, unitK},
      {*rhsScaleColumnMap, *rhsScaleBlockMap}, rhsScaleType.getOwner());
  FragmentType outputPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM, unitN},
      {*rowMap, *columnMap}, resultType.getOwner());

  Value rows = builder.create<MakeRangeOp>(
      location, rowIndexType, rowStart, blockM.getResult(), one,
      rowRange->getLogicalStart(), rowRange->getLogicalStop(),
      rowMap->getSourceId(), rowMap->getSourceAxis(), rowMap->getDerived());
  inheritRangeAuthority(rows, *rowRange);
  Value columns = builder.create<MakeRangeOp>(
      location, columnIndexType, columnStart, blockN.getResult(), one,
      columnRange->getLogicalStart(), columnRange->getLogicalStop(),
      columnMap->getSourceId(), columnMap->getSourceAxis(),
      columnMap->getDerived());
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
            blockRange->getLogicalStart(), blockRange->getLogicalStop(),
            lhsBlockMap->getSourceId(), lhsBlockMap->getSourceAxis(),
            lhsBlockMap->getDerived());
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

        FailureOr<Value> retargetedLhs = materializeRetargetedValidity(
            nested, nestedLocation, lhsLoad.getValid(), lhsTailRanges,
            lhsValid, lhsPredicateType);
        FailureOr<Value> retargetedLhsScale = materializeRetargetedValidity(
            nested, nestedLocation, lhsScaleLoad.getValid(),
            lhsScaleTailRanges, lhsScaleValid, lhsScalePredicateType);
        FailureOr<Value> retargetedRhs = materializeRetargetedValidity(
            nested, nestedLocation, rhsLoad.getValid(), rhsTailRanges,
            rhsValid, rhsPredicateType);
        FailureOr<Value> retargetedRhsScale = materializeRetargetedValidity(
            nested, nestedLocation, rhsScaleLoad.getValid(),
            rhsScaleTailRanges, rhsScaleValid, rhsScalePredicateType);
        if (failed(retargetedLhs) || failed(retargetedLhsScale) ||
            failed(retargetedRhs) || failed(retargetedRhsScale)) {
          loopBodyFailed = true;
          return;
        }
        lhsValid = *retargetedLhs;
        lhsScaleValid = *retargetedLhsScale;
        rhsValid = *retargetedRhs;
        rhsScaleValid = *retargetedRhsScale;

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
            sourceAxisIdentity(*columnMap),
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
        sourceAxisIdentity(*rowMap));
    FailureOr<unsigned> storeColumn = queryCoordinatePosition(
        path.store.getCoordinates(),
        sourceAxisIdentity(*columnMap));
    if (failed(storeRow) || failed(storeColumn))
      return path.store.emitOpError(
          "blocked scaled-contract output lost source coordinates");
    coordinates[*storeRow] = rows;
    coordinates[*storeColumn] = columns;
    FailureOr<Value> valid = materializeRetargetedValidity(
        builder, location, path.store.getValid(), outputTailRanges,
        outputValid, outputPredicateType);
    if (failed(valid))
      return reject("result store residual validity could not be retargeted");
    auto replacement = builder.create<StoreOp>(
        location, path.store.getResource(), coordinates, output, *valid,
        path.store.getSourceAxes());
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
  SmallVector<SparseContractOp> sparseContracts;
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
      [&](SparseContractOp contract) { sparseContracts.push_back(contract); });
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
  for (SparseContractOp contract : sparseContracts)
    if (!hasFragmentSchema(contract))
      return contract.emitOpError(
          "shared sparse-contraction blocking requires fragment operands, metadata, accumulator, and result");
  for (ContractOp contract : contracts)
    if (requiresPhysicalRealization(contract)) {
      FailureOr<bool> nativeSegment =
          realizeSegmentNativeReduction(contract, kernel);
      if (failed(nativeSegment))
        return failure();
      if (*nativeSegment)
        continue;
      FailureOr<bool> nativeStructured =
          realizeStructuredNativeReduction(contract, kernel);
      if (failed(nativeStructured))
        return failure();
      if (*nativeStructured)
        continue;
      const bool rangeSingleReduction = hasRangeContractForm(contract);
      if (!rangeSingleReduction &&
          contract.getLhsReductionAxes().size() == 1 &&
          contract.getRhsReductionAxes().size() == 1 &&
          contract.getLhsBatchAxes().empty() &&
          contract.getRhsBatchAxes().empty() &&
          !freeAxesNeedRealization(contract, kernel) &&
          hasSelectedFreeAxes(contract) &&
          hasExplicitPairedReductionRanges(contract)) {
        if (failed(realizeReductionTraversal(contract, kernel)))
          return failure();
      } else if (!rangeSingleReduction &&
                 contract.getLhsReductionAxes().size() == 1 &&
                 contract.getRhsReductionAxes().size() == 1 &&
                 contract.getLhsBatchAxes().empty() &&
                 contract.getRhsBatchAxes().empty() &&
                 hasCompleteStorePath(contract) &&
                 freeAxesNeedRealization(contract, kernel)) {
        if (failed(realizeContract(contract, kernel)))
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
  for (SparseContractOp contract : sparseContracts)
    if (failed(realizeSparseReductionTraversal(contract, kernel)))
      return failure();
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
