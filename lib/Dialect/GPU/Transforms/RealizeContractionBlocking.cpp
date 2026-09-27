#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <functional>
#include <limits>
#include <tuple>

using namespace mlir;

namespace intent::gpu {
namespace {

constexpr int64_t contractionReductionCandidates[] = {16, 32, 64, 128, 256};

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

void fuseContractionAdds(func::FuncOp kernel) {
  SmallVector<BinaryOp> additions;
  kernel.walk([&](BinaryOp binary) {
    if (binary.getOperatorKind() == BinaryOperator::Add)
      additions.push_back(binary);
  });
  for (BinaryOp add : additions) {
    for (unsigned operand : {1u, 0u}) {
      Value projected = add->getOperand(operand);
      if (projected.getType() != add.getResult().getType())
        continue;
      Value value =
          stripAdditiveProjection(projected, /*singleUse=*/true);
      auto contract = value ? value.getDefiningOp<ContractOp>() : ContractOp();
      if (!contract || contract->getBlock() != add->getBlock() ||
          !isLiteralZeroProjection(contract.getAccumulator()))
        continue;
      SmallVector<Operation *> projections;
      for (Value current = projected; current != value;) {
        Operation *projection = current.getDefiningOp();
        projections.push_back(projection);
        current = projection->getOperand(0);
      }
      OpBuilder builder(add);
      FailureOr<Value> carry = projectPhysicalValueToSchema(
          builder, add.getLoc(), add->getOperand(1 - operand),
          contract.getResult().getType());
      if (failed(carry))
        continue;
      // Clone only the pure contraction at its consumer. Inputs already
      // dominate this point; memory reads and other effects are not moved.
      auto fused = cast<ContractOp>(builder.clone(*contract));
      fused.getAccumulatorMutable().assign(*carry);
      IRMapping mapping;
      mapping.map(value, fused.getResult());
      Value result = fused.getResult();
      for (Operation *projection : llvm::reverse(projections)) {
        Operation *clone = builder.clone(*projection, mapping);
        result = clone->getResult(0);
        mapping.map(projection->getResult(0), result);
      }
      add.getResult().replaceAllUsesWith(result);
      add.erase();
      break;
    }
  }
  eraseDeadPhysicalValues(kernel);
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

bool fullReductionNeedsTraversal(ContractOp contract);

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
  return fullReductionNeedsTraversal(contract) ||
         hasUnrealizedPhysicalAxis(contract.getLhs()) ||
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
  // Read-dependent coordinates use indirect-row replay, not direct range blocking.
  if (!fact.accesses.empty())
    return {};
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
  bool otherCoordinate = false;
  llvm::SmallPtrSet<Operation *, 16> visited;
  std::function<void(Value)> collect = [&](Value current) {
    for (auto [axis, coordinate] : llvm::enumerate(mapping.getCoordinates()))
      if (current == coordinate) {
        if (roles[axis] == pointwiseOwnership)
          axes.insert(axis);
        else
          otherCoordinate = true;
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
  return !otherCoordinate && axes.size() == 1
             ? FailureOr<unsigned>(*axes.begin())
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
                              coverageDimensionAttr, coverageBoundAttr}) {
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

LogicalResult collectReductionAxisRanges(
    Value value, unsigned axis, SmallVectorImpl<MakeRangeOp> &ranges) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  auto type = dyn_cast<FragmentType>(value.getType());
  if (!kernel || !type || axis >= type.getShape().size())
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.axisRanges(value, axis);
  FailureOr<MakeRangeOp> authority = queryExactLogicalRange(fact);
  if (failed(authority) || !fact.unitStep ||
      !analysis.lockstepRanges(fact.roots).isExact())
    return failure();
  PhysicalRangeAxisFact selected = analysis.rangeAxes(value, fact.roots);
  if (!selected.isExact() ||
      selected.fragmentAxes != ArrayRef<unsigned>{axis} ||
      llvm::any_of(fact.roots, [&](MakeRangeOp range) {
        return !sameLogicalRange(range, *authority) ||
               range.getResult().getType().getShape()[0] !=
                   type.getShape()[axis];
      }))
    return failure();
  ranges.assign(fact.roots.begin(), fact.roots.end());
  return success();
}

LogicalResult collectPairedReductionRanges(
    ContractOp contract, SmallVectorImpl<MakeRangeOp> &lhsRanges,
    SmallVectorImpl<MakeRangeOp> &rhsRanges) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1)
    return failure();
  auto lhsType = dyn_cast<FragmentType>(contract.getLhs().getType());
  auto rhsType = dyn_cast<FragmentType>(contract.getRhs().getType());
  if (!lhsType || !rhsType)
    return failure();
  if (failed(collectReductionAxisRanges(
          contract.getLhs(), contract.getLhsReductionAxes().front(), lhsRanges)) ||
      failed(collectReductionAxisRanges(
          contract.getRhs(), contract.getRhsReductionAxes().front(), rhsRanges)))
    return failure();
  MakeRangeOp lhs = lhsRanges.front(), rhs = rhsRanges.front();
  if (samePhysicalScalarExpression(lhs.getLogicalStart(), rhs.getLogicalStart()) &&
      samePhysicalScalarExpression(lhs.getLogicalStop(), rhs.getLogicalStop()))
    return success();
  auto lhsSize = constantLogicalRangeCardinality(lhs);
  auto rhsSize = constantLogicalRangeCardinality(rhs);
  return success(lhsSize && rhsSize && *lhsSize == *rhsSize);
}

bool hasExplicitPairedReductionRanges(ContractOp contract) {
  SmallVector<MakeRangeOp> lhsRanges, rhsRanges;
  return succeeded(collectPairedReductionRanges(contract, lhsRanges, rhsRanges));
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
                                       func::FuncOp kernel, Value value,
                                       PhysicalExprAttr blockedExtent,
                                       ArrayRef<MakeRangeOp> roots,
                                       Value replacement, IRMapping &mapping,
                                       Operation *insertionAnchor,
                                       DominanceInfo *dominance) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  if (auto extract = value.getDefiningOp<ExtractOp>()) {
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
      Value field = record.getFields()[extract.getField()];
      FailureOr<Value> replayed = replaySourceValueImpl(
          builder, location, kernel, field, blockedExtent, roots, replacement,
          mapping, insertionAnchor, dominance);
      if (failed(replayed))
        return failure();
      mapping.map(value, *replayed);
      return *replayed;
    }
  }
  if (auto range = value.getDefiningOp<MakeRangeOp>())
    if (llvm::any_of(roots, [&](MakeRangeOp root) {
          return range == root || sameLogicalRange(range, root);
        })) {
      auto original = range.getResult().getType();
      auto coordinate = cast<FragmentType>(replacement.getType());
      auto target = FragmentType::get(
          value.getContext(), original.getElementType(), coordinate.getShape(),
          original.getAxisMaps(), original.getValidity(), original.getOwner());
      return projectPhysicalValueToSchema(builder, location, replacement, target);
    }
  bool relocate = insertionAnchor && dominance &&
                  !dominance->dominates(value, insertionAnchor);
  auto originalResultType = dyn_cast<FragmentType>(value.getType());
  if (!originalResultType) {
    if (!relocate)
      return value;
    Operation *producer = value.getDefiningOp();
    if (!producer || producer->getNumRegions() != 0 ||
        producer->getNumResults() != 1 ||
        (!isa<arith::ConstantOp>(producer) &&
         !isPhysicalReplayNode(producer, PhysicalReplayScope::ValueGraph,
                               /*allowAccesses=*/true)))
      return failure();
    for (Value operand : producer->getOperands()) {
      FailureOr<Value> replayed = replaySourceValueImpl(
          builder, location, kernel, operand, blockedExtent, roots,
          replacement, mapping, insertionAnchor, dominance);
      if (failed(replayed))
        return failure();
      if (*replayed != operand && !mapping.lookupOrNull(operand))
        mapping.map(operand, *replayed);
    }
    Operation *clone = builder.clone(*producer, mapping);
    mapping.map(value, clone->getResult(0));
    return clone->getResult(0);
  }
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
  if (selected.fragmentAxes.empty() && !relocate)
    return value;
  Operation *producer = value.getDefiningOp();
  if (!producer || (isa<MakeRangeOp>(producer) && !relocate)) {
    if (producer)
      producer->emitOpError(
          "selected range root was not bound to its blocked replacement");
    return failure();
  }
  if (!isa<MakeRangeOp>(producer) &&
      !isPhysicalReplayNode(producer, PhysicalReplayScope::ValueGraph,
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
  SmallVector<MakeRangeOp> operandRoots(roots.begin(), roots.end());
  for (unsigned axis : selected.fragmentAxes)
    for (MakeRangeOp root :
         PhysicalProgramAnalysis(kernel).axisRanges(value, axis).roots)
      if (!llvm::is_contained(operandRoots, root))
        operandRoots.push_back(root);
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replayed = replaySourceValueImpl(
        builder, location, kernel, operand, blockedExtent, operandRoots,
        replacement, mapping, insertionAnchor, dominance);
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
  if (isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp, LoadOp>(producer)) {
    for (Value operand : producer->getOperands()) {
      if (auto load = dyn_cast<LoadOp>(producer);
          load && operand != load.getValid() && operand != load.getFill())
        continue;
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
        InFlightDiagnostic diagnostic = producer->emitOpError(
            "replayed operand cannot adopt the selected range relation")
            << "; operand=" << current.getType()
            << "; target=" << operandTarget;
        if (Operation *definition = current.getDefiningOp())
          diagnostic << "; operand_producer=" << definition->getName()
                     << "; operand_location=" << definition->getLoc();
        auto selectedOperand =
            PhysicalProgramAnalysis(kernel).rangeAxes(operand, roots);
        diagnostic << "; selected_operand_axes="
                   << selectedOperand.fragmentAxes.size();
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
                                   IRMapping &mapping,
                                   Operation *insertionAnchor) {
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
  bool preserveValue = !replay.isReplayable() ||
      llvm::any_of(replay.accesses, [&](Operation *access) {
        auto load = dyn_cast<LoadOp>(access);
        return load && !canReplayReadAt(load, insertionAnchor);
      });
  if (preserveValue) {
    Operation *owner = value.getDefiningOp() ? value.getDefiningOp()
                                           : kernel.getOperation();
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeAxisFact selected = analysis.rangeAxes(value, roots);
    auto original = dyn_cast<FragmentType>(value.getType());
    DominanceInfo dominance(kernel);
    if (!original || !selected.isExact() || selected.fragmentAxes.size() != 1 ||
        !dominance.dominates(value, insertionAnchor))
      return owner->emitError("contraction blocking cannot slice the original value");
    FailureOr<Value> sliced = materializeRetainedSlice(
        builder, location, value, selected.fragmentAxes.front(), blockedExtent,
        replacement, insertionAnchor);
    if (failed(sliced))
      return failure();
    mapping.map(value, *sliced);
    return sliced;
  }
  FailureOr<Value> result = replaySourceValueImpl(
      builder, location, kernel, value, blockedExtent, roots, replacement,
      mapping, /*insertionAnchor=*/nullptr, /*dominance=*/nullptr);
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
                                 ArrayRef<MakeRangeOp> ranges,
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
            "blocked contraction tail requires a fragment load result");
      auto predicateType = FragmentType::get(
          resultType.getContext(), IntegerType::get(resultType.getContext(), 1),
          resultType.getShape(),
          resultType.getAxisMaps(), resultType.getValidity(),
          resultType.getOwner());
      OpBuilder validityBuilder(load);
      auto kernel = originalLoad->getParentOfType<func::FuncOp>();
      auto axes = PhysicalProgramAnalysis(kernel).rangeAxes(
          originalLoad.getResult(), ranges);
      if (!axes.isExact() || axes.fragmentAxes.size() != 1)
        return load.emitOpError(
            "blocked contraction tail has no unique load-axis relation");
      FailureOr<Value> projected = projectPredicateToFragmentAxis(
          validityBuilder, location, tailPredicate, predicateType,
          axes.fragmentAxes.front());
      if (failed(projected))
        return load.emitOpError(
            "blocked contraction tail cannot project to its load coordinates");
      Value valid = *projected;
      if (load.getValid()) {
        Value existing = load.getValid();
        if (existing.getType() != predicateType) {
          FailureOr<Value> projectedExisting = projectPhysicalValueToSchema(
              validityBuilder, location, existing, predicateType);
          if (failed(projectedExisting))
            return load.emitOpError(
                "blocked contraction tail cannot preserve existing load validity");
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
                                   func::FuncOp kernel, Value value,
                                   PhysicalSourceAxis source,
                                   PhysicalExprAttr blockedExtent,
                                   MakeRangeOp root, Value replacement,
                                   IRMapping &mapping,
                                   Operation *insertionAnchor = nullptr) {
  FailureOr<int64_t> dimension = queryRangeDimension(root);
  if (!kernel || !(sourceAxisIdentity(root) == source) || failed(dimension))
    return failure();
  PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
      value, source, PhysicalReplayScope::ValueGraph,
      /*allowAccesses=*/true, insertionAnchor, *dimension);
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
  std::optional<DominanceInfo> dominance;
  if (insertionAnchor)
    dominance.emplace(kernel);
  return replaySourceValueImpl(
      builder, location, kernel, value, blockedExtent,
      ArrayRef<MakeRangeOp>(root), replacement, mapping, insertionAnchor,
      dominance ? &*dominance : nullptr);
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
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    if (extent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) ||
        extent.getValue() <= 0 || llvm::isPowerOf2_64(extent.getValue()))
      continue;
    if (failed(realizeFullCoverageDimension(kernel, source, axis)))
      return failure();
    fragment = cast<FragmentType>(source.getType());
  }
  for (int64_t axis : axes) {
    if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
      return failure();
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    FailureOr<ParameterOp> parameter = parameterForExtent(kernel, extent);
    if (failed(parameter)) {
      PhysicalAxisRealizationFact coverage =
          PhysicalProgramAnalysis(kernel).axisRealization(source, axis);
      if (!coverage.isExact() || !coverage.physicalized ||
          coverage.constructionScalarSeed) {
        if (!coverage.roots.empty() &&
            llvm::all_of(coverage.roots, [](MakeRangeOp range) {
              return isUnitStepRange(range) && samePhysicalScalarExpression(
                  range.getStart(), range.getLogicalStart());
            }) && succeeded(realizeFullCoverageDimension(kernel, source, axis))) {
          fragment = cast<FragmentType>(source.getType());
          continue;
        }
        return kernel.emitError(
            "native contraction axis has no exact physical coverage")
               << "; axis=" << axis << "; source=" << source.getType()
               << "; construction_seed=" << coverage.constructionScalarSeed;
      }
      continue;
    }
    uint32_t role = parameter->getParameter().getRole();
    if (role == static_cast<uint32_t>(ParameterRole::ScanChunk) ||
        role == static_cast<uint32_t>(ParameterRole::Reduction) ||
        role == static_cast<uint32_t>(ParameterRole::FullCoverage))
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

bool isZeroScalar(Value value);

Value stripCoverageProjection(Value value) {
  while (Operation *producer = value.getDefiningOp()) {
    if (auto broadcast = dyn_cast<BroadcastOp>(producer))
      value = broadcast.getValue();
    else if (auto reshape = dyn_cast<ReshapeOp>(producer))
      value = reshape.getValue();
    else if (auto transpose = dyn_cast<TransposeOp>(producer))
      value = transpose.getValue();
    else if (auto splat = dyn_cast<SplatOp>(producer))
      value = splat.getValue();
    else
      break;
  }
  return value;
}

bool impliesLogicalUpperBound(Value predicate, MakeRangeOp range,
                               llvm::SmallDenseSet<Value> &visited) {
  if (!predicate || !visited.insert(predicate).second)
    return false;
  predicate = stripCoverageProjection(predicate);
  if (isZeroScalar(predicate))
    return true;
  if (auto binary = predicate.getDefiningOp<BinaryOp>())
    if (binary.getOperatorKind() == BinaryOperator::LogicalAnd ||
        binary.getOperatorKind() == BinaryOperator::BitwiseAnd)
      return impliesLogicalUpperBound(binary.getLhs(), range, visited) ||
             impliesLogicalUpperBound(binary.getRhs(), range, visited);
  auto compare = predicate.getDefiningOp<CompareOp>();
  if (!compare || compare.getPredicate() != ComparePredicate::Lt)
    return false;
  Value coordinate = stripCoverageProjection(compare.getLhs());
  Value bound = stripCoverageProjection(compare.getRhs());
  auto coordinateRange = coordinate.getDefiningOp<MakeRangeOp>();
  return coordinateRange && sameLogicalRange(coordinateRange, range) &&
         samePhysicalScalarExpression(coordinateRange.getStart(), range.getStart()) &&
         samePhysicalScalarExpression(coordinateRange.getExtent(), range.getExtent()) &&
         samePhysicalScalarExpression(bound, range.getLogicalStop());
}

bool isZeroPastLogicalEnd(Value value, MakeRangeOp range) {
  while (true) {
    value = stripCoverageProjection(value);
    if (auto cast = value.getDefiningOp<CastOp>()) {
      value = cast.getValue();
      continue;
    }
    break;
  }
  if (isZeroScalar(value))
    return true;
  Value valid;
  Value fill;
  if (auto load = value.getDefiningOp<LoadOp>()) {
    valid = load.getValid();
    fill = load.getFill();
  } else if (auto gather = value.getDefiningOp<GatherOp>()) {
    valid = gather.getValid();
    fill = gather.getFill();
  } else if (auto select = value.getDefiningOp<SelectOp>()) {
    valid = select.getCondition();
    fill = select.getFalseValue();
  }
  if (!fill || !isZeroScalar(fill))
    return false;
  llvm::SmallDenseSet<Value> visited;
  return impliesLogicalUpperBound(valid, range, visited);
}

LogicalResult neutralizeFullCoverageOperand(ContractOp contract,
                                            OpOperand &operand,
                                            ArrayRef<int64_t> axes,
                                            func::FuncOp kernel) {
  for (int64_t axis : axes) {
    Value source = operand.get();
    auto type = cast<FragmentType>(source.getType());
    bool fullCoverage = isFullCoverageExtent(contract, type.getShape()[axis]);
    auto extent = cast<PhysicalExprAttr>(type.getShape()[axis]);
    if (!fullCoverage && extent.getKind() !=
                             static_cast<uint32_t>(PhysicalExprKind::Constant))
      continue;
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact fact = analysis.axisRanges(source, static_cast<unsigned>(axis));
    auto traversal = analysis.lockstepRanges(fact.roots);
    FailureOr<MakeRangeOp> range =
        fact.state != PhysicalFactState::Unknown && fact.blockers.empty() &&
                traversal.isExact()
            ? FailureOr<MakeRangeOp>(traversal.authority)
            : FailureOr<MakeRangeOp>(failure());
    if (!fullCoverage && (failed(range) || !isUnitStepRange(*range)))
      continue;
    if (failed(range) || !isUnitStepRange(*range))
      return contract.emitOpError(
          "full-coverage contraction tail requires an exact unit-step range");
    FailureOr<Value> stop = resolveLogicalRangeEnd(kernel, *range);
    if (failed(stop))
      return contract.emitOpError(
          "full-coverage contraction tail has no logical bound");
    // Preserve a proven zero extension instead of materializing a second mask.
    // Derived operations such as exp do not preserve it and still need the
    // explicit contraction identity below.
    if (isZeroPastLogicalEnd(source, *range))
      continue;
    OpBuilder builder(contract);
    Location location = contract.getLoc();
    auto coordinateType = range->getResult().getType();
    auto predicateType = FragmentType::get(
        kernel.getContext(), builder.getI1Type(), coordinateType.getShape(),
        coordinateType.getAxisMaps(), coordinateType.getValidity(),
        coordinateType.getOwner());
    Value stopFragment =
        builder.create<BroadcastOp>(location, coordinateType, *stop);
    Value tail = builder.create<CompareOp>(
        location, predicateType, range->getResult(), stopFragment,
        ComparePredicate::Lt);
    FailureOr<Value> valid = projectPredicateToFragmentAxis(
        builder, location, tail, type, static_cast<unsigned>(axis));
    FailureOr<Value> zero = materializeZeroFragment(builder, location, type);
    if (failed(valid) || failed(zero))
      return contract.emitOpError(
          "could not materialize full-coverage contraction identity");
    // Load padding is insufficient for derived operands: exp(padding) can be
    // infinite, and multiplying it by the other operand's zero produces NaN.
    operand.set(builder.create<SelectOp>(location, type, *valid, source, *zero));
  }
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
    // A rank-lifted outer ownership axis can sit beside a logical unit free
    // axis introduced by reshape (for example [H, 1, K]).  The unit carries no
    // independent matrix work.  Squeeze it, perform the canonical matrix
    // contraction, and restore the logical result shape afterwards.  This
    // keeps the outer axis as M/N rather than degrading it into a batch of
    // one-row contractions.
    {
      auto freeAxes = [](FragmentType type, ArrayRef<int64_t> reduction) {
        SmallVector<unsigned> result;
        for (unsigned axis = 0; axis < type.getShape().size(); ++axis)
          if (!llvm::is_contained(reduction, static_cast<int64_t>(axis)))
            result.push_back(axis);
        return result;
      };
      PhysicalProgramAnalysis analysis(kernel);
      auto unitAxis = [&](Value value, unsigned axis) {
        auto type = cast<FragmentType>(value.getType());
        auto extent = dyn_cast<PhysicalExprAttr>(type.getShape()[axis]);
        if (!extent || extent.getKind() !=
                           static_cast<uint32_t>(PhysicalExprKind::Constant) ||
            extent.getValue() != 1)
          return false;
        auto realization = analysis.axisRealization(value, axis);
        return realization.hasExtentAuthority() &&
               !realization.constructionScalarSeed;
      };
      auto lhs = contract.getLhs().getType();
      auto rhs = contract.getRhs().getType();
      SmallVector<unsigned> lhsFree =
          freeAxes(lhs, contract.getLhsReductionAxes());
      SmallVector<unsigned> rhsFree =
          freeAxes(rhs, contract.getRhsReductionAxes());
      SmallVector<unsigned> lhsErased;
      SmallVector<unsigned> rhsErased;
      if (lhsFree.size() > 1)
        for (unsigned axis : lhsFree)
          if (unitAxis(contract.getLhs(), axis))
            lhsErased.push_back(axis);
      if (rhsFree.size() > 1)
        for (unsigned axis : rhsFree)
          if (unitAxis(contract.getRhs(), axis))
            rhsErased.push_back(axis);
      const size_t lhsRemaining = lhsFree.size() - lhsErased.size();
      const size_t rhsRemaining = rhsFree.size() - rhsErased.size();
      bool removesBatches = llvm::all_of(contract.getLhsBatchAxes(), [&](int64_t axis) {
        return llvm::is_contained(lhsErased, axis);
      }) && llvm::all_of(contract.getRhsBatchAxes(), [&](int64_t axis) {
        return llvm::is_contained(rhsErased, axis);
      });
      if ((!lhsErased.empty() || !rhsErased.empty()) && lhsRemaining == 1 &&
          rhsRemaining == 1 && removesBatches) {
        auto eraseAxes = [&](FragmentType source,
                             ArrayRef<unsigned> erased) {
          SmallVector<Attribute> shape;
          SmallVector<Attribute> mappings;
          for (auto [axis, extent] : llvm::enumerate(source.getShape())) {
            if (llvm::is_contained(erased, axis))
              continue;
            shape.push_back(extent);
            auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
            mappings.push_back(AxisMapAttr::get(
                source.getContext(), mapping.getSourceId(),
                mapping.getSourceAxis(), mapping.getDimensionId(),
                mappings.size(), mapping.getDerived()));
          }
          return FragmentType::get(
              source.getContext(), source.getElementType(),
              ArrayAttr::get(source.getContext(), shape),
              ArrayAttr::get(source.getContext(), mappings),
              source.getValidity(), source.getOwner());
        };
        auto remap = [](ArrayRef<int64_t> axes,
                        ArrayRef<unsigned> erased) {
          SmallVector<int64_t> result;
          for (int64_t axis : axes) {
            int64_t shift = llvm::count_if(erased, [&](unsigned removed) {
              return removed < static_cast<unsigned>(axis);
            });
            result.push_back(axis - shift);
          }
          return result;
        };
        SmallVector<unsigned> resultErased;
        FragmentType originalResult = contract.getResult().getType();
        auto appendResultAxes = [&](FragmentType operand,
                                    ArrayRef<unsigned> erased,
                                    ArrayRef<int64_t> pairedBatch) {
          for (unsigned axis : erased) {
            if (llvm::is_contained(pairedBatch, axis))
              continue;
            auto mapping = cast<AxisMapAttr>(operand.getAxisMaps()[axis]);
            auto projection = queryFragmentDimension(
                originalResult, mapping.getDimensionId());
            if (!projection.isExact() ||
                llvm::is_contained(resultErased, projection.fragmentAxis))
              return failure();
            resultErased.push_back(projection.fragmentAxis);
          }
          return success();
        };
        if (failed(appendResultAxes(lhs, lhsErased, {})) ||
            failed(appendResultAxes(rhs, rhsErased, contract.getRhsBatchAxes())))
          return contract.emitOpError(
              "singleton matrix axes have no exact result projection");
        FragmentType squeezedLhs = eraseAxes(lhs, lhsErased);
        FragmentType squeezedRhs = eraseAxes(rhs, rhsErased);
        FragmentType squeezedResult = eraseAxes(originalResult, resultErased);
        OpBuilder builder(contract);
        auto eraseRelation = [&](unsigned sourceRank,
                                 ArrayRef<unsigned> erased) {
          SmallVector<Attribute> groups;
          unsigned resultAxis = 0;
          for (unsigned sourceAxis = 0; sourceAxis < sourceRank; ++sourceAxis) {
            SmallVector<int64_t> resultAxes;
            if (!llvm::is_contained(erased, sourceAxis))
              resultAxes.push_back(resultAxis++);
            groups.push_back(ReshapeGroupAttr::get(
                contract.getContext(),
                DenseI64ArrayAttr::get(
                    contract.getContext(),
                    {static_cast<int64_t>(sourceAxis)}),
                DenseI64ArrayAttr::get(contract.getContext(), resultAxes)));
          }
          return ArrayAttr::get(contract.getContext(), groups);
        };
        auto reshapeTo = [&](Value value, FragmentType target,
                             ArrayRef<unsigned> erased) -> FailureOr<Value> {
          auto source = dyn_cast<FragmentType>(value.getType());
          if (!source)
            return failure();
          if (source == target)
            return value;
          auto reshape = builder.create<ReshapeOp>(
              contract.getLoc(), target, value,
              eraseRelation(source.getShape().size(), erased));
          if (Attribute origin = contract->getAttr(originAttr))
            reshape->setAttr(originAttr, origin);
          return reshape.getResult();
        };
        FailureOr<Value> normalizedLhs =
            reshapeTo(contract.getLhs(), squeezedLhs, lhsErased);
        FailureOr<Value> normalizedRhs =
            reshapeTo(contract.getRhs(), squeezedRhs, rhsErased);
        FailureOr<Value> normalizedAccumulator =
            reshapeTo(contract.getAccumulator(), squeezedResult, resultErased);
        if (failed(normalizedLhs) || failed(normalizedRhs) ||
            failed(normalizedAccumulator))
          return contract.emitOpError(
              "cannot squeeze logical unit free axes for matrix ownership");
        contract->setOperand(0, *normalizedLhs);
        contract->setOperand(1, *normalizedRhs);
        contract->setOperand(2, *normalizedAccumulator);
        contract->setAttr(
            "lhs_reduction_axes",
            builder.getDenseI64ArrayAttr(
                remap(contract.getLhsReductionAxes(), lhsErased)));
        contract->setAttr(
            "rhs_reduction_axes",
            builder.getDenseI64ArrayAttr(
                remap(contract.getRhsReductionAxes(), rhsErased)));
        contract->setAttr("lhs_batch_axes", builder.getDenseI64ArrayAttr({}));
        contract->setAttr("rhs_batch_axes", builder.getDenseI64ArrayAttr({}));
        contract.getResult().setType(squeezedResult);

        SmallVector<Attribute> restoredGroups;
        unsigned sourceAxis = 0;
        for (unsigned resultAxis = 0;
             resultAxis < originalResult.getShape().size(); ++resultAxis) {
          SmallVector<int64_t> sourceAxes;
          if (!llvm::is_contained(resultErased, resultAxis))
            sourceAxes.push_back(sourceAxis++);
          restoredGroups.push_back(ReshapeGroupAttr::get(
              contract.getContext(),
              DenseI64ArrayAttr::get(contract.getContext(), sourceAxes),
              DenseI64ArrayAttr::get(
                  contract.getContext(),
                  {static_cast<int64_t>(resultAxis)})));
        }
        builder.setInsertionPointAfter(contract);
        auto restored = builder.create<ReshapeOp>(
            contract.getLoc(), originalResult, contract.getResult(),
            ArrayAttr::get(contract.getContext(), restoredGroups));
        if (Attribute origin = contract->getAttr(originAttr))
          restored->setAttr(originAttr, origin);
        contract.getResult().replaceUsesWithIf(
            restored.getResult(), [&](OpOperand &use) {
              return use.getOwner() != restored.getOperation();
            });
      }
    }
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
  while (isa<FragmentType>(value.getType())) {
    UniformExpression expression = describeUniformValue(value);
    if (expression.kind != UniformKind::Forward ||
        expression.operands.size() != 1)
      return failure();
    value = expression.operands.front();
  }
  return value;
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

Value broadcastAxis(OpBuilder &builder, Location location, FragmentType result,
                    Value value, unsigned targetAxis) {
  auto source = cast<FragmentType>(value.getType());
  assert(source.getShape().size() == 1 && targetAxis < result.getShape().size());
  SmallVector<Attribute> shape, groups;
  for (unsigned axis = 0; axis < result.getShape().size(); ++axis) {
    shape.push_back(axis == targetAxis ? source.getShape()[0] :
                    expression(builder.getContext(), PhysicalExprKind::Constant, 1));
    SmallVector<int64_t> sourceAxes;
    if (axis == targetAxis) sourceAxes.push_back(0);
    groups.push_back(ReshapeGroupAttr::get(builder.getContext(),
        builder.getDenseI64ArrayAttr(sourceAxes),
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(axis)})));
  }
  auto expanded = FragmentType::get(builder.getContext(), source.getElementType(),
      builder.getArrayAttr(shape), result.getAxisMaps(), source.getValidity(), source.getOwner());
  Value reshaped = builder.create<ReshapeOp>(location, expanded, value, builder.getArrayAttr(groups));
  return builder.create<BroadcastOp>(location, result, reshaped);
}

Value rangeBoundsValidity(OpBuilder &builder, Location location,
                          FragmentType indexType,
                          FragmentType predicateType, Value coordinate,
                          Value logicalStop) {
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value lower = builder.create<CompareOp>(
      location, predicateType, coordinate,
      broadcast(builder, location, indexType, zero), ComparePredicate::Ge);
  Value upper = compare(builder, location, predicateType, coordinate,
                        broadcast(builder, location, indexType, logicalStop),
                        ComparePredicate::Lt);
  return binary(builder, location, predicateType, lower, upper,
                BinaryOperator::LogicalAnd);
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
  SmallVector<Operation *> operations;
  StoreOp store;
};

bool collectStorePaths(Value value, SmallVector<Operation *> operations,
                       SmallVectorImpl<StorePath> &paths,
                       llvm::SmallPtrSetImpl<Operation *> &visited,
                       Value root = {}) {
  if (!root)
    root = value;
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;
    if (auto broadcast = dyn_cast<BroadcastOp>(user)) {
      auto source = dyn_cast<FragmentType>(value.getType());
      auto result = dyn_cast<FragmentType>(broadcast.getResult().getType());
      if (!source || !result || source.getShape() != result.getShape())
        return false;
      auto projection = queryAxisProjection(source, result);
      if (!projection.isExact() ||
          llvm::any_of(llvm::enumerate(projection.targetToSource), [](auto entry) {
            return !entry.value() || *entry.value() != entry.index();
          }))
        return false;
    }
    if (isa<CastOp, BitcastOp, BroadcastOp, UnaryOp, BinaryOp, CompareOp,
            SelectOp>(user)) {
      if (!isa<CastOp, BroadcastOp>(user) &&
          (user->getBlock() != root.getParentBlock() ||
           !root.hasOneUse() || !value.hasOneUse() ||
           llvm::any_of(operations, [](Operation *operation) {
             return !operation->getResult(0).hasOneUse();
           })))
        return false;
      SmallVector<Operation *> next(operations);
      next.push_back(user);
      if (!collectStorePaths(user->getResult(0), std::move(next), paths,
                             visited, root))
        return false;
      continue;
    }
    auto store = dyn_cast<StoreOp>(user);
    if (!store || store.getValue() != value)
      return false;
    paths.push_back({operations, store});
  }
  return !paths.empty();
}

FailureOr<Value> materializeStorePath(
    OpBuilder &builder, func::FuncOp kernel, StorePath &path, Value original,
    Value blocked, Value rows, Value columns, Operation *insertionAnchor) {
  auto tile = cast<FragmentType>(blocked.getType());
  auto schema = [&](Type type) {
    auto fragment = cast<FragmentType>(type);
    return FragmentType::get(
        kernel.getContext(), fragment.getElementType(), tile.getShape(),
        tile.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
  };
  auto capture = [&](Value value) -> FailureOr<Value> {
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment)
      return value;
    FragmentType target = schema(fragment);
    if (value.getType() == target &&
        DominanceInfo(kernel).dominates(value, insertionAnchor))
      return value;
    if (auto scalar = scalarSource(value); succeeded(scalar))
      return projectPhysicalValueToSchema(builder, path.store.getLoc(), *scalar,
                                           target);
    if (fragment.getShape().size() != 2)
      return failure();
    Value captured = value;
    SmallVector<Value, 2> coordinates{rows, columns};
    for (auto [axis, coordinate] : llvm::enumerate(coordinates)) {
      auto replacement = coordinate.getDefiningOp<MakeRangeOp>();
      PhysicalProgramAnalysis analysis(kernel);
      PhysicalRangeFact ranges = analysis.axisRanges(captured, axis);
      auto root = queryExactLogicalRange(ranges);
      if (!replacement || failed(root) || !isUnitStepRange(*root) ||
          !samePhysicalScalarExpression(root->getLogicalStart(),
                                        replacement.getLogicalStart()) ||
          !samePhysicalScalarExpression(root->getLogicalStop(),
                                        replacement.getLogicalStop())) {
        path.store.emitOpError("epilogue capture has no matching result-axis range")
            << "; axis=" << axis << "; value=" << value
            << "; root=" << (succeeded(root) ? root->getResult() : Value())
            << "; replacement=" << coordinate;
        return failure();
      }
      auto source = sourceAxisIdentity(cast<AxisMapAttr>(
          cast<FragmentType>(value.getType()).getAxisMaps()[axis]));
      if (!analysis.replayability(value, source, PhysicalReplayScope::ValueGraph,
                                 /*allowAccesses=*/true, insertionAnchor)
               .isReplayable()) {
        path.store.emitOpError("epilogue capture cannot be replayed at its write")
            << "; axis=" << axis << "; value=" << value;
        return failure();
      }
      auto indexType = root->getResult().getType();
      auto extent = cast<PhysicalExprAttr>(tile.getShape()[axis]);
      indexType = FragmentType::get(
          kernel.getContext(), indexType.getElementType(),
          builder.getArrayAttr({extent}), indexType.getAxisMaps(),
          indexType.getValidity(), indexType.getOwner());
      Value range = builder.create<MakeRangeOp>(
          path.store.getLoc(), indexType, replacement.getStart(),
          replacement.getExtent(), root->getStep(), root->getLogicalStart(),
          root->getLogicalStop(), root->getSourceId(), root->getSourceAxis(),
          root->getDerived());
      inheritRangeAuthority(range, *root);
      auto predicateType = FragmentType::get(
          kernel.getContext(), builder.getI1Type(), indexType.getShape(),
          indexType.getAxisMaps(), indexType.getValidity(), indexType.getOwner());
      Value tail = rangeBoundsValidity(builder, path.store.getLoc(), indexType,
                                       predicateType, range, root->getLogicalStop());
      IRMapping mapping;
      for (MakeRangeOp sourceRange : ranges.roots)
        mapping.map(sourceRange.getResult(), range);
      ReplayMaterializationOptions options;
      options.fragmentAxis = axis;
      options.segmentTail = tail;
      options.materializeZeroFill = true;
      auto replayed = materializeReplayedValue(
          builder, path.store.getLoc(), value, source, extent, mapping, options);
      if (failed(replayed)) {
        path.store.emitOpError("epilogue capture axis could not be materialized")
            << "; axis=" << axis << "; value=" << value;
        return failure();
      }
      value = *replayed;
    }
    return projectPhysicalValueToSchema(builder, path.store.getLoc(), value,
                                         target);
  };
  IRMapping mapping;
  mapping.map(original, blocked);
  for (Operation *operation : path.operations) {
    for (Value operand : operation->getOperands()) {
      if (mapping.contains(operand))
        continue;
      auto replayed = capture(operand);
      if (failed(replayed)) {
        operation->emitOpError("contraction epilogue operand cannot be tiled")
            << "; operand=" << operand;
        return failure();
      }
      mapping.map(operand, *replayed);
    }
    Operation *clone = builder.clone(*operation, mapping);
    clone->getResult(0).setType(schema(operation->getResult(0).getType()));
  }
  Value result = mapping.lookupOrNull(path.store.getValue());
  return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
}

bool isTransparentMatrixReshape(ReshapeOp reshape) {
  auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
  auto target = dyn_cast<FragmentType>(reshape.getResult().getType());
  if (!source || !target || source.getElementType() != target.getElementType())
    return false;
  auto projectedAxes = [](FragmentType type) {
    SmallVector<std::pair<PhysicalSourceAxis, Attribute>> axes;
    for (auto [extent, mapping] :
         llvm::zip(type.getShape(), type.getAxisMaps())) {
      auto axis = cast<AxisMapAttr>(mapping);
      auto expression = cast<PhysicalExprAttr>(extent);
      bool introducedUnit =
          axis.getDerived() &&
          expression.getKind() ==
              static_cast<uint32_t>(PhysicalExprKind::Constant) &&
          expression.getValue() == 1;
      if (!introducedUnit)
        axes.emplace_back(sourceAxisIdentity(axis), extent);
    }
    return axes;
  };
  return projectedAxes(source) == projectedAxes(target);
}

Value stripTransparentMatrixReshapes(Value value) {
  while (auto reshape = value.getDefiningOp<ReshapeOp>()) {
    if (!isTransparentMatrixReshape(reshape))
      break;
    value = reshape.getValue();
  }
  return value;
}

LoadOp matrixOperandLoad(Value value) {
  value = stripTransparentMatrixReshapes(value);
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
  return stripTransparentMatrixReshapes(transpose.getValue())
      .getDefiningOp<LoadOp>();
}

FailureOr<unsigned> accessCoordinatePosition(LoadOp load,
                                             AxisMapAttr mapping,
                                             Value operand) {
  return PhysicalProgramAnalysis(load->getParentOfType<func::FuncOp>())
      .accessCoordinatePosition(load, mapping, operand);
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

bool freeAxesNeedRealization(ContractOp contract, func::FuncOp kernel) {
  return PhysicalProgramAnalysis(kernel)
      .contractFreeAxes(contract.getOperation())
      .needsRealization();
}

bool freeAxesReadyForReductionTraversal(ContractOp contract,
                                        func::FuncOp kernel) {
  PhysicalContractFreeAxisFact freeAxes =
      PhysicalProgramAnalysis(kernel).contractFreeAxes(contract.getOperation());
  if (!freeAxes.isExact() || freeAxes.axes.empty())
    return false;
  return llvm::all_of(freeAxes.axes, [&](const PhysicalContractFreeAxis &axis) {
    if (axis.realization.physicalized)
      return true;
    auto fragment = cast<FragmentType>(axis.operand.getType());
    auto extent = cast<PhysicalExprAttr>(
        fragment.getShape()[axis.operandAxis]);
    if (extent.getKind() ==
        static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
      FailureOr<ParameterOp> parameter =
          queryParameterBySymbol(kernel, extent.getSymbol());
      if (failed(parameter))
        return false;
      auto role = static_cast<ParameterRole>(
          parameter->getParameter().getRole());
      return role == ParameterRole::OwnershipM ||
             role == ParameterRole::OwnershipN ||
             role == ParameterRole::FullCoverage;
    }
    return extent.getKind() ==
               static_cast<uint32_t>(PhysicalExprKind::Constant) &&
           extent.getValue() == 1 &&
           axis.realization.isExact() &&
           !axis.realization.constructionScalarSeed &&
           axis.realization.roots.empty();
  });
}

bool reductionAxesNeedTraversal(ContractOp contract, func::FuncOp kernel) {
  PhysicalProgramAnalysis analysis(kernel);
  auto operandNeedsTraversal = [&](Value operand, ArrayRef<int64_t> axes) {
    auto fragment = cast<FragmentType>(operand.getType());
    return llvm::any_of(axes, [&](int64_t axis) {
      if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
        return false;
      PhysicalAxisRealizationFact fact =
          analysis.axisRealization(operand, static_cast<unsigned>(axis));
      return isOwnershipExtent(contract, fragment.getShape()[axis]) ||
             isFullCoverageExtent(contract, fragment.getShape()[axis]) ||
             fact.constructionScalarSeed ||
             (fact.isExact() && !fact.physicalized);
    });
  };
  return operandNeedsTraversal(contract.getLhs(),
                               contract.getLhsReductionAxes()) ||
         operandNeedsTraversal(contract.getRhs(),
                               contract.getRhsReductionAxes()) ||
         fullReductionNeedsTraversal(contract);
}

bool hasCompleteStorePath(ContractOp contract) {
  SmallVector<StorePath> paths;
  llvm::SmallPtrSet<Operation *, 8> visited;
  return collectStorePaths(contract.getResult(), {}, paths, visited);
}

bool outputCoordinatesNeedRealization(ContractOp contract) {
  if (contract.getResult().getType().getShape().size() != 2 ||
      !contract.getLhsBatchAxes().empty() ||
      !contract.getRhsBatchAxes().empty())
    return false;
  SmallVector<StorePath> paths;
  llvm::SmallPtrSet<Operation *, 8> visited;
  if (!collectStorePaths(contract.getResult(), {}, paths, visited))
    return false;
  auto kernel = contract->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  return llvm::any_of(paths, [&](StorePath &path) {
    if (path.store.getCoordinates().size() != 2 ||
        path.store.getSourceAxes() != ArrayRef<int64_t>{0, 1})
      return false;
    MakeRangeOp outputRanges[2], sourceRanges[2];
    for (unsigned axis = 0; axis < 2; ++axis) {
      auto output = sourceRange(path.store.getCoordinates()[axis]);
      auto source = queryExactLogicalRange(
          analysis.axisRanges(contract.getResult(), axis));
      if (!output || failed(source))
        continue;
      if (samePhysicalScalarExpression(output.getLogicalStart(), source->getLogicalStart()) &&
          samePhysicalScalarExpression(output.getLogicalStop(), source->getLogicalStop()) &&
          samePhysicalScalarExpression(output.getStep(), source->getStep())) {
        outputRanges[axis] = output;
        sourceRanges[axis] = *source;
        if (output.getResult().getType().getShape() ==
                source->getResult().getType().getShape() &&
            !samePhysicalScalarExpression(output.getStart(), source->getStart()))
          return true;
      }
    }
    // Reusing an operand in two matrix roles does not equate the independent
    // output coordinates. Realize their M/N ownership before a K-only loop.
    return sourceRanges[0] && sourceRanges[1] &&
           sameLogicalRange(sourceRanges[0], sourceRanges[1]) &&
           outputRanges[0] && outputRanges[1] &&
           outputRanges[0] != outputRanges[1];
  });
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

FailureOr<bool> collapseMultiReductionContract(ContractOp contract) {
  SmallVector<int64_t> lhsAxes(contract.getLhsReductionAxes());
  SmallVector<int64_t> rhsAxes(contract.getRhsReductionAxes());
  if (lhsAxes.size() <= 1 || lhsAxes.size() != rhsAxes.size() ||
      !contract.getLhsBatchAxes().empty() || !contract.getRhsBatchAxes().empty())
    return false;
  if (!contract.getLhs().getDefiningOp<LoadOp>() ||
      !contract.getRhs().getDefiningOp<LoadOp>())
    return false;
  func::FuncOp kernel = contract->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  auto unit = expression(kernel.getContext(), PhysicalExprKind::Constant, 1);
  for (Value operand : {contract.getLhs(), contract.getRhs()}) {
    auto load = operand.getDefiningOp<LoadOp>();
    if (!isa<ViewType>(load.getResource().getType()))
      return false;
    auto type = cast<FragmentType>(operand.getType());
    llvm::DenseMap<Operation *, unsigned> rootAxes;
    for (unsigned axis = 0; axis < type.getShape().size(); ++axis) {
      PhysicalRangeFact ranges = analysis.axisRanges(operand, axis);
      auto realization = analysis.axisRealization(operand, axis);
      bool introducedUnit = ranges.roots.empty() && realization.isExact() &&
                            !realization.constructionScalarSeed &&
                            type.getShape()[axis] == unit;
      if (ranges.state != PhysicalFactState::Exact && !introducedUnit)
        return false;
      for (MakeRangeOp root : ranges.roots) {
        auto [found, inserted] = rootAxes.try_emplace(root.getOperation(), axis);
        if (!inserted && found->second != axis)
          return false;
      }
    }
  }
  SmallVector<PhysicalExprAttr> reductionExtents;
  for (auto [lhsAxis, rhsAxis] : llvm::zip(lhsAxes, rhsAxes)) {
    SmallVector<MakeRangeOp> paired;
    for (auto [operand, axis] :
         {std::pair<Value, int64_t>{contract.getLhs(), lhsAxis},
          std::pair<Value, int64_t>{contract.getRhs(), rhsAxis}}) {
      auto load = operand.getDefiningOp<LoadOp>();
      if (!canReplayReadAt(load, contract))
        return false;
      PhysicalRangeFact ranges = analysis.axisRanges(operand, axis);
      if (failed(queryExactLogicalRange(ranges)))
        return false;
      auto projection = analysis.rangeAxes(operand, ranges.roots);
      if (!projection.isExact() ||
          projection.fragmentAxes != ArrayRef<unsigned>{static_cast<unsigned>(axis)})
        return false;
      auto dimension = queryRangeDimension(ranges.roots.front());
      if (failed(dimension))
        return false;
      auto source = sourceAxisIdentity(ranges.roots.front());
      auto replay = analysis.replayability(
          operand, source, PhysicalReplayScope::ValueGraph,
          /*allowAccesses=*/true, /*insertionAnchor=*/nullptr, *dimension);
      if (!replay.isReplayable() ||
          llvm::any_of(replay.accesses, [&](Operation *access) {
            auto read = dyn_cast<LoadOp>(access);
            return read && !canReplayReadAt(read, contract);
          }))
        return false;
      for (MakeRangeOp range : ranges.roots) {
        auto realization = analysis.axisRealization(range.getResult(), 0);
        auto currentDimension = queryRangeDimension(range);
        auto cardinality = constantLogicalRangeCardinality(range);
        auto physical = constantPhysicalExpression(
            cast<PhysicalExprAttr>(range.getResult().getType().getShape()[0]));
        bool complete = cardinality && physical && *physical >= *cardinality;
        if (!isZeroScalar(range.getStart()) ||
            !isZeroScalar(range.getLogicalStart()) || !isUnitStepRange(range) ||
            range->hasAttr(sourceSubregionAttr) ||
            !queryLaunchExpression(range.getLogicalStop()) ||
            !(sourceAxisIdentity(range) == source) || failed(currentDimension) ||
            *currentDimension != *dimension ||
            (!complete && !realization.constructionScalarSeed &&
             !samePhysicalScalarExpression(range.getExtent(),
                                           range.getLogicalStop())))
          return false;
        paired.push_back(range);
      }
    }
    if (!analysis.lockstepRanges(paired).isExact())
      return false;
    reductionExtents.push_back(
        queryLaunchExpression(paired.front().getLogicalStop()));
  }

  // Put a complete power-of-two tile inside a non-power-of-two outer group.
  // The outer coordinates then stay constant across aligned K lanes, allowing
  // the provider's axis analysis to share their delinearization arithmetic.
  auto tail = reductionExtents.back();
  if (tail.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
      tail.getValue() > 0 && !llvm::isPowerOf2_64(tail.getValue())) {
    unsigned inner = lhsAxes.size() - 1;
    int64_t largest = 0;
    for (auto [position, extent] : llvm::enumerate(reductionExtents)) {
      if (extent.getKind() !=
              static_cast<uint32_t>(PhysicalExprKind::Constant) ||
          extent.getValue() < contractionReductionCandidates[0] ||
          !llvm::isPowerOf2_64(extent.getValue()) ||
          extent.getValue() <= largest)
        continue;
      inner = position;
      largest = extent.getValue();
    }
    std::rotate(lhsAxes.begin() + inner, lhsAxes.begin() + inner + 1,
                lhsAxes.end());
    std::rotate(rhsAxes.begin() + inner, rhsAxes.begin() + inner + 1,
                rhsAxes.end());
  }

  auto [sourceId, dimensionId] = nextPhysicalAxisIdentities(kernel);
  OpBuilder builder(contract);
  // Expand only the contraction's replayable operands, using logical extents
  // before flattening. Padding each source axis would change the flattened
  // coordinate relation; the eventual K tile owns the single tail instead.
  auto complete = [&](Value value, ArrayRef<int64_t> axes) -> FailureOr<Value> {
    for (int64_t axis : axes) {
      auto ranges = PhysicalProgramAnalysis(kernel).axisRanges(value, axis);
      MakeRangeOp root = ranges.roots.front();
      if (llvm::all_of(ranges.roots, [](MakeRangeOp range) {
            return samePhysicalScalarExpression(range.getExtent(),
                                                range.getLogicalStop());
          }))
        continue;
      auto extent = queryLaunchExpression(root.getLogicalStop());
      auto original = root.getResult().getType();
      auto type = FragmentType::get(
          kernel.getContext(), original.getElementType(),
          builder.getArrayAttr({extent}), original.getAxisMaps(),
          original.getValidity(), original.getOwner());
      Value full = builder.create<MakeRangeOp>(
          contract.getLoc(), type, root.getStart(), root.getLogicalStop(),
          root.getStep(), root.getLogicalStart(), root.getLogicalStop(),
          root.getSourceId(), root.getSourceAxis(), root.getDerived());
      inheritRangeAuthority(full, root);
      IRMapping mapping;
      auto replayed = replaySourceValue(builder, contract.getLoc(), value,
                                       extent, ranges.roots, full, mapping,
                                       contract.getOperation());
      if (failed(replayed))
        return failure();
      value = *replayed;
    }
    return value;
  };
  auto completeLhs = complete(contract.getLhs(), lhsAxes);
  auto completeRhs = complete(contract.getRhs(), rhsAxes);
  if (failed(completeLhs) || failed(completeRhs))
    return contract.emitOpError("cannot replay complete paired reduction ranges");
  auto collapse = [&](Value value, ArrayRef<int64_t> reductionAxes)
      -> std::pair<Value, int64_t> {
    SmallVector<int64_t> reductions(reductionAxes);
    auto source = cast<FragmentType>(value.getType());
    bool contiguous = llvm::all_of(
        llvm::enumerate(reductions), [&](auto entry) {
          return entry.value() == reductions.front() +
                                      static_cast<int64_t>(entry.index());
        });
    if (!contiguous) {
      // Preserve free-axis order and use the same paired-reduction order on
      // both operands to make their exact ranges one collapsible group.
      SmallVector<int64_t> permutation;
      for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
        if (!llvm::is_contained(reductions, static_cast<int64_t>(axis)))
          permutation.push_back(axis);
      int64_t firstReduction = permutation.size();
      llvm::append_range(permutation, reductions);
      SmallVector<Attribute> transposedShape, transposedMappings;
      for (auto [resultAxis, sourceAxis] : llvm::enumerate(permutation)) {
        transposedShape.push_back(source.getShape()[sourceAxis]);
        auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
        transposedMappings.push_back(AxisMapAttr::get(
            kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), resultAxis, mapping.getDerived()));
      }
      source = FragmentType::get(
          kernel.getContext(), source.getElementType(),
          builder.getArrayAttr(transposedShape),
          builder.getArrayAttr(transposedMappings), source.getValidity(),
          source.getOwner());
      value = builder.create<TransposeOp>(contract.getLoc(), source, value,
                                           permutation);
      for (auto [position, axis] : llvm::enumerate(reductions))
        axis = firstReduction + static_cast<int64_t>(position);
    }
    SmallVector<Attribute> shape, mappings, groups;
    for (unsigned axis = 0; axis < source.getShape().size();) {
      unsigned resultAxis = shape.size();
      SmallVector<int64_t> sourceAxes;
      auto extent = cast<PhysicalExprAttr>(source.getShape()[axis]);
      AxisMapAttr mapping;
      if (axis == static_cast<unsigned>(reductions.front())) {
        sourceAxes.append(reductions.begin(), reductions.end());
        for (int64_t reduced : ArrayRef<int64_t>(reductions).drop_front()) {
          auto factor = cast<PhysicalExprAttr>(source.getShape()[reduced]);
          if (extent == unit)
            extent = factor;
          else if (factor != unit)
            extent = binaryExpression(kernel.getContext(), PhysicalExprKind::Multiply,
                                      extent, factor);
        }
        mapping = AxisMapAttr::get(kernel.getContext(), sourceId, 0,
                                   dimensionId, resultAxis, true);
        axis += reductions.size();
      } else {
        sourceAxes.push_back(axis);
        auto original = cast<AxisMapAttr>(source.getAxisMaps()[axis++]);
        mapping = AxisMapAttr::get(
            kernel.getContext(), original.getSourceId(), original.getSourceAxis(),
            original.getDimensionId(), resultAxis, original.getDerived());
      }
      shape.push_back(extent);
      mappings.push_back(mapping);
      groups.push_back(ReshapeGroupAttr::get(
          kernel.getContext(), builder.getDenseI64ArrayAttr(sourceAxes),
          builder.getDenseI64ArrayAttr({static_cast<int64_t>(resultAxis)})));
    }
    auto target = FragmentType::get(
        kernel.getContext(), source.getElementType(), builder.getArrayAttr(shape),
        builder.getArrayAttr(mappings), source.getValidity(), source.getOwner());
    return {builder.create<ReshapeOp>(contract.getLoc(), target, value,
                                      builder.getArrayAttr(groups)),
            reductions.front()};
  };
  auto [lhs, lhsAxis] = collapse(*completeLhs, lhsAxes);
  auto [rhs, rhsAxis] = collapse(*completeRhs, rhsAxes);
  auto replacement = builder.create<ContractOp>(
      contract.getLoc(), contract.getResult().getType(), lhs, rhs,
      contract.getAccumulator(), ArrayRef<int64_t>{lhsAxis},
      ArrayRef<int64_t>{rhsAxis}, ArrayRef<int64_t>{}, ArrayRef<int64_t>{});
  if (Attribute origin = contract->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  contract.getResult().replaceAllUsesWith(replacement.getResult());
  contract.erase();
  return true;
}

LogicalResult decomposeMultiReductionContract(ContractOp contract) {
  if (!contract->getBlock() || contract.getLhsReductionAxes().size() <= 1)
    return success();
  if (contract.getLhsReductionAxes().size() !=
          contract.getRhsReductionAxes().size() ||
      contract.getLhsBatchAxes().size() != contract.getRhsBatchAxes().size())
    return contract.emitOpError(
        "multi-pair contraction decomposition requires paired reduction and batch axes");
  func::FuncOp kernel = contract->getParentOfType<func::FuncOp>();
  Location location = contract.getLoc();
  std::string failureReason;
  auto unit = expression(kernel.getContext(), PhysicalExprKind::Constant, 1);
  std::function<FailureOr<Value>(OpBuilder &, Value, Value,
                                SmallVector<int64_t>, SmallVector<int64_t>,
                                SmallVector<int64_t>, SmallVector<int64_t>, Value)>
      build;
  build = [&](OpBuilder &builder, Value lhs, Value rhs,
              SmallVector<int64_t> lhsReductions,
              SmallVector<int64_t> rhsReductions,
              SmallVector<int64_t> lhsBatch, SmallVector<int64_t> rhsBatch,
              Value accumulator) -> FailureOr<Value> {
    if (lhsReductions.size() == 1) {
      auto product = builder.create<ContractOp>(
          location, contract.getResult().getType(), lhs, rhs, accumulator,
          lhsReductions, rhsReductions, lhsBatch, rhsBatch);
      if (Attribute origin = contract->getAttr(originAttr))
        product->setAttr(originAttr, origin);
      return product.getResult();
    }

    unsigned lhsAxis = lhsReductions.front();
    unsigned rhsAxis = rhsReductions.front();
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact lhsFact = analysis.axisRanges(lhs, lhsAxis);
    PhysicalRangeFact rhsFact = analysis.axisRanges(rhs, rhsAxis);
    FailureOr<MakeRangeOp> lhsRange = queryExactLogicalRange(lhsFact);
    FailureOr<MakeRangeOp> rhsRange = queryExactLogicalRange(rhsFact);
    if (failed(lhsRange) || failed(rhsRange)) {
      failureReason = "a reduction pair has no exact source-axis ranges";
      return failure();
    }
    SmallVector<MakeRangeOp> pairedRanges(lhsFact.roots.begin(), lhsFact.roots.end());
    llvm::append_range(pairedRanges, rhsFact.roots);
    PhysicalLockstepTraversalFact lockstep = analysis.lockstepRanges(pairedRanges);
    if (!lockstep.isExact()) {
      failureReason = "a reduction pair does not have one lockstep traversal";
      return failure();
    }
    FailureOr<Value> logicalEnd = resolveLogicalRangeEnd(kernel, lockstep.authority);
    FailureOr<Value> step = scalarSource(lockstep.authority.getStep());
    if (failed(logicalEnd) || failed(step)) {
      failureReason = "a reduction pair has no exact logical bounds and step";
      return failure();
    }

    bool failedBody = false;
    auto loop = builder.create<scf::ForOp>(
        location, lockstep.authority.getLogicalStart(), *logicalEnd, *step,
        ValueRange{accumulator},
        [](OpBuilder &, Location, Value, ValueRange) {});
    auto buildBody = [&](OpBuilder &nested, Location nestedLocation, Value coordinate,
            ValueRange carries) {
          auto slice = [&](Value value, unsigned axis,
                           const PhysicalRangeFact &fact) -> FailureOr<Value> {
            MakeRangeOp root = fact.roots.front();
            auto rangeType = cast<FragmentType>(root.getResult().getType());
            auto indexType = FragmentType::get(
                kernel.getContext(), nested.getIndexType(),
                nested.getArrayAttr({unit}), rangeType.getAxisMaps(),
                rangeType.getValidity(), rangeType.getOwner());
            Value one = nested.create<arith::ConstantIndexOp>(nestedLocation, 1);
            Value selected = nested.create<MakeRangeOp>(
                nestedLocation, indexType, coordinate, one, root.getStep(),
                root.getLogicalStart(), root.getLogicalStop(), root.getSourceId(),
                root.getSourceAxis(), root.getDerived());
            inheritRangeAuthority(selected, root);
            IRMapping mapping;
            for (MakeRangeOp range : fact.roots)
              mapping.map(range.getResult(), selected);
            FailureOr<Value> replayed = replaySourceValue(
                nested, nestedLocation, value, unit, fact.roots, selected,
                mapping, loop.getOperation());
            if (failed(replayed))
              return failure();
            auto source = cast<FragmentType>((*replayed).getType());
            auto target = eraseFragmentAxis(source, axis);
            SmallVector<Attribute> groups;
            unsigned resultAxis = 0;
            for (unsigned sourceAxis = 0;
                 sourceAxis < source.getShape().size(); ++sourceAxis) {
              SmallVector<int64_t> resultAxes;
              if (sourceAxis != axis)
                resultAxes.push_back(resultAxis++);
              groups.push_back(ReshapeGroupAttr::get(
                  kernel.getContext(),
                  nested.getDenseI64ArrayAttr({static_cast<int64_t>(sourceAxis)}),
                  nested.getDenseI64ArrayAttr(resultAxes)));
            }
            return Value(nested.create<ReshapeOp>(
                nestedLocation, target, *replayed, nested.getArrayAttr(groups)));
          };
          FailureOr<Value> lhsSlice = slice(lhs, lhsAxis, lhsFact);
          FailureOr<Value> rhsSlice = slice(rhs, rhsAxis, rhsFact);
          if (failed(lhsSlice) || failed(rhsSlice)) {
            failureReason = "a reduction-pair slice could not preserve its value graph";
            failedBody = true;
            return;
          }
          FailureOr<Value> product = build(
              nested, *lhsSlice, *rhsSlice, eraseAxis(lhsReductions, lhsAxis),
              eraseAxis(rhsReductions, rhsAxis), eraseAxis(lhsBatch, lhsAxis),
              eraseAxis(rhsBatch, rhsAxis), carries.front());
          if (failed(product)) {
            failedBody = true;
            if (failureReason.empty())
              failureReason = "nested reduction-pair decomposition failed";
            return;
          }
          nested.create<scf::YieldOp>(nestedLocation, *product);
        };
    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(loop.getBody());
    buildBody(bodyBuilder, location, loop.getInductionVar(), loop.getRegionIterArgs());
    if (failedBody) {
      loop.erase();
      return failure();
    }
    return loop.getResult(0);
  };

  OpBuilder builder(contract);
  FailureOr<Value> replacement = build(
      builder, contract.getLhs(), contract.getRhs(),
      SmallVector<int64_t>(contract.getLhsReductionAxes()),
      SmallVector<int64_t>(contract.getRhsReductionAxes()),
      SmallVector<int64_t>(contract.getLhsBatchAxes()),
      SmallVector<int64_t>(contract.getRhsBatchAxes()), contract.getAccumulator());
  if (failed(replacement))
    return contract.emitOpError(
               "multi-pair contraction could not be decomposed into provider-native contractions: ")
           << failureReason;
  contract.getResult().replaceAllUsesWith(*replacement);
  contract.erase();
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

FailureOr<ParameterOp>
regionContractionParameter(func::FuncOp kernel, PhysicalExprAttr extent) {
  if (!extent ||
      extent.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Parameter))
    return failure();
  FailureOr<ParameterOp> parameter =
      queryParameterBySymbol(kernel, extent.getSymbol());
  if (failed(parameter) ||
      (*parameter).getParameter().getRole() !=
          static_cast<uint32_t>(ParameterRole::ScanChunk) ||
      (*parameter).getParameter().getCategory() !=
          static_cast<uint32_t>(ParameterCategory::RegionContraction))
    return failure();
  return *parameter;
}

FailureOr<bool> realizeSegmentNativeReduction(ContractOp contract,
                                              func::FuncOp kernel) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1)
    return false;
  auto segmentParameter = [&](Value operand,
                              int64_t axis) -> FailureOr<ParameterOp> {
    auto fragment = dyn_cast<FragmentType>(operand.getType());
    if (!fragment || axis < 0 ||
        axis >= static_cast<int64_t>(fragment.getShape().size()))
      return failure();
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    return regionContractionParameter(kernel, extent);
  };
  FailureOr<ParameterOp> lhsSegment = segmentParameter(
      contract.getLhs(), contract.getLhsReductionAxes().front());
  FailureOr<ParameterOp> rhsSegment = segmentParameter(
      contract.getRhs(), contract.getRhsReductionAxes().front());
  if (failed(lhsSegment) || failed(rhsSegment) ||
      *lhsSegment != *rhsSegment)
    return false;
  scf::ForOp segmentLoop =
      enclosingRegionContractionSegment(contract.getOperation());
  if (segmentLoop &&
      segmentLoop.getStep().getDefiningOp<ParameterOp>() != *lhsSegment)
    return contract.emitOpError(
               "reduction extent disagrees with its enclosing region-contraction segment"),
           failure();
  if (failed(markNativeCoverage(kernel, contract)))
    return failure();
  return true;
}

LogicalResult realizeReductionTraversal(ContractOp contract, func::FuncOp kernel,
                                        SmallVectorImpl<ContractOp> &replayed);

FailureOr<bool> realizeStructuredNativeReduction(
    ContractOp contract, func::FuncOp kernel,
    SmallVectorImpl<ContractOp> &replayed) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1)
    return false;
  PhysicalProgramAnalysis analysis(kernel);
  bool fullReduction = true;
  bool retainedSource = false;
  for (auto [operand, axis] :
       {std::pair<Value, int64_t>{contract.getLhs(),
                                  contract.getLhsReductionAxes().front()},
        std::pair<Value, int64_t>{contract.getRhs(),
                                  contract.getRhsReductionAxes().front()}}) {
    auto type = cast<FragmentType>(operand.getType());
    auto parameter = parameterForExtent(
        kernel, cast<PhysicalExprAttr>(type.getShape()[axis]));
    fullReduction &= succeeded(parameter) &&
        parameter->getParameter().getRole() ==
            static_cast<uint32_t>(ParameterRole::FullCoverage) &&
        parameter->getParameter().getCategory() ==
            static_cast<uint32_t>(ParameterCategory::Coverage);
    auto mapping = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
    retainedSource |= !analysis.replayability(
        operand, sourceAxisIdentity(mapping), PhysicalReplayScope::ValueGraph,
        /*allowAccesses=*/true, contract, mapping.getDimensionId()).isReplayable();
  }
  if (fullReduction && retainedSource &&
      freeAxesReadyForReductionTraversal(contract, kernel)) {
    if (failed(markNativeCoverage(kernel, contract)))
      return failure();
    return true;
  }
  LoadOp lhsLoad = matrixOperandLoad(contract.getLhs());
  LoadOp rhsLoad = matrixOperandLoad(contract.getRhs());
  if (!lhsLoad || !rhsLoad)
    return false;
  struct SegmentFact {
    bool present = false;
    ParameterOp parameter;
    std::optional<PhysicalSourceAxis> source;
  };
  auto operandSegment = [&](Value operand,
                            ArrayRef<int64_t> reductionAxes)
      -> FailureOr<SegmentFact> {
    SegmentFact result;
    auto fragment = dyn_cast<FragmentType>(operand.getType());
    if (!fragment)
      return failure();
    for (auto [axis, attribute] : llvm::enumerate(fragment.getShape())) {
      if (llvm::is_contained(reductionAxes, static_cast<int64_t>(axis)))
        continue;
      auto extent = cast<PhysicalExprAttr>(attribute);
      FailureOr<ParameterOp> parameter =
          regionContractionParameter(kernel, extent);
      PhysicalRangeFact ranges = analysis.axisRanges(operand, axis);
      bool subregion = llvm::any_of(ranges.roots, [](MakeRangeOp range) {
        return range->hasAttr(sourceSubregionAttr);
      });
      if (failed(parameter) && !subregion)
        continue;
      auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
      PhysicalSourceAxis source = sourceAxisIdentity(mapping);
      if (result.source && !(*result.source == source))
        return failure();
      if (succeeded(parameter) && result.parameter &&
          result.parameter != *parameter)
        return failure();
      if (succeeded(parameter))
        result.parameter = *parameter;
      result.present = true;
      result.source = source;
      if (subregion &&
          llvm::any_of(ranges.roots, [](MakeRangeOp range) {
            return !range->hasAttr(sourceSubregionAttr);
          })) {
        return failure();
      }
    }
    return result;
  };
  FailureOr<SegmentFact> lhsSegment = operandSegment(
      contract.getLhs(), contract.getLhsReductionAxes());
  FailureOr<SegmentFact> rhsSegment = operandSegment(
      contract.getRhs(), contract.getRhsReductionAxes());
  if (failed(lhsSegment) || failed(rhsSegment))
    return contract.emitOpError(
               "operand carries an ambiguous region-contraction segment relation"),
           failure();
  if (lhsSegment->present == rhsSegment->present)
    return false;
  SegmentFact &segment = lhsSegment->present ? *lhsSegment : *rhsSegment;
  // A logical subregion only identifies the operand-local member relation; it
  // is not itself an executable region-contraction segment.  Native segment
  // coverage requires the typed parameter created by the structured traversal
  // realization.  Otherwise leave the contract to ordinary M/N/K blocking,
  // which materializes the subregion traversal explicitly.
  if (!segment.parameter)
    return false;
  scf::ForOp segmentLoop =
      enclosingRegionContractionSegment(contract.getOperation());
  if (segmentLoop && segmentLoop.getStep().getDefiningOp<ParameterOp>() !=
                         segment.parameter)
    return contract.emitOpError(
               "operand extent disagrees with its enclosing region-contraction segment"),
           failure();

  bool lhsInvariant = !lhsSegment->present;
  bool rhsInvariant = !rhsSegment->present;
  if (segmentLoop) {
    DominanceInfo dominance(kernel);
    bool lhsDominates = dominance.dominates(lhsLoad.getOperation(),
                                            segmentLoop.getOperation());
    bool rhsDominates = dominance.dominates(rhsLoad.getOperation(),
                                            segmentLoop.getOperation());
    if (lhsDominates != lhsInvariant || rhsDominates != rhsInvariant)
      return contract.emitOpError(
                 "typed segment ownership disagrees with lexical load invariance"),
             failure();
  }

  Value segmented = lhsInvariant ? contract.getRhs() : contract.getLhs();
  if (reductionUsesOwnershipExtent(
          contract, segmented,
          lhsInvariant ? contract.getRhsReductionAxes()
                       : contract.getLhsReductionAxes())) {
    unsigned lhsAxis = contract.getLhsReductionAxes().front();
    unsigned rhsAxis = contract.getRhsReductionAxes().front();
    if (contract.getLhs().getType().getShape()[lhsAxis] !=
        contract.getRhs().getType().getShape()[rhsAxis])
      return false;
    SmallVector<MakeRangeOp> ranges;
    for (auto [operand, axis] :
         {std::pair<Value, unsigned>{contract.getLhs(), lhsAxis},
          std::pair<Value, unsigned>{contract.getRhs(), rhsAxis}}) {
      FailureOr<MakeRangeOp> range =
          queryExactLogicalRange(analysis.axisRanges(operand, axis));
      if (failed(range) || !isUnitStepRange(*range) ||
          !isZeroPastLogicalEnd(operand, *range))
        return false;
      ranges.push_back(*range);
    }
    OpBuilder builder(contract);
    Location location = contract.getLoc();
    Value complete;
    for (MakeRangeOp range : ranges) {
      Value span = binary(builder, location, builder.getIndexType(),
                          range.getLogicalStop(), range.getLogicalStart(),
                          BinaryOperator::Subtract);
      Value beginsAtStart = compare(
          builder, location, builder.getI1Type(), range.getStart(),
          range.getLogicalStart(), ComparePredicate::Eq);
      Value coversExtent = compare(builder, location, builder.getI1Type(),
                                    range.getExtent(), span,
                                    ComparePredicate::Ge);
      Value covered = binary(builder, location, builder.getI1Type(),
                              beginsAtStart, coversExtent,
                              BinaryOperator::LogicalAnd);
      complete = complete
                     ? binary(builder, location, builder.getI1Type(), complete,
                              covered, BinaryOperator::LogicalAnd)
                     : covered;
    }
    auto choice = builder.create<scf::IfOp>(
        location, TypeRange{contract.getResult().getType()}, complete,
        /*withElseRegion=*/true);
    OpBuilder native = choice.getThenBodyBuilder();
    auto product = cast<ContractOp>(native.clone(*contract));
    native.create<scf::YieldOp>(location, product.getResult());
    OpBuilder blocked = choice.getElseBodyBuilder();
    auto reduction = cast<ContractOp>(blocked.clone(*contract));
    blocked.create<scf::YieldOp>(location, reduction.getResult());
    if (failed(realizeReductionTraversal(reduction, kernel, replayed)))
      return failure();
    contract.getResult().replaceAllUsesWith(choice.getResult(0));
    contract.erase();
    return true;
  }

  Value invariant = lhsInvariant ? contract.getLhs() : contract.getRhs();
  unsigned reductionAxis = static_cast<unsigned>(
      lhsInvariant ? contract.getLhsReductionAxes().front()
                   : contract.getRhsReductionAxes().front());
  auto invariantType = cast<FragmentType>(invariant.getType());
  auto reductionMapping =
      cast<AxisMapAttr>(invariantType.getAxisMaps()[reductionAxis]);
  LoadOp invariantLoad = lhsInvariant ? lhsLoad : rhsLoad;
  PhysicalAxisProjection loadProjection = queryFragmentAxis(
      invariantLoad.getResult().getType(),
      sourceAxisIdentity(reductionMapping));
  if (!loadProjection.isExact() ||
      failed(realizeFullCoverageDimension(
          kernel, invariantLoad.getResult(), loadProjection.fragmentAxis)))
    return contract.emitOpError(
        "structured contraction invariant has no exact full-coverage reduction realization");
  if (failed(markNativeCoverage(kernel, contract)))
    return failure();
  return true;
}

/// Block only the reduction traversal of a contraction whose result axes have
/// already been materialized by pointwise ownership.  This form is used when
/// several contractions feed one pure output expression: each term keeps its
/// own K loop while the shared free/batch tile and final store remain in the
/// ordinary value graph.
LogicalResult realizeReductionTraversal(ContractOp contract,
                                        func::FuncOp kernel,
                                        SmallVectorImpl<ContractOp> &replayed) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1)
    return contract.emitOpError(
        "reduction-only contraction blocking requires one reduction pair");

  FragmentType lhsType = contract.getLhs().getType();
  FragmentType rhsType = contract.getRhs().getType();
  FragmentType resultType = contract.getResult().getType();
  // Selected tiles can be clamped by a launch dimension. Their typed
  // realization, rather than a constant-only expression, owns the free axes.
  if (!freeAxesReadyForReductionTraversal(contract, kernel))
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
  SmallVector<MakeRangeOp> lhsRanges;
  SmallVector<MakeRangeOp> rhsRanges;
  if (failed(collectPairedReductionRanges(contract, lhsRanges, rhsRanges))) {
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

  // These are logical full ranges, even when their current fragment has a
  // constant padded extent. Establish the identity before replaying the
  // producer graph into K tiles; masking loads alone does not neutralize exp.
  if (failed(neutralizeFullCoverageOperand(
          contract, contract.getLhsMutable(), contract.getLhsReductionAxes(),
          kernel)) ||
      failed(neutralizeFullCoverageOperand(
          contract, contract.getRhsMutable(), contract.getRhsReductionAxes(),
          kernel)))
    return failure();

  std::string suffix =
      ("_" + Twine(lhsMap->getSourceId()) + "_" +
       Twine(rhsMap->getSourceId()))
          .str();
  ParameterOp blockK = getOrCreatePhysicalParameter(
      kernel, "BLOCK_K" + suffix, ParameterRole::Reduction,
      ParameterCategory::Contraction,
      lhsType.getElementType().getIntOrFloatBitWidth(),
      contractionReductionCandidates);
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
      context, IndexType::get(context), {unitK},
      {cast<AxisMapAttr>(lhsRange.getResult().getType().getAxisMaps()[0])},
      lhsType.getOwner());
  auto rhsIndexType = fragmentType(
      context, IndexType::get(context), {unitK},
      {cast<AxisMapAttr>(rhsRange.getResult().getType().getAxisMaps()[0])},
      rhsType.getOwner());

  Location location = contract.getLoc();
  OpBuilder builder(contract);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  bool bodyFailed = false;
  SmallVector<ContractOp> nativeProducts;
  auto loop = builder.create<scf::ForOp>(
      location, lhsRange.getLogicalStart(), *logicalEnd, blockK.getResult(),
      ValueRange{contract.getAccumulator()},
      [&](OpBuilder &nested, Location nestedLocation, Value kStart,
          ValueRange carries) {
        Value lhsK = nested.create<MakeRangeOp>(
            nestedLocation, lhsIndexType, kStart, blockK.getResult(), one,
            lhsRange.getLogicalStart(), lhsRange.getLogicalStop(),
            lhsRange.getSourceId(), lhsRange.getSourceAxis(),
            lhsRange.getDerived());
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
            rhsRange.getSourceId(), rhsRange.getSourceAxis(),
            rhsRange.getDerived());
        inheritRangeAuthority(rhsK, rhsRange);
        FailureOr<Value> lhsTail = buildRangeTailPredicate(
            nested, nestedLocation, lhsK, lhsRange);
        FailureOr<Value> rhsTail = buildRangeTailPredicate(
            nested, nestedLocation, rhsK, rhsRange);
        if (failed(lhsTail) || failed(rhsTail)) {
          bodyFailed = true;
          return;
        }
        IRMapping lhsReplay;
        IRMapping rhsReplay;
        for (MakeRangeOp range : lhsRanges)
          lhsReplay.map(range.getResult(), lhsK);
        for (MakeRangeOp range : rhsRanges)
          rhsReplay.map(range.getResult(), rhsK);
        FailureOr<Value> lhs = replaySourceValue(
            nested, nestedLocation, contract.getLhs(), unitK, lhsRanges, lhsK,
            lhsReplay, contract.getOperation());
        FailureOr<Value> rhs = replaySourceValue(
            nested, nestedLocation, contract.getRhs(), unitK, rhsRanges, rhsK,
            rhsReplay, contract.getOperation());
        if (failed(lhs) || failed(rhs)) {
          bodyFailed = true;
          return;
        }
        if (failed(appendTailValidity(nestedLocation, contract.getLhs(),
                                      lhsRanges, *lhsTail, lhsReplay)) ||
            failed(appendTailValidity(nestedLocation, contract.getRhs(),
                                      rhsRanges, *rhsTail, rhsReplay))) {
          bodyFailed = true;
          return;
        }
        auto product = nested.create<ContractOp>(
            nestedLocation, resultType, *lhs, *rhs, carries.front(),
            contract.getLhsReductionAxes(), contract.getRhsReductionAxes(),
            contract.getLhsBatchAxes(), contract.getRhsBatchAxes());
        nativeProducts.push_back(product);
        if (Attribute origin = contract->getAttr(originAttr))
          product->setAttr(originAttr, origin);
        nested.create<scf::YieldOp>(nestedLocation, product.getResult());
      });
  if (bodyFailed) {
    loop.erase();
    return contract.emitOpError(
        "reduction-only contraction blocking could not replay its load graph");
  }
  loop.walk([&](ContractOp nested) {
    if (!llvm::is_contained(nativeProducts, nested))
      replayed.push_back(nested);
  });

  contract.getResult().replaceAllUsesWith(loop.getResult(0));
  contract.erase();
  return success();
}

bool canReplayContractionReads(ContractOp contract) {
  SmallVector<Value> pending(contract->getOperands());
  llvm::DenseSet<Value> seen;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    Operation *producer = value.getDefiningOp();
    if (!producer)
      continue;
    if (auto load = dyn_cast<LoadOp>(producer);
        load && !canReplayReadAt(load, contract))
      return false;
    llvm::append_range(pending, producer->getOperands());
  }
  return true;
}

bool fullReductionNeedsTraversal(ContractOp contract) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1)
    return false;
  auto kernel = contract->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  int64_t smallestTile = *std::min_element(
      std::begin(contractionReductionCandidates),
      std::end(contractionReductionCandidates));
  SmallVector<MakeRangeOp> paired;
  for (auto [operand, axis] :
       {std::pair<Value, int64_t>{contract.getLhs(),
                                  contract.getLhsReductionAxes().front()},
        std::pair<Value, int64_t>{contract.getRhs(),
                                  contract.getRhsReductionAxes().front()}}) {
    auto type = cast<FragmentType>(operand.getType());
    auto physical = cast<PhysicalExprAttr>(type.getShape()[axis]);
    auto extent = constantPhysicalExpression(physical);
    if (extent && *extent <= smallestTile && *extent > 0 &&
        llvm::isPowerOf2_64(*extent))
      return false;
    PhysicalRangeFact ranges = analysis.axisRanges(operand, axis);
    if (!ranges.unitStep || failed(queryExactLogicalRange(ranges)))
      return false;
    for (MakeRangeOp range : ranges.roots) {
      auto size = constantLogicalRangeCardinality(range);
      bool complete = size && extent && *size > 0 && *size <= *extent;
      complete |= isZeroScalar(range.getLogicalStart()) &&
                  samePhysicalScalarExpression(range.getExtent(),
                                               range.getLogicalStop());
      if (!complete || queryLaunchExpression(range.getExtent()) != physical ||
          !samePhysicalScalarExpression(range.getStart(),
                                        range.getLogicalStart()))
        return false;
      paired.push_back(range);
    }
  }
  // A complete logical extent does not select a hardware reduction tile.
  // Only retile a complete range here; an existing segment retains its bounds.
  return analysis.lockstepRanges(paired).isExact() &&
         canReplayContractionReads(contract);
}

FailureOr<bool> realizeFullResultTraversal(
    ContractOp contract, func::FuncOp kernel,
    SmallVectorImpl<ContractOp> &pending) {
  if (!contract.getLhsBatchAxes().empty() ||
      !contract.getRhsBatchAxes().empty())
    return false;
  FragmentType resultType = contract.getResult().getType();
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  uint64_t retainedBytes =
      (resultType.getElementType().getIntOrFloatBitWidth() + 7) / 8;
  bool boundedResult = static_cast<bool>(capabilities);
  for (Attribute attribute : resultType.getShape()) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    auto width = constantPhysicalExpression(extent);
    if (!width && extent.getKind() ==
                      static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
      auto parameter = parameterForExtent(kernel, extent);
      if (succeeded(parameter)) {
        auto candidates = (*parameter).getParameter().getCandidates().asArrayRef();
        if (!candidates.empty())
          width = *std::min_element(candidates.begin(), candidates.end());
      }
    }
    if (!width || *width <= 0 ||
        retainedBytes > std::numeric_limits<uint64_t>::max() / *width) {
      boundedResult = false;
      break;
    }
    retainedBytes *= *width;
  }
  // Introduce retained-result traversal only when even the smallest existing
  // fragment exceeds the local budget; larger candidates do not force a loop
  // on otherwise small ownership tiles.
  const bool largeRetainedResult =
      boundedResult && retainedBytes >
                           static_cast<uint64_t>(
                               capabilities.getMaxDynamicSharedMemoryPerBlock());
  auto needsTraversal = [&](PhysicalExprAttr extent) {
    return isFullCoverageExtent(contract, extent) ||
           (largeRetainedResult && extent.getKind() ==
                                      static_cast<uint32_t>(PhysicalExprKind::Constant) &&
            extent.getValue() > 1);
  };
  if (llvm::none_of(resultType.getShape(), [&](Attribute extent) {
        return needsTraversal(cast<PhysicalExprAttr>(extent));
      }))
    return false;
  SmallVector<std::pair<OpOperand *, unsigned>> freeAxes(resultType.getShape().size());
  unsigned freeAxisCount = 0;
  for (auto [operand, reductions] :
       {std::pair<OpOperand *, ArrayRef<int64_t>>{&contract.getLhsMutable(),
                                          contract.getLhsReductionAxes()},
        std::pair<OpOperand *, ArrayRef<int64_t>>{&contract.getRhsMutable(),
                                          contract.getRhsReductionAxes()}}) {
    auto type = cast<FragmentType>(operand->get().getType());
    for (unsigned axis = 0; axis < type.getShape().size(); ++axis) {
      if (llvm::is_contained(reductions, static_cast<int64_t>(axis)))
        continue;
      // Contract results concatenate lhs and rhs free axes. Source identities
      // may repeat, including when the same matrix supplies both operands.
      unsigned resultAxis = freeAxisCount;
      if (resultAxis >= freeAxes.size() ||
          resultType.getShape()[resultAxis] != type.getShape()[axis])
        return false;
      freeAxes[resultAxis] = {operand, axis};
      ++freeAxisCount;
    }
  }
  if (freeAxisCount != resultType.getShape().size())
    return contract.emitOpError(
        "full-result traversal lost its free-axis relation");
  // Slicing may replay loads at the contraction.  External views can alias,
  // so retain their original snapshots across every intervening write.
  if (!canReplayContractionReads(contract))
    return false;

  for (auto [resultAxis, selected] : llvm::enumerate(freeAxes)) {
    auto fullExtent = cast<PhysicalExprAttr>(resultType.getShape()[resultAxis]);
    if (!needsTraversal(fullExtent))
      continue;
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact fact = analysis.axisRanges(selected.first->get(), selected.second);
    auto traversal = analysis.lockstepRanges(fact.roots);
    FailureOr<MakeRangeOp> authority =
        fact.state != PhysicalFactState::Unknown && fact.blockers.empty() &&
                traversal.isExact()
            ? FailureOr<MakeRangeOp>(traversal.authority)
            : FailureOr<MakeRangeOp>(failure());
    FailureOr<ParameterOp> fullParameter = parameterForExtent(kernel, fullExtent);
    FailureOr<AxisMapAttr> axisMap =
        queryAxisMap(selected.first->get().getType(), selected.second);
    if (failed(authority) || failed(axisMap) ||
        (failed(fullParameter) && fullExtent.getKind() !=
                                     static_cast<uint32_t>(PhysicalExprKind::Constant)) ||
        !fact.unitStep ||
        !samePhysicalScalarExpression((*authority).getStart(),
                                      (*authority).getLogicalStart()))
      continue;
    if (failed(fullParameter)) {
      auto cardinality = constantLogicalRangeCardinality(*authority);
      UniformValueAnalysis constants(describeUniformValue);
      auto actualExtent = dyn_cast_or_null<IntegerAttr>(
          constants.evaluate((*authority).getExtent()));
      if (!cardinality || *cardinality <= 0 ||
          *cardinality > fullExtent.getValue() || !actualExtent ||
          actualExtent.getInt() != fullExtent.getValue())
        continue;
    }
    bool lhsAxis = selected.first == &contract.getLhsMutable();
    std::string name =
        ((lhsAxis ? "BLOCK_M_CACHE_" : "BLOCK_N_CACHE_") +
         Twine(axisMap->getSourceId()) + "_" + Twine(axisMap->getSourceAxis()) +
         "_" + Twine(axisMap->getDerived() ? 1 : 0) + "_D" +
         Twine(axisMap->getDimensionId()))
            .str();
    Type inputElement = cast<FragmentType>(selected.first->get().getType())
                            .getElementType();
    ParameterOp block = getOrCreatePhysicalParameter(
        kernel, name,
        lhsAxis ? ParameterRole::OwnershipM : ParameterRole::OwnershipN,
        ParameterCategory::Contraction,
        inputElement.isIndex() ? 64 : inputElement.getIntOrFloatBitWidth(),
        {32, 64, 128, 256, 512});
    if (!block)
      return failure();
    if (FailureOr<int64_t> dimension = queryRangeDimension(*authority);
        succeeded(dimension))
      block->setAttr(dimensionAttr,
                     IntegerAttr::get(IntegerType::get(kernel.getContext(), 64),
                                      *dimension));
    auto tileExtent = parameterExpression(kernel.getContext(), name);
    SmallVector<Attribute> tileShape(resultType.getShape().begin(),
                                     resultType.getShape().end());
    tileShape[resultAxis] = tileExtent;
    auto tileResultType = FragmentType::get(
        kernel.getContext(), resultType.getElementType(),
        ArrayAttr::get(kernel.getContext(), tileShape), resultType.getAxisMaps(),
        resultType.getValidity(), resultType.getOwner());
    auto fullRangeType = FragmentType::get(
        kernel.getContext(), IndexType::get(kernel.getContext()),
        ArrayAttr::get(kernel.getContext(), {fullExtent}),
        ArrayAttr::get(kernel.getContext(), {AxisMapAttr::get(
            kernel.getContext(), axisMap->getSourceId(), axisMap->getSourceAxis(),
            axisMap->getDimensionId(), 0, axisMap->getDerived())}),
        resultType.getValidity(), resultType.getOwner());
    auto tileRangeType = FragmentType::get(
        kernel.getContext(), fullRangeType.getElementType(),
        ArrayAttr::get(kernel.getContext(), {tileExtent}), fullRangeType.getAxisMaps(),
        fullRangeType.getValidity(), fullRangeType.getOwner());
    auto indexType = FragmentType::get(
        kernel.getContext(), IndexType::get(kernel.getContext()),
        resultType.getShape(), resultType.getAxisMaps(), resultType.getValidity(),
        resultType.getOwner());
    auto predicateType = FragmentType::get(
        kernel.getContext(), IntegerType::get(kernel.getContext(), 1),
        resultType.getShape(), resultType.getAxisMaps(), resultType.getValidity(),
        resultType.getOwner());
    OpBuilder builder(contract);
    Location location = contract.getLoc();
    Value fullSize = succeeded(fullParameter)
                         ? (*fullParameter).getResult()
                         : Value(builder.create<arith::ConstantIndexOp>(
                               location, fullExtent.getValue()));
    Value fullRange = builder.create<MakeRangeOp>(
        location, fullRangeType, (*authority).getLogicalStart(),
        fullSize, (*authority).getStep(),
        (*authority).getLogicalStart(), (*authority).getLogicalStop(),
        axisMap->getSourceId(), axisMap->getSourceAxis(), axisMap->getDerived());
    inheritRangeAuthority(fullRange, *authority);
    Value coordinates = broadcastAxis(builder, location, indexType, fullRange,
                                      static_cast<unsigned>(resultAxis));
    // A retained output that fits exactly in the selected native tile needs
    // no slice/update traversal. This binding is launch-uniform and becomes
    // a compile-time branch in the target DSL. Keep the original snapshot and
    // accumulator directly in the native product.
    scf::IfOp fullTile;
    PhysicalProgramAnalysis nativeAnalysis(kernel);
    bool hasReductionSeed = false;
    for (auto [operand, axes] :
         {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
          std::pair{contract.getRhs(), contract.getRhsReductionAxes()}})
      for (int64_t axis : axes)
        hasReductionSeed |=
            nativeAnalysis.axisRealization(operand, axis).constructionScalarSeed;
    // A scalar construction seed is not a selected reduction tile. Retain
    // the traversal so its reduction can be blocked before using native MMA.
    if (!hasReductionSeed && !fullReductionNeedsTraversal(contract)) {
      Value fits = builder.create<CompareOp>(location, builder.getI1Type(),
          fullSize, block.getResult(), ComparePredicate::Eq);
      fullTile = builder.create<scf::IfOp>(location, TypeRange{resultType}, fits, true);
      builder.setInsertionPointToStart(fullTile.thenBlock());
      auto native = cast<ContractOp>(builder.clone(*contract));
      if (failed(markNativeCoverage(kernel, native))) return failure();
      builder.create<scf::YieldOp>(location, native.getResult());
      builder.setInsertionPointToStart(fullTile.elseBlock());
    }
    // Each iteration fills one disjoint slice of an immutable full result.
    // The operand and dot fragments retain their independent, bounded tiles.
    auto loop = builder.create<scf::ForOp>(
        location, (*authority).getLogicalStart(), (*authority).getLogicalStop(),
        block.getResult(), ValueRange{contract.getAccumulator()},
        [](OpBuilder &body, Location location, Value, ValueRange carries) {
          body.create<scf::YieldOp>(location, carries);
        });
    Operation *yield = loop.getBody()->getTerminator();
    OpBuilder nested(yield);
    Value tileRange = nested.create<MakeRangeOp>(
        location, tileRangeType, loop.getInductionVar(), block.getResult(),
        (*authority).getStep(), (*authority).getLogicalStart(),
        (*authority).getLogicalStop(), axisMap->getSourceId(),
        axisMap->getSourceAxis(), axisMap->getDerived());
    inheritRangeAuthority(tileRange, *authority);
    FailureOr<Value> tail =
        buildRangeTailPredicate(nested, location, tileRange, *authority);
    if (failed(tail))
      return failure();
    Value origin = nested.create<BroadcastOp>(
        location, tileRangeType, (*authority).getLogicalStart());
    Value ordinal = binary(nested, location, tileRangeType, tileRange, origin,
                           BinaryOperator::Subtract);
    SmallVector<Value> operands;
    for (OpOperand *operand :
         {&contract.getLhsMutable(), &contract.getRhsMutable(),
          &contract.getAccumulatorMutable()}) {
      Value value = operand->get();
      bool accumulator = operand == &contract.getAccumulatorMutable();
      if (operand != selected.first && !accumulator) {
        operands.push_back(value);
        continue;
      }
      unsigned axis = accumulator ? resultAxis : selected.second;
      auto source = cast<FragmentType>(value.getType());
      SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
      shape[axis] = tileExtent;
      auto slicedType = FragmentType::get(
          kernel.getContext(), source.getElementType(), nested.getArrayAttr(shape),
          source.getAxisMaps(), source.getValidity(), source.getOwner());
      FailureOr<Value> fill = materializeZeroFragment(nested, location, slicedType);
      if (failed(fill))
        return failure();
      if (accumulator && isLiteralZeroProjection(value)) {
        operands.push_back(*fill);
        continue;
      }
      auto sourceMap = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
      auto coordinateType = FragmentType::get(
          kernel.getContext(), nested.getIndexType(), nested.getArrayAttr({tileExtent}),
          nested.getArrayAttr({AxisMapAttr::get(
              kernel.getContext(), sourceMap.getSourceId(), sourceMap.getSourceAxis(),
              sourceMap.getDimensionId(), 0, sourceMap.getDerived())}),
          source.getValidity(), source.getOwner());
      Value coordinate = nested.create<ReshapeOp>(
          location, coordinateType, ordinal, nested.getArrayAttr({ReshapeGroupAttr::get(
              kernel.getContext(), nested.getDenseI64ArrayAttr({0}),
              nested.getDenseI64ArrayAttr({0}))}));
      auto valid = projectPredicateToFragmentAxis(nested, location, *tail, slicedType, axis);
      if (failed(valid))
        return failure();
      auto coordinatePredicate = FragmentType::get(
          kernel.getContext(), nested.getI1Type(), coordinateType.getShape(),
          coordinateType.getAxisMaps(), coordinateType.getValidity(), coordinateType.getOwner());
      Value zero = nested.create<arith::ConstantIndexOp>(location, 0);
      Value lower = nested.create<BroadcastOp>(location, coordinateType, zero);
      Value upper = nested.create<BroadcastOp>(location, coordinateType, fullSize);
      Value bounded = binary(nested, location, coordinatePredicate,
          compare(nested, location, coordinatePredicate, coordinate, lower, ComparePredicate::Ge),
          compare(nested, location, coordinatePredicate, coordinate, upper, ComparePredicate::Lt),
          BinaryOperator::LogicalAnd);
      auto projectedBounds = projectPredicateToFragmentAxis(nested, location, bounded, slicedType, axis);
      if (failed(projectedBounds))
        return failure();
      Value activeSlice = binary(nested, location, (*valid).getType(), *valid,
                                 *projectedBounds, BinaryOperator::LogicalAnd);
      operands.push_back(nested.create<GatherOp>(
          location, slicedType, value, ValueRange{coordinate}, activeSlice, *fill,
          ArrayRef<int64_t>{static_cast<int64_t>(axis)}));
    }
    FailureOr<Value> accumulator = projectPhysicalValueToSchema(
        nested, location, operands[2], tileResultType);
    if (failed(accumulator))
      return contract.emitOpError(
          "full-result traversal could not project its accumulator slice");
    auto tile = nested.create<ContractOp>(
        location, tileResultType, operands[0], operands[1],
        *accumulator, contract.getLhsReductionAxes(), contract.getRhsReductionAxes(),
        contract.getLhsBatchAxes(), contract.getRhsBatchAxes());
    if (Attribute origin = contract->getAttr(originAttr))
      tile->setAttr(originAttr, origin);
    Value start = nested.create<BroadcastOp>(location, indexType,
                                            loop.getInductionVar());
    Value local = binary(nested, location, indexType, coordinates, start,
                         BinaryOperator::Subtract);
    Value zero = nested.create<arith::ConstantIndexOp>(location, 0);
    Value zeroes = nested.create<BroadcastOp>(location, indexType, zero);
    Value width = nested.create<BroadcastOp>(location, indexType, block.getResult());
    Value stop = nested.create<BroadcastOp>(location, indexType,
                                           (*authority).getLogicalStop());
    Value lower = compare(nested, location, predicateType, local, zeroes,
                          ComparePredicate::Ge);
    Value upper = compare(nested, location, predicateType, local, width,
                          ComparePredicate::Lt);
    Value active = compare(nested, location, predicateType, coordinates, stop,
                           ComparePredicate::Lt);
    Value valid = binary(nested, location, predicateType, lower, upper,
                         BinaryOperator::LogicalAnd);
    valid = binary(nested, location, predicateType, valid, active,
                   BinaryOperator::LogicalAnd);
    Value assembled = nested.create<GatherOp>(
        location, resultType, tile.getResult(), ValueRange{local}, valid,
        loop.getRegionIterArgs().front(),
        ArrayRef<int64_t>{static_cast<int64_t>(resultAxis)});
    yield->setOperands(ValueRange{assembled});
    loop.walk([&](ContractOp product) { pending.push_back(product); });
    if (fullTile) {
      builder.setInsertionPointAfter(loop);
      builder.create<scf::YieldOp>(location, loop.getResult(0));
    }
    contract.getResult().replaceAllUsesWith(fullTile ? fullTile.getResult(0) : loop.getResult(0));
    contract.erase();
    return true;
  }
  return false;
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
            compressedRanges, compressedK, compressedReplay,
            contract.getOperation());
        FailureOr<Value> rhs = replaySourceValue(
            nested, nestedLocation, contract.getRhs(), unitDenseK, denseRanges,
            denseK, denseReplay, contract.getOperation());
        SmallVector<Value> replayedMetadata;
        for (Value component : metadataComponents) {
          FailureOr<Value> replayed = replaySourceValue(
              nested, nestedLocation, component, unitMetadataK, metadataRanges,
              metadataK, metadataReplay, contract.getOperation());
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
                                      compressedRanges, *compressedTail,
                                      compressedReplay)) ||
            failed(appendTailValidity(nestedLocation, contract.getRhs(),
                                      denseRanges, *denseTail, denseReplay)) ||
            failed(appendTailValidity(nestedLocation, contract.getMetadata(),
                                      metadataRanges, *metadataTail,
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

LogicalResult realizeContract(ContractOp contract, func::FuncOp kernel,
                              SmallVectorImpl<ContractOp> &pending) {
  if (!contract->getBlock())
    return success();
  llvm::SmallPtrSet<Operation *, 16> existingProducts;
  kernel.walk([&](ContractOp product) {
    existingProducts.insert(product.getOperation());
  });
  const bool required = requiresPhysicalRealization(contract) ||
                        outputCoordinatesNeedRealization(contract);
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
           << "; selected_free_axes="
           << freeAxesReadyForReductionTraversal(contract, kernel);
  };
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1 ||
      !contract.getLhsBatchAxes().empty() ||
      !contract.getRhsBatchAxes().empty())
    return unhandled("requires one reduction pair and no batch axes");

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
      rhsReductionMap->getDimensionId() <= 0)
    return unhandled("paired reduction axes have no logical dimensions");

  auto matrixAccess = [&](Value operand, AxisMapAttr free,
                          AxisMapAttr reduction) -> LoadOp {
    auto replay = PhysicalProgramAnalysis(kernel).replayability(
        operand, std::nullopt, PhysicalReplayScope::ValueGraph,
        /*allowAccesses=*/true);
    if (!replay.isReplayable())
      return {};
    for (Operation *access : replay.accesses) {
      auto load = dyn_cast<LoadOp>(access);
      if (!load)
        continue;
      auto freeCoordinate = accessCoordinatePosition(load, free, operand);
      auto reductionCoordinate = accessCoordinatePosition(load, reduction, operand);
      if (succeeded(freeCoordinate) && succeeded(reductionCoordinate) &&
          *freeCoordinate != *reductionCoordinate)
        return load;
    }
    return {};
  };
  auto lhsLoad = matrixAccess(contract.getLhs(), *rowMap, *lhsReductionMap);
  auto rhsLoad = matrixAccess(contract.getRhs(), *columnMap, *rhsReductionMap);
  if (!lhsLoad || !rhsLoad || !canReplayContractionReads(contract))
    return unhandled("operands have no replayable matrix coordinate accesses");

  FailureOr<unsigned> lhsRowCoordinate =
      accessCoordinatePosition(lhsLoad, *rowMap, contract.getLhs());
  FailureOr<unsigned> lhsReductionCoordinate =
      accessCoordinatePosition(lhsLoad, *lhsReductionMap, contract.getLhs());
  FailureOr<unsigned> rhsReductionCoordinate =
      accessCoordinatePosition(rhsLoad, *rhsReductionMap, contract.getRhs());
  FailureOr<unsigned> rhsColumnCoordinate =
      accessCoordinatePosition(rhsLoad, *columnMap, contract.getRhs());
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
      !PhysicalProgramAnalysis(kernel)
           .lockstepRanges({lhsReductionRange, rhsReductionRange}).isExact())
    return unhandled("physical coordinates are not explicit compatible ranges");
  for (auto [operand, axis, range] : {
           std::tuple<Value, unsigned, MakeRangeOp>{contract.getLhs(), *lhsFree,
                                                   rowRange},
           {contract.getLhs(), lhsReduction, lhsReductionRange},
           {contract.getRhs(), rhsReduction, rhsReductionRange},
           {contract.getRhs(), *rhsFree, columnRange}}) {
    auto selected = PhysicalProgramAnalysis(kernel).rangeAxes(operand, {range});
    if (!selected.isExact() || selected.fragmentAxes != ArrayRef<unsigned>{axis})
      return unhandled("matrix coordinates do not select independent operand axes");
  }
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
  PhysicalExprAttr rowRangeExtent = queryLaunchRangeExtent(rowRange);
  PhysicalExprAttr columnRangeExtent = queryLaunchRangeExtent(columnRange);
  const bool runtimeRowTraversal =
      indirectRow ||
      (!rowRangeExtent &&
       (rowRange->hasAttr(sourceSubregionAttr) ||
        (rowSourceRange && rowSourceRange->hasAttr(sourceSubregionAttr))));
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
  SmallVector<std::pair<MakeRangeOp, Value>> outputTailRanges = {
      {rowRange, *rowLogicalEnd},
      {columnRange, *columnLogicalEnd},
  };

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
  auto sourceSuffix = [](AxisMapAttr mapping) {
    return ("_" + Twine(mapping.getSourceId()) + "_" +
            Twine(mapping.getSourceAxis()) + "_" +
            Twine(static_cast<unsigned>(mapping.getDerived())))
        .str();
  };
  std::string suffix = sourceSuffix(*rowMap);
  suffix += sourceSuffix(*lhsReductionMap);
  suffix += sourceSuffix(*rhsReductionMap);
  suffix += sourceSuffix(*columnMap);
  suffix +=
      ("_" + Twine(static_cast<unsigned>(contractionCategory)) + "_" +
       Twine(static_cast<unsigned>(indirectRow)))
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
      {16, 32, 64, 128, 256});
  ParameterOp blockK = getOrCreatePhysicalParameter(
      kernel, "BLOCK_K" + suffix, ParameterRole::Reduction,
      contractionCategory,
      std::max(lhsType.getElementType().getIntOrFloatBitWidth(),
               rhsType.getElementType().getIntOrFloatBitWidth()),
      contractionReductionCandidates);
  if (!blockM || !blockN || !blockK)
    return failure();
  FailureOr<unsigned> existingRowAxis =
      mappingAxisForScalar(rowRange.getStart(), mapping);
  FailureOr<unsigned> existingColumnAxis =
      mappingAxisForScalar(columnRange.getStart(), mapping);
  Value reductionStart = lhsReductionRange.getStart();
  if (succeeded(mappingAxisForScalar(reductionStart, mapping)))
    reductionStart = lhsReductionRange.getLogicalStart();
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
  ArrayAttr parameterGroup = ArrayAttr::get(
      context,
      {PhysicalSourceAttr::get(context, rowMap->getSourceId(),
                               rowMap->getSourceAxis(), rowMap->getDerived()),
       PhysicalSourceAttr::get(
           context, lhsReductionMap->getSourceId(),
           lhsReductionMap->getSourceAxis(), lhsReductionMap->getDerived()),
       PhysicalSourceAttr::get(
           context, rhsReductionMap->getSourceId(),
           rhsReductionMap->getSourceAxis(), rhsReductionMap->getDerived()),
       PhysicalSourceAttr::get(context, columnMap->getSourceId(),
                               columnMap->getSourceAxis(),
                               columnMap->getDerived()),
       IntegerAttr::get(IntegerType::get(context, 32),
                        static_cast<uint32_t>(contractionCategory)),
       BoolAttr::get(context, indirectRow)});
  for (ParameterOp parameter : {blockM, blockN, blockK})
    parameter->setAttr(parameterGroupAttr, parameterGroup);
  if (rowWorkers)
    rowWorkers->setAttr(parameterGroupAttr, parameterGroup);
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
  if (rowRangeExtent)
    rowExtent = mapBuilder.create<PhysicalExprOp>(
        location, mapBuilder.getIndexType(), rowRangeExtent);
  if (columnRangeExtent)
    columnExtent = mapBuilder.create<PhysicalExprOp>(
        location, mapBuilder.getIndexType(), columnRangeExtent);
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
  if (rowRangeExtent)
    rowExpression = rowRangeExtent;
  if (columnRangeExtent)
    columnExpression = columnRangeExtent;
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
  FragmentType rhsReductionIndexType = fragmentType(
      context, builder.getIndexType(), {unitK}, {*rhsReductionMap},
      rhsType.getOwner());
  FragmentType rowPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM}, {*rowMap}, lhsType.getOwner());
  FragmentType columnPredicateType = fragmentType(
      context, builder.getI1Type(), {unitN}, {*columnMap}, rhsType.getOwner());
  FragmentType reductionPredicateType = fragmentType(
      context, builder.getI1Type(), {unitK}, {*lhsReductionMap},
      lhsType.getOwner());
  FragmentType rhsReductionPredicateType = fragmentType(
      context, builder.getI1Type(), {unitK}, {*rhsReductionMap},
      rhsType.getOwner());
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
  if (!columnRange->hasAttr(sourceSubregionAttr))
    columns.getDefiningOp()->setAttr(programBoundedOriginAttr,
                                    builder.getUnitAttr());
  Value columnValid = rangeBoundsValidity(
      builder, location, columnIndexType, columnPredicateType, columns,
      columnStop);
  auto emitRowBlock = [&](OpBuilder &rowBuilder,
                          Value rowStart) -> LogicalResult {
    Value rows = rowBuilder.create<MakeRangeOp>(
        location, rowIndexType, rowStart, blockM.getResult(), one,
        rowRange.getLogicalStart(), rowRange.getLogicalStop(),
        rowMap->getSourceId(), rowMap->getSourceAxis(), rowMap->getDerived());
    inheritRangeAuthority(rows, rowRange);
    if (!runtimeRowTraversal && !rowRange->hasAttr(sourceSubregionAttr))
      rows.getDefiningOp()->setAttr(programBoundedOriginAttr,
                                   rowBuilder.getUnitAttr());
    Value rowValid = rangeBoundsValidity(rowBuilder, location, rowIndexType,
                                         rowPredicateType, rows, rowStop);
    IRMapping rowReplay;
    if (runtimeRowTraversal)
      rowReplay.map(rowRange.getResult(), rows);
    SmallVector<SmallVector<Value>> replayedStoreCoordinates;
    SmallVector<Value> replayedStoreValidities;
    if (indirectRow) {
      for (AssumeInBoundsOp assumption : rowAssumptions) {
        FailureOr<Value> index = replaySourceValue(
            rowBuilder, location, kernel, assumption.getIndex(),
            sourceAxisIdentity(*rowMap),
            unitM, rowRange, rows, rowReplay);
        if (failed(index))
          return assumption.emitOpError(
              "blocked contraction could not replay an in-bounds assertion");
        auto replacement = rowBuilder.create<AssumeInBoundsOp>(
            location, *index, assumption.getResource(), assumption.getAxis());
        if (Attribute origin = assumption->getAttr(originAttr))
          replacement->setAttr(originAttr, origin);
      }
      for (auto [pathIndex, path] : llvm::enumerate(paths)) {
        SmallVector<Value> coordinates;
        for (Value coordinate : path.store.getCoordinates()) {
          FailureOr<Value> replayed = replaySourceValue(
              rowBuilder, location, kernel, coordinate,
              sourceAxisIdentity(*rowMap),
              unitM, rowRange, rows, rowReplay);
          if (failed(replayed))
            return path.store.emitOpError(
                "blocked contraction could not replay an output coordinate graph");
          coordinates.push_back(*replayed);
        }
        replayedStoreCoordinates.push_back(std::move(coordinates));
        Value validity;
        if (path.store.getValid()) {
          FailureOr<Value> replayed = replaySourceValue(
              rowBuilder, location, kernel, path.store.getValid(),
              sourceAxisIdentity(*rowMap), unitM, rowRange, rows, rowReplay,
              contract.getOperation());
          if (failed(replayed))
            return path.store.emitOpError(
                "blocked contraction could not replay output validity");
          validity = *replayed;
        }
        replayedStoreValidities.push_back(validity);
      }
    }
    Value initialAccumulator = contract.getAccumulator();
    if (failed(scalarSource(initialAccumulator)))
      for (auto [range, extent, coordinates] : {
               std::tuple<MakeRangeOp, PhysicalExprAttr, Value>{rowRange, unitM, rows},
               {columnRange, unitN, columns}}) {
        IRMapping accumulatorReplay;
        accumulatorReplay.map(range.getResult(), coordinates);
        FailureOr<Value> replayed = replaySourceValue(
            rowBuilder, location, initialAccumulator, extent,
            ArrayRef<MakeRangeOp>{range}, coordinates, accumulatorReplay,
            contract.getOperation());
        if (failed(replayed))
          return contract.emitOpError(
              "blocked contraction could not replay its output-dependent accumulator");
        initialAccumulator = *replayed;
      }
    FailureOr<Value> accumulator = projectPhysicalValueToSchema(
        rowBuilder, location, initialAccumulator, blockedResultType);
    if (failed(accumulator))
      return contract.emitOpError(
          "blocked contraction accumulator has no exact result projection");

    std::string loopBodyFailure;
    auto loop = rowBuilder.create<scf::ForOp>(
        location, reductionStart, reductionStop,
        blockK.getResult(), ValueRange{*accumulator},
        [](OpBuilder &body, Location location, Value, ValueRange carries) {
          body.create<scf::YieldOp>(location, carries);
        });
    OpBuilder reductionBuilder(loop.getBody()->getTerminator());
    auto emitReductionBody =
        [&](OpBuilder &nested, Location nestedLocation, Value kStart,
            ValueRange carries) {
          Value reductions = nested.create<MakeRangeOp>(
              nestedLocation, reductionIndexType, kStart, blockK.getResult(), one,
              lhsReductionRange.getLogicalStart(),
              lhsReductionRange.getLogicalStop(),
              lhsReductionMap->getSourceId(), lhsReductionMap->getSourceAxis(),
              lhsReductionMap->getDerived());
          inheritRangeAuthority(reductions, lhsReductionRange);
          Value rhsReductionCoordinates = nested.create<MakeRangeOp>(
              nestedLocation, rhsReductionIndexType, kStart, blockK.getResult(), one,
              rhsReductionRange.getLogicalStart(), rhsReductionRange.getLogicalStop(),
              rhsReductionMap->getSourceId(), rhsReductionMap->getSourceAxis(),
              rhsReductionMap->getDerived());
          inheritRangeAuthority(rhsReductionCoordinates, rhsReductionRange);
          Value reductionValid = rangeBoundsValidity(
              nested, nestedLocation, reductionIndexType,
              reductionPredicateType, reductions, reductionStop);
          Value rhsReductionValid = rangeBoundsValidity(
              nested, nestedLocation, rhsReductionIndexType,
              rhsReductionPredicateType, rhsReductionCoordinates, reductionStop);
          Value lhsRows =
              broadcastAxis(nested, nestedLocation, lhsPredicateType, rowValid, 0);
          Value lhsReductions = broadcastAxis(nested, nestedLocation,
                                          lhsPredicateType, reductionValid, 1);
          Value lhsValid = binary(nested, nestedLocation, lhsPredicateType,
                                  lhsRows, lhsReductions,
                                  BinaryOperator::LogicalAnd);
          Value rhsReductions = broadcastAxis(nested, nestedLocation,
                                          rhsPredicateType, rhsReductionValid, 0);
          Value rhsColumns =
              broadcastAxis(nested, nestedLocation, rhsPredicateType, columnValid, 1);
          Value rhsValid = binary(nested, nestedLocation, rhsPredicateType,
                                  rhsReductions, rhsColumns,
                                  BinaryOperator::LogicalAnd);
          // Replay each operand independently: a shared source can occupy
          // different M/N roles, and pointwise producers remain part of the dot.
          auto replayOperand = [&](Value source, const IRMapping &freeSeeds,
                                   MakeRangeOp freeRange,
                                   Value freeCoordinates,
                                   PhysicalExprAttr freeExtent, Value freeValid,
                                   MakeRangeOp reductionRange,
                                   Value reductionCoordinates, Value kValid,
                                   FragmentType target, Value valid)
              -> FailureOr<Value> {
            IRMapping freeReplay(freeSeeds);
            freeReplay.map(freeRange.getResult(), freeCoordinates);
            auto freeValue = replaySourceValue(
                nested, nestedLocation, source, freeExtent,
                ArrayRef<MakeRangeOp>(freeRange), freeCoordinates, freeReplay,
                contract.getOperation());
            if (failed(freeValue) ||
                failed(appendTailValidity(nestedLocation, source,
                                         ArrayRef<MakeRangeOp>(freeRange),
                                         freeValid, freeReplay)))
              return failure();
            IRMapping reductionReplay;
            reductionReplay.map(reductionRange.getResult(),
                                reductionCoordinates);
            auto value = replaySourceValue(
                nested, nestedLocation, *freeValue, unitK,
                ArrayRef<MakeRangeOp>(reductionRange), reductionCoordinates,
                reductionReplay, loop.getBody()->getTerminator());
            if (failed(value) ||
                failed(appendTailValidity(nestedLocation, *freeValue,
                                         ArrayRef<MakeRangeOp>(reductionRange),
                                         kValid, reductionReplay)))
              return failure();
            auto projected = projectPhysicalValueToSchema(
                nested, nestedLocation, *value, target);
            auto zero = materializeZeroFragment(nested, nestedLocation, target);
            if (failed(projected) || failed(zero))
              return failure();
            return nested
                .create<SelectOp>(nestedLocation, target, valid, *projected, *zero)
                .getResult();
          };
          auto lhs = replayOperand(
              contract.getLhs(), rowReplay, rowRange, rows, unitM, rowValid,
              lhsReductionRange, reductions, reductionValid, blockedLhsType,
              lhsValid);
          auto rhs = replayOperand(
              contract.getRhs(), IRMapping{}, columnRange, columns, unitN,
              columnValid,
              rhsReductionRange, rhsReductionCoordinates, rhsReductionValid,
              blockedRhsType, rhsValid);
          if (failed(lhs) || failed(rhs)) {
            loopBodyFailure = "operand value graph could not be replayed";
            return;
          }
          Value product = nested.create<ContractOp>(
              nestedLocation, blockedResultType, *lhs, *rhs, carries.front(),
              ArrayRef<int64_t>{1}, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{},
              ArrayRef<int64_t>{});
          loop.getBody()->getTerminator()->setOperands(product);
        };
    emitReductionBody(reductionBuilder, location, loop.getInductionVar(),
                      loop.getRegionIterArgs());
    if (!loopBodyFailure.empty()) {
      loop.erase();
      return contract.emitOpError(
                 "blocked contraction could not materialize its loop body: ")
             << loopBodyFailure;
    }

    Value outputRows =
        broadcastAxis(rowBuilder, location, outputPredicateType, rowValid, 0);
    Value outputColumns =
        broadcastAxis(rowBuilder, location, outputPredicateType, columnValid, 1);
    Value outputValid = binary(rowBuilder, location, outputPredicateType,
                               outputRows, outputColumns,
                               BinaryOperator::LogicalAnd);
    for (auto [pathIndex, path] : llvm::enumerate(paths)) {
      OpBuilder::InsertionGuard storeInsertion(rowBuilder);
      if (!runtimeRowTraversal)
        rowBuilder.setInsertionPoint(path.store);
      auto output = materializeStorePath(
          rowBuilder, kernel, path, contract.getResult(), loop.getResult(0),
          rows, columns,
          runtimeRowTraversal ? contract.getOperation() : path.store.getOperation());
      if (failed(output))
        return path.store.emitOpError(
            "blocked contraction could not replay its pointwise epilogue");
      SmallVector<Value> coordinates =
          indirectRow ? replayedStoreCoordinates[pathIndex]
                      : SmallVector<Value>(path.store.getCoordinates());
      FailureOr<unsigned> storeColumn = directRankOneAccessPosition(
          path.store.getCoordinates(), path.store.getValue().getType(), 1);
      if (failed(storeColumn))
        storeColumn = queryCoordinatePosition(
            path.store.getCoordinates(), sourceAxisIdentity(*columnMap));
      FailureOr<unsigned> storeRow = failure();
      if (!indirectRow) {
        storeRow = directRankOneAccessPosition(
            path.store.getCoordinates(), path.store.getValue().getType(), 0);
        if (failed(storeRow))
          storeRow = queryCoordinatePosition(
              path.store.getCoordinates(), sourceAxisIdentity(*rowMap));
      }
      if (failed(storeColumn))
        return path.store.emitOpError(
            "blocked contract output lost its logical source coordinates");
      SmallVector<std::pair<MakeRangeOp, Value>> storeTailRanges(outputTailRanges);
      if (!indirectRow) {
        if (failed(storeRow))
          return path.store.emitOpError(
              "blocked contract output lost its logical source coordinates");
        for (auto [position, authority, end] : {
                 std::tuple<unsigned, MakeRangeOp, Value>{*storeRow, rowRange, rowStop},
                 std::tuple<unsigned, MakeRangeOp, Value>{*storeColumn, columnRange, columnStop}}) {
          MakeRangeOp stored = sourceRange(path.store.getCoordinates()[position]);
          if (!stored)
            continue;
          FailureOr<Value> storedEnd = resolveLogicalRangeEnd(kernel, stored);
          if (succeeded(storedEnd) && samePhysicalScalarExpression(*storedEnd, end) &&
              samePhysicalScalarExpression(stored.getLogicalStart(), authority.getLogicalStart()) &&
              samePhysicalScalarExpression(stored.getStep(), authority.getStep()))
            storeTailRanges.emplace_back(stored, *storedEnd);
        }
        coordinates[*storeRow] = rows;
      }
      coordinates[*storeColumn] = columns;
      FailureOr<Value> valid = failure();
      if (indirectRow) {
        Value replayed = replayedStoreValidities[pathIndex];
        if (!replayed) {
          valid = outputValid;
        } else {
          IRMapping columnReplay;
          columnReplay.map(columnRange.getResult(), columns);
          FailureOr<Value> columnValidity = replaySourceValue(
              rowBuilder, location, kernel, replayed,
              sourceAxisIdentity(*columnMap), unitN, columnRange, columns,
              columnReplay);
          if (failed(columnValidity))
            return path.store.emitOpError(
                "blocked contraction could not replay output column validity");
          valid = materializeValidityConjunction(
              rowBuilder, location, outputValid, *columnValidity,
              outputPredicateType);
        }
      } else {
        Value originalValidity = path.store.getValid();
        // The new row/column mask covers proven tails. Other predicates are
        // replayed through each exact range occurrence below.
        if (originalValidity &&
            PhysicalProgramAnalysis(kernel).isTailPredicate(originalValidity, storeTailRanges))
          originalValidity = Value();
        if (originalValidity) {
          IRMapping replay;
          replay.map(rowRange.getResult(), rows);
          FailureOr<Value> replayed = replaySourceValue(
              rowBuilder, location, kernel, originalValidity,
              sourceAxisIdentity(*rowMap), unitM, rowRange, rows, replay,
              &*rowBuilder.getInsertionPoint());
          if (failed(replayed))
            return path.store.emitOpError(
                "blocked contraction could not relocate output validity");
          originalValidity = *replayed;
        }
        if (originalValidity) {
          IRMapping replay;
          replay.map(columnRange.getResult(), columns);
          FailureOr<Value> replayed = replaySourceValue(
              rowBuilder, location, kernel, originalValidity,
              sourceAxisIdentity(*columnMap), unitN, columnRange, columns,
              replay, &*rowBuilder.getInsertionPoint());
          if (failed(replayed))
            return path.store.emitOpError(
                "blocked contraction could not relocate output column validity");
          originalValidity = *replayed;
        }
        valid = materializeValidityConjunction(
            rowBuilder, location, outputValid, originalValidity,
            outputPredicateType);
      }
      if (failed(valid))
        return path.store.emitOpError(
            "blocked contract output validity could not be retargeted");
      auto replacement = rowBuilder.create<StoreOp>(
          location, path.store.getResource(), coordinates, *output, *valid,
          path.store.getSourceAxes());
      if (Attribute origin = path.store->getAttr(originAttr))
        replacement->setAttr(originAttr, origin);
    }
    return success();
  };

  if (runtimeRowTraversal) {
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
        location, rowStart, rowStop, rowStep);
    OpBuilder nested(rowLoop.getBody()->getTerminator());
    if (failed(emitRowBlock(nested, rowLoop.getInductionVar()))) {
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
  // Retiling a pointwise epilogue can replay a sibling matrix expression.
  // Retire the obsolete epilogue before its original products are visited,
  // and let the new products receive their own bounded reduction traversal.
  // Keep products themselves alive until their worklist entry is consumed.
  llvm::SmallPtrSet<Operation *, 16> retired;
  for (StorePath &path : paths)
    for (Operation *operation : llvm::reverse(path.operations))
      if (!retired.contains(operation) && operation->use_empty()) {
        retired.insert(operation);
        operation->erase();
      }
  kernel.walk([&](ContractOp product) {
    if (!existingProducts.contains(product.getOperation()))
      pending.push_back(product);
  });
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
  Value rowStop = *rowLogicalEnd;
  Value columnStart = columnRange->getStart();
  if (failed(existingColumnAxis))
    columnStart = binary(
        builder, location, builder.getIndexType(), columnStart,
        binary(builder, location, builder.getIndexType(), columnTile,
               blockN.getResult(), BinaryOperator::Multiply),
        BinaryOperator::Add);
  Value columnStop = *columnLogicalEnd;
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
  if (!(*rowRange)->hasAttr(sourceSubregionAttr))
    rows.getDefiningOp()->setAttr(programBoundedOriginAttr,
                                 builder.getUnitAttr());
  Value columns = builder.create<MakeRangeOp>(
      location, columnIndexType, columnStart, blockN.getResult(), one,
      columnRange->getLogicalStart(), columnRange->getLogicalStop(),
      columnMap->getSourceId(), columnMap->getSourceAxis(),
      columnMap->getDerived());
  inheritRangeAuthority(columns, *columnRange);
  if (!(*columnRange)->hasAttr(sourceSubregionAttr))
    columns.getDefiningOp()->setAttr(programBoundedOriginAttr,
                                    builder.getUnitAttr());
  Value rowValid = rangeBoundsValidity(builder, location, rowIndexType,
                                       rowPredicateType, rows, rowStop);
  Value columnValid = rangeBoundsValidity(
      builder, location, columnIndexType, columnPredicateType, columns,
      columnStop);
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
        Value blockValid = rangeBoundsValidity(
            nested, nestedLocation, blockIndexType, blockPredicateType, blocks,
            blockStop);
        Value lhsRows = broadcastAxis(nested, nestedLocation, lhsPredicateType,
                                  rowValid, 0);
        Value lhsBlocks = broadcastAxis(nested, nestedLocation, lhsPredicateType,
                                    blockValid, 1);
        Value lhsValid = binary(nested, nestedLocation, lhsPredicateType,
                                lhsRows, lhsBlocks,
                                BinaryOperator::LogicalAnd);
        Value lhsScaleRows = broadcastAxis(
            nested, nestedLocation, lhsScalePredicateType, rowValid, 0);
        Value lhsScaleBlocks = broadcastAxis(
            nested, nestedLocation, lhsScalePredicateType, blockValid, 1);
        Value lhsScaleValid = binary(
            nested, nestedLocation, lhsScalePredicateType, lhsScaleRows,
            lhsScaleBlocks, BinaryOperator::LogicalAnd);
        Value rhsBlocks = broadcastAxis(nested, nestedLocation, rhsPredicateType,
                                    blockValid, 0);
        Value rhsColumns = broadcastAxis(nested, nestedLocation, rhsPredicateType,
                                     columnValid, 2);
        Value rhsValid = binary(nested, nestedLocation, rhsPredicateType,
                                rhsBlocks, rhsColumns,
                                BinaryOperator::LogicalAnd);
        Value rhsScaleBlocks = broadcastAxis(
            nested, nestedLocation, rhsScalePredicateType, blockValid, 1);
        Value rhsScaleColumns = broadcastAxis(
            nested, nestedLocation, rhsScalePredicateType, columnValid, 0);
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
            nested, nestedLocation, kernel,
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

  Value outputRows = broadcastAxis(builder, location, outputPredicateType, rowValid, 0);
  Value outputColumns =
      broadcastAxis(builder, location, outputPredicateType, columnValid, 1);
  Value outputValid = binary(builder, location, outputPredicateType, outputRows,
                             outputColumns, BinaryOperator::LogicalAnd);
  for (StorePath &path : paths) {
    OpBuilder::InsertionGuard storeInsertion(builder);
    builder.setInsertionPoint(path.store);
    auto output = materializeStorePath(
        builder, kernel, path, contract.getResult(), loop.getResult(0), rows,
        columns, path.store.getOperation());
    if (failed(output))
      return reject("pointwise epilogue could not be replayed");
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
    Value originalValidity = path.store.getValid();
    if (originalValidity) {
      IRMapping replay;
      replay.map(rowRange->getResult(), rows);
      FailureOr<Value> replayed = replaySourceValue(
          builder, location, kernel, originalValidity,
          sourceAxisIdentity(*rowMap), unitM, *rowRange, rows, replay,
          contract.getOperation());
      if (failed(replayed))
        return reject("result store validity could not be relocated");
      originalValidity = *replayed;
    }
    FailureOr<Value> valid = materializeRetargetedValidity(
        builder, location, originalValidity, outputTailRanges,
        outputValid, outputPredicateType);
    if (failed(valid))
      return reject("result store residual validity could not be retargeted");
    auto replacement = builder.create<StoreOp>(
        location, path.store.getResource(), coordinates, *output, *valid,
        path.store.getSourceAxes());
    if (Attribute origin = path.store->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
  }

  for (StorePath &path : paths)
    path.store.erase();
  return success();
}

Type transposeLoopSchema(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragment.getShape().size() == 2 ? transposeLastTwo(fragment) : type;
  if (auto record = dyn_cast<RecordType>(type)) {
    SmallVector<Attribute> fields;
    for (Attribute field : record.getFieldTypes())
      fields.push_back(TypeAttr::get(
          transposeLoopSchema(cast<TypeAttr>(field).getValue())));
    return RecordType::get(type.getContext(), record.getFieldNames(),
                           ArrayAttr::get(type.getContext(), fields),
                           record.getOwner());
  }
  return type;
}

void matrixFields(Type type, SmallVectorImpl<FragmentType> &fields) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    if (fragment.getShape().size() == 2)
      fields.push_back(fragment);
  } else if (auto record = dyn_cast<RecordType>(type)) {
    for (Attribute field : record.getFieldTypes())
      matrixFields(cast<TypeAttr>(field).getValue(), fields);
  }
}

Value transposeLoopValue(OpBuilder &builder, Location location, Value value) {
  if (auto fragment = dyn_cast<FragmentType>(value.getType())) {
    if (fragment.getShape().size() != 2)
      return value;
    auto target = transposeLastTwo(fragment);
    if (auto splat = value.getDefiningOp<SplatOp>())
      return builder.create<SplatOp>(location, target, splat.getValue());
    return builder.create<TransposeOp>(location, target, value,
                                       ArrayRef<int64_t>{1, 0});
  }
  if (auto record = dyn_cast<RecordType>(value.getType())) {
    SmallVector<Value> fields;
    for (auto [index, type] : llvm::enumerate(record.getFieldTypes())) {
      Value field = builder.create<ExtractOp>(
          location, cast<TypeAttr>(type).getValue(), value, index);
      fields.push_back(transposeLoopValue(builder, location, field));
    }
    return builder.create<MakeRecordOp>(
        location, cast<RecordType>(transposeLoopSchema(record)), fields);
  }
  return value;
}

bool supportsMatrixLoopTranspose(scf::ForOp loop) {
  auto supportedType = [&](Type type) {
    std::function<bool(Type)> supported = [&](Type current) {
      if (auto fragment = dyn_cast<FragmentType>(current))
        return fragment.getShape().size() <= 2;
      if (auto record = dyn_cast<RecordType>(current))
        return llvm::all_of(record.getFieldTypes(), [&](Attribute field) {
          return supported(cast<TypeAttr>(field).getValue());
        });
      return true;
    };
    return supported(type);
  };
  return !loop.walk([&](Operation *operation) {
    if (!llvm::all_of(operation->getOperandTypes(), supportedType) ||
        !llvm::all_of(operation->getResultTypes(), supportedType))
      return WalkResult::interrupt();
    if (auto load = dyn_cast<LoadOp>(operation))
      return isa<ViewType>(load.getResource().getType())
                 ? WalkResult::advance() : WalkResult::interrupt();
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      if (contract.getLhs().getType().getShape().size() != 2 ||
          contract.getRhs().getType().getShape().size() != 2 ||
          !contract.getLhsBatchAxes().empty() ||
          !contract.getRhsBatchAxes().empty() ||
          contract.getLhsReductionAxes().size() != 1 ||
          contract.getRhsReductionAxes().size() != 1)
        return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
      auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
      auto target = dyn_cast<FragmentType>(reshape.getResult().getType());
      if (!source || !target)
        return WalkResult::interrupt();
      auto nonUnit = [](FragmentType type) {
        return llvm::count_if(type.getShape(), [](Attribute extent) {
          auto expression = cast<PhysicalExprAttr>(extent);
          return expression.getKind() !=
                     static_cast<uint32_t>(PhysicalExprKind::Constant) ||
                 expression.getValue() != 1;
        });
      };
      if (source.getShape() != target.getShape() &&
          (nonUnit(source) > 1 || nonUnit(target) > 1))
        return WalkResult::interrupt();
      return succeeded(inferReshapeReassociation(
                 cast<FragmentType>(transposeLoopSchema(source)),
                 cast<FragmentType>(transposeLoopSchema(target))))
                 ? WalkResult::advance() : WalkResult::interrupt();
    }
    if (auto reduce = dyn_cast<ReduceOp>(operation))
      return reduce.getAxes().size() == 1 ? WalkResult::advance()
                                          : WalkResult::interrupt();
    return isa<arith::ConstantOp, scf::ForOp, scf::IfOp, scf::YieldOp,
               UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
               SplatOp, BroadcastOp, TransposeOp, MakeRecordOp, ExtractOp,
               MakeRangeOp, RangeBoundOp, DimOp, AssumeInBoundsOp, YieldOp>(operation)
               ? WalkResult::advance() : WalkResult::interrupt();
  }).wasInterrupted();
}

void transposeClonedLoop(scf::ForOp loop) {
  loop.walk([&](Operation *operation) {
    for (Value result : operation->getResults())
      result.setType(transposeLoopSchema(result.getType()));
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          argument.setType(transposeLoopSchema(argument.getType()));
  });
  loop.walk([&](Operation *operation) {
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      Value lhs = contract.getLhs(), rhs = contract.getRhs();
      auto lhsAxes = remapAxes(contract.getRhsReductionAxes(), {1, 0});
      auto rhsAxes = remapAxes(contract.getLhsReductionAxes(), {1, 0});
      contract->setOperand(0, rhs);
      contract->setOperand(1, lhs);
      contract.setLhsReductionAxesAttr(
          DenseI64ArrayAttr::get(loop.getContext(), lhsAxes));
      contract.setRhsReductionAxesAttr(
          DenseI64ArrayAttr::get(loop.getContext(), rhsAxes));
    } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      auto source = dyn_cast<FragmentType>(reduce.getInputs().front().getType());
      if (source && source.getShape().size() == 2)
        reduce.setAxesAttr(DenseI64ArrayAttr::get(
            loop.getContext(), remapAxes(reduce.getAxes(), {1, 0})));
    } else if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
      auto relation = inferReshapeReassociation(
          cast<FragmentType>(reshape.getValue().getType()),
          cast<FragmentType>(reshape.getResult().getType()));
      assert(succeeded(relation) && "loop transpose preflight checked reshapes");
      reshape.setReassociationAttr(*relation);
    }
  });
}

FailureOr<bool> projectContractResult(ContractOp contract) {
  if (!contract.getResult().hasOneUse())
    return false;
  auto gather = dyn_cast<GatherOp>(*contract.getResult().getUsers().begin());
  auto resultType = contract.getResult().getType();
  if (!gather || gather.getSource() != contract.getResult() ||
      gather.getCoordinates().size() > resultType.getShape().size() ||
      failed(scalarSource(contract.getAccumulator())) ||
      !canReplayContractionReads(contract))
    return false;
  auto gatheredType = dyn_cast<FragmentType>(gather.getResult().getType());
  if (gatheredType && gatheredType.getShape().size() > resultType.getShape().size())
    return false;
  auto kernel = contract->getParentOfType<func::FuncOp>();
  DominanceInfo dominance(kernel);
  llvm::SmallDenseSet<Value> checked;
  std::function<bool(Value)> canMoveCoordinate = [&](Value value) {
    if (dominance.dominates(value, contract.getOperation()))
      return true;
    if (!checked.insert(value).second)
      return true;
    Operation *producer = value.getDefiningOp();
    return producer && producer->getNumRegions() == 0 &&
           !isa<RecordType>(value.getType()) &&
           (gatheredType || !isa<FragmentType>(value.getType())) &&
           (isa<arith::ConstantOp, PhysicalExprOp>(producer) ||
            isPhysicalReplayNode(producer, PhysicalReplayScope::Coordinate,
                                 /*allowAccesses=*/false)) &&
           llvm::all_of(producer->getOperands(), canMoveCoordinate);
  };
  if (!gatheredType && gather.getValid() && !canMoveCoordinate(gather.getValid()))
    return false;
  SmallVector<Value> selected(resultType.getShape().size());
  SmallVector<int64_t> resultOrder(resultType.getShape().size(), -1);
  llvm::SmallDenseSet<unsigned> selectedResultAxes;
  for (auto [coordinate, axis] : llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (auto scalar = scalarSource(coordinate); succeeded(scalar))
      coordinate = *scalar;
    if (axis < 0 || axis >= static_cast<int64_t>(selected.size()) ||
        selected[axis] ||
        !canMoveCoordinate(coordinate))
      return false;
    if (gatheredType && isa<FragmentType>(coordinate.getType())) {
      auto range = coordinate.getDefiningOp<MakeRangeOp>();
      if (!range || !isUnitStepRange(range))
        return false;
      auto projections = queryRangeProjections(gatheredType, range);
      if (projections.size() != 1 ||
          !selectedResultAxes.insert(projections.front().fragmentAxis).second ||
          gatheredType.getShape()[projections.front().fragmentAxis] !=
              range.getResult().getType().getShape()[0])
        return false;
      resultOrder[axis] = projections.front().fragmentAxis;
    } else if (isa<FragmentType>(coordinate.getType())) {
      return false;
    }
    selected[axis] = coordinate;
  }
  for (unsigned axis = 0; axis < selected.size(); ++axis) {
    if (selected[axis])
      continue;
    if (!gatheredType)
      return false;
    auto mapping = cast<AxisMapAttr>(resultType.getAxisMaps()[axis]);
    auto retained = queryFragmentAxis(gatheredType, sourceAxisIdentity(mapping));
    if (!retained.isExact() || retained.dimensionId != mapping.getDimensionId() ||
        gatheredType.getShape()[retained.fragmentAxis] != resultType.getShape()[axis] ||
        !selectedResultAxes.insert(retained.fragmentAxis).second)
      return false;
    resultOrder[axis] = retained.fragmentAxis;
  }
  if (gatheredType && selectedResultAxes.size() != gatheredType.getShape().size())
    return false;
  auto isUnit = [](Attribute attribute) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    return extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
           extent.getValue() == 1;
  };
  SmallVector<SmallVector<unsigned>, 2> resultAxes(2);
  unsigned nextResultAxis = 0;
  unsigned side = 0;
  for (auto [operand, reductions] :
       {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
        std::pair{contract.getRhs(), contract.getRhsReductionAxes()}}) {
    auto type = cast<FragmentType>(operand.getType());
    resultAxes[side].resize(type.getShape().size());
    for (unsigned axis = 0; axis < type.getShape().size(); ++axis) {
      if (llvm::is_contained(reductions, static_cast<int64_t>(axis)))
        continue;
      auto batch = llvm::find(contract.getRhsBatchAxes(), static_cast<int64_t>(axis));
      if (side == 1 && batch != contract.getRhsBatchAxes().end()) {
        unsigned pair = std::distance(contract.getRhsBatchAxes().begin(), batch);
        resultAxes[side][axis] = resultAxes[0][contract.getLhsBatchAxes()[pair]];
      } else {
        resultAxes[side][axis] = nextResultAxis++;
      }
    }
    ++side;
  }
  PhysicalProgramAnalysis analysis(kernel);
  side = 0;
  for (auto [operand, reductions] :
       {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
        std::pair{contract.getRhs(), contract.getRhsReductionAxes()}}) {
    auto type = cast<FragmentType>(operand.getType());
    for (unsigned axis = 0; axis < type.getShape().size(); ++axis) {
      if (llvm::is_contained(reductions, static_cast<int64_t>(axis)))
        continue;
      auto mapping = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
      auto resultMapping = cast<AxisMapAttr>(
          resultType.getAxisMaps()[resultAxes[side][axis]]);
      bool pairedRhsBatch = side == 1 && llvm::is_contained(
          contract.getRhsBatchAxes(), static_cast<int64_t>(axis));
      if (!pairedRhsBatch &&
          (mapping.getDimensionId() <= 0 ||
           mapping.getDimensionId() != resultMapping.getDimensionId()))
        return false;
      if (!selected[resultAxes[side][axis]])
        continue;
      if (isUnit(type.getShape()[axis]) &&
          isIntegerConstant(selected[resultAxes[side][axis]], 0))
        continue;
      auto roots = analysis.axisRanges(operand, axis);
      auto range = queryExactLogicalRange(roots);
      auto occurrences = analysis.rangeAxes(operand, roots.roots);
      if (failed(range) || !isUnitStepRange(*range) ||
          !analysis.lockstepRanges(roots.roots).isExact() ||
          !occurrences.isExact() ||
          occurrences.fragmentAxes != SmallVector<unsigned>{axis} ||
          !isIntegerConstant((*range).getStart(), 0) ||
          !isIntegerConstant((*range).getLogicalStart(), 0) ||
          (!samePhysicalScalarExpression((*range).getExtent(),
                                         (*range).getLogicalStop()) &&
           !isFullCoverageExtent(contract,
                                 cast<PhysicalExprAttr>(type.getShape()[axis]))))
        return false;
    }
    ++side;
  }
  OpBuilder builder(contract);
  IRMapping coordinates;
  std::function<Value(Value)> moveCoordinate = [&](Value value) -> Value {
    if (Value mapped = coordinates.lookupOrNull(value))
      return mapped;
    if (dominance.dominates(value, contract.getOperation()))
      return value;
    Operation *producer = value.getDefiningOp();
    for (Value operand : producer->getOperands())
      coordinates.map(operand, moveCoordinate(operand));
    Operation *clone = builder.clone(*producer, coordinates);
    return clone->getResult(cast<OpResult>(value).getResultNumber());
  };
  for (Value &coordinate : selected)
    if (coordinate)
      coordinate = moveCoordinate(coordinate);
  Value gatherValidity = !gatheredType && gather.getValid()
                             ? moveCoordinate(gather.getValid()) : Value();
  PhysicalExprAttr unit = expression(kernel.getContext(), PhysicalExprKind::Constant, 1);
  Value one = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 1);
  side = 0;
  SmallVector<Value> operands;
  for (auto [original, reductions] :
       {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
        std::pair{contract.getRhs(), contract.getRhsReductionAxes()}}) {
    Value operand = original;
    auto originalType = cast<FragmentType>(original.getType());
    for (unsigned axis = 0; axis < originalType.getShape().size(); ++axis) {
      if (llvm::is_contained(reductions, static_cast<int64_t>(axis)))
        continue;
      Value coordinate = selected[resultAxes[side][axis]];
      if (!coordinate)
        continue;
      if (isUnit(originalType.getShape()[axis]) && isIntegerConstant(coordinate, 0))
        continue;
      auto roots = PhysicalProgramAnalysis(kernel).axisRanges(operand, axis);
      auto authority = queryExactLogicalRange(roots);
      if (failed(authority))
        return contract.emitOpError("contraction projection lost its free-axis range");
      auto mapping = cast<AxisMapAttr>(originalType.getAxisMaps()[axis]);
      unsigned resultAxis = resultAxes[side][axis];
      auto selectedRange = coordinate.getDefiningOp<MakeRangeOp>();
      auto projectedExtent = selectedRange
                                 ? cast<PhysicalExprAttr>(
                                       gatheredType.getShape()[resultOrder[resultAxis]])
                                 : unit;
      auto projectedMapping = selectedRange
                                  ? cast<AxisMapAttr>(
                                        gatheredType.getAxisMaps()[resultOrder[resultAxis]])
                                  : mapping;
      auto coordinateType = fragmentType(kernel.getContext(), builder.getIndexType(),
                                          {projectedExtent}, {projectedMapping},
                                          originalType.getOwner());
      auto range = builder.create<MakeRangeOp>(
          contract.getLoc(), coordinateType,
          selectedRange ? selectedRange.getStart() : coordinate,
          selectedRange ? selectedRange.getExtent() : one, one,
          (*authority).getLogicalStart(), (*authority).getLogicalStop(),
          projectedMapping.getSourceId(), projectedMapping.getSourceAxis(),
          projectedMapping.getDerived());
      inheritRangeAuthority(range, *authority);
      IRMapping replay;
      for (MakeRangeOp root : roots.roots)
        replay.map(root.getResult(), range.getResult());
      ReplayMaterializationOptions options;
      options.fragmentAxis = axis;
      options.traversalRanges = roots.roots;
      if (selectedRange)
        options.segmentMapping = projectedMapping;
      auto tail = buildRangeTailPredicate(builder, contract.getLoc(), range, *authority);
      if (failed(tail))
        return contract.emitOpError("contraction projection has no range validity");
      options.segmentTail = *tail;
      if (gatherValidity) {
        auto valid = materializeValidityConjunction(
            builder, contract.getLoc(), *tail, gatherValidity, coordinateType);
        if (failed(valid))
          return contract.emitOpError("scalar contraction cannot preserve gather validity");
        options.segmentTail = *valid;
      }
      options.materializeZeroFill = true;
      auto projected = materializeReplayedValue(
          builder, contract.getLoc(), operand, sourceAxisIdentity(mapping),
          projectedExtent, replay, options);
      if (failed(projected))
        return contract.emitOpError("cannot project a contraction input");
      operand = *projected;
    }
    operands.push_back(operand);
    ++side;
  }
  SmallVector<Attribute> shape(resultType.getShape().size(), unit);
  auto projectedType = FragmentType::get(
      kernel.getContext(), resultType.getElementType(), builder.getArrayAttr(shape),
      resultType.getAxisMaps(), resultType.getValidity(), resultType.getOwner());
  if (gatheredType) {
    SmallVector<Attribute> projectedShape, projectedMaps;
    for (auto [axis, gatheredAxis] : llvm::enumerate(resultOrder)) {
      projectedShape.push_back(gatheredAxis >= 0
                                   ? gatheredType.getShape()[gatheredAxis] : unit);
      auto mapping = cast<AxisMapAttr>(gatheredAxis >= 0
          ? gatheredType.getAxisMaps()[gatheredAxis] : resultType.getAxisMaps()[axis]);
      projectedMaps.push_back(AxisMapAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), axis, mapping.getDerived()));
    }
    projectedType = FragmentType::get(
        kernel.getContext(), gatheredType.getElementType(),
        builder.getArrayAttr(projectedShape), builder.getArrayAttr(projectedMaps),
        gatheredType.getValidity(), gatheredType.getOwner());
  }
  auto accumulator = projectPhysicalValueToSchema(
      builder, contract.getLoc(), contract.getAccumulator(), projectedType);
  if (failed(accumulator))
    return contract.emitOpError("contraction has no uniform accumulator projection");
  auto projected = builder.create<ContractOp>(
      contract.getLoc(), projectedType, operands[0], operands[1], *accumulator,
      contract.getLhsReductionAxes(), contract.getRhsReductionAxes(),
      contract.getLhsBatchAxes(), contract.getRhsBatchAxes());
  builder.setInsertionPoint(gather);
  if (gatheredType) {
    Value result = projected.getResult();
    SmallVector<Attribute> retainedShape, retainedMaps;
    SmallVector<int64_t> permutation(gatheredType.getShape().size());
    for (auto [axis, gatheredAxis] : llvm::enumerate(resultOrder)) {
      if (gatheredAxis < 0)
        continue;
      unsigned retainedAxis = retainedShape.size();
      permutation[gatheredAxis] = retainedAxis;
      retainedShape.push_back(projectedType.getShape()[axis]);
      auto mapping = cast<AxisMapAttr>(projectedType.getAxisMaps()[axis]);
      retainedMaps.push_back(AxisMapAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), retainedAxis, mapping.getDerived()));
    }
    if (retainedShape.size() != resultOrder.size()) {
      auto retainedType = FragmentType::get(
          kernel.getContext(), gatheredType.getElementType(),
          builder.getArrayAttr(retainedShape), builder.getArrayAttr(retainedMaps),
          gatheredType.getValidity(), gatheredType.getOwner());
      auto reassociation = inferReshapeReassociation(projectedType, retainedType);
      if (failed(reassociation))
        return contract.emitOpError("contraction projection cannot drop its selected singleton axes");
      result = builder.create<ReshapeOp>(gather.getLoc(), retainedType, result,
                                        *reassociation);
    }
    if (llvm::any_of(llvm::enumerate(permutation), [](auto entry) {
          return entry.index() != static_cast<unsigned>(entry.value());
        }))
      result = builder.create<TransposeOp>(gather.getLoc(), gatheredType, result,
                                           permutation);
    if (gather.getValid())
      result = builder.create<SelectOp>(gather.getLoc(), gatheredType,
                                       gather.getValid(), result, gather.getFill());
    gather.getResult().replaceAllUsesWith(result);
    gather.erase();
    contract.erase();
    return true;
  }
  Value zero = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 0);
  SmallVector<Value> zeros(selected.size(), zero);
  auto replacement = builder.create<GatherOp>(
      gather.getLoc(), gather.getResult().getType(), projected, zeros,
      gather.getValid(), gather.getFill(), gather.getSourceAxes());
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  contract.erase();
  return true;
}

} // namespace

bool hasRangeContractForm(ContractOp contract,
                         SmallVectorImpl<StoreOp> *stores) {
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
  SmallVector<std::tuple<LoadOp, AxisMapAttr, Value>> axes;
  for (auto [load, operand, axis] :
       {std::tuple<LoadOp, Value, unsigned>{lhsLoad, contract.getLhs(), *lhsFree},
        std::tuple<LoadOp, Value, unsigned>{
            lhsLoad, contract.getLhs(),
            static_cast<unsigned>(contract.getLhsReductionAxes().front())},
        std::tuple<LoadOp, Value, unsigned>{
            rhsLoad, contract.getRhs(),
            static_cast<unsigned>(contract.getRhsReductionAxes().front())},
        std::tuple<LoadOp, Value, unsigned>{rhsLoad, contract.getRhs(), *rhsFree}}) {
    FailureOr<AxisMapAttr> mapping = queryAxisMap(operand.getType(), axis);
    if (failed(mapping))
      return false;
    axes.emplace_back(load, *mapping, operand);
  }
  for (auto [load, mapping, operand] : axes) {
    FailureOr<unsigned> coordinate = accessCoordinatePosition(load, mapping, operand);
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
  if (!collectStorePaths(contract.getResult(), {}, paths, visited))
    return false;
  if (stores)
    for (const StorePath &path : paths)
      stores->push_back(path.store);
  return true;
}

LogicalResult normalizeMatrixContractShapes(func::FuncOp kernel) {
  llvm::DenseSet<StringAttr> units;
  kernel.walk([&](ParameterOp parameter) {
    auto schema = parameter.getParameter();
    auto role = static_cast<ParameterRole>(schema.getRole());
    if ((role != ParameterRole::OwnershipM &&
         role != ParameterRole::OwnershipN) ||
        schema.getCandidates().size() != 1 || schema.getCandidates()[0] != 1)
      return;
    units.insert(schema.getName());
    OpBuilder builder(parameter);
    Value constant = builder.create<arith::ConstantOp>(
        parameter.getLoc(), parameter.getResult().getType(),
        builder.getIntegerAttr(parameter.getResult().getType(), 1));
    parameter.getResult().replaceAllUsesWith(constant);
  });
  // Ownership and coverage are already closed at this boundary. A singleton
  // physical candidate is an exact extent, independent of the target API.
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](PhysicalExprAttr extent) -> std::optional<Attribute> {
    if (extent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Parameter) ||
        !units.contains(extent.getSymbol()))
      return std::nullopt;
    return expression(kernel.getContext(), PhysicalExprKind::Constant, 1);
  });
  replacer.recursivelyReplaceElementsIn(kernel.getOperation(),
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
  auto [nextSource, nextDimension] = nextPhysicalAxisIdentities(kernel);
  SmallVector<ContractOp> contracts;
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  for (ContractOp contract : contracts) {
    auto lhs = contract.getLhs().getType();
    auto rhs = contract.getRhs().getType();
    if (contract.getLhsReductionAxes().size() != 1 ||
        contract.getRhsReductionAxes().size() != 1 ||
        (lhs.getShape().size() <= 2 && rhs.getShape().size() <= 2))
      continue;
    ArrayRef<int64_t> lhsBatch = contract.getLhsBatchAxes();
    ArrayRef<int64_t> rhsBatch = contract.getRhsBatchAxes();
    unsigned batchRank = lhsBatch.size();
    if (batchRank > 1)
      continue;
    auto canonicalBatch = [&](ArrayRef<int64_t> axes) {
      return llvm::all_of(llvm::enumerate(axes), [](auto entry) {
        return static_cast<int64_t>(entry.index()) == entry.value();
      });
    };
    if (batchRank && lhs.getShape().size() == batchRank + 2 &&
        rhs.getShape().size() == batchRank + 2 &&
        canonicalBatch(lhsBatch) && canonicalBatch(rhsBatch) &&
        contract.getLhsReductionAxes().front() == batchRank + 1 &&
        contract.getRhsReductionAxes().front() == batchRank)
      continue;
    auto freeAxes = [](FragmentType type, int64_t reduction,
                       ArrayRef<int64_t> batch) {
      SmallVector<int64_t> axes;
      for (int64_t axis = 0; axis < static_cast<int64_t>(type.getShape().size()); ++axis)
        if (axis != reduction && !llvm::is_contained(batch, axis))
          axes.push_back(axis);
      return axes;
    };
    auto lhsFree = freeAxes(lhs, contract.getLhsReductionAxes().front(), lhsBatch);
    auto rhsFree = freeAxes(rhs, contract.getRhsReductionAxes().front(), rhsBatch);
    if (lhsFree.empty() || rhsFree.empty())
      return contract.emitOpError("matrix normalization requires free axes on both operands");
    OpBuilder builder(contract);
    Location location = contract.getLoc();
    auto remap = [&](AxisMapAttr axis, unsigned position) {
      return AxisMapAttr::get(kernel.getContext(), axis.getSourceId(),
          axis.getSourceAxis(), axis.getDimensionId(), position, axis.getDerived());
    };
    auto collapsedAxis = [&](FragmentType type, ArrayRef<int64_t> axes,
                             unsigned position) {
      if (axes.size() == 1)
        return remap(cast<AxisMapAttr>(type.getAxisMaps()[axes.front()]), position);
      return AxisMapAttr::get(kernel.getContext(), nextSource++, 0,
                                   nextDimension++, position, true);
    };
    auto product = [&](FragmentType type, ArrayRef<int64_t> axes) {
      auto extent = cast<PhysicalExprAttr>(type.getShape()[axes.front()]);
      for (int64_t axis : axes.drop_front())
        extent = PhysicalExprAttr::get(kernel.getContext(),
            static_cast<uint32_t>(PhysicalExprKind::Multiply), 0,
            builder.getStringAttr(""),
            builder.getArrayAttr({extent, type.getShape()[axis]}));
      return extent;
    };
    auto makeType = [&](FragmentType source, ArrayRef<Attribute> shape,
                        ArrayRef<Attribute> mappings) {
      return FragmentType::get(kernel.getContext(), source.getElementType(),
          builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
          source.getValidity(), source.getOwner());
    };
    auto transpose = [&](Value value, ArrayRef<int64_t> permutation) {
      if (llvm::all_of(llvm::enumerate(permutation), [](auto entry) {
            return static_cast<int64_t>(entry.index()) == entry.value();
          }))
        return value;
      auto type = cast<FragmentType>(value.getType());
      SmallVector<Attribute> shape, mappings;
      for (auto [position, axis] : llvm::enumerate(permutation)) {
        shape.push_back(type.getShape()[axis]);
        mappings.push_back(remap(cast<AxisMapAttr>(type.getAxisMaps()[axis]), position));
      }
      return Value(builder.create<TransposeOp>(location,
          makeType(type, shape, mappings), value, permutation));
    };
    auto reshape = [&](Value value, FragmentType target) -> FailureOr<Value> {
      if (value.getType() == target) return value;
      auto relation = inferReshapeReassociation(
          cast<FragmentType>(value.getType()), target);
      if (failed(relation)) return failure();
      return Value(builder.create<ReshapeOp>(location, target, value, *relation));
    };
    auto originalResult = contract.getResult().getType();
    SmallVector<int64_t> resultPermutation;
    auto appendResultAxes = [&](FragmentType operand, ArrayRef<int64_t> axes) {
      for (int64_t axis : axes) {
        auto mapping = cast<AxisMapAttr>(operand.getAxisMaps()[axis]);
        auto source = queryFragmentAxis(originalResult, sourceAxisIdentity(mapping));
        auto dimension = queryFragmentDimension(originalResult, mapping.getDimensionId());
        std::optional<int64_t> position;
        if (source.isExact()) position = source.fragmentAxis;
        else if (dimension.isExact()) position = dimension.fragmentAxis;
        if (!position || llvm::is_contained(resultPermutation, *position))
          return failure();
        resultPermutation.push_back(*position);
      }
      return success();
    };
    if (failed(appendResultAxes(lhs, lhsBatch)) ||
        failed(appendResultAxes(lhs, lhsFree)) ||
        failed(appendResultAxes(rhs, rhsFree)) ||
        resultPermutation.size() != originalResult.getShape().size())
      return contract.emitOpError("matrix free axes have no bijective result projection");
    auto m = product(lhs, lhsFree);
    auto n = product(rhs, rhsFree);
    auto mAxis = collapsedAxis(lhs, lhsFree, batchRank);
    auto nAxis = collapsedAxis(rhs, rhsFree, batchRank + 1);
    int64_t lhsK = contract.getLhsReductionAxes().front();
    int64_t rhsK = contract.getRhsReductionAxes().front();
    SmallVector<int64_t> lhsPermutation(lhsBatch);
    llvm::append_range(lhsPermutation, lhsFree);
    lhsPermutation.push_back(lhsK);
    SmallVector<int64_t> rhsPermutation(rhsBatch);
    rhsPermutation.push_back(rhsK);
    llvm::append_range(rhsPermutation, rhsFree);
    SmallVector<Attribute> lhsShape, lhsMappings, rhsShape, rhsMappings;
    SmallVector<Attribute> resultShape, resultMappings;
    SmallVector<int64_t> matrixBatch;
    for (auto [position, pair] : llvm::enumerate(llvm::zip(lhsBatch, rhsBatch))) {
      auto [left, right] = pair;
      lhsShape.push_back(lhs.getShape()[left]);
      lhsMappings.push_back(remap(
          cast<AxisMapAttr>(lhs.getAxisMaps()[left]), position));
      rhsShape.push_back(rhs.getShape()[right]);
      rhsMappings.push_back(remap(
          cast<AxisMapAttr>(rhs.getAxisMaps()[right]), position));
      int64_t resultAxis = resultPermutation[position];
      resultShape.push_back(originalResult.getShape()[resultAxis]);
      resultMappings.push_back(remap(
          cast<AxisMapAttr>(originalResult.getAxisMaps()[resultAxis]), position));
      matrixBatch.push_back(position);
    }
    llvm::append_range(lhsShape, ArrayRef<Attribute>{m, lhs.getShape()[lhsK]});
    llvm::append_range(lhsMappings, ArrayRef<Attribute>{mAxis,
        remap(cast<AxisMapAttr>(lhs.getAxisMaps()[lhsK]), batchRank + 1)});
    llvm::append_range(rhsShape, ArrayRef<Attribute>{rhs.getShape()[rhsK], n});
    llvm::append_range(rhsMappings, ArrayRef<Attribute>{
        remap(cast<AxisMapAttr>(rhs.getAxisMaps()[rhsK]), batchRank), nAxis});
    llvm::append_range(resultShape, ArrayRef<Attribute>{m, n});
    llvm::append_range(resultMappings, ArrayRef<Attribute>{mAxis, nAxis});
    auto matrixResult = makeType(originalResult, resultShape, resultMappings);
    FailureOr<Value> matrixLhs = reshape(transpose(contract.getLhs(), lhsPermutation),
        makeType(lhs, lhsShape, lhsMappings));
    FailureOr<Value> matrixRhs = reshape(transpose(contract.getRhs(), rhsPermutation),
        makeType(rhs, rhsShape, rhsMappings));
    Value orderedAccumulator = transpose(contract.getAccumulator(), resultPermutation);
    auto orderedResult = cast<FragmentType>(orderedAccumulator.getType());
    FailureOr<Value> matrixAccumulator = reshape(orderedAccumulator, matrixResult);
    if (failed(matrixLhs) || failed(matrixRhs) || failed(matrixAccumulator))
      return contract.emitOpError("matrix form has no exact row-major reshape");
    contract->setOperands(ValueRange{*matrixLhs, *matrixRhs, *matrixAccumulator});
    contract->setAttr("lhs_reduction_axes", builder.getDenseI64ArrayAttr({batchRank + 1}));
    contract->setAttr("rhs_reduction_axes", builder.getDenseI64ArrayAttr({batchRank}));
    contract->setAttr("lhs_batch_axes", builder.getDenseI64ArrayAttr(matrixBatch));
    contract->setAttr("rhs_batch_axes", builder.getDenseI64ArrayAttr(matrixBatch));
    contract.getResult().setType(matrixResult);
    builder.setInsertionPointAfter(contract);
    FailureOr<Value> restored = reshape(contract.getResult(), orderedResult);
    if (failed(restored))
      return contract.emitOpError("matrix result has no inverse row-major reshape");
    Operation *firstRestore = (*restored).getDefiningOp();
    SmallVector<int64_t> inverse(resultPermutation.size());
    for (auto [position, axis] : llvm::enumerate(resultPermutation))
      inverse[axis] = position;
    Value result = transpose(*restored, inverse);
    contract.getResult().replaceUsesWithIf(result, [&](OpOperand &use) {
      return use.getOwner() != firstRestore &&
             use.getOwner() != result.getDefiningOp();
    });
  }
  for (ContractOp contract : contracts) {
    auto unitBatch = [](FragmentType type) {
      return type.getShape().size() == 3 &&
             constantPhysicalExpression(
                 cast<PhysicalExprAttr>(type.getShape()[0])) == 1;
    };
    if (contract.getLhsBatchAxes() != ArrayRef<int64_t>{0} ||
        contract.getRhsBatchAxes() != ArrayRef<int64_t>{0} ||
        contract.getLhsReductionAxes() != ArrayRef<int64_t>{2} ||
        contract.getRhsReductionAxes() != ArrayRef<int64_t>{1} ||
        !unitBatch(contract.getLhs().getType()) ||
        !unitBatch(contract.getRhs().getType()) ||
        !unitBatch(contract.getAccumulator().getType()) ||
        !unitBatch(contract.getResult().getType()))
      continue;
    OpBuilder builder(contract);
    auto matrixType = [&](FragmentType type) {
      SmallVector<Attribute> axes;
      for (Attribute attribute : type.getAxisMaps().getValue().drop_front()) {
        auto axis = cast<AxisMapAttr>(attribute);
        axes.push_back(AxisMapAttr::get(
            kernel.getContext(), axis.getSourceId(), axis.getSourceAxis(),
            axis.getDimensionId(), axes.size(), axis.getDerived()));
      }
      return FragmentType::get(
          kernel.getContext(), type.getElementType(),
          builder.getArrayAttr(type.getShape().getValue().drop_front()),
          builder.getArrayAttr(axes), type.getValidity(), type.getOwner());
    };
    SmallVector<Value> operands;
    for (Value operand : {contract.getLhs(), contract.getRhs(),
                          contract.getAccumulator()}) {
      auto source = cast<FragmentType>(operand.getType());
      auto target = matrixType(source);
      auto reassociation = inferReshapeReassociation(source, target);
      if (failed(reassociation))
        return contract.emitOpError("unit-batch operand has no exact row-major reshape");
      operands.push_back(builder.create<ReshapeOp>(
          contract.getLoc(), target, operand, *reassociation));
    }
    auto original = contract.getResult().getType();
    auto projected = matrixType(original);
    auto reassociation = inferReshapeReassociation(projected, original);
    if (failed(reassociation))
      return contract.emitOpError("unit-batch result has no inverse row-major reshape");
    auto matrix = builder.create<ContractOp>(
        contract.getLoc(), projected, operands[0], operands[1], operands[2],
        ArrayRef<int64_t>{1}, ArrayRef<int64_t>{0},
        ArrayRef<int64_t>{}, ArrayRef<int64_t>{});
    auto restored = builder.create<ReshapeOp>(
        contract.getLoc(), original, matrix, *reassociation);
    if (Attribute origin = contract->getAttr(originAttr)) {
      matrix->setAttr(originAttr, origin);
      restored->setAttr(originAttr, origin);
    }
    contract.getResult().replaceAllUsesWith(restored);
    contract.erase();
  }
  kernel.walk([&](scf::ForOp loop) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [index, carried] : llvm::enumerate(loop.getRegionIterArgs())) {
      if (!carried.hasOneUse())
        continue;
      auto projected = dyn_cast<ReshapeOp>(*carried.getUsers().begin());
      auto restored = yield.getOperand(index).getDefiningOp<ReshapeOp>();
      if (!projected || !restored ||
          projected->getBlock() != loop.getBody() ||
          restored->getBlock() != loop.getBody() ||
          !restored.getResult().hasOneUse() ||
          restored.getResult().getType() != carried.getType() ||
          restored.getValue().getType() != projected.getResult().getType() ||
          projected.getResult().getType() == carried.getType())
        continue;

      // Keep the carry in the computation's shape. The inverse pure views at
      // the loop boundaries also preserve the value of a zero-trip loop.
      OpBuilder builder(loop);
      IRMapping initialMapping;
      initialMapping.map(carried, loop.getInitArgs()[index]);
      Operation *initial = builder.clone(*projected, initialMapping);
      Type type = projected.getResult().getType();
      loop.getInitArgsMutable()[index].assign(initial->getResult(0));
      carried.setType(type);
      loop.getResult(index).setType(type);
      yield->setOperand(index, restored.getValue());

      builder.setInsertionPointAfter(loop);
      IRMapping resultMapping;
      resultMapping.map(restored.getValue(), loop.getResult(index));
      Operation *result = builder.clone(*restored, resultMapping);
      loop.getResult(index).replaceUsesWithIf(result->getResult(0),
          [&](OpOperand &use) { return use.getOwner() != result; });
      projected.getResult().replaceAllUsesWith(carried);
      restored.erase();
      projected.erase();
    }
  });
  return success();
}

LogicalResult orientLoopContractions(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  SmallVector<scf::ForOp> loops;
  kernel->walk([&](scf::ForOp loop) {
    if (!loop->getParentOfType<scf::ForOp>())
      loops.push_back(loop);
  });
  for (scf::ForOp loop : loops) {
    SmallVector<FragmentType> fields;
    for (Value value : loop.getInitArgs())
      matrixFields(value.getType(), fields);
    if (fields.empty())
      continue;
    FragmentType matrix = fields.front();
    if (!isa<FloatType>(matrix.getElementType()) ||
        !llvm::all_of(fields, [&](FragmentType field) {
          return field == matrix;
        }))
      continue;
    auto row = parameterForExtent(
        *kernel, cast<PhysicalExprAttr>(matrix.getShape()[0]));
    auto column = parameterForExtent(
        *kernel, cast<PhysicalExprAttr>(matrix.getShape()[1]));
    if (failed(row) || failed(column) ||
        row->getParameter().getCategory() !=
            static_cast<uint32_t>(ParameterCategory::RegionContraction) ||
        column->getParameter().getRole() !=
            static_cast<uint32_t>(ParameterRole::FullCoverage))
      continue;
    llvm::DenseSet<Value> visited;
    ContractOp carriedContract;
    std::function<bool(Value)> dependsOnMatrixContract = [&](Value value) {
      if (!visited.insert(value).second)
        return false;
      Operation *producer = value.getDefiningOp();
      if (!producer || !loop->isProperAncestor(producer))
        return false;
      if (auto contract = dyn_cast<ContractOp>(producer))
        if (contract.getResult().getType().getShape() == matrix.getShape() &&
            contract.getResult().getType().getElementType() ==
                matrix.getElementType()) {
          carriedContract = contract;
          return true;
        }
      return llvm::any_of(producer->getOperands(), dependsOnMatrixContract);
    };
    bool carriesContract = llvm::any_of(
        loop.getBody()->getTerminator()->getOperands(), dependsOnMatrixContract);
    if (!carriesContract || !supportsMatrixLoopTranspose(loop))
      continue;
    // Orient the complete connected product graph, including joins. Every
    // matrix value is transposed together, so shared producers remain shared
    // and pointwise joins retain their original arithmetic order.
    llvm::DenseSet<Operation *> chain;
    SmallVector<ContractOp> pending{carriedContract};
    bool hasProducer = false;
    while (!pending.empty()) {
      ContractOp consumer = pending.pop_back_val();
      if (!chain.insert(consumer.getOperation()).second)
        continue;
      visited.clear();
      std::function<void(Value)> traceProducer = [&](Value value) {
        if (!visited.insert(value).second)
          return;
        Operation *producer = value.getDefiningOp();
        if (!producer || !loop->isProperAncestor(producer))
          return;
        if (auto contract = dyn_cast<ContractOp>(producer)) {
          hasProducer |= contract != carriedContract;
          pending.push_back(contract);
          return;
        }
        for (Value operand : producer->getOperands())
          traceProducer(operand);
      };
      traceProducer(consumer.getLhs());
      traceProducer(consumer.getRhs());
      traceProducer(consumer.getAccumulator());
    }
    bool disconnectedProducts = false;
    loop.walk([&](ContractOp contract) {
      disconnectedProducts |= !chain.contains(contract.getOperation());
    });
    if (!hasProducer || disconnectedProducts)
      continue;

    // Keep a strongly rectangular matrix's short axis in the column position.
    // The predicate uses existing tile parameters, not a new structural tuner.
    OpBuilder builder(loop);
    Location location = loop.getLoc();
    Value four = builder.create<arith::ConstantIndexOp>(location, 4);
    Value quarter = binary(builder, location, builder.getIndexType(),
                           column->getResult(), four,
                           BinaryOperator::FloorDivide);
    Value rectangular = builder.create<CompareOp>(
        location, builder.getI1Type(), row->getResult(), quarter,
        ComparePredicate::Le);
    auto choice = builder.create<scf::IfOp>(
        location, loop.getResultTypes(), rectangular, true);
    auto branchBuilder = [](Region &region) {
      Block &block = region.front();
      if (!block.empty() && isa<scf::YieldOp>(block.back()))
        block.back().erase();
      return OpBuilder(&block, block.end());
    };
    OpBuilder transposed = branchBuilder(choice.getThenRegion());
    IRMapping mapping;
    llvm::DenseSet<Value> captured;
    loop.walk([&](Operation *operation) {
      for (Value operand : operation->getOperands()) {
        Operation *owner = operand.getParentRegion()->getParentOp();
        if (owner == loop || loop->isProperAncestor(owner) ||
            !captured.insert(operand).second)
          continue;
        SmallVector<FragmentType> matrices;
        matrixFields(operand.getType(), matrices);
        if (!matrices.empty())
          mapping.map(operand,
                      transposeLoopValue(transposed, location, operand));
      }
    });
    auto replacement = cast<scf::ForOp>(transposed.clone(*loop, mapping));
    transposeClonedLoop(replacement);
    SmallVector<Value> results;
    for (Value result : replacement.getResults())
      results.push_back(transposeLoopValue(transposed, location, result));
    transposed.create<scf::YieldOp>(location, results);
    OpBuilder original = branchBuilder(choice.getElseRegion());
    IRMapping originalMapping;
    auto unchanged = cast<scf::ForOp>(original.clone(*loop, originalMapping));
    original.create<scf::YieldOp>(location, unchanged.getResults());
    loop.replaceAllUsesWith(choice.getResults());
    loop.erase();
  }
  eraseDeadPhysicalValues(*kernel);
  return success();
}

static ReshapeOp exposeTransposedContractSplit(ContractOp contract) {
  auto transpose = dyn_cast<TransposeOp>(*contract.getResult().getUsers().begin());
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>{1, 0} ||
      !transpose.getResult().hasOneUse())
    return {};
  auto output = dyn_cast<ReshapeOp>(*transpose.getResult().getUsers().begin());
  if (!output)
    return {};
  auto target = cast<FragmentType>(output.getResult().getType());
  SmallVector<int64_t> freeAxes;
  std::optional<int64_t> columnAxis;
  llvm::SmallDenseSet<int64_t> units;
  for (Attribute attribute : output.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().empty()) {
      for (int64_t axis : group.getResultAxes().asArrayRef()) {
        auto extent = cast<PhysicalExprAttr>(target.getShape()[axis]);
        if (extent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) ||
            extent.getValue() != 1)
          return {};
        units.insert(axis);
      }
    } else if (group.getSourceAxes().asArrayRef() == ArrayRef<int64_t>{0} &&
               group.getResultAxes().size() == 1 && !columnAxis) {
      columnAxis = group.getResultAxes()[0];
    } else if (group.getSourceAxes().asArrayRef() == ArrayRef<int64_t>{1} && freeAxes.empty()) {
      llvm::append_range(freeAxes, group.getResultAxes().asArrayRef());
    } else {
      return {};
    }
  }
  if (!columnAxis || freeAxes.size() < 2 ||
      freeAxes.size() + units.size() + 1 != target.getShape().size())
    return {};
  auto rowExtent = cast<PhysicalExprAttr>(contract.getResult().getType().getShape()[0]);
  if (rowExtent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) ||
      rowExtent.getValue() <= 0)
    return {};
  int64_t product = 1;
  for (int64_t axis : freeAxes) {
    auto extent = cast<PhysicalExprAttr>(target.getShape()[axis]);
    if (extent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) ||
        extent.getValue() <= 0 || product > rowExtent.getValue() / extent.getValue())
      return {};
    product *= extent.getValue();
  }
  if (product != rowExtent.getValue() ||
      target.getShape()[*columnAxis] != contract.getResult().getType().getShape()[1])
    return {};

  // Commute a pure transpose with the declared row-major split, exposing
  // [split rows..., column] to the existing contraction composition below.
  OpBuilder builder(transpose);
  SmallVector<Attribute> shape, mappings;
  SmallVector<int64_t> splitAxes;
  auto appendAxis = [&](int64_t axis) {
    auto mapping = cast<AxisMapAttr>(target.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(
        contract.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), shape.size(), mapping.getDerived()));
    shape.push_back(target.getShape()[axis]);
  };
  for (int64_t axis : freeAxes) {
    splitAxes.push_back(shape.size());
    appendAxis(axis);
  }
  appendAxis(*columnAxis);
  auto expanded = FragmentType::get(
      contract.getContext(), target.getElementType(), builder.getArrayAttr(shape),
      builder.getArrayAttr(mappings), target.getValidity(), target.getOwner());
  auto split = builder.create<ReshapeOp>(
      output.getLoc(), expanded, contract.getResult(), builder.getArrayAttr({
          ReshapeGroupAttr::get(contract.getContext(), builder.getDenseI64ArrayAttr({0}),
                               builder.getDenseI64ArrayAttr(splitAxes)),
          ReshapeGroupAttr::get(contract.getContext(), builder.getDenseI64ArrayAttr({1}),
                               builder.getDenseI64ArrayAttr({static_cast<int64_t>(freeAxes.size())}))}));
  if (Attribute origin = output->getAttr(originAttr))
    split->setAttr(originAttr, origin);
  shape.clear();
  mappings.clear();
  SmallVector<int64_t> permutation;
  SmallVector<Attribute> restore;
  for (int64_t axis = 0; axis < static_cast<int64_t>(target.getShape().size()); ++axis) {
    if (units.contains(axis)) {
      restore.push_back(ReshapeGroupAttr::get(
          contract.getContext(), builder.getDenseI64ArrayAttr({}),
          builder.getDenseI64ArrayAttr({axis})));
      continue;
    }
    int64_t inputAxis = axis == *columnAxis
                            ? static_cast<int64_t>(freeAxes.size())
                            : std::distance(freeAxes.begin(), llvm::find(freeAxes, axis));
    permutation.push_back(inputAxis);
    restore.push_back(ReshapeGroupAttr::get(
        contract.getContext(), builder.getDenseI64ArrayAttr({static_cast<int64_t>(shape.size())}),
        builder.getDenseI64ArrayAttr({axis})));
    appendAxis(axis);
  }
  auto reordered = FragmentType::get(
      contract.getContext(), target.getElementType(), builder.getArrayAttr(shape),
      builder.getArrayAttr(mappings), target.getValidity(), target.getOwner());
  Value value = builder.create<TransposeOp>(
      output.getLoc(), reordered, split, builder.getDenseI64ArrayAttr(permutation));
  if (Attribute origin = output->getAttr(originAttr))
    value.getDefiningOp()->setAttr(originAttr, origin);
  if (!units.empty()) {
    value = builder.create<ReshapeOp>(output.getLoc(), target, value, builder.getArrayAttr(restore));
    if (Attribute origin = output->getAttr(originAttr))
      value.getDefiningOp()->setAttr(originAttr, origin);
  }
  output.getResult().replaceAllUsesWith(value);
  output.erase();
  transpose.erase();
  return split;
}

static void orientContractionOutputs(func::FuncOp kernel) {
  SmallVector<ContractOp> contracts;
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  for (ContractOp contract : contracts) {
    auto result = contract.getResult().getType();
    auto lhs = contract.getLhs().getType();
    auto rhs = contract.getRhs().getType();
    if (!lhs.getElementType().isF32() || !rhs.getElementType().isF32() ||
        !result.getElementType().isF32() ||
        !contract.getLhsBatchAxes().empty() ||
        !contract.getRhsBatchAxes().empty())
      continue;
    unsigned lhsFree =
        lhs.getShape().size() - contract.getLhsReductionAxes().size();
    unsigned rhsFree =
        rhs.getShape().size() - contract.getRhsReductionAxes().size();
    if (!lhsFree || !rhsFree)
      continue;
    PhysicalProgramAnalysis analysis(kernel);

    SmallVector<Value> pending{contract.getResult()};
    llvm::DenseSet<Value> visited;
    bool compatible = true;
    bool sawStore = false;
    while (compatible && !pending.empty()) {
      Value value = pending.pop_back_val();
      if (!visited.insert(value).second)
        continue;
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (auto store = dyn_cast<StoreOp>(user)) {
          auto axes = store.getSourceAxes();
          if (store.getValue() != value || axes.empty()) {
            compatible = false;
            break;
          }
          auto position = llvm::find(axes, static_cast<int64_t>(axes.size() - 1));
          if (position == axes.end()) {
            compatible = false;
            break;
          }
          Value coordinate = store.getCoordinates()[position - axes.begin()];
          while (auto projection = coordinate.getDefiningOp()) {
            if (!isa<BroadcastOp, ReshapeOp>(projection))
              break;
            coordinate = projection->getOperand(0);
          }
          auto type = dyn_cast<FragmentType>(coordinate.getType());
          if (!type || type.getShape().size() != 1) {
            compatible = false;
            break;
          }
          auto destination =
              queryExactLogicalRange(analysis.sourceRanges(coordinate));
          if (failed(destination)) {
            compatible = false;
            break;
          }
          auto matchesFreeAxis = [&](Value operand, ArrayRef<int64_t> reduction) {
            auto fragment = cast<FragmentType>(operand.getType());
            for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
              if (llvm::is_contained(reduction, static_cast<int64_t>(axis)))
                continue;
              auto ranges = analysis.axisRanges(operand, axis);
              if (ranges.isExact() && !ranges.roots.empty() &&
                  ranges.blockers.empty() &&
                  llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
                    return sameLogicalRange(range, *destination);
                  }))
                return true;
            }
            return false;
          };
          if (!matchesFreeAxis(contract.getLhs(),
                               contract.getLhsReductionAxes()) ||
              matchesFreeAxis(contract.getRhs(), contract.getRhsReductionAxes())) {
            compatible = false;
            break;
          }
          sawStore = true;
          continue;
        }
        if (isa<scf::YieldOp>(user) &&
            isa<scf::ForOp, scf::IfOp>(user->getParentOp())) {
          pending.push_back(
              user->getParentOp()->getResult(use.getOperandNumber()));
          continue;
        }
        if (!isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
                 SplatOp, BroadcastOp, ReshapeOp, TransposeOp>(user)) {
          compatible = false;
          break;
        }
        llvm::append_range(pending, user->getResults());
      }
    }
    if (!compatible || !sawStore)
      continue;

    // Matrix columns follow the innermost output coordinate. Swap the free-axis
    // groups and restore the original result schema for all existing users.
    SmallVector<int64_t> permutation;
    for (unsigned axis = lhsFree; axis < lhsFree + rhsFree; ++axis)
      permutation.push_back(axis);
    for (unsigned axis = 0; axis < lhsFree; ++axis)
      permutation.push_back(axis);
    SmallVector<Attribute> shape, mappings;
    SmallVector<int64_t> inverse(permutation.size());
    for (auto [position, axis] : llvm::enumerate(permutation)) {
      shape.push_back(result.getShape()[axis]);
      auto mapping = cast<AxisMapAttr>(result.getAxisMaps()[axis]);
      mappings.push_back(AxisMapAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), position, mapping.getDerived()));
      inverse[axis] = position;
    }
    OpBuilder builder(contract);
    auto transposed = FragmentType::get(
        kernel.getContext(), result.getElementType(), builder.getArrayAttr(shape),
        builder.getArrayAttr(mappings), result.getValidity(), result.getOwner());
    Value accumulator = builder.create<TransposeOp>(
        contract.getLoc(), transposed, contract.getAccumulator(), permutation);
    auto replacement = cast<ContractOp>(builder.clone(*contract));
    replacement->setOperand(0, contract.getRhs());
    replacement->setOperand(1, contract.getLhs());
    replacement.getAccumulatorMutable().assign(accumulator);
    replacement.setLhsReductionAxesAttr(contract.getRhsReductionAxesAttr());
    replacement.setRhsReductionAxesAttr(contract.getLhsReductionAxesAttr());
    replacement.getResult().setType(transposed);
    Value restored = builder.create<TransposeOp>(
        contract.getLoc(), result, replacement.getResult(), inverse);
    contract.getResult().replaceAllUsesWith(restored);
    contract.erase();
  }
}

LogicalResult fuseMultiplyReductions(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  SmallVector<ReduceOp> reductions;
  kernel->walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
  for (ReduceOp reduce : reductions) {
    if (reduce.getSourceCount() != 1 || reduce.getIdentityCount() != 1 ||
        reduce.getCaptureCount() != 0 || reduce.getNumResults() != 1 ||
        !isLiteralZeroProjection(reduce.getInputs()[1]))
      continue;
    auto resultType = dyn_cast<FragmentType>(reduce.getResult(0).getType());
    // The existing matrix path provides full-precision f32 accumulation.
    // Other accumulator formats keep their native reduction until that path
    // can preserve their precision without imposing new provider restrictions.
    if (!resultType || !resultType.getElementType().isF32())
      continue;
    if (queryBinaryCombineKind(reduce.getCombine()) != BinaryOperator::Add)
      continue;

    Value product = reduce.getInputs().front();
    // Builtin sum may retain an identity cast. A numeric conversion, including
    // default accumulator widening, is a rounding boundary and cannot fuse.
    while (product.hasOneUse()) {
      auto cast = product.getDefiningOp<CastOp>();
      if (!cast || cast.getValue().getType() != product.getType())
        break;
      product = cast.getValue();
    }
    auto multiply = product.getDefiningOp<BinaryOp>();
    auto productType = dyn_cast<FragmentType>(product.getType());
    if (!multiply || !product.hasOneUse() || !productType ||
        multiply.getOperatorKind() != BinaryOperator::Multiply ||
        productType.getElementType() != resultType.getElementType())
      continue;
    auto unbroadcast = [&](Value value) {
      auto source = dyn_cast<FragmentType>(value.getType());
      BroadcastProjection projection;
      if (source)
        projection = queryBroadcastProjection(source, productType);
      if (!projection.isExact())
        return std::pair{value, projection};
      while (Operation *producer = value.getDefiningOp()) {
        if (auto broadcast = dyn_cast<BroadcastOp>(producer)) {
          auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
          auto output = cast<FragmentType>(value.getType());
          if (!input || input.getValidity() != output.getValidity())
            break;
          auto step = queryBroadcastProjection(input, output);
          if (!step.isExact())
            break;
          for (auto &axis : projection.targetToSource)
            if (axis)
              axis = step.targetToSource[*axis];
          value = broadcast.getValue();
        } else if (auto conversion = dyn_cast<CastOp>(producer)) {
          if (conversion.getValue().getType() != value.getType())
            break;
          value = conversion.getValue();
        } else if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
          auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
          auto target = dyn_cast<FragmentType>(value.getType());
          if (!source || !target || source.getShape() != target.getShape() ||
              source.getValidity() != target.getValidity() ||
              !llvm::all_of(reshape.getReassociation(), [](Attribute attribute) {
                auto group = cast<ReshapeGroupAttr>(attribute);
                return group.getSourceAxes().size() == 1 &&
                       group.getSourceAxes() == group.getResultAxes();
              }))
            break;
          value = reshape.getValue();
        } else {
          break;
        }
      }
      return std::pair{value, projection};
    };
    auto [lhs, lhsProjection] = unbroadcast(multiply.getLhs());
    auto [rhs, rhsProjection] = unbroadcast(multiply.getRhs());
    auto lhsType = dyn_cast<FragmentType>(lhs.getType());
    auto rhsType = dyn_cast<FragmentType>(rhs.getType());
    if (!lhsType || !rhsType ||
        lhsType.getElementType() != productType.getElementType() ||
        rhsType.getElementType() != productType.getElementType())
      continue;
    if (!lhsProjection.isExact() || !rhsProjection.isExact())
      continue;
    PhysicalProgramAnalysis analysis(*kernel);
    auto broadcastsAxis = [&](Value value, std::optional<unsigned> axis) {
      if (!axis)
        return true;
      auto type = cast<FragmentType>(value.getType());
      if (constantPhysicalExpression(
              cast<PhysicalExprAttr>(type.getShape()[*axis])) != 1)
        return false;
      // A one-element physical tile can still carry a non-unit logical axis.
      // Drop only introduced units or ranges proven logically singleton.
      auto ranges = analysis.axisRanges(value, *axis);
      return ranges.isExact() &&
             llvm::all_of(ranges.roots, [](MakeRangeOp range) {
               // A lifted workset coordinate starts as one logical iteration;
               // ownership may subsequently pack several iterations together.
               return !range.getStart().getDefiningOp<WorksetCoordinateOp>() &&
                      isProvablySingletonLogicalRange(range);
             });
    };
    SmallVector<int64_t> lhsKept, rhsKept, lhsReduced, rhsReduced;
    SmallVector<int64_t> lhsBatch, rhsBatch, lhsOutput, rhsOutput;
    bool compatible = true;
    unsigned lhsFree = 0, rhsFree = 0;
    for (unsigned axis = 0; axis < productType.getShape().size(); ++axis) {
      auto left = lhsProjection.targetToSource[axis];
      auto right = rhsProjection.targetToSource[axis];
      bool leftBroadcast = broadcastsAxis(lhs, left);
      bool rightBroadcast = broadcastsAxis(rhs, right);
      bool reduced = llvm::is_contained(reduce.getAxes(), axis);
      if (reduced && (!left || !right || leftBroadcast || rightBroadcast)) {
        compatible = false;
        break;
      }
      // Retain a common unit result axis once, just like a batch axis. Other
      // singleton broadcasts introduce no independent contraction work.
      if (!reduced && leftBroadcast && rightBroadcast) {
        if (!left) {
          compatible = false;
          break;
        }
        leftBroadcast = false;
      }
      int64_t leftAxis = lhsKept.size(), rightAxis = rhsKept.size();
      if (!leftBroadcast)
        lhsKept.push_back(*left);
      if (!rightBroadcast)
        rhsKept.push_back(*right);
      if (reduced) {
        lhsReduced.push_back(leftAxis);
        rhsReduced.push_back(rightAxis);
      } else {
        if (!leftBroadcast)
          lhsOutput.push_back(axis);
        if (!leftBroadcast && !rightBroadcast) {
          lhsBatch.push_back(leftAxis);
          rhsBatch.push_back(rightAxis);
        } else if (!rightBroadcast) {
          rhsOutput.push_back(axis);
          ++rhsFree;
        } else {
          ++lhsFree;
        }
      }
    }
    // Preserve matrix-vector reuse before ownership can lift an independent
    // workset axis onto the other operand. Remaining vector contractions are
    // realized as native reductions after blocking.
    if (!compatible || (!lhsFree && !rhsFree))
      continue;
    OpBuilder builder(reduce);
    auto squeeze = [&](Value value, ArrayRef<int64_t> kept) -> Value {
      auto source = cast<FragmentType>(value.getType());
      if (kept.size() == source.getShape().size())
        return value;
      SmallVector<Attribute> shape, mappings, groups;
      for (unsigned axis = 0; axis < source.getShape().size(); ++axis) {
        SmallVector<int64_t> resultAxes;
        if (llvm::is_contained(kept, axis)) {
          resultAxes.push_back(shape.size());
          shape.push_back(source.getShape()[axis]);
          auto map = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
          mappings.push_back(AxisMapAttr::get(
              module.getContext(), map.getSourceId(), map.getSourceAxis(),
              map.getDimensionId(), mappings.size(), map.getDerived()));
        }
        groups.push_back(ReshapeGroupAttr::get(
            module.getContext(), builder.getDenseI64ArrayAttr({axis}),
            builder.getDenseI64ArrayAttr(resultAxes)));
      }
      auto target = FragmentType::get(
          module.getContext(), source.getElementType(),
          builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
          source.getValidity(), source.getOwner());
      return builder.create<ReshapeOp>(reduce.getLoc(), target, value,
                                       builder.getArrayAttr(groups));
    };
    lhs = squeeze(lhs, lhsKept);
    rhs = squeeze(rhs, rhsKept);
    // Preserve the innermost result coordinates as the matrix column axes.
    // The multiplication's operand order does not define matrix orientation.
    if (lhsBatch.empty() && rhsBatch.empty() && !lhsOutput.empty() &&
        !rhsOutput.empty() &&
        lhsOutput.back() > rhsOutput.back()) {
      std::swap(lhs, rhs);
      std::swap(lhsReduced, rhsReduced);
      std::swap(lhsOutput, rhsOutput);
    }
    SmallVector<int64_t> outputAxes(lhsOutput);
    llvm::append_range(outputAxes, rhsOutput);
    SmallVector<Attribute> shape, mappings;
    auto appendOutput = [&](Value value, ArrayRef<int64_t> reduced,
                            ArrayRef<int64_t> batched) {
      auto type = cast<FragmentType>(value.getType());
      for (unsigned axis = 0; axis < type.getShape().size(); ++axis) {
        if (llvm::is_contained(reduced, axis) ||
            llvm::is_contained(batched, axis))
          continue;
        shape.push_back(type.getShape()[axis]);
        auto map = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
        mappings.push_back(AxisMapAttr::get(
            module.getContext(), map.getSourceId(), map.getSourceAxis(),
            map.getDimensionId(), mappings.size(), map.getDerived()));
      }
    };
    appendOutput(lhs, lhsReduced, {});
    appendOutput(rhs, rhsReduced, rhsBatch);
    auto contractedType = FragmentType::get(
        module.getContext(), resultType.getElementType(),
        builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
        resultType.getValidity(), resultType.getOwner());
    auto zero = projectPhysicalValueToSchema(
        builder, reduce.getLoc(), reduce.getInputs()[1], contractedType);
    if (failed(zero))
      return reduce.emitOpError("cannot form the multiply-reduction identity");
    auto contract = builder.create<ContractOp>(
        reduce.getLoc(), contractedType, lhs, rhs, *zero, lhsReduced, rhsReduced,
        lhsBatch, rhsBatch);
    if (Attribute origin = reduce->getAttr(originAttr))
      contract->setAttr(originAttr, origin);
    SmallVector<int64_t> permutation;
    for (unsigned axis = 0; axis < productType.getShape().size(); ++axis) {
      if (llvm::is_contained(reduce.getAxes(), axis))
        continue;
      permutation.push_back(llvm::find(outputAxes, axis) - outputAxes.begin());
    }
    Value replacement = contract.getResult();
    if (!llvm::all_of(llvm::enumerate(permutation), [](auto item) {
          return item.index() == static_cast<unsigned>(item.value());
        })) {
      SmallVector<Attribute> reorderedShape, reorderedMappings;
      for (int64_t axis : permutation) {
        reorderedShape.push_back(shape[axis]);
        auto map = cast<AxisMapAttr>(mappings[axis]);
        reorderedMappings.push_back(AxisMapAttr::get(
            module.getContext(), map.getSourceId(), map.getSourceAxis(),
            map.getDimensionId(), reorderedMappings.size(), map.getDerived()));
      }
      auto reorderedType = FragmentType::get(
          module.getContext(), resultType.getElementType(),
          builder.getArrayAttr(reorderedShape),
          builder.getArrayAttr(reorderedMappings),
          resultType.getValidity(), resultType.getOwner());
      replacement = builder.create<TransposeOp>(
          reduce.getLoc(), reorderedType, replacement, permutation);
    }
    if (replacement.getType() != resultType)
      replacement = builder.create<BroadcastOp>(reduce.getLoc(), resultType,
                                                replacement);
    reduce.getResult(0).replaceAllUsesWith(replacement);
    reduce.erase();
  }
  eraseDeadPhysicalValues(*kernel);
  return success();
}

LogicalResult normalizeContractionSources(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  if (failed(fuseMultiplyReductions(module)) ||
      failed(realizeAccessComposition(module)))
    return failure();
  orientContractionOutputs(*kernel);
  SmallVector<ContractOp> contracts;
  kernel->walk([&](ContractOp contract) { contracts.push_back(contract); });
  // Collapse complete logical reduction ranges before ownership introduces
  // physical padding or scalar tiles that no longer admit this reassociation.
  bool collapsed = false;
  for (ContractOp contract : contracts) {
    auto result = collapseMultiReductionContract(contract);
    if (failed(result))
      return failure();
    collapsed |= *result;
  }
  if (collapsed && failed(realizeAccessComposition(module)))
    return failure();
  contracts.clear();
  kernel->walk([&](ContractOp contract) { contracts.push_back(contract); });
  bool changed = false;
  for (ContractOp contract : contracts) {
    if (!contract.getResult().hasOneUse() ||
        contract.getLhs().getType().getShape().size() != 2 ||
        contract.getRhs().getType().getShape().size() != 2 ||
        contract.getLhsReductionAxes() != ArrayRef<int64_t>{1} ||
        (contract.getRhsReductionAxes() != ArrayRef<int64_t>{0} &&
         contract.getRhsReductionAxes() != ArrayRef<int64_t>{1}) ||
        !contract.getLhsBatchAxes().empty() ||
        !contract.getRhsBatchAxes().empty())
      continue;
    auto input = contract.getLhs().getDefiningOp<ReshapeOp>();
    auto output = dyn_cast<ReshapeOp>(*contract.getResult().getUsers().begin());
    auto identity = contract.getAccumulator().getDefiningOp<SplatOp>();
    if (!identity)
      continue;
    if (!output) {
      output = exposeTransposedContractSplit(contract);
      changed |= static_cast<bool>(output);
    }
    if (!output || !identity || output.getReassociation().size() != 2)
      continue;
    auto target = cast<FragmentType>(output.getResult().getType());
    auto outputFree = cast<ReshapeGroupAttr>(output.getReassociation()[0]);
    auto outputColumn = cast<ReshapeGroupAttr>(output.getReassociation()[1]);
    unsigned freeRank = outputFree.getResultAxes().size();
    if (freeRank < 2 || target.getShape().size() != freeRank + 1 ||
        outputFree.getSourceAxes().asArrayRef() != ArrayRef<int64_t>{0} ||
        outputColumn.getSourceAxes().asArrayRef() != ArrayRef<int64_t>{1} ||
        outputColumn.getResultAxes().asArrayRef() !=
            ArrayRef<int64_t>{static_cast<int64_t>(freeRank)} ||
        llvm::any_of(llvm::enumerate(outputFree.getResultAxes().asArrayRef()),
                     [](auto item) {
                       return item.value() != static_cast<int64_t>(item.index());
                     }))
      continue;
    Value sourceValue = contract.getLhs();
    auto source = contract.getLhs().getType();
    ReshapeGroupAttr inputReduction;
    bool cancelsInput = false;
    if (input && input.getReassociation().size() == 2) {
      auto inputType = cast<FragmentType>(input.getValue().getType());
      auto inputFree = cast<ReshapeGroupAttr>(input.getReassociation()[0]);
      inputReduction = cast<ReshapeGroupAttr>(input.getReassociation()[1]);
      cancelsInput =
          inputFree.getResultAxes().asArrayRef() == ArrayRef<int64_t>{0} &&
          inputReduction.getResultAxes().asArrayRef() == ArrayRef<int64_t>{1} &&
          inputFree.getSourceAxes() == outputFree.getResultAxes();
      for (unsigned axis = 0; cancelsInput && axis < freeRank; ++axis) {
        auto original = cast<AxisMapAttr>(inputType.getAxisMaps()[axis]);
        auto restored = cast<AxisMapAttr>(target.getAxisMaps()[axis]);
        cancelsInput = original.getDimensionId() > 0 &&
                       original.getDimensionId() == restored.getDimensionId() &&
                       inputType.getShape()[axis] == target.getShape()[axis];
      }
      if (cancelsInput) {
        sourceValue = input.getValue();
        source = inputType;
      }
    }

    // Move the declared row-major free-axis split through the contraction.
    // An inverse input reshape can be cancelled while retaining its original
    // coordinate roots; otherwise the same split applies to the flat lhs.
    OpBuilder builder(contract);
    if (contract.getRhsReductionAxes() == ArrayRef<int64_t>{1}) {
      Value rhs = contract.getRhs();
      contract.getRhsMutable().assign(builder.create<TransposeOp>(
          contract.getLoc(), transposeLastTwo(cast<FragmentType>(rhs.getType())),
          rhs, builder.getDenseI64ArrayAttr({1, 0})));
      contract->setAttr("rhs_reduction_axes", builder.getDenseI64ArrayAttr({0}));
    }
    SmallVector<Attribute> shape, mappings, groups;
    for (unsigned axis = 0; axis < freeRank; ++axis) {
      shape.push_back(target.getShape()[axis]);
      mappings.push_back(cancelsInput ? source.getAxisMaps()[axis]
                                     : target.getAxisMaps()[axis]);
      if (cancelsInput)
        groups.push_back(ReshapeGroupAttr::get(
            module.getContext(), builder.getDenseI64ArrayAttr({axis}),
            builder.getDenseI64ArrayAttr({axis})));
    }
    if (!cancelsInput)
      groups.push_back(outputFree);
    auto matrix = contract.getLhs().getType();
    auto reduction = cast<AxisMapAttr>(matrix.getAxisMaps()[1]);
    shape.push_back(matrix.getShape()[1]);
    mappings.push_back(AxisMapAttr::get(
        module.getContext(), reduction.getSourceId(), reduction.getSourceAxis(),
        reduction.getDimensionId(), freeRank, reduction.getDerived()));
    groups.push_back(ReshapeGroupAttr::get(
        module.getContext(), cancelsInput ? inputReduction.getSourceAxes()
                                         : builder.getDenseI64ArrayAttr({1}),
        builder.getDenseI64ArrayAttr({freeRank})));
    auto operandType = FragmentType::get(
        module.getContext(), matrix.getElementType(), builder.getArrayAttr(shape),
        builder.getArrayAttr(mappings), matrix.getValidity(), matrix.getOwner());
    while (auto cast = sourceValue.getDefiningOp<CastOp>()) {
      if (cast.getValue().getType() != sourceValue.getType())
        break;
      sourceValue = cast.getValue();
    }
    Value operand = builder.create<ReshapeOp>(
        contract.getLoc(), operandType, sourceValue, builder.getArrayAttr(groups));
    SmallVector<Attribute> resultMappings(mappings.begin(),
                                          mappings.begin() + freeRank);
    auto column = cast<AxisMapAttr>(contract.getRhs().getType().getAxisMaps()[1]);
    resultMappings.push_back(AxisMapAttr::get(
        module.getContext(), column.getSourceId(), column.getSourceAxis(),
        column.getDimensionId(), freeRank, column.getDerived()));
    auto resultType = FragmentType::get(
        module.getContext(), target.getElementType(), target.getShape(),
        builder.getArrayAttr(resultMappings), target.getValidity(),
        target.getOwner());
    Value accumulator = builder.create<SplatOp>(output.getLoc(), resultType,
                                               identity.getValue());
    contract.getLhsMutable().assign(operand);
    contract.getAccumulatorMutable().assign(accumulator);
    contract->setAttr("lhs_reduction_axes",
                      builder.getDenseI64ArrayAttr({freeRank}));
    contract.getResult().setType(resultType);
    builder.setInsertionPointAfter(contract);
    Value projected = builder.create<BroadcastOp>(output.getLoc(), target,
                                                 contract.getResult());
    output.getResult().replaceAllUsesWith(projected);
    output.erase();
    changed = true;
  }
  if (changed) {
    SmallVector<CastOp> identityCasts;
    kernel->walk([&](CastOp cast) {
      if (cast.getValue().getType() == cast.getResult().getType())
        identityCasts.push_back(cast);
    });
    for (CastOp cast : identityCasts) {
      cast.getResult().replaceAllUsesWith(cast.getValue());
      cast.erase();
    }
    eraseDeadPhysicalValues(*kernel);
    if (failed(realizeAccessComposition(module)))
      return failure();
  }
  return success();
}

LogicalResult realizeContractionBlocking(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  fuseContractionAdds(kernel);
  SmallVector<ContractOp> contracts;
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  for (ContractOp contract : contracts)
    if (failed(projectContractResult(contract)))
      return failure();
  eraseDeadPhysicalValues(kernel);
  if (failed(realizeAccessComposition(module)))
    return failure();
  contracts.clear();
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  bool collapsed = false;
  for (ContractOp contract : contracts) {
    auto result = collapseMultiReductionContract(contract);
    if (failed(result))
      return failure();
    collapsed |= *result;
  }
  if (collapsed && failed(realizeAccessComposition(module)))
    return failure();
  contracts.clear();
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  for (ContractOp contract : contracts)
    if (contract.getLhsReductionAxes().size() > 1 &&
        failed(decomposeMultiReductionContract(contract)))
      return failure();
  contracts.clear();
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  if (failed(normalizeMatrixContractForms(kernel)))
    return failure();
  // Close retained read snapshots before creating any contraction slices.
  // Materializing one operand later can otherwise retarget an existing slice
  // that shares the same logical dimension with that snapshot.
  WalkResult snapshots = kernel.walk([&](ContractOp contract) {
    for (auto [value, axes] :
         {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
          std::pair{contract.getRhs(), contract.getRhsReductionAxes()}}) {
      for (int64_t axis : axes) {
        PhysicalProgramAnalysis analysis(kernel);
        auto fragment = cast<FragmentType>(value.getType());
        auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
        PhysicalReplayFact replay = analysis.replayability(
            value, sourceAxisIdentity(mapping), PhysicalReplayScope::ValueGraph,
            /*allowAccesses=*/true, /*insertionAnchor=*/nullptr,
            mapping.getDimensionId());
        if (!replay.isReplayable())
          continue;
        bool retain = llvm::any_of(replay.accesses, [&](Operation *access) {
          auto load = dyn_cast<LoadOp>(access);
          return load && !canReplayReadAt(load, contract);
        });
        if (retain && !analysis.axisRealization(value, axis).physicalized &&
            failed(realizeFullCoverageDimension(kernel, value, axis)))
          return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (snapshots.wasInterrupted())
    return failure();
  llvm::SmallPtrSet<Operation *, 16> epilogueProducts;
  auto realizeSelectedContract = [&](ContractOp contract) -> LogicalResult {
    size_t previousSize = contracts.size();
    if (failed(realizeContract(contract, kernel, contracts))) return failure();
    for (ContractOp replayed : ArrayRef<ContractOp>(contracts).drop_front(previousSize))
      epilogueProducts.insert(replayed.getOperation());
    return success();
  };
  for (size_t contractIndex = 0; contractIndex < contracts.size();
       ++contractIndex) {
    ContractOp contract = contracts[contractIndex];
    if (contract.getResult().use_empty()) {
      contract.erase();
      continue;
    }
    if (!hasFragmentSchema(contract))
      return contract.emitOpError(
          "shared contraction blocking requires fragment operands and result");
    // A complete matrix-to-store path can tile the result and its epilogue
    // together. Retaining the full result first would hide that ownership.
    if (!requiresPhysicalRealization(contract) || !hasRangeContractForm(contract)) {
      FailureOr<bool> fullResult =
          realizeFullResultTraversal(contract, kernel, contracts);
      if (failed(fullResult))
        return failure();
      if (*fullResult)
        continue;
    }
    bool realizeOutput = outputCoordinatesNeedRealization(contract);
    if (realizeOutput) {
      if (failed(realizeSelectedContract(contract)))
        return failure();
      continue;
    }
    if (requiresPhysicalRealization(contract)) {
      FailureOr<bool> nativeSegment =
          realizeSegmentNativeReduction(contract, kernel);
      if (failed(nativeSegment))
        return failure();
      if (*nativeSegment)
        continue;
      FailureOr<bool> nativeStructured =
          realizeStructuredNativeReduction(contract, kernel, contracts);
      if (failed(nativeStructured))
        return failure();
      if (*nativeStructured)
        continue;
      const bool rangeSingleReduction = hasRangeContractForm(contract);
      if ((!rangeSingleReduction || epilogueProducts.contains(contract)) &&
          contract.getLhsReductionAxes().size() == 1 &&
          contract.getRhsReductionAxes().size() == 1 &&
          freeAxesReadyForReductionTraversal(contract, kernel) &&
          reductionAxesNeedTraversal(contract, kernel) &&
          hasExplicitPairedReductionRanges(contract)) {
        SmallVector<ContractOp> replayed;
        if (failed(realizeReductionTraversal(contract, kernel, replayed)))
          return failure();
        contracts.append(replayed.begin(), replayed.end());
      } else if (!rangeSingleReduction &&
                 contract.getLhsReductionAxes().size() == 1 &&
                 contract.getRhsReductionAxes().size() == 1 &&
                 contract.getLhsBatchAxes().empty() &&
                 contract.getRhsBatchAxes().empty() &&
                 hasCompleteStorePath(contract) &&
                 freeAxesNeedRealization(contract, kernel)) {
        if (failed(realizeSelectedContract(contract)))
          return failure();
      } else if (!rangeSingleReduction &&
                 contract.getLhsReductionAxes().size() == 1 &&
                 contract.getRhsReductionAxes().size() == 1) {
        if (failed(markNativeCoverage(kernel, contract)))
          return failure();
      } else if (failed(realizeSelectedContract(contract))) {
        return failure();
      }
    } else if (failed(markNativeCoverage(kernel, contract))) {
      return failure();
    }
  }

  SmallVector<SparseContractOp> sparseContracts;
  SmallVector<ScaledContractOp> scaledContracts;
  kernel.walk(
      [&](SparseContractOp contract) { sparseContracts.push_back(contract); });
  kernel.walk(
      [&](ScaledContractOp contract) { scaledContracts.push_back(contract); });
  for (ScaledContractOp contract : scaledContracts)
    if (!hasFragmentSchema(contract))
      return contract.emitOpError(
          "shared scaled-contraction blocking requires fragment operands, accumulator, and result");
  for (SparseContractOp contract : sparseContracts)
    if (!hasFragmentSchema(contract))
      return contract.emitOpError(
          "shared sparse-contraction blocking requires fragment operands, metadata, accumulator, and result");
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
  WalkResult tails = kernel.walk([&](ContractOp contract) {
    if (failed(neutralizeFullCoverageOperand(
            contract, contract.getLhsMutable(), contract.getLhsReductionAxes(),
            kernel)) ||
        failed(neutralizeFullCoverageOperand(
            contract, contract.getRhsMutable(), contract.getRhsReductionAxes(),
            kernel)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (tails.wasInterrupted())
    return failure();
  eraseDeadPhysicalValues(kernel);
  // A reduction can consume a provisional pointwise axis. Once its old value
  // graph is gone, that unused axis must not multiply the output workset.
  SmallVector<DelinearizeOp> mappings;
  kernel.walk([&](DelinearizeOp mapping) { mappings.push_back(mapping); });
  for (DelinearizeOp mapping : mappings) {
    auto roles = mapping->getAttrOfType<DenseI64ArrayAttr>(coordinateRolesAttr);
    auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
    auto offset = mapping->getAttrOfType<PhysicalExprAttr>(segmentOffsetAttr);
    auto length = mapping->getAttrOfType<PhysicalExprAttr>(segmentLengthAttr);
    auto program = mapping.getLinear().getDefiningOp<ProgramIdOp>();
    if (!roles || roles.size() != mapping.getNumResults() ||
        !llvm::is_contained(roles.asArrayRef(),
                            static_cast<int64_t>(CoordinateRole::ContractionM)) ||
        !llvm::is_contained(roles.asArrayRef(),
                            static_cast<int64_t>(CoordinateRole::ContractionN)) ||
        !program || program.getAxis() != 0 ||
        !program.getResult().hasOneUse() || !space ||
        space.size() != 1 || !offset || !length || space[0] != length ||
        offset.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) ||
        offset.getValue() != 0)
      continue;
    SmallVector<Type> types;
    SmallVector<Value> extents;
    SmallVector<Attribute> launch;
    SmallVector<int64_t> retainedRoles;
    SmallVector<Value> retainedCoordinates;
    for (auto [axis, coordinate] : llvm::enumerate(mapping.getCoordinates())) {
      if (coordinate.use_empty() &&
          roles[axis] == static_cast<int64_t>(CoordinateRole::PointwiseOwnership))
        continue;
      types.push_back(coordinate.getType());
      extents.push_back(mapping.getExtents()[axis]);
      launch.push_back(mapping.getLaunchExtents()[axis]);
      retainedRoles.push_back(roles[axis]);
      retainedCoordinates.push_back(coordinate);
    }
    if (types.size() == mapping.getNumResults())
      continue;
    OpBuilder builder(mapping);
    auto compact = builder.create<DelinearizeOp>(
        mapping.getLoc(), types, mapping.getLinear(), extents,
        builder.getArrayAttr(launch));
    compact->setAttrs(mapping->getAttrs());
    compact.setLaunchExtentsAttr(builder.getArrayAttr(launch));
    compact->setAttr(coordinateRolesAttr,
                     builder.getDenseI64ArrayAttr(retainedRoles));
    length = cast<PhysicalExprAttr>(launch.front());
    for (Attribute extent : llvm::drop_begin(launch))
      length = binaryExpression(kernel.getContext(), PhysicalExprKind::Multiply,
                                length, cast<PhysicalExprAttr>(extent));
    compact->setAttr(segmentLengthAttr, length);
    kernel->setAttr(programSpaceAttr, builder.getArrayAttr({length}));
    for (auto [old, current] :
         llvm::zip(retainedCoordinates, compact.getCoordinates()))
      old.replaceAllUsesWith(current);
    mapping.erase();
  }
  eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace intent::gpu
