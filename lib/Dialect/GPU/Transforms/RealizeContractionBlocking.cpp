#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"

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

bool requiresPhysicalRealization(ContractOp contract) {
  for (FragmentType type : {contract.getLhs().getType(),
                            contract.getRhs().getType(),
                            contract.getResult().getType()})
    if (llvm::any_of(type.getShape(), [](Attribute extent) {
          return !isCompileTimeExtent(cast<PhysicalExprAttr>(extent));
        }))
      return true;
  return false;
}

Value binary(OpBuilder &builder, Location location, Type result, Value lhs,
             Value rhs, uint64_t kind) {
  return builder.create<BinaryOp>(location, result, lhs, rhs, kind);
}

Value compare(OpBuilder &builder, Location location, Type result, Value lhs,
              Value rhs, uint64_t predicate) {
  return builder.create<CompareOp>(location, result, lhs, rhs, predicate);
}

FailureOr<AxisMapAttr> axisMap(FragmentType fragment, unsigned axis) {
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getFragmentAxis() == axis)
      return mapping;
  }
  return failure();
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

FailureOr<unsigned> coordinateForSource(ValueRange coordinates,
                                        uint64_t sourceId) {
  std::optional<unsigned> result;
  for (auto [index, coordinate] : llvm::enumerate(coordinates)) {
    auto fragment = dyn_cast<FragmentType>(coordinate.getType());
    if (!fragment)
      continue;
    bool matches = llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
      return cast<AxisMapAttr>(attribute).getSourceId() == sourceId;
    });
    if (!matches)
      continue;
    if (result)
      return failure();
    result = index;
  }
  return result ? FailureOr<unsigned>(*result) : FailureOr<unsigned>(failure());
}

FragmentType fragmentType(MLIRContext *context, Type element,
                          ArrayRef<PhysicalExprAttr> shape,
                          ArrayRef<AxisMapAttr> sourceMappings,
                          uint64_t owner) {
  SmallVector<Attribute> extents(shape.begin(), shape.end());
  SmallVector<Attribute> mappings;
  for (auto [axis, source] : llvm::enumerate(sourceMappings))
    mappings.push_back(AxisMapAttr::get(context, source.getSourceId(),
                                        source.getSourceAxis(), axis));
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
    zero = builder.create<arith::ConstantOp>(
        location, integer, builder.getIntegerAttr(integer, 0));
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
  auto lhsLoad = contract.getLhs().getDefiningOp<LoadOp>();
  auto rhsLoad = contract.getRhs().getDefiningOp<LoadOp>();
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
  FailureOr<AxisMapAttr> rowMap = axisMap(lhsType, *lhsFree);
  FailureOr<AxisMapAttr> lhsReductionMap = axisMap(lhsType, lhsReduction);
  FailureOr<AxisMapAttr> rhsReductionMap = axisMap(rhsType, rhsReduction);
  FailureOr<AxisMapAttr> columnMap = axisMap(rhsType, *rhsFree);
  if (failed(rowMap) || failed(lhsReductionMap) || failed(rhsReductionMap) ||
      failed(columnMap) ||
      lhsReductionMap->getSourceId() != rhsReductionMap->getSourceId())
    return unhandled("paired reduction coordinates lack one shared provenance");

  FailureOr<unsigned> lhsRowCoordinate =
      coordinateForSource(lhsLoad.getCoordinates(), rowMap->getSourceId());
  FailureOr<unsigned> lhsReductionCoordinate = coordinateForSource(
      lhsLoad.getCoordinates(), lhsReductionMap->getSourceId());
  FailureOr<unsigned> rhsReductionCoordinate = coordinateForSource(
      rhsLoad.getCoordinates(), rhsReductionMap->getSourceId());
  FailureOr<unsigned> rhsColumnCoordinate =
      coordinateForSource(rhsLoad.getCoordinates(), columnMap->getSourceId());
  if (failed(lhsRowCoordinate) || failed(lhsReductionCoordinate) ||
      failed(rhsReductionCoordinate) || failed(rhsColumnCoordinate))
    return unhandled("load coordinates do not cover all free/reduction axes");
  auto rowRange = lhsLoad.getCoordinates()[*lhsRowCoordinate]
                      .getDefiningOp<MakeRangeOp>();
  auto lhsReductionRange =
      lhsLoad.getCoordinates()[*lhsReductionCoordinate]
          .getDefiningOp<MakeRangeOp>();
  auto rhsReductionRange =
      rhsLoad.getCoordinates()[*rhsReductionCoordinate]
          .getDefiningOp<MakeRangeOp>();
  auto columnRange = rhsLoad.getCoordinates()[*rhsColumnCoordinate]
                         .getDefiningOp<MakeRangeOp>();
  if (!rowRange || !lhsReductionRange || !rhsReductionRange || !columnRange ||
      lhsReductionRange.getSourceId() != rhsReductionRange.getSourceId())
    return unhandled("physical coordinates are not explicit compatible ranges");
  auto rowBound = rowRange.getStart().getDefiningOp<RangeBoundOp>();
  auto rowSourceRange =
      rowBound ? rowBound.getRange().getDefiningOp<RangeOp>() : RangeOp();
  const bool runtimeRowTraversal =
      rowSourceRange && rowSourceRange->hasAttr(sourceSubregionAttr);

  FailureOr<Value> rowStep = scalarSource(rowRange.getStep());
  FailureOr<Value> reductionStep = scalarSource(lhsReductionRange.getStep());
  FailureOr<Value> columnStep = scalarSource(columnRange.getStep());
  auto isUnit = [](FailureOr<Value> value) {
    if (failed(value))
      return false;
    Value current = *value;
    while (true) {
      if (auto bound = current.getDefiningOp<RangeBoundOp>()) {
        if (bound.getBound() != 2)
          return false;
        auto range = bound.getRange().getDefiningOp<RangeOp>();
        if (!range)
          return false;
        current = range.getStep();
        continue;
      }
      if (auto cast = current.getDefiningOp<CastOp>()) {
        current = cast.getValue();
        continue;
      }
      break;
    }
    auto constant = current.getDefiningOp<arith::ConstantOp>();
    auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                            : IntegerAttr();
    return integer && integer.getInt() == 1;
  };
  if (!isUnit(rowStep) || !isUnit(reductionStep) || !isUnit(columnStep))
    return unhandled("blocking currently requires unit-step source ranges");

  SmallVector<StorePath> paths;
  llvm::SmallPtrSet<Operation *, 8> visited;
  if (!collectStorePaths(contract.getResult(), {}, paths, visited))
    return unhandled("result does not have a complete unique-store path");
  for (StorePath &path : paths)
    if (path.store.getValid() && failed(scalarSource(path.store.getValid())))
      return unhandled("result store has non-scalar residual validity");
  if (lhsLoad.getValid() && failed(scalarSource(lhsLoad.getValid())))
    return unhandled("lhs load has non-scalar residual validity");
  if (rhsLoad.getValid() && failed(scalarSource(rhsLoad.getValid())))
    return unhandled("rhs load has non-scalar residual validity");
  if (lhsLoad.getFill() && !isZeroScalar(lhsLoad.getFill()))
    return unhandled("lhs invalid fill is not the contraction zero");
  if (rhsLoad.getFill() && !isZeroScalar(rhsLoad.getFill()))
    return unhandled("rhs invalid fill is not the contraction zero");
  FailureOr<Value> initialAccumulator = scalarSource(contract.getAccumulator());
  if (failed(initialAccumulator) ||
      (*initialAccumulator).getType() !=
          contract.getResult().getType().getElementType())
    return unhandled("accumulator is not an explicit scalarizable value");

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
      segmentOffset.getValue() != 0 || programSpace[0] != segmentLength)
    return unhandled("current rule requires one full-program execution segment");

  unsigned rowResourceAxis = lhsLoad.getSourceAxes()[*lhsRowCoordinate];
  unsigned columnResourceAxis = rhsLoad.getSourceAxes()[*rhsColumnCoordinate];
  MLIRContext *context = kernel.getContext();
  Location location = contract.getLoc();
  std::string suffix =
      ("_" + Twine(rowMap->getSourceId()) + "_" +
       Twine(columnMap->getSourceId()))
          .str();
  ParameterOp blockM = getOrCreateParameter(
      kernel, "BLOCK_M" + suffix, ParameterRole::OwnershipM, {32, 64});
  ParameterOp blockN = getOrCreateParameter(
      kernel, "BLOCK_N" + suffix, ParameterRole::OwnershipN, {64, 128});
  ParameterOp blockK = getOrCreateParameter(
      kernel, "BLOCK_K" + suffix, ParameterRole::Reduction, {32, 64});
  if (!blockM || !blockN || !blockK)
    return failure();
  ParameterOp rowWorkers;
  if (runtimeRowTraversal)
    rowWorkers = getOrCreateParameter(
        kernel, "ROW_WORKERS" + suffix, ParameterRole::OwnershipM, {1, 2, 4});
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
    Value adjusted = binary(mapBuilder, location, mapBuilder.getIndexType(), extent,
                            binary(mapBuilder, location, mapBuilder.getIndexType(),
                                   divisor, one, 1),
                            0);
    return binary(mapBuilder, location, mapBuilder.getIndexType(), adjusted,
                  divisor, 4);
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
  Value rowStop = binary(builder, location, builder.getIndexType(),
                         rowRange.getStart(), rowRange.getExtent(), 0);
  Value columnStart = binary(
      builder, location, builder.getIndexType(), columnRange.getStart(),
      binary(builder, location, builder.getIndexType(), columnTile,
             blockN.getResult(), 2),
      0);
  Value columnStop = binary(builder, location, builder.getIndexType(),
                            columnRange.getStart(), columnRange.getExtent(), 0);
  Value reductionStop =
      binary(builder, location, builder.getIndexType(),
             lhsReductionRange.getStart(), lhsReductionRange.getExtent(), 0);

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
  Value columnEnd = broadcast(builder, location, columnIndexType, columnStop);
  Value columnValid =
      compare(builder, location, columnPredicateType, columns, columnEnd, 2);
  auto emitRowBlock = [&](OpBuilder &rowBuilder,
                          Value rowStart) -> LogicalResult {
    Value rows = rowBuilder.create<MakeRangeOp>(
        location, rowIndexType, rowStart, blockM.getResult(), one,
        rowMap->getSourceId(), rowMap->getSourceAxis());
    Value rowEnd = broadcast(rowBuilder, location, rowIndexType, rowStop);
    Value rowValid =
        compare(rowBuilder, location, rowPredicateType, rows, rowEnd, 2);
    Value accumulator = rowBuilder.create<SplatOp>(
        location, blockedResultType, *initialAccumulator);

    bool loopBodyFailed = false;
    auto loop = rowBuilder.create<scf::ForOp>(
        location, lhsReductionRange.getStart(), reductionStop,
        blockK.getResult(), ValueRange{accumulator},
        [&](OpBuilder &nested, Location nestedLocation, Value kStart,
            ValueRange carries) {
          Value reductions = nested.create<MakeRangeOp>(
              nestedLocation, reductionIndexType, kStart, blockK.getResult(), one,
              lhsReductionMap->getSourceId(), lhsReductionMap->getSourceAxis());
          Value reductionEnd =
              broadcast(nested, nestedLocation, reductionIndexType, reductionStop);
          Value reductionValid = compare(nested, nestedLocation,
                                         reductionPredicateType, reductions,
                                         reductionEnd, 2);
          Value lhsRows =
              broadcast(nested, nestedLocation, lhsPredicateType, rowValid);
          Value lhsReductions = broadcast(nested, nestedLocation,
                                          lhsPredicateType, reductionValid);
          Value lhsValid = binary(nested, nestedLocation, lhsPredicateType,
                                  lhsRows, lhsReductions, 11);
          Value rhsReductions = broadcast(nested, nestedLocation,
                                          rhsPredicateType, reductionValid);
          Value rhsColumns =
              broadcast(nested, nestedLocation, rhsPredicateType, columnValid);
          Value rhsValid = binary(nested, nestedLocation, rhsPredicateType,
                                  rhsReductions, rhsColumns, 11);
          if (lhsLoad.getValid()) {
            FailureOr<Value> original = retargetPredicate(
                nested, nestedLocation, lhsLoad.getValid(), lhsPredicateType);
            if (failed(original)) {
              loopBodyFailed = true;
              return;
            }
            lhsValid = binary(nested, nestedLocation, lhsPredicateType, lhsValid,
                              *original, 11);
          }
          if (rhsLoad.getValid()) {
            FailureOr<Value> original = retargetPredicate(
                nested, nestedLocation, rhsLoad.getValid(), rhsPredicateType);
            if (failed(original)) {
              loopBodyFailed = true;
              return;
            }
            rhsValid = binary(nested, nestedLocation, rhsPredicateType, rhsValid,
                              *original, 11);
          }
          SmallVector<Value> lhsCoordinates(lhsLoad.getCoordinates());
          lhsCoordinates[*lhsRowCoordinate] = rows;
          lhsCoordinates[*lhsReductionCoordinate] = reductions;
          SmallVector<Value> rhsCoordinates(rhsLoad.getCoordinates());
          rhsCoordinates[*rhsReductionCoordinate] = reductions;
          rhsCoordinates[*rhsColumnCoordinate] = columns;
          FailureOr<Value> lhsFill = retargetFill(
              nested, nestedLocation, lhsLoad.getFill(), blockedLhsType);
          FailureOr<Value> rhsFill = retargetFill(
              nested, nestedLocation, rhsLoad.getFill(), blockedRhsType);
          if (failed(lhsFill) || failed(rhsFill)) {
            loopBodyFailed = true;
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
    if (loopBodyFailed) {
      loop.erase();
      return contract.emitOpError(
          "blocked contraction could not materialize its loop body");
    }

    Value outputRows =
        broadcast(rowBuilder, location, outputPredicateType, rowValid);
    Value outputColumns =
        broadcast(rowBuilder, location, outputPredicateType, columnValid);
    Value outputValid = binary(rowBuilder, location, outputPredicateType,
                               outputRows, outputColumns, 11);
    for (StorePath &path : paths) {
      Value output = loop.getResult(0);
      for (CastOp conversion : path.casts) {
        auto original = llvm::cast<FragmentType>(conversion.getResult().getType());
        FragmentType converted = fragmentType(
            context, original.getElementType(), {unitM, unitN},
            {*rowMap, *columnMap}, original.getOwner());
        output = rowBuilder.create<CastOp>(location, converted, output);
      }
      SmallVector<Value> coordinates(path.store.getCoordinates());
      FailureOr<unsigned> storeRow = coordinateForSource(
          path.store.getCoordinates(), rowMap->getSourceId());
      FailureOr<unsigned> storeColumn = coordinateForSource(
          path.store.getCoordinates(), columnMap->getSourceId());
      if (failed(storeRow) || failed(storeColumn))
        return path.store.emitOpError(
            "blocked contract output lost its logical source coordinates");
      coordinates[*storeRow] = rows;
      coordinates[*storeColumn] = columns;
      Value valid = outputValid;
      if (path.store.getValid()) {
        FailureOr<Value> original = retargetPredicate(
            rowBuilder, location, path.store.getValid(), outputPredicateType);
        if (failed(original))
          return path.store.emitOpError(
              "blocked contract output has non-scalar residual validity");
        valid = binary(rowBuilder, location, outputPredicateType, valid,
                       *original, 11);
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
               blockM.getResult(), 2),
        0);
    Value rowStep = binary(builder, location, builder.getIndexType(),
                           blockM.getResult(), rowWorkers.getResult(), 2);
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
               blockM.getResult(), 2),
        0);
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
  eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace

LogicalResult realizeContractionBlocking(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<ContractOp> contracts;
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  if (contracts.size() != 1) {
    for (ContractOp contract : contracts)
      if (requiresPhysicalRealization(contract))
        return contract.emitOpError(
            "joint blocking for multiple runtime-shaped contractions is not materialized");
    return success();
  }
  for (ContractOp contract : contracts)
    if (failed(realizeContract(contract, kernel)))
      return failure();
  return success();
}

} // namespace intent::gpu
