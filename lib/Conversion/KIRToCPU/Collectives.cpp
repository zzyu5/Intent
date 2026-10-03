#include "Construction.h"
#include "Intent/Analysis/ProductSchema.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CollectiveHelpers.h"
#include "Intent/Interfaces/StructuredOpInterface.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace intent::kir_to_cpu {

LogicalResult Construction::helper(Region &original, Region &target, ValueRange captures) {
  auto savedDimensions = dimensions;
  OpBuilder::InsertionGuard guard(builder);
  Block *body = new Block;
  target.push_back(body);
  SmallVector<Type> inputTypes;
  appendProductLeafTypes(original.front().getArgumentTypes(), inputTypes);
  if (captures.size() > inputTypes.size())
    return original.getParentOp()->emitError("CPU helper capture schema is incomplete");
  for (auto [index, capture] : llvm::enumerate(captures))
    inputTypes[inputTypes.size() - captures.size() + index] = capture.getType();
  for (Type type : inputTypes) body->addArgument(valueType(type), original.getLoc());
  builder.setInsertionPointToStart(body);
  auto ranges = getProductLeafRanges(original.front().getArgumentTypes());
  for (auto [argument, range] : llvm::zip(original.front().getArguments(), ranges)) {
    SmallVector<Type> leaves;
    appendProductLeafTypes(argument.getType(), leaves);
    SmallVector<Value> parts;
    for (auto [leaf, target] :
         llvm::zip(leaves, body->getArguments().slice(range.offset, range.size))) {
      bindDimensions(leaf, target, original.getLoc());
      parts.push_back(target);
    }
    bindProduct(argument, parts);
  }
  if (failed(lowerBlock(original.front()))) return failure();
  auto results = flattened(original.front().getTerminator()->getOperands());
  if (isa<cpu::SliceReduceOp>(target.getParentOp()))
    builder.create<cpu::SliceReduceYieldOp>(original.getLoc(), results);
  else if (isa<cpu::ScanOp>(target.getParentOp()))
    builder.create<cpu::ScanYieldOp>(original.getLoc(), results);
  else builder.create<cpu::RegionYieldOp>(original.getLoc(), results);
  dimensions = std::move(savedDimensions);
  return success();
}

LogicalResult Construction::region(Operation *operation) {
  auto schema = cast<StructuredOpInterface>(operation);
  const bool scan = schema.getStructuredKind() == StructuredOpKind::RegionScan;
  int64_t axis = schema.getIterationAxes().front();
  auto sources = flattened(schema.getSources());
  auto identityValues = flattened(schema.getIdentities());
  auto stateValues = flattened(schema.getInitialStates());
  auto captures = flattened(schema.getCaptures());
  Location loc = operation->getLoc();
  SmallVector<Type> results;
  appendProductLeafTypes(operation->getResultTypes(), results);
  for (Type &type : results) type = valueType(type);
  auto outputs = emptyResults(operation->getResultTypes(), loc);
  if (failed(outputs)) return failure();
  auto outputFields = fieldPaths(schema.getEmittedResults().getTypes());
  SmallVector<int64_t> outputAxes;
  if (scan) {
    Block &emit = schema.getEmitRegion()->front();
    SmallVector<Type> sourceLeaves;
    appendProductLeafTypes(schema.getEmitSources().front().getType(), sourceLeaves);
    auto sourceType = cast<RankedTensorType>(sourceLeaves.front());
    int64_t member = cast<TensorShapeAttr>(sourceType.getEncoding()).getDimensions()[axis];
    for (Type type : emit.getTerminator()->getOperandTypes()) {
      SmallVector<Type> leaves; appendProductLeafTypes(type, leaves);
      for (Type leaf : leaves) {
        auto tensor = cast<RankedTensorType>(leaf);
        auto axes = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions().asArrayRef();
        auto found = llvm::find(axes, member);
        if (found == axes.end()) return operation->emitError("CPU region emit has no source member axis");
        outputAxes.push_back(found - axes.begin());
      }
    }
  }
  Operation *target;
  if (scan)
    target = builder.create<cpu::RegionScanOp>(loc,
        TypeRange(results).take_front(outputFields.size()),
        TypeRange(results).drop_front(outputFields.size()),
        sources, identityValues, stateValues, captures,
        ValueRange(*outputs).take_front(outputFields.size()),
        ValueRange(*outputs).drop_front(outputFields.size()), axis,
        fieldPaths(schema.getIdentities().getTypes()), fieldPaths(schema.getInitialStates().getTypes()),
        builder.getDenseI64ArrayAttr(outputAxes), IntegerAttr());
  else
    target = builder.create<cpu::RegionFoldOp>(loc, results, sources, identityValues, captures,
        *outputs, axis, fieldPaths(schema.getIdentities().getTypes()), IntegerAttr());
  auto targetSchema = cast<cpu::RegionOpInterface>(target);
  if (failed(helper(*schema.getSummarizeRegion(), targetSchema.getSummarize(), captures))) return failure();
  if (failed(helper(schema.getCombine(), targetSchema.getCombine()))) return failure();
  if (scan) {
    if (failed(helper(*schema.getApplyRegion(), *targetSchema.getApplyRegion()))) return failure();
    if (failed(helper(*schema.getEmitRegion(), *targetSchema.getEmitRegion(), captures))) return failure();
  }
  bindValues(operation->getResults(), target->getResults(), loc);
  return success();
}

LogicalResult Construction::scan(ScanOp operation) {
  SmallVector<Type> results;
  appendProductLeafTypes(operation.getResultTypes(), results);
  for (Type &type : results) type = valueType(type);
  auto outputs = emptyResults(operation.getResultTypes(), operation.getLoc());
  if (failed(outputs)) return failure();
  auto target = builder.create<cpu::ScanOp>(operation.getLoc(), results,
      flattened(operation.getSources()), flattened(operation.getIdentities()),
      flattened(operation.getCaptures()), *outputs,
      operation.getAxis(), operation.getInclusive(), operation.getReverse());
  if (failed(helper(operation.getCombine(), target.getCombine(), flattened(operation.getCaptures())))) return failure();
  bindValues(operation.getResults(), target.getResults(), operation.getLoc());
  return success();
}

LogicalResult Construction::reduce(ReduceOp operation) {
  auto sources = flattened(operation.getSources());
  auto identities = flattened(operation.getIdentities());
  auto captures = flattened(operation.getCaptures());
  Location loc = operation.getLoc();
  SmallVector<Type> results;
  appendProductLeafTypes(operation.getResultTypes(), results);
  for (Type &type : results) type = valueType(type);
  auto outputs = emptyResults(operation.getResultTypes(), loc);
  if (failed(outputs)) return failure();
  if (results.size() != identities.size() || sources.size() != identities.size())
    return operation.emitError("CPU reduction source, identity and output leaves differ");
  SmallVector<int64_t> axes;
  for (Attribute axis : operation.getAxes())
    axes.push_back(cast<IntegerAttr>(axis).getInt());
  auto reduction = builder.create<cpu::SliceReduceOp>(loc, results, sources, identities,
      captures, *outputs, axes,
      cpu::ReductionOrderAttr::get(builder.getContext(), true, true));
  if (failed(helper(operation.getCombine(), reduction.getCombine(), captures))) return failure();
  bindValues(operation.getResults(), reduction.getResults(), loc);
  return success();
}

LogicalResult Construction::lower(HistogramOp op) {
  Location loc = op.getLoc();
  auto type = cast<RankedTensorType>(op.getResult().getType());
  Value bins = values.lookup(op.getBins());
  Value output = emptyTensor(type, {bins}, loc);
  auto integer = dyn_cast<IntegerType>(getElementTypeOrSelf(op.getValues().getType()));
  auto result = builder.create<cpu::HistogramOp>(loc, tensorType(type), values.lookup(op.getValues()), values.lookup(op.getValid()), output,
      integer && integer.isUnsigned());
  values.map(op.getResult(), result.getResult());
  bindDimensions(type, result.getResult(), loc);
  return success();
}

} // namespace intent::kir_to_cpu
