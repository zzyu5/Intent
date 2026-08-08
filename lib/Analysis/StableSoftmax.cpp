#include "Intent/Analysis/StableSoftmax.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>

using namespace mlir;

namespace intent {
namespace {

StringRef getStringAttribute(Operation *operation, StringRef name) {
  auto attribute = operation->getAttrOfType<StringAttr>(name);
  return attribute ? attribute.getValue() : StringRef();
}

bool hasAxisZero(Operation *operation) {
  auto axes = operation->getAttrOfType<ArrayAttr>("intent.axes");
  auto axis = axes && axes.size() == 1 ? dyn_cast<IntegerAttr>(axes[0])
                                      : IntegerAttr();
  return axis && axis.getInt() == 0;
}

bool hasReductionIdentity(Operation *reduction, StringRef identity) {
  if (!reduction || reduction->getNumOperands() != 2)
    return false;
  Operation *constant = reduction->getOperand(1).getDefiningOp();
  auto value =
      constant ? constant->getAttrOfType<FloatAttr>("intent.value") : FloatAttr();
  if (!constant || constant->getName().getStringRef() != "intent.constant" ||
      !value)
    return false;
  double literal = value.getValueAsDouble();
  if (identity == "negative_infinity")
    return std::isinf(literal) && literal < 0.0;
  if (identity == "zero")
    return literal == 0.0;
  return false;
}

bool isSourceDomain(Operation *domain, func::FuncOp entry, int64_t axis) {
  if (!domain || domain->getName().getStringRef() != "intent.domain" ||
      domain->getNumOperands() != 2 || domain->getNumResults() != 1)
    return false;
  Operation *start = domain->getOperand(0).getDefiningOp();
  auto startValue =
      start ? start->getAttrOfType<IntegerAttr>("intent.value") : IntegerAttr();
  Operation *stop = domain->getOperand(1).getDefiningOp();
  auto stopAxis =
      stop ? stop->getAttrOfType<IntegerAttr>("intent.axis") : IntegerAttr();
  return start && start->getName().getStringRef() == "intent.constant" &&
         startValue && startValue.getInt() == 0 && stop &&
         stop->getName().getStringRef() == "intent.dim" && stopAxis &&
         stopAxis.getInt() == axis && stop->getNumOperands() == 1 &&
         stop->getOperand(0) == entry.getArgument(0);
}

bool isSoftmaxView(func::FuncOp entry, unsigned index, StringRef access,
                   ArrayAttr &canonicalShape) {
  auto parameters = entry->getAttrOfType<ArrayAttr>("intent.parameters");
  if (!parameters || index >= parameters.size())
    return false;
  auto metadata = dyn_cast<DictionaryAttr>(parameters[index]);
  auto viewType = dyn_cast<ViewType>(entry.getArgument(index).getType());
  auto tensorType = viewType ? dyn_cast<RankedTensorType>(viewType.getTensor())
                             : RankedTensorType();
  if (!metadata || !viewType || viewType.getAccess() != access || !tensorType ||
      tensorType.getRank() != 2 || !tensorType.getElementType().isF32() ||
      !tensorType.isDynamicDim(0) || !tensorType.isDynamicDim(1))
    return false;

  auto kind = metadata.getAs<StringAttr>("kind");
  auto viewKind = metadata.getAs<StringAttr>("view_kind");
  auto shape = metadata.getAs<ArrayAttr>("shape");
  auto constraints = metadata.getAs<DictionaryAttr>("constraints");
  if (!kind || kind.getValue() != "view" || !viewKind ||
      viewKind.getValue() != access || !shape || shape.size() != 2 ||
      !constraints)
    return false;
  for (Attribute dimension : shape) {
    auto symbol = dyn_cast<StringAttr>(dimension);
    if (!symbol || symbol.getValue().empty())
      return false;
  }
  if (!canonicalShape)
    canonicalShape = shape;
  else if (canonicalShape != shape)
    return false;

  auto strides = constraints.getAs<ArrayAttr>("strides");
  auto layout = constraints.getAs<StringAttr>("layout");
  auto noalias = constraints.getAs<BoolAttr>("noalias");
  return strides && strides.size() == 2 && isa<UnitAttr>(strides[0]) &&
         isa<IntegerAttr>(strides[1]) &&
         cast<IntegerAttr>(strides[1]).getInt() == 1 && layout &&
         layout.getValue() == "row_major" && noalias && noalias.getValue() &&
         isa<UnitAttr>(constraints.get("alignment")) &&
         isa<UnitAttr>(constraints.get("alias"));
}

bool hasIndexRelation(Operation *operation, BlockArgument row, Value columns,
                      unsigned rowOperand, unsigned columnOperand) {
  auto relation = operation->getAttrOfType<ArrayAttr>("intent.index");
  if (!relation || relation.size() != 2 ||
      rowOperand >= operation->getNumOperands() ||
      columnOperand >= operation->getNumOperands())
    return false;
  auto rowTerm = dyn_cast<DictionaryAttr>(relation[0]);
  auto columnTerm = dyn_cast<DictionaryAttr>(relation[1]);
  if (!rowTerm || !columnTerm)
    return false;
  auto rowKind = rowTerm.getAs<StringAttr>("kind");
  auto columnKind = columnTerm.getAs<StringAttr>("kind");
  auto rowPositions = rowTerm.getAs<ArrayAttr>("operands");
  auto columnPositions = columnTerm.getAs<ArrayAttr>("operands");
  if (!rowKind || rowKind.getValue() != "value_index" || !columnKind ||
      columnKind.getValue() != "region_index" || !rowPositions ||
      rowPositions.size() != 1 || !columnPositions ||
      columnPositions.size() != 1)
    return false;
  auto encodedRow = dyn_cast<IntegerAttr>(rowPositions[0]);
  auto encodedColumn = dyn_cast<IntegerAttr>(columnPositions[0]);
  return encodedRow && encodedRow.getInt() == rowOperand && encodedColumn &&
         encodedColumn.getInt() == columnOperand &&
         operation->getOperand(rowOperand) == row &&
         operation->getOperand(columnOperand) == columns;
}

Operation *findUnique(ArrayRef<Operation *> operations, StringRef name,
                      StringRef attribute = {}, StringRef value = {}) {
  Operation *result = nullptr;
  for (Operation *operation : operations) {
    if (operation->getName().getStringRef() != name ||
        (!attribute.empty() && getStringAttribute(operation, attribute) != value))
      continue;
    if (result)
      return nullptr;
    result = operation;
  }
  return result;
}

} // namespace

int64_t getIntentNodeID(Operation *operation) {
  return operation->getAttrOfType<IntegerAttr>("intent.node").getInt();
}

FailureOr<StableSoftmaxMatch> matchStableSoftmax(ModuleOp module) {
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  if (functions.size() != 1) {
    module.emitError("stable softmax realization requires exactly one function");
    return failure();
  }
  func::FuncOp entry = functions.front();
  auto functionKind = entry->getAttrOfType<StringAttr>("intent.kind");
  if (!functionKind || functionKind.getValue() != "kernel" ||
      entry.getNumArguments() != 2) {
    entry.emitOpError("stable softmax requires one kernel with input/output views");
    return failure();
  }

  ArrayAttr abiShape;
  if (!isSoftmaxView(entry, 0, "in", abiShape) ||
      !isSoftmaxView(entry, 1, "out", abiShape)) {
    entry.emitOpError(
        "stable softmax requires equal symbolic rank-two f32 row-major noalias views");
    return failure();
  }
  auto parameterNodes = entry->getAttrOfType<ArrayAttr>("intent.parameter_nodes");
  if (!parameterNodes || parameterNodes.size() != 2 ||
      !isa<IntegerAttr>(parameterNodes[0]) ||
      !isa<IntegerAttr>(parameterNodes[1])) {
    entry.emitOpError("stable softmax ABI lacks stable value IDs");
    return failure();
  }

  Block &topLevel = entry.getBody().front();
  SmallVector<Operation *> rowLoops;
  SmallVector<Operation *> domains;
  unsigned topLevelConstants = 0;
  unsigned dimensions = 0;
  unsigned returns = 0;
  for (Operation &operation : topLevel) {
    StringRef name = operation.getName().getStringRef();
    if (name == "intent.parallel")
      rowLoops.push_back(&operation);
    else if (name == "intent.domain")
      domains.push_back(&operation);
    else if (name == "intent.constant")
      ++topLevelConstants;
    else if (name == "intent.dim")
      ++dimensions;
    else if (name == "intent.return")
      ++returns;
    else {
      operation.emitOpError("is not part of the stable softmax top-level workset");
      return failure();
    }
  }
  if (rowLoops.size() != 1 || domains.size() != 2 || topLevelConstants != 2 ||
      dimensions != 2 || returns != 1) {
    entry.emitOpError("stable softmax top level must contain exactly M/N domains and one row loop");
    return failure();
  }
  Operation *rowLoop = rowLoops.front();
  if (rowLoop->getNumOperands() != 1 || rowLoop->getNumRegions() != 1 ||
      !llvm::hasSingleElement(rowLoop->getRegion(0))) {
    rowLoop->emitOpError("stable softmax row loop has an incompatible region schema");
    return failure();
  }
  Operation *rowDomain = rowLoop->getOperand(0).getDefiningOp();
  Operation *columnDomain = domains[0] == rowDomain ? domains[1] : domains[0];
  if (!llvm::is_contained(domains, rowDomain) ||
      !isSourceDomain(rowDomain, entry, 0) ||
      !isSourceDomain(columnDomain, entry, 1)) {
    rowLoop->emitOpError("row/column domains do not preserve the source M/N workset");
    return failure();
  }

  Block &body = rowLoop->getRegion(0).front();
  if (body.getNumArguments() != 1) {
    rowLoop->emitOpError("stable softmax row loop requires one logical index");
    return failure();
  }
  SmallVector<Operation *> operations;
  SmallVector<Operation *> broadcasts;
  unsigned loads = 0;
  unsigned stores = 0;
  unsigned constants = 0;
  unsigned reductions = 0;
  unsigned binaries = 0;
  unsigned unary = 0;
  unsigned yields = 0;
  for (Operation &operation : body) {
    StringRef name = operation.getName().getStringRef();
    if (name != "intent.view_load" && name != "intent.constant" &&
        name != "intent.reduce" && name != "intent.broadcast" &&
        name != "intent.binary" && name != "intent.unary" &&
        name != "intent.view_store" && name != "intent.yield") {
      operation.emitOpError("is not part of the stable softmax source algorithm");
      return failure();
    }
    operations.push_back(&operation);
    if (name == "intent.view_load")
      ++loads;
    else if (name == "intent.view_store")
      ++stores;
    else if (name == "intent.constant")
      ++constants;
    else if (name == "intent.reduce")
      ++reductions;
    else if (name == "intent.broadcast")
      broadcasts.push_back(&operation);
    else if (name == "intent.binary")
      ++binaries;
    else if (name == "intent.unary")
      ++unary;
    else if (name == "intent.yield")
      ++yields;
  }
  if (loads != 1 || stores != 1 || constants != 2 || reductions != 2 ||
      broadcasts.size() != 2 || binaries != 2 || unary != 1 || yields != 1) {
    rowLoop->emitOpError("body contains extra or missing stable softmax operations");
    return failure();
  }

  Operation *load = findUnique(operations, "intent.view_load");
  Operation *store = findUnique(operations, "intent.view_store");
  Operation *reduceMax =
      findUnique(operations, "intent.reduce", "intent.combine", "maximum");
  Operation *reduceSum =
      findUnique(operations, "intent.reduce", "intent.combine", "add");
  Operation *subtract =
      findUnique(operations, "intent.binary", "intent.operator", "subtract");
  Operation *divide = findUnique(operations, "intent.binary", "intent.operator",
                                 "true_divide");
  Operation *exponential =
      findUnique(operations, "intent.unary", "intent.operator", "exp");
  if (!load || !store || !reduceMax || !reduceSum || !subtract || !divide ||
      !exponential || broadcasts.size() != 2 || !hasAxisZero(reduceMax) ||
      !hasAxisZero(reduceSum) ||
      !hasReductionIdentity(reduceMax, "negative_infinity") ||
      !hasReductionIdentity(reduceSum, "zero")) {
    rowLoop->emitOpError("body is not the canonical stable softmax operation set");
    return failure();
  }

  Operation *maxBroadcast = nullptr;
  Operation *sumBroadcast = nullptr;
  for (Operation *broadcast : broadcasts) {
    if (broadcast->getNumOperands() != 1 || broadcast->getNumResults() != 1)
      continue;
    if (broadcast->getOperand(0) == reduceMax->getResult(0))
      maxBroadcast = broadcast;
    if (broadcast->getOperand(0) == reduceSum->getResult(0))
      sumBroadcast = broadcast;
  }
  if (!maxBroadcast || !sumBroadcast || load->getNumResults() != 1 ||
      reduceMax->getOperand(0) != load->getResult(0) ||
      subtract->getOperand(0) != load->getResult(0) ||
      subtract->getOperand(1) != maxBroadcast->getResult(0) ||
      exponential->getOperand(0) != subtract->getResult(0) ||
      reduceSum->getOperand(0) != exponential->getResult(0) ||
      divide->getOperand(0) != exponential->getResult(0) ||
      divide->getOperand(1) != sumBroadcast->getResult(0)) {
    rowLoop->emitOpError("def-use chain is not stable softmax");
    return failure();
  }

  auto valueIndex = store->getAttrOfType<IntegerAttr>("intent.value_operand_index");
  if (!valueIndex || valueIndex.getInt() != 1 ||
      store->getNumOperands() != 4 || load->getNumOperands() != 3 ||
      store->getOperand(1) != divide->getResult(0) ||
      load->getOperand(0) != entry.getArgument(0) ||
      store->getOperand(0) != entry.getArgument(1) ||
      !hasIndexRelation(load, body.getArgument(0), columnDomain->getResult(0),
                        1, 2) ||
      !hasIndexRelation(store, body.getArgument(0), columnDomain->getResult(0),
                        2, 3)) {
    rowLoop->emitOpError("memory flow/index relation is not row-major softmax");
    return failure();
  }

  return StableSoftmaxMatch{
      entry,
      columnDomain,
      rowLoop,
      load,
      reduceMax,
      maxBroadcast,
      subtract,
      exponential,
      reduceSum,
      sumBroadcast,
      divide,
      store,
      cast<IntegerAttr>(parameterNodes[0]).getInt(),
      cast<IntegerAttr>(parameterNodes[1]).getInt(),
      cast<StringAttr>(abiShape[0]).getValue().str(),
      cast<StringAttr>(abiShape[1]).getValue().str(),
  };
}

} // namespace intent
