#include "Intent/Target/Triton/Realization/Realize.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "Intent/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace intent::triton {
namespace {

struct RowwiseKernel {
  func::FuncOp entry;
  Operation *rowDomain;
  Operation *columnDomain;
  Operation *parallel;
  Block *body;
  int64_t inputValueID;
  int64_t outputValueID;
};

IntegerAttr i64(OpBuilder &builder, int64_t value) {
  return builder.getI64IntegerAttr(value);
}

StringAttr string(OpBuilder &builder, StringRef value) {
  return builder.getStringAttr(value);
}

FailureOr<int64_t> nodeID(Operation *operation) {
  auto node = operation->getAttrOfType<IntegerAttr>("intent.node");
  if (!node) {
    operation->emitOpError("requires intent.node for target realization");
    return failure();
  }
  return node.getInt();
}

bool isSourceDomain(Operation *domain, Value source, int64_t axis) {
  if (!domain || domain->getName().getStringRef() != "intent.domain" ||
      domain->getNumOperands() != 2 || domain->getNumResults() != 1)
    return false;
  Operation *start = domain->getOperand(0).getDefiningOp();
  Operation *stop = domain->getOperand(1).getDefiningOp();
  auto startValue =
      start ? start->getAttrOfType<IntegerAttr>("intent.value") : IntegerAttr();
  auto stopAxis =
      stop ? stop->getAttrOfType<IntegerAttr>("intent.axis") : IntegerAttr();
  return start && start->getName().getStringRef() == "intent.constant" &&
         startValue && startValue.getInt() == 0 && stop &&
         stop->getName().getStringRef() == "intent.dim" && stopAxis &&
         stopAxis.getInt() == axis && stop->getNumOperands() == 1 &&
         stop->getOperand(0) == source;
}

bool hasRowColumnIndex(Operation *operation, BlockArgument row, Value columns,
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
  auto encodedRow = rowPositions && rowPositions.size() == 1
                        ? dyn_cast<IntegerAttr>(rowPositions[0])
                        : IntegerAttr();
  auto encodedColumn = columnPositions && columnPositions.size() == 1
                           ? dyn_cast<IntegerAttr>(columnPositions[0])
                           : IntegerAttr();
  return rowKind && rowKind.getValue() == "value_index" && columnKind &&
         columnKind.getValue() == "region_index" && encodedRow &&
         encodedRow.getInt() == rowOperand && encodedColumn &&
         encodedColumn.getInt() == columnOperand &&
         operation->getOperand(rowOperand) == row &&
         operation->getOperand(columnOperand) == columns;
}

bool isRankTwoF32View(func::FuncOp entry, unsigned index, StringRef access) {
  auto view = dyn_cast<intent::ViewType>(entry.getArgument(index).getType());
  auto tensor = view ? dyn_cast<RankedTensorType>(view.getTensor())
                     : RankedTensorType();
  if (!view || view.getAccess() != access || !tensor || tensor.getRank() != 2 ||
      !tensor.getElementType().isF32())
    return false;
  auto parameters = entry->getAttrOfType<ArrayAttr>("intent.parameters");
  auto metadata = parameters && index < parameters.size()
                      ? dyn_cast<DictionaryAttr>(parameters[index])
                      : DictionaryAttr();
  auto constraints = metadata ? metadata.getAs<DictionaryAttr>("constraints")
                              : DictionaryAttr();
  auto strides = constraints ? constraints.getAs<ArrayAttr>("strides")
                             : ArrayAttr();
  auto layout = constraints ? constraints.getAs<StringAttr>("layout")
                            : StringAttr();
  auto noalias = constraints ? constraints.getAs<BoolAttr>("noalias")
                             : BoolAttr();
  return strides && strides.size() == 2 && isa<UnitAttr>(strides[0]) &&
         isa<IntegerAttr>(strides[1]) &&
         cast<IntegerAttr>(strides[1]).getInt() == 1 && layout &&
         layout.getValue() == "row_major" && noalias && noalias.getValue();
}

bool hasAxisZero(Operation *operation) {
  auto axes = operation->getAttrOfType<ArrayAttr>("intent.axes");
  auto axis = axes && axes.size() == 1 ? dyn_cast<IntegerAttr>(axes[0])
                                      : IntegerAttr();
  return axis && axis.getInt() == 0;
}

StringRef mapReduction(Operation *operation) {
  auto combine = operation->getAttrOfType<StringAttr>("intent.combine");
  if (!combine || !hasAxisZero(operation))
    return {};
  if (combine.getValue() == "maximum")
    return "tl.max";
  if (combine.getValue() == "add")
    return "tl.sum";
  return {};
}

StringRef mapPointwise(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  if (name == "intent.broadcast")
    return "alias";
  auto logical = operation->getAttrOfType<StringAttr>("intent.operator");
  if (!logical)
    return {};
  if (name == "intent.unary") {
    if (logical.getValue() == "exp")
      return "tl.exp";
    if (logical.getValue() == "exp2")
      return "tl.exp2";
    if (logical.getValue() == "log")
      return "tl.log";
    if (logical.getValue() == "rsqrt")
      return "tl.rsqrt";
    if (logical.getValue() == "negate")
      return "python_negate";
    return {};
  }
  if (name == "intent.binary") {
    if (logical.getValue() == "add")
      return "python_add";
    if (logical.getValue() == "subtract")
      return "python_subtract";
    if (logical.getValue() == "multiply")
      return "python_multiply";
    if (logical.getValue() == "true_divide")
      return "python_true_divide";
  }
  return {};
}

bool preservesMaskedLanes(Value loaded) {
  Operation *maximum = nullptr;
  SmallVector<Operation *> subtracts;
  for (Operation *user : loaded.getUsers()) {
    StringRef name = user->getName().getStringRef();
    if (name == "intent.reduce" && mapReduction(user) == "tl.max") {
      if (maximum)
        return false;
      maximum = user;
      continue;
    }
    auto logical = user->getAttrOfType<StringAttr>("intent.operator");
    if (name == "intent.binary" && logical &&
        logical.getValue() == "subtract" && user->getOperand(0) == loaded) {
      subtracts.push_back(user);
      continue;
    }
    return false;
  }
  if (!maximum || subtracts.empty())
    return false;
  for (Operation *subtract : subtracts) {
    Operation *broadcast = subtract->getOperand(1).getDefiningOp();
    if (!broadcast || broadcast->getName().getStringRef() != "intent.broadcast" ||
        broadcast->getNumOperands() != 1 ||
        broadcast->getOperand(0) != maximum->getResult(0))
      return false;
  }
  return true;
}

FailureOr<RowwiseKernel> analyzeRowwiseKernel(ModuleOp module) {
  SmallVector<func::FuncOp> kernels;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto kind = function->getAttrOfType<StringAttr>("intent.kind");
    if (kind && kind.getValue() == "kernel")
      kernels.push_back(function);
  }
  if (kernels.size() != 1) {
    module.emitError("Triton rowwise realization requires exactly one kernel entry");
    return failure();
  }
  func::FuncOp entry = kernels.front();
  if (entry.getNumArguments() != 2 || !isRankTwoF32View(entry, 0, "in") ||
      !isRankTwoF32View(entry, 1, "out")) {
    entry.emitOpError(
        "rowwise Triton realization requires rank-two f32 input/output views");
    return failure();
  }
  auto parameterNodes = entry->getAttrOfType<ArrayAttr>("intent.parameter_nodes");
  if (!parameterNodes || parameterNodes.size() != 2 ||
      !isa<IntegerAttr>(parameterNodes[0]) || !isa<IntegerAttr>(parameterNodes[1])) {
    entry.emitOpError("rowwise realization requires stable ABI value IDs");
    return failure();
  }

  Operation *parallel = nullptr;
  for (Operation &operation : entry.getBody().front()) {
    if (operation.getName().getStringRef() != "intent.parallel")
      continue;
    if (parallel) {
      operation.emitOpError("rowwise realization supports one parallel region");
      return failure();
    }
    parallel = &operation;
  }
  if (!parallel || parallel->getNumOperands() != 1 ||
      parallel->getNumRegions() != 1 ||
      !llvm::hasSingleElement(parallel->getRegion(0))) {
    entry.emitOpError("rowwise realization requires one single-block parallel region");
    return failure();
  }
  Operation *rowDomain = parallel->getOperand(0).getDefiningOp();
  Operation *columnDomain = nullptr;
  for (Operation &operation : entry.getBody().front()) {
    if (&operation != rowDomain &&
        isSourceDomain(&operation, entry.getArgument(0), 1)) {
      if (columnDomain) {
        operation.emitOpError("rowwise realization found multiple column domains");
        return failure();
      }
      columnDomain = &operation;
    }
  }
  if (!isSourceDomain(rowDomain, entry.getArgument(0), 0) || !columnDomain) {
    parallel->emitOpError("requires source row and column domains");
    return failure();
  }
  Block &body = parallel->getRegion(0).front();
  if (body.getNumArguments() != 1 || parallel->getNumResults() != 0) {
    parallel->emitOpError("rowwise parallel region cannot carry state");
    return failure();
  }

  bool sawLoad = false;
  bool sawStore = false;
  for (Operation &operation : body) {
    StringRef name = operation.getName().getStringRef();
    if (name == "intent.constant" || name == "intent.yield")
      continue;
    if (name == "intent.view_load") {
      if (operation.getNumOperands() != 3 || operation.getNumResults() != 1 ||
          operation.getOperand(0) != entry.getArgument(0) ||
          !hasRowColumnIndex(&operation, body.getArgument(0),
                             columnDomain->getResult(0), 1, 2) ||
          !preservesMaskedLanes(operation.getResult(0))) {
        operation.emitOpError(
            "cannot preserve this load under the rowwise masked-lane mechanism");
        return failure();
      }
      sawLoad = true;
      continue;
    }
    if (name == "intent.view_store") {
      auto valueIndex =
          operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
      if (!valueIndex || valueIndex.getInt() != 1 ||
          operation.getNumOperands() != 4 ||
          operation.getOperand(0) != entry.getArgument(1) ||
          !hasRowColumnIndex(&operation, body.getArgument(0),
                             columnDomain->getResult(0), 2, 3)) {
        operation.emitOpError("cannot realize this rowwise store");
        return failure();
      }
      sawStore = true;
      continue;
    }
    if (name == "intent.reduce") {
      if (mapReduction(&operation).empty()) {
        operation.emitOpError("has no Triton rowwise reduction realization");
        return failure();
      }
      continue;
    }
    if (name == "intent.broadcast" || name == "intent.unary" ||
        name == "intent.binary") {
      if (mapPointwise(&operation).empty()) {
        operation.emitOpError("has no Triton pointwise realization");
        return failure();
      }
      continue;
    }
    operation.emitOpError("has no Triton rowwise realization");
    return failure();
  }
  if (!sawLoad || !sawStore) {
    parallel->emitOpError("rowwise realization requires observable load/store work");
    return failure();
  }

  return RowwiseKernel{
      entry,
      rowDomain,
      columnDomain,
      parallel,
      &body,
      cast<IntegerAttr>(parameterNodes[0]).getInt(),
      cast<IntegerAttr>(parameterNodes[1]).getInt(),
  };
}

} // namespace

LogicalResult realizeKernel(ModuleOp module, const TargetOptions &target) {
  if (target.architecture.empty() || target.device < 0 || target.warpSize <= 0)
    return module.emitError("Triton realization requires a complete target");
  if (!module.getOps<intent::plan::RealizationOp>().empty() ||
      !module.getOps<intent::plan::SearchSpaceOp>().empty())
    return module.emitError("module already contains realization state");
  if (failed(verifyKernelModule(module)))
    return failure();
  FailureOr<RowwiseKernel> analyzed = analyzeRowwiseKernel(module);
  if (failed(analyzed))
    return failure();
  RowwiseKernel &kernel = *analyzed;

  FailureOr<int64_t> rowDomainNode = nodeID(kernel.rowDomain);
  FailureOr<int64_t> columnDomainNode = nodeID(kernel.columnDomain);
  FailureOr<int64_t> loopNode = nodeID(kernel.parallel);
  if (failed(rowDomainNode) || failed(columnDomainNode) || failed(loopNode))
    return failure();

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto realization = builder.create<intent::plan::RealizationOp>(
      kernel.entry.getLoc(),
      FlatSymbolRefAttr::get(module.getContext(), kernel.entry.getName()),
      string(builder, "triton"));
  Block &body = realization.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  builder.create<plan::TargetOp>(
      kernel.entry.getLoc(), string(builder, target.architecture),
      i64(builder, target.device), i64(builder, target.warpSize));
  builder.create<plan::AxisOp>(
      kernel.rowDomain->getLoc(), i64(builder, *rowDomainNode), i64(builder, 0),
      string(builder, "row"), string(builder, "one"));
  builder.create<plan::AxisOp>(
      kernel.columnDomain->getLoc(), i64(builder, *columnDomainNode),
      i64(builder, 1), string(builder, "column"),
      string(builder, "next_power_of_two"));
  builder.create<plan::ProgramOp>(
      kernel.parallel->getLoc(), i64(builder, *loopNode), i64(builder, 0),
      string(builder, "persistent"), string(builder, "grid_stride"));
  for (int64_t value : {kernel.inputValueID, kernel.outputValueID}) {
    builder.create<plan::StorageOp>(kernel.entry.getLoc(), i64(builder, value),
                                    string(builder, "global"));
    builder.create<plan::LayoutOp>(
        kernel.entry.getLoc(), i64(builder, value), string(builder, "row_major"),
        builder.getDenseI64ArrayAttr({0, 1}));
  }

  for (Operation &operation : *kernel.body) {
    FailureOr<int64_t> node = nodeID(&operation);
    if (failed(node))
      return failure();
    StringRef name = operation.getName().getStringRef();
    if (name == "intent.reduce") {
      builder.create<plan::ReductionOp>(
          operation.getLoc(), i64(builder, *node),
          string(builder, mapReduction(&operation)), i64(builder, 0));
    } else if (name == "intent.broadcast" || name == "intent.unary" ||
               name == "intent.binary") {
      builder.create<plan::PointwiseOp>(
          operation.getLoc(), i64(builder, *node),
          string(builder, mapPointwise(&operation)));
    } else if (name == "intent.view_load") {
      builder.create<plan::BoundaryOp>(
          operation.getLoc(), i64(builder, *node), i64(builder, *columnDomainNode),
          string(builder, "index_lt_extent"),
          string(builder, "negative_infinity"), string(builder, "predicate"));
    }
  }
  builder.create<plan::PipelineOp>(
      kernel.parallel->getLoc(), i64(builder, *loopNode), i64(builder, 2),
      i64(builder, 4), i64(builder, 200000), builder.getBoolAttr(false),
      builder.getBoolAttr(false));
  builder.create<plan::LaunchOp>(
      kernel.parallel->getLoc(), i64(builder, *loopNode),
      string(builder, "persistent_occupancy"), i64(builder, 8));
  builder.create<intent::plan::YieldOp>(kernel.entry.getLoc());
  return plan::verifyTritonRealization(realization);
}

} // namespace intent::triton
