#include "Intent/Dialect/Plan/IR/PlanOps.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <cmath>
#include <initializer_list>

using namespace mlir;
using namespace intent::plan;

namespace {

LogicalResult requireOneOf(Operation *operation, StringRef attribute,
                           StringRef value,
                           std::initializer_list<StringRef> allowed) {
  for (StringRef candidate : allowed)
    if (value == candidate)
      return success();
  return operation->emitOpError()
         << "attribute '" << attribute << "' has unsupported value '" << value
         << "'";
}

StringRef getStringAttribute(Operation *operation, StringRef name) {
  auto attribute = operation->getAttrOfType<StringAttr>(name);
  return attribute ? attribute.getValue() : StringRef();
}

LogicalResult verifySoftmaxViewABI(func::FuncOp entry, unsigned index,
                                   DictionaryAttr metadata,
                                   ArrayAttr &canonicalShape) {
  auto viewType = dyn_cast<intent::ViewType>(entry.getArgument(index).getType());
  auto tensorType = viewType ? dyn_cast<RankedTensorType>(viewType.getTensor())
                             : RankedTensorType();
  auto kind = metadata.getAs<StringAttr>("kind");
  auto viewKind = metadata.getAs<StringAttr>("view_kind");
  auto shape = metadata.getAs<ArrayAttr>("shape");
  auto constraints = metadata.getAs<DictionaryAttr>("constraints");
  if (!viewType || !tensorType || tensorType.getRank() != 2 ||
      !tensorType.getElementType().isF32() || !tensorType.isDynamicDim(0) ||
      !tensorType.isDynamicDim(1) || !kind || kind.getValue() != "view" ||
      !viewKind || viewKind.getValue() != viewType.getAccess() || !shape ||
      shape.size() != 2 || !constraints)
    return entry.emitOpError(
        "initial softmax Plan requires two symbolic rank-two f32 view parameters");
  for (Attribute dimension : shape) {
    auto symbol = dyn_cast<StringAttr>(dimension);
    if (!symbol || symbol.getValue().empty())
      return entry.emitOpError(
          "softmax ABI shape dimensions must be non-empty symbolic names");
  }
  if (!canonicalShape)
    canonicalShape = shape;
  else if (canonicalShape != shape)
    return entry.emitOpError("softmax input/output symbolic shapes must match");

  auto strides = constraints.getAs<ArrayAttr>("strides");
  auto layout = constraints.getAs<StringAttr>("layout");
  auto noalias = constraints.getAs<BoolAttr>("noalias");
  if (!strides || strides.size() != 2 || !isa<UnitAttr>(strides[0]) ||
      !isa<IntegerAttr>(strides[1]) ||
      cast<IntegerAttr>(strides[1]).getInt() != 1 || !layout ||
      layout.getValue() != "row_major" || !noalias || !noalias.getValue() ||
      !isa<UnitAttr>(constraints.get("alignment")) ||
      !isa<UnitAttr>(constraints.get("alias")))
    return entry.emitOpError(
        "softmax ABI requires strides=[dynamic, 1], row_major, noalias views");
  return success();
}

LogicalResult verifySoftmaxDomain(Operation *domain, func::FuncOp entry,
                                  int64_t axis) {
  if (!domain || domain->getName().getStringRef() != "intent.domain" ||
      domain->getNumOperands() != 2 || domain->getNumResults() != 1)
    return failure();
  Operation *start = domain->getOperand(0).getDefiningOp();
  auto startValue =
      start ? start->getAttrOfType<IntegerAttr>("intent.value") : IntegerAttr();
  Operation *stop = domain->getOperand(1).getDefiningOp();
  auto stopAxis =
      stop ? stop->getAttrOfType<IntegerAttr>("intent.axis") : IntegerAttr();
  if (!start || start->getName().getStringRef() != "intent.constant" ||
      !startValue || startValue.getInt() != 0 || !stop ||
      stop->getName().getStringRef() != "intent.dim" || !stopAxis ||
      stopAxis.getInt() != axis || stop->getNumOperands() != 1)
    return failure();
  return success(stop->getOperand(0) == entry.getArgument(0));
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

} // namespace

LogicalResult PlanOp::verify() {
  if (getBackend() != "triton")
    return emitOpError("currently supports only the Triton backend");
  if (getDeviceAttr().getInt() < 0 || getWarpSize() <= 0 || getArchitecture().empty())
    return emitOpError("requires complete non-negative target information");

  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError("must be nested directly in an MLIR module");
  auto entry = module.lookupSymbol<func::FuncOp>(getEntry());
  if (!entry)
    return emitOpError("references a missing kernel entry");
  auto kind = entry->getAttrOfType<StringAttr>("intent.kind");
  if (!kind || kind.getValue() != "kernel")
    return emitOpError("entry must reference an Intent kernel function");

  DenseMap<int64_t, Operation *> nodes;
  DenseSet<int64_t> values;
  DenseMap<int64_t, DictionaryAttr> parameterMetadata;
  auto parameterNodes = entry->getAttrOfType<ArrayAttr>("intent.parameter_nodes");
  auto parameters = entry->getAttrOfType<ArrayAttr>("intent.parameters");
  if (!parameterNodes || !parameters ||
      parameterNodes.size() != entry.getNumArguments() ||
      parameters.size() != entry.getNumArguments())
    return emitOpError("kernel entry lacks complete intent.parameter_nodes metadata");
  if (entry.getNumArguments() != 2)
    return emitOpError("initial softmax Plan requires exactly two ABI views");
  ArrayAttr abiShape;
  unsigned parameterIndex = 0;
  for (auto [nodeAttribute, metadataAttribute] :
       llvm::zip(parameterNodes, parameters)) {
    auto id = dyn_cast<IntegerAttr>(nodeAttribute);
    auto metadata = dyn_cast<DictionaryAttr>(metadataAttribute);
    if (!id || !metadata || !values.insert(id.getInt()).second)
      return emitOpError("kernel value node IDs must be unique integers");
    if (failed(verifySoftmaxViewABI(entry, parameterIndex, metadata, abiShape)))
      return failure();
    parameterMetadata[id.getInt()] = metadata;
    ++parameterIndex;
  }
  WalkResult walk = entry.walk([&](Operation *operation) -> WalkResult {
    if (operation == entry.getOperation())
      return WalkResult::advance();
    auto node = operation->getAttrOfType<IntegerAttr>("intent.node");
    auto resultNodes = operation->getAttrOfType<ArrayAttr>("intent.result_nodes");
    auto resultNames = operation->getAttrOfType<ArrayAttr>("intent.result_names");
    if (!node || !nodes.try_emplace(node.getInt(), operation).second) {
      operation->emitOpError("requires a unique integer intent.node attribute");
      return WalkResult::interrupt();
    }
    if (!resultNodes || resultNodes.size() != operation->getNumResults()) {
      operation->emitOpError("requires one intent.result_nodes ID per SSA result");
      return WalkResult::interrupt();
    }
    if (!resultNames || resultNames.size() != operation->getNumResults()) {
      operation->emitOpError("requires one intent.result_names entry per SSA result");
      return WalkResult::interrupt();
    }
    for (Attribute attribute : resultNames) {
      auto name = dyn_cast<StringAttr>(attribute);
      if (!name || name.getValue().empty()) {
        operation->emitOpError("result names must be non-empty strings");
        return WalkResult::interrupt();
      }
    }
    for (Attribute attribute : resultNodes) {
      auto id = dyn_cast<IntegerAttr>(attribute);
      if (!id || !values.insert(id.getInt()).second) {
        operation->emitOpError("result node IDs must be unique integers");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return failure();

  DenseSet<int64_t> storageValues;
  DenseSet<int64_t> layoutValues;
  DenseSet<int64_t> primitiveNodes;
  DenseMap<int64_t, ExtentOp> extentBindings;
  DenseMap<int64_t, BoundaryOp> boundaryBindings;
  DenseMap<int64_t, PrimitiveOp> primitiveBindings;
  int ownershipCount = 0;
  int pipelineCount = 0;
  int launchCount = 0;
  int64_t ownershipLoop = -1;
  int64_t launchLoop = -1;

  for (Operation &operation : getBody().front()) {
    if (isa<YieldOp>(operation))
      continue;
    if (auto extent = dyn_cast<ExtentOp>(operation)) {
      if (!nodes.count(extent.getNode()))
        return extent.emitOpError("references a missing Kernel IR node");
      if (extent.getAxisAttr().getInt() < 0 || extent.getLogical().empty())
        return extent.emitOpError("requires a non-negative axis and logical extent");
      if (failed(requireOneOf(&operation, "tile", extent.getTile(),
                              {"one", "next_power_of_two"})))
        return failure();
      if (extentBindings.count(extent.getAxis()))
        return extent.emitOpError("duplicates a physical axis extent");
      Operation *source = nodes.lookup(extent.getNode());
      if ((extent.getAxis() == 0 &&
           source->getName().getStringRef() != "intent.parallel") ||
          (extent.getAxis() == 1 &&
           source->getName().getStringRef() != "intent.domain"))
        return extent.emitOpError("references an incompatible Kernel IR node kind");
      extentBindings[extent.getAxis()] = extent;
      continue;
    }
    if (auto ownership = dyn_cast<OwnershipOp>(operation)) {
      ++ownershipCount;
      ownershipLoop = ownership.getLoopNode();
      if (!nodes.count(ownershipLoop) ||
          nodes.lookup(ownershipLoop)->getName().getStringRef() != "intent.parallel")
        return ownership.emitOpError("references a missing Kernel IR loop");
      if (ownership.getWorker() != "program" || ownership.getWorkerAxis() != 0 ||
          ownership.getTraversal() != "persistent" ||
          ownership.getMapping() != "grid_stride")
        return ownership.emitOpError(
            "initial softmax realization requires persistent program-axis-zero grid-stride ownership");
      continue;
    }
    if (auto storage = dyn_cast<StorageOp>(operation)) {
      if (!parameterMetadata.count(storage.getValue()) ||
          !storageValues.insert(storage.getValue()).second)
        return storage.emitOpError("requires a unique kernel ABI value");
      if (storage.getSpace() != "global" ||
          failed(requireOneOf(&operation, "access", storage.getAccess(),
                              {"read", "write", "read_write"})))
        return failure();
      auto metadata = parameterMetadata.lookup(storage.getValue());
      auto parameterKind = metadata.getAs<StringAttr>("kind");
      auto viewKind = metadata.getAs<StringAttr>("view_kind");
      if (!parameterKind || parameterKind.getValue() != "view" || !viewKind)
        return storage.emitOpError("must bind an ABI view parameter");
      StringRef requiredAccess = viewKind.getValue() == "in"       ? "read"
                                 : viewKind.getValue() == "out"    ? "write"
                                                                  : "read_write";
      if (storage.getAccess() != requiredAccess)
        return storage.emitOpError("changes the source View access contract");
      continue;
    }
    if (auto layout = dyn_cast<LayoutOp>(operation)) {
      if (!values.count(layout.getValue()) ||
          !layoutValues.insert(layout.getValue()).second)
        return layout.emitOpError("requires a unique existing Kernel IR value");
      if (layout.getKind() != "row_major" || layout.getOrder().size() != 2 ||
          layout.getOrder()[0] != 0 || layout.getOrder()[1] != 1)
        return layout.emitOpError("initial softmax layout must preserve row-major axes [0, 1]");
      continue;
    }
    if (auto primitive = dyn_cast<PrimitiveOp>(operation)) {
      if (!nodes.count(primitive.getNode()) ||
          !primitiveNodes.insert(primitive.getNode()).second)
        return primitive.emitOpError("requires a unique existing Kernel IR operation");
      if (failed(requireOneOf(&operation, "kind", primitive.getKind(),
                              {"reduction", "pointwise"})))
        return failure();
      Operation *source = nodes.lookup(primitive.getNode());
      StringRef sourceName = source->getName().getStringRef();
      if (primitive.getKind() == "reduction") {
        if (sourceName != "intent.reduce" ||
            getStringAttribute(source, "intent.combine") != primitive.getOperatorName())
          return primitive.emitOpError("does not match its Kernel IR reduction");
        auto axes = source->getAttrOfType<ArrayAttr>("intent.axes");
        auto sourceAxis = axes && axes.size() == 1
                              ? dyn_cast<IntegerAttr>(axes[0])
                              : IntegerAttr();
        if (!sourceAxis ||
            primitive.getAxisAttr().getInt() != sourceAxis.getInt())
          return primitive.emitOpError(
              "reduction axis does not match the Kernel IR reduction");
        StringRef requiredIdentity =
            primitive.getOperatorName() == "maximum" ? "negative_infinity"
            : primitive.getOperatorName() == "add"   ? "zero"
                                                       : StringRef();
        if (requiredIdentity.empty() ||
            primitive.getIdentity() != requiredIdentity ||
            !hasReductionIdentity(source, requiredIdentity))
          return primitive.emitOpError(
              "does not preserve the Kernel IR reduction identity");
      } else {
        StringRef sourceOperator;
        if (sourceName == "intent.broadcast")
          sourceOperator = "broadcast";
        else if (sourceName == "intent.unary" || sourceName == "intent.binary")
          sourceOperator = getStringAttribute(source, "intent.operator");
        if (sourceOperator.empty() || sourceOperator != primitive.getOperatorName())
          return primitive.emitOpError("does not match its Kernel IR pointwise node");
        if (primitive.getAxisAttr().getInt() != -1 ||
            primitive.getIdentity() != "none")
          return primitive.emitOpError(
              "pointwise bindings cannot introduce an axis or identity");
      }
      primitiveBindings[primitive.getNode()] = primitive;
      continue;
    }
    if (auto boundary = dyn_cast<BoundaryOp>(operation)) {
      if (!nodes.count(boundary.getNode()) ||
          nodes.lookup(boundary.getNode())->getName().getStringRef() != "intent.domain" ||
          boundary.getTail() != "masked" ||
          boundary.getPredicate() != "index_lt_extent" ||
          boundary.getLoadFill() != "negative_infinity" ||
          boundaryBindings.count(boundary.getNode()))
        return boundary.emitOpError("requires a valid masked Kernel IR extent binding");
      boundaryBindings[boundary.getNode()] = boundary;
      continue;
    }
    if (auto pipeline = dyn_cast<PipelineOp>(operation)) {
      ++pipelineCount;
      if (pipeline.getLowStages() <= 0 || pipeline.getHighStages() <= 0 ||
          pipeline.getSmemThreshold() <= 0 || pipeline.getPrefetch() ||
          pipeline.getAsyncCopy())
        return pipeline.emitOpError("contains an unsupported pipeline policy");
      continue;
    }
    if (auto launch = dyn_cast<LaunchOp>(operation)) {
      ++launchCount;
      launchLoop = launch.getLoopNode();
      if (!nodes.count(launchLoop) || launch.getGridPolicy() != "persistent_occupancy" ||
          launch.getNumWarps() <= 0)
        return launch.emitOpError("contains an unsupported launch policy");
      continue;
    }
    return operation.emitOpError("is not legal inside intent_plan.plan");
  }

  if (ownershipCount != 1 || pipelineCount != 1 || launchCount != 1)
    return emitOpError("requires exactly one ownership, pipeline and launch binding");
  if (ownershipLoop != launchLoop)
    return emitOpError("ownership and launch must reference the same Kernel IR loop");
  if (storageValues != layoutValues)
    return emitOpError("storage and layout bindings must cover the same values");
  if (storageValues.size() != entry.getNumArguments())
    return emitOpError("storage/layout bindings must cover the complete kernel ABI");
  if (extentBindings.size() != 2 || !extentBindings.count(0) ||
      !extentBindings.count(1) || extentBindings.lookup(0).getTile() != "one" ||
      extentBindings.lookup(1).getTile() != "next_power_of_two")
    return emitOpError("softmax Plan requires row-one and next-power-of-two column extents");
  if (extentBindings.lookup(0).getNodeAttr().getInt() != ownershipLoop)
    return emitOpError("row extent, ownership and launch must reference the same loop");
  for (int64_t axis : {int64_t{0}, int64_t{1}}) {
    auto dimension = dyn_cast<StringAttr>(abiShape[axis]);
    if (!dimension || extentBindings.lookup(axis).getLogical() != dimension.getValue())
      return emitOpError("Plan logical extents must match the source ABI shape");
  }
  auto columnExtent = extentBindings.lookup(1);
  auto boundary = boundaryBindings.lookup(columnExtent.getNode());
  if (!boundary || boundary.getLogical() != columnExtent.getLogical())
    return emitOpError("column extent requires a matching logical boundary binding");

  Operation *rowLoop = nodes.lookup(ownershipLoop);
  if (rowLoop->getNumOperands() != 1 ||
      failed(verifySoftmaxDomain(rowLoop->getOperand(0).getDefiningOp(), entry,
                                 0)) ||
      failed(verifySoftmaxDomain(nodes.lookup(columnExtent.getNode()), entry,
                                 1)))
    return emitOpError(
        "row/column extents must be the source ABI M/N domains");
  DenseSet<int64_t> expectedPrimitives;
  for (Region &region : rowLoop->getRegions())
    region.walk([&](Operation *nested) {
      StringRef name = nested->getName().getStringRef();
      if (name == "intent.reduce" || name == "intent.broadcast" ||
          name == "intent.unary" || name == "intent.binary")
        expectedPrimitives.insert(
            nested->getAttrOfType<IntegerAttr>("intent.node").getInt());
    });
  if (primitiveNodes != expectedPrimitives)
    return emitOpError("primitive bindings must exactly cover row-loop computations");
  return success();
}

#define GET_OP_CLASSES
#include "Intent/Dialect/Plan/IR/PlanOps.cpp.inc"
