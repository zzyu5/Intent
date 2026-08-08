#include "Intent/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace intent {
namespace {

bool isIntentOperation(Operation *operation) {
  return operation->getName().getDialectNamespace() == "intent";
}

LogicalResult verifyResultMetadata(Operation *operation,
                                   DenseSet<int64_t> &valueIDs) {
  auto resultNodes = operation->getAttrOfType<ArrayAttr>("intent.result_nodes");
  auto resultNames = operation->getAttrOfType<ArrayAttr>("intent.result_names");
  auto resultTypes = operation->getAttrOfType<ArrayAttr>("intent.result_types");
  if (!resultNodes || resultNodes.size() != operation->getNumResults())
    return operation->emitOpError(
        "requires one intent.result_nodes entry per SSA result");
  if (!resultTypes || resultTypes.size() != operation->getNumResults())
    return operation->emitOpError(
        "requires one intent.result_types entry per SSA result");
  if (!resultNames || resultNames.size() != operation->getNumResults())
    return operation->emitOpError(
        "requires one intent.result_names entry per SSA result");
  if (auto resultShapes =
          operation->getAttrOfType<ArrayAttr>("intent.result_shapes"))
    if (resultShapes.size() != operation->getNumResults())
      return operation->emitOpError(
          "intent.result_shapes must align with SSA results");
  for (Attribute attribute : resultNodes) {
    auto id = dyn_cast<IntegerAttr>(attribute);
    if (!id || id.getInt() < 0 || !valueIDs.insert(id.getInt()).second)
      return operation->emitOpError(
          "result node IDs must be unique non-negative integers");
  }
  for (Attribute attribute : resultTypes) {
    auto type = dyn_cast<StringAttr>(attribute);
    if (!type || type.getValue().empty())
      return operation->emitOpError(
          "intent.result_types entries must be non-empty strings");
  }
  for (Attribute attribute : resultNames) {
    auto name = dyn_cast<StringAttr>(attribute);
    if (!name || name.getValue().empty())
      return operation->emitOpError(
          "intent.result_names entries must be non-empty strings");
  }
  if (auto resultShapes =
          operation->getAttrOfType<ArrayAttr>("intent.result_shapes")) {
    for (Attribute attribute : resultShapes) {
      auto shape = dyn_cast<ArrayAttr>(attribute);
      if (!shape)
        return operation->emitOpError(
            "intent.result_shapes entries must be arrays");
      for (Attribute dimension : shape)
        if (!isa<StringAttr>(dimension))
          return operation->emitOpError(
              "intent.result_shapes dimensions must be canonical strings");
    }
  }
  return success();
}

LogicalResult verifyRegionArgumentMetadata(Operation *operation,
                                           DenseSet<int64_t> &valueIDs) {
  auto regionNodes =
      operation->getAttrOfType<ArrayAttr>("intent.region_argument_nodes");
  auto regionNames =
      operation->getAttrOfType<ArrayAttr>("intent.region_argument_names");
  if (operation->getNumRegions() == 0) {
    if (regionNodes || regionNames)
      return operation->emitOpError(
          "region argument metadata is only legal on region-owning operations");
    return success();
  }
  if (!regionNodes || !regionNames ||
      regionNodes.size() != operation->getNumRegions() ||
      regionNames.size() != operation->getNumRegions())
    return operation->emitOpError(
        "requires region argument metadata aligned with every region");

  for (auto [regionIndex, region] : llvm::enumerate(operation->getRegions())) {
    auto blockNodes = dyn_cast<ArrayAttr>(regionNodes[regionIndex]);
    auto blockNames = dyn_cast<ArrayAttr>(regionNames[regionIndex]);
    if (!blockNodes || !blockNames || blockNodes.size() != region.getBlocks().size() ||
        blockNames.size() != region.getBlocks().size())
      return operation->emitOpError(
          "region argument metadata must align with every block");
    unsigned blockIndex = 0;
    for (Block &block : region) {
      auto argumentNodes = dyn_cast<ArrayAttr>(blockNodes[blockIndex]);
      auto argumentNames = dyn_cast<ArrayAttr>(blockNames[blockIndex]);
      if (!argumentNodes || !argumentNames ||
          argumentNodes.size() != block.getNumArguments() ||
          argumentNames.size() != block.getNumArguments())
        return operation->emitOpError(
            "block argument metadata must align with every block argument");
      for (auto [nodeAttribute, nameAttribute] :
           llvm::zip(argumentNodes, argumentNames)) {
        auto node = dyn_cast<IntegerAttr>(nodeAttribute);
        auto name = dyn_cast<StringAttr>(nameAttribute);
        if (!node || node.getInt() < 0 ||
            !valueIDs.insert(node.getInt()).second || !name ||
            name.getValue().empty())
          return operation->emitOpError(
              "block arguments require unique IDs and non-empty names");
      }
      ++blockIndex;
    }
  }
  return success();
}

LogicalResult verifyTypeMetadata(Operation *owner, DictionaryAttr metadata) {
  auto type = metadata.getAs<StringAttr>("type");
  if (!type || type.getValue().empty())
    return owner->emitOpError("type metadata requires a non-empty type spelling");
  if (Attribute shapeAttribute = metadata.get("shape")) {
    auto shape = dyn_cast<ArrayAttr>(shapeAttribute);
    if (!shape)
      return owner->emitOpError("shape metadata must be an array");
    for (Attribute dimension : shape)
      if (!isa<StringAttr>(dimension))
        return owner->emitOpError("shape dimensions must be canonical strings");
  }
  return success();
}

LogicalResult verifyParameterMetadata(func::FuncOp function,
                                      ArrayAttr parameters) {
  for (Attribute attribute : parameters) {
    auto metadata = dyn_cast<DictionaryAttr>(attribute);
    if (!metadata)
      return function.emitOpError("parameter metadata entries must be dictionaries");
    auto name = metadata.getAs<StringAttr>("name");
    auto kind = metadata.getAs<StringAttr>("kind");
    if (!name || name.getValue().empty() || !kind)
      return function.emitOpError("parameter metadata requires name and kind");
    if (failed(verifyTypeMetadata(function, metadata)))
      return failure();
    if (kind.getValue() == "view") {
      auto viewKind = metadata.getAs<StringAttr>("view_kind");
      auto constraints = metadata.getAs<DictionaryAttr>("constraints");
      if (!viewKind || !constraints ||
          (viewKind.getValue() != "in" && viewKind.getValue() != "out" &&
           viewKind.getValue() != "inout"))
        return function.emitOpError("view metadata requires kind and constraints");
      if (!constraints.get("strides") || !constraints.get("layout") ||
          !constraints.get("alignment") || !constraints.get("alias") ||
          !constraints.getAs<BoolAttr>("noalias"))
        return function.emitOpError("view constraints metadata is incomplete");
      Attribute stridesAttribute = constraints.get("strides");
      if (!isa<UnitAttr, ArrayAttr>(stridesAttribute))
        return function.emitOpError("view strides must be unit or an array");
      if (auto strides = dyn_cast<ArrayAttr>(stridesAttribute))
        for (Attribute stride : strides)
          if (!isa<UnitAttr, IntegerAttr>(stride))
            return function.emitOpError(
                "view stride entries must be integers or dynamic unit slots");
      if (!isa<UnitAttr, StringAttr>(constraints.get("layout")) ||
          !isa<UnitAttr, IntegerAttr>(constraints.get("alignment")) ||
          !isa<UnitAttr, StringAttr>(constraints.get("alias")))
        return function.emitOpError(
            "view layout/alignment/alias constraints have incompatible MLIR kinds");
    } else if (kind.getValue() != "runtime_scalar" &&
               kind.getValue() != "constexpr" && kind.getValue() != "value") {
      return function.emitOpError("parameter metadata has an invalid kind");
    }
  }
  return success();
}

LogicalResult verifyEffects(Operation *operation) {
  auto effects = operation->getAttrOfType<ArrayAttr>("intent.effects");
  if (!effects)
    return success();
  for (Attribute attribute : effects) {
    auto effect = dyn_cast<DictionaryAttr>(attribute);
    if (!effect)
      return operation->emitOpError("effect metadata has an invalid schema");
    auto kind = effect.getAs<StringAttr>("kind");
    auto resource = effect.getAs<StringAttr>("resource");
    if (!kind || !resource ||
        (kind.getValue() != "read" && kind.getValue() != "write" &&
         kind.getValue() != "atomic" && kind.getValue() != "fence" &&
         kind.getValue() != "rng") ||
        (resource.getValue() != "external_view" &&
         resource.getValue() != "logical_buffer" &&
         resource.getValue() != "rng_state" &&
         resource.getValue() != "ordering"))
      return operation->emitOpError("effect metadata contains an unknown kind/resource");
    auto target = effect.getAs<IntegerAttr>("target");
    if (!target || target.getInt() < -1 ||
        target.getInt() >= static_cast<int64_t>(operation->getNumOperands()))
      return operation->emitOpError("effect target does not reference an operand");
  }
  return success();
}

LogicalResult verifyIndexMetadata(Operation *operation) {
  auto relation = operation->getAttrOfType<ArrayAttr>("intent.index");
  if (!relation)
    return success();
  if (relation.empty())
    return operation->emitOpError("index relation must not be empty");
  for (Attribute attribute : relation) {
    auto term = dyn_cast<DictionaryAttr>(attribute);
    if (!term)
      return operation->emitOpError("index relation term has an invalid schema");
    auto kind = term.getAs<StringAttr>("kind");
    auto operands = term.getAs<ArrayAttr>("operands");
    auto staticValues = term.getAs<ArrayAttr>("static");
    if (!kind || !operands || !staticValues)
      return operation->emitOpError("index relation term has an invalid schema");

    for (Attribute operand : operands) {
      if (isa<UnitAttr>(operand))
        continue;
      auto position = dyn_cast<IntegerAttr>(operand);
      if (!position || position.getInt() < 0 ||
          position.getInt() >= static_cast<int64_t>(operation->getNumOperands()))
        return operation->emitOpError(
            "index relation references an invalid operand position");
    }
    for (Attribute staticValue : staticValues)
      if (!isa<UnitAttr, IntegerAttr>(staticValue))
        return operation->emitOpError(
            "index relation static payload must contain integers or unit slots");

    StringRef termKind = kind.getValue();
    if (termKind == "full_slice" || termKind == "new_axis") {
      if (!operands.empty() || !staticValues.empty())
        return operation->emitOpError(
            "full_slice/new_axis index terms cannot carry payload");
      continue;
    }
    if (termKind == "static_index") {
      if (!operands.empty() || staticValues.size() != 1 ||
          !isa<IntegerAttr>(staticValues[0]))
        return operation->emitOpError(
            "static_index requires exactly one integer literal");
      continue;
    }
    if (termKind == "value_index" || termKind == "region_index") {
      if (operands.size() != 1 || !isa<IntegerAttr>(operands[0]) ||
          !staticValues.empty())
        return operation->emitOpError(
            "value_index/region_index requires exactly one dynamic operand");
      continue;
    }
    if (termKind == "slice") {
      if (operands.size() != 3 || staticValues.size() != 3)
        return operation->emitOpError(
            "slice requires three dynamic/static payload slots");
      for (auto [operand, staticValue] : llvm::zip(operands, staticValues))
        if (!isa<UnitAttr>(operand) && !isa<UnitAttr>(staticValue))
          return operation->emitOpError(
              "slice payload slot cannot be both dynamic and static");
      continue;
    }
    return operation->emitOpError("index relation has an unknown term kind");
  }
  return success();
}

LogicalResult verifyRegionTerminator(Operation *operation, Region &region,
                                     StringRef expected) {
  if (!llvm::hasSingleElement(region))
    return operation->emitOpError("structured regions must contain one block");
  Block &block = region.front();
  if (block.empty())
    return operation->emitOpError("structured regions require a terminator");
  StringRef terminator = block.back().getName().getStringRef();
  if (terminator == expected)
    return success();
  if (terminator == "intent.break" || terminator == "intent.continue")
    return success();
  return operation->emitOpError()
         << "region requires " << expected << " terminator";
}

LogicalResult verifyStructuredRegions(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  if (name == "intent.parallel" || name == "intent.ordered" ||
      name == "intent.state_stream" || name == "intent.for") {
    if (operation->getNumRegions() != 1)
      return operation->emitOpError("requires exactly one structured body");
    return verifyRegionTerminator(operation, operation->getRegion(0),
                                  "intent.yield");
  }
  if (name == "intent.if") {
    if (operation->getNumRegions() < 1 || operation->getNumRegions() > 2)
      return operation->emitOpError("requires one or two branch regions");
    for (Region &region : operation->getRegions())
      if (failed(verifyRegionTerminator(operation, region, "intent.yield")))
        return failure();
    return success();
  }
  if (name == "intent.while") {
    if (operation->getNumRegions() != 2)
      return operation->emitOpError("requires before and after regions");
    if (failed(verifyRegionTerminator(operation, operation->getRegion(0),
                                      "intent.condition")))
      return failure();
    return verifyRegionTerminator(operation, operation->getRegion(1),
                                  "intent.yield");
  }
  if (operation->getNumRegions() != 0)
    return operation->emitOpError("does not own a structured region");
  return success();
}

template <typename AttributeType>
LogicalResult requireAttribute(Operation *operation, StringRef name) {
  Attribute attribute = operation->getAttr(name);
  if (!attribute)
    return operation->emitOpError() << "requires attribute '" << name << "'";
  if (!isa<AttributeType>(attribute))
    return operation->emitOpError()
           << "attribute '" << name << "' has an incompatible MLIR kind";
  return success();
}

LogicalResult verifySemanticAttributeShape(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  if (name == "intent.constant")
    return operation->getAttr("intent.value")
               ? success()
               : operation->emitOpError("requires intent.value");
  if (name == "intent.dim")
    return requireAttribute<IntegerAttr>(operation, "intent.axis");
  if (name == "intent.partition")
    return requireAttribute<StringAttr>(operation, "intent.mode");
  if (name == "intent.transpose")
    return requireAttribute<ArrayAttr>(operation, "intent.permutation");
  if (name == "intent.make_record")
    return requireAttribute<ArrayAttr>(operation, "intent.fields");
  if (name == "intent.extract")
    return requireAttribute<StringAttr>(operation, "intent.key");
  if (name == "intent.unary" || name == "intent.binary")
    return requireAttribute<StringAttr>(operation, "intent.operator");
  if (name == "intent.compare")
    return requireAttribute<StringAttr>(operation, "intent.predicate");
  if (name == "intent.reduce") {
    if (failed(requireAttribute<ArrayAttr>(operation, "intent.axes")))
      return failure();
    return requireAttribute<StringAttr>(operation, "intent.combine");
  }
  if (name == "intent.scan") {
    if (failed(requireAttribute<IntegerAttr>(operation, "intent.axis")))
      return failure();
    if (failed(requireAttribute<BoolAttr>(operation, "intent.inclusive")))
      return failure();
    return requireAttribute<StringAttr>(operation, "intent.combine");
  }
  if (name == "intent.contract")
    return requireAttribute<ArrayAttr>(operation, "intent.reduce");
  if (name == "intent.view_load" || name == "intent.view_store" ||
      name == "intent.gather" || name == "intent.scatter_unique" ||
      name == "intent.scatter_reduce" || name == "intent.buffer_load" ||
      name == "intent.buffer_store" || name == "intent.atomic_add" ||
      name == "intent.atomic_cas")
    return requireAttribute<ArrayAttr>(operation, "intent.index");
  if (name == "intent.call")
    return requireAttribute<StringAttr>(operation, "intent.callee");
  return success();
}

class VerifyKernelIRPass
    : public PassWrapper<VerifyKernelIRPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyKernelIRPass)

  StringRef getArgument() const final { return "verify-intent-kernel"; }
  StringRef getDescription() const final {
    return "Verify Intent Kernel IR metadata and structured-region boundary";
  }
  void runOnOperation() final {
    if (failed(verifyKernelModule(getOperation())))
      signalPassFailure();
  }
};

} // namespace

LogicalResult verifyKernelModule(ModuleOp module) {
  DenseSet<int64_t> operationIDs;
  DenseSet<int64_t> valueIDs;
  bool sawKernel = false;

  for (auto function : module.getOps<func::FuncOp>()) {
    auto kind = function->getAttrOfType<StringAttr>("intent.kind");
    auto parameters = function->getAttrOfType<ArrayAttr>("intent.parameters");
    auto parameterNodes =
        function->getAttrOfType<ArrayAttr>("intent.parameter_nodes");
    auto results = function->getAttrOfType<ArrayAttr>("intent.results");
    if (!kind || !parameters || !parameterNodes || !results)
      return function.emitOpError("requires complete Intent function metadata");
    if (kind.getValue() != "kernel" && kind.getValue() != "helper")
      return function.emitOpError("has an invalid intent.kind");
    if (kind.getValue() == "kernel") {
      if (sawKernel)
        return function.emitOpError("module contains more than one kernel entry");
      sawKernel = true;
    }
    if (parameters.size() != function.getNumArguments() ||
        parameterNodes.size() != function.getNumArguments() ||
        results.size() != function.getNumResults())
      return function.emitOpError("Intent ABI metadata does not match function type");
    if (failed(verifyParameterMetadata(function, parameters)))
      return failure();
    for (Attribute attribute : results) {
      auto metadata = dyn_cast<DictionaryAttr>(attribute);
      if (!metadata || failed(verifyTypeMetadata(function, metadata)))
        return failure();
    }
    for (Attribute attribute : parameterNodes) {
      auto id = dyn_cast<IntegerAttr>(attribute);
      if (!id || id.getInt() < 0 || !valueIDs.insert(id.getInt()).second)
        return function.emitOpError(
            "parameter node IDs must be unique non-negative integers");
    }
    if (!llvm::hasSingleElement(function.getBody()) ||
        function.getBody().front().empty() ||
        function.getBody().front().back().getName().getStringRef() !=
            "intent.return")
      return function.emitOpError(
          "requires one structured entry block ending in intent.return");

    WalkResult walk = function.walk([&](Operation *operation) -> WalkResult {
      if (operation == function.getOperation())
        return WalkResult::advance();
      if (!isIntentOperation(operation))
        return operation->emitOpError("is not legal inside an Intent function"),
               WalkResult::interrupt();
      auto node = operation->getAttrOfType<IntegerAttr>("intent.node");
      if (!node || node.getInt() < 0 ||
          !operationIDs.insert(node.getInt()).second) {
        operation->emitOpError(
            "requires a unique non-negative integer intent.node attribute");
        return WalkResult::interrupt();
      }
      if (failed(verifyResultMetadata(operation, valueIDs)) ||
          failed(verifyRegionArgumentMetadata(operation, valueIDs)) ||
          failed(verifyStructuredRegions(operation)) ||
          failed(verifySemanticAttributeShape(operation)) ||
          failed(verifyEffects(operation)) ||
          failed(verifyIndexMetadata(operation)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walk.wasInterrupted())
      return failure();
  }
  if (!sawKernel)
    return module.emitError("Intent module requires exactly one kernel entry");
  return success();
}

std::unique_ptr<Pass> createVerifyKernelIRPass() {
  return std::make_unique<VerifyKernelIRPass>();
}

void registerIntentPasses() { PassRegistration<VerifyKernelIRPass>(); }

} // namespace intent
