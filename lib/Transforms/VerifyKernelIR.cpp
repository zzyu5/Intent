#include "Intent/Transforms/Passes.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace intent {
namespace {

bool isIntentOperation(Operation *operation) {
  return operation->getName().getDialectNamespace() == "intent";
}

std::optional<std::string> builtinMetadataSpelling(Type type) {
  std::string spelling;
  llvm::raw_string_ostream stream(spelling);
  type.print(stream);
  stream.flush();
  if (spelling == "i1")
    return std::string("bool");
  if (StringRef(spelling).starts_with("ui"))
    return "u" + StringRef(spelling).drop_front(2).str();
  if (spelling == "f8E4M3FN")
    return std::string("f8e4m3fn");
  if (spelling == "f8E5M2")
    return std::string("f8e5m2");
  if (spelling == "f8E8M0FNU")
    return std::string("f8e8m0fnu");
  if (isa<IntegerType, IndexType, FloatType>(type))
    return spelling;
  return std::nullopt;
}

std::optional<StringRef> logicalMetadataSpelling(Type type) {
  if (auto value = dyn_cast<intent::LogicalIndexType>(type))
    return value.getSpec();
  if (auto value = dyn_cast<intent::DomainType>(type))
    return value.getSpec();
  if (auto value = dyn_cast<intent::RegionType>(type))
    return value.getSpec();
  if (auto value = dyn_cast<intent::PartitionType>(type))
    return value.getSpec();
  if (auto value = dyn_cast<intent::RaggedType>(type))
    return value.getSpec();
  if (auto value = dyn_cast<intent::BufferType>(type))
    return value.getSpec();
  if (auto value = dyn_cast<intent::RecordType>(type))
    return value.getSpec();
  if (auto value = dyn_cast<intent::ConstexprType>(type))
    return value.getSpec();
  if (auto value = dyn_cast<intent::EnumType>(type))
    return value.getSpec();
  return std::nullopt;
}

FailureOr<std::string> tensorMetadataSpelling(Operation *owner,
                                              RankedTensorType tensor,
                                              ArrayAttr shape) {
  if (!shape || shape.size() != static_cast<size_t>(tensor.getRank())) {
    owner->emitOpError("tensor metadata shape does not match the SSA rank");
    return failure();
  }
  std::optional<std::string> element =
      builtinMetadataSpelling(tensor.getElementType());
  if (!element) {
    owner->emitOpError("tensor metadata has no canonical element type spelling");
    return failure();
  }
  std::string result = "tensor<";
  for (auto [axis, attribute] : llvm::enumerate(shape)) {
    auto dimension = dyn_cast<StringAttr>(attribute);
    if (!dimension) {
      owner->emitOpError("tensor shape metadata contains a non-string dimension");
      return failure();
    }
    if (tensor.isDynamicDim(axis)) {
      if (dimension.getValue().empty()) {
        owner->emitOpError("dynamic tensor metadata dimension is empty");
        return failure();
      }
    } else {
      int64_t metadataExtent = -1;
      if (dimension.getValue().getAsInteger(10, metadataExtent) ||
          metadataExtent != tensor.getDimSize(axis)) {
        owner->emitOpError(
            "static tensor metadata dimension does not match the SSA type");
        return failure();
      }
    }
    result += dimension.getValue().str() + "x";
  }
  result += *element + ">";
  return result;
}

LogicalResult verifyMetadataMatchesType(Operation *owner, Type actualType,
                                        StringRef metadataType,
                                        ArrayAttr shape = {}) {
  std::string expected;
  if (auto view = dyn_cast<intent::ViewType>(actualType)) {
    auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
    FailureOr<std::string> spelling =
        tensor ? tensorMetadataSpelling(owner, tensor, shape)
               : FailureOr<std::string>(failure());
    if (!tensor)
      owner->emitOpError("view metadata requires a ranked tensor SSA type");
    if (failed(spelling))
      return failure();
    expected = std::move(*spelling);
  } else if (auto tensor = dyn_cast<RankedTensorType>(actualType)) {
    FailureOr<std::string> spelling =
        tensorMetadataSpelling(owner, tensor, shape);
    if (failed(spelling))
      return failure();
    expected = std::move(*spelling);
  } else if (std::optional<StringRef> spelling =
                 logicalMetadataSpelling(actualType)) {
    expected = spelling->str();
  } else if (std::optional<std::string> spelling =
                 builtinMetadataSpelling(actualType)) {
    expected = std::move(*spelling);
  } else {
    return owner->emitOpError(
        "SSA type has no canonical Intent metadata spelling");
  }
  if (metadataType != expected)
    return owner->emitOpError()
           << "type metadata '" << metadataType << "' does not match SSA type '"
           << expected << "'";
  return success();
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
  auto resultShapes =
      operation->getAttrOfType<ArrayAttr>("intent.result_shapes");
  for (auto [index, attribute] : llvm::enumerate(resultTypes)) {
    auto type = cast<StringAttr>(attribute);
    ArrayAttr shape =
        resultShapes ? dyn_cast<ArrayAttr>(resultShapes[index]) : ArrayAttr();
    if (failed(verifyMetadataMatchesType(
            operation, operation->getResult(index).getType(), type.getValue(),
            shape)))
      return failure();
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
  for (auto [index, attribute] : llvm::enumerate(parameters)) {
    auto metadata = dyn_cast<DictionaryAttr>(attribute);
    if (!metadata)
      return function.emitOpError("parameter metadata entries must be dictionaries");
    auto name = metadata.getAs<StringAttr>("name");
    auto kind = metadata.getAs<StringAttr>("kind");
    if (!name || name.getValue().empty() || !kind)
      return function.emitOpError("parameter metadata requires name and kind");
    if (failed(verifyTypeMetadata(function, metadata)))
      return failure();
    if (failed(verifyMetadataMatchesType(
            function, function.getArgument(index).getType(),
            metadata.getAs<StringAttr>("type").getValue(),
            metadata.getAs<ArrayAttr>("shape"))))
      return failure();
    if (kind.getValue() == "view") {
      auto viewKind = metadata.getAs<StringAttr>("view_kind");
      auto constraints = metadata.getAs<DictionaryAttr>("constraints");
      if (!viewKind || !constraints ||
          (viewKind.getValue() != "in" && viewKind.getValue() != "out" &&
           viewKind.getValue() != "inout"))
        return function.emitOpError("view metadata requires kind and constraints");
      auto view = dyn_cast<intent::ViewType>(function.getArgument(index).getType());
      if (!view || view.getAccess() != viewKind.getValue())
        return function.emitOpError(
            "view access metadata does not match the SSA view type");
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
         kind.getValue() != "atomic") ||
        (resource.getValue() != "external_view" &&
         resource.getValue() != "logical_buffer"))
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
  return operation->emitOpError()
         << "region requires " << expected << " terminator";
}

LogicalResult verifyStructuredRegions(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  if (name == "intent.parallel" || name == "intent.state_stream" ||
      name == "intent.for") {
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
  if (name == "intent.assume_in_bounds")
    return requireAttribute<IntegerAttr>(operation, "intent.axis");
  if (name == "intent.partition") {
    auto mode = operation->getAttrOfType<StringAttr>("intent.mode");
    if (!mode || (mode.getValue() != "extent" && mode.getValue() != "count"))
      return operation->emitOpError("requires extent or count partition semantics");
    if (operation->getNumOperands() != 2 || operation->getNumResults() != 1 ||
        !isa<intent::DomainType, intent::RegionType>(
            operation->getOperand(0).getType()) ||
        !isa<IntegerType, IndexType>(operation->getOperand(1).getType()))
      return operation->emitOpError("has no canonical partition operand schema");
    auto result = dyn_cast<intent::PartitionType>(operation->getResult(0).getType());
    std::string resultPrefix =
        (Twine("partition<") + mode.getValue() + ",").str();
    if (!result || !result.getSpec().starts_with(resultPrefix))
      return operation->emitOpError(
          "partition result type does not match its semantic mode");
    Operation *definition = operation->getOperand(1).getDefiningOp();
    auto fixed = definition && definition->getName().getStringRef() ==
                                   "intent.constant"
                     ? definition->getAttrOfType<IntegerAttr>("intent.value")
                     : IntegerAttr();
    if (mode.getValue() == "extent" && (!fixed || fixed.getInt() <= 0))
      return operation->emitOpError(
          "extent partition requires a positive constant extent");
    if (mode.getValue() == "count" && fixed && fixed.getInt() <= 0)
      return operation->emitOpError("partition count must be positive");
    return success();
  }
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
  if (name == "intent.random") {
    if (operation->getNumOperands() != 2 || operation->getNumResults() != 1)
      return operation->emitOpError(
          "requires seed/counter operands and one uniform result");
    auto algorithm =
        operation->getAttrOfType<StringAttr>("intent.algorithm");
    auto elementType = [](Type type) {
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        return tensor.getElementType();
      return type;
    };
    Type seedType = elementType(operation->getOperand(0).getType());
    Type counterType = elementType(operation->getOperand(1).getType());
    Type resultType = operation->getResult(0).getType();
    if (auto tensor = dyn_cast<RankedTensorType>(resultType))
      resultType = tensor.getElementType();
    if (!algorithm || algorithm.getValue() != "counter_xorshift32" ||
        !isa<IntegerType, IndexType>(seedType) ||
        !isa<IntegerType, IndexType, intent::LogicalIndexType>(counterType) ||
        !resultType.isF32())
      return operation->emitOpError(
          "requires integer seed/counter and counter_xorshift32 f32 output");
    return success();
  }
  auto verifyCombinerUse = [&](bool scan) -> LogicalResult {
    auto components =
        operation->getAttrOfType<IntegerAttr>("intent.component_count");
    auto captures = operation->getAttrOfType<IntegerAttr>("intent.capture_count");
    Attribute combine = operation->getAttr("intent.combine");
    if (!components || components.getInt() <= 0 || !captures ||
        captures.getInt() < 0 || !combine)
      return operation->emitOpError(
          "requires component_count, capture_count, and combine semantics");
    unsigned componentCount = static_cast<unsigned>(components.getInt());
    unsigned captureCount = static_cast<unsigned>(captures.getInt());
    if (operation->getNumOperands() != 2 * componentCount + captureCount ||
        operation->getNumResults() != componentCount)
      return operation->emitOpError(
          "combiner operands/results do not match its component schema");

    for (unsigned index = 0; index < componentCount; ++index) {
      auto source = dyn_cast<RankedTensorType>(operation->getOperand(index).getType());
      Type identity = operation->getOperand(componentCount + index).getType();
      Type result = operation->getResult(index).getType();
      Type resultElement = result;
      if (auto tensor = dyn_cast<RankedTensorType>(result))
        resultElement = tensor.getElementType();
      if (!source || !isa<IntegerType, IndexType, FloatType>(identity) ||
          source.getElementType() != identity || resultElement != identity)
        return operation->emitOpError(
            "combiner source, identity, and result component types disagree");
      if (scan) {
        auto resultTensor = dyn_cast<RankedTensorType>(result);
        if (!resultTensor || resultTensor.getShape() != source.getShape())
          return operation->emitOpError(
              "scan result components must preserve their source shape");
      }
    }

    if (auto builtin = dyn_cast<StringAttr>(combine)) {
      if (captureCount != 0 ||
          !llvm::is_contained(
              {StringRef("add"), StringRef("maximum"),
               StringRef("logical_or"), StringRef("logical_and")},
              builtin.getValue()))
        return operation->emitOpError(
            "has unsupported built-in combiner semantics");
      return success();
    }

    auto reference = dyn_cast<FlatSymbolRefAttr>(combine);
    auto module = operation->getParentOfType<ModuleOp>();
    auto function = reference && module
                        ? module.lookupSymbol<func::FuncOp>(reference.getValue())
                        : func::FuncOp();
    auto role = function
                    ? function->getAttrOfType<StringAttr>("intent.role")
                    : StringAttr();
    auto helperComponents =
        function ? function->getAttrOfType<IntegerAttr>(
                       "intent.component_count")
                 : IntegerAttr();
    auto helperCaptures =
        function ? function->getAttrOfType<IntegerAttr>("intent.capture_count")
                 : IntegerAttr();
    if (!function || !role || role.getValue() != "combiner" ||
        !helperComponents || helperComponents.getInt() != components.getInt() ||
        !helperCaptures || helperCaptures.getInt() != captures.getInt() ||
        function.getNumArguments() != 2 * componentCount + captureCount ||
        function.getNumResults() != componentCount)
      return operation->emitOpError(
          "does not reference a typed combiner helper with matching arity");
    auto builtin =
        operation->getAttrOfType<StringAttr>("intent.combine_builtin");
    auto helperBuiltin =
        function->getAttrOfType<StringAttr>("intent.builtin");
    if (static_cast<bool>(builtin) != static_cast<bool>(helperBuiltin) ||
        (builtin && builtin.getValue() != helperBuiltin.getValue()) ||
        (builtin &&
         (builtin.getValue() != "argmax_lowest" || componentCount != 2 ||
          captureCount != 0)))
      return operation->emitOpError(
          "combiner built-in role does not match its typed helper contract");
    for (unsigned index = 0; index < componentCount; ++index) {
      Type identity = operation->getOperand(componentCount + index).getType();
      if (function.getArgument(index).getType() != identity ||
          function.getArgument(componentCount + index).getType() != identity ||
          function.getResultTypes()[index] != identity)
        return operation->emitOpError(
            "combiner helper accumulator types do not match identities");
    }
    for (unsigned index = 0; index < captureCount; ++index)
      if (!isa<IntegerType, IndexType, FloatType>(
              operation->getOperand(2 * componentCount + index).getType()) ||
          function.getArgument(2 * componentCount + index).getType() !=
              operation->getOperand(2 * componentCount + index).getType())
        return operation->emitOpError(
            "combiner helper captures must be matching explicit scalar operands");
    return success();
  };
  if (name == "intent.reduce") {
    if (failed(requireAttribute<ArrayAttr>(operation, "intent.axes")))
      return failure();
    return verifyCombinerUse(false);
  }
  if (name == "intent.scan") {
    if (failed(requireAttribute<IntegerAttr>(operation, "intent.axis")))
      return failure();
    if (failed(requireAttribute<BoolAttr>(operation, "intent.inclusive")))
      return failure();
    return verifyCombinerUse(true);
  }
  if (name == "intent.contract" || name == "intent.scaled_contract") {
    if (failed(requireAttribute<ArrayAttr>(operation, "intent.reduce")))
      return failure();
    if (failed(requireAttribute<ArrayAttr>(operation, "intent.batch")) ||
        failed(requireAttribute<StringAttr>(operation, "intent.multiply")) ||
        failed(requireAttribute<StringAttr>(operation, "intent.combine")))
      return failure();
    auto multiply = operation->getAttrOfType<StringAttr>("intent.multiply");
    auto combine = operation->getAttrOfType<StringAttr>("intent.combine");
    if (multiply.getValue() != "multiply" || combine.getValue() != "add")
      return operation->emitOpError(
          "supports only the multiply/add semiring; use explicit pointwise "
          "operations and intent.reduce for another semiring");
    if (name == "intent.scaled_contract") {
      auto lhsGroup =
          operation->getAttrOfType<IntegerAttr>("intent.lhs_group_size");
      auto rhsGroup =
          operation->getAttrOfType<IntegerAttr>("intent.rhs_group_size");
      auto lhsFormat =
          operation->getAttrOfType<StringAttr>("intent.lhs_format");
      auto rhsFormat =
          operation->getAttrOfType<StringAttr>("intent.rhs_format");
      if (operation->getNumOperands() != 4 || operation->getNumResults() != 1 ||
          !lhsGroup || !rhsGroup || lhsGroup.getInt() <= 0 ||
          rhsGroup.getInt() <= 0 || !lhsFormat || !rhsFormat)
        return operation->emitOpError(
            "has no canonical four-operand scaled-contraction contract");
      auto lhs = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
      auto rhs = dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
      auto lhsScale =
          dyn_cast<RankedTensorType>(operation->getOperand(2).getType());
      auto rhsScale =
          dyn_cast<RankedTensorType>(operation->getOperand(3).getType());
      bool flat = lhs && rhs && lhs.getRank() == 2 && rhs.getRank() == 2;
      bool grouped = lhs && rhs && lhs.getRank() == 3 && rhs.getRank() == 3;
      if (!lhs || !rhs || !lhsScale || !rhsScale || (!flat && !grouped) ||
          lhsScale.getRank() != 2 || rhsScale.getRank() != 2 ||
          lhs.getElementType() != rhs.getElementType() ||
          !isa<Float8E4M3FNType, Float8E5M2Type>(lhs.getElementType()) ||
          !isa<Float8E8M0FNUType>(lhsScale.getElementType()) ||
          lhsScale.getElementType() != rhsScale.getElementType())
        return operation->emitOpError(
            "requires matching rank-two or grouped rank-three FP8 data and "
            "rank-two E8M0 scale tensors");
    }
    return success();
  }
  if (name == "intent.atomic_cas") {
    auto relation = operation->getAttrOfType<ArrayAttr>("intent.index");
    auto compareIndex = operation->getAttrOfType<IntegerAttr>(
        "intent.compare_operand_index");
    auto valueIndex = operation->getAttrOfType<IntegerAttr>(
        "intent.value_operand_index");
    auto ordering = operation->getAttrOfType<StringAttr>("intent.ordering");
    auto scope = operation->getAttrOfType<StringAttr>("intent.scope");
    bool knownOrdering =
        ordering && llvm::is_contained(
                        {StringRef("relaxed"), StringRef("acquire"),
                         StringRef("release"), StringRef("acq_rel"),
                         StringRef("seq_cst")},
                        ordering.getValue());
    bool knownScope =
        scope && llvm::is_contained(
                     {StringRef("work_item"), StringRef("subgroup"),
                      StringRef("workgroup"), StringRef("device"),
                      StringRef("system")},
                     scope.getValue());
    if (!relation || !compareIndex || !valueIndex ||
        compareIndex.getInt() <= 0 ||
        valueIndex.getInt() != compareIndex.getInt() + 1 ||
        static_cast<unsigned>(valueIndex.getInt()) !=
            operation->getNumOperands() - 1 ||
        operation->getNumResults() != 1 ||
        operation->getOperand(compareIndex.getInt()).getType() !=
            operation->getOperand(valueIndex.getInt()).getType() ||
        operation->getResult(0).getType() !=
            operation->getOperand(valueIndex.getInt()).getType() ||
        !knownOrdering || !knownScope)
      return operation->emitOpError(
          "requires trailing type-matched compare/value operands and canonical memory semantics");
    return success();
  }
  if (name == "intent.view_load" || name == "intent.view_store" ||
      name == "intent.gather" || name == "intent.scatter_unique" ||
      name == "intent.scatter_reduce" || name == "intent.buffer_load" ||
      name == "intent.buffer_store" || name == "intent.atomic_add")
    return requireAttribute<ArrayAttr>(operation, "intent.index");
  return success();
}

LogicalResult verifyCombinerHelper(func::FuncOp function) {
  auto role = function->getAttrOfType<StringAttr>("intent.role");
  if (!role || role.getValue() != "combiner")
    return function.emitOpError("helper requires a supported intent.role");
  auto components =
      function->getAttrOfType<IntegerAttr>("intent.component_count");
  auto captures = function->getAttrOfType<IntegerAttr>("intent.capture_count");
  if (!components || components.getInt() <= 0 || !captures ||
      captures.getInt() < 0)
    return function.emitOpError(
        "combiner helper requires positive component_count and non-negative capture_count");
  unsigned componentCount = static_cast<unsigned>(components.getInt());
  unsigned captureCount = static_cast<unsigned>(captures.getInt());
  if (auto builtin =
          function->getAttrOfType<StringAttr>("intent.builtin"))
    if (builtin.getValue() != "argmax_lowest" || componentCount != 2 ||
        captureCount != 0)
      return function.emitOpError("has an unsupported combiner built-in role");
  if (function.getNumArguments() != 2 * componentCount + captureCount ||
      function.getNumResults() != componentCount)
    return function.emitOpError(
        "combiner helper signature does not match its component schema");
  for (unsigned index = 0; index < componentCount; ++index) {
    Type type = function.getResultTypes()[index];
    if (!isa<IntegerType, IndexType, FloatType>(type) ||
        function.getArgument(index).getType() != type ||
        function.getArgument(componentCount + index).getType() != type)
      return function.emitOpError(
          "combiner accumulator parameters/results must be matching scalar types");
  }
  for (Operation &operation : function.getBody().front()) {
    StringRef name = operation.getName().getStringRef();
    if (!llvm::is_contained(
            {StringRef("intent.constant"), StringRef("intent.make_record"),
             StringRef("intent.extract"), StringRef("intent.unary"),
             StringRef("intent.binary"), StringRef("intent.compare"),
             StringRef("intent.select"), StringRef("intent.cast"),
             StringRef("intent.return")},
            name))
      return operation.emitOpError(
          "is not legal in a pure scalar combiner helper");
    if (operation.getNumRegions() != 0 || operation.getAttr("intent.effects"))
      return operation.emitOpError(
          "combiner helper operations must be effect-free scalar SSA");
    if (name == "intent.return" &&
        &operation != &function.getBody().front().back())
      return operation.emitOpError(
          "combiner helper return must be the final operation");
  }
  Operation &terminator = function.getBody().front().back();
  if (terminator.getNumOperands() != componentCount)
    return terminator.emitOpError(
        "combiner return schema does not match accumulator components");
  for (unsigned index = 0; index < componentCount; ++index)
    if (terminator.getOperand(index).getType() != function.getResultTypes()[index])
      return terminator.emitOpError(
          "combiner return type does not match accumulator component");
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
    if (kind.getValue() == "helper" && failed(verifyCombinerHelper(function)))
      return failure();
    if (parameters.size() != function.getNumArguments() ||
        parameterNodes.size() != function.getNumArguments() ||
        results.size() != function.getNumResults())
      return function.emitOpError("Intent ABI metadata does not match function type");
    if (failed(verifyParameterMetadata(function, parameters)))
      return failure();
    for (auto [index, attribute] : llvm::enumerate(results)) {
      auto metadata = dyn_cast<DictionaryAttr>(attribute);
      if (!metadata || failed(verifyTypeMetadata(function, metadata)))
        return failure();
      if (failed(verifyMetadataMatchesType(
              function, function.getResultTypes()[index],
              metadata.getAs<StringAttr>("type").getValue(),
              metadata.getAs<ArrayAttr>("shape"))))
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
      if (kind.getValue() == "kernel" &&
          operation->getName().getStringRef() == "intent.indices" &&
          operation->getParentOp() == function.getOperation())
        return operation->emitOpError(
                   "requires an enclosing execution region that selects the indexed range"),
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
