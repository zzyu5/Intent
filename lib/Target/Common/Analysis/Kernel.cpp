#include "Intent/Target/Common/Analysis/Kernel.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::target {
namespace {

IntegerAttr nodeAttribute(Operation &operation) {
  if (auto node = operation.getAttrOfType<IntegerAttr>("intent.node"))
    return node;
  if (operation.getName().getDialectNamespace() == "intent_plan")
    return operation.getAttrOfType<IntegerAttr>("node");
  return {};
}

} // namespace

FailureOr<int64_t> getNodeID(Operation &operation, StringRef consumer) {
  IntegerAttr node = nodeAttribute(operation);
  if (!node) {
    operation.emitOpError() << "requires a stable node ID for " << consumer;
    return failure();
  }
  return node.getInt();
}

FailureOr<int64_t> getValueID(Value value, const KernelModel &kernel,
                              Operation &consumer, StringRef purpose) {
  auto found = kernel.valueIDs.find(value);
  if (found == kernel.valueIDs.end()) {
    consumer.emitOpError() << "references a value without intent ID for "
                           << purpose;
    return failure();
  }
  return found->second;
}

FailureOr<SmallVector<std::string>>
getLogicalShape(Value value, const KernelModel &kernel, Operation &consumer,
                StringRef purpose) {
  ArrayAttr shape;
  auto argument = llvm::find_if(kernel.abi.arguments,
                                [&](const ABIArgument &candidate) {
                                  return candidate.value == value;
                                });
  if (argument != kernel.abi.arguments.end())
    shape = argument->metadata.getAs<ArrayAttr>("shape");
  else if (Operation *definition = value.getDefiningOp()) {
    auto shapes =
        definition->getAttrOfType<ArrayAttr>("intent.result_shapes");
    auto result = dyn_cast<OpResult>(value);
    if (result && shapes && result.getResultNumber() < shapes.size())
      shape = dyn_cast<ArrayAttr>(shapes[result.getResultNumber()]);
  }
  if (!shape) {
    consumer.emitOpError() << "cannot recover canonical logical shape for "
                           << purpose;
    return failure();
  }
  SmallVector<std::string> result;
  result.reserve(shape.size());
  for (Attribute attribute : shape) {
    auto extent = dyn_cast<StringAttr>(attribute);
    if (!extent || extent.getValue().empty()) {
      consumer.emitOpError() << "has malformed canonical logical shape for "
                             << purpose;
      return failure();
    }
    result.push_back(extent.getValue().str());
  }
  return result;
}

FailureOr<KernelABI> analyzeKernelABI(func::FuncOp entry) {
  auto parameterNodes = entry->getAttrOfType<ArrayAttr>("intent.parameter_nodes");
  auto parameters = entry->getAttrOfType<ArrayAttr>("intent.parameters");
  if (!parameterNodes || !parameters ||
      parameterNodes.size() != entry.getNumArguments() ||
      parameters.size() != entry.getNumArguments()) {
    entry.emitOpError("has incomplete canonical ABI metadata");
    return failure();
  }

  KernelABI abi{entry, {}};
  for (auto [index, pair] :
       llvm::enumerate(llvm::zip(parameterNodes, parameters))) {
    auto valueID = dyn_cast<IntegerAttr>(std::get<0>(pair));
    auto metadata = dyn_cast<DictionaryAttr>(std::get<1>(pair));
    auto name = metadata ? metadata.getAs<StringAttr>("name") : StringAttr();
    if (!valueID || valueID.getInt() < 0 || !metadata || !name) {
      entry.emitOpError() << "has malformed ABI metadata for argument " << index;
      return failure();
    }
    abi.arguments.push_back(ABIArgument{
        static_cast<unsigned>(index), valueID.getInt(), name.getValue().str(),
        entry.getArgument(index), entry.getArgument(index).getType(), metadata});
  }
  return abi;
}

FailureOr<RegionStructure> analyzeRegionStructure(func::FuncOp entry) {
  RegionStructure structure;
  WalkResult result = entry.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (operation == entry.getOperation())
      return WalkResult::advance();
    if (operation->getNumRegions() == 0)
      return WalkResult::advance();
    Operation *parent = operation->getParentOp();
    while (parent && parent != entry.getOperation() &&
           parent->getNumRegions() == 0)
      parent = parent->getParentOp();
    if (parent && parent != entry.getOperation()) {
      auto found = structure.positions.find(parent);
      if (found == structure.positions.end()) {
        operation->emitOpError("has an unindexed structured parent");
        return WalkResult::interrupt();
      }
    } else {
      parent = nullptr;
    }
    unsigned position = structure.nodes.size();
    structure.nodes.push_back(RegionNode{operation, parent});
    structure.positions[operation] = position;
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return structure;
}

FailureOr<KernelModel> analyzeKernel(func::FuncOp entry) {
  FailureOr<KernelABI> abi = analyzeKernelABI(entry);
  FailureOr<RegionStructure> regions = analyzeRegionStructure(entry);
  if (failed(abi) || failed(regions))
    return failure();

  KernelModel model{entry,
                    std::move(*abi),
                    std::move(*regions),
                    llvm::DenseMap<int64_t, Operation *>(),
                    llvm::DenseMap<int64_t, Value>(),
                    llvm::DenseMap<Value, int64_t>(),
                    llvm::DenseMap<int64_t, RaggedStructure>(),
                    llvm::DenseMap<int64_t, StateStreamStructure>(),
                    llvm::DenseMap<Value, SmallVector<Value, 2>>(),
                    llvm::DenseMap<Value, SmallVector<Value, 2>>(),
                    llvm::DenseMap<Value, SmallVector<Value, 2>>()};
  auto indexValue = [&](int64_t id, Value value, Operation &owner) {
    if (id < 0 || !model.values.try_emplace(id, value).second ||
        !model.valueIDs.try_emplace(value, id).second) {
      owner.emitOpError("duplicates an intent value ID in canonical Kernel IR");
      return failure();
    }
    return success();
  };
  for (const ABIArgument &argument : model.abi.arguments)
    if (failed(indexValue(argument.valueID, argument.value,
                          *model.entry.getOperation())))
      return failure();
  WalkResult result = model.entry.walk([&](Operation *operation) {
    IntegerAttr node = nodeAttribute(*operation);
    if (node && !model.nodes.try_emplace(node.getInt(), operation).second) {
      operation->emitOpError("duplicates a stable node ID in the executable function");
      return WalkResult::interrupt();
    }
    if (operation != model.entry.getOperation() &&
        operation->getNumRegions() != 0) {
      auto regionNodes =
          operation->getAttrOfType<ArrayAttr>("intent.region_argument_nodes");
      if (!regionNodes || regionNodes.size() != operation->getNumRegions()) {
        operation->emitOpError(
            "has incomplete canonical region argument value IDs");
        return WalkResult::interrupt();
      }
      for (auto [regionIndex, region] :
           llvm::enumerate(operation->getRegions())) {
        auto blockNodes = dyn_cast<ArrayAttr>(regionNodes[regionIndex]);
        if (!blockNodes || blockNodes.size() != region.getBlocks().size()) {
          operation->emitOpError(
              "has region argument value IDs misaligned with blocks");
          return WalkResult::interrupt();
        }
        for (auto [blockIndex, block] : llvm::enumerate(region)) {
          auto argumentNodes = dyn_cast<ArrayAttr>(blockNodes[blockIndex]);
          if (!argumentNodes ||
              argumentNodes.size() != block.getNumArguments()) {
            operation->emitOpError(
                "has region argument value IDs misaligned with block arguments");
            return WalkResult::interrupt();
          }
          for (auto [argumentIndex, attribute] :
               llvm::enumerate(argumentNodes)) {
            auto valueID = dyn_cast<IntegerAttr>(attribute);
            if (!valueID ||
                failed(indexValue(valueID.getInt(),
                                  block.getArgument(argumentIndex), *operation)))
              return WalkResult::interrupt();
          }
        }
      }
    }
    auto resultNodes =
        operation->getAttrOfType<ArrayAttr>("intent.result_nodes");
    if (operation->getNumResults() != 0 &&
        (!resultNodes || resultNodes.size() != operation->getNumResults())) {
      operation->emitOpError("has incomplete canonical result value IDs");
      return WalkResult::interrupt();
    }
    if (resultNodes)
      for (auto [index, attribute] : llvm::enumerate(resultNodes)) {
        auto valueID = dyn_cast<IntegerAttr>(attribute);
        if (!valueID ||
            failed(indexValue(valueID.getInt(), operation->getResult(index),
                              *operation)))
          return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  result = model.entry.walk([&](Operation *operation) {
    auto shapes = operation->getAttrOfType<ArrayAttr>("intent.result_shapes");
    if (!shapes)
      return WalkResult::advance();
    if (shapes.size() != operation->getNumResults()) {
      operation->emitOpError(
          "has result-axis provenance misaligned with canonical results");
      return WalkResult::interrupt();
    }
    for (auto [resultIndex, shapeAttribute] : llvm::enumerate(shapes)) {
      auto shape = dyn_cast<ArrayAttr>(shapeAttribute);
      if (!shape) {
        operation->emitOpError(
            "has malformed canonical result-axis provenance");
        return WalkResult::interrupt();
      }
      SmallVector<Value, 2> arguments(shape.size());
      for (auto [tensorAxis, extentAttribute] : llvm::enumerate(shape)) {
        auto extent = dyn_cast<StringAttr>(extentAttribute);
        if (!extent) {
          operation->emitOpError(
              "has non-symbolic canonical result-axis provenance");
          return WalkResult::interrupt();
        }
        StringRef spelling = extent.getValue();
        if (!spelling.consume_front("?region_"))
          continue;
        auto [valueSpelling, regionAxisSpelling] = spelling.rsplit('_');
        int64_t valueID = -1;
        int64_t regionAxis = -1;
        if (valueSpelling.getAsInteger(10, valueID) ||
            regionAxisSpelling.getAsInteger(10, regionAxis) || regionAxis < 0) {
          operation->emitOpError(
              "has malformed canonical region-axis provenance");
          return WalkResult::interrupt();
        }
        Value source = model.values.lookup(valueID);
        if (!source) {
          operation->emitOpError(
              "references an unknown value as result-axis provenance");
          return WalkResult::interrupt();
        }
        if (isa<BlockArgument>(source))
          arguments[tensorAxis] = source;
      }
      model.resultAxisRegionArguments[operation->getResult(resultIndex)] =
          std::move(arguments);
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  for (const auto &entry : model.nodes) {
    Operation *operation = entry.second;
    StringRef name = ::intent::target::semanticOperationName(*operation);
    if (name == "intent.ragged") {
      if (operation->getNumResults() != 1)
        return operation->emitOpError("has no canonical ragged result");
      RaggedStructure relation{operation, entry.first, -1, {}};
      for (Operation *user : operation->getResult(0).getUsers()) {
        StringRef userName = ::intent::target::semanticOperationName(*user);
        if (userName != "intent.ragged_outer" &&
            userName != "intent.ragged_member")
          continue;
        FailureOr<int64_t> userNode =
            getNodeID(*user, "canonical ragged relation analysis");
        if (failed(userNode))
          return failure();
        if (userName == "intent.ragged_outer") {
          if (relation.outerNode >= 0)
            return operation->emitOpError(
                "has multiple canonical ragged outer domains");
          relation.outerNode = *userNode;
          continue;
        }
        relation.memberNodes.push_back(*userNode);
        auto selector = dyn_cast<BlockArgument>(user->getOperand(1));
        Operation *owner = selector ? selector.getOwner()->getParentOp() : nullptr;
        Operation *outer =
            owner && selector.getArgNumber() == 0 && owner->getNumOperands() > 0
                ? owner->getOperand(0).getDefiningOp()
                : nullptr;
        if (relation.outerNode < 0 && outer &&
            ::intent::target::semanticOperationName(*outer) == "intent.ragged_member") {
          FailureOr<int64_t> outerNode =
              getNodeID(*outer, "nested ragged outer analysis");
          if (failed(outerNode))
            return failure();
          relation.outerNode = *outerNode;
        }
      }
      llvm::sort(relation.memberNodes);
      if (relation.outerNode < 0 || relation.memberNodes.empty())
        return operation->emitOpError(
            "has incomplete canonical ragged ownership domains");
      model.raggedRelations[relation.node] = std::move(relation);
      continue;
    }
    if (name != "intent.state_stream")
      continue;
    Operation *axis = operation->getNumOperands() > 0
                          ? resolveStructuralDomain(operation->getOperand(0))
                          : nullptr;
    FailureOr<int64_t> axisNode =
        axis ? getNodeID(*axis, "canonical state-stream axis analysis")
             : FailureOr<int64_t>(failure());
    if (failed(axisNode))
      return operation->emitOpError("has no canonical state-stream axis");
    int64_t stopValue = -1;
    if (auto stopIndex = operation->getAttrOfType<IntegerAttr>(
            "intent.stop_operand_index")) {
      int64_t operand = stopIndex.getInt();
      FailureOr<int64_t> stopID =
          operand >= 0 && static_cast<unsigned>(operand) < operation->getNumOperands()
              ? getValueID(operation->getOperand(operand), model, *operation,
                           "canonical state-stream stop analysis")
              : FailureOr<int64_t>(failure());
      if (failed(stopID))
        return operation->emitOpError("has no canonical logical stream stop");
      stopValue = *stopID;
    }
    model.stateStreams[entry.first] =
        StateStreamStructure{operation, entry.first, *axisNode, stopValue};
    if (operation->getNumRegions() != 1 ||
        !llvm::hasSingleElement(operation->getRegion(0)))
      return operation->emitOpError(
          "has no canonical single-block state-stream body");
    Operation *terminator = operation->getRegion(0).front().getTerminator();
    if (!terminator || ::intent::target::semanticOperationName(*terminator) != "intent.yield" ||
        terminator->getNumOperands() != operation->getNumResults() ||
        operation->getNumOperands() < operation->getNumResults() + 1)
      return operation->emitOpError(
          "has no canonical state-stream result source schema");
    for (auto [index, result] : llvm::enumerate(operation->getResults())) {
      for (Value source : {operation->getOperand(index + 1),
                           terminator->getOperand(index)}) {
        model.structuredResultSources[result].push_back(source);
        model.structuredValueUsers[source].push_back(result);
      }
    }
  }
  return model;
}

FailureOr<KernelModel> analyzeKernel(ModuleOp module) {
  SmallVector<func::FuncOp> entries;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto kind = function->getAttrOfType<StringAttr>("intent.kind");
    if (kind && kind.getValue() == "kernel")
      entries.push_back(function);
  }
  if (entries.size() != 1) {
    module.emitError("target lowering requires exactly one Intent kernel entry");
    return failure();
  }
  return analyzeKernel(entries.front());
}

} // namespace intent::target
