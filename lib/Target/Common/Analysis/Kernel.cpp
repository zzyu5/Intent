#include "Intent/Target/Common/Analysis/Kernel.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::target {

FailureOr<int64_t> getNodeID(Operation &operation, StringRef consumer) {
  auto node = operation.getAttrOfType<IntegerAttr>("intent.node");
  if (!node) {
    operation.emitOpError() << "requires intent.node for " << consumer;
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

  FailureOr<KernelABI> abi = analyzeKernelABI(entries.front());
  FailureOr<RegionStructure> regions = analyzeRegionStructure(entries.front());
  if (failed(abi) || failed(regions))
    return failure();

  KernelModel model{entries.front(),
                    std::move(*abi),
                    std::move(*regions),
                    llvm::DenseMap<int64_t, Operation *>(),
                    llvm::DenseMap<int64_t, Value>(),
                    llvm::DenseMap<Value, int64_t>()};
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
    auto node = operation->getAttrOfType<IntegerAttr>("intent.node");
    if (node && !model.nodes.try_emplace(node.getInt(), operation).second) {
      operation->emitOpError("duplicates an intent.node in canonical Kernel IR");
      return WalkResult::interrupt();
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
  return model;
}

} // namespace intent::target
