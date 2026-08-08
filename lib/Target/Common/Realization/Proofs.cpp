#include "Intent/Target/Common/Realization/KernelFacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace intent::target {
namespace {

bool provesSoftmaxNegativeInfinity(Value loaded) {
  Operation *maximum = nullptr;
  SmallVector<Operation *> subtracts;
  for (Operation *user : loaded.getUsers()) {
    StringRef name = user->getName().getStringRef();
    if (name == "intent.reduce") {
      auto combine = user->getAttrOfType<StringAttr>("intent.combine");
      auto axes = user->getAttrOfType<ArrayAttr>("intent.axes");
      auto axis = axes && axes.size() == 1 ? dyn_cast<IntegerAttr>(axes[0])
                                          : IntegerAttr();
      if (!combine || combine.getValue() != "maximum" || !axis ||
          axis.getInt() != 0 || maximum)
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
  return llvm::all_of(subtracts, [&](Operation *subtract) {
    Operation *broadcast = subtract->getOperand(1).getDefiningOp();
    return broadcast &&
           broadcast->getName().getStringRef() == "intent.broadcast" &&
           broadcast->getNumOperands() == 1 &&
           broadcast->getOperand(0) == maximum->getResult(0);
  });
}

bool proveZeroPaddedUses(Value value, bool zero,
                         llvm::DenseMap<Value, bool> &visited) {
  auto found = visited.find(value);
  if (found != visited.end())
    return !zero || found->second;
  visited[value] = zero;
  for (Operation *user : value.getUsers()) {
    StringRef name = user->getName().getStringRef();
    if (name == "intent.view_store")
      continue;
    if (name == "intent.reduce") {
      auto combine = user->getAttrOfType<StringAttr>("intent.combine");
      if (!zero || !combine || combine.getValue() != "add")
        return false;
      continue;
    }
    if (name == "intent.cast") {
      if (user->getNumResults() != 1 ||
          !proveZeroPaddedUses(user->getResult(0), zero, visited))
        return false;
      continue;
    }
    if (name == "intent.unary") {
      auto logical = user->getAttrOfType<StringAttr>("intent.operator");
      bool resultZero =
          zero && logical && logical.getValue() == "negate";
      if (user->getNumResults() != 1 ||
          !proveZeroPaddedUses(user->getResult(0), resultZero, visited))
        return false;
      continue;
    }
    if (name == "intent.binary") {
      auto logical = user->getAttrOfType<StringAttr>("intent.operator");
      bool resultZero =
          zero && logical && logical.getValue() == "multiply";
      if (user->getNumResults() != 1 ||
          !proveZeroPaddedUses(user->getResult(0), resultZero, visited))
        return false;
      continue;
    }
    return false;
  }
  return true;
}

} // namespace

std::optional<std::string> inferMaskedLaneFill(Value loaded) {
  if (provesSoftmaxNegativeInfinity(loaded))
    return std::string("negative_infinity");
  llvm::DenseMap<Value, bool> visited;
  if (proveZeroPaddedUses(loaded, true, visited))
    return std::string("zero");
  return std::nullopt;
}

} // namespace intent::target
