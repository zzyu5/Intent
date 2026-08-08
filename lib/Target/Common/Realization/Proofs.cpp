#include "Intent/Target/Common/Realization/KernelFacts.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::target {

bool proveMaskedLaneNeutrality(Value loaded) {
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

} // namespace intent::target
