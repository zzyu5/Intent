#include "Support/Operations.h"

#include "Intent/Target/Common/Analysis/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;

namespace intent::gpu::realization {
namespace {

std::string physicalOperationName(Operation &operation) {
  StringRef source = ::intent::target::semanticOperationName(operation);
  assert(source.starts_with("intent.") &&
         "physical conversion expects a canonical Intent operation");
  return (Twine("intent_plan.exec_") + source.drop_front(7)).str();
}

LogicalResult convertOperation(Operation &operation) {
  if (operation.getName().getDialectNamespace() != "intent")
    return success();

  OperationState state(operation.getLoc(), physicalOperationName(operation));
  state.addOperands(operation.getOperands());
  state.addTypes(operation.getResultTypes());
  state.addAttributes(operation.getAttrs());
  state.addAttribute("intent_plan.source_op",
                     StringAttr::get(operation.getContext(),
                                     ::intent::target::semanticOperationName(operation)));
  for (unsigned index = 0; index < operation.getNumRegions(); ++index)
    state.addRegion();

  OpBuilder builder(&operation);
  Operation *physical = builder.create(state);
  for (unsigned index = 0; index < operation.getNumRegions(); ++index)
    physical->getRegion(index).takeBody(operation.getRegion(index));
  for (auto [source, target] :
       llvm::zip(operation.getResults(), physical->getResults()))
    source.replaceAllUsesWith(target);
  operation.erase();
  return success();
}

} // namespace

LogicalResult materializePhysicalOperations(func::FuncOp entry) {
  SmallVector<Operation *> canonical;
  entry.walk([&](Operation *operation) {
    if (operation->getName().getDialectNamespace() == "intent")
      canonical.push_back(operation);
  });
  for (Operation *operation : llvm::reverse(canonical))
    if (failed(convertOperation(*operation)))
      return failure();
  return success();
}

} // namespace intent::gpu::realization
