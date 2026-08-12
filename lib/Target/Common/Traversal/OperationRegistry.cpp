#include "Intent/Target/Common/Traversal/OperationRegistry.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Block.h"

using namespace mlir;

namespace intent::target {
namespace {

LogicalResult traverseBlock(Block &block,
                            const OperationHandlerRegistry &registry,
                            StringRef stage) {
  for (Operation &operation : block) {
    const OperationHandler *handler =
        registry.lookup(operation.getName().getStringRef());
    if (!handler)
      return operation.emitOpError()
             << "has no registered handler during " << stage;
    if (handler->enter && failed(handler->enter(operation)))
      return failure();
    for (Region &region : operation.getRegions())
      for (Block &nested : region)
        if (failed(traverseBlock(nested, registry, stage)))
          return failure();
    if (handler->leave && failed(handler->leave(operation)))
      return failure();
  }
  return success();
}

} // namespace

LogicalResult OperationHandlerRegistry::add(StringRef operationName,
                                            OperationHandler handler) {
  if (!handlers.try_emplace(operationName, std::move(handler)).second)
    return failure();
  return success();
}

const OperationHandler *
OperationHandlerRegistry::lookup(StringRef operationName) const {
  auto found = handlers.find(operationName);
  return found == handlers.end() ? nullptr : &found->second;
}

LogicalResult OperationHandlerRegistry::dispatch(Operation &operation,
                                                 StringRef stage) const {
  const OperationHandler *handler = lookup(operation.getName().getStringRef());
  if (!handler)
    return operation.emitOpError() << "has no registered handler during " << stage;
  if (handler->enter && failed(handler->enter(operation)))
    return failure();
  if (handler->leave && failed(handler->leave(operation)))
    return failure();
  return success();
}

LogicalResult traverseKernel(func::FuncOp entry,
                             const OperationHandlerRegistry &registry,
                             StringRef stage) {
  if (!llvm::hasSingleElement(entry.getBody()))
    return entry.emitOpError("requires a single canonical entry block");
  return traverseBlock(entry.getBody().front(), registry, stage);
}

} // namespace intent::target
