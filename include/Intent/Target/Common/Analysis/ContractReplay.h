#ifndef INTENT_TARGET_COMMON_ANALYSIS_CONTRACTREPLAY_H
#define INTENT_TARGET_COMMON_ANALYSIS_CONTRACTREPLAY_H

#include "Intent/Target/Common/Analysis/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include <functional>
#include <optional>

namespace intent::target {

struct ContractOperandReplay {
  llvm::SmallVector<mlir::Operation *> producers;
  llvm::SmallVector<mlir::Operation *> exclusiveProducers;
  llvm::SmallVector<mlir::Operation *> loadDependentProducers;
  llvm::SmallVector<mlir::Operation *> transfers;
};

inline bool isReplayableContractProducer(mlir::Operation &operation) {
  if (operation.getNumRegions() != 0 || operation.getNumResults() != 1)
    return false;
  llvm::StringRef name = semanticOperationName(operation);
  return name == "intent.view_load" || name == "intent.indices" ||
         name == "intent.broadcast" || name == "intent.unary" ||
         name == "intent.binary" || name == "intent.compare" ||
         name == "intent.mask" || name == "intent.select" ||
         name == "intent.cast" || name == "intent.bitcast" ||
         name == "intent.full" ||
         name == "intent.zeros" || name == "intent.members" ||
         name == "intent.gather" ||
         name == "intent.reshape" || name == "intent.transpose";
}

inline std::optional<ContractOperandReplay>
analyzeContractOperandReplay(mlir::Value operand,
                             mlir::Operation &contract) {
  if (!mlir::isa<mlir::RankedTensorType>(operand.getType()))
    return std::nullopt;

  ContractOperandReplay replay;
  llvm::DenseSet<mlir::Operation *> visited;
  std::function<bool(mlir::Value)> collect = [&](mlir::Value value) {
    if (!mlir::isa<mlir::RankedTensorType>(value.getType()))
      return true;
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || !isReplayableContractProducer(*definition))
      return false;
    if (!visited.insert(definition).second)
      return true;
    for (mlir::Value input : definition->getOperands())
      if (!collect(input))
        return false;
    replay.producers.push_back(definition);
    if (semanticOperationName(*definition) == "intent.view_load")
      replay.transfers.push_back(definition);
    return true;
  };
  if (!collect(operand))
    return std::nullopt;

  llvm::DenseSet<mlir::Operation *> loadDependent;
  for (mlir::Operation *producer : replay.producers) {
    bool dependsOnLoad =
        semanticOperationName(*producer) == "intent.view_load" ||
        llvm::any_of(producer->getOperands(), [&](mlir::Value input) {
          mlir::Operation *definition = input.getDefiningOp();
          return definition && loadDependent.contains(definition);
        });
    if (dependsOnLoad)
      loadDependent.insert(producer);
    if (dependsOnLoad)
      replay.loadDependentProducers.push_back(producer);
    bool externalUser = llvm::any_of(
        producer->getResult(0).getUsers(), [&](mlir::Operation *user) {
          return user != &contract && !visited.contains(user);
        });
    if (externalUser && dependsOnLoad)
      return std::nullopt;
    if (!externalUser)
      replay.exclusiveProducers.push_back(producer);
  }
  return replay;
}

} // namespace intent::target

#endif // INTENT_TARGET_COMMON_ANALYSIS_CONTRACTREPLAY_H
