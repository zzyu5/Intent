#ifndef INTENT_TARGET_COMMON_ANALYSIS_KERNEL_H
#define INTENT_TARGET_COMMON_ANALYSIS_KERNEL_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace intent::target {

struct ABIArgument {
  unsigned index;
  int64_t valueID;
  std::string name;
  mlir::Value value;
  mlir::Type type;
  mlir::DictionaryAttr metadata;
};

struct KernelABI {
  mlir::func::FuncOp entry;
  llvm::SmallVector<ABIArgument> arguments;
};

struct RegionNode {
  mlir::Operation *operation;
  mlir::Operation *parent;
};

struct RegionStructure {
  llvm::SmallVector<RegionNode> nodes;
  llvm::DenseMap<mlir::Operation *, unsigned> positions;
};

struct RaggedStructure {
  mlir::Operation *operation;
  int64_t node;
  int64_t outerNode;
  llvm::SmallVector<int64_t> memberNodes;
};

struct StateStreamStructure {
  mlir::Operation *operation;
  int64_t node;
  int64_t axisNode;
  int64_t stopNode;
};

struct KernelModel {
  mlir::func::FuncOp entry;
  KernelABI abi;
  RegionStructure regions;
  llvm::DenseMap<int64_t, mlir::Operation *> nodes;
  llvm::DenseMap<int64_t, mlir::Value> values;
  llvm::DenseMap<mlir::Value, int64_t> valueIDs;
  llvm::DenseMap<int64_t, RaggedStructure> raggedRelations;
  llvm::DenseMap<int64_t, StateStreamStructure> stateStreams;
};

mlir::FailureOr<int64_t> getNodeID(mlir::Operation &operation,
                                   llvm::StringRef consumer);

mlir::FailureOr<int64_t> getValueID(mlir::Value value,
                                    const KernelModel &kernel,
                                    mlir::Operation &consumer,
                                    llvm::StringRef purpose);

mlir::FailureOr<KernelABI> analyzeKernelABI(mlir::func::FuncOp entry);

mlir::FailureOr<RegionStructure>
analyzeRegionStructure(mlir::func::FuncOp entry);

mlir::FailureOr<KernelModel> analyzeKernel(mlir::ModuleOp module);

} // namespace intent::target

#endif
