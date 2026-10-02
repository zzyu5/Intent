#ifndef INTENT_DIALECT_CPU_ANALYSIS_STORAGE_H
#define INTENT_DIALECT_CPU_ANALYSIS_STORAGE_H

#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace intent::cpu {

void registerStorageInterfaces(mlir::DialectRegistry &registry);

// Check that explicit release operands connect to owned storage, no heap origin
// is disconnected from all releases, and unconditional lexical frees precede
// no later accesses. Native dealloc verifies its condition/retention schema;
// its transformation establishes the complex control-flow ownership algorithm.
// This structural audit is not a second path-sensitive deallocation proof.
mlir::LogicalResult verifyStorageOwnership(mlir::func::FuncOp function);

struct StorageOriginFacts {
  llvm::SmallVector<mlir::Value> values;
  // All origin frontiers have a declared identity: allocation, public argument
  // or collective invocation formal. Completeness does not imply disjointness.
  bool complete = false;

  mlir::Value uniqueOrigin() const;
};

struct StorageEffect {
  mlir::Operation *operation;
  mlir::MemoryEffects::EffectInstance effect;
};

struct StorageEffects {
  // Addressable storage effects. The registered cache-hint resource preserves
  // prefetch instructions for MLIR, but does not read or write buffer contents.
  llvm::SmallVector<StorageEffect> entries;
  bool complete = true;
  bool ordered = false;
};

// A closed upper bound for a current descriptor dimension. This query does not
// use observed runtime shapes or interpret wrapping index arithmetic as affine.
std::optional<int64_t> constantDimensionUpperBound(mlir::Value memory, unsigned axis);

// Current-IR aliases of the supplied origin, not an execution or allocation plan.
// Unknown users and escapes remain visible and make the closure incomplete.
struct StorageAliasFacts {
  mlir::Value root;
  llvm::SmallVector<mlir::Value> values;
  llvm::SmallVector<mlir::Operation *> users;
  bool complete = true;
};

bool isStorageAliasOperation(mlir::Operation *operation);

struct StorageLifetime {
  mlir::memref::AllocOp allocation;
  mlir::memref::DeallocOp end;
  StorageAliasFacts aliases;

  bool contains(mlir::Operation *operation) const;
};

// An analysis of one current IR snapshot. Recreate it after changing operands,
// control flow, views, storage or effects; it never maintains a shadow program.
class StorageAnalysis {
public:
  explicit StorageAnalysis(mlir::func::FuncOp function);

  StorageOriginFacts origins(mlir::Value memory) const;
  mlir::Value uniqueOrigin(mlir::Value memory) const;
  intent::ViewType externalView(mlir::Value memory) const;
  bool isReadOnly(mlir::Value memory) const;
  StorageAliasFacts aliases(mlir::Value memory) const;
  // Requires one lexical release covering every alias use and a complete alias
  // closure. Representation, access and effect legality remain caller duties.
  std::optional<StorageLifetime> lifetime(mlir::memref::AllocOp allocation) const;
  StorageEffects effects(mlir::Operation *scope) const;
  bool disjoint(mlir::Value first, mlir::Value second);
  bool preserves(mlir::Operation *scope, mlir::Value memory);

  // Observe the value after 'from' and before 'to'. Neither endpoint's effects
  // participate; both endpoints must belong to the same lexical sequence.
  bool unchangedBetween(mlir::Value memory, mlir::Operation *from,
                        mlir::Operation *to);
  // Replay a read inside 'from' at 'to': the complete producer and enclosing
  // consumer are checked, including effects in their nested execution regions.
  bool readStable(mlir::Value memory, mlir::Operation *from,
                  mlir::Operation *to);

private:
  bool isLiveAt(mlir::Value memory, mlir::Operation *operation) const;
  mlir::func::FuncOp function;
  mlir::BufferViewFlowAnalysis flow;
  mlir::BufferOriginAnalysis bufferOrigins;
  mlir::AliasAnalysis aliasAnalysis;
  mlir::DominanceInfo dominance;
};

} // namespace intent::cpu
#endif
