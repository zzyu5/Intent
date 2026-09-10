#include "Intent/Dialect/CPU/Analysis/AxisRelations.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace intent::cpu {

unsigned AxisRelations::root(unsigned item) {
  if (parents[item] != item) parents[item] = root(parents[item]);
  return parents[item];
}

void AxisRelations::unite(unsigned lhs, unsigned rhs) {
  parents[root(rhs)] = root(lhs);
}

void AxisRelations::equate(Value lhs, Value rhs) {
  if (!isa<MemRefType>(lhs.getType()) || !isa<MemRefType>(rhs.getType())) return;
  for (auto [left, right] : llvm::zip_equal(positions.at(lhs), positions.at(rhs))) unite(left, right);
}

AxisRelations::AxisRelations(func::FuncOp function) {
  SmallVector<Value> memories;
  auto add = [&](Value value) {
    auto memory = dyn_cast<MemRefType>(value.getType());
    if (!memory || positions.count(value)) return;
    memories.push_back(value);
    auto &axes = positions[value];
    for (int64_t axis = 0; axis < memory.getRank(); ++axis) {
      unsigned identity = parents.size();
      parents.push_back(identity);
      axes.push_back(identity);
    }
  };
  function.walk([&](Operation *operation) {
    for (Value result : operation->getResults()) add(result);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments()) add(argument);
  });
  function.walk([&](Operation *operation) {
    if (auto view = dyn_cast<memref::SubViewOp>(operation)) {
      unsigned position = 0;
      auto dropped = view.getDroppedDims();
      for (auto [axis, source] : llvm::enumerate(positions.at(view.getSource())))
        if (!dropped.test(axis)) unite(source, positions.at(view.getResult())[position++]);
    } else if (auto cast = dyn_cast<memref::CastOp>(operation)) {
      equate(cast.getSource(), cast.getResult());
    } else if (auto tasks = dyn_cast<TasksOp>(operation)) {
      for (auto [argument, capture] : llvm::zip(tasks.getBody().front().getArguments().drop_front(), tasks.getCaptures()))
        equate(argument, capture);
    } else if (auto quantize = dyn_cast<QuantizeOp>(operation)) {
      unite(positions.at(quantize.getInput())[0], positions.at(quantize.getOutput())[0]);
    } else if (auto dot = dyn_cast<QuantizedDotOp>(operation)) {
      unite(positions.at(dot.getLhs())[0], positions.at(dot.getRhs())[0]);
    }
  });
  auto identify = [&](unsigned position) {
    unsigned representative = root(position);
    if (!identities.count(representative)) identities[representative] = identities.size() + 1;
  };
  for (BlockArgument argument : function.getArguments()) {
    auto memory = dyn_cast<MemRefType>(argument.getType());
    if (!memory) continue;
    for (auto [axis, position] : llvm::enumerate(positions.at(argument)))
      if (memory.isDynamicDim(axis)) identify(position);
  }
  dynamicAxes = identities.size();
  for (Value memory : memories)
    for (unsigned position : positions.at(memory)) identify(position);
}

SmallVector<int64_t> AxisRelations::axes(Value memory) {
  SmallVector<int64_t> result;
  for (unsigned position : positions.at(memory)) result.push_back(identities.at(root(position)));
  return result;
}

}
