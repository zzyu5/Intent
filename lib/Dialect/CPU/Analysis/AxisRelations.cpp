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
    } else if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
      equate(copy.getSource(), copy.getTarget());
    } else if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
      llvm::DenseMap<unsigned, unsigned> dimensions;
      for (auto [input, map] : llvm::zip(generic->getOperands(), generic.getIndexingMapsArray())) {
        if (!isa<MemRefType>(input.getType())) continue;
        for (auto [position, expression] : llvm::enumerate(map.getResults())) {
          auto dim = dyn_cast<AffineDimExpr>(expression);
          if (!dim) continue;
          unsigned axis = positions.at(input)[position];
          auto [it, inserted] = dimensions.try_emplace(dim.getPosition(), axis);
          if (!inserted) unite(it->second, axis);
        }
      }
    } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      std::optional<unsigned> reduction;
      for (auto [input, attribute] : llvm::zip(reduce.getInputs(), reduce.getIndexingMaps())) {
        if (!isa<MemRefType>(input.getType())) continue;
        auto map = mlir::cast<AffineMapAttr>(attribute).getValue();
        for (auto [position, expression] : llvm::enumerate(map.getResults())) {
          if (!isa<AffineDimExpr>(expression)) continue;
          unsigned axis = positions.at(input)[position];
          if (reduction) unite(*reduction, axis);
          else reduction = axis;
        }
      }
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
