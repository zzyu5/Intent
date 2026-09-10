#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "../../../Dialect/CPU/Transforms/Utilities.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::mojo {
using namespace intent::cpu;
namespace {

struct LocalEpilogue {
  memref::AllocaOp allocation;
  linalg::FillOp initialization;
  linalg::GenericOp consumer;
};

std::optional<LocalEpilogue> localEpilogue(linalg::GenericOp contraction) {
  Value partial = contraction.getOutputs()[0];
  auto allocation = partial.getDefiningOp<memref::AllocaOp>();
  auto initialization = dyn_cast_or_null<linalg::FillOp>(contraction->getPrevNode());
  auto consumer = dyn_cast_or_null<linalg::GenericOp>(contraction->getNextNode());
  if (!allocation || !initialization || !consumer ||
      initialization.getOutputs().size() != 1 || initialization.getOutputs()[0] != partial ||
      consumer.getOutputs().size() != 1 || consumer.getNumResults() ||
      consumer.getOutputs()[0] == partial || !llvm::is_contained(consumer.getInputs(), partial) ||
      consumer.getNumReductionLoops() ||
      !llvm::all_of(consumer.getIndexingMapsArray(), [](AffineMap map) { return map.isIdentity(); }) ||
      !consumer.getRegion().front().getArguments().back().use_empty()) return std::nullopt;
  for (Operation *user : partial.getUsers())
    if (user != initialization && user != contraction && user != consumer) return std::nullopt;
  auto shape = cast<MemRefType>(partial.getType()).getShape();
  PhysicalProgramAnalysis analysis(contraction->getParentOfType<func::FuncOp>());
  Value destination = consumer.getOutputs()[0];
  for (Value memory : consumer->getOperands()) {
    auto type = dyn_cast<MemRefType>(memory.getType());
    if (!type || type.getShape() != shape || !type.getElementType().isF32()) return std::nullopt;
    if (memory != destination && analysis.storageRoot(memory) == analysis.storageRoot(destination))
      return std::nullopt;
  }
  for (Operation &operation : consumer.getRegion().front().without_terminator())
    if (operation.getNumRegions() || operation.getNumResults() != 1 ||
        !operation.getResult(0).getType().isF32() || !isMemoryEffectFree(&operation)) return std::nullopt;
  return LocalEpilogue{allocation, initialization, consumer};
}

}

LogicalResult materializeRegisterContractions(func::FuncOp function) {
  SmallVector<linalg::GenericOp> operations;
  SmallVector<scf::ForOp> reductions;
  function.walk([&](linalg::GenericOp operation) {
    if (operation->hasAttr("intent_cpu.microtile")) operations.push_back(operation);
  });
  for (auto operation : operations) {
    auto tile = operation->getAttrOfType<MicrotileAttr>("intent_cpu.microtile");
    auto outputType = cast<MemRefType>(operation.getOutputs()[0].getType());
    if (!isMatrixContraction(operation) || outputType.getShape() !=
        ArrayRef<int64_t>({tile.getRows(), tile.getColumns()}))
      return operation.emitError("CPU register contraction does not match its formed microtile");
    Value lhs = operation.getInputs()[0], rhs = operation.getInputs()[1];
    Value output = operation.getOutputs()[0];
    auto epilogue = localEpilogue(operation);
    OpBuilder b(operation);
    Location loc = operation.getLoc();
    int64_t width = tile.getVectorWidth(), columns = tile.getColumns() / width;
    auto vectorType = VectorType::get({width}, b.getF32Type());
    SmallVector<Value> rows, offsets, accumulators;
    for (int64_t row = 0; row < tile.getRows(); ++row) rows.push_back(index(b, loc, row));
    for (int64_t column = 0; column < columns; ++column) offsets.push_back(index(b, loc, column * width));
    auto load = [&](Value source, Value row, Value column) -> Value {
      if (width == 1) return b.create<memref::LoadOp>(loc, source, ValueRange{row, column});
      return b.create<vector::LoadOp>(loc, vectorType, source, ValueRange{row, column});
    };
    for (Value row : rows)
      for (Value offset : offsets) {
        if (!epilogue) accumulators.push_back(load(output, row, offset));
        else {
          Value initial = epilogue->initialization.getInputs()[0];
          accumulators.push_back(width == 1 ? initial
              : Value(b.create<vector::BroadcastOp>(loc, vectorType, initial)));
        }
      }
    Value depth = b.create<memref::DimOp>(loc, lhs, 1);
    auto reduction = b.create<scf::ForOp>(loc, index(b, loc, 0), depth, index(b, loc, 1), accumulators);
    reductions.push_back(reduction);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(reduction.getBody());
      Value k = reduction.getInductionVar();
      if (width > 1) {
        Value last = b.create<arith::SubIOp>(loc, depth, index(b, loc, 1));
        Value ahead = b.create<arith::MinSIOp>(loc, add(b, loc, k, index(b, loc, 4)), last);
        for (Value offset : offsets)
          b.create<memref::PrefetchOp>(loc, rhs, ValueRange{ahead, offset}, false, 3, true);
      }
      SmallVector<Value> right;
      for (Value offset : offsets) right.push_back(load(rhs, k, offset));
      SmallVector<Value> next;
      for (auto [number, row] : llvm::enumerate(rows)) {
        Value left = b.create<memref::LoadOp>(loc, lhs, ValueRange{row, k});
        if (width != 1) left = b.create<vector::BroadcastOp>(loc, vectorType, left);
        for (int64_t column = 0; column < columns; ++column)
          next.push_back(b.create<math::FmaOp>(loc, left, right[column],
              reduction.getRegionIterArgs()[number * columns + column]));
      }
      b.create<scf::YieldOp>(loc, next);
    }
    for (auto [number, row] : llvm::enumerate(rows))
      for (int64_t column = 0; column < columns; ++column) {
        Value value = reduction.getResult(number * columns + column);
        Value destination = output;
        if (epilogue) {
          IRMapping mapping;
          Block &body = epilogue->consumer.getRegion().front();
          for (auto [position, input] : llvm::enumerate(epilogue->consumer.getInputs()))
            mapping.map(body.getArgument(position), input == output ? value : load(input, row, offsets[column]));
          for (Operation &nested : body.without_terminator()) {
            if (auto constant = dyn_cast<arith::ConstantOp>(&nested); constant && width != 1) {
              Value splat = b.create<arith::ConstantOp>(loc, vectorType,
                  DenseElementsAttr::get(vectorType, cast<FloatAttr>(constant.getValue())));
              mapping.map(constant.getResult(), splat);
            } else {
              Operation *cloned = b.clone(nested, mapping);
              if (width != 1) cloned->getResult(0).setType(vectorType);
            }
          }
          value = mapping.lookup(body.getTerminator()->getOperand(0));
          destination = epilogue->consumer.getOutputs()[0];
        }
        if (width == 1) b.create<memref::StoreOp>(loc, value, destination, ValueRange{row, offsets[column]});
        else b.create<vector::StoreOp>(loc, value, destination, ValueRange{row, offsets[column]});
      }
    operation.erase();
    if (epilogue) {
      epilogue->consumer.erase();
      epilogue->initialization.erase();
      epilogue->allocation.erase();
    }
  }
  for (scf::ForOp reduction : reductions)
    if (failed(mlir::loopUnrollByFactor(reduction, 4)))
      return function.emitError("CPU register contraction loop cannot realize its adjacent four-step issue group");
  return success();
}

}
