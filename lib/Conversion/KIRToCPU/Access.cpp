#include "Construction.h"
#include "Intent/Analysis/ProductSchema.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include <functional>

using namespace mlir;

namespace intent::kir_to_cpu {

LogicalResult Construction::verifyIndexTerms(Operation *operation, const IndexRelationFact &fact) {
  for (const auto &term : fact.terms) {
    if (term.kind == 3 && term.operands.size() == 1) continue;
    if (term.kind == 4) {
      if (term.operands.size() != 1 || !domains.count(term.operands[0]))
        return operation->emitError("CPU indexed access requires a realized domain");
    } else if (term.kind != 0 && term.kind != 1 && term.kind != 2) {
      return operation->emitError("CPU indexed access does not implement this coordinate term");
    }
  }
  return success();
}

SmallVector<Value> Construction::indexedCoordinates(const IndexRelationFact &fact,
                                     Value source, ValueRange members, OpBuilder &nested, Location loc) {
  SmallVector<Value> coordinates;
  for (const auto &term : fact.terms) {
    if (term.kind == 1) continue;
    Value coordinate;
    if (term.kind == 0 || term.kind == 4) {
      coordinate = members[term.resultAxes.front()];
      if (term.kind == 4) {
        Domain domain = domains.lookup(term.operands[0]);
        coordinate = nested.createOrFold<arith::AddIOp>(loc, domain.begin,
            nested.createOrFold<arith::MulIOp>(loc, coordinate, domain.step));
      }
    } else if (term.kind == 2) {
      int64_t literal = *term.staticValues[0];
      coordinate = nested.create<arith::ConstantIndexOp>(loc, literal);
      if (literal < 0) coordinate = nested.create<arith::AddIOp>(loc,
          dimension(nested, loc, source, *term.sourceAxis), coordinate);
    } else {
      coordinate = values.lookup(term.operands[0]);
      if (auto type = dyn_cast<ShapedType>(coordinate.getType())) {
        SmallVector<Value> indices;
        for (int64_t axis = 0; axis < type.getRank(); ++axis)
          indices.push_back(type.getDimSize(axis) == 1
              ? Value(nested.create<arith::ConstantIndexOp>(loc, 0))
              : members[term.indexAxes[axis]]);
        coordinate = extractElement(nested, loc, coordinate, indices);
      }
      if (!coordinate.getType().isIndex()) {
        auto logical = cast<IntegerType>(getElementTypeOrSelf(term.operands[0].getType()));
        coordinate = logical.isUnsigned()
            ? Value(nested.create<arith::IndexCastUIOp>(loc, nested.getIndexType(), coordinate))
            : Value(nested.create<arith::IndexCastOp>(loc, nested.getIndexType(), coordinate));
      }
    }
    coordinates.push_back(coordinate);
  }
  return coordinates;
}

LogicalResult Construction::indexedWrite(Operation *operation) {
  auto fact = analysis.indexRelation(operation);
  if (failed(fact)) return failure();
  if (failed(verifyIndexTerms(operation, *fact))) return failure();
  auto access = cast<IndexedAccessOpInterface>(operation);
  Value input = values.lookup(access.getStoredValue()), destination = values.lookup(fact->source);
  Location loc = operation->getLoc();
  SmallVector<Value> sizes, members;
  if (auto type = dyn_cast<RankedTensorType>(input.getType()))
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      sizes.push_back(dimension(builder, loc, input, axis));
  std::function<void(unsigned)> traverse = [&](unsigned axis) {
    if (axis == sizes.size()) {
      auto coordinates = indexedCoordinates(*fact, destination, members, builder, loc);
      Value value = elementAt(input, members, builder, loc);
      builder.create<memref::StoreOp>(loc, value, destination, coordinates);
      return;
    }
    auto loop = builder.create<scf::ForOp>(loc, constant(loc, 0), sizes[axis], constant(loc, 1));
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    members.push_back(loop.getInductionVar());
    traverse(axis + 1);
    members.pop_back();
  };
  traverse(0);
  return success();
}

FailureOr<Value> Construction::indexedRead(Operation *operation, const IndexRelationFact &fact, Value source) {
  auto access = cast<IndexedAccessOpInterface>(operation);
  Type resultType = operation->getResult(0).getType();
  auto tensor = dyn_cast<RankedTensorType>(resultType);
  if (failed(verifyIndexTerms(operation, fact))) return failure();
  auto read = [&](OpBuilder &nested, Location loc, ValueRange members) -> Value {
    auto load = [&]() -> Value {
      auto coordinates = indexedCoordinates(fact, source, members, nested, loc);
      return extractElement(nested, loc, source, coordinates);
    };
    if (alwaysValid(operation)) return load();
    Value active = elementAt(values.lookup(access.getAccessValidity()), members, nested, loc);
    auto conditional = nested.create<scf::IfOp>(loc, TypeRange{getElementTypeOrSelf(resultType)}, active, true);
    {
      OpBuilder::InsertionGuard guard(nested);
      nested.setInsertionPointToStart(conditional.thenBlock());
      nested.create<scf::YieldOp>(loc, load());
      nested.setInsertionPointToStart(conditional.elseBlock());
      Value fill = elementAt(values.lookup(access.getAccessFill()), members, nested, loc);
      nested.create<scf::YieldOp>(loc, fill);
    }
    return conditional.getResult(0);
  };
  if (!tensor) return read(builder, operation->getLoc(), {});
  auto shape = extents(operation->getResult(0), operation->getLoc());
  if (failed(shape)) return failure();
  Value output = emptyTensor(tensor, *shape, operation->getLoc());
  auto result = builder.create<linalg::GenericOp>(operation->getLoc(), TypeRange{output.getType()}, ValueRange{}, ValueRange{output},
      SmallVector<AffineMap>{builder.getMultiDimIdentityMap(tensor.getRank())},
      SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
      [&](OpBuilder &nested, Location loc, ValueRange) {
        SmallVector<Value> members;
        for (int64_t axis = 0; axis < tensor.getRank(); ++axis)
          members.push_back(nested.create<linalg::IndexOp>(loc, axis));
        nested.create<linalg::YieldOp>(loc, read(nested, loc, members));
      });
  return result.getResult(0);
}

LogicalResult Construction::atomicAccess(Operation *operation) {
  auto access = cast<IndexedAccessOpInterface>(operation);
  auto fact = analysis.indexRelation(operation);
  if (failed(fact)) return failure();
  if (failed(verifyIndexTerms(operation, *fact))) return failure();
  Location loc = operation->getLoc();
  Value target = values.lookup(fact->source);
  Type element = cast<MemRefType>(target.getType()).getElementType();
  auto ordering = operation->getAttrOfType<AtomicOrderingAttr>("ordering");
  SmallVector<Type> resultTypes;
  appendProductLeafTypes(operation->getResultTypes(), resultTypes);
  Type accessType = resultTypes.empty()
      ? access.getStoredValue().getType()
      : resultTypes.front();
  auto tensor = dyn_cast<RankedTensorType>(accessType);
  SmallVector<Value> sizes, members, outputs;
  bool used = llvm::any_of(operation->getResults(), [](Value value) { return !value.use_empty(); });
  if (tensor) {
    Value shapeSource = operation->getNumResults()
        ? operation->getResult(0) : access.getStoredValue();
    SmallVector<unsigned, 2> path;
    bool first = true;
    walkProductLeaves(shapeSource.getType(), [&](Type, ArrayRef<unsigned> fieldPath) {
      if (!first) return;
      path.assign(fieldPath.begin(), fieldPath.end());
      first = false;
    });
    auto extentsOr = extents(shapeSource, loc, path);
    if (failed(extentsOr)) return failure();
    sizes = *extentsOr;
    if (used)
      for (Type result : resultTypes)
        outputs.push_back(emptyTensor(cast<RankedTensorType>(result), sizes, loc));
  }
  auto operand = [&](Value value) {
    return elementAt(values.lookup(value), members, builder, loc);
  };
  std::function<SmallVector<Value>(unsigned, ValueRange)> traverse =
      [&](unsigned axis, ValueRange carried) -> SmallVector<Value> {
    if (axis < sizes.size()) {
      auto loop = builder.create<scf::ForOp>(loc, constant(loc, 0), sizes[axis], constant(loc, 1), carried);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      members.push_back(loop.getInductionVar());
      auto results = traverse(axis + 1, loop.getRegionIterArgs());
      members.pop_back();
      if (!loop.getBody()->empty() &&
          loop.getBody()->back().hasTrait<OpTrait::IsTerminator>())
        loop.getBody()->back().erase();
      builder.setInsertionPointToEnd(loop.getBody());
      builder.create<scf::YieldOp>(loc, results);
      return SmallVector<Value>(loop.getResults().begin(), loop.getResults().end());
    }
    auto coordinates = indexedCoordinates(*fact, target, members, builder, loc);
    SmallVector<Value> results;
    if (isa<AtomicLoadOp>(operation)) {
      results.push_back(builder.create<cpu::AtomicLoadOp>(loc, element, target, coordinates, ordering));
    } else if (isa<AtomicStoreOp>(operation)) {
      builder.create<cpu::AtomicStoreOp>(loc, target, operand(access.getStoredValue()), coordinates, ordering);
    } else if (isa<AtomicRMWOp>(operation)) {
      results.push_back(builder.create<cpu::AtomicRMWOp>(loc, element, target, operand(access.getStoredValue()),
          coordinates, ordering, operation->getAttrOfType<AtomicRMWKindAttr>("kind"),
          builder.getBoolAttr(isa<IntegerType>(element) && cast<IntegerType>(element).isUnsigned())));
    } else {
      auto exchange = builder.create<cpu::AtomicCompareExchangeOp>(loc, element, builder.getI1Type(),
          target, operand(access.getCompareValue()), operand(access.getReplacementValue()), coordinates, ordering);
      llvm::append_range(results, exchange.getResults());
    }
    if (!used) return {};
    if (!tensor) {
      bindValues(operation->getResults(), results);
      return {};
    }
    SmallVector<Value> updated;
    for (auto [value, output] : llvm::zip(results, carried))
      updated.push_back(builder.create<tensor::InsertOp>(loc, value, output, members));
    return updated;
  };
  auto results = traverse(0, outputs);
  if (used && tensor) bindValues(operation->getResults(), results);
  return success();
}

LogicalResult Construction::scatterReduce(ScatterReduceOp operation) {
  auto fact = analysis.indexRelation(operation);
  if (failed(fact)) return failure();
  if (failed(verifyIndexTerms(operation, *fact))) return failure();
  Location loc = operation.getLoc();
  Value target = values.lookup(fact->source);
  Value input = values.lookup(operation.getValue());
  SmallVector<Value> sizes, members;
  if (auto memory = dyn_cast<RankedTensorType>(input.getType()))
    for (int64_t axis = 0; axis < memory.getRank(); ++axis)
      sizes.push_back(dimension(builder, loc, input, axis));
  std::function<LogicalResult(unsigned)> traverse = [&](unsigned axis) -> LogicalResult {
    if (axis < sizes.size()) {
      auto loop = builder.create<scf::ForOp>(loc, constant(loc, 0), sizes[axis], constant(loc, 1));
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      members.push_back(loop.getInductionVar());
      auto status = traverse(axis + 1);
      members.pop_back();
      return status;
    }
    Value value = elementAt(input, members, builder, loc);
    auto coordinates = indexedCoordinates(*fact, target, members, builder, loc);
    auto update = builder.create<memref::GenericAtomicRMWOp>(loc, target, coordinates);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&update.getRegion().front());
    Block &combine = operation.getCombine().front();
    values.map(combine.getArgument(0), update.getCurrentValue());
    values.map(combine.getArgument(1), value);
    for (Operation &instruction : combine.without_terminator())
      if (failed(lowerOperation(&instruction))) return failure();
    builder.create<memref::AtomicYieldOp>(loc, values.lookup(combine.getTerminator()->getOperand(0)));
    return success();
  };
  return traverse(0);
}

FailureOr<Value> Construction::indexed(Operation *operation) {
  auto fact = analysis.indexRelation(operation);
  if (failed(fact))
    return failure();
  Value source = values.lookup(fact->source);
  if (isa<ViewLoadOp, BufferLoadOp, GatherOp>(operation) && (!alwaysValid(operation) || llvm::any_of(fact->terms, [&](const auto &term) {
        return term.kind == 3 && term.operands.size() == 1 &&
            isa<RankedTensorType>(values.lookup(term.operands[0]).getType());
      })))
    return indexedRead(operation, *fact, source);
  auto type = cast<ShapedType>(source.getType());
  SmallVector<OpFoldResult> offsets, sizes, strides;
  SmallVector<int64_t> resultShape;
  SmallVector<AffineExpr> projection;
  bool inserted = false;
  for (const IndexTermFact &term : fact->terms) {
    OpFoldResult stride = builder.getIndexAttr(1);
    if (term.kind == 1) {
      inserted = true;
      continue;
    } else if (term.kind == 0) {
      unsigned axis = *term.sourceAxis;
      offsets.push_back(builder.getIndexAttr(0));
      sizes.push_back(type.isDynamicDim(axis) ? OpFoldResult(dimension(builder, operation->getLoc(), source, axis)) : builder.getIndexAttr(type.getDimSize(axis)));
      resultShape.push_back(type.getDimSize(axis));
      projection.push_back(builder.getAffineDimExpr(term.resultAxes.front()));
    } else if (term.kind == 3 && term.operands.size() == 1) {
      auto coordinate = indexValue(values.lookup(term.operands[0]), term.operands[0].getType(), operation->getLoc());
      if (failed(coordinate)) return failure();
      offsets.push_back(*coordinate);
      sizes.push_back(builder.getIndexAttr(1));
    } else if (term.kind == 2 && !term.staticValues.empty() && term.staticValues[0]) {
      int64_t literal = *term.staticValues[0];
      if (literal < 0) offsets.push_back(builder.create<arith::AddIOp>(operation->getLoc(),
          dimension(builder, operation->getLoc(), source, *term.sourceAxis),
          constant(operation->getLoc(), literal)).getResult());
      else offsets.push_back(builder.getIndexAttr(literal));
      sizes.push_back(builder.getIndexAttr(1));
    } else if (term.kind == 4 && term.operands.size() == 1 && domains.count(term.operands[0])) {
      Domain domain = domains.lookup(term.operands[0]);
      offsets.push_back(domain.begin);
      stride = domain.step;
      projection.push_back(builder.getAffineDimExpr(term.resultAxes.front()));
      llvm::APInt extent;
      if (matchPattern(domain.extent, m_ConstantInt(&extent))) {
        sizes.push_back(builder.getIndexAttr(extent.getSExtValue()));
        resultShape.push_back(extent.getSExtValue());
      } else {
        sizes.push_back(domain.extent);
        resultShape.push_back(ShapedType::kDynamic);
      }
    } else {
      operation->emitError("CPU construction does not implement this index relation");
      return failure();
    }
    strides.push_back(stride);
  }
  if (offsets.size() != static_cast<size_t>(type.getRank())) {
    operation->emitError("CPU indexed access must resolve every source axis");
    return failure();
  }
  if (resultShape.empty() && !inserted) {
    SmallVector<Value> indices;
    for (OpFoldResult offset : offsets)
      indices.push_back(isa<Value>(offset) ? cast<Value>(offset)
                         : constant(operation->getLoc(), cast<IntegerAttr>(cast<Attribute>(offset)).getInt()));
    if (isa<ViewLoadOp, BufferLoadOp, GatherOp>(operation) &&
        !isa<RankedTensorType>(operation->getResult(0).getType()))
      return extractElement(builder, operation->getLoc(), source, indices);
  }
  Value selected;
  if (isa<RankedTensorType>(type)) {
    auto resultType = RankedTensorType::get(resultShape, type.getElementType());
    selected = builder.create<tensor::ExtractSliceOp>(operation->getLoc(),
        resultType, source, offsets, sizes, strides);
  } else {
    auto resultType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        resultShape, cast<MemRefType>(type), offsets, sizes, strides));
    selected = builder.create<memref::SubViewOp>(operation->getLoc(), resultType,
                                                source, offsets, sizes, strides);
  }
  if (!inserted) {
    if (isa<ViewLoadOp, BufferLoadOp, GatherOp>(operation) &&
        isa<MemRefType>(selected.getType()))
      return Value(builder.create<cpu::ReadOp>(operation->getLoc(),
          tensorType(operation->getResult(0).getType()), selected));
    return selected;
  }
  if (operation->getNumResults() != 1 || !isa<RankedTensorType>(operation->getResult(0).getType()))
    return operation->emitError("CPU inserted write axes are not implemented"), failure();
  auto tensor = cast<RankedTensorType>(operation->getResult(0).getType());
  auto shape = extents(operation->getResult(0), operation->getLoc());
  if (failed(shape)) return failure();
  Value output = emptyTensor(tensor, *shape, operation->getLoc());
  if (isa<MemRefType>(selected.getType()))
    selected = builder.create<cpu::ReadOp>(operation->getLoc(),
        RankedTensorType::get(resultShape, type.getElementType()), selected);
  auto result = builder.create<linalg::GenericOp>(operation->getLoc(), TypeRange{output.getType()}, ValueRange{selected}, ValueRange{output},
      SmallVector<AffineMap>{AffineMap::get(tensor.getRank(), 0, projection, builder.getContext()), builder.getMultiDimIdentityMap(tensor.getRank())},
      SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
      [](OpBuilder &nested, Location loc, ValueRange scalars) { nested.create<linalg::YieldOp>(loc, scalars[0]); });
  return result.getResult(0);
}

bool Construction::alwaysValid(Operation *operation) {
  Value predicate = cast<IndexedAccessOpInterface>(operation).getAccessValidity();
  if (!predicate) return true;
  while (auto producer = predicate.getDefiningOp()) {
    if (auto literal = dyn_cast<ConstantOp>(producer)) {
      auto value = dyn_cast<IntegerAttr>(literal.getValue());
      return value && value.getType().isInteger(1) && !value.getValue().isZero();
    }
    if (isa<FullOp, BroadcastOp, ReshapeOp>(producer)) predicate = producer->getOperand(0);
    else return false;
  }
  return false;
}

LogicalResult Construction::lower(BufferOp op) {
  Location loc = op.getLoc();
  if (!analysis.logicalBuffer(op).isExact())
    return op.emitError("CPU logical buffer requires an exact lexical allocation fact");
  auto tensor = cast<RankedTensorType>(cast<BufferType>(op.getResult().getType()).getTensor());
  auto sizes = extents(op.getResult(), loc);
  if (failed(sizes)) return failure();
  SmallVector<Value> dynamic;
  for (auto [axis, size] : llvm::enumerate(*sizes))
    if (tensor.isDynamicDim(axis)) dynamic.push_back(size);
  Value storage = builder.create<memref::AllocOp>(loc,
      MemRefType::get(tensor.getShape(), tensor.getElementType()), dynamic);
  if (op.getInitial()) {
    Value initial = values.lookup(op.getInitial());
    if (isa<RankedTensorType>(initial.getType()))
      builder.create<cpu::WriteOp>(loc, initial, storage);
    else builder.create<linalg::FillOp>(loc, ValueRange{initial}, ValueRange{storage});
  }
  values.map(op.getResult(), storage);
  return success();
}

LogicalResult Construction::load(Operation *operation) {
  auto value = indexed(operation);
  if (failed(value)) return failure();
  values.map(operation->getResult(0), *value);
  return success();
}

LogicalResult Construction::store(Operation *operation) {
  Location loc = operation->getLoc();
  if (!alwaysValid(operation))
    return operation->emitError("CPU predicated destination stores are not implemented");
  auto fact = analysis.indexRelation(operation);
  if (failed(fact)) return failure();
  if (isa<ScatterUniqueOp>(operation) || llvm::any_of(fact->terms, [&](const auto &term) {
        return term.kind == 1 || (term.kind == 3 && term.operands.size() == 1 &&
                                 isa<RankedTensorType>(values.lookup(term.operands[0]).getType()));
      })) return indexedWrite(operation);
  auto destination = indexed(operation);
  if (failed(destination)) return failure();
  Value input = values.lookup(cast<IndexedAccessOpInterface>(operation).getStoredValue());
  if (isa<RankedTensorType>(input.getType()))
    builder.create<cpu::WriteOp>(loc, input, *destination);
  else
    builder.create<memref::StoreOp>(loc, input, *destination, ValueRange{});
  return success();
}

} // namespace intent::kir_to_cpu
