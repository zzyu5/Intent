#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Utilities.h"
#include "VectorReductions.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::cpu {
namespace {

SmallVector<Value> coordinates(OpBuilder &b, Location loc, AffineMap map,
                               ValueRange domain) {
  SmallVector<Value> result;
  for (AffineExpr expression : map.getResults()) {
    if (auto axis = dyn_cast<AffineDimExpr>(expression)) result.push_back(domain[axis.getPosition()]);
    else result.push_back(index(b, loc, cast<AffineConstantExpr>(expression).getValue()));
  }
  return result;
}

bool supportedMap(AffineMap map) {
  return !map.getNumSymbols() && llvm::all_of(map.getResults(), [](AffineExpr expression) {
    return isa<AffineDimExpr, AffineConstantExpr>(expression);
  });
}

bool materializeProductReduction(linalg::GenericOp operation) {
  auto order = operation->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
  auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
  auto widthAttr = binding ? dyn_cast_or_null<IntegerAttr>(binding.getParameters().get("vector_width")) : IntegerAttr{};
  int64_t width = widthAttr ? widthAttr.getInt() : 0;
  if (!order || !order.getAdjacentReassociation() || width <= 1)
    return false;
  if (width & (width - 1)) return false;
  unsigned components = operation.getOutputs().size(), inputs = operation.getInputs().size();
  // Multi-output generics retain the source/combine boundary established by
  // structured reduction construction; single-output producer fusion does not.
  if (components < 2 || inputs < components || !operation.getNumLoops()) return false;
  auto iterators = operation.getIteratorTypesArray();
  if (iterators.back() != utils::IteratorType::reduction) return false;
  auto maps = operation.getIndexingMapsArray();
  auto sourceType = dyn_cast<MemRefType>(operation.getInputs()[0].getType());
  if (!sourceType || sourceType.getRank() != operation.getNumLoops()) return false;
  auto scalarType = [](Type type) { return isa<FloatType, IntegerType, IndexType>(type); };
  SmallVector<AffineExpr> freeAxes;
  SmallVector<int64_t> outputShape;
  for (unsigned axis = 0; axis + 1 < operation.getNumLoops(); ++axis)
    if (iterators[axis] == utils::IteratorType::parallel) {
      freeAxes.push_back(getAffineDimExpr(axis, operation.getContext()));
      outputShape.push_back(sourceType.getDimSize(axis));
    }
  auto outputMap = AffineMap::get(operation.getNumLoops(), 0, freeAxes, operation.getContext());
  SmallVector<bool> vectorLoads(components, false);
  for (unsigned component = 0; component < components; ++component) {
    auto source = dyn_cast<MemRefType>(operation.getInputs()[component].getType());
    auto output = dyn_cast<MemRefType>(operation.getOutputs()[component].getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    if (!source || !output || source.getShape() != sourceType.getShape() ||
        source.getElementType() != output.getElementType() ||
        !scalarType(source.getElementType()) ||
        !maps[component].isIdentity() || maps[inputs + component] != outputMap ||
        output.getShape() != ArrayRef<int64_t>(outputShape) ||
        failed(source.getStridesAndOffset(strides, offset)))
      return false;
    vectorLoads[component] = strides.back() == 1 && !source.getElementType().isInteger(1);
  }
  for (unsigned capture = components; capture < inputs; ++capture)
    for (AffineExpr expression : maps[capture].getResults())
      if (auto dimension = dyn_cast<AffineDimExpr>(expression))
        if (dimension.getPosition() + 1 == operation.getNumLoops()) return false;
  Block &body = operation.getRegion().front();
  if (!llvm::all_of(body.getArgumentTypes(), scalarType) ||
      llvm::any_of(body.getArguments().drop_front(inputs), [](BlockArgument argument) {
        return argument.use_empty();
      })) return false;
  for (Operation &instruction : body.without_terminator())
    if (instruction.getNumRegions() || instruction.getNumResults() != 1 ||
        !scalarType(instruction.getResult(0).getType()) ||
        !llvm::all_of(instruction.getOperandTypes(), scalarType) ||
        !isMemoryEffectFree(&instruction) ||
        (!isa<arith::ConstantOp>(instruction) && !instruction.hasTrait<OpTrait::Elementwise>()))
      return false;
  auto function = operation->getParentOfType<func::FuncOp>();
  bool accumulatePartials = order.getElementPermutation();
  PhysicalProgramAnalysis physical(function);
  AliasAnalysis aliases(function);
  auto abi = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  auto disjoint = [&](Value first, Value second) {
    first = physical.storageRoot(first);
    second = physical.storageRoot(second);
    if (first == second) return false;
    if (aliases.alias(first, second).isNo()) return true;
    auto left = physical.externalView(first), right = physical.externalView(second);
    return abi && abi.getDisjointOutputs() && left && right &&
        (left.getAccess() != 0 || right.getAccess() != 0);
  };
  for (auto [component, output] : llvm::enumerate(operation.getOutputs())) {
    for (Value input : operation.getInputs())
      if (isa<MemRefType>(input.getType()) && !disjoint(input, output)) return false;
    for (Value previous : operation.getOutputs().take_front(component))
      if (!disjoint(previous, output)) return false;
  }

  OpBuilder b(operation);
  Location loc = operation.getLoc();
  SmallVector<Value> sizes, position(sourceType.getRank());
  for (int64_t axis = 0; axis < sourceType.getRank(); ++axis)
    sizes.push_back(b.create<memref::DimOp>(loc, operation.getInputs()[0], axis));
  auto combine = [&](ValueRange left, ValueRange right, ValueRange captures, int64_t lanes) {
    IRMapping mapping;
    for (unsigned component = 0; component < components; ++component) {
      mapping.map(body.getArgument(component), right[component]);
      mapping.map(body.getArgument(inputs + component), left[component]);
    }
    auto broadcast = [&](Value value) -> Value {
      return lanes ? Value(b.create<vector::BroadcastOp>(loc, VectorType::get({lanes}, value.getType()), value)) : value;
    };
    for (auto [number, capture] : llvm::enumerate(captures))
      mapping.map(body.getArgument(components + number), broadcast(capture));
    auto mapped = [&](Value value) {
      if (!mapping.contains(value)) mapping.map(value, broadcast(value));
      return mapping.lookup(value);
    };
    for (Operation &instruction : body.without_terminator()) {
      if (lanes && isa<arith::ConstantOp>(instruction)) {
        mapping.map(instruction.getResult(0), broadcast(b.clone(instruction)->getResult(0)));
        continue;
      }
      for (Value operand : instruction.getOperands()) mapped(operand);
      Operation *cloned = b.clone(instruction, mapping);
      if (lanes) cloned->getResult(0).setType(VectorType::get({lanes}, instruction.getResult(0).getType()));
    }
    SmallVector<Value> result;
    for (Value value : body.getTerminator()->getOperands()) result.push_back(mapped(value));
    return result;
  };
  std::function<void(unsigned)> traverse = [&](unsigned axis) {
    if (axis + 1 < sizes.size()) {
      loop(b, loc, index(b, loc, 0), sizes[axis], 1, [&](Value coordinate) {
        position[axis] = coordinate;
        traverse(axis + 1);
      });
      return;
    }
    Value zero = index(b, loc, 0), one = index(b, loc, 1), extent = sizes.back();
    Value nonempty = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, extent, zero);
    auto active = b.create<scf::IfOp>(loc, nonempty, false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(active.thenBlock());
    position.back() = zero;
    SmallVector<Value> initials, captures;
    auto outputPosition = coordinates(b, loc, outputMap, position);
    for (Value output : operation.getOutputs())
      initials.push_back(b.create<memref::LoadOp>(loc, output, outputPosition));
    for (unsigned number = components; number < inputs; ++number) {
      Value value = operation.getInputs()[number];
      if (isa<MemRefType>(value.getType()))
        value = b.create<memref::LoadOp>(loc, value, coordinates(b, loc, maps[number], position));
      captures.push_back(value);
    }
    Value step = index(b, loc, width);
    Value completeEnd = b.create<arith::SubIOp>(loc, extent, b.create<arith::RemSIOp>(loc, extent, step));
    auto horizontal = [&](ValueRange partial) {
      return horizontalReduce(b, loc, partial, [&](ValueRange left, ValueRange right, int64_t lanes) {
        return combine(left, right, captures, lanes);
      });
    };
    auto loadBlock = [&](Value begin) {
      position.back() = begin;
      SmallVector<Value> partial;
      for (auto [component, source] : llvm::enumerate(operation.getInputs().take_front(components))) {
        auto type = VectorType::get({width}, cast<MemRefType>(source.getType()).getElementType());
        if (vectorLoads[component]) {
          partial.push_back(b.create<vector::LoadOp>(loc, type, source, position));
          continue;
        }
        // Scalar lane reads retain the memref's element representation and
        // strides, including bool storage whose vector packing is target-specific.
        SmallVector<Value> lanes, coordinate(position);
        for (int64_t lane = 0; lane < width; ++lane) {
          coordinate.back() = b.create<arith::AddIOp>(loc, position.back(), index(b, loc, lane));
          lanes.push_back(b.create<memref::LoadOp>(loc, source, coordinate));
        }
        partial.push_back(b.create<vector::FromElementsOp>(loc, type, lanes));
      }
      return partial;
    };
    SmallVector<Value> reduced;
    if (accumulatePartials) {
      // Seed from members, then include the original accumulator exactly once.
      // It need not be an identity after an enclosing reduction is blocked.
      Value hasBlock = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, completeEnd, zero);
      auto full = b.create<scf::IfOp>(loc, TypeRange(initials), hasBlock, true);
      {
        OpBuilder::InsertionGuard fullGuard(b);
        b.setInsertionPointToStart(full.thenBlock());
        SmallVector<Value> seeds = loadBlock(zero);
        auto blocks = b.create<scf::ForOp>(loc, step, completeEnd, step, seeds);
        {
          OpBuilder::InsertionGuard blockGuard(b);
          b.setInsertionPointToStart(blocks.getBody());
          auto partial = loadBlock(blocks.getInductionVar());
          b.create<scf::YieldOp>(loc, combine(blocks.getRegionIterArgs(), partial, captures, width));
        }
        auto partial = horizontal(SmallVector<Value>(blocks.getResults()));
        b.create<scf::YieldOp>(loc, combine(initials, partial, captures, 0));
        b.setInsertionPointToStart(full.elseBlock());
        b.create<scf::YieldOp>(loc, initials);
      }
      reduced.assign(full.getResults().begin(), full.getResults().end());
    } else {
      auto blocks = b.create<scf::ForOp>(loc, zero, completeEnd, step, initials);
      {
        OpBuilder::InsertionGuard blockGuard(b);
        b.setInsertionPointToStart(blocks.getBody());
        auto partial = horizontal(loadBlock(blocks.getInductionVar()));
        b.create<scf::YieldOp>(loc, combine(blocks.getRegionIterArgs(), partial, captures, 0));
      }
      reduced.assign(blocks.getResults().begin(), blocks.getResults().end());
    }
    auto tail = b.create<scf::ForOp>(loc, completeEnd, extent, one, reduced);
    {
      OpBuilder::InsertionGuard tailGuard(b);
      b.setInsertionPointToStart(tail.getBody());
      position.back() = tail.getInductionVar();
      SmallVector<Value> values;
      for (Value source : operation.getInputs().take_front(components))
        values.push_back(b.create<memref::LoadOp>(loc, source, position));
      b.create<scf::YieldOp>(loc, combine(tail.getRegionIterArgs(), values, captures, 0));
    }
    for (auto [value, output] : llvm::zip(tail.getResults(), operation.getOutputs()))
      b.create<memref::StoreOp>(loc, value, output, outputPosition);
  };
  traverse(0);
  operation.erase();
  return true;
}

void collapseProductReductionAxes(linalg::GenericOp operation) {
  auto maps = operation.getIndexingMapsArray();
  auto order = operation->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
  if (order && order.getElementPermutation() && operation.getOutputs().size() > 1) {
    auto iterators = operation.getIteratorTypesArray();
    SmallVector<ReassociationIndices> groups;
    for (auto [axis, iterator] : llvm::enumerate(iterators)) {
      if (iterator == utils::IteratorType::reduction && axis &&
          iterators[axis - 1] == utils::IteratorType::reduction)
        groups.back().push_back(axis);
      else groups.push_back({static_cast<int64_t>(axis)});
    }
    if (groups.size() < iterators.size() &&
        linalg::areDimSequencesPreserved(maps, groups)) {
      IRRewriter rewriter(operation.getContext());
      rewriter.setInsertionPoint(operation);
      auto collapsed = linalg::collapseOpIterationDims(operation, groups, rewriter);
      if (succeeded(collapsed)) {
        auto replacement = cast<linalg::GenericOp>(collapsed->collapsedOp.getOperation());
        replacement->setDiscardableAttrs(llvm::to_vector(operation->getDiscardableAttrs()));
        operation.erase();
      }
    }
  }
}

LogicalResult materialize(linalg::GenericOp operation) {
  if (operation.getOutputs().empty() || operation.getNumResults())
    return operation.emitError("CPU structured materialization requires destination buffers");
  auto maps = operation.getIndexingMapsArray();
  if (!llvm::all_of(maps, supportedMap))
    return operation.emitError("CPU pointwise coordinate map has no implemented scalar projection");
  if (materializeProductReduction(operation)) return success();
  OpBuilder b(operation);
  Location loc = operation.getLoc();
  SmallVector<Value> sizes(operation.getNumLoops()), position;
  for (auto [memory, map] : llvm::zip(operation->getOperands(), maps)) {
    if (!isa<MemRefType>(memory.getType())) continue;
    for (auto [axis, expression] : llvm::enumerate(map.getResults()))
      if (auto dimension = dyn_cast<AffineDimExpr>(expression))
        if (!sizes[dimension.getPosition()])
          sizes[dimension.getPosition()] = b.create<memref::DimOp>(loc, memory, axis);
  }
  if (llvm::any_of(sizes, [](Value size) { return !size; }))
    return operation.emitError("CPU structured traversal has an unbound loop extent");
  Block &body = operation.getRegion().front();
  auto order = operation->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
  auto iterators = operation.getIteratorTypesArray();
  SmallVector<AffineExpr> freeAxes;
  for (unsigned axis = 0; axis + 1 < sizes.size(); ++axis)
    freeAxes.push_back(b.getAffineDimExpr(axis));
  auto freeMap = AffineMap::get(sizes.size(), 0, freeAxes, b.getContext());
  bool carryReduction = order && operation.getOutputs().size() == 1 && !sizes.empty() &&
      iterators.back() == utils::IteratorType::reduction &&
      llvm::all_of(ArrayRef(iterators).drop_back(), [](utils::IteratorType iterator) {
        return iterator == utils::IteratorType::parallel;
      }) && maps.back() == freeMap && !body.getArguments().back().use_empty() &&
      llvm::all_of(body.without_terminator(), [](Operation &instruction) {
        return !instruction.getNumRegions() && isMemoryEffectFree(&instruction);
      });
  if (carryReduction) {
    auto function = operation->getParentOfType<func::FuncOp>();
    PhysicalProgramAnalysis physical(function);
    AliasAnalysis aliases(function);
    auto abi = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
    Value output = physical.storageRoot(operation.getOutputs()[0]);
    for (Value input : operation.getInputs()) {
      if (!isa<MemRefType>(input.getType())) continue;
      input = physical.storageRoot(input);
      if (input != output && aliases.alias(input, output).isNo()) continue;
      auto source = physical.externalView(input), destination = physical.externalView(output);
      if (input != output && abi && abi.getDisjointOutputs() && source && destination &&
          (source.getAccess() != 0 || destination.getAccess() != 0)) continue;
      carryReduction = false;
      break;
    }
  }
  auto compute = [&](Value carry = {}) {
    IRMapping mapping;
    for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
      Value value = input;
      if (isa<MemRefType>(input.getType()))
        value = b.create<memref::LoadOp>(loc, input, coordinates(b, loc, maps[number], position));
      mapping.map(body.getArgument(number), value);
    }
    for (auto [number, output] : llvm::enumerate(operation.getOutputs())) {
      unsigned argument = operation.getInputs().size() + number;
      if (!body.getArgument(argument).use_empty())
        mapping.map(body.getArgument(argument), carry ? carry : Value(b.create<memref::LoadOp>(
            loc, output, coordinates(b, loc, maps[argument], position))));
    }
    for (Operation &nested : body.without_terminator()) {
      if (auto index = dyn_cast<linalg::IndexOp>(nested)) mapping.map(index.getResult(), position[index.getDim()]);
      else b.clone(nested, mapping);
    }
    SmallVector<Value> results;
    for (Value value : body.getTerminator()->getOperands()) results.push_back(mapping.lookupOrDefault(value));
    return results;
  };
  std::function<void(unsigned)> visit = [&](unsigned axis) {
    if (carryReduction && axis + 1 == sizes.size()) {
      Value zero = index(b, loc, 0), one = index(b, loc, 1);
      auto active = b.create<scf::IfOp>(loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, sizes[axis], zero), false);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(active.thenBlock());
      Value output = operation.getOutputs()[0];
      Value initial = b.create<memref::LoadOp>(loc, output, position);
      auto reduction = b.create<scf::ForOp>(loc, zero, sizes[axis], one, ValueRange{initial});
      reduction->setAttr("intent_cpu.reduction_order", order);
      {
        OpBuilder::InsertionGuard loopGuard(b);
        b.setInsertionPointToStart(reduction.getBody());
        position.push_back(reduction.getInductionVar());
        b.create<scf::YieldOp>(loc, compute(reduction.getRegionIterArgs()[0]));
        position.pop_back();
      }
      b.create<memref::StoreOp>(loc, reduction.getResult(0), output, position);
      return;
    }
    if (axis != sizes.size()) {
      loop(b, loc, index(b, loc, 0), sizes[axis], 1, [&](Value i) {
        position.push_back(i); visit(axis + 1); position.pop_back();
      });
      return;
    }
    auto results = compute();
    for (auto [number, output] : llvm::enumerate(operation.getOutputs()))
      b.create<memref::StoreOp>(loc, results[number], output,
          coordinates(b, loc, maps[operation.getInputs().size() + number], position));
  };
  visit(0);
  operation.erase();
  return success();
}

LogicalResult materialize(ReduceOp operation) {
  OpBuilder b(operation);
  Location loc = operation.getLoc();
  auto reduction = b.create<scf::ForOp>(loc, index(b, loc, 0), operation.getExtent(),
      index(b, loc, 1), ValueRange{operation.getInitial()});
  reduction->setAttr("intent_cpu.reduction_order", operation.getOrder());
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(reduction.getBody());
    IRMapping mapping;
    Block &body = operation.getCombine().front();
    mapping.map(body.getArgument(0), reduction.getRegionIterArgs()[0]);
    for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
      Value value = input;
      if (isa<MemRefType>(input.getType()))
        value = b.create<memref::LoadOp>(loc, input,
            coordinates(b, loc, cast<AffineMapAttr>(operation.getIndexingMaps()[number]).getValue(),
                        ValueRange{reduction.getInductionVar()}));
      mapping.map(body.getArgument(number + 1), value);
    }
    for (Operation &nested : body.without_terminator()) b.clone(nested, mapping);
    b.create<scf::YieldOp>(loc, mapping.lookupOrDefault(cast<YieldOp>(body.getTerminator()).getValue()));
  }
  operation.getResult().replaceAllUsesWith(reduction.getResult(0));
  operation.erase();
  return success();
}

LogicalResult materialize(ScanOp operation, int64_t width) {
  OpBuilder b(operation);
  Location loc = operation.getLoc();
  auto type = cast<MemRefType>(operation.getSources()[0].getType());
  int64_t scanAxis = operation.getAxis();
  if (width > 1 && !supportsVectorScan(operation))
    return operation.emitError("selected vector scan requires an elementwise last-axis scan with supported strides");
  SmallVector<memref::ExtractStridedMetadataOp> descriptors;
  Value contiguous;
  if (width > 1)
    for (Value memory : llvm::concat<const Value>(operation.getSources(), operation.getOutputs())) {
      auto memoryType = cast<MemRefType>(memory.getType());
      auto [strides, offset] = memoryType.getStridesAndOffset();
      if (!ShapedType::isDynamic(strides.back())) continue;
      auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, memory);
      descriptors.push_back(metadata);
      Value unit = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
          metadata.getStrides().back(), index(b, loc, 1));
      contiguous = contiguous ? Value(b.create<arith::AndIOp>(loc, contiguous, unit)) : unit;
    }
  if (contiguous) {
    auto dispatch = b.create<scf::IfOp>(loc, contiguous, true);
    b.setInsertionPointToStart(dispatch.thenBlock());
    IRMapping mapping;
    for (auto metadata : descriptors) {
      auto sourceType = cast<MemRefType>(metadata.getSource().getType());
      auto [strides, offset] = sourceType.getStridesAndOffset();
      strides.back() = 1;
      SmallVector<OpFoldResult> sizes, steps;
      for (int64_t axis = 0; axis < sourceType.getRank(); ++axis) {
        sizes.push_back(sourceType.isDynamicDim(axis) ? OpFoldResult(metadata.getSizes()[axis])
            : OpFoldResult(b.getIndexAttr(sourceType.getDimSize(axis))));
        steps.push_back(ShapedType::isDynamic(strides[axis]) ? OpFoldResult(metadata.getStrides()[axis])
            : OpFoldResult(b.getIndexAttr(strides[axis])));
      }
      auto viewType = MemRefType::get(sourceType.getShape(), sourceType.getElementType(),
          StridedLayoutAttr::get(b.getContext(), offset, strides), sourceType.getMemorySpace());
      OpFoldResult begin = ShapedType::isDynamic(offset) ? OpFoldResult(metadata.getOffset())
          : OpFoldResult(b.getIndexAttr(offset));
      mapping.map(metadata.getSource(), b.create<memref::ReinterpretCastOp>(loc,
          viewType, metadata.getBaseBuffer(), begin, sizes, steps));
    }
    auto vectorScan = cast<ScanOp>(b.clone(*operation, mapping));
    if (failed(materialize(vectorScan, width))) return failure();
    operation->moveBefore(dispatch.elseBlock(), dispatch.elseBlock()->begin());
    return materialize(operation, 1);
  }
  SmallVector<Value> sizes, position(type.getRank());
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    sizes.push_back(b.create<memref::DimOp>(loc, operation.getSources()[0], axis));
  auto combine = [&](ValueRange left, ValueRange right, bool vectorized) {
    Block &body = operation.getCombine().front();
    IRMapping mapping;
    unsigned component = 0;
    for (Value value : left) mapping.map(body.getArgument(component++), value);
    for (Value value : right) mapping.map(body.getArgument(component++), value);
    for (Value capture : operation.getCaptures()) {
      Value value = vectorized ? Value(b.create<vector::BroadcastOp>(loc, VectorType::get({width}, capture.getType()), capture)) : capture;
      mapping.map(body.getArgument(component++), value);
    }
    for (Operation &instruction : body.without_terminator()) {
      if (vectorized && isa<arith::ConstantOp>(instruction)) {
        Value scalar = b.clone(instruction)->getResult(0);
        mapping.map(instruction.getResult(0), b.create<vector::BroadcastOp>(loc, VectorType::get({width}, scalar.getType()), scalar));
      } else {
        Operation *cloned = b.clone(instruction, mapping);
        if (vectorized) cloned->getResult(0).setType(VectorType::get({width}, instruction.getResult(0).getType()));
      }
    }
    SmallVector<Value> result;
    for (Value value : body.getTerminator()->getOperands()) result.push_back(mapping.lookupOrDefault(value));
    return result;
  };
  auto broadcast = [&](ValueRange scalars) {
    SmallVector<Value> result;
    for (Value scalar : scalars) result.push_back(b.create<vector::BroadcastOp>(loc, VectorType::get({width}, scalar.getType()), scalar));
    return result;
  };
  auto shift = [&](ValueRange vectors, ValueRange leading, int64_t offset) {
    SmallVector<int64_t> lanes;
    for (int64_t lane = 0; lane < width; ++lane) lanes.push_back(lane < offset ? lane : width + lane - offset);
    SmallVector<Value> result;
    for (auto [value, initial] : llvm::zip(vectors, leading))
      result.push_back(b.create<vector::ShuffleOp>(loc, initial, value, lanes));
    return result;
  };
  std::function<void(int64_t)> traverse = [&](int64_t axis) {
    if (axis != type.getRank()) {
      if (axis == scanAxis) { traverse(axis + 1); return; }
      loop(b, loc, index(b, loc, 0), sizes[axis], 1, [&](Value coordinate) {
        position[axis] = coordinate;
        traverse(axis + 1);
      });
      return;
    }
    Value extent = sizes[operation.getAxis()];
    Value begin = index(b, loc, 0);
    SmallVector<Value> initials(operation.getInitials());
    if (width > 1) {
      Value step = index(b, loc, width);
      Value completeEnd = b.create<arith::SubIOp>(loc, extent, b.create<arith::RemSIOp>(loc, extent, step));
      auto blocks = b.create<scf::ForOp>(loc, begin, completeEnd, step, initials);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(blocks.getBody());
        Value coordinate = blocks.getInductionVar();
        if (operation.getReverse()) coordinate = b.create<arith::SubIOp>(loc,
            b.create<arith::SubIOp>(loc, extent, step), coordinate);
        position[scanAxis] = coordinate;
        SmallVector<int64_t> reverseLanes;
        for (int64_t lane = width - 1; lane >= 0; --lane) reverseLanes.push_back(lane);
        SmallVector<Value> prefix;
        for (Value source : operation.getSources()) {
          auto vectorType = VectorType::get({width}, cast<MemRefType>(source.getType()).getElementType());
          Value values = b.create<vector::LoadOp>(loc, vectorType, source, position);
          if (operation.getReverse()) values = b.create<vector::ShuffleOp>(loc, values, values, reverseLanes);
          prefix.push_back(values);
        }
        auto identity = broadcast(operation.getInitials());
        // Each stage combines adjacent ranges in the selected traversal order.
        for (int64_t offset = 1; offset < width; offset *= 2)
          prefix = combine(shift(prefix, identity, offset), prefix, true);
        auto incoming = broadcast(blocks.getRegionIterArgs());
        prefix = combine(incoming, prefix, true);
        auto output = operation.getInclusive() ? prefix : shift(prefix, incoming, 1);
        for (auto [value, destination] : llvm::zip(output, operation.getOutputs())) {
          if (operation.getReverse()) value = b.create<vector::ShuffleOp>(loc, value, value, reverseLanes);
          b.create<vector::StoreOp>(loc, value, destination, position);
        }
        SmallVector<Value> next;
        for (Value value : prefix)
          next.push_back(b.create<vector::ExtractElementOp>(loc, value, index(b, loc, width - 1)));
        b.create<scf::YieldOp>(loc, next);
      }
      begin = completeEnd;
      initials.assign(blocks.getResults().begin(), blocks.getResults().end());
    }
    auto scan = b.create<scf::ForOp>(loc, begin, extent, index(b, loc, 1), initials);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(scan.getBody());
    Value coordinate = scan.getInductionVar();
    if (operation.getReverse()) coordinate = b.create<arith::SubIOp>(loc,
        b.create<arith::SubIOp>(loc, extent, index(b, loc, 1)), coordinate);
    position[operation.getAxis()] = coordinate;
    SmallVector<Value> inputs;
    for (Value source : operation.getSources())
      inputs.push_back(b.create<memref::LoadOp>(loc, source, position));
    auto next = combine(scan.getRegionIterArgs(), inputs, false);
    ValueRange output = operation.getInclusive() ? ValueRange(next) : ValueRange(scan.getRegionIterArgs());
    for (auto [value, destination] : llvm::zip(output, operation.getOutputs()))
      b.create<memref::StoreOp>(loc, value, destination, position);
    b.create<scf::YieldOp>(loc, next);
  };
  traverse(0);
  operation.erase();
  return success();
}

LogicalResult materialize(ScanOp operation) {
  int64_t width = 1;
  auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
  if (binding)
    if (auto parameter = dyn_cast_or_null<IntegerAttr>(binding.getParameters().get("scan_width")))
      width = parameter.getInt();
  return materialize(operation, width);
}

}

LogicalResult realizeHistograms(func::FuncOp function) {
  auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  auto configuration = function->getAttrOfType<ConfigurationAttr>("intent_cpu.configuration");
  SmallVector<HistogramOp> histograms;
  function.walk([&](HistogramOp operation) { histograms.push_back(operation); });
  for (HistogramOp operation : histograms) {
    OpBuilder b(operation);
    Location loc = operation.getLoc();
    auto inputType = cast<MemRefType>(operation.getValues().getType());
    auto outputType = cast<MemRefType>(operation.getOutput().getType());
    int64_t groups = operation->getParentOfType<scf::ParallelOp>() ? 1
        : capabilities.getWorkers() * configuration.getTaskGrain();
    Value zero = index(b, loc, 0), one = index(b, loc, 1), groupCount = index(b, loc, groups);
    Value size = one;
    SmallVector<Value> extents;
    for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
      extents.push_back(b.create<memref::DimOp>(loc, operation.getValues(), axis));
      size = multiply(b, loc, size, extents.back());
    }
    Value bins = b.create<memref::DimOp>(loc, operation.getOutput(), 0);
    SmallVector<Value> dynamicSizes;
    if (outputType.isDynamicDim(0)) dynamicSizes.push_back(bins);
    Value partials = b.create<memref::AllocOp>(loc,
        MemRefType::get({groups, outputType.getDimSize(0)}, outputType.getElementType()), dynamicSizes);
    Value countZero = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(outputType.getElementType(), 0));
    Value countOne = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(outputType.getElementType(), 1));
    b.create<linalg::FillOp>(loc, ValueRange{countZero}, ValueRange{partials});
    Value chunk = b.create<arith::CeilDivSIOp>(loc, size, groupCount);
    auto tasks = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{groupCount}, ValueRange{one});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(tasks.getBody());
      Value group = tasks.getInductionVars()[0];
      Value begin = multiply(b, loc, group, chunk);
      Value end = b.create<arith::MinSIOp>(loc, add(b, loc, begin, chunk), size);
      loop(b, loc, begin, end, 1, [&](Value ordinal) {
        SmallVector<Value> coordinates(inputType.getRank());
        Value remaining = ordinal;
        for (int64_t axis = inputType.getRank() - 1; axis >= 0; --axis) {
          coordinates[axis] = axis ? Value(b.create<arith::RemSIOp>(loc, remaining, extents[axis])) : remaining;
          if (axis) remaining = b.create<arith::DivSIOp>(loc, remaining, extents[axis]);
        }
        Value active = b.create<memref::LoadOp>(loc, operation.getValid(), coordinates);
        auto conditional = b.create<scf::IfOp>(loc, active, false);
        OpBuilder::InsertionGuard predicateGuard(b);
        b.setInsertionPointToStart(conditional.thenBlock());
        Value value = b.create<memref::LoadOp>(loc, operation.getValues(), coordinates);
        if (!value.getType().isIndex()) {
          if (operation.getUnsignedValues()) value = b.create<arith::IndexCastUIOp>(loc, b.getIndexType(), value);
          else value = b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
        }
        SmallVector<Value> bin{group, value};
        Value previous = b.create<memref::LoadOp>(loc, partials, bin);
        Value next = b.create<arith::AddIOp>(loc, previous, countOne);
        b.create<memref::StoreOp>(loc, next, partials, bin);
      });
    }
    b.create<linalg::FillOp>(loc, ValueRange{countZero}, ValueRange{operation.getOutput()});
    auto bin = b.getAffineDimExpr(0), group = b.getAffineDimExpr(1);
    b.create<linalg::GenericOp>(loc, ValueRange{partials}, ValueRange{operation.getOutput()},
        SmallVector<AffineMap>{AffineMap::get(2, 0, {group, bin}, b.getContext()),
                              AffineMap::get(2, 0, {bin}, b.getContext())},
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::reduction},
        [](OpBuilder &nested, Location location, ValueRange arguments) {
          nested.create<linalg::YieldOp>(location, nested.create<arith::AddIOp>(location, arguments[0], arguments[1]).getResult());
        });
    b.create<memref::DeallocOp>(loc, partials);
    operation.erase();
  }
  return success();
}

LogicalResult materializeStructuredComputations(func::FuncOp function) {
  function.walk([&](linalg::GenericOp operation) { collapseProductReductionAxes(operation); });
  SmallVector<Operation *> views;
  function.walk([&](Operation *operation) {
    if (isa<memref::ExpandShapeOp, memref::CollapseShapeOp>(operation)) views.push_back(operation);
  });
  for (Operation *operation : views) {
    auto axes = unitReshapeAxes(operation);
    auto collapse = dyn_cast<memref::CollapseShapeOp>(operation);
    if (!axes && !collapse) return operation->emitError("CPU shape view materialization requires unit-axis reassociation");
    OpBuilder b(operation);
    Location loc = operation->getLoc();
    auto type = cast<MemRefType>(operation->getResult(0).getType());
    SmallVector<int64_t> staticStrides;
    int64_t staticOffset;
    if (failed(type.getStridesAndOffset(staticStrides, staticOffset)))
      return operation->emitError("CPU shape view materialization requires strided storage");
    auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, operation->getOperand(0));
    SmallVector<OpFoldResult> sizes(type.getRank()), strides;
    for (int64_t axis = 0; axis < type.getRank(); ++axis) {
      if (!type.isDynamicDim(axis)) sizes[axis] = b.getIndexAttr(type.getDimSize(axis));
      strides.push_back(b.getIndexAttr(ShapedType::isDynamic(staticStrides[axis]) ? 1 : staticStrides[axis]));
    }
    if (axes)
      for (auto [source, result] : *axes) {
        if (type.isDynamicDim(result)) sizes[result] = metadata.getSizes()[source];
        if (ShapedType::isDynamic(staticStrides[result])) strides[result] = metadata.getStrides()[source];
      }
    if (collapse)
      for (auto [result, group] : llvm::enumerate(collapse.getReassociationIndices())) {
        if (type.isDynamicDim(result)) {
          Value size = index(b, loc, 1);
          for (int64_t source : group)
            size = b.create<arith::MulIOp>(loc, size, metadata.getSizes()[source]);
          sizes[result] = size;
        }
        if (!axes && ShapedType::isDynamic(staticStrides[result])) {
          Value stride = metadata.getStrides()[group.back()];
          Value found = b.create<arith::ConstantIntOp>(loc, 0, 1);
          for (int64_t source : llvm::reverse(group)) {
            Value nonunit = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                metadata.getSizes()[source], index(b, loc, 1));
            Value choose = b.create<arith::AndIOp>(loc, nonunit,
                b.create<arith::XOrIOp>(loc, found, b.create<arith::ConstantIntOp>(loc, 1, 1)));
            stride = b.create<arith::SelectOp>(loc, choose, metadata.getStrides()[source], stride);
            found = b.create<arith::OrIOp>(loc, found, nonunit);
          }
          strides[result] = stride;
        }
      }
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(operation))
      for (auto [source, group] : llvm::enumerate(expand.getReassociationIndices())) {
        Value stride = metadata.getStrides()[source];
        for (int64_t result : llvm::reverse(group)) {
          if (ShapedType::isDynamic(staticStrides[result])) strides[result] = stride;
          if (result != group.front() && type.getDimSize(result) != 1)
            stride = b.create<arith::MulIOp>(loc, stride, getValueOrCreateConstantIndexOp(b, loc, sizes[result]));
        }
      }
    OpFoldResult offset = ShapedType::isDynamic(staticOffset) ? OpFoldResult(metadata.getOffset())
                                                            : OpFoldResult(b.getIndexAttr(staticOffset));
    Value view = b.create<memref::ReinterpretCastOp>(loc, type, metadata.getBaseBuffer(), offset, sizes, strides);
    operation->getResult(0).replaceAllUsesWith(view);
    operation->erase();
  }
  SmallVector<Operation *> operations;
  function.walk([&](Operation *operation) {
    if (isa<linalg::GenericOp, linalg::FillOp, ReduceOp, ScanOp>(operation)) operations.push_back(operation);
  });
  for (Operation *operation : operations) {
    if (auto fill = dyn_cast<linalg::FillOp>(operation)) {
      if (fill.getOutputs().size() != 1 || fill.getNumResults())
        return fill.emitError("CPU fill requires one physical buffer");
      OpBuilder b(fill);
      int64_t rank = cast<MemRefType>(fill.getOutputs()[0].getType()).getRank();
      auto generic = b.create<linalg::GenericOp>(fill.getLoc(), fill.getInputs(), fill.getOutputs(),
          SmallVector<AffineMap>{AffineMap::get(rank, 0, {}, b.getContext()), b.getMultiDimIdentityMap(rank)},
          SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
          [](OpBuilder &nested, Location loc, ValueRange arguments) {
            nested.create<linalg::YieldOp>(loc, arguments[0]);
          });
      fill.erase();
      if (failed(materialize(generic))) return failure();
    } else if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
      if (failed(materialize(generic))) return failure();
    } else if (auto scan = dyn_cast<ScanOp>(operation)) {
      if (failed(materialize(scan))) return failure();
    } else if (failed(materialize(cast<ReduceOp>(operation)))) return failure();
  }
  return success();
}

}
