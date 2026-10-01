#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Predication.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <limits>

using namespace mlir;

namespace intent::gpu {
namespace {

bool isScalar(Type type) {
  return isa<IntegerType, IndexType, FloatType>(type);
}

bool isFloatToIntegerCast(CastOp cast) {
  auto elementType = [](Type type) {
    auto fragment = dyn_cast<FragmentType>(type);
    return fragment ? fragment.getElementType() : type;
  };
  return isa<FloatType>(elementType(cast.getValue().getType())) &&
         elementType(cast.getType()).isIntOrIndex();
}

bool isPredicatableProduct(Type type) {
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::all_of(record.getFieldTypes(), [](Attribute field) {
      return isPredicatableProduct(cast<TypeAttr>(field).getValue());
    });
  return isScalar(type) || isa<FragmentType>(type);
}

bool dependsOnWorksetCoordinate(Value value) {
  SmallVector<Value> worklist{value};
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (!worklist.empty()) {
    Operation *producer = worklist.pop_back_val().getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (isa<WorksetCoordinateOp>(producer))
      return true;
    llvm::append_range(worklist, producer->getOperands());
  }
  return false;
}

bool canPredicate(Block &block, bool allowStores = false,
                  bool allowProducts = false, bool allowLoops = false,
                  bool allowAssumptions = false) {
  for (Operation &operation : block.without_terminator()) {
    if (auto assumption = dyn_cast<AssumeInBoundsOp>(operation)) {
      if (!allowAssumptions || !isScalar(assumption.getIndex().getType()))
        return false;
      continue;
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      APInt lower, upper, step;
      if (!allowLoops ||
          !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
          !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
          !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
          !step.isStrictlyPositive() ||
          !llvm::all_of(loop.getResultTypes(), isPredicatableProduct) ||
          !canPredicate(*loop.getBody(), /*allowStores=*/false,
                        /*allowProducts=*/true, /*allowLoops=*/true))
        return false;
      continue;
    }
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      if (!allowLoops || !canPredicateScalarWhile(loop))
        return false;
      continue;
    }
    if (auto store = dyn_cast<StoreOp>(operation)) {
      if (!allowStores || !isScalar(store.getValue().getType()) ||
          !llvm::all_of(store.getCoordinates(), [&](Value coordinate) {
            return isScalar(coordinate.getType()) &&
                   (!allowAssumptions || coordinate.getType().isIntOrIndex());
          }))
        return false;
      continue;
    }
    if (allowProducts &&
        isa<ContractOp, ReduceOp, ReshapeOp, TransposeOp>(operation)) {
      if (!llvm::all_of(operation.getResultTypes(), isPredicatableProduct) ||
          !isMemoryEffectFree(&operation))
        return false;
      if (auto reduce = dyn_cast<ReduceOp>(operation))
        if (!canPredicate(reduce.getCombine().front(), false, true, false))
          return false;
      continue;
    }
    if (operation.getNumRegions() || !operation.getNumResults() ||
        !llvm::all_of(operation.getResultTypes(), [&](Type type) {
          return allowProducts ? isPredicatableProduct(type) : isScalar(type);
        }))
      return false;
    ValueRange coordinates;
    if (auto load = dyn_cast<LoadOp>(operation))
      coordinates = load.getCoordinates();
    else if (auto gather = dyn_cast<GatherOp>(operation))
      coordinates = gather.getCoordinates();
    if (!llvm::all_of(coordinates, [&](Value coordinate) {
          if (allowAssumptions && !coordinate.getType().isIntOrIndex())
            return false;
          return allowProducts ? isPredicatableProduct(coordinate.getType())
                               : isScalar(coordinate.getType());
        }))
      return false;
    bool product = allowProducts && isa<MakeRecordOp, ExtractOp>(operation) &&
                   isSpeculatable(&operation) && isMemoryEffectFree(&operation);
    auto conversion = dyn_cast<CastOp>(operation);
    bool guardedConversion = conversion && isFloatToIntegerCast(conversion);
    if (!product && !guardedConversion && !canPredicateValueOperation(&operation))
      return false;
  }
  return true;
}

void dischargeScalarAssumptions(Block &block) {
  SmallVector<AssumeInBoundsOp> assumptions;
  for (Operation &operation : block.without_terminator())
    if (auto assumption = dyn_cast<AssumeInBoundsOp>(operation))
      assumptions.push_back(assumption);
  if (assumptions.empty())
    return;
  // A fact inside a branch is not true for inactive lanes after if-conversion.
  // Close the accesses with explicit validity before removing that lexical fact.
  // These guards are true for every active access of a legal source program.
  for (Operation &operation : llvm::make_early_inc_range(block.without_terminator())) {
    auto load = dyn_cast<LoadOp>(operation);
    auto store = dyn_cast<StoreOp>(operation);
    if (!load && !store)
      continue;
    Value resource = load ? load.getResource() : store.getResource();
    SmallVector<Value> coordinates = llvm::to_vector(
        load ? load.getCoordinates() : store.getCoordinates());
    if (coordinates.empty())
      continue;
    ArrayRef<int64_t> axes = load ? load.getSourceAxes() : store.getSourceAxes();
    Value valid = load ? load.getValid() : store.getValid();
    OpBuilder builder(&operation);
    Location location = operation.getLoc();
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    for (unsigned position = 0; position < coordinates.size(); ++position) {
      Value &coordinate = coordinates[position];
      int64_t axis = axes[position];
      if (!coordinate.getType().isIndex())
        coordinate = builder.create<CastOp>(location, builder.getIndexType(), coordinate);
      Value extent;
      if (auto buffer = dyn_cast<BufferType>(resource.getType()))
        extent = builder.create<PhysicalExprOp>(location, builder.getIndexType(),
            cast<PhysicalExprAttr>(buffer.getShape()[axis]));
      else
        extent = builder.create<DimOp>(location, builder.getIndexType(), resource, axis);
      Value lower = builder.create<CompareOp>(location, builder.getI1Type(),
          coordinate, zero, ComparePredicate::Ge);
      Value upper = builder.create<CompareOp>(location, builder.getI1Type(),
          coordinate, extent, ComparePredicate::Lt);
      Value bounded = builder.create<BinaryOp>(location, builder.getI1Type(),
          lower, upper, BinaryOperator::LogicalAnd);
      valid = valid ? Value(builder.create<BinaryOp>(location, builder.getI1Type(),
          valid, bounded, BinaryOperator::LogicalAnd)) : bounded;
    }
    if (load) {
      load.getCoordinatesMutable().assign(coordinates);
      load.getValidMutable().assign(valid);
      if (!load.getFill())
        load.getFillMutable().assign(builder.create<arith::ConstantOp>(
            location, builder.getZeroAttr(load.getType())).getResult());
    } else {
      store.getCoordinatesMutable().assign(coordinates);
      store.getValidMutable().assign(valid);
    }
  }
  for (AssumeInBoundsOp assumption : assumptions)
    assumption.erase();
}

SmallVector<Value> predicateBlock(OpBuilder &builder, Block &block,
                                  Value predicate) {
  IRMapping mapping;
  for (Operation &operation : block.without_terminator())
    clonePredicatedScalarOperation(builder, &operation, mapping, predicate);
  SmallVector<Value> results;
  for (Value value : block.getTerminator()->getOperands())
    results.push_back(mapping.lookupOrDefault(value));
  return results;
}

Value selectScalarProduct(OpBuilder &builder, Location location, Value condition,
                          Value lhs, Value rhs) {
  auto record = dyn_cast<RecordType>(lhs.getType());
  if (!record) {
    if (auto fragment = dyn_cast<FragmentType>(lhs.getType())) {
      auto predicate = FragmentType::get(
          fragment.getContext(), builder.getI1Type(), fragment.getShape(),
          fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
      condition = builder.create<BroadcastOp>(location, predicate, condition);
    }
    return builder.create<SelectOp>(location, lhs.getType(), condition, lhs, rhs);
  }
  auto field = [&](Value value, unsigned index, Type type) -> Value {
    if (auto made = value.getDefiningOp<MakeRecordOp>())
      return made.getFields()[index];
    return builder.create<ExtractOp>(location, type, value, index);
  };
  SmallVector<Value> fields;
  for (auto [index, attribute] : llvm::enumerate(record.getFieldTypes())) {
    Type type = cast<TypeAttr>(attribute).getValue();
    fields.push_back(selectScalarProduct(builder, location, condition,
                                         field(lhs, index, type),
                                         field(rhs, index, type)));
  }
  return builder.create<MakeRecordOp>(location, record, fields);
}

Value anyActiveLane(OpBuilder &builder, Location location, Value predicate) {
  auto shape = dyn_cast<FragmentType>(predicate.getType());
  if (!shape)
    return predicate;
  Value identity = builder.create<arith::ConstantOp>(location,
                                                    builder.getBoolAttr(false));
  SmallVector<int64_t> axes;
  for (unsigned axis = 0; axis < shape.getShape().size(); ++axis)
    axes.push_back(axis);
  auto any = builder.create<ReduceOp>(
      location, ValueRange{predicate},
      ValueRange{identity}, ValueRange{}, axes);
  OpBuilder::InsertionGuard guard(builder);
  Block *combine = builder.createBlock(
      &any.getCombine(), {}, {builder.getI1Type(), builder.getI1Type()},
      {location, location});
  Value joined = builder.create<BinaryOp>(
      location, builder.getI1Type(), combine->getArgument(0),
      combine->getArgument(1), BinaryOperator::LogicalOr);
  builder.create<YieldOp>(location, joined);
  return any.getResult(0);
}

} // namespace

bool canPredicateScalarBlock(Block &block) {
  return canPredicate(block, /*allowStores=*/true);
}

bool canPredicateScalarWhile(scf::WhileOp loop) {
  if (!loop.getNumResults() || !llvm::hasSingleElement(loop.getBefore()) ||
      !llvm::hasSingleElement(loop.getAfter()) ||
      !llvm::all_of(loop.getResultTypes(), isScalar))
    return false;
  Block &before = loop.getBefore().front();
  Block &after = loop.getAfter().front();
  auto condition = dyn_cast<scf::ConditionOp>(before.getTerminator());
  if (!condition || before.getArgumentTypes() != loop.getResultTypes() ||
      after.getArgumentTypes() != loop.getResultTypes() ||
      !llvm::equal(condition.getArgs(), before.getArguments()))
    return false;
  // Only pure per-element recurrences are coarsened here. Cross-lane effects,
  // nested regions and changes to the state in the condition need other proofs.
  return llvm::all_of(ArrayRef<Block *>{&before, &after}, [&](Block *block) {
    return canPredicate(*block) &&
           llvm::all_of(block->without_terminator(), [](Operation &operation) {
             return isMemoryEffectFree(&operation);
           });
  });
}

void clonePredicatedScalarOperation(OpBuilder &builder, Operation *operation,
                                   IRMapping &mapping, Value predicate,
                                   FragmentType shape,
                                   bool nonemptyIterations) {
  Location location = operation->getLoc();
  auto hasIterationAxes = [&](Type type) {
    auto fragment = dyn_cast<FragmentType>(type);
    return shape && fragment &&
           llvm::all_of(shape.getAxisMaps(), [&](Attribute axis) {
             auto source = cast<AxisMapAttr>(axis);
             return llvm::any_of(fragment.getAxisMaps(), [&](Attribute candidate) {
               auto target = cast<AxisMapAttr>(candidate);
               return sourceAxisIdentity(source) == sourceAxisIdentity(target) &&
                      source.getDimensionId() == target.getDimensionId();
             });
           });
  };
  auto resultType = [&](Type element) -> Type {
    if (!shape || hasIterationAxes(element))
      return element;
    // Independent iterations form a new prefix; an existing column fragment
    // keeps its own axes. BroadcastOp projects by these identities, so a row
    // vector embeds as [R, 1] and a column vector as [1, C].
    SmallVector<Attribute> extents(shape.getShape().begin(), shape.getShape().end());
    SmallVector<Attribute> axes(shape.getAxisMaps().begin(),
                               shape.getAxisMaps().end());
    if (auto fragment = dyn_cast<FragmentType>(element)) {
      element = fragment.getElementType();
      llvm::append_range(extents, fragment.getShape());
      for (Attribute attribute : fragment.getAxisMaps()) {
        auto axis = cast<AxisMapAttr>(attribute);
        axes.push_back(AxisMapAttr::get(
            shape.getContext(), axis.getSourceId(), axis.getSourceAxis(),
            axis.getDimensionId(), axes.size(), axis.getDerived()));
      }
    }
    return FragmentType::get(
        shape.getContext(), element, builder.getArrayAttr(extents),
        builder.getArrayAttr(axes), shape.getValidity(), shape.getOwner());
  };
  auto project = [&](Value value, Type target) -> Value {
    return value.getType() == target
               ? value
               : Value(builder.create<BroadcastOp>(location, target, value));
  };
  auto lift = [&](Value value) -> Value {
    return project(value, resultType(value.getType()));
  };
  auto mapped = [&](Value value) { return mapping.lookupOrDefault(value); };
  auto maskedValidity = [&](Value valid, Type dataType) -> Value {
    Type maskType = builder.getI1Type();
    if (auto fragment = dyn_cast<FragmentType>(resultType(dataType)))
      maskType = FragmentType::get(
          fragment.getContext(), builder.getI1Type(), fragment.getShape(),
          fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
    Value condition = project(predicate, maskType);
    if (!valid)
      return condition;
    return builder.create<BinaryOp>(
        location, maskType, condition, project(mapped(valid), maskType),
        BinaryOperator::LogicalAnd);
  };
  auto coordinates = [&](ValueRange values) {
    return llvm::to_vector(llvm::map_range(values, mapped));
  };
  auto fill = [&](Value value, Type type) -> Value {
    if (value)
      return project(mapped(value), resultType(type));
    auto fragment = dyn_cast<FragmentType>(type);
    Type element = fragment ? fragment.getElementType() : type;
    Value zero = builder.create<arith::ConstantOp>(
        location, builder.getZeroAttr(element));
    if (fragment)
      zero = builder.create<SplatOp>(location, fragment, zero);
    return lift(zero);
  };
  Operation *clone;
  if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    SmallVector<Value> initial = coordinates(loop.getInitArgs());
    for (Value &value : initial)
      value = lift(value);
    auto result = builder.create<scf::ForOp>(
        location, mapped(loop.getLowerBound()), mapped(loop.getUpperBound()),
        mapped(loop.getStep()), initial);
    result->setAttrs(loop->getAttrs());
    if (!result.getBody()->empty())
      result.getBody()->back().erase();
    IRMapping bodyMapping(mapping);
    bodyMapping.map(loop.getInductionVar(), result.getInductionVar());
    bodyMapping.map(loop.getRegionIterArgs(), result.getRegionIterArgs());
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(result.getBody());
      for (Operation &nested : loop.getBody()->without_terminator())
        clonePredicatedScalarOperation(builder, &nested, bodyMapping, predicate,
                                       shape, nonemptyIterations);
      SmallVector<Value> yielded;
      for (auto [value, carried] :
           llvm::zip(loop.getBody()->getTerminator()->getOperands(),
                     result.getRegionIterArgs()))
        yielded.push_back(selectScalarProduct(
            builder, location, predicate, lift(bodyMapping.lookupOrDefault(value)),
            carried));
      builder.create<scf::YieldOp>(location, yielded);
    }
    clone = result;
  } else if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
    SmallVector<Value> initial = coordinates(loop.getInits());
    for (Value &value : initial)
      value = lift(value);
    if (shape)
      initial.push_back(lift(predicate));
    SmallVector<Type> types;
    for (Value value : initial)
      types.push_back(value.getType());
    auto result = builder.create<scf::WhileOp>(location, types, initial);
    result->setAttrs(loop->getAttrs());
    OpBuilder::InsertionGuard guard(builder);
    Block *before = builder.createBlock(
        &result.getBefore(), {}, types, SmallVector<Location>(types.size(), location));
    IRMapping beforeMapping(mapping);
    beforeMapping.map(loop.getBeforeArguments(),
                      before->getArguments().take_front(loop.getNumResults()));
    Value active = shape ? Value(before->getArguments().back()) : predicate;
    for (Operation &nested : loop.getBefore().front().without_terminator())
      clonePredicatedScalarOperation(builder, &nested, beforeMapping, active, shape);
    auto condition = cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
    Value nextActive = builder.create<BinaryOp>(
        location, active.getType(), active,
        project(beforeMapping.lookupOrDefault(condition.getCondition()), active.getType()),
        BinaryOperator::LogicalAnd);
    Value anyActive = anyActiveLane(builder, location, nextActive);
    SmallVector<Value> forwarded(before->getArguments().take_front(loop.getNumResults()));
    if (shape)
      forwarded.push_back(nextActive);
    builder.create<scf::ConditionOp>(location, anyActive, forwarded);
    Block *after = builder.createBlock(
        &result.getAfter(), {}, types, SmallVector<Location>(types.size(), location));
    IRMapping afterMapping(mapping);
    afterMapping.map(loop.getAfterArguments(),
                     after->getArguments().take_front(loop.getNumResults()));
    active = shape ? Value(after->getArguments().back()) : predicate;
    for (Operation &nested : loop.getAfter().front().without_terminator())
      clonePredicatedScalarOperation(builder, &nested, afterMapping, active, shape);
    SmallVector<Value> yielded;
    for (auto [value, carried] :
         llvm::zip(loop.getAfter().front().getTerminator()->getOperands(),
                   after->getArguments().take_front(loop.getNumResults())))
      yielded.push_back(selectScalarProduct(
          builder, location, active, lift(afterMapping.lookupOrDefault(value)), carried));
    if (shape)
      yielded.push_back(active);
    builder.create<scf::YieldOp>(location, yielded);
    clone = result;
  } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
    // An iteration-invariant condition remains scalar after remapping. Keep
    // its branches lazy while predicating the accesses in the selected branch.
    if (shape && mapped(branch.getCondition()).getType().isInteger(1)) {
      SmallVector<Type> types;
      for (Type type : branch.getResultTypes())
        types.push_back(resultType(type));
      auto result = builder.create<scf::IfOp>(location, types,
          mapped(branch.getCondition()), !branch.getElseRegion().empty());
      for (auto [original, region] :
           llvm::zip(branch->getRegions(), result->getRegions())) {
        if (original.empty())
          continue;
        Block &body = region.front();
        if (!body.empty())
          body.back().erase();
        IRMapping branchMapping(mapping);
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&body);
        for (Operation &nested : original.front().without_terminator())
          clonePredicatedScalarOperation(builder, &nested, branchMapping,
                                         predicate, shape, nonemptyIterations);
        SmallVector<Value> yielded;
        for (Value value : original.front().getTerminator()->getOperands())
          yielded.push_back(lift(branchMapping.lookupOrDefault(value)));
        builder.create<scf::YieldOp>(location, yielded);
      }
      if (Attribute origin = operation->getAttr(originAttr))
        result->setAttr(originAttr, origin);
      mapping.map(branch.getResults(), result.getResults());
      return;
    }
    // A complete chunk can have scalar `true` validity even when the branch
    // condition varies by iteration. Lift both predicates to that iteration
    // schema before combining them.
    Value activePredicate = lift(predicate);
    Value condition = project(lift(mapped(branch.getCondition())),
                              activePredicate.getType());
    Value zero = builder.create<arith::ConstantOp>(location,
                                                 builder.getBoolAttr(false));
    if (auto fragment = dyn_cast<FragmentType>(condition.getType()))
      zero = builder.create<SplatOp>(location, fragment, zero);
    Value inverse = builder.create<CompareOp>(
        location, condition.getType(), condition, zero, ComparePredicate::Eq);
    auto cloneBranch = [&](Region &region, Value selected) {
      SmallVector<Value> results;
      if (region.empty())
        return results;
      Value active = builder.create<BinaryOp>(
          location, activePredicate.getType(), activePredicate, selected,
          BinaryOperator::LogicalAnd);
      IRMapping branchMapping(mapping);
      for (Operation &nested : region.front().without_terminator())
        // This branch may have no active iteration, even in a nonempty chunk.
        clonePredicatedScalarOperation(builder, &nested, branchMapping, active,
                                       shape);
      for (Value value : region.front().getTerminator()->getOperands())
        results.push_back(lift(branchMapping.lookupOrDefault(value)));
      return results;
    };
    SmallVector<Value> thenValues = cloneBranch(branch.getThenRegion(), condition);
    SmallVector<Value> elseValues = cloneBranch(branch.getElseRegion(), inverse);
    for (auto [result, lhs, rhs] :
         llvm::zip(branch.getResults(), thenValues, elseValues))
      mapping.map(result, selectScalarProduct(builder, location, condition,
                                              lhs, rhs));
    return;
  } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
    // New independent iteration axes are free axes of the original reduction.
    auto liftGroup = [&](ValueRange values) {
      SmallVector<Value> lifted;
      for (Value value : values) lifted.push_back(lift(mapped(value)));
      return lifted;
    };
    auto sources = liftGroup(reduce.getSources());
    auto identities = liftGroup(reduce.getIdentities());
    auto captures = liftGroup(reduce.getCaptures());
    SmallVector<int64_t> axes(reduce.getAxes());
    for (int64_t &axis : axes)
      axis += shape ? shape.getShape().size() : 0;
    // The recursive clone lifts the declared result and helper schemas together;
    // source relations close after the complete predicated graph is built.
    SmallVector<Type> types;
    for (Type type : reduce.getResultTypes())
      types.push_back(resultType(type));
    auto result = builder.create<ReduceOp>(location, types, sources, identities,
                                           captures, axes);
    for (NamedAttribute attribute : reduce->getDiscardableAttrs())
      result->setAttr(attribute.getName(), attribute.getValue());
    {
      OpBuilder::InsertionGuard guard(builder);
      SmallVector<Type> arguments;
      for (Type type : reduce.getCombine().front().getArgumentTypes())
        arguments.push_back(resultType(type));
      Block *body = builder.createBlock(&result.getCombine(), {}, arguments,
                                        SmallVector<Location>(arguments.size(), location));
      IRMapping nestedMapping(mapping);
      nestedMapping.map(reduce.getCombine().front().getArguments(), body->getArguments());
      for (Operation &nested : reduce.getCombine().front().without_terminator())
        clonePredicatedScalarOperation(builder, &nested, nestedMapping, predicate, shape);
      SmallVector<Value> yielded;
      for (Value value : reduce.getCombine().front().getTerminator()->getOperands())
        yielded.push_back(lift(nestedMapping.lookupOrDefault(value)));
      builder.create<YieldOp>(location, yielded);
    }
    clone = result;
  } else if (auto store = dyn_cast<StoreOp>(operation)) {
    clone = builder.create<StoreOp>(
        location, mapped(store.getResource()), coordinates(store.getCoordinates()),
        project(mapped(store.getValue()), resultType(store.getValue().getType())),
        maskedValidity(store.getValid(), store.getValue().getType()),
        store.getSourceAxes());
  } else if (auto gather = dyn_cast<GatherOp>(operation)) {
    clone = builder.create<GatherOp>(
        location, resultType(gather.getType()), mapped(gather.getSource()),
        coordinates(gather.getCoordinates()),
        maskedValidity(gather.getValid(), gather.getType()),
        fill(gather.getFill(), gather.getType()), gather.getSourceAxes());
  } else if (auto load = dyn_cast<LoadOp>(operation)) {
    // An executing independent chunk has at least one original iteration.
    // If every read operand is invariant across those iterations, the original
    // read (including its mask/fill) is already required. Keep its own axes and
    // let its consumers broadcast, rather than issuing one copy per new lane.
    bool safeRead = shape && nonemptyIterations &&
        llvm::none_of(operation->getOperands(), [&](Value value) {
          return hasIterationAxes(mapped(value).getType());
        });
    auto view = dyn_cast<ViewType>(load.getResource().getType());
    if (!shape && isa<FragmentType>(load.getType()) && view &&
        view.getAccess() == 0) {
      PhysicalProgramAnalysis analysis(load->getParentOfType<func::FuncOp>());
      auto bounds = analysis.accessBounds(load);
      safeRead = bounds.isExact() && bounds.assumedAxes.empty();
    }
    // Value branches contain no writes. An independently bounded input read
    // can keep its original mask without acquiring unrelated lane dependence.
    if (safeRead)
      clone = builder.clone(*operation, mapping);
    else
      clone = builder.create<LoadOp>(
          location, resultType(load.getType()), mapped(load.getResource()),
          coordinates(load.getCoordinates()),
          maskedValidity(load.getValid(), load.getType()),
          fill(load.getFill(), load.getType()), load.getSourceAxes());
  } else if (auto reshape = dyn_cast<ReshapeOp>(operation);
             reshape && hasIterationAxes(mapped(reshape.getValue()).getType())) {
    // Reassociation describes the original suffix. The independent iteration
    // prefix is preserved on both sides of the reshape.
    clone = builder.create<ReshapeOp>(
        location, resultType(reshape.getType()), mapped(reshape.getValue()),
        reshape.getReassociation());
  } else if (auto transpose = dyn_cast<TransposeOp>(operation);
             transpose && hasIterationAxes(mapped(transpose.getValue()).getType())) {
    SmallVector<int64_t> permutation;
    unsigned prefix = shape.getShape().size();
    for (unsigned axis = 0; axis < prefix; ++axis)
      permutation.push_back(axis);
    for (int64_t axis : transpose.getPermutation())
      permutation.push_back(axis + prefix);
    clone = builder.create<TransposeOp>(
        location, resultType(transpose.getType()), mapped(transpose.getValue()),
        permutation);
  } else if (auto conversion = dyn_cast<CastOp>(operation);
             conversion && isFloatToIntegerCast(conversion)) {
    // Inactive branches may contain NaN or out-of-range floating values.
    // Select a defined operand before conversion, not its possibly poison result.
    Value value = lift(mapped(conversion.getValue()));
    auto fragment = dyn_cast<FragmentType>(value.getType());
    Type element = fragment ? fragment.getElementType() : value.getType();
    Value zero = builder.create<arith::ConstantOp>(
        location, builder.getZeroAttr(element));
    zero = project(zero, value.getType());
    Value guarded = selectScalarProduct(builder, location, predicate, value, zero);
    clone = builder.create<CastOp>(location, resultType(conversion.getType()), guarded);
  } else {
    bool vector = shape && llvm::any_of(operation->getOperands(), [&](Value value) {
      return hasIterationAxes(mapped(value).getType());
    });
    if (vector && isa<SplatOp, BroadcastOp>(operation)) {
      clone = builder.create<BroadcastOp>(
          location, resultType(operation->getResult(0).getType()),
          mapped(operation->getOperand(0)));
    } else {
      clone = builder.clone(*operation, mapping);
    }
    if (vector &&
        isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(operation)) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(clone);
      for (Value result : clone->getResults())
        result.setType(resultType(result.getType()));
      auto schema = cast<FragmentType>(clone->getResult(0).getType());
      for (OpOperand &operand : clone->getOpOperands()) {
        Type element = operand.get().getType();
        if (auto fragment = dyn_cast<FragmentType>(element))
          element = fragment.getElementType();
        auto target = FragmentType::get(
            schema.getContext(), element, schema.getShape(), schema.getAxisMaps(),
            schema.getValidity(), schema.getOwner());
        operand.set(project(operand.get(), target));
      }
    }
  }
  if (Attribute origin = operation->getAttr(originAttr))
    clone->setAttr(originAttr, origin);
  for (auto [previous, replacement] :
       llvm::zip(operation->getResults(), clone->getResults()))
    mapping.map(previous, replacement);
}

LogicalResult predicateScalarControl(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  SmallVector<scf::IfOp> conditionals;
  kernel->walk<WalkOrder::PostOrder>(
      [&](scf::IfOp conditional) { conditionals.push_back(conditional); });
  for (scf::IfOp conditional : conditionals) {
    if (conditional->hasAttr(executionGroupAttr))
      continue;
    if (!conditional.getNumResults()) {
      bool emptyElse = conditional.getElseRegion().empty() ||
          conditional.getElseRegion().front().without_terminator().empty();
      if (!emptyElse ||
          !canPredicate(conditional.getThenRegion().front(),
                        /*allowStores=*/true, /*allowProducts=*/false,
                        /*allowLoops=*/false, /*allowAssumptions=*/true))
        continue;
      dischargeScalarAssumptions(conditional.getThenRegion().front());
      OpBuilder builder(conditional);
      predicateBlock(builder, conditional.getThenRegion().front(),
                     conditional.getCondition());
      conditional.erase();
      continue;
    }
    if (conditional.getElseRegion().empty() ||
        !llvm::all_of(conditional.getResultTypes(), isPredicatableProduct))
      continue;
    bool scalarEffects =
        llvm::all_of(conditional.getResultTypes(), isScalar) &&
        canPredicate(conditional.getThenRegion().front(), true, false, false, true) &&
        canPredicate(conditional.getElseRegion().front(), true, false, false, true);
    if (!scalarEffects &&
        (!canPredicate(conditional.getThenRegion().front(), false, true, true) ||
         !canPredicate(conditional.getElseRegion().front(), false, true, true)))
      continue;
    bool hasFragment = false;
    for (Type type : conditional.getResultTypes())
      type.walk([&](FragmentType) { hasFragment = true; });
    // Keep launch-wide choices lazy even before their scalar values are
    // lifted to fragments; inactive transcendental/loop paths can be costly.
    if (isLaunchUniformScalar(conditional.getCondition(), *kernel) ||
        (hasFragment && !dependsOnWorksetCoordinate(conditional.getCondition())))
      continue;
    if (scalarEffects) {
      dischargeScalarAssumptions(conditional.getThenRegion().front());
      dischargeScalarAssumptions(conditional.getElseRegion().front());
    }
    OpBuilder builder(conditional);
    Value otherwise = builder.create<UnaryOp>(
        conditional.getLoc(), builder.getI1Type(), conditional.getCondition(),
        UnaryOperator::Not);
    SmallVector<Value> thenValues = predicateBlock(
        builder, conditional.getThenRegion().front(), conditional.getCondition());
    SmallVector<Value> elseValues = predicateBlock(
        builder, conditional.getElseRegion().front(), otherwise);
    for (auto [result, thenValue, elseValue] :
         llvm::zip(conditional.getResults(), thenValues, elseValues)) {
      Value replacement = selectScalarProduct(
          builder, conditional.getLoc(), conditional.getCondition(), thenValue, elseValue);
      if (Attribute origin = conditional->getAttr(originAttr))
        replacement.getDefiningOp()->setAttr(originAttr, origin);
      result.replaceAllUsesWith(replacement);
    }
    conditional.erase();
  }
  eraseDeadPhysicalValues(*kernel);
  return success();
}

LogicalResult guardInactivePredicatedLoops(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  SmallVector<scf::ForOp> loops;
  kernel->walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops) {
    APInt lower, upper, step;
    if (!loop.getNumResults() || !isMemoryEffectFree(loop) ||
        llvm::any_of(loop.getBody()->without_terminator(), [](Operation &nested) {
          return nested.getNumRegions() != 0;
        }) ||
        !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
        !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
        lower.getBitWidth() > 64 || upper.getBitWidth() > 64 ||
        step.getBitWidth() > 64 || !step.isStrictlyPositive() ||
        static_cast<__int128>(upper.getSExtValue()) - lower.getSExtValue() <=
            step.getSExtValue() ||
        static_cast<__int128>(upper.getSExtValue()) + step.getSExtValue() - 1 >
            std::numeric_limits<int64_t>::max())
      continue;
    // If-conversion freezes an inactive recurrence at its incoming value.
    // Recover that loop-invariant predicate from the carry selects; when every
    // lane is inactive, a finite effect-free loop is exactly its initial state.
    SmallVector<SmallVector<std::pair<Value, bool>>> guards;
    Type predicateType;
    bool supported = true;
    for (auto [yielded, carried] :
         llvm::zip(loop.getBody()->getTerminator()->getOperands(),
                   loop.getRegionIterArgs())) {
      if (yielded == carried)
        continue;
      SmallVector<std::pair<Value, bool>> conjunction;
      while (auto select = yielded.getDefiningOp<SelectOp>()) {
        bool inverted = select.getTrueValue() == carried;
        if ((!inverted && select.getFalseValue() != carried) ||
            !loop.isDefinedOutsideOfLoop(select.getCondition()))
          break;
        Type type = select.getCondition().getType();
        auto fragment = dyn_cast<FragmentType>(type);
        if (!fragment || fragment.getShape().empty() ||
            (predicateType && predicateType != type))
          break;
        predicateType = type;
        conjunction.emplace_back(select.getCondition(), inverted);
        yielded = inverted ? select.getFalseValue() : select.getTrueValue();
      }
      if (conjunction.empty()) {
        supported = false;
        break;
      }
      guards.push_back(std::move(conjunction));
    }
    if (!supported || guards.empty())
      continue;
    OpBuilder builder(loop);
    Location location = loop.getLoc();
    Value active;
    for (auto &conjunction : guards) {
      Value selected;
      for (auto [condition, inverted] : conjunction) {
        if (inverted)
          condition = builder.create<UnaryOp>(location, predicateType, condition,
                                               UnaryOperator::Not);
        selected = selected ? Value(builder.create<BinaryOp>(
                                  location, predicateType, selected, condition,
                                  BinaryOperator::LogicalAnd)) : condition;
      }
      active = active ? Value(builder.create<BinaryOp>(
                            location, predicateType, active, selected,
                            BinaryOperator::LogicalOr)) : selected;
    }
    auto conditional = builder.create<scf::IfOp>(
        location, loop.getResultTypes(), anyActiveLane(builder, location, active),
        /*withElseRegion=*/true);
    loop->replaceAllUsesWith(conditional.getResults());
    Block &thenBody = conditional.getThenRegion().front();
    loop->moveBefore(&thenBody, thenBody.begin());
    builder.setInsertionPointToEnd(&thenBody);
    builder.create<scf::YieldOp>(location, loop.getResults());
    builder.setInsertionPointToStart(&conditional.getElseRegion().front());
    builder.create<scf::YieldOp>(location, loop.getInitArgs());
  }
  return success();
}

} // namespace intent::gpu
