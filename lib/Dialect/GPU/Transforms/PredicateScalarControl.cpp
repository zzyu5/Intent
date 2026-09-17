#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool isScalar(Type type) {
  return isa<IntegerType, IndexType, FloatType>(type);
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
                  bool allowProducts = false, bool allowLoops = false) {
  for (Operation &operation : block.without_terminator()) {
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
    if (auto store = dyn_cast<StoreOp>(operation)) {
      if (!allowStores || !isScalar(store.getValue().getType()) ||
          !llvm::all_of(store.getCoordinates(), [](Value coordinate) {
            return isScalar(coordinate.getType());
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
          return allowProducts ? isPredicatableProduct(coordinate.getType())
                               : isScalar(coordinate.getType());
        }))
      return false;
    bool product = allowProducts && isa<MakeRecordOp, ExtractOp>(operation) &&
                   isSpeculatable(&operation) && isMemoryEffectFree(&operation);
    if (!product && !canPredicateValueOperation(&operation))
      return false;
  }
  return true;
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

} // namespace

bool canPredicateValueOperation(Operation *operation) {
  if (isa<LoadOp, GatherOp>(operation))
    return true;
  if (auto binary = dyn_cast<BinaryOp>(operation)) {
    auto kind = binary.getOperatorKind();
    if (kind == BinaryOperator::FloorDivide ||
        kind == BinaryOperator::Remainder ||
        kind == BinaryOperator::LeftShift ||
        kind == BinaryOperator::RightShift)
      return false;
  }
  auto elementType = [](Type type) {
    auto fragment = dyn_cast<FragmentType>(type);
    return fragment ? fragment.getElementType() : type;
  };
  if (auto cast = dyn_cast<CastOp>(operation)) {
    Type source = elementType(cast.getValue().getType());
    Type result = elementType(cast.getType());
    if ((isa<FloatType>(source) && !isa<FloatType>(result)) ||
        isa<Float8E4M3FNType>(result))
      return false;
  }
  return isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
             SplatOp, BroadcastOp, MakeRangeOp, DimOp, PhysicalExprOp,
             arith::ConstantOp>(operation) &&
         isSpeculatable(operation) && isMemoryEffectFree(operation);
}

bool canPredicateScalarBlock(Block &block) {
  return canPredicate(block, /*allowStores=*/true);
}

void clonePredicatedScalarOperation(OpBuilder &builder, Operation *operation,
                                   IRMapping &mapping, Value predicate,
                                   FragmentType shape) {
  Location location = operation->getLoc();
  auto resultType = [&](Type element) -> Type {
    return shape ? FragmentType::get(shape.getContext(), element,
                                     shape.getShape(), shape.getAxisMaps(),
                                     shape.getValidity(), shape.getOwner())
                 : element;
  };
  auto lift = [&](Value value) -> Value {
    if (!shape)
      return value;
    auto fragment = dyn_cast<FragmentType>(value.getType());
    Type target =
        resultType(fragment ? fragment.getElementType() : value.getType());
    return value.getType() == target
               ? value
               : Value(builder.create<BroadcastOp>(location, target, value));
  };
  auto mapped = [&](Value value) { return mapping.lookupOrDefault(value); };
  auto maskedValidity = [&](Value valid, Type dataType) -> Value {
    Type maskType = builder.getI1Type();
    if (auto fragment = dyn_cast<FragmentType>(resultType(dataType)))
      maskType = FragmentType::get(
          fragment.getContext(), builder.getI1Type(), fragment.getShape(),
          fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
    auto project = [&](Value value) -> Value {
      return value.getType() == maskType
                 ? value
                 : Value(builder.create<BroadcastOp>(
                       location, cast<FragmentType>(maskType), value));
    };
    Value condition = project(predicate);
    if (!valid)
      return condition;
    return builder.create<BinaryOp>(
        location, maskType, condition, project(mapped(valid)),
        BinaryOperator::LogicalAnd);
  };
  auto coordinates = [&](ValueRange values) {
    return llvm::to_vector(llvm::map_range(values, mapped));
  };
  auto fill = [&](Value value, Type type) -> Value {
    if (value)
      return lift(mapped(value));
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
    assert(!shape && "ordered loops are predicated before fragment lifting");
    auto result = builder.create<scf::ForOp>(
        location, mapped(loop.getLowerBound()), mapped(loop.getUpperBound()),
        mapped(loop.getStep()), coordinates(loop.getInitArgs()));
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
        clonePredicatedScalarOperation(builder, &nested, bodyMapping, predicate);
      SmallVector<Value> yielded;
      for (auto [value, carried] :
           llvm::zip(loop.getBody()->getTerminator()->getOperands(),
                     result.getRegionIterArgs()))
        yielded.push_back(selectScalarProduct(
            builder, location, predicate, bodyMapping.lookupOrDefault(value),
            carried));
      builder.create<scf::YieldOp>(location, yielded);
    }
    clone = result;
  } else if (auto store = dyn_cast<StoreOp>(operation)) {
    clone = builder.create<StoreOp>(
        location, mapped(store.getResource()), coordinates(store.getCoordinates()),
        lift(mapped(store.getValue())),
        maskedValidity(store.getValid(), store.getValue().getType()),
        store.getSourceAxes());
  } else if (auto gather = dyn_cast<GatherOp>(operation)) {
    clone = builder.create<GatherOp>(
        location, resultType(gather.getType()), mapped(gather.getSource()),
        coordinates(gather.getCoordinates()),
        maskedValidity(gather.getValid(), gather.getType()),
        fill(gather.getFill(), gather.getType()), gather.getSourceAxes());
  } else if (auto load = dyn_cast<LoadOp>(operation)) {
    bool safeRead = false;
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
  } else {
    bool vector = shape && llvm::any_of(operation->getOperands(), [&](Value value) {
      return isa<FragmentType>(mapped(value).getType());
    });
    clone = builder.clone(*operation, mapping);
    if (vector) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(clone);
      for (OpOperand &operand : clone->getOpOperands())
        operand.set(lift(operand.get()));
      for (Value result : clone->getResults())
        result.setType(resultType(result.getType()));
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
                        /*allowStores=*/true))
        continue;
      OpBuilder builder(conditional);
      predicateBlock(builder, conditional.getThenRegion().front(),
                     conditional.getCondition());
      conditional.erase();
      continue;
    }
    if (conditional.getElseRegion().empty() ||
        !llvm::all_of(conditional.getResultTypes(), isPredicatableProduct) ||
        !canPredicate(conditional.getThenRegion().front(), false, true, true) ||
        !canPredicate(conditional.getElseRegion().front(), false, true, true))
      continue;
    bool hasFragment = false;
    for (Type type : conditional.getResultTypes())
      type.walk([&](FragmentType) { hasFragment = true; });
    // Tensor predication must enable workset lifting, not duplicate both
    // sides of a launch-wide algorithm choice.
    if (hasFragment && !dependsOnWorksetCoordinate(conditional.getCondition()))
      continue;
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

} // namespace intent::gpu
