#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool isScalar(Type type) {
  return isa<IntegerType, IndexType, FloatType>(type);
}

bool isScalarProduct(Type type) {
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::all_of(record.getFieldTypes(), [](Attribute field) {
      return isScalarProduct(cast<TypeAttr>(field).getValue());
    });
  return isScalar(type);
}

bool canPredicate(Block &block, bool allowStores = false,
                  bool allowProducts = false) {
  for (Operation &operation : block.without_terminator()) {
    if (auto store = dyn_cast<StoreOp>(operation)) {
      if (!allowStores || !isScalar(store.getValue().getType()) ||
          !llvm::all_of(store.getCoordinates(), [](Value coordinate) {
            return isScalar(coordinate.getType());
          }))
        return false;
      continue;
    }
    if (operation.getNumRegions() || !operation.getNumResults() ||
        !llvm::all_of(operation.getResultTypes(), [&](Type type) {
          return allowProducts ? isScalarProduct(type) : isScalar(type);
        }))
      return false;
    ValueRange coordinates;
    if (auto load = dyn_cast<LoadOp>(operation))
      coordinates = load.getCoordinates();
    else if (auto gather = dyn_cast<GatherOp>(operation))
      coordinates = gather.getCoordinates();
    if (!llvm::all_of(coordinates, [](Value coordinate) {
          return isScalar(coordinate.getType());
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
  if (!record)
    return builder.create<SelectOp>(location, lhs.getType(), condition, lhs, rhs);
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
  auto maskedValidity = [&](Value valid) -> Value {
    if (!valid)
      return predicate;
    return builder.create<BinaryOp>(
        location, resultType(builder.getI1Type()), predicate, lift(mapped(valid)),
        BinaryOperator::LogicalAnd);
  };
  auto coordinates = [&](ValueRange values) {
    return llvm::to_vector(llvm::map_range(values, mapped));
  };
  auto fill = [&](Value value, Type type) -> Value {
    return lift(value ? mapped(value)
                      : builder.create<arith::ConstantOp>(
                            location, builder.getZeroAttr(type)));
  };
  Operation *clone;
  if (auto store = dyn_cast<StoreOp>(operation)) {
    clone = builder.create<StoreOp>(
        location, mapped(store.getResource()), coordinates(store.getCoordinates()),
        lift(mapped(store.getValue())), maskedValidity(store.getValid()),
        store.getSourceAxes());
  } else if (auto gather = dyn_cast<GatherOp>(operation)) {
    clone = builder.create<GatherOp>(
        location, resultType(gather.getType()), mapped(gather.getSource()),
        coordinates(gather.getCoordinates()), maskedValidity(gather.getValid()),
        fill(gather.getFill(), gather.getType()), gather.getSourceAxes());
  } else if (auto load = dyn_cast<LoadOp>(operation)) {
    // Predicated iterations must not issue accesses in inactive lanes.
    clone = builder.create<LoadOp>(
        location, resultType(load.getType()), mapped(load.getResource()),
        coordinates(load.getCoordinates()), maskedValidity(load.getValid()),
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
        !llvm::all_of(conditional.getResultTypes(), isScalarProduct) ||
        !canPredicate(conditional.getThenRegion().front(), false, true) ||
        !canPredicate(conditional.getElseRegion().front(), false, true))
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
