#include "Intent/Conversion/KIRToCPU/KIRToCPU.h"
#include "Intent/Analysis/CanonicalKernel.h"
#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace intent {
namespace {

struct Domain {
  Value begin, end, step;
};

class Construction {
public:
  Construction(ModuleOp original, ModuleOp physical)
      : analysis(original), module(physical), builder(physical.getContext()) {}

  LogicalResult lower(func::FuncOp source) {
    auto parameters = source->getAttrOfType<ArrayAttr>("intent.parameters");
    if (!parameters || parameters.size() != source.getNumArguments())
      return source.emitError("CPU construction requires canonical parameter metadata");
    SmallVector<Type> types;
    SmallVector<Attribute> interface;
    for (auto [i, argument] : llvm::enumerate(source.getArguments())) {
      auto parameter = cast<ParameterAttr>(parameters[i]);
      NamedAttrList item;
      item.set("name", parameter.getName());
      if (auto view = dyn_cast<ViewType>(argument.getType())) {
        auto tensor = cast<RankedTensorType>(view.getTensor());
        if (!tensor.getElementType().isF32() || view.getAccess() == 2)
          return source.emitError("CPU construction supports f32 In/Out views; InOut is not implemented");
        if (view.getConstraints().getHasStrides())
          return source.emitError("CPU construction does not yet implement declared view stride constraints");
        auto shape = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
        if (!shape)
          return source.emitError("CPU view is missing canonical dimension identities");
        types.push_back(MemRefType::get(tensor.getShape(), tensor.getElementType()));
        item.set("kind", builder.getStringAttr("view"));
        item.set("access", builder.getI64IntegerAttr(view.getAccess()));
        item.set("shape", builder.getDenseI64ArrayAttr(tensor.getShape()));
        item.set("dimensions", shape.getDimensions());
        item.set("alias", view.getConstraints().getAlias());
        item.set("noalias", builder.getBoolAttr(view.getConstraints().getNoalias()));
      } else if (argument.getType().isF32() || argument.getType().isIndex() ||
                 argument.getType().isInteger(64)) {
        types.push_back(argument.getType());
        item.set("kind", builder.getStringAttr("scalar"));
        item.set("dtype", builder.getStringAttr(argument.getType().isF32() ? "f32" : "i64"));
      } else {
        return source.emitError("CPU construction does not implement this parameter type");
      }
      interface.push_back(builder.getDictionaryAttr(item));
    }
    builder.setInsertionPointToEnd(module.getBody());
    function = builder.create<func::FuncOp>(source.getLoc(), source.getName(),
                                           builder.getFunctionType(types, {}));
    function->setAttr("cpu.interface", builder.getArrayAttr(interface));
    function->setAttr("cpu.contiguous_views", builder.getUnitAttr());
    function->setAttr("cpu.disjoint_outputs", builder.getUnitAttr());
    function.addEntryBlock();
    builder.setInsertionPointToStart(&function.front());
    for (auto [oldValue, newValue] : llvm::zip(source.getArguments(), function.getArguments())) {
      values.map(oldValue, newValue);
      if (auto view = dyn_cast<ViewType>(oldValue.getType())) {
        auto type = cast<RankedTensorType>(view.getTensor());
        auto ids = cast<TensorShapeAttr>(type.getEncoding()).getDimensions().asArrayRef();
        for (int64_t axis = 0; axis < type.getRank(); ++axis) {
          Value extent = type.isDynamicDim(axis)
              ? Value(builder.create<memref::DimOp>(source.getLoc(), newValue, axis))
              : constant(source.getLoc(), type.getDimSize(axis));
          dimensions.try_emplace(ids[axis], extent);
        }
      }
    }
    if (failed(lowerBlock(source.front())))
      return failure();
    builder.create<func::ReturnOp>(source.getLoc());
    return success();
  }

private:
  Value constant(Location loc, int64_t value) {
    return builder.create<arith::ConstantIndexOp>(loc, value);
  }

  FailureOr<SmallVector<Value>> extents(RankedTensorType tensor, Location loc) {
    SmallVector<Value> result;
    auto shape = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
    for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
      if (!tensor.isDynamicDim(axis)) {
        result.push_back(constant(loc, tensor.getDimSize(axis)));
        continue;
      }
      if (!shape || !dimensions.count(shape.getDimensions()[axis])) {
        emitError(loc, "CPU construction cannot resolve the canonical dynamic extent");
        return failure();
      }
      result.push_back(dimensions.lookup(shape.getDimensions()[axis]));
    }
    return result;
  }

  Value allocate(RankedTensorType tensor, ArrayRef<Value> sizes, Location loc) {
    SmallVector<Value> dynamic;
    for (auto [axis, size] : llvm::enumerate(sizes))
      if (tensor.isDynamicDim(axis))
        dynamic.push_back(size);
    Value allocation = builder.create<memref::AllocOp>(
        loc, MemRefType::get(tensor.getShape(), tensor.getElementType()), dynamic);
    allocations.back().push_back(allocation);
    return allocation;
  }

  void elements(Location loc, ArrayRef<Value> sizes,
                llvm::function_ref<void(ValueRange)> body) {
    SmallVector<Value> indices;
    std::function<void(unsigned)> visit = [&](unsigned axis) {
      if (axis == sizes.size()) {
        body(indices);
        return;
      }
      auto loop = builder.create<scf::ForOp>(loc, constant(loc, 0), sizes[axis], constant(loc, 1));
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      indices.push_back(loop.getInductionVar());
      visit(axis + 1);
      indices.pop_back();
    };
    visit(0);
  }

  Value element(Value value, ValueRange coordinates, Location loc) {
    auto memory = dyn_cast<MemRefType>(value.getType());
    if (!memory)
      return value;
    SmallVector<Value> indices;
    unsigned start = coordinates.size() - memory.getRank();
    for (int64_t axis = 0; axis < memory.getRank(); ++axis)
      indices.push_back(memory.getDimSize(axis) == 1 ? constant(loc, 0) : coordinates[start + axis]);
    return builder.create<memref::LoadOp>(loc, value, indices);
  }

  FailureOr<Value> indexed(Operation *operation) {
    auto fact = analysis.indexRelation(operation);
    if (failed(fact))
      return failure();
    Value source = values.lookup(fact->source);
    auto type = dyn_cast<MemRefType>(source.getType());
    SmallVector<OpFoldResult> offsets, sizes, strides;
    SmallVector<int64_t> resultShape;
    for (const IndexTermFact &term : fact->terms) {
      if (term.kind == 3 && term.operands.size() == 1) {
        offsets.push_back(values.lookup(term.operands[0]));
        sizes.push_back(builder.getIndexAttr(1));
      } else if (term.kind == 2 && !term.staticValues.empty() && term.staticValues[0]) {
        offsets.push_back(builder.getIndexAttr(*term.staticValues[0]));
        sizes.push_back(builder.getIndexAttr(1));
      } else if (term.kind == 4 && term.operands.size() == 1 && domains.count(term.operands[0])) {
        Domain domain = domains.lookup(term.operands[0]);
        if (!matchPattern(domain.begin, m_Zero()) || !matchPattern(domain.step, m_One())) {
          operation->emitError("CPU indexed domain requires zero begin and unit stride");
          return failure();
        }
        offsets.push_back(builder.getIndexAttr(0));
        sizes.push_back(domain.end);
        resultShape.push_back(ShapedType::kDynamic);
      } else {
        operation->emitError("CPU construction does not implement this index relation");
        return failure();
      }
      strides.push_back(builder.getIndexAttr(1));
    }
    if (offsets.size() != static_cast<size_t>(type.getRank())) {
      operation->emitError("CPU indexed access must resolve every source axis");
      return failure();
    }
    if (resultShape.empty()) {
      SmallVector<Value> indices;
      for (OpFoldResult offset : offsets)
        indices.push_back(isa<Value>(offset) ? cast<Value>(offset)
                           : constant(operation->getLoc(), cast<IntegerAttr>(cast<Attribute>(offset)).getInt()));
      if (isa<ViewLoadOp>(operation))
        return Value(builder.create<memref::LoadOp>(operation->getLoc(), source, indices));
    }
    auto resultType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        resultShape, type, offsets, sizes, strides));
    return Value(builder.create<memref::SubViewOp>(operation->getLoc(), resultType,
                                                 source, offsets, sizes, strides));
  }

  FailureOr<Value> arithmetic(Operation *operation, ValueRange arguments) {
    Location loc = operation->getLoc();
    if (auto binary = dyn_cast<BinaryOp>(operation)) {
      if (binary.getApproximate() || binary.getFlushToZero())
        return binary.emitError("CPU approximate arithmetic is not implemented"), failure();
      Value a = arguments[0], b = arguments[1];
      bool fp = isa<FloatType>(a.getType());
      switch (binary.getOperatorKind()) {
      case BinaryOperator::Add:
        return fp ? Value(builder.create<arith::AddFOp>(loc, a, b)) : Value(builder.create<arith::AddIOp>(loc, a, b));
      case BinaryOperator::Subtract:
        return fp ? Value(builder.create<arith::SubFOp>(loc, a, b)) : Value(builder.create<arith::SubIOp>(loc, a, b));
      case BinaryOperator::Multiply:
        return fp ? Value(builder.create<arith::MulFOp>(loc, a, b)) : Value(builder.create<arith::MulIOp>(loc, a, b));
      case BinaryOperator::TrueDivide:
        if (fp) return Value(builder.create<arith::DivFOp>(loc, a, b));
        break;
      default: break;
      }
    } else if (auto unary = dyn_cast<UnaryOp>(operation)) {
      if (unary.getApproximate() || unary.getFlushToZero())
        return unary.emitError("CPU approximate arithmetic is not implemented"), failure();
      switch (unary.getOperatorKind()) {
      case UnaryOperator::Rsqrt: {
        Value root = builder.create<math::SqrtOp>(loc, arguments[0]);
        Value one = builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(1.0));
        return Value(builder.create<arith::DivFOp>(loc, one, root));
      }
      case UnaryOperator::Sqrt: return Value(builder.create<math::SqrtOp>(loc, arguments[0]));
      case UnaryOperator::Negate: return Value(builder.create<arith::NegFOp>(loc, arguments[0]));
      default: break;
      }
    }
    operation->emitError("CPU construction does not implement this arithmetic operation");
    return failure();
  }

  LogicalResult pointwise(Operation *operation) {
    Type resultType = operation->getResult(0).getType();
    auto tensor = dyn_cast<RankedTensorType>(resultType);
    SmallVector<Value> arguments;
    for (Value input : operation->getOperands())
      arguments.push_back(values.lookup(input));
    if (!tensor || tensor.getRank() == 0) {
      auto result = arithmetic(operation, arguments);
      if (failed(result)) return failure();
      values.map(operation->getResult(0), *result);
      return success();
    }
    auto sizes = extents(tensor, operation->getLoc());
    if (failed(sizes)) return failure();
    Value output = allocate(tensor, *sizes, operation->getLoc());
    LogicalResult status = success();
    elements(operation->getLoc(), *sizes, [&](ValueRange indices) {
      SmallVector<Value> scalars;
      for (Value input : arguments)
        scalars.push_back(element(input, indices, operation->getLoc()));
      auto result = arithmetic(operation, scalars);
      if (failed(result)) { status = failure(); return; }
      builder.create<memref::StoreOp>(operation->getLoc(), *result, output, indices);
    });
    values.map(operation->getResult(0), output);
    return status;
  }

  LogicalResult reduce(ReduceOp operation) {
    if (operation.getSourceCount() != 1 || operation.getIdentityCount() != 1 ||
        operation.getCaptureCount() != 0 || operation.getAxes().size() != 1 ||
        cast<IntegerAttr>(operation.getAxes()[0]).getInt() != 0)
      return operation.emitError("CPU reduction currently requires one uncaptured axis");
    Value input = values.lookup(operation.getInputs()[0]);
    auto type = dyn_cast<MemRefType>(input.getType());
    if (!type || type.getRank() != 1)
      return operation.emitError("CPU reduction currently requires a rank-one source");
    Value initial = values.lookup(operation.getInputs()[1]);
    Location loc = operation.getLoc();
    auto loop = builder.create<scf::ForOp>(loc, constant(loc, 0),
        builder.create<memref::DimOp>(loc, input, 0), constant(loc, 1), ValueRange{initial});
    loop->setAttr("cpu.ordered_reassociation", builder.getUnitAttr());
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      Value current = builder.create<memref::LoadOp>(loc, input, ValueRange{loop.getInductionVar()});
      Block &combine = operation.getCombine().front();
      values.map(combine.getArgument(0), loop.getRegionIterArgs()[0]);
      values.map(combine.getArgument(1), current);
      for (Operation &nested : combine.without_terminator())
        if (failed(lowerOperation(&nested))) return failure();
      Value result = values.lookup(combine.getTerminator()->getOperand(0));
      builder.create<scf::YieldOp>(loc, result);
    }
    values.map(operation.getResults()[0], loop.getResult(0));
    return success();
  }

  LogicalResult contract(ContractOp operation) {
    auto lhsType = cast<RankedTensorType>(operation.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(operation.getRhs().getType());
    auto resultType = cast<RankedTensorType>(operation.getResult().getType());
    auto pairs = operation.getReduce();
    if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || pairs.size() != 1 ||
        !operation.getBatch().empty() ||
        cast<IntegerAttr>(cast<ArrayAttr>(pairs[0])[0]).getInt() != 1 ||
        cast<IntegerAttr>(cast<ArrayAttr>(pairs[0])[1]).getInt() != 0 ||
        !lhsType.getElementType().isF32() || !rhsType.getElementType().isF32() ||
        !resultType.getElementType().isF32())
      return operation.emitError("CPU construction supports ordinary rank-two f32 contraction");
    Location loc = operation.getLoc();
    auto sizes = extents(resultType, loc);
    if (failed(sizes)) return failure();
    Value output = allocate(resultType, *sizes, loc);
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(0.0));
    builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{output});
    AffineExpr m, n, k;
    bindDims(builder.getContext(), m, n, k);
    SmallVector<AffineMap> maps = {
        AffineMap::get(3, 0, {m, k}, builder.getContext()),
        AffineMap::get(3, 0, {k, n}, builder.getContext()),
        AffineMap::get(3, 0, {m, n}, builder.getContext())};
    SmallVector<utils::IteratorType> iterators = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::reduction};
    builder.create<linalg::GenericOp>(loc,
        ValueRange{values.lookup(operation.getLhs()), values.lookup(operation.getRhs())},
        ValueRange{output}, maps, iterators,
        [](OpBuilder &b, Location loc, ValueRange arguments) {
          Value value = b.create<math::FmaOp>(loc, arguments[0], arguments[1], arguments[2]);
          b.create<linalg::YieldOp>(loc, value);
        });
    values.map(operation.getResult(), output);
    return success();
  }

  LogicalResult lowerBlock(Block &block) {
    allocations.emplace_back();
    for (Operation &operation : block.without_terminator())
      if (failed(lowerOperation(&operation))) return failure();
    for (Value allocation : llvm::reverse(allocations.back()))
      builder.create<memref::DeallocOp>(block.getParentOp()->getLoc(), allocation);
    allocations.pop_back();
    return success();
  }

  LogicalResult lowerOperation(Operation *operation) {
    Location loc = operation->getLoc();
    if (auto op = dyn_cast<ConstantOp>(operation)) {
      Type type = op.getResult().getType();
      TypedAttr attribute;
      if (auto integer = dyn_cast<IntegerAttr>(op.getValue())) {
        attribute = builder.getIntegerAttr(type, integer.getValue());
      } else if (auto floating = dyn_cast<FloatAttr>(op.getValue())) {
        llvm::APFloat value = floating.getValue();
        bool losesInformation;
        value.convert(cast<FloatType>(type).getFloatSemantics(),
                      llvm::APFloat::rmNearestTiesToEven, &losesInformation);
        attribute = FloatAttr::get(type, value);
      } else return op.emitError("CPU constant payload is not a scalar numeric literal");
      values.map(op.getResult(), builder.create<arith::ConstantOp>(loc, type, attribute));
    } else if (auto op = dyn_cast<DimOp>(operation)) {
      if (!dimensions.count(op.getDimension()))
        return op.emitError("CPU dimension has no runtime ABI binding");
      values.map(op.getResult(), dimensions.lookup(op.getDimension()));
    } else if (auto op = dyn_cast<DomainOp>(operation)) {
      Domain domain{values.lookup(op.getBounds()[0]), values.lookup(op.getBounds()[1]),
                    op.getBounds().size() == 3 ? values.lookup(op.getBounds()[2]) : constant(loc, 1)};
      if (!matchPattern(domain.begin, m_Zero()) || !matchPattern(domain.step, m_One()))
        return op.emitError("CPU construction currently supports zero-based unit-step domains");
      domains[op.getResult()] = domain;
      dimensions[cast<IntegerAttr>(op.getExtentDimensions()[0]).getInt()] = domain.end;
    } else if (auto op = dyn_cast<ParallelOp>(operation)) {
      if (!domains.count(op.getSource()))
        return op.emitError("CPU parallel source is not a supported domain");
      Domain domain = domains.lookup(op.getSource());
      auto parallel = builder.create<scf::ParallelOp>(loc, ValueRange{domain.begin},
          ValueRange{domain.end}, ValueRange{domain.step});
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(parallel.getBody());
      values.map(op.getBody().front().getArgument(0), parallel.getInductionVars()[0]);
      return lowerBlock(op.getBody().front());
    } else if (auto op = dyn_cast<ViewLoadOp>(operation)) {
      if (cast<ViewType>(op.getInputs()[0].getType()).getAccess() != 0)
        return op.emitError("CPU construction does not implement reads from writable external views");
      if (op.getValidOperandIndex() || op.getFillOperandIndex())
        return op.emitError("CPU predicated source loads are not implemented");
      auto value = indexed(operation);
      if (failed(value)) return failure();
      values.map(op.getResult(), *value);
    } else if (auto op = dyn_cast<ViewStoreOp>(operation)) {
      auto destination = indexed(operation);
      if (failed(destination)) return failure();
      Value input = values.lookup(op.getInputs()[op.getValueOperandIndex()]);
      if (isa<MemRefType>(input.getType()))
        builder.create<memref::CopyOp>(loc, input, *destination);
      else
        builder.create<memref::StoreOp>(loc, input, *destination, ValueRange{});
    } else if (isa<BinaryOp, UnaryOp>(operation)) {
      return pointwise(operation);
    } else if (auto op = dyn_cast<BroadcastOp>(operation)) {
      auto tensor = cast<RankedTensorType>(op.getResult().getType());
      auto sizes = extents(tensor, loc);
      if (failed(sizes)) return failure();
      Value input = values.lookup(op.getInputs()[0]);
      Value output = allocate(tensor, *sizes, loc);
      elements(loc, *sizes, [&](ValueRange indices) {
        builder.create<memref::StoreOp>(loc, element(input, indices, loc), output, indices);
      });
      values.map(op.getResult(), output);
    } else if (auto op = dyn_cast<ReduceOp>(operation)) {
      return reduce(op);
    } else if (auto op = dyn_cast<ContractOp>(operation)) {
      return contract(op);
    } else if (operation->getName().getDialectNamespace() == "arith") {
      builder.clone(*operation, values);
    } else {
      return operation->emitError("CPU construction does not implement this canonical operation");
    }
    return success();
  }

  CanonicalKernelAnalysis analysis;
  ModuleOp module;
  OpBuilder builder;
  func::FuncOp function;
  IRMapping values;
  llvm::DenseMap<int64_t, Value> dimensions;
  llvm::DenseMap<Value, Domain> domains;
  SmallVector<SmallVector<Value>> allocations;
};

}

LogicalResult lowerCanonicalKIRToCPU(ModuleOp module) {
  CanonicalKernelAnalysis analysis(module);
  if (failed(analysis.verify())) return failure();
  auto physical = OwningOpRef<ModuleOp>(ModuleOp::create(module.getLoc()));
  (*physical)->setAttr("cpu.execution_family", StringAttr::get(module.getContext(), "cpu"));
  Construction construction(module, *physical);
  unsigned count = 0;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (++count != 1)
      return function.emitError("CPU construction requires an inlined single kernel");
    if (failed(construction.lower(function))) return failure();
  }
  if (count == 0) return module.emitError("CPU construction found no kernel");
  if (failed(mlir::verify(*physical))) return failure();
  module.getBodyRegion().takeBody(physical->getBodyRegion());
  module->setAttrs((*physical)->getAttrs());
  return success();
}

}
