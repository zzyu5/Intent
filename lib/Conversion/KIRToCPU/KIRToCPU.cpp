#include "Intent/Conversion/KIRToCPU/KIRToCPU.h"
#include "Intent/Analysis/CanonicalKernel.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/RegionProgram.h"
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
#include <functional>

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
      if (auto view = dyn_cast<ViewType>(argument.getType())) {
        auto tensor = cast<RankedTensorType>(view.getTensor());
        if ((!tensor.getElementType().isF32() && !tensor.getElementType().isUnsignedInteger(8)) ||
            view.getAccess() == 2)
          return source.emitError("CPU construction supports f32/u8 In/Out views; InOut is not implemented");
        auto memory = MemRefType::get(tensor.getShape(), tensor.getElementType());
        if (view.getConstraints().getHasStrides()) {
          if (view.getConstraints().getStrides().size() != static_cast<size_t>(tensor.getRank()))
            return source.emitError("CPU declared stride constraints must cover the view rank");
          SmallVector<int64_t> strides;
          int64_t offset;
          if (failed(memory.getStridesAndOffset(strides, offset)))
            return source.emitError("CPU contiguous view has no derived strides");
          for (auto [constraint, stride] : llvm::zip(view.getConstraints().getStrides(), strides)) {
            if (isa<UnitAttr>(constraint)) continue;
            auto fixed = dyn_cast<IntegerAttr>(constraint);
            if (!fixed || ShapedType::isDynamic(stride) || fixed.getInt() != stride)
              return source.emitError("CPU contiguous ABI cannot discharge this declared stride constraint");
          }
        }
        auto shape = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
        if (!shape)
          return source.emitError("CPU view is missing canonical dimension identities");
        types.push_back(memory);
        interface.push_back(cpu::ViewArgumentAttr::get(builder.getContext(),
            parameter.getName(), tensor.getElementType(),
            builder.getDenseI64ArrayAttr(tensor.getShape()), shape.getDimensions(),
            view.getAccess(), view.getConstraints().getAlias(),
            view.getConstraints().getNoalias()));
      } else if (argument.getType().isF32() || argument.getType().isIndex() ||
                 argument.getType().isInteger(64)) {
        types.push_back(argument.getType());
        interface.push_back(cpu::ScalarArgumentAttr::get(builder.getContext(),
            parameter.getName(), argument.getType()));
      } else {
        return source.emitError("CPU construction does not implement this parameter type");
      }
    }
    builder.setInsertionPointToEnd(module.getBody());
    function = builder.create<func::FuncOp>(source.getLoc(), source.getName(),
                                           builder.getFunctionType(types, {}));
    function->setAttr("intent_cpu.interface", cpu::InterfaceAttr::get(
        builder.getContext(), builder.getArrayAttr(interface), true, true));
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

  FailureOr<Value> indexValue(Value value, Location loc) {
    if (value.getType().isIndex()) return value;
    if (value.getType().isSignlessInteger(64))
      return Value(builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), value));
    return emitError(loc, "CPU coordinates require index or signed 64-bit integer values"), failure();
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

  ArrayAttr componentTypes(Type type) {
    if (auto record = dyn_cast<RecordType>(type)) return record.getFieldTypes();
    if (auto tuple = dyn_cast<intent::TupleType>(type)) return tuple.getComponentTypes();
    return {};
  }

  void flattenTypes(Type type, SmallVectorImpl<Type> &result) {
    if (auto components = componentTypes(type)) {
      for (Attribute component : components) flattenTypes(cast<TypeAttr>(component).getValue(), result);
    } else result.push_back(type);
  }

  SmallVector<Value> flattened(Value value) {
    if (componentTypes(value.getType())) return products.at(value);
    return {values.lookup(value)};
  }

  SmallVector<Value> flattened(ValueRange inputs) {
    SmallVector<Value> result;
    for (Value value : inputs) llvm::append_range(result, flattened(value));
    return result;
  }

  void bindProduct(Value original, ValueRange components) {
    if (componentTypes(original.getType())) products[original] = llvm::to_vector(components);
    else values.map(original, components.front());
  }

  void bindDimensions(Type original, Value value, Location loc) {
    auto tensor = dyn_cast<RankedTensorType>(original);
    if (!tensor || !isa<MemRefType>(value.getType())) return;
    auto identities = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions();
    for (auto [axis, identity] : llvm::enumerate(identities.asArrayRef()))
      dimensions[identity] = builder.create<memref::DimOp>(loc, value, axis);
  }

  SmallVector<Value> makeSlots(TypeRange types, Location loc) {
    SmallVector<Value> result;
    for (Type type : types) {
      SmallVector<Type> leaves;
      flattenTypes(type, leaves);
      for (Type leaf : leaves) {
        auto tensor = dyn_cast<RankedTensorType>(leaf);
        if (!tensor) tensor = RankedTensorType::get({}, leaf);
        auto sizes = extents(tensor, loc);
        if (failed(sizes)) return {};
        result.push_back(allocate(tensor, *sizes, loc));
      }
    }
    return result;
  }

  Value slotValue(Value slot, Location loc) {
    if (cast<MemRefType>(slot.getType()).getRank() == 0)
      return builder.create<memref::LoadOp>(loc, slot, ValueRange{});
    return slot;
  }

  void copyToSlot(Value value, Value slot, Location loc) {
    if (isa<MemRefType>(value.getType())) builder.create<memref::CopyOp>(loc, value, slot);
    else if (cast<MemRefType>(slot.getType()).getRank())
      builder.create<linalg::FillOp>(loc, ValueRange{value}, ValueRange{slot});
    else builder.create<memref::StoreOp>(loc, value, slot, ValueRange{});
  }

  void bindSlots(ValueRange originals, ValueRange slots, Location loc) {
    unsigned offset = 0;
    for (Value original : originals) {
      SmallVector<Type> leaves;
      flattenTypes(original.getType(), leaves);
      SmallVector<Value> parts;
      for (Type leaf : leaves) {
        Value value = slots[offset++];
        bindDimensions(leaf, value, loc);
        parts.push_back(isa<RankedTensorType>(leaf) ? value : slotValue(value, loc));
      }
      bindProduct(original, parts);
    }
  }

  ArrayAttr fieldPaths(TypeRange types) {
    SmallVector<Attribute> fields;
    std::function<void(Type, std::string)> visit = [&](Type type, std::string path) {
      if (auto components = componentTypes(type)) {
        auto record = dyn_cast<RecordType>(type);
        for (auto [index, component] : llvm::enumerate(components)) {
          auto name = record ? cast<StringAttr>(record.getFieldNames()[index]).getValue().str() : std::to_string(index);
          visit(cast<TypeAttr>(component).getValue(), path + "." + name);
        }
      } else fields.push_back(builder.getStringAttr(path));
    };
    for (auto [index, type] : llvm::enumerate(types)) visit(type, std::to_string(index));
    return builder.getArrayAttr(fields);
  }

  LogicalResult helper(Region &original, Region &target, TypeRange inputTypes,
                       TypeRange destinationTypes) {
    auto savedDimensions = dimensions;
    OpBuilder::InsertionGuard guard(builder);
    Block *body = new Block;
    target.push_back(body);
    for (Type type : inputTypes) body->addArgument(type, original.getLoc());
    for (Type type : destinationTypes) body->addArgument(type, original.getLoc());
    builder.setInsertionPointToStart(body);
    unsigned cursor = 0;
    for (Value argument : original.front().getArguments()) {
      SmallVector<Type> leaves;
      flattenTypes(argument.getType(), leaves);
      SmallVector<Value> parts;
      for (Type leaf : leaves) {
        Value value = body->getArgument(cursor++);
        bindDimensions(leaf, value, original.getLoc());
        if (!isa<RankedTensorType>(leaf) && isa<MemRefType>(value.getType()))
          value = slotValue(value, original.getLoc());
        parts.push_back(value);
      }
      bindProduct(argument, parts);
    }
    if (cursor != inputTypes.size()) return original.getParentOp()->emitError("CPU helper argument partition mismatch");
    allocations.emplace_back();
    for (Operation &operation : original.front().without_terminator())
      if (failed(lowerOperation(&operation))) return failure();
    auto results = flattened(original.front().getTerminator()->getOperands());
    if (results.size() != destinationTypes.size())
      return original.getParentOp()->emitError("CPU helper destination schema mismatch");
    for (auto [value, slot] : llvm::zip(results, body->getArguments().drop_front(cursor)))
      copyToSlot(value, slot, original.getLoc());
    for (Value allocation : llvm::reverse(allocations.back())) builder.create<memref::DeallocOp>(original.getLoc(), allocation);
    allocations.pop_back();
    builder.create<cpu::RegionYieldOp>(original.getLoc());
    dimensions = std::move(savedDimensions);
    return success();
  }

  LogicalResult region(Operation *operation) {
    const bool scan = isa<RegionScanOp>(operation);
    auto count = [&](StringRef name) { return operation->getAttrOfType<IntegerAttr>(name).getInt(); };
    int64_t sourceCount = count("source_count"), identityCount = count("identity_count");
    int64_t stateCount = scan ? count("state_count") : 0, outputCount = scan ? count("output_count") : 0;
    int64_t axis = count("axis");
    auto original = operation->getOperands();
    auto sources = flattened(original.take_front(sourceCount));
    auto identityValues = flattened(original.slice(sourceCount, identityCount));
    auto stateValues = flattened(original.slice(sourceCount + identityCount, stateCount));
    auto captures = flattened(original.drop_front(sourceCount + identityCount + stateCount));
    Location loc = operation->getLoc();
    auto destinations = makeSlots(operation->getResultTypes(), loc);
    if (destinations.empty()) return failure();
    SmallVector<Type> identityTypes;
    for (Type type : original.slice(sourceCount, identityCount).getTypes()) flattenTypes(type, identityTypes);
    SmallVector<Type> summarySlots;
    for (Type type : identityTypes) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      summarySlots.push_back(tensor ? MemRefType::get(tensor.getShape(), tensor.getElementType()) : MemRefType::get({}, type));
    }
    auto outputFields = fieldPaths(TypeRange(operation->getResultTypes()).take_front(outputCount));
    SmallVector<int64_t> outputAxes;
    if (scan) {
      Block &emit = operation->getRegion(3).front();
      SmallVector<Type> sourceLeaves; flattenTypes(emit.getArgument(0).getType(), sourceLeaves);
      auto sourceType = cast<RankedTensorType>(sourceLeaves.front());
      int64_t member = cast<TensorShapeAttr>(sourceType.getEncoding()).getDimensions()[axis];
      for (Type type : emit.getTerminator()->getOperandTypes()) {
        SmallVector<Type> leaves; flattenTypes(type, leaves);
        for (Type leaf : leaves) {
          auto tensor = cast<RankedTensorType>(leaf);
          auto axes = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions().asArrayRef();
          auto found = llvm::find(axes, member);
          if (found == axes.end()) return operation->emitError("CPU region emit has no source member axis");
          outputAxes.push_back(found - axes.begin());
        }
      }
    }
    SmallVector<Type> stateSlots;
    if (scan) llvm::append_range(stateSlots, TypeRange(destinations).drop_front(outputFields.size()));
    SmallVector<Value> inputs(sources);
    llvm::append_range(inputs, identityValues); llvm::append_range(inputs, stateValues);
    llvm::append_range(inputs, captures); llvm::append_range(inputs, destinations);
    OperationState state(loc, scan ? cpu::RegionScanOp::getOperationName() : cpu::RegionFoldOp::getOperationName());
    state.addOperands(inputs);
    state.addAttribute("axis", builder.getI64IntegerAttr(axis));
    state.addAttribute("source_count", builder.getI64IntegerAttr(sources.size()));
    state.addAttribute("identity_count", builder.getI64IntegerAttr(identityValues.size()));
    state.addAttribute("state_count", builder.getI64IntegerAttr(stateValues.size()));
    state.addAttribute("capture_count", builder.getI64IntegerAttr(captures.size()));
    state.addAttribute("output_count", builder.getI64IntegerAttr(outputFields.size()));
    state.addAttribute("summary_fields", fieldPaths(original.slice(sourceCount, identityCount).getTypes()));
    state.addAttribute("state_fields", fieldPaths(original.slice(sourceCount + identityCount, stateCount).getTypes()));
    state.addAttribute("output_axes", builder.getDenseI64ArrayAttr(outputAxes));
    for (unsigned i = 0; i < operation->getNumRegions(); ++i) state.addRegion();
    Operation *target = builder.create(state);
    SmallVector<Type> sliceTypes;
    for (Value source : sources) {
      auto type = cast<MemRefType>(source.getType());
      SmallVector<int64_t> shape(type.getShape());
      shape[axis] = ShapedType::kDynamic;
      auto layout = StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic,
          SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic));
      sliceTypes.push_back(MemRefType::get(shape, type.getElementType(), layout));
    }
    SmallVector<Type> summarizeInputs(sliceTypes);
    llvm::append_range(summarizeInputs, TypeRange(captures));
    if (failed(helper(operation->getRegion(0), target->getRegion(0), summarizeInputs, summarySlots))) return failure();
    SmallVector<Type> combineInputs(summarySlots);
    llvm::append_range(combineInputs, summarySlots);
    if (failed(helper(operation->getRegion(1), target->getRegion(1), combineInputs, summarySlots))) return failure();
    if (scan) {
      SmallVector<Type> applyInputs(summarySlots);
      llvm::append_range(applyInputs, stateSlots);
      if (failed(helper(operation->getRegion(2), target->getRegion(2), applyInputs, stateSlots))) return failure();
      SmallVector<Type> emitInputs(sliceTypes), emitSlots;
      llvm::append_range(emitInputs, stateSlots); llvm::append_range(emitInputs, TypeRange(captures));
      for (auto [output, outputAxis] : llvm::zip(ValueRange(destinations).take_front(outputFields.size()), outputAxes)) {
        auto type = cast<MemRefType>(output.getType());
        SmallVector<int64_t> shape(type.getShape()); shape[outputAxis] = ShapedType::kDynamic;
        emitSlots.push_back(MemRefType::get(shape, type.getElementType(),
            StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic,
                SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic))));
      }
      if (failed(helper(operation->getRegion(3), target->getRegion(3), emitInputs, emitSlots))) return failure();
    }
    bindSlots(operation->getResults(), destinations, loc);
    return success();
  }

  SmallVector<AffineMap> pointwiseMaps(ValueRange inputs, int64_t rank) {
    SmallVector<AffineMap> maps;
    for (Value input : inputs) {
      SmallVector<AffineExpr> axes;
      if (auto memory = dyn_cast<MemRefType>(input.getType()))
        for (int64_t axis = 0; axis < memory.getRank(); ++axis)
          axes.push_back(memory.getDimSize(axis) == 1
              ? builder.getAffineConstantExpr(0)
              : builder.getAffineDimExpr(rank - memory.getRank() + axis));
      maps.push_back(AffineMap::get(rank, 0, axes, builder.getContext()));
    }
    maps.push_back(builder.getMultiDimIdentityMap(rank));
    return maps;
  }

  FailureOr<Value> indexed(Operation *operation) {
    auto fact = analysis.indexRelation(operation);
    if (failed(fact))
      return failure();
    Value source = values.lookup(fact->source);
    auto type = dyn_cast<MemRefType>(source.getType());
    if (llvm::any_of(fact->terms, [](const IndexTermFact &term) { return term.kind == 1; })) {
      auto tensor = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
      if (!tensor || llvm::any_of(fact->terms, [](const IndexTermFact &term) { return term.kind != 0 && term.kind != 1; }))
        return operation->emitError("CPU inserted index axes require a full tensor projection"), failure();
      auto shape = extents(tensor, operation->getLoc());
      if (failed(shape)) return failure();
      Value output = allocate(tensor, *shape, operation->getLoc());
      SmallVector<AffineExpr> coordinates;
      for (auto [axis, term] : llvm::enumerate(fact->terms))
        if (term.kind == 0) coordinates.push_back(builder.getAffineDimExpr(axis));
      builder.create<linalg::GenericOp>(operation->getLoc(), ValueRange{source}, ValueRange{output},
          SmallVector<AffineMap>{AffineMap::get(tensor.getRank(), 0, coordinates, builder.getContext()), builder.getMultiDimIdentityMap(tensor.getRank())},
          SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
          [](OpBuilder &nested, Location loc, ValueRange scalars) { nested.create<linalg::YieldOp>(loc, scalars[0]); });
      return output;
    }
    SmallVector<OpFoldResult> offsets, sizes, strides;
    SmallVector<int64_t> resultShape;
    for (const IndexTermFact &term : fact->terms) {
      if (term.kind == 0) {
        unsigned axis = *term.sourceAxis;
        offsets.push_back(builder.getIndexAttr(0));
        sizes.push_back(type.isDynamicDim(axis) ? OpFoldResult(builder.create<memref::DimOp>(operation->getLoc(), source, axis).getResult()) : builder.getIndexAttr(type.getDimSize(axis)));
        resultShape.push_back(type.getDimSize(axis));
      } else if (term.kind == 3 && term.operands.size() == 1) {
        auto coordinate = indexValue(values.lookup(term.operands[0]), operation->getLoc());
        if (failed(coordinate)) return failure();
        offsets.push_back(*coordinate);
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
        llvm::APInt extent;
        if (matchPattern(domain.end, m_ConstantInt(&extent))) {
          sizes.push_back(builder.getIndexAttr(extent.getSExtValue()));
          resultShape.push_back(extent.getSExtValue());
        } else {
          sizes.push_back(domain.end);
          resultShape.push_back(ShapedType::kDynamic);
        }
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
      if (isa<ViewLoadOp, GatherOp>(operation))
        return Value(builder.create<memref::LoadOp>(operation->getLoc(), source, indices));
    }
    auto resultType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        resultShape, type, offsets, sizes, strides));
    return Value(builder.create<memref::SubViewOp>(operation->getLoc(), resultType,
                                                 source, offsets, sizes, strides));
  }

  FailureOr<Value> arithmetic(Operation *operation, ValueRange arguments, OpBuilder &builder) {
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
      case BinaryOperator::MaximumNum:
        if (fp) return Value(builder.create<arith::MaxNumFOp>(loc, a, b));
        break;
      case BinaryOperator::Maximum:
        return fp ? Value(builder.create<arith::MaximumFOp>(loc, a, b)) : Value(builder.create<arith::MaxSIOp>(loc, a, b));
      case BinaryOperator::Minimum:
        return fp ? Value(builder.create<arith::MinimumFOp>(loc, a, b)) : Value(builder.create<arith::MinSIOp>(loc, a, b));
      case BinaryOperator::LogicalAnd:
      case BinaryOperator::BitwiseAnd: return Value(builder.create<arith::AndIOp>(loc, a, b));
      case BinaryOperator::LogicalOr:
      case BinaryOperator::BitwiseOr: return Value(builder.create<arith::OrIOp>(loc, a, b));
      case BinaryOperator::BitwiseXor: return Value(builder.create<arith::XOrIOp>(loc, a, b));
      default: break;
      }
    } else if (auto unary = dyn_cast<UnaryOp>(operation)) {
      if (unary.getApproximate() || unary.getFlushToZero())
        return unary.emitError("CPU approximate arithmetic is not implemented"), failure();
      switch (unary.getOperatorKind()) {
      case UnaryOperator::Rsqrt: return Value(builder.create<math::RsqrtOp>(loc, arguments[0]));
      case UnaryOperator::Sqrt: return Value(builder.create<math::SqrtOp>(loc, arguments[0]));
      case UnaryOperator::Exp: return Value(builder.create<math::ExpOp>(loc, arguments[0]));
      case UnaryOperator::Exp2: return Value(builder.create<math::Exp2Op>(loc, arguments[0]));
      case UnaryOperator::Negate: return Value(builder.create<arith::NegFOp>(loc, arguments[0]));
      case UnaryOperator::Not: return Value(builder.create<arith::XOrIOp>(loc, arguments[0],
          builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(true))));
      default: break;
      }
    } else if (auto compare = dyn_cast<CompareOp>(operation)) {
      static const arith::CmpFPredicate floating[] = {arith::CmpFPredicate::OEQ, arith::CmpFPredicate::UNE,
          arith::CmpFPredicate::OLT, arith::CmpFPredicate::OLE, arith::CmpFPredicate::OGT, arith::CmpFPredicate::OGE};
      static const arith::CmpIPredicate integer[] = {arith::CmpIPredicate::eq, arith::CmpIPredicate::ne,
          arith::CmpIPredicate::slt, arith::CmpIPredicate::sle, arith::CmpIPredicate::sgt, arith::CmpIPredicate::sge};
      unsigned predicate = static_cast<unsigned>(compare.getPredicate());
      return isa<FloatType>(arguments[0].getType())
          ? Value(builder.create<arith::CmpFOp>(loc, floating[predicate], arguments[0], arguments[1]))
          : Value(builder.create<arith::CmpIOp>(loc, integer[predicate], arguments[0], arguments[1]));
    } else if (isa<SelectOp>(operation)) {
      return Value(builder.create<arith::SelectOp>(loc, arguments[0], arguments[1], arguments[2]));
    } else if (isa<MaskOp>(operation)) {
      return Value(builder.create<arith::SelectOp>(loc, arguments[1], arguments[0], arguments[2]));
    } else if (auto cast = dyn_cast<CastOp>(operation)) {
      Type type = getElementTypeOrSelf(cast.getType());
      Type input = arguments[0].getType();
      if (input == type) return arguments[0];
      if (cast.getRounding()) return cast.emitError("CPU explicit-rounding cast is not implemented"), failure();
      if (input.isIndex() && type.isInteger(64)) return Value(builder.create<arith::IndexCastOp>(loc, type, arguments[0]));
      if (input.isInteger(64) && type.isIndex()) return Value(builder.create<arith::IndexCastOp>(loc, type, arguments[0]));
      if ((input.isIndex() || input.isInteger(64)) && type.isF32()) {
        Value value = input.isIndex() ? Value(builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), arguments[0])) : arguments[0];
        return Value(builder.create<arith::SIToFPOp>(loc, type, value));
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
    if (!tensor || scalarized) {
      auto result = arithmetic(operation, arguments, builder);
      if (failed(result)) return failure();
      values.map(operation->getResult(0), *result);
      return success();
    }
    auto sizes = extents(tensor, operation->getLoc());
    if (failed(sizes)) return failure();
    Value output = allocate(tensor, *sizes, operation->getLoc());
    LogicalResult status = success();
    builder.create<linalg::GenericOp>(operation->getLoc(), arguments,
        ValueRange{output}, pointwiseMaps(arguments, tensor.getRank()),
        SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &nested, Location loc, ValueRange scalars) {
          auto result = arithmetic(operation, scalars.take_front(arguments.size()), nested);
          if (failed(result)) { status = failure(); return; }
          nested.create<linalg::YieldOp>(loc, *result);
        });
    values.map(operation->getResult(0), output);
    return status;
  }

  LogicalResult reduce(ReduceOp operation) {
    auto first = dyn_cast<MemRefType>(flattened(operation.getInputs()[0]).front().getType());
    if (operation.getSourceCount() != 1 || operation.getIdentityCount() != 1 ||
        operation.getCaptureCount() != 0 || operation.getAxes().size() != 1 ||
        cast<IntegerAttr>(operation.getAxes()[0]).getInt() != 0 ||
        !first || first.getRank() != 1 || !first.getElementType().isF32())
      return structuredReduction(operation);
    Value input = values.lookup(operation.getInputs()[0]);
    auto type = dyn_cast<MemRefType>(input.getType());
    if (!type || type.getRank() != 1)
      return operation.emitError("CPU reduction currently requires a rank-one source");
    Value initial = values.lookup(operation.getInputs()[1]);
    Location loc = operation.getLoc();
    Value extent = builder.create<memref::DimOp>(loc, input, 0);
    auto reduction = builder.create<cpu::ReduceOp>(loc, initial.getType(),
        extent, initial, ValueRange{input},
        builder.getArrayAttr({AffineMapAttr::get(builder.getMultiDimIdentityMap(1))}),
        cpu::ReductionOrderAttr::get(builder.getContext(), false));
    Block *body = &reduction.getCombine().emplaceBlock();
    body->addArgument(initial.getType(), loc);
    body->addArgument(type.getElementType(), loc);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(body);
      Block &combine = operation.getCombine().front();
      values.map(combine.getArgument(0), body->getArgument(0));
      values.map(combine.getArgument(1), body->getArgument(1));
      for (Operation &nested : combine.without_terminator())
        if (failed(lowerOperation(&nested))) return failure();
      Value result = values.lookup(combine.getTerminator()->getOperand(0));
      builder.create<cpu::YieldOp>(loc, result);
      auto *combineOp = result.getDefiningOp();
      if (combineOp && isa<arith::AddFOp, arith::MaxNumFOp>(combineOp) &&
          body->getArgument(0).hasOneUse() &&
          llvm::is_contained(combineOp->getOperands(), body->getArgument(0)))
        reduction.setOrderAttr(cpu::ReductionOrderAttr::get(builder.getContext(), true));
    }
    values.map(operation.getResults()[0], reduction.getResult());
    return success();
  }

  LogicalResult structuredReduction(ReduceOp operation) {
    Block &combine = operation.getCombine().front();
    for (Operation &nested : combine.without_terminator())
      if (!isa<ConstantOp, BinaryOp, UnaryOp, CompareOp, SelectOp, MaskOp, CastOp,
               MakeRecordOp, MakeTupleOp, ExtractOp>(nested))
        return nested.emitError("CPU tensor reduction requires a pointwise combine; non-pointwise summary reduction is not implemented");
    auto sources = flattened(operation.getInputs().take_front(operation.getSourceCount()));
    auto identities = flattened(operation.getInputs().slice(operation.getSourceCount(), operation.getIdentityCount()));
    auto captures = flattened(operation.getInputs().drop_front(operation.getSourceCount() + operation.getIdentityCount()));
    if (sources.size() != identities.size()) return operation.emitError("CPU reduction source and state leaves differ");
    auto type = cast<MemRefType>(sources.front().getType());
    for (Value value : sources)
      if (cast<MemRefType>(value.getType()).getShape() != type.getShape())
        return operation.emitError("CPU product reduction with differing free shapes is not implemented");
    SmallVector<bool> reduced(type.getRank(), false);
    for (Attribute axis : operation.getAxes()) reduced[cast<IntegerAttr>(axis).getInt()] = true;
    SmallVector<AffineExpr> freeAxes;
    SmallVector<utils::IteratorType> iterators;
    for (unsigned axis = 0; axis < reduced.size(); ++axis) {
      iterators.push_back(reduced[axis] ? utils::IteratorType::reduction : utils::IteratorType::parallel);
      if (!reduced[axis]) freeAxes.push_back(builder.getAffineDimExpr(axis));
    }
    Location loc = operation.getLoc();
    auto outputs = makeSlots(operation.getResultTypes(), loc);
    if (outputs.size() != identities.size()) return failure();
    for (auto [identity, output] : llvm::zip(identities, outputs)) copyToSlot(identity, output, loc);
    SmallVector<Value> inputs(sources);
    llvm::append_range(inputs, captures);
    SmallVector<AffineMap> maps(sources.size(), builder.getMultiDimIdentityMap(type.getRank()));
    for (Value capture : captures) {
      SmallVector<AffineExpr> expressions;
      if (auto memory = dyn_cast<MemRefType>(capture.getType())) {
        if (memory.getRank() > static_cast<int64_t>(freeAxes.size())) return operation.emitError("CPU reduction capture is not free-axis aligned");
        for (int64_t axis = 0; axis < memory.getRank(); ++axis)
          expressions.push_back(memory.getDimSize(axis) == 1 ? builder.getAffineConstantExpr(0) : freeAxes[freeAxes.size() - memory.getRank() + axis]);
      }
      maps.push_back(AffineMap::get(type.getRank(), 0, expressions, builder.getContext()));
    }
    for (Value output : outputs) {
      if (cast<MemRefType>(output.getType()).getRank() != static_cast<int64_t>(freeAxes.size()))
        return operation.emitError("CPU reduction result does not retain the free axes");
      maps.push_back(AffineMap::get(type.getRank(), 0, freeAxes, builder.getContext()));
    }
    LogicalResult status = success();
    builder.create<linalg::GenericOp>(loc, inputs, outputs, maps, iterators,
        [&](OpBuilder &nestedBuilder, Location, ValueRange scalars) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(nestedBuilder.getInsertionBlock(), nestedBuilder.getInsertionPoint());
      SmallVector<Value> arguments(scalars.drop_front(inputs.size()));
      llvm::append_range(arguments, scalars.take_front(sources.size()));
      llvm::append_range(arguments, scalars.slice(sources.size(), captures.size()));
      unsigned offset = 0;
      for (Value argument : combine.getArguments()) {
        SmallVector<Type> leaves; flattenTypes(argument.getType(), leaves);
        bindProduct(argument, ValueRange(arguments).slice(offset, leaves.size()));
        offset += leaves.size();
      }
      scalarized = true;
      for (Operation &nested : combine.without_terminator())
        if (failed(lowerOperation(&nested))) { status = failure(); break; }
      scalarized = false;
      if (succeeded(status)) builder.create<linalg::YieldOp>(loc, flattened(combine.getTerminator()->getOperands()));
    });
    if (failed(status)) return failure();
    bindSlots(operation.getResults(), outputs, loc);
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

  FailureOr<Value> lowerResults(Block &block, ValueRange destinations, bool condition = false) {
    allocations.emplace_back();
    for (Operation &operation : block.without_terminator())
      if (failed(lowerOperation(&operation))) return failure();
    auto operands = block.getTerminator()->getOperands();
    Value predicate = condition ? values.lookup(operands.front()) : Value();
    auto results = flattened(condition ? operands.drop_front() : operands);
    if (results.size() != destinations.size())
      return block.getParentOp()->emitError("CPU ordered control result partition mismatch"), failure();
    for (auto [result, destination] : llvm::zip(results, destinations)) copyToSlot(result, destination, block.getParentOp()->getLoc());
    for (Value allocation : llvm::reverse(allocations.back())) builder.create<memref::DeallocOp>(block.getParentOp()->getLoc(), allocation);
    allocations.pop_back();
    return predicate;
  }

  LogicalResult orderedControl(Operation *operation) {
    Location loc = operation->getLoc();
    auto destinations = makeSlots(operation->getResultTypes(), loc);
    if (operation->getNumResults() && destinations.empty()) return failure();
    auto savedDimensions = dimensions;
    if (auto conditional = dyn_cast<IfOp>(operation)) {
      auto target = builder.create<scf::IfOp>(loc, values.lookup(conditional.getCondition()), true);
      for (auto [source, destination] : llvm::zip(operation->getRegions(), target->getRegions())) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&destination.front());
        if (failed(lowerResults(source.front(), destinations))) return failure();
        dimensions = savedDimensions;
      }
    } else {
      auto loop = dyn_cast<ForOp>(operation);
      auto initial = flattened(loop ? operation->getOperands().drop_front() : operation->getOperands());
      if (initial.size() != destinations.size()) return operation->emitError("CPU loop initial and result schemas differ");
      auto next = makeSlots(operation->getResultTypes(), loc);
      for (auto [value, destination] : llvm::zip(initial, destinations)) copyToSlot(value, destination, loc);
      auto advance = [&]() {
        for (auto [source, destination] : llvm::zip(next, destinations)) copyToSlot(source, destination, loc);
      };
      if (loop) {
        if (!domains.count(loop.getInputs().front())) return operation->emitError("CPU ordered for requires a supported domain");
        Domain domain = domains.lookup(loop.getInputs().front());
        auto target = builder.create<scf::ForOp>(loc, domain.begin, domain.end, domain.step);
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(target.getBody());
        Block &body = loop.getBody().front();
        values.map(body.getArgument(0), target.getInductionVar());
        bindSlots(body.getArguments().drop_front(), destinations, loc);
        if (failed(lowerResults(body, next))) return failure();
        advance();
      } else {
        auto target = builder.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
        target.getBefore().emplaceBlock(); target.getAfter().emplaceBlock();
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&target.getBefore().front());
          Block &before = operation->getRegion(0).front();
          bindSlots(before.getArguments(), destinations, loc);
          auto condition = lowerResults(before, next, true);
          if (failed(condition)) return failure();
          advance();
          builder.create<scf::ConditionOp>(loc, *condition, ValueRange{});
        }
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&target.getAfter().front());
          Block &after = operation->getRegion(1).front();
          bindSlots(after.getArguments(), destinations, loc);
          if (failed(lowerResults(after, next))) return failure();
          advance();
          builder.create<scf::YieldOp>(loc);
        }
      }
    }
    dimensions = std::move(savedDimensions);
    bindSlots(operation->getResults(), destinations, loc);
    return success();
  }

  bool alwaysValid(Operation *operation) {
    auto position = operation->getAttrOfType<IntegerAttr>("valid_operand_index");
    if (!position) return true;
    Value predicate = operation->getOperand(position.getInt());
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
      Value begin = values.lookup(op.getBounds()[0]);
      Value end = values.lookup(op.getBounds()[1]);
      Value step = op.getBounds().size() == 3 ? values.lookup(op.getBounds()[2]) : constant(loc, 1);
      if (!matchPattern(begin, m_Zero()) || !matchPattern(step, m_One()))
        return op.emitError("CPU construction currently supports zero-based unit-step domains");
      auto physicalBegin = indexValue(begin, loc), physicalEnd = indexValue(end, loc), physicalStep = indexValue(step, loc);
      if (failed(physicalBegin) || failed(physicalEnd) || failed(physicalStep)) return failure();
      Domain domain{constant(loc, 0), *physicalEnd, constant(loc, 1)};
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
    } else if (isa<IfOp, ForOp, WhileOp>(operation)) {
      return orderedControl(operation);
    } else if (auto op = dyn_cast<ViewLoadOp>(operation)) {
      if (cast<ViewType>(op.getInputs()[0].getType()).getAccess() != 0)
      return op.emitError("CPU construction does not implement reads from writable external views");
      if (!alwaysValid(operation))
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
    } else if (auto op = dyn_cast<GatherOp>(operation)) {
      if (!alwaysValid(operation))
        return op.emitError("CPU predicated tensor gathers are not implemented");
      auto value = indexed(operation);
      if (failed(value)) return failure();
      values.map(op.getResult(), *value);
    } else if (isa<BinaryOp, UnaryOp, CompareOp, SelectOp, CastOp, MaskOp>(operation)) {
      return pointwise(operation);
    } else if (auto op = dyn_cast<FullOp>(operation)) {
      auto tensor = cast<RankedTensorType>(op.getResult().getType());
      auto sizes = extents(tensor, loc);
      if (failed(sizes)) return failure();
      Value output = allocate(tensor, *sizes, loc);
      builder.create<linalg::FillOp>(loc, ValueRange{values.lookup(op.getInputs()[0])}, ValueRange{output});
      values.map(op.getResult(), output);
    } else if (auto op = dyn_cast<IndicesOp>(operation)) {
      if (!domains.count(op.getSource()) || op.getTensorAxis())
        return op.emitError("CPU indices requires an explicit one-dimensional domain");
      Domain domain = domains.lookup(op.getSource());
      auto type = cast<RankedTensorType>(op.getResult().getType());
      Value output = allocate(type, {domain.end}, loc);
      builder.create<linalg::GenericOp>(loc, ValueRange{}, ValueRange{output},
          SmallVector<AffineMap>{builder.getMultiDimIdentityMap(1)},
          SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
          [&](OpBuilder &nested, Location location, ValueRange) {
            Value coordinate = nested.create<linalg::IndexOp>(location, 0);
            if (!type.getElementType().isIndex()) coordinate = nested.create<arith::IndexCastOp>(location, type.getElementType(), coordinate);
            nested.create<linalg::YieldOp>(location, coordinate);
          });
      values.map(op.getResult(), output);
    } else if (isa<TransposeOp, ReshapeOp>(operation)) {
      Value input = values.lookup(operation->getOperand(0));
      auto source = cast<MemRefType>(input.getType());
      auto tensor = cast<RankedTensorType>(operation->getResult(0).getType());
      auto sizes = extents(tensor, loc);
      if (failed(sizes)) return failure();
      SmallVector<AffineExpr> coordinates(source.getRank());
      if (auto transpose = dyn_cast<TransposeOp>(operation)) {
        for (auto [axis, permuted] : llvm::enumerate(transpose.getPermutation()))
          coordinates[cast<IntegerAttr>(permuted).getInt()] = builder.getAffineDimExpr(axis);
      } else {
        unsigned next = 0;
        for (int64_t axis = 0; axis < source.getRank(); ++axis) {
          if (source.getDimSize(axis) == 1) { coordinates[axis] = builder.getAffineConstantExpr(0); continue; }
          while (next < tensor.getRank() && tensor.getDimSize(next) == 1) ++next;
          if (next == tensor.getRank() || source.getDimSize(axis) != tensor.getDimSize(next))
            return operation->emitError("CPU non-unit-axis reshape requires explicit contiguous regrouping, which is not implemented");
          coordinates[axis] = builder.getAffineDimExpr(next++);
        }
        while (next < tensor.getRank() && tensor.getDimSize(next) == 1) ++next;
        if (next != tensor.getRank()) return operation->emitError("CPU reshape cannot introduce non-unit axes");
      }
      Value output = allocate(tensor, *sizes, loc);
      builder.create<linalg::GenericOp>(loc, ValueRange{input}, ValueRange{output},
          SmallVector<AffineMap>{AffineMap::get(tensor.getRank(), 0, coordinates, builder.getContext()), builder.getMultiDimIdentityMap(tensor.getRank())},
          SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
          [](OpBuilder &nested, Location location, ValueRange args) { nested.create<linalg::YieldOp>(location, args[0]); });
      values.map(operation->getResult(0), output);
    } else if (auto op = dyn_cast<BroadcastOp>(operation)) {
      auto tensor = cast<RankedTensorType>(op.getResult().getType());
      auto sizes = extents(tensor, loc);
      if (failed(sizes)) return failure();
      Value input = values.lookup(op.getInputs()[0]);
      Value output = allocate(tensor, *sizes, loc);
      builder.create<linalg::GenericOp>(loc, ValueRange{input}, ValueRange{output},
          pointwiseMaps(ValueRange{input}, tensor.getRank()),
          SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
          [](OpBuilder &nested, Location loc, ValueRange scalars) {
            nested.create<linalg::YieldOp>(loc, scalars[0]);
          });
      values.map(op.getResult(), output);
    } else if (auto op = dyn_cast<MakeRecordOp>(operation)) {
      bindProduct(op.getResult(), flattened(op.getFields()));
    } else if (auto op = dyn_cast<MakeTupleOp>(operation)) {
      bindProduct(op.getResult(), flattened(op.getComponents()));
    } else if (auto op = dyn_cast<ExtractOp>(operation)) {
      auto components = componentTypes(op.getProduct().getType());
      unsigned offset = 0;
      SmallVector<Type> leaves;
      for (uint64_t field = 0; field < op.getField(); ++field)
        flattenTypes(cast<TypeAttr>(components[field]).getValue(), leaves);
      offset = leaves.size(); leaves.clear();
      flattenTypes(op.getResult().getType(), leaves);
      bindProduct(op.getResult(), ValueRange(products.at(op.getProduct())).slice(offset, leaves.size()));
    } else if (isa<RegionFoldOp, RegionScanOp>(operation)) {
      return region(operation);
    } else if (auto op = dyn_cast<ReduceOp>(operation)) {
      return reduce(op);
    } else if (auto op = dyn_cast<QuantizeOp>(operation)) {
      auto tensor = cast<RankedTensorType>(op.getResult().getType());
      auto sizes = extents(tensor, loc);
      if (failed(sizes)) return failure();
      Value output = allocate(tensor, *sizes, loc);
      output.getDefiningOp<memref::AllocOp>().setAlignment(4);
      if (tensor.getShape()[0] != 0)
        builder.create<cpu::QuantizeOp>(loc, values.lookup(op.getInput()), output, op.getFormatAttr());
      values.map(op.getResult(), output);
    } else if (auto op = dyn_cast<QuantizedDotOp>(operation)) {
      if (cast<RankedTensorType>(op.getLhs().getType()).getShape()[0] == 0) {
        values.map(op.getResult(), builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(0.0)));
        return success();
      }
      values.map(op.getResult(), builder.create<cpu::QuantizedDotOp>(loc,
          builder.getF32Type(), values.lookup(op.getLhs()), values.lookup(op.getRhs()),
          op.getLhsFormatAttr(), op.getRhsFormatAttr()));
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
  llvm::DenseMap<Value, SmallVector<Value>> products;
  llvm::DenseMap<int64_t, Value> dimensions;
  llvm::DenseMap<Value, Domain> domains;
  SmallVector<SmallVector<Value>> allocations;
  bool scalarized = false;
};

}

LogicalResult lowerCanonicalKIRToCPU(ModuleOp module) {
  CanonicalKernelAnalysis analysis(module);
  if (failed(analysis.verify())) return failure();
  auto physical = OwningOpRef<ModuleOp>(ModuleOp::create(module.getLoc()));
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
