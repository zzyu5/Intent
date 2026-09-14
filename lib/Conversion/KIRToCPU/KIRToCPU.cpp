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
  Value begin, end, step, extent;
};

Type integerStorageType(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return IntegerType::get(type.getContext(), integer.getWidth());
  if (auto memory = dyn_cast<MemRefType>(type))
    return MemRefType::get(memory.getShape(), integerStorageType(memory.getElementType()), memory.getLayout(), memory.getMemorySpace());
  if (auto function = dyn_cast<FunctionType>(type)) {
    SmallVector<Type> inputs, outputs;
    for (Type input : function.getInputs()) inputs.push_back(integerStorageType(input));
    for (Type output : function.getResults()) outputs.push_back(integerStorageType(output));
    return FunctionType::get(type.getContext(), inputs, outputs);
  }
  if (auto vector = dyn_cast<VectorType>(type))
    return VectorType::get(vector.getShape(), integerStorageType(vector.getElementType()), vector.getScalableDims());
  return type;
}

void realizeIntegerStorage(ModuleOp module) {
  module.walk([&](Operation *operation) {
    for (Value result : operation->getResults()) result.setType(integerStorageType(result.getType()));
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments()) argument.setType(integerStorageType(argument.getType()));
    SmallVector<NamedAttribute> attributes(operation->getAttrs());
    for (NamedAttribute attribute : attributes) {
      if (auto type = dyn_cast<TypeAttr>(attribute.getValue()))
        operation->setAttr(attribute.getName(), TypeAttr::get(integerStorageType(type.getValue())));
      else if (auto integer = dyn_cast<IntegerAttr>(attribute.getValue()))
        operation->setAttr(attribute.getName(), IntegerAttr::get(integerStorageType(integer.getType()), integer.getValue()));
    }
  });
}

class Construction {
public:
  Construction(ModuleOp original, ModuleOp physical, CPUEntryLayout entryLayout)
      : analysis(original), module(physical), builder(physical.getContext()), entryLayout(entryLayout) {}

  LogicalResult lower(func::FuncOp source) {
    auto parameters = source->getAttrOfType<ArrayAttr>("intent.parameters");
    if (!parameters || parameters.size() != source.getNumArguments())
      return source.emitError("CPU construction requires canonical parameter metadata");
    SmallVector<Type> types;
    SmallVector<Attribute> interface;
    SmallVector<Value> runtimeArguments;
    for (auto [i, argument] : llvm::enumerate(source.getArguments())) {
      auto parameter = cast<ParameterAttr>(parameters[i]);
      if (parameter.getKind() == 2) {
        if (!isa<ConstexprType>(argument.getType()) || !argument.use_empty())
          return source.emitError("CPU physical ABI requires fully specialized constexpr parameters");
        continue;
      }
      runtimeArguments.push_back(argument);
      if (auto view = dyn_cast<ViewType>(argument.getType())) {
        auto tensor = cast<RankedTensorType>(view.getTensor());
        Type element = tensor.getElementType();
        if ((!element.isF16() && !element.isBF16() && !element.isF32() && !element.isF64() &&
             !isa<Float8E4M3FNType, Float8E5M2Type>(element) &&
             !element.isInteger(8) && !element.isInteger(16) && !element.isInteger(32) &&
             !element.isInteger(64) && !element.isInteger(1)))
          return source.emitError("CPU construction requires supported numeric views");
        auto memory = MemRefType::get(tensor.getShape(), tensor.getElementType());
        SmallVector<Attribute> constraints(tensor.getRank(), builder.getUnitAttr());
        if (view.getConstraints().getHasStrides()) {
          if (view.getConstraints().getStrides().size() != static_cast<size_t>(tensor.getRank()))
            return source.emitError("CPU declared stride constraints must cover the view rank");
          constraints.assign(view.getConstraints().getStrides().begin(), view.getConstraints().getStrides().end());
        }
        for (Attribute constraint : constraints)
          if (!isa<UnitAttr, IntegerAttr>(constraint))
            return source.emitError("CPU symbolic stride constraints are not implemented");
        if (entryLayout == CPUEntryLayout::StridedInputs && view.getAccess() == 0) {
          SmallVector<int64_t> strides;
          for (Attribute constraint : constraints) {
            auto fixed = dyn_cast<IntegerAttr>(constraint);
            strides.push_back(fixed ? fixed.getInt() : ShapedType::kDynamic);
          }
          memory = MemRefType::get(tensor.getShape(), tensor.getElementType(),
              StridedLayoutAttr::get(builder.getContext(), 0, strides));
        } else {
          SmallVector<int64_t> strides;
          int64_t offset;
          if (failed(memory.getStridesAndOffset(strides, offset)))
            return source.emitError("CPU contiguous view has no derived strides");
          for (auto [constraint, stride] : llvm::zip(constraints, strides)) {
            if (isa<UnitAttr>(constraint)) continue;
            auto fixed = dyn_cast<IntegerAttr>(constraint);
            if (!fixed || (!ShapedType::isDynamic(stride) && fixed.getInt() != stride))
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
            builder.getArrayAttr(constraints), view.getAccess(), view.getConstraints().getAlias(),
            view.getConstraints().getNoalias()));
      } else if (argument.getType().isF32() || argument.getType().isF64() || argument.getType().isIndex() ||
                 argument.getType().isInteger(8) || argument.getType().isInteger(16) ||
                 argument.getType().isInteger(32) || argument.getType().isInteger(64) ||
                 argument.getType().isInteger(1)) {
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
        builder.getContext(), builder.getArrayAttr(interface), entryLayout == CPUEntryLayout::Contiguous, true));
    function.addEntryBlock();
    builder.setInsertionPointToStart(&function.front());
    for (auto [oldValue, newValue] : llvm::zip(runtimeArguments, function.getArguments())) {
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

  FailureOr<Value> indexValue(Value value, Type logicalType, Location loc) {
    if (value.getType().isIndex()) return value;
    if (isa<IntegerType>(value.getType()) && !value.getType().isInteger(1)) {
      auto integer = dyn_cast<IntegerType>(logicalType);
      if (!integer && !logicalType.isIndex() && !isa<LogicalIndexType>(logicalType))
        return emitError(loc, "CPU coordinate has no integer interpretation"), failure();
      return integer && integer.isUnsigned()
          ? Value(builder.create<arith::IndexCastUIOp>(loc, builder.getIndexType(), value))
          : Value(builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), value));
    }
    return emitError(loc, "CPU coordinates require index or integer values"), failure();
  }

  Value domainExtent(Value begin, Value end, Value step, Location loc) {
    if (matchPattern(begin, m_Zero()) && matchPattern(step, m_One()) && end.getDefiningOp<memref::DimOp>())
      return end;
    Value distance = builder.createOrFold<arith::SubIOp>(loc, end, begin);
    Value nonnegative = builder.createOrFold<arith::MaxSIOp>(loc, distance, constant(loc, 0));
    return builder.createOrFold<arith::CeilDivSIOp>(loc, nonnegative, step);
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

  Value elementAt(Value input, ValueRange members, OpBuilder &nested, Location loc) {
    auto type = dyn_cast<MemRefType>(input.getType());
    if (!type) return input;
    SmallVector<Value> indices;
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      indices.push_back(type.getDimSize(axis) == 1
          ? Value(nested.create<arith::ConstantIndexOp>(loc, 0))
          : members[members.size() - type.getRank() + axis]);
    return nested.create<memref::LoadOp>(loc, input, indices);
  }

  FailureOr<unsigned> advancedIndexRank(Operation *operation, const IndexRelationFact &fact) {
    unsigned advancedRank = 0;
    for (const auto &term : fact.terms) {
      if (term.kind == 3 && term.operands.size() == 1) {
        if (auto type = dyn_cast<MemRefType>(values.lookup(term.operands[0]).getType()))
          advancedRank = std::max<unsigned>(advancedRank, type.getRank());
      } else if (term.kind == 4) {
        if (term.operands.size() != 1 || !domains.count(term.operands[0]))
          return operation->emitError("CPU indexed access requires a realized domain"), failure();
      } else if (term.kind != 0 && term.kind != 1 && term.kind != 2) {
        return operation->emitError("CPU indexed access does not implement this coordinate term"), failure();
      }
    }
    return advancedRank;
  }

  SmallVector<Value> indexedCoordinates(const IndexRelationFact &fact, unsigned advancedRank,
                                       Value source, ValueRange members, OpBuilder &nested, Location loc) {
    SmallVector<Value> coordinates;
    unsigned axis = 0;
    std::optional<unsigned> advancedBegin;
    for (const auto &term : fact.terms) {
      if (term.kind == 1) { ++axis; continue; }
      Value coordinate;
      if (term.kind == 0 || term.kind == 4) {
        coordinate = members[axis++];
        if (term.kind == 4) {
          Domain domain = domains.lookup(term.operands[0]);
          coordinate = nested.createOrFold<arith::AddIOp>(loc, domain.begin,
              nested.createOrFold<arith::MulIOp>(loc, coordinate, domain.step));
        }
      } else if (term.kind == 2) {
        int64_t literal = *term.staticValues[0];
        coordinate = nested.create<arith::ConstantIndexOp>(loc, literal);
        if (literal < 0) coordinate = nested.create<arith::AddIOp>(loc,
            nested.create<memref::DimOp>(loc, source, *term.sourceAxis), coordinate);
      } else {
        coordinate = values.lookup(term.operands[0]);
        if (auto type = dyn_cast<MemRefType>(coordinate.getType())) {
          if (!advancedBegin) { advancedBegin = axis; axis += advancedRank; }
          coordinate = elementAt(coordinate, members.slice(*advancedBegin, advancedRank), nested, loc);
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

  LogicalResult indexedWrite(Operation *operation) {
    auto fact = analysis.indexRelation(operation);
    if (failed(fact)) return failure();
    auto rank = advancedIndexRank(operation, *fact);
    if (failed(rank)) return failure();
    unsigned position = operation->getAttrOfType<IntegerAttr>("value_operand_index").getInt();
    Value input = values.lookup(operation->getOperand(position)), destination = values.lookup(fact->source);
    Location loc = operation->getLoc();
    SmallVector<Value> sizes, members;
    if (auto type = dyn_cast<MemRefType>(input.getType()))
      for (int64_t axis = 0; axis < type.getRank(); ++axis)
        sizes.push_back(builder.create<memref::DimOp>(loc, input, axis));
    std::function<void(unsigned)> traverse = [&](unsigned axis) {
      if (axis == sizes.size()) {
        auto coordinates = indexedCoordinates(*fact, *rank, destination, members, builder, loc);
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

  FailureOr<Value> indexedRead(Operation *operation, const IndexRelationFact &fact, Value source) {
    Type resultType = operation->getResult(0).getType();
    auto tensor = dyn_cast<RankedTensorType>(resultType);
    auto rank = advancedIndexRank(operation, fact);
    if (failed(rank)) return failure();
    auto read = [&](OpBuilder &nested, Location loc, ValueRange members) -> Value {
      auto load = [&]() -> Value {
        auto coordinates = indexedCoordinates(fact, *rank, source, members, nested, loc);
        return nested.create<memref::LoadOp>(loc, source, coordinates);
      };
      if (alwaysValid(operation)) return load();
      unsigned validPosition = operation->getAttrOfType<IntegerAttr>("valid_operand_index").getInt();
      unsigned fillPosition = operation->getAttrOfType<IntegerAttr>("fill_operand_index").getInt();
      Value active = elementAt(values.lookup(operation->getOperand(validPosition)), members, nested, loc);
      auto conditional = nested.create<scf::IfOp>(loc, TypeRange{getElementTypeOrSelf(resultType)}, active, true);
      {
        OpBuilder::InsertionGuard guard(nested);
        nested.setInsertionPointToStart(conditional.thenBlock());
        nested.create<scf::YieldOp>(loc, load());
        nested.setInsertionPointToStart(conditional.elseBlock());
        Value fill = elementAt(values.lookup(operation->getOperand(fillPosition)), members, nested, loc);
        nested.create<scf::YieldOp>(loc, fill);
      }
      return conditional.getResult(0);
    };
    if (!tensor) return read(builder, operation->getLoc(), {});
    auto shape = extents(tensor, operation->getLoc());
    if (failed(shape)) return failure();
    Value output = allocate(tensor, *shape, operation->getLoc());
    builder.create<linalg::GenericOp>(operation->getLoc(), ValueRange{}, ValueRange{output},
        SmallVector<AffineMap>{builder.getMultiDimIdentityMap(tensor.getRank())},
        SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &nested, Location loc, ValueRange) {
          SmallVector<Value> members;
          for (int64_t axis = 0; axis < tensor.getRank(); ++axis)
            members.push_back(nested.create<linalg::IndexOp>(loc, axis));
          nested.create<linalg::YieldOp>(loc, read(nested, loc, members));
        });
    return output;
  }

  LogicalResult atomicAccess(Operation *operation) {
    auto fact = analysis.indexRelation(operation);
    if (failed(fact)) return failure();
    auto rank = advancedIndexRank(operation, *fact);
    if (failed(rank)) return failure();
    Location loc = operation->getLoc();
    Value target = values.lookup(fact->source);
    Type element = cast<MemRefType>(target.getType()).getElementType();
    auto ordering = operation->getAttrOfType<AtomicOrderingAttr>("ordering");
    SmallVector<Type> resultTypes;
    for (Type type : operation->getResultTypes()) flattenTypes(type, resultTypes);
    Type accessType = resultTypes.empty()
        ? operation->getOperand(cast<AtomicStoreOp>(operation).getValueOperand()).getType()
        : resultTypes.front();
    auto tensor = dyn_cast<RankedTensorType>(accessType);
    SmallVector<Value> sizes, members, outputs;
    bool used = llvm::any_of(operation->getResults(), [](Value value) { return !value.use_empty(); });
    if (tensor) {
      auto extentsOr = extents(tensor, loc);
      if (failed(extentsOr)) return failure();
      sizes = *extentsOr;
      if (used) outputs = makeSlots(operation->getResultTypes(), loc);
    }
    auto operand = [&](StringRef attribute) {
      auto position = operation->getAttrOfType<IntegerAttr>(attribute).getInt();
      return elementAt(values.lookup(operation->getOperand(position)), members, builder, loc);
    };
    std::function<void(unsigned)> traverse = [&](unsigned axis) {
      if (axis < sizes.size()) {
        auto loop = builder.create<scf::ForOp>(loc, constant(loc, 0), sizes[axis], constant(loc, 1));
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(loop.getBody());
        members.push_back(loop.getInductionVar());
        traverse(axis + 1);
        members.pop_back();
        return;
      }
      auto coordinates = indexedCoordinates(*fact, *rank, target, members, builder, loc);
      SmallVector<Value> results;
      if (isa<AtomicLoadOp>(operation)) {
        results.push_back(builder.create<cpu::AtomicLoadOp>(loc, element, target, coordinates, ordering));
      } else if (isa<AtomicStoreOp>(operation)) {
        builder.create<cpu::AtomicStoreOp>(loc, target, operand("value_operand"), coordinates, ordering);
      } else if (isa<AtomicRMWOp>(operation)) {
        results.push_back(builder.create<cpu::AtomicRMWOp>(loc, element, target, operand("value_operand"),
            coordinates, ordering, operation->getAttrOfType<AtomicRMWKindAttr>("kind"),
            builder.getBoolAttr(isa<IntegerType>(element) && cast<IntegerType>(element).isUnsigned())));
      } else {
        auto exchange = builder.create<cpu::AtomicCompareExchangeOp>(loc, element, builder.getI1Type(),
            target, operand("expected_operand"), operand("desired_operand"), coordinates, ordering);
        llvm::append_range(results, exchange.getResults());
      }
      if (!used) return;
      if (!tensor) bindScalars(operation->getResults(), results);
      else for (auto [value, output] : llvm::zip(results, outputs))
        builder.create<memref::StoreOp>(loc, value, output, members);
    };
    traverse(0);
    if (used && tensor) bindSlots(operation->getResults(), outputs, loc);
    return success();
  }

  LogicalResult scatterReduce(ScatterReduceOp operation) {
    auto fact = analysis.indexRelation(operation);
    if (failed(fact)) return failure();
    auto rank = advancedIndexRank(operation, *fact);
    if (failed(rank)) return failure();
    Location loc = operation.getLoc();
    Value target = values.lookup(fact->source);
    Value input = values.lookup(operation->getOperand(operation.getValueOperandIndex()));
    SmallVector<Value> sizes, members;
    if (auto memory = dyn_cast<MemRefType>(input.getType()))
      for (int64_t axis = 0; axis < memory.getRank(); ++axis)
        sizes.push_back(builder.create<memref::DimOp>(loc, input, axis));
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
      auto coordinates = indexedCoordinates(*fact, *rank, target, members, builder, loc);
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

  FailureOr<Value> indexed(Operation *operation) {
    auto fact = analysis.indexRelation(operation);
    if (failed(fact))
      return failure();
    Value source = values.lookup(fact->source);
    if (isa<ViewLoadOp, BufferLoadOp, GatherOp>(operation) && (!alwaysValid(operation) || llvm::any_of(fact->terms, [&](const auto &term) {
          return term.kind == 3 && term.operands.size() == 1 &&
              isa<MemRefType>(values.lookup(term.operands[0]).getType());
        })))
      return indexedRead(operation, *fact, source);
    auto type = dyn_cast<MemRefType>(source.getType());
    SmallVector<OpFoldResult> offsets, sizes, strides;
    SmallVector<int64_t> resultShape;
    SmallVector<AffineExpr> projection;
    unsigned outputAxis = 0;
    bool inserted = false;
    for (const IndexTermFact &term : fact->terms) {
      OpFoldResult stride = builder.getIndexAttr(1);
      if (term.kind == 1) {
        inserted = true;
        ++outputAxis;
        continue;
      } else if (term.kind == 0) {
        unsigned axis = *term.sourceAxis;
        offsets.push_back(builder.getIndexAttr(0));
        sizes.push_back(type.isDynamicDim(axis) ? OpFoldResult(builder.create<memref::DimOp>(operation->getLoc(), source, axis).getResult()) : builder.getIndexAttr(type.getDimSize(axis)));
        resultShape.push_back(type.getDimSize(axis));
        projection.push_back(builder.getAffineDimExpr(outputAxis++));
      } else if (term.kind == 3 && term.operands.size() == 1) {
        auto coordinate = indexValue(values.lookup(term.operands[0]), term.operands[0].getType(), operation->getLoc());
        if (failed(coordinate)) return failure();
        offsets.push_back(*coordinate);
        sizes.push_back(builder.getIndexAttr(1));
      } else if (term.kind == 2 && !term.staticValues.empty() && term.staticValues[0]) {
        int64_t literal = *term.staticValues[0];
        if (literal < 0) offsets.push_back(builder.create<arith::AddIOp>(operation->getLoc(),
            builder.create<memref::DimOp>(operation->getLoc(), source, *term.sourceAxis),
            constant(operation->getLoc(), literal)).getResult());
        else offsets.push_back(builder.getIndexAttr(literal));
        sizes.push_back(builder.getIndexAttr(1));
      } else if (term.kind == 4 && term.operands.size() == 1 && domains.count(term.operands[0])) {
        Domain domain = domains.lookup(term.operands[0]);
        offsets.push_back(domain.begin);
        stride = domain.step;
        projection.push_back(builder.getAffineDimExpr(outputAxis++));
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
      if (isa<ViewLoadOp, BufferLoadOp, GatherOp>(operation))
        return Value(builder.create<memref::LoadOp>(operation->getLoc(), source, indices));
    }
    auto resultType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        resultShape, type, offsets, sizes, strides));
    Value selected = builder.create<memref::SubViewOp>(operation->getLoc(), resultType,
                                                      source, offsets, sizes, strides);
    if (!inserted) {
      auto view = dyn_cast<ViewType>(fact->source.getType());
      if (isa<BufferLoadOp>(operation) || (isa<ViewLoadOp>(operation) && view && view.getAccess() != 0)) {
        auto tensor = cast<RankedTensorType>(operation->getResult(0).getType());
        SmallVector<Value> shape;
        for (int64_t axis = 0; axis < resultType.getRank(); ++axis)
          shape.push_back(builder.create<memref::DimOp>(operation->getLoc(), selected, axis));
        Value snapshot = allocate(tensor, shape, operation->getLoc());
        builder.create<memref::CopyOp>(operation->getLoc(), selected, snapshot);
        return snapshot;
      }
      return selected;
    }
    if (operation->getNumResults() != 1 || !isa<RankedTensorType>(operation->getResult(0).getType()))
      return operation->emitError("CPU inserted write axes are not implemented"), failure();
    auto tensor = cast<RankedTensorType>(operation->getResult(0).getType());
    auto shape = extents(tensor, operation->getLoc());
    if (failed(shape)) return failure();
    Value output = allocate(tensor, *shape, operation->getLoc());
    builder.create<linalg::GenericOp>(operation->getLoc(), ValueRange{selected}, ValueRange{output},
        SmallVector<AffineMap>{AffineMap::get(tensor.getRank(), 0, projection, builder.getContext()), builder.getMultiDimIdentityMap(tensor.getRank())},
        SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
        [](OpBuilder &nested, Location loc, ValueRange scalars) { nested.create<linalg::YieldOp>(loc, scalars[0]); });
    return output;
  }

  FailureOr<Value> arithmetic(Operation *operation, ValueRange arguments, OpBuilder &builder) {
    Location loc = operation->getLoc();
    auto flushF32 = [&](Value value) -> Value {
      Value zero = builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(0.0f));
      Value normal = builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(0x1.0p-126f));
      Value magnitude = builder.create<math::AbsFOp>(loc, value);
      Value subnormal = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, magnitude, normal);
      Value signedZero = builder.create<arith::MulFOp>(loc, value, zero);
      return builder.create<arith::SelectOp>(loc, subnormal, signedZero, value);
    };
    if (isa<RandomBitsOp>(operation)) {
      Type u32 = IntegerType::get(builder.getContext(), 32, IntegerType::Unsigned);
      Type u64 = IntegerType::get(builder.getContext(), 64, IntegerType::Unsigned);
      auto literal = [&](Type type, uint64_t value) -> Value {
        return builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(type, value));
      };
      auto low = [&](Value value) -> Value { return builder.create<arith::TruncIOp>(loc, u32, value); };
      auto high = [&](Value value) -> Value {
        return low(builder.create<arith::ShRUIOp>(loc, value, literal(u64, 32)));
      };
      Value counter = arguments[1];
      if (counter.getType().isIndex()) counter = builder.create<arith::IndexCastUIOp>(loc, u64, counter);
      Value block = builder.create<arith::ShRUIOp>(loc, counter, literal(u64, 2));
      Value word = builder.create<arith::AndIOp>(loc, counter, literal(u64, 3));
      Value key0 = low(arguments[0]), key1 = high(arguments[0]);
      SmallVector<Value> words{low(block), high(block), literal(u32, 0), literal(u32, 0)};
      auto product = [&](Value value, uint64_t multiplier) {
        Value wide = builder.create<arith::ExtUIOp>(loc, u64, value);
        Value result = builder.create<arith::MulIOp>(loc, wide, literal(u64, multiplier));
        return std::make_pair(high(result), low(result));
      };
      auto mix = [&](Value a, Value b, Value key) -> Value {
        return builder.create<arith::XOrIOp>(loc, builder.create<arith::XOrIOp>(loc, a, b), key);
      };
      for (unsigned round = 0; round < 10; ++round) {
        auto [hi0, lo0] = product(words[0], 0xD2511F53);
        auto [hi1, lo1] = product(words[2], 0xCD9E8D57);
        words = {mix(hi1, words[1], key0), lo1, mix(hi0, words[3], key1), lo0};
        key0 = builder.create<arith::AddIOp>(loc, key0, literal(u32, 0x9E3779B9));
        key1 = builder.create<arith::AddIOp>(loc, key1, literal(u32, 0xBB67AE85));
      }
      Value result = words[3];
      for (int i = 2; i >= 0; --i) {
        Value selected = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, word, literal(u64, i));
        result = builder.create<arith::SelectOp>(loc, selected, words[i], result);
      }
      return result;
    } else if (auto binary = dyn_cast<BinaryOp>(operation)) {
      Value a = arguments[0], b = arguments[1];
      if (binary.getApproximate() || binary.getFlushToZero()) {
        if (binary.getOperatorKind() != BinaryOperator::TrueDivide || !binary.getApproximate() ||
            !a.getType().isF32() || !b.getType().isF32())
          return binary.emitError("CPU non-default arithmetic requires the closed f32 approximate division contract"), failure();
        auto literal = [&](float value) -> Value {
          return builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(value));
        };
        Value zero = literal(0.0f), negativeZero = literal(-0.0f);
        if (binary.getFlushToZero()) { a = flushF32(a); b = flushF32(b); }
        Value quotient = builder.create<arith::DivFOp>(loc, a, b);
        Value large = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
            builder.create<math::AbsFOp>(loc, b), literal(0x1.0p126f));
        Value negative = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, b, zero);
        Value denominatorSign = builder.create<arith::SelectOp>(loc, negative, negativeZero, zero);
        Value limit = builder.create<arith::MulFOp>(loc, a, denominatorSign);
        Value result = builder.create<arith::SelectOp>(loc, large, limit, quotient);
        return binary.getFlushToZero() ? flushF32(result) : result;
      }
      bool fp = isa<FloatType>(a.getType());
      auto logical = dyn_cast<IntegerType>(getElementTypeOrSelf(binary.getOperand(0).getType()));
      bool unsignedInteger = logical && logical.isUnsigned();
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
      case BinaryOperator::FloorDivide:
        if (!fp) return unsignedInteger ? Value(builder.create<arith::DivUIOp>(loc, a, b))
            : Value(builder.create<arith::FloorDivSIOp>(loc, a, b));
        break;
      case BinaryOperator::Remainder:
        if (!fp) {
          if (unsignedInteger) return Value(builder.create<arith::RemUIOp>(loc, a, b));
          Value quotient = builder.create<arith::FloorDivSIOp>(loc, a, b);
          Value product = builder.create<arith::MulIOp>(loc, quotient, b);
          return Value(builder.create<arith::SubIOp>(loc, a, product));
        }
        break;
      case BinaryOperator::MaximumNum:
        if (fp) return Value(builder.create<arith::MaxNumFOp>(loc, a, b));
        break;
      case BinaryOperator::MinimumNum:
        if (fp) return Value(builder.create<arith::MinNumFOp>(loc, a, b));
        break;
      case BinaryOperator::Power:
        if (fp) return Value(builder.create<math::PowFOp>(loc, a, b));
        break;
      case BinaryOperator::Maximum:
        return fp ? Value(builder.create<arith::MaximumFOp>(loc, a, b)) : unsignedInteger
            ? Value(builder.create<arith::MaxUIOp>(loc, a, b)) : Value(builder.create<arith::MaxSIOp>(loc, a, b));
      case BinaryOperator::Minimum:
        return fp ? Value(builder.create<arith::MinimumFOp>(loc, a, b)) : unsignedInteger
            ? Value(builder.create<arith::MinUIOp>(loc, a, b)) : Value(builder.create<arith::MinSIOp>(loc, a, b));
      case BinaryOperator::LogicalAnd:
      case BinaryOperator::BitwiseAnd: return Value(builder.create<arith::AndIOp>(loc, a, b));
      case BinaryOperator::LogicalOr:
      case BinaryOperator::BitwiseOr: return Value(builder.create<arith::OrIOp>(loc, a, b));
      case BinaryOperator::BitwiseXor: return Value(builder.create<arith::XOrIOp>(loc, a, b));
      case BinaryOperator::LeftShift: return Value(builder.create<arith::ShLIOp>(loc, a, b));
      case BinaryOperator::RightShift: return unsignedInteger ? Value(builder.create<arith::ShRUIOp>(loc, a, b))
          : Value(builder.create<arith::ShRSIOp>(loc, a, b));
      default: break;
      }
    } else if (auto unary = dyn_cast<UnaryOp>(operation)) {
      if (unary.getApproximate() || unary.getFlushToZero()) {
        auto kind = unary.getOperatorKind();
        if (!unary.getApproximate() || !arguments[0].getType().isF32() ||
            (kind != UnaryOperator::Exp2 && kind != UnaryOperator::Tanh) ||
            (unary.getFlushToZero() && kind != UnaryOperator::Exp2))
          return unary.emitError("CPU non-default unary arithmetic requires the closed f32 exp2/tanh contract"), failure();
        Value input = unary.getFlushToZero() ? flushF32(arguments[0]) : arguments[0];
        Value result = kind == UnaryOperator::Exp2 ? Value(builder.create<math::Exp2Op>(loc, input))
                                                  : Value(builder.create<math::TanhOp>(loc, input));
        return unary.getFlushToZero() ? flushF32(result) : result;
      }
      switch (unary.getOperatorKind()) {
      case UnaryOperator::Rsqrt: return Value(builder.create<math::RsqrtOp>(loc, arguments[0]));
      case UnaryOperator::Sqrt: return Value(builder.create<math::SqrtOp>(loc, arguments[0]));
      case UnaryOperator::Exp: return Value(builder.create<math::ExpOp>(loc, arguments[0]));
      case UnaryOperator::Exp2: return Value(builder.create<math::Exp2Op>(loc, arguments[0]));
      case UnaryOperator::Log: return Value(builder.create<math::LogOp>(loc, arguments[0]));
      case UnaryOperator::Sin: return Value(builder.create<math::SinOp>(loc, arguments[0]));
      case UnaryOperator::Cos: return Value(builder.create<math::CosOp>(loc, arguments[0]));
      case UnaryOperator::Floor: return Value(builder.create<math::FloorOp>(loc, arguments[0]));
      case UnaryOperator::Erf: return Value(builder.create<math::ErfOp>(loc, arguments[0]));
      case UnaryOperator::Tanh: return Value(builder.create<math::TanhOp>(loc, arguments[0]));
      case UnaryOperator::Abs:
        if (isa<FloatType>(arguments[0].getType())) return Value(builder.create<math::AbsFOp>(loc, arguments[0]));
        if (auto type = dyn_cast<IntegerType>(getElementTypeOrSelf(unary.getOperand().getType())); type && type.isUnsigned()) return arguments[0];
        return Value(builder.create<math::AbsIOp>(loc, arguments[0]));
      case UnaryOperator::Sigmoid: {
        Value one = builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(arguments[0].getType(), 1.0));
        Value negative = builder.create<arith::NegFOp>(loc, arguments[0]);
        Value denominator = builder.create<arith::AddFOp>(loc, one, builder.create<math::ExpOp>(loc, negative));
        return Value(builder.create<arith::DivFOp>(loc, one, denominator));
      }
      case UnaryOperator::Negate:
        if (isa<FloatType>(arguments[0].getType())) return Value(builder.create<arith::NegFOp>(loc, arguments[0]));
        return Value(builder.create<arith::SubIOp>(loc,
            builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(arguments[0].getType(), 0)), arguments[0]));
      case UnaryOperator::Not: return Value(builder.create<arith::XOrIOp>(loc, arguments[0],
          builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(true))));
      default: break;
      }
    } else if (auto compare = dyn_cast<CompareOp>(operation)) {
      static const arith::CmpFPredicate floating[] = {arith::CmpFPredicate::OEQ, arith::CmpFPredicate::UNE,
          arith::CmpFPredicate::OLT, arith::CmpFPredicate::OLE, arith::CmpFPredicate::OGT, arith::CmpFPredicate::OGE};
      static const arith::CmpIPredicate integer[] = {arith::CmpIPredicate::eq, arith::CmpIPredicate::ne,
          arith::CmpIPredicate::slt, arith::CmpIPredicate::sle, arith::CmpIPredicate::sgt, arith::CmpIPredicate::sge};
      static const arith::CmpIPredicate unsignedPredicates[] = {arith::CmpIPredicate::eq, arith::CmpIPredicate::ne,
          arith::CmpIPredicate::ult, arith::CmpIPredicate::ule, arith::CmpIPredicate::ugt, arith::CmpIPredicate::uge};
      auto type = dyn_cast<IntegerType>(getElementTypeOrSelf(compare.getOperand(0).getType()));
      unsigned predicate = static_cast<unsigned>(compare.getPredicate());
      return isa<FloatType>(arguments[0].getType())
          ? Value(builder.create<arith::CmpFOp>(loc, floating[predicate], arguments[0], arguments[1]))
          : Value(builder.create<arith::CmpIOp>(loc, type && type.isUnsigned() ? unsignedPredicates[predicate] : integer[predicate], arguments[0], arguments[1]));
    } else if (isa<SelectOp>(operation)) {
      return Value(builder.create<arith::SelectOp>(loc, arguments[0], arguments[1], arguments[2]));
    } else if (isa<MaskOp>(operation)) {
      return Value(builder.create<arith::SelectOp>(loc, arguments[1], arguments[0], arguments[2]));
    } else if (isa<BitcastOp>(operation)) {
      return Value(builder.create<arith::BitcastOp>(loc, getElementTypeOrSelf(operation->getResult(0).getType()), arguments[0]));
    } else if (auto cast = dyn_cast<CastOp>(operation)) {
      Type type = getElementTypeOrSelf(cast.getType());
      Type input = getElementTypeOrSelf(cast.getOperand().getType());
      if (isa<LogicalIndexType>(input)) input = builder.getIndexType();
      auto inputInteger = dyn_cast<IntegerType>(input), outputInteger = dyn_cast<IntegerType>(type);
      bool unsignedInput = inputInteger && inputInteger.isUnsigned();
      bool unsignedOutput = outputInteger && outputInteger.isUnsigned();
      if (input == type) return arguments[0];
      if (cast.getRounding() && *cast.getRounding() != 0)
        return cast.emitError("CPU non-default-rounding cast is not implemented"), failure();
      if (type.isInteger(1)) return cast.emitError("CPU numeric-to-bool cast is not implemented"), failure();
      if (input.isIndex() && outputInteger) return unsignedOutput
          ? Value(builder.create<arith::IndexCastUIOp>(loc, type, arguments[0])) : Value(builder.create<arith::IndexCastOp>(loc, type, arguments[0]));
      if (inputInteger && type.isIndex()) return unsignedInput
          ? Value(builder.create<arith::IndexCastUIOp>(loc, type, arguments[0])) : Value(builder.create<arith::IndexCastOp>(loc, type, arguments[0]));
      if ((input.isIndex() || inputInteger) && isa<FloatType>(type)) {
        Value value = input.isIndex() ? Value(builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), arguments[0])) : arguments[0];
        if (input.isInteger(1) || unsignedInput) return Value(builder.create<arith::UIToFPOp>(loc, type, value));
        return Value(builder.create<arith::SIToFPOp>(loc, type, value));
      }
      if (isa<FloatType>(input) && isa<FloatType>(type)) {
        if (input.getIntOrFloatBitWidth() < type.getIntOrFloatBitWidth())
          return Value(builder.create<arith::ExtFOp>(loc, type, arguments[0]));
        if (input.getIntOrFloatBitWidth() > type.getIntOrFloatBitWidth())
          return Value(builder.create<arith::TruncFOp>(loc, type, arguments[0]));
        Value widened = builder.create<arith::ExtFOp>(loc, builder.getF32Type(), arguments[0]);
        return Value(builder.create<arith::TruncFOp>(loc, type, widened));
      }
      if (isa<FloatType>(input) && outputInteger && !type.isInteger(1))
        return unsignedOutput ? Value(builder.create<arith::FPToUIOp>(loc, type, arguments[0]))
            : Value(builder.create<arith::FPToSIOp>(loc, type, arguments[0]));
      if (inputInteger && outputInteger) {
        if (input.getIntOrFloatBitWidth() == type.getIntOrFloatBitWidth()) return arguments[0];
        if (input.getIntOrFloatBitWidth() > type.getIntOrFloatBitWidth())
          return Value(builder.create<arith::TruncIOp>(loc, type, arguments[0]));
        if (input.isInteger(1) || unsignedInput) return Value(builder.create<arith::ExtUIOp>(loc, type, arguments[0]));
        return Value(builder.create<arith::ExtSIOp>(loc, type, arguments[0]));
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

  LogicalResult scan(ScanOp operation) {
    int64_t count = operation.getSourceCount();
    auto inputs = operation.getInputs();
    auto first = cast<RankedTensorType>(inputs[0].getType());
    SmallVector<Value> sources, initials, captures, outputs;
    for (int64_t component = 0; component < count; ++component) {
      auto type = cast<RankedTensorType>(inputs[component].getType());
      if (type.getShape() != first.getShape() || type.getEncoding() != first.getEncoding() ||
          inputs[count + component].getType() != type.getElementType())
        return operation.emitError("CPU scalar scan requires matching logical source axes and scalar identities; slice-valued combines are not implemented");
      sources.push_back(values.lookup(inputs[component]));
      initials.push_back(values.lookup(inputs[count + component]));
      auto sizes = extents(type, operation.getLoc());
      if (failed(sizes)) return failure();
      outputs.push_back(allocate(type, *sizes, operation.getLoc()));
    }
    for (Value input : inputs.drop_front(count * 2)) {
      Value capture = values.lookup(input);
      if (!isa<IntegerType, IndexType, FloatType>(capture.getType()))
        return operation.emitError("CPU scalar scan requires scalar captures");
      captures.push_back(capture);
    }
    auto target = builder.create<cpu::ScanOp>(operation.getLoc(), sources, initials, captures, outputs,
        operation.getAxis(), operation.getInclusive(), operation.getReverse());
    Block &source = operation.getCombine().front();
    Block *body = &target.getCombine().emplaceBlock();
    for (BlockArgument argument : source.getArguments()) {
      auto mapped = body->addArgument(argument.getType(), operation.getLoc());
      values.map(argument, mapped);
    }
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(body);
      if (failed(lowerBlock(source))) return failure();
      SmallVector<Value> yielded;
      for (Value value : source.getTerminator()->getOperands()) yielded.push_back(values.lookup(value));
      builder.create<cpu::ScanYieldOp>(operation.getLoc(), yielded);
    }
    for (auto [result, output] : llvm::zip(operation.getResults(), outputs)) values.map(result, output);
    return success();
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
      if (combineOp && isa<arith::AddFOp, arith::MaxNumFOp, arith::MaximumFOp>(combineOp) &&
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

  Value dotProduct(Value lhs, Value rhs, Value extent, Type accumulator,
                   ArrayRef<AffineMap> maps, Location loc) {
    Type element = cast<MemRefType>(lhs.getType()).getElementType();
    Value initial = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(accumulator));
    auto reduction = builder.create<cpu::ReduceOp>(loc, accumulator, extent, initial,
        ValueRange{lhs, rhs}, builder.getAffineMapArrayAttr(maps),
        cpu::ReductionOrderAttr::get(builder.getContext(), true));
    Block *body = &reduction.getCombine().emplaceBlock();
    body->addArguments(TypeRange{accumulator, element, element}, {loc, loc, loc});
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(body);
      Value left = body->getArgument(1), right = body->getArgument(2);
      if (element != accumulator) {
        left = builder.create<arith::ExtFOp>(loc, accumulator, left);
        right = builder.create<arith::ExtFOp>(loc, accumulator, right);
      }
      Value product = builder.create<arith::MulFOp>(loc, left, right);
      Value sum = builder.create<arith::AddFOp>(loc, body->getArgument(0), product);
      builder.create<cpu::YieldOp>(loc, sum);
    }
    return reduction.getResult();
  }

  void matrix(Value lhs, Value rhs, Value destination, Location loc) {
    Type accumulator = cast<MemRefType>(destination.getType()).getElementType();
    if (isa<FloatType>(accumulator) && cast<MemRefType>(rhs.getType()).getDimSize(1) == 1) {
      Value rows = builder.create<memref::DimOp>(loc, lhs, 0);
      Value extent = builder.create<memref::DimOp>(loc, lhs, 1);
      Value zero = constant(loc, 0), one = constant(loc, 1);
      auto vectorSlice = [&](Value source, int64_t reductionAxis,
                             ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) -> Value {
        auto type = cast<MemRefType>(source.getType());
        SmallVector<OpFoldResult> strides(2, builder.getIndexAttr(1));
        SmallVector<OpFoldResult> sliceSizes(sizes);
        if (!type.isDynamicDim(reductionAxis))
          sliceSizes[reductionAxis] = builder.getIndexAttr(type.getDimSize(reductionAxis));
        auto sliced = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
            ArrayRef<int64_t>{type.getDimSize(reductionAxis)}, type, offsets, sliceSizes, strides));
        return builder.create<memref::SubViewOp>(loc, sliced, source, offsets, sliceSizes, strides);
      };
      Value right = vectorSlice(rhs, 0, {builder.getIndexAttr(0), builder.getIndexAttr(0)},
          {extent, builder.getIndexAttr(1)});
      auto parallel = builder.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{rows}, ValueRange{one});
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(parallel.getBody());
      Value row = parallel.getInductionVars()[0];
      Value source = vectorSlice(lhs, 1, {row, builder.getIndexAttr(0)}, {builder.getIndexAttr(1), extent});
      auto map = builder.getMultiDimIdentityMap(1);
      Value value = dotProduct(source, right, extent, accumulator, {map, map}, loc);
      builder.create<memref::StoreOp>(loc, value, destination, ValueRange{row, zero});
      return;
    }
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(accumulator));
    AffineExpr m, n, k;
    bindDims(builder.getContext(), m, n, k);
    SmallVector<AffineMap> maps = {
        AffineMap::get(3, 0, {m, k}, builder.getContext()),
        AffineMap::get(3, 0, {k, n}, builder.getContext()),
        AffineMap::get(3, 0, {m, n}, builder.getContext())};
    builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{destination});
    builder.create<linalg::GenericOp>(loc, ValueRange{lhs, rhs}, ValueRange{destination}, maps,
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::parallel,
                                        utils::IteratorType::reduction},
        [](OpBuilder &b, Location loc, ValueRange arguments) {
          Value value;
          if (isa<FloatType>(arguments[2].getType())) {
            Value lhs = arguments[0], rhs = arguments[1];
            if (lhs.getType() != arguments[2].getType()) lhs = b.create<arith::ExtFOp>(loc, arguments[2].getType(), lhs);
            if (rhs.getType() != arguments[2].getType()) rhs = b.create<arith::ExtFOp>(loc, arguments[2].getType(), rhs);
            value = b.create<math::FmaOp>(loc, lhs, rhs, arguments[2]);
          } else {
            Value lhs = b.create<arith::ExtSIOp>(loc, arguments[2].getType(), arguments[0]);
            Value rhs = b.create<arith::ExtSIOp>(loc, arguments[2].getType(), arguments[1]);
            value = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, lhs, rhs), arguments[2]);
          }
          b.create<linalg::YieldOp>(loc, value);
        });
  }

  LogicalResult contract(ContractOp operation) {
    auto lhsType = cast<RankedTensorType>(operation.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(operation.getRhs().getType());
    auto resultType = cast<RankedTensorType>(operation.getResult().getType());
    auto pairs = operation.getReduce();
    Type inputElement = lhsType.getElementType(), accumulator = resultType.getElementType();
    bool floating = isa<FloatType>(inputElement) && inputElement == rhsType.getElementType() &&
        (accumulator.isF32() || accumulator.isF64()) &&
        inputElement.getIntOrFloatBitWidth() <= accumulator.getIntOrFloatBitWidth();
    bool integer = lhsType.getElementType().isSignlessInteger(8) && rhsType.getElementType().isSignlessInteger(8) &&
        resultType.getElementType().isSignlessInteger(32);
    int64_t batchRank = operation.getBatch().size();
    if (floating && !batchRank && lhsType.getRank() == 1 && rhsType.getRank() == 1 &&
        resultType.getRank() == 0 && pairs.size() == 1 &&
        cast<IntegerAttr>(cast<ArrayAttr>(pairs[0])[0]).getInt() == 0 &&
        cast<IntegerAttr>(cast<ArrayAttr>(pairs[0])[1]).getInt() == 0) {
      Location loc = operation.getLoc();
      Value lhs = values.lookup(operation.getLhs()), rhs = values.lookup(operation.getRhs());
      Value extent = builder.create<memref::DimOp>(loc, lhs, 0);
      auto map = builder.getMultiDimIdentityMap(1);
      values.map(operation.getResult(), dotProduct(lhs, rhs, extent, accumulator, {map, map}, loc));
      return success();
    }
    if (!batchRank && pairs.size() > 1 && resultType.getRank() == 2 &&
        lhsType.getRank() == static_cast<int64_t>(pairs.size()) + 1 &&
        rhsType.getRank() == static_cast<int64_t>(pairs.size()) + 1 && (floating || integer)) {
      Location loc = operation.getLoc();
      Value lhs = values.lookup(operation.getLhs()), rhs = values.lookup(operation.getRhs());
      SmallVector<unsigned> leftAxes, rightAxes;
      SmallVector<Value> reductionSizes;
      Value depth = constant(loc, 1);
      for (Attribute pair : pairs) {
        auto axes = cast<ArrayAttr>(pair);
        leftAxes.push_back(cast<IntegerAttr>(axes[0]).getInt());
        rightAxes.push_back(cast<IntegerAttr>(axes[1]).getInt());
        Value size = builder.create<memref::DimOp>(loc, lhs, leftAxes.back());
        reductionSizes.push_back(size);
        depth = builder.create<arith::MulIOp>(loc, depth, size);
      }
      auto pack = [&](Value source, ArrayRef<unsigned> axes, bool left) {
        auto type = cast<MemRefType>(source.getType());
        unsigned freeAxis = 0;
        while (llvm::is_contained(axes, freeAxis)) ++freeAxis;
        Value freeSize = builder.create<memref::DimOp>(loc, source, freeAxis);
        SmallVector<int64_t> shape = left
            ? SmallVector<int64_t>{type.getDimSize(freeAxis), ShapedType::kDynamic}
            : SmallVector<int64_t>{ShapedType::kDynamic, type.getDimSize(freeAxis)};
        SmallVector<Value> sizes = left ? SmallVector<Value>{freeSize, depth}
                                       : SmallVector<Value>{depth, freeSize};
        Value packed = allocate(RankedTensorType::get(shape, inputElement), sizes, loc);
        // Sliced paired axes need not be physically contiguous. Materialize
        // their logical coordinates before exposing the rank-two contraction.
        builder.create<linalg::GenericOp>(loc, ValueRange{}, ValueRange{packed},
            SmallVector<AffineMap>{builder.getMultiDimIdentityMap(2)},
            SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
            [&](OpBuilder &b, Location loc, ValueRange) {
              SmallVector<Value> coordinates(type.getRank());
              coordinates[freeAxis] = b.create<linalg::IndexOp>(loc, left ? 0 : 1);
              Value linear = b.create<linalg::IndexOp>(loc, left ? 1 : 0);
              for (int64_t position = axes.size() - 1; position > 0; --position) {
                coordinates[axes[position]] = b.create<arith::RemSIOp>(loc, linear, reductionSizes[position]);
                linear = b.create<arith::DivSIOp>(loc, linear, reductionSizes[position]);
              }
              coordinates[axes[0]] = linear;
              b.create<linalg::YieldOp>(loc, ValueRange{b.create<memref::LoadOp>(loc, source, coordinates)});
            });
        return packed;
      };
      Value left = pack(lhs, leftAxes, true), right = pack(rhs, rightAxes, false);
      auto sizes = extents(resultType, loc);
      if (failed(sizes)) return failure();
      Value output = allocate(resultType, *sizes, loc);
      matrix(left, right, output, loc);
      values.map(operation.getResult(), output);
      return success();
    }
    if (lhsType.getRank() != batchRank + 2 || rhsType.getRank() != batchRank + 2 || pairs.size() != 1 ||
        cast<IntegerAttr>(cast<ArrayAttr>(pairs[0])[0]).getInt() != batchRank + 1 ||
        cast<IntegerAttr>(cast<ArrayAttr>(pairs[0])[1]).getInt() != batchRank ||
        (!floating && !integer))
      return operation.emitError("CPU construction requires a floating vector dot or a matrix contraction with lossless floating widening or i8 to i32 accumulation");
    for (auto [axis, pair] : llvm::enumerate(operation.getBatch())) {
      auto relation = cast<ArrayAttr>(pair);
      if (cast<IntegerAttr>(relation[0]).getInt() != static_cast<int64_t>(axis) ||
          cast<IntegerAttr>(relation[1]).getInt() != static_cast<int64_t>(axis))
        return operation.emitError("CPU contraction batch axes must form a shared leading domain");
    }
    Location loc = operation.getLoc();
    auto sizes = extents(resultType, loc);
    if (failed(sizes)) return failure();
    Value output = allocate(resultType, *sizes, loc);
    Value lhs = values.lookup(operation.getLhs()), rhs = values.lookup(operation.getRhs());
    if (!batchRank) matrix(lhs, rhs, output, loc);
    else {
      SmallVector<Value> begins(batchRank, constant(loc, 0)), steps(batchRank, constant(loc, 1)), ends;
      for (int64_t axis = 0; axis < batchRank; ++axis)
        ends.push_back(builder.create<memref::DimOp>(loc, lhs, axis));
      auto batches = builder.create<scf::ParallelOp>(loc, begins, ends, steps);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(batches.getBody());
      auto slice = [&](Value source) -> Value {
        auto type = cast<MemRefType>(source.getType());
        SmallVector<OpFoldResult> offsets, extents, strides(type.getRank(), builder.getIndexAttr(1));
        for (Value coordinate : batches.getInductionVars()) {
          offsets.push_back(coordinate);
          extents.push_back(builder.getIndexAttr(1));
        }
        for (int64_t axis = batchRank; axis < type.getRank(); ++axis) {
          offsets.push_back(builder.getIndexAttr(0));
          extents.push_back(type.isDynamicDim(axis)
              ? OpFoldResult(builder.create<memref::DimOp>(loc, source, axis).getResult())
              : builder.getIndexAttr(type.getDimSize(axis)));
        }
        auto result = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
            type.getShape().take_back(2), type, offsets, extents, strides));
        return builder.create<memref::SubViewOp>(loc, result, source, offsets, extents, strides);
      };
      matrix(slice(lhs), slice(rhs), slice(output), loc);
    }
    values.map(operation.getResult(), output);
    return success();
  }

  LogicalResult sparseContract(SparseContractOp operation) {
    auto lhsType = cast<RankedTensorType>(operation.getCompressed().getType());
    auto rhsType = cast<RankedTensorType>(operation.getRhs().getType());
    auto resultType = cast<RankedTensorType>(operation.getResult().getType());
    Type element = lhsType.getElementType(), accumulator = resultType.getElementType();
    auto pairs = operation.getReduce();
    if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || !operation.getBatch().empty() ||
        operation.getFormat().getCompressionAxis() != 1 || pairs.size() != 1 ||
        cast<IntegerAttr>(cast<ArrayAttr>(pairs[0])[0]).getInt() != 1 ||
        cast<IntegerAttr>(cast<ArrayAttr>(pairs[0])[1]).getInt() != 0 ||
        !isa<FloatType>(element) || element != rhsType.getElementType() ||
        (!accumulator.isF32() && !accumulator.isF64()) ||
        element.getIntOrFloatBitWidth() > accumulator.getIntOrFloatBitWidth())
      return operation.emitError("CPU sparse contraction requires rank-two compressed matrices and lossless floating widening");
    Location loc = operation.getLoc();
    auto sizes = extents(resultType, loc);
    if (failed(sizes)) return failure();
    Value output = allocate(resultType, *sizes, loc);
    Value compressed = values.lookup(operation.getCompressed()), rhs = values.lookup(operation.getRhs());
    SmallVector<Value> positions = flattened(ValueRange{operation.getMetadata()});
    int64_t nonzeros = operation.getFormat().getKind() == 0 ? 1 : 2;
    int64_t groupSize = nonzeros * 2;
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(accumulator));
    builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{output});
    AffineExpr m, compressedK, n;
    bindDims(builder.getContext(), m, compressedK, n);
    SmallVector<AffineMap> maps{
        AffineMap::get(3, 0, {m, compressedK}, builder.getContext()),
        AffineMap::get(3, 0, {m, n}, builder.getContext())};
    builder.create<linalg::GenericOp>(loc, ValueRange{compressed}, ValueRange{output}, maps,
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::reduction,
                                        utils::IteratorType::parallel},
        [&](OpBuilder &b, Location loc, ValueRange arguments) {
          auto index = [&](int64_t value) -> Value { return b.create<arith::ConstantIndexOp>(loc, value); };
          Value row = b.create<linalg::IndexOp>(loc, 0), column = b.create<linalg::IndexOp>(loc, 2);
          Value ordinal = b.create<linalg::IndexOp>(loc, 1);
          Value group = b.create<arith::DivSIOp>(loc, ordinal, index(nonzeros));
          auto position = [&](Value memory) -> Value {
            Value value = b.create<memref::LoadOp>(loc, memory, ValueRange{row, group});
            if (value.getType().isIndex()) return value;
            return cast<IntegerType>(value.getType()).isUnsigned()
                ? Value(b.create<arith::IndexCastUIOp>(loc, b.getIndexType(), value))
                : Value(b.create<arith::IndexCastOp>(loc, b.getIndexType(), value));
          };
          Value relative = position(positions[0]);
          if (nonzeros == 2) {
            Value first = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                b.create<arith::RemSIOp>(loc, ordinal, index(nonzeros)), index(0));
            relative = b.create<arith::SelectOp>(loc, first, relative, position(positions[1]));
          }
          Value reduction = b.create<arith::AddIOp>(loc,
              b.create<arith::MulIOp>(loc, group, index(groupSize)), relative);
          Value left = arguments[0], right = b.create<memref::LoadOp>(loc, rhs, ValueRange{reduction, column});
          if (element != accumulator) {
            left = b.create<arith::ExtFOp>(loc, accumulator, left);
            right = b.create<arith::ExtFOp>(loc, accumulator, right);
          }
          b.create<linalg::YieldOp>(loc, ValueRange{b.create<math::FmaOp>(loc, left, right, arguments[1])});
        });
    values.map(operation.getResult(), output);
    return success();
  }

  LogicalResult scaledContract(ScaledContractOp operation) {
    auto outputType = cast<RankedTensorType>(operation.getResult().getType());
    auto carrier = [](Value value, ScaledFormat format) {
      Type element = cast<RankedTensorType>(value.getType()).getElementType();
      return format == ScaledFormat::E4M3 ? isa<Float8E4M3FNType>(element)
           : format == ScaledFormat::E2M1 && element.isInteger(8);
    };
    auto scale = [](Value value) {
      Type element = cast<RankedTensorType>(value.getType()).getElementType();
      return element.isInteger(8) || element.isF32();
    };
    if (!outputType.getElementType().isF32() ||
        !carrier(operation.getLhs(), operation.getLhsFormat()) ||
        !carrier(operation.getRhs(), operation.getRhsFormat()) ||
        !scale(operation.getLhsScale()) || !scale(operation.getRhsScale()))
      return operation.emitError("CPU scaled contraction requires E4M3 or E2M1 carriers, byte E8M0 or f32 scales and f32 accumulation");
    Location loc = operation.getLoc();
    int64_t groupSize = operation.getLhsGroupSize();
    Value lhs = values.lookup(operation.getLhs()), rhs = values.lookup(operation.getRhs());
    Value groups = builder.create<memref::DimOp>(loc, lhs, 1);
    Value depth = builder.create<arith::MulIOp>(loc, groups, constant(loc, groupSize));
    auto sizes = extents(outputType, loc);
    if (failed(sizes)) return failure();
    auto decode = [&](Value source, Value scales, ScaledFormat format, bool left) {
      SmallVector<Value> shape = left ? SmallVector<Value>{(*sizes)[0], depth}
                                     : SmallVector<Value>{depth, (*sizes)[1]};
      auto type = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, builder.getF32Type());
      Value decoded = allocate(type, shape, loc);
      builder.create<linalg::GenericOp>(loc, ValueRange{}, ValueRange{decoded},
          SmallVector<AffineMap>{builder.getMultiDimIdentityMap(2)},
          SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
          [&](OpBuilder &b, Location loc, ValueRange) {
            auto integer = [&](int64_t value) -> Value { return b.create<arith::ConstantIntOp>(loc, value, 32); };
            auto index = [&](int64_t value) -> Value { return b.create<arith::ConstantIndexOp>(loc, value); };
            Value row = b.create<linalg::IndexOp>(loc, 0), column = b.create<linalg::IndexOp>(loc, 1);
            Value free = left ? row : column, reduction = left ? column : row;
            Value group = b.create<arith::DivSIOp>(loc, reduction, index(groupSize));
            Value inner = b.create<arith::RemSIOp>(loc, reduction, index(groupSize));
            Value position = format == ScaledFormat::E2M1
                ? Value(b.create<arith::DivSIOp>(loc, inner, index(2))) : inner;
            SmallVector<Value> coordinates = left ? SmallVector<Value>{free, group, position}
                                                  : SmallVector<Value>{group, position, free};
            Value raw = b.create<memref::LoadOp>(loc, source, coordinates);
            Value number;
            if (format == ScaledFormat::E4M3) number = b.create<arith::ExtFOp>(loc, b.getF32Type(), raw);
            else {
              Value bits = b.create<arith::ExtUIOp>(loc, b.getI32Type(), raw);
              Value lane = b.create<arith::IndexCastOp>(loc, b.getI32Type(), b.create<arith::RemSIOp>(loc, inner, index(2)));
              Value shift = b.create<arith::MulIOp>(loc, lane, integer(4));
              Value nibble = b.create<arith::ShRUIOp>(loc, bits, shift);
              Value magnitude = b.create<arith::AndIOp>(loc, nibble, integer(7));
              static constexpr float table[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
              number = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(table[0]));
              for (int64_t code = 1; code < 8; ++code) {
                Value matches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, magnitude, integer(code));
                number = b.create<arith::SelectOp>(loc, matches,
                    b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(table[code])), number);
              }
              Value sign = b.create<arith::AndIOp>(loc, nibble, integer(8));
              Value negative = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, sign, integer(0));
              number = b.create<arith::SelectOp>(loc, negative, b.create<arith::NegFOp>(loc, number), number);
            }
            Value rawScale = b.create<memref::LoadOp>(loc, scales, ValueRange{free, group});
            Value scale = rawScale;
            if (!scale.getType().isF32()) {
              Value exponent = b.create<arith::ExtUIOp>(loc, b.getI32Type(), rawScale);
              Value bits = b.create<arith::ShLIOp>(loc, exponent, integer(23));
              Value subnormal = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, exponent, integer(0));
              bits = b.create<arith::SelectOp>(loc, subnormal, integer(0x00400000), bits);
              Value nan = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, exponent, integer(255));
              bits = b.create<arith::SelectOp>(loc, nan, integer(0x7fc00000), bits);
              scale = b.create<arith::BitcastOp>(loc, b.getF32Type(), bits);
            }
            b.create<linalg::YieldOp>(loc, ValueRange{b.create<arith::MulFOp>(loc, number, scale)});
          });
      return decoded;
    };
    Value left = decode(lhs, values.lookup(operation.getLhsScale()), operation.getLhsFormat(), true);
    Value right = decode(rhs, values.lookup(operation.getRhsScale()), operation.getRhsFormat(), false);
    Value output = allocate(outputType, *sizes, loc);
    matrix(left, right, output, loc);
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

  void bindScalars(ValueRange originals, ValueRange components) {
    unsigned offset = 0;
    for (Value original : originals) {
      SmallVector<Type> leaves;
      flattenTypes(original.getType(), leaves);
      bindProduct(original, components.slice(offset, leaves.size()));
      offset += leaves.size();
    }
  }

  LogicalResult scalarControl(Operation *operation, TypeRange resultTypes) {
    auto savedDimensions = dimensions;
    Location loc = operation->getLoc();
    auto body = [&](Block &source, Block *target) {
      if (!target->empty() && target->back().hasTrait<OpTrait::IsTerminator>()) target->back().erase();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(target);
      if (failed(lowerBlock(source))) return failure();
      builder.create<scf::YieldOp>(loc, flattened(source.getTerminator()->getOperands()));
      dimensions = savedDimensions;
      return success();
    };
    if (auto conditional = dyn_cast<IfOp>(operation)) {
      auto target = builder.create<scf::IfOp>(loc, resultTypes, values.lookup(conditional.getCondition()), true);
      if (failed(body(conditional.getThenRegion().front(), target.thenBlock())) ||
          failed(body(conditional.getElseRegion().front(), target.elseBlock()))) return failure();
      bindScalars(operation->getResults(), target.getResults());
      return success();
    }
    auto loop = cast<ForOp>(operation);
    if (!domains.count(loop.getInputs().front())) return operation->emitError("CPU ordered for requires a realized domain");
    Domain domain = domains.lookup(loop.getInputs().front());
    auto target = builder.create<scf::ForOp>(loc, domain.begin, domain.end, domain.step,
                                            flattened(loop.getInputs().drop_front()));
    Block &source = loop.getBody().front();
    values.map(source.getArgument(0), target.getInductionVar());
    bindScalars(source.getArguments().drop_front(), target.getRegionIterArgs());
    if (failed(body(source, target.getBody()))) return failure();
    bindScalars(operation->getResults(), target.getResults());
    return success();
  }

  LogicalResult orderedControl(Operation *operation) {
    SmallVector<Type> leaves;
    for (Type type : operation->getResultTypes()) flattenTypes(type, leaves);
    if (isa<IfOp, ForOp>(operation) && llvm::all_of(leaves, [](Type type) {
          return isa<IntegerType, IndexType, FloatType>(type);
        })) return scalarControl(operation, leaves);
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
      auto physicalBegin = indexValue(begin, op.getBounds()[0].getType(), loc);
      auto physicalEnd = indexValue(end, op.getBounds()[1].getType(), loc);
      auto physicalStep = indexValue(step, op.getBounds().size() == 3 ? op.getBounds()[2].getType() : builder.getIndexType(), loc);
      if (failed(physicalBegin) || failed(physicalEnd) || failed(physicalStep)) return failure();
      Value extent = domainExtent(*physicalBegin, *physicalEnd, *physicalStep, loc);
      Domain domain{*physicalBegin, *physicalEnd, *physicalStep, extent};
      domains[op.getResult()] = domain;
      dimensions[cast<IntegerAttr>(op.getExtentDimensions()[0]).getInt()] = domain.extent;
    } else if (auto op = dyn_cast<SubregionOp>(operation)) {
      auto source = domains.find(op.getInputs()[0]);
      if (source == domains.end()) return op.emitError("CPU subregion requires a realized source domain");
      Domain domain = source->second;
      unsigned position = 1;
      if (op.getHasStart()) {
        Value input = op.getInputs()[position++];
        auto start = indexValue(values.lookup(input), input.getType(), loc);
        if (failed(start)) return failure();
        domain.begin = *start;
      }
      if (op.getHasStop()) {
        Value input = op.getInputs()[position];
        auto stop = indexValue(values.lookup(input), input.getType(), loc);
        if (failed(stop)) return failure();
        domain.end = *stop;
      }
      domain.extent = domainExtent(domain.begin, domain.end, domain.step, loc);
      domains[op.getResult()] = domain;
      dimensions[cast<IntegerAttr>(op.getExtentDimensions()[0]).getInt()] = domain.extent;
    } else if (auto op = dyn_cast<RegionEndOp>(operation)) {
      auto source = domains.find(op.getSource());
      if (source == domains.end()) return op.emitError("CPU region end requires a realized source domain");
      values.map(op.getResult(), source->second.end);
    } else if (auto op = dyn_cast<ParallelOp>(operation)) {
      if (!domains.count(op.getSource()))
        return op.emitError("CPU parallel source is not a supported domain");
      Domain domain = domains.lookup(op.getSource());
      auto parallel = builder.create<scf::ParallelOp>(loc, ValueRange{constant(loc, 0)},
          ValueRange{domain.extent}, ValueRange{constant(loc, 1)});
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(parallel.getBody());
      Value coordinate = builder.createOrFold<arith::AddIOp>(loc, domain.begin,
          builder.createOrFold<arith::MulIOp>(loc, parallel.getInductionVars()[0], domain.step));
      values.map(op.getBody().front().getArgument(0), coordinate);
      return lowerBlock(op.getBody().front());
    } else if (isa<IfOp, ForOp, WhileOp>(operation)) {
      return orderedControl(operation);
    } else if (auto op = dyn_cast<BufferOp>(operation)) {
      if (!analysis.logicalBuffer(op).isExact())
        return op.emitError("CPU logical buffer requires an exact lexical allocation fact");
      auto tensor = cast<RankedTensorType>(cast<BufferType>(op.getResult().getType()).getTensor());
      auto sizes = extents(tensor, loc);
      if (failed(sizes)) return failure();
      Value storage = allocate(tensor, *sizes, loc);
      if (op.getInitialOperand())
        copyToSlot(values.lookup(op.getInputs()[*op.getInitialOperand()]), storage, loc);
      values.map(op.getResult(), storage);
      bindDimensions(tensor, storage, loc);
    } else if (isa<AssumeInBoundsOp>(operation)) {
      return success();
    } else if (isa<AtomicLoadOp, AtomicStoreOp, AtomicRMWOp, AtomicCompareExchangeOp>(operation)) {
      return atomicAccess(operation);
    } else if (auto op = dyn_cast<ScatterReduceOp>(operation)) {
      return scatterReduce(op);
    } else if (isa<ViewLoadOp, BufferLoadOp>(operation)) {
      auto value = indexed(operation);
      if (failed(value)) return failure();
      values.map(operation->getResult(0), *value);
    } else if (isa<ViewStoreOp, BufferStoreOp, ScatterUniqueOp>(operation)) {
      if (!alwaysValid(operation))
        return operation->emitError("CPU predicated destination stores are not implemented");
      auto fact = analysis.indexRelation(operation);
      if (failed(fact)) return failure();
      if (isa<ScatterUniqueOp>(operation) || llvm::any_of(fact->terms, [&](const auto &term) {
            return term.kind == 1 || (term.kind == 3 && term.operands.size() == 1 &&
                                     isa<MemRefType>(values.lookup(term.operands[0]).getType()));
          })) return indexedWrite(operation);
      auto destination = indexed(operation);
      if (failed(destination)) return failure();
      unsigned position = isa<ViewStoreOp>(operation) ? cast<ViewStoreOp>(operation).getValueOperandIndex()
                                                     : cast<BufferStoreOp>(operation).getValueOperandIndex();
      Value input = values.lookup(operation->getOperand(position));
      if (isa<MemRefType>(input.getType()))
        builder.create<memref::CopyOp>(loc, input, *destination);
      else
        builder.create<memref::StoreOp>(loc, input, *destination, ValueRange{});
    } else if (auto op = dyn_cast<GatherOp>(operation)) {
      auto value = indexed(operation);
      if (failed(value)) return failure();
      values.map(op.getResult(), *value);
    } else if (isa<BinaryOp, UnaryOp, CompareOp, SelectOp, CastOp, BitcastOp, MaskOp, RandomBitsOp>(operation)) {
      return pointwise(operation);
    } else if (auto op = dyn_cast<FullOp>(operation)) {
      auto tensor = cast<RankedTensorType>(op.getResult().getType());
      auto sizes = extents(tensor, loc);
      if (failed(sizes)) return failure();
      Value output = allocate(tensor, *sizes, loc);
      builder.create<linalg::FillOp>(loc, ValueRange{values.lookup(op.getInputs()[0])}, ValueRange{output});
      values.map(op.getResult(), output);
    } else if (auto op = dyn_cast<IndicesOp>(operation)) {
      auto type = cast<RankedTensorType>(op.getResult().getType());
      auto sizes = extents(type, loc);
      if (failed(sizes)) return failure();
      Value begin = constant(loc, 0), step = constant(loc, 1);
      int64_t axis = 0;
      if (op.getTensorAxis()) {
        axis = *op.getTensorAxis();
      } else {
        if (!domains.count(op.getSource()))
          return op.emitError("CPU indices requires a realized source domain or a tensor axis");
        Domain domain = domains.lookup(op.getSource());
        begin = domain.begin; step = domain.step;
      }
      Value output = allocate(type, *sizes, loc);
      builder.create<linalg::GenericOp>(loc, ValueRange{}, ValueRange{output},
          SmallVector<AffineMap>{builder.getMultiDimIdentityMap(type.getRank())},
          SmallVector<utils::IteratorType>(type.getRank(), utils::IteratorType::parallel),
          [&](OpBuilder &nested, Location location, ValueRange) {
            Value coordinate = nested.create<linalg::IndexOp>(location, axis);
            coordinate = nested.createOrFold<arith::AddIOp>(location, begin,
                nested.createOrFold<arith::MulIOp>(location, coordinate, step));
            if (!type.getElementType().isIndex()) coordinate = nested.create<arith::IndexCastOp>(location, type.getElementType(), coordinate);
            nested.create<linalg::YieldOp>(location, coordinate);
          });
      values.map(op.getResult(), output);
    } else if (auto op = dyn_cast<JoinOp>(operation)) {
      auto tensor = cast<RankedTensorType>(op.getResult().getType());
      auto sizes = extents(tensor, loc);
      if (failed(sizes)) return failure();
      Value output = allocate(tensor, *sizes, loc);
      SmallVector<AffineExpr> prefix;
      for (int64_t axis = 0; axis + 1 < tensor.getRank(); ++axis)
        prefix.push_back(builder.getAffineDimExpr(axis));
      auto inputMap = AffineMap::get(tensor.getRank(), 0, prefix, builder.getContext());
      builder.create<linalg::GenericOp>(loc,
          ValueRange{values.lookup(op.getLhs()), values.lookup(op.getRhs())}, ValueRange{output},
          SmallVector<AffineMap>{inputMap, inputMap, builder.getMultiDimIdentityMap(tensor.getRank())},
          SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
          [&](OpBuilder &nested, Location location, ValueRange inputs) {
            Value component = nested.create<linalg::IndexOp>(location, tensor.getRank() - 1);
            Value first = nested.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq,
                component, nested.create<arith::ConstantIndexOp>(location, 0));
            Value selected = nested.create<arith::SelectOp>(location, first, inputs[0], inputs[1]);
            nested.create<linalg::YieldOp>(location, selected);
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
        bool unitAxesOnly = true;
        auto sourceTensor = cast<RankedTensorType>(operation->getOperand(0).getType());
        auto sourceShape = dyn_cast_or_null<TensorShapeAttr>(sourceTensor.getEncoding());
        auto resultShape = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
        for (int64_t axis = 0; axis < source.getRank(); ++axis) {
          if (source.getDimSize(axis) == 1) { coordinates[axis] = builder.getAffineConstantExpr(0); continue; }
          while (next < tensor.getRank() && tensor.getDimSize(next) == 1) ++next;
          if (next == tensor.getRank() || source.getDimSize(axis) != tensor.getDimSize(next)) {
            unitAxesOnly = false;
            break;
          }
          if (source.isDynamicDim(axis) && (!sourceShape || !resultShape ||
              sourceShape.getDimensions()[axis] != resultShape.getDimensions()[next])) {
            unitAxesOnly = false;
            break;
          }
          coordinates[axis] = builder.getAffineDimExpr(next++);
        }
        while (next < tensor.getRank() && tensor.getDimSize(next) == 1) ++next;
        unitAxesOnly &= next == tensor.getRank();
        if (!unitAxesOnly) {
          SmallVector<Value> sourceSizes;
          for (int64_t axis = 0; axis < source.getRank(); ++axis)
            sourceSizes.push_back(builder.create<memref::DimOp>(loc, input, axis));
          Value output = allocate(tensor, *sizes, loc);
          builder.create<linalg::GenericOp>(loc, ValueRange{}, ValueRange{output},
              SmallVector<AffineMap>{builder.getMultiDimIdentityMap(tensor.getRank())},
              SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
              [&](OpBuilder &nested, Location location, ValueRange) {
                Value linear = nested.create<arith::ConstantIndexOp>(location, 0);
                for (auto [axis, size] : llvm::enumerate(*sizes))
                  linear = nested.create<arith::AddIOp>(location,
                      nested.create<arith::MulIOp>(location, linear, size), nested.create<linalg::IndexOp>(location, axis));
                SmallVector<Value> indices(source.getRank());
                for (int64_t axis = source.getRank() - 1; axis >= 0; --axis) {
                  if (axis == 0) { indices[axis] = linear; break; }
                  indices[axis] = nested.create<arith::RemSIOp>(location, linear, sourceSizes[axis]);
                  linear = nested.create<arith::DivSIOp>(location, linear, sourceSizes[axis]);
                }
                Value value = nested.create<memref::LoadOp>(location, input, indices);
                nested.create<linalg::YieldOp>(location, value);
              });
          values.map(operation->getResult(0), output);
          return success();
        }
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
    } else if (auto op = dyn_cast<HistogramOp>(operation)) {
      auto type = cast<RankedTensorType>(op.getResult().getType());
      Value bins = values.lookup(op.getBins());
      Value output = allocate(type, {bins}, loc);
      auto integer = dyn_cast<IntegerType>(getElementTypeOrSelf(op.getValues().getType()));
      builder.create<cpu::HistogramOp>(loc, values.lookup(op.getValues()), values.lookup(op.getValid()), output,
          integer && integer.isUnsigned());
      values.map(op.getResult(), output);
      bindDimensions(type, output, loc);
    } else if (auto op = dyn_cast<ScanOp>(operation)) {
      return scan(op);
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
      Value output = allocate(cast<RankedTensorType>(op.getResult().getType()), {}, loc);
      builder.create<cpu::QuantizedDotOp>(loc, values.lookup(op.getLhs()), values.lookup(op.getRhs()),
          output, op.getLhsFormatAttr(), op.getRhsFormatAttr());
      values.map(op.getResult(), output);
    } else if (auto op = dyn_cast<ContractOp>(operation)) {
      return contract(op);
    } else if (auto op = dyn_cast<ScaledContractOp>(operation)) {
      return scaledContract(op);
    } else if (auto op = dyn_cast<SparseContractOp>(operation)) {
      return sparseContract(op);
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
  CPUEntryLayout entryLayout;
  func::FuncOp function;
  IRMapping values;
  llvm::DenseMap<Value, SmallVector<Value>> products;
  llvm::DenseMap<int64_t, Value> dimensions;
  llvm::DenseMap<Value, Domain> domains;
  SmallVector<SmallVector<Value>> allocations;
  bool scalarized = false;
};

}

LogicalResult lowerCanonicalKIRToCPU(ModuleOp module, CPUEntryLayout entryLayout) {
  CanonicalKernelAnalysis analysis(module);
  if (failed(analysis.verify())) return failure();
  auto physical = OwningOpRef<ModuleOp>(ModuleOp::create(module.getLoc()));
  Construction construction(module, *physical, entryLayout);
  unsigned count = 0;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (++count != 1)
      return function.emitError("CPU construction requires an inlined single kernel");
    if (failed(construction.lower(function))) return failure();
  }
  if (count == 0) return module.emitError("CPU construction found no kernel");
  realizeIntegerStorage(*physical);
  if (failed(mlir::verify(*physical))) return failure();
  module.getBodyRegion().takeBody(physical->getBodyRegion());
  module->setAttrs((*physical)->getAttrs());
  return success();
}

}
