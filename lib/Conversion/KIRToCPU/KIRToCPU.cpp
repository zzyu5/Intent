#include "Intent/Conversion/KIRToCPU/KIRToCPU.h"
#include "Intent/Conversion/ScalarLowering.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "Intent/Analysis/CanonicalKernel.h"
#include "Intent/Analysis/ProductSchema.h"
#include "Intent/Analysis/ContractionAxes.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CollectiveHelpers.h"
#include "Intent/Interfaces/StructuredOpInterface.h"
#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

class Construction {
public:
  Construction(ModuleOp original, ModuleOp physical, CPUEntryLayout entryLayout)
      : analysis(original), module(physical), builder(physical.getContext()), entryLayout(entryLayout) {}

  LogicalResult lower(func::FuncOp source) {
    SmallVector<Type> types;
    auto interface = buildPublicInterface(source);
    if (failed(interface)) return failure();
    SmallVector<Value> runtimeArguments;
    for (BlockArgument argument : source.getArguments()) {
      if (isa<ConstexprType>(argument.getType())) {
        if (!argument.use_empty())
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
      } else if (argument.getType().isF32() || argument.getType().isF64() || argument.getType().isIndex() ||
                 argument.getType().isInteger(8) || argument.getType().isInteger(16) ||
                 argument.getType().isInteger(32) || argument.getType().isInteger(64) ||
                 argument.getType().isInteger(1)) {
        types.push_back(argument.getType());
      } else {
        return source.emitError("CPU construction does not implement this parameter type");
      }
    }
    builder.setInsertionPointToEnd(module.getBody());
    function = builder.create<func::FuncOp>(source.getLoc(), source.getName(),
                                           builder.getFunctionType(types, {}));
    function->setAttr(interfaceAttr, *interface);
    function->setAttr(cpu::entryRequirementsAttr, cpu::EntryRequirementsAttr::get(
        builder.getContext(), entryLayout == CPUEntryLayout::Contiguous, true));
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

  LogicalResult bindShape(Operation *operation) {
    auto shape = operation->getAttrOfType<ShapeRelationAttr>("shape");
    if (!shape) return success();
    Location loc = operation->getLoc();
    SmallVector<Value> sizes;
    std::optional<int64_t> inferred;
    for (Attribute axis : shape.getAxes()) {
      auto expression = cast<ShapeExprAttr>(axis);
      if (expression.getKind() == 2) {
        inferred = expression.getDimension();
        continue;
      }
      Value size;
      if (expression.getKind() == 0) {
        size = constant(loc, expression.getPayload());
      } else {
        Value operand = isa<BufferOp>(operation)
            ? cast<BufferOp>(operation).getExtents()[expression.getPayload()]
            : operation->getOperand(expression.getPayload());
        auto extent = indexValue(values.lookup(operand), operand.getType(), loc);
        if (failed(extent)) return failure();
        size = *extent;
      }
      dimensions[expression.getDimension()] = size;
      sizes.push_back(size);
    }
    if (inferred) {
      Value knownElements = constant(loc, 1);
      for (Value size : sizes)
        knownElements = builder.createOrFold<arith::MulIOp>(loc, knownElements, size);
      if (matchPattern(knownElements, m_Zero()))
        return operation->emitError("CPU inferred reshape requires a uniquely determined extent");
      Value source = values.lookup(operation->getOperand(0));
      Value sourceElements = constant(loc, 1);
      for (int64_t axis = 0; axis < cast<ShapedType>(source.getType()).getRank(); ++axis)
        sourceElements = builder.createOrFold<arith::MulIOp>(loc, sourceElements,
            dimension(builder, loc, source, axis));
      // A legal inferred reshape has a nonzero known product and exact quotient.
      dimensions[*inferred] = builder.createOrFold<arith::DivSIOp>(loc, sourceElements, knownElements);
    }
    return success();
  }

  static RankedTensorType tensorType(Type type) {
    auto tensor = cast<RankedTensorType>(type);
    return RankedTensorType::get(tensor.getShape(), tensor.getElementType());
  }

  static Type valueType(Type type) {
    return isa<RankedTensorType>(type) ? Type(tensorType(type)) : type;
  }

  static Value dimension(OpBuilder &b, Location loc, Value value, int64_t axis) {
    return isa<RankedTensorType>(value.getType())
        ? Value(b.createOrFold<tensor::DimOp>(loc, value, axis))
        : Value(b.createOrFold<memref::DimOp>(loc, value, axis));
  }

  static Value extractElement(OpBuilder &b, Location loc, Value value,
                              ValueRange indices) {
    return isa<RankedTensorType>(value.getType())
        ? Value(b.create<tensor::ExtractOp>(loc, value, indices))
        : Value(b.create<memref::LoadOp>(loc, value, indices));
  }

  Value emptyTensor(RankedTensorType tensor, ArrayRef<Value> sizes, Location loc) {
    SmallVector<Value> dynamic;
    for (auto [axis, size] : llvm::enumerate(sizes))
      if (tensor.isDynamicDim(axis))
        dynamic.push_back(size);
    return builder.create<tensor::EmptyOp>(loc, tensor.getShape(),
                                           tensor.getElementType(), dynamic);
  }

  FailureOr<SmallVector<Value>> emptyResults(TypeRange types, Location loc) {
    SmallVector<Type> leaves;
    appendProductLeafTypes(types, leaves);
    SmallVector<Value> result;
    for (Type leaf : leaves) {
      auto tensor = dyn_cast<RankedTensorType>(leaf);
      if (!tensor) tensor = RankedTensorType::get({}, leaf);
      auto sizes = extents(tensor, loc);
      if (failed(sizes)) return failure();
      result.push_back(emptyTensor(tensor, *sizes, loc));
    }
    return result;
  }

  SmallVector<Value> flattened(Value value) {
    if (getProductComponents(value.getType())) return products.at(value);
    return {values.lookup(value)};
  }

  SmallVector<Value> flattened(ValueRange inputs) {
    SmallVector<Value> result;
    for (Value value : inputs) llvm::append_range(result, flattened(value));
    return result;
  }

  void bindProduct(Value original, ValueRange components) {
    if (getProductComponents(original.getType())) products[original] = llvm::to_vector(components);
    else values.map(original, components.front());
  }

  void bindDimensions(Type original, Value value, Location loc) {
    auto tensor = dyn_cast<RankedTensorType>(original);
    if (!tensor || !isa<ShapedType>(value.getType())) return;
    auto identities = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions();
    for (auto [axis, identity] : llvm::enumerate(identities.asArrayRef()))
      dimensions[identity] = dimension(builder, loc, value, axis);
  }

  void bindValues(ValueRange originals, ValueRange components, Location loc) {
    auto ranges = getProductLeafRanges(originals.getTypes());
    for (auto [original, range] : llvm::zip(originals, ranges)) {
      SmallVector<Type> leaves;
      appendProductLeafTypes(original.getType(), leaves);
      SmallVector<Value> parts;
      for (auto [leaf, value] :
           llvm::zip(leaves, components.slice(range.offset, range.size))) {
        bindDimensions(leaf, value, loc);
        parts.push_back(value);
      }
      bindProduct(original, parts);
    }
  }

  ArrayAttr fieldPaths(TypeRange types) {
    SmallVector<Attribute> fields;
    for (auto [index, type] : llvm::enumerate(types))
      walkProductLeaves(type, [&](Type, ArrayRef<unsigned> path) {
        std::string name = std::to_string(index);
        if (!path.empty()) name += "." + getProductPathName(type, path);
        fields.push_back(builder.getStringAttr(name));
      });
    return builder.getArrayAttr(fields);
  }

  LogicalResult helper(Region &original, Region &target, ValueRange captures = {}) {
    auto savedDimensions = dimensions;
    OpBuilder::InsertionGuard guard(builder);
    Block *body = new Block;
    target.push_back(body);
    SmallVector<Type> inputTypes;
    appendProductLeafTypes(original.front().getArgumentTypes(), inputTypes);
    if (captures.size() > inputTypes.size())
      return original.getParentOp()->emitError("CPU helper capture schema is incomplete");
    for (auto [index, capture] : llvm::enumerate(captures))
      inputTypes[inputTypes.size() - captures.size() + index] = capture.getType();
    for (Type type : inputTypes) body->addArgument(valueType(type), original.getLoc());
    builder.setInsertionPointToStart(body);
    auto ranges = getProductLeafRanges(original.front().getArgumentTypes());
    for (auto [argument, range] : llvm::zip(original.front().getArguments(), ranges)) {
      SmallVector<Type> leaves;
      appendProductLeafTypes(argument.getType(), leaves);
      SmallVector<Value> parts;
      for (auto [leaf, target] :
           llvm::zip(leaves, body->getArguments().slice(range.offset, range.size))) {
        bindDimensions(leaf, target, original.getLoc());
        parts.push_back(target);
      }
      bindProduct(argument, parts);
    }
    if (failed(lowerBlock(original.front()))) return failure();
    auto results = flattened(original.front().getTerminator()->getOperands());
    if (isa<cpu::SliceReduceOp>(target.getParentOp()))
      builder.create<cpu::SliceReduceYieldOp>(original.getLoc(), results);
    else if (isa<cpu::ScanOp>(target.getParentOp()))
      builder.create<cpu::ScanYieldOp>(original.getLoc(), results);
    else builder.create<cpu::RegionYieldOp>(original.getLoc(), results);
    dimensions = std::move(savedDimensions);
    return success();
  }

  LogicalResult region(Operation *operation) {
    auto schema = cast<StructuredOpInterface>(operation);
    const bool scan = schema.getStructuredKind() == StructuredOpKind::RegionScan;
    int64_t axis = schema.getIterationAxes().front();
    auto sources = flattened(schema.getSources());
    auto identityValues = flattened(schema.getIdentities());
    auto stateValues = flattened(schema.getInitialStates());
    auto captures = flattened(schema.getCaptures());
    Location loc = operation->getLoc();
    SmallVector<Type> results;
    appendProductLeafTypes(operation->getResultTypes(), results);
    for (Type &type : results) type = valueType(type);
    auto outputs = emptyResults(operation->getResultTypes(), loc);
    if (failed(outputs)) return failure();
    auto outputFields = fieldPaths(schema.getEmittedResults().getTypes());
    SmallVector<int64_t> outputAxes;
    if (scan) {
      Block &emit = schema.getEmitRegion()->front();
      SmallVector<Type> sourceLeaves;
      appendProductLeafTypes(schema.getEmitSources().front().getType(), sourceLeaves);
      auto sourceType = cast<RankedTensorType>(sourceLeaves.front());
      int64_t member = cast<TensorShapeAttr>(sourceType.getEncoding()).getDimensions()[axis];
      for (Type type : emit.getTerminator()->getOperandTypes()) {
        SmallVector<Type> leaves; appendProductLeafTypes(type, leaves);
        for (Type leaf : leaves) {
          auto tensor = cast<RankedTensorType>(leaf);
          auto axes = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions().asArrayRef();
          auto found = llvm::find(axes, member);
          if (found == axes.end()) return operation->emitError("CPU region emit has no source member axis");
          outputAxes.push_back(found - axes.begin());
        }
      }
    }
    Operation *target;
    if (scan)
      target = builder.create<cpu::RegionScanOp>(loc,
          TypeRange(results).take_front(outputFields.size()),
          TypeRange(results).drop_front(outputFields.size()),
          sources, identityValues, stateValues, captures,
          ValueRange(*outputs).take_front(outputFields.size()),
          ValueRange(*outputs).drop_front(outputFields.size()), axis,
          fieldPaths(schema.getIdentities().getTypes()), fieldPaths(schema.getInitialStates().getTypes()),
          builder.getDenseI64ArrayAttr(outputAxes), IntegerAttr());
    else
      target = builder.create<cpu::RegionFoldOp>(loc, results, sources, identityValues, captures,
          *outputs, axis, fieldPaths(schema.getIdentities().getTypes()), IntegerAttr());
    auto targetSchema = cast<cpu::RegionOpInterface>(target);
    if (failed(helper(*schema.getSummarizeRegion(), targetSchema.getSummarize(), captures))) return failure();
    if (failed(helper(schema.getCombine(), targetSchema.getCombine()))) return failure();
    if (scan) {
      if (failed(helper(*schema.getApplyRegion(), *targetSchema.getApplyRegion()))) return failure();
      if (failed(helper(*schema.getEmitRegion(), *targetSchema.getEmitRegion(), captures))) return failure();
    }
    bindValues(operation->getResults(), target->getResults(), loc);
    return success();
  }

  SmallVector<AffineMap> pointwiseMaps(ValueRange inputs, int64_t rank) {
    SmallVector<AffineMap> maps;
    for (Value input : inputs) {
      SmallVector<AffineExpr> axes;
      if (auto memory = dyn_cast<RankedTensorType>(input.getType()))
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
    auto type = dyn_cast<ShapedType>(input.getType());
    if (!type) return input;
    SmallVector<Value> indices;
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      indices.push_back(type.getDimSize(axis) == 1
          ? Value(nested.create<arith::ConstantIndexOp>(loc, 0))
          : members[members.size() - type.getRank() + axis]);
    return extractElement(nested, loc, input, indices);
  }

  LogicalResult verifyIndexTerms(Operation *operation, const IndexRelationFact &fact) {
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

  SmallVector<Value> indexedCoordinates(const IndexRelationFact &fact,
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

  LogicalResult indexedWrite(Operation *operation) {
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

  FailureOr<Value> indexedRead(Operation *operation, const IndexRelationFact &fact, Value source) {
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
    auto shape = extents(tensor, operation->getLoc());
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

  LogicalResult atomicAccess(Operation *operation) {
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
      auto extentsOr = extents(tensor, loc);
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
        bindValues(operation->getResults(), results, loc);
        return {};
      }
      SmallVector<Value> updated;
      for (auto [value, output] : llvm::zip(results, carried))
        updated.push_back(builder.create<tensor::InsertOp>(loc, value, output, members));
      return updated;
    };
    auto results = traverse(0, outputs);
    if (used && tensor) bindValues(operation->getResults(), results, loc);
    return success();
  }

  LogicalResult scatterReduce(ScatterReduceOp operation) {
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

  FailureOr<Value> indexed(Operation *operation) {
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
    auto shape = extents(tensor, operation->getLoc());
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
      if (unary.getOperatorKind() == UnaryOperator::Sigmoid) {
        Value one = builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(arguments[0].getType(), 1.0));
        Value negative = builder.create<arith::NegFOp>(loc, arguments[0]);
        Value denominator = builder.create<arith::AddFOp>(loc, one, builder.create<math::ExpOp>(loc, negative));
        return Value(builder.create<arith::DivFOp>(loc, one, denominator));
      }
    } else if (auto cast = dyn_cast<CastOp>(operation)) {
      if (getElementTypeOrSelf(cast.getType()).isInteger(1) &&
          !getElementTypeOrSelf(cast.getInput().getType()).isInteger(1))
        return cast.emitError("CPU numeric-to-bool cast is not implemented"), failure();
    }
    return intent::lowerScalarOperation(operation, arguments, builder);
  }

  LogicalResult pointwise(Operation *operation) {
    Type resultType = operation->getResult(0).getType();
    auto tensor = dyn_cast<RankedTensorType>(resultType);
    SmallVector<Value> arguments;
    for (Value input : operation->getOperands())
      arguments.push_back(values.lookup(input));
    if (!tensor) {
      auto result = arithmetic(operation, arguments, builder);
      if (failed(result)) return failure();
      values.map(operation->getResult(0), *result);
      return success();
    }
    auto sizes = extents(tensor, operation->getLoc());
    if (failed(sizes)) return failure();
    Value output = emptyTensor(tensor, *sizes, operation->getLoc());
    LogicalResult status = success();
    auto result = builder.create<linalg::GenericOp>(operation->getLoc(), TypeRange{output.getType()}, arguments,
        ValueRange{output}, pointwiseMaps(arguments, tensor.getRank()),
        SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &nested, Location loc, ValueRange scalars) {
          auto result = arithmetic(operation, scalars.take_front(arguments.size()), nested);
          if (failed(result)) { status = failure(); return; }
          nested.create<linalg::YieldOp>(loc, *result);
        });
    values.map(operation->getResult(0), result.getResult(0));
    return status;
  }

  LogicalResult scan(ScanOp operation) {
    SmallVector<Type> results;
    appendProductLeafTypes(operation.getResultTypes(), results);
    for (Type &type : results) type = valueType(type);
    auto outputs = emptyResults(operation.getResultTypes(), operation.getLoc());
    if (failed(outputs)) return failure();
    auto target = builder.create<cpu::ScanOp>(operation.getLoc(), results,
        flattened(operation.getSources()), flattened(operation.getIdentities()),
        flattened(operation.getCaptures()), *outputs,
        operation.getAxis(), operation.getInclusive(), operation.getReverse());
    if (failed(helper(operation.getCombine(), target.getCombine(), flattened(operation.getCaptures())))) return failure();
    bindValues(operation.getResults(), target.getResults(), operation.getLoc());
    return success();
  }

  LogicalResult reduce(ReduceOp operation) {
    auto sources = flattened(operation.getSources());
    auto identities = flattened(operation.getIdentities());
    auto captures = flattened(operation.getCaptures());
    Location loc = operation.getLoc();
    SmallVector<Type> results;
    appendProductLeafTypes(operation.getResultTypes(), results);
    for (Type &type : results) type = valueType(type);
    auto outputs = emptyResults(operation.getResultTypes(), loc);
    if (failed(outputs)) return failure();
    if (results.size() != identities.size() || sources.size() != identities.size())
      return operation.emitError("CPU reduction source, identity and output leaves differ");
    SmallVector<int64_t> axes;
    for (Attribute axis : operation.getAxes())
      axes.push_back(cast<IntegerAttr>(axis).getInt());
    auto reduction = builder.create<cpu::SliceReduceOp>(loc, results, sources, identities,
        captures, *outputs, axes,
        cpu::ReductionOrderAttr::get(builder.getContext(), true, true));
    if (failed(helper(operation.getCombine(), reduction.getCombine(), captures))) return failure();
    bindValues(operation.getResults(), reduction.getResults(), loc);
    return success();
  }

  Value emitContraction(Value lhs, Value rhs, Value destination,
                       ArrayRef<AffineMap> maps, unsigned parallelRank,
                       unsigned reductionRank, Location loc) {
    Type accumulator = cast<RankedTensorType>(destination.getType()).getElementType();
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(accumulator));
    destination = builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{destination}).getResult(0);
    SmallVector<utils::IteratorType> iterators(parallelRank, utils::IteratorType::parallel);
    iterators.append(reductionRank, utils::IteratorType::reduction);
    auto contraction = builder.create<linalg::GenericOp>(loc, TypeRange{destination.getType()}, ValueRange{lhs, rhs}, ValueRange{destination},
        maps, iterators, [](OpBuilder &b, Location loc, ValueRange arguments) {
          Value value;
          if (isa<FloatType>(arguments[2].getType())) {
            Value lhs = arguments[0], rhs = arguments[1];
            if (lhs.getType() != arguments[2].getType())
              lhs = b.create<arith::ExtFOp>(loc, arguments[2].getType(), lhs);
            if (rhs.getType() != arguments[2].getType())
              rhs = b.create<arith::ExtFOp>(loc, arguments[2].getType(), rhs);
            value = b.create<math::FmaOp>(loc, lhs, rhs, arguments[2]);
          } else {
            Value lhs = b.create<arith::ExtSIOp>(loc, arguments[2].getType(), arguments[0]);
            Value rhs = b.create<arith::ExtSIOp>(loc, arguments[2].getType(), arguments[1]);
            value = b.create<arith::AddIOp>(loc,
                b.create<arith::MulIOp>(loc, lhs, rhs), arguments[2]);
          }
          b.create<linalg::YieldOp>(loc, value);
        });
    contraction->setAttr("intent_cpu.reduction_order",
        cpu::ReductionOrderAttr::get(builder.getContext(), true, true));
    return contraction.getResult(0);
  }

  Value matrix(Value lhs, Value rhs, Value destination, Location loc) {
    AffineExpr m, n, k;
    bindDims(builder.getContext(), m, n, k);
    SmallVector<AffineMap> maps = {
        AffineMap::get(3, 0, {m, k}, builder.getContext()),
        AffineMap::get(3, 0, {k, n}, builder.getContext()),
        AffineMap::get(3, 0, {m, n}, builder.getContext())};
    return emitContraction(lhs, rhs, destination, maps, 2, 1, loc);
  }

  LogicalResult contract(ContractOp operation) {
    auto lhsType = cast<RankedTensorType>(operation.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(operation.getRhs().getType());
    auto resultType = cast<RankedTensorType>(operation.getResult().getType());
    SmallVector<int64_t> lhsReduction, rhsReduction, lhsBatch, rhsBatch;
    auto appendPairs = [](ArrayAttr pairs, SmallVectorImpl<int64_t> &left,
                          SmallVectorImpl<int64_t> &right) {
      for (Attribute attribute : pairs) {
        auto pair = cast<ArrayAttr>(attribute);
        left.push_back(cast<IntegerAttr>(pair[0]).getInt());
        right.push_back(cast<IntegerAttr>(pair[1]).getInt());
      }
    };
    appendPairs(operation.getReduce(), lhsReduction, rhsReduction);
    appendPairs(operation.getBatch(), lhsBatch, rhsBatch);
    std::string reason;
    auto axes = ContractionAxes::get(lhsType.getRank(), rhsType.getRank(),
        lhsReduction, rhsReduction, lhsBatch, rhsBatch, &reason);
    if (!axes)
      return operation.emitError("CPU contraction axis schema: ") << reason;
    if (axes->results.size() != static_cast<size_t>(resultType.getRank()))
      return operation.emitError("CPU contraction result rank disagrees with its axis schema");
    Type inputElement = lhsType.getElementType(), accumulator = resultType.getElementType();
    bool floating = isa<FloatType>(inputElement) && inputElement == rhsType.getElementType() &&
        (accumulator.isF32() || accumulator.isF64()) &&
        inputElement.getIntOrFloatBitWidth() <= accumulator.getIntOrFloatBitWidth();
    bool integer = inputElement.isSignlessInteger(8) &&
        rhsType.getElementType().isSignlessInteger(8) && accumulator.isSignlessInteger(32);
    if (!floating && !integer)
      return operation.emitError(
          "CPU contraction requires lossless floating widening to f32/f64 or i8 to i32 accumulation");

    unsigned parallelRank = resultType.getRank();
    unsigned reductionRank = axes->reduction.size();
    unsigned loopRank = parallelRank + reductionRank;
    SmallVector<AffineExpr> lhsMap(lhsType.getRank()), rhsMap(rhsType.getRank()), resultMap;
    for (auto [axis, result] : llvm::enumerate(axes->lhsResultAxes))
      if (result) lhsMap[axis] = builder.getAffineDimExpr(*result);
    for (auto [axis, result] : llvm::enumerate(axes->rhsResultAxes))
      if (result) rhsMap[axis] = builder.getAffineDimExpr(*result);
    for (auto [number, pair] : llvm::enumerate(axes->reduction)) {
      AffineExpr reduction = builder.getAffineDimExpr(parallelRank + number);
      lhsMap[pair.lhs] = reduction;
      rhsMap[pair.rhs] = reduction;
    }
    for (unsigned axis = 0; axis < parallelRank; ++axis)
      resultMap.push_back(builder.getAffineDimExpr(axis));
    SmallVector<AffineMap> maps = {
        AffineMap::get(loopRank, 0, lhsMap, builder.getContext()),
        AffineMap::get(loopRank, 0, rhsMap, builder.getContext()),
        AffineMap::get(loopRank, 0, resultMap, builder.getContext())};
    Location loc = operation.getLoc();
    auto sizes = extents(resultType, loc);
    if (failed(sizes)) return failure();
    Value output = emptyTensor(resultType, *sizes, loc);
    Value result = emitContraction(values.lookup(operation.getLhs()), values.lookup(operation.getRhs()),
                    output, maps, parallelRank, reductionRank, loc);
    values.map(operation.getResult(), result);
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
    Value output = emptyTensor(resultType, *sizes, loc);
    Value compressed = values.lookup(operation.getCompressed()), rhs = values.lookup(operation.getRhs());
    SmallVector<Value> positions = flattened(ValueRange{operation.getMetadata()});
    int64_t nonzeros = operation.getFormat().getKind() == 0 ? 1 : 2;
    int64_t groupSize = nonzeros * 2;
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(accumulator));
    output = builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{output}).getResult(0);
    AffineExpr m, compressedK, n;
    bindDims(builder.getContext(), m, compressedK, n);
    SmallVector<AffineMap> maps{
        AffineMap::get(3, 0, {m, compressedK}, builder.getContext()),
        AffineMap::get(3, 0, {m, n}, builder.getContext())};
    auto result = builder.create<linalg::GenericOp>(loc, TypeRange{output.getType()}, ValueRange{compressed}, ValueRange{output}, maps,
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::reduction,
                                        utils::IteratorType::parallel},
        [&](OpBuilder &b, Location loc, ValueRange arguments) {
          auto index = [&](int64_t value) -> Value { return b.create<arith::ConstantIndexOp>(loc, value); };
          Value row = b.create<linalg::IndexOp>(loc, 0), column = b.create<linalg::IndexOp>(loc, 2);
          Value ordinal = b.create<linalg::IndexOp>(loc, 1);
          Value group = b.create<arith::DivSIOp>(loc, ordinal, index(nonzeros));
          auto position = [&](Value memory) -> Value {
            Value value = extractElement(b, loc, memory, ValueRange{row, group});
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
          Value left = arguments[0], right = extractElement(b, loc, rhs, ValueRange{reduction, column});
          if (element != accumulator) {
            left = b.create<arith::ExtFOp>(loc, accumulator, left);
            right = b.create<arith::ExtFOp>(loc, accumulator, right);
          }
          b.create<linalg::YieldOp>(loc, ValueRange{b.create<math::FmaOp>(loc, left, right, arguments[1])});
        });
    values.map(operation.getResult(), result.getResult(0));
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
    Value groups = dimension(builder, loc, lhs, 1);
    Value depth = builder.create<arith::MulIOp>(loc, groups, constant(loc, groupSize));
    auto sizes = extents(outputType, loc);
    if (failed(sizes)) return failure();
    auto decode = [&](Value source, Value scales, ScaledFormat format, bool left) {
      SmallVector<Value> shape = left ? SmallVector<Value>{(*sizes)[0], depth}
                                     : SmallVector<Value>{depth, (*sizes)[1]};
      auto type = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, builder.getF32Type());
      Value decoded = emptyTensor(type, shape, loc);
      auto result = builder.create<linalg::GenericOp>(loc, TypeRange{decoded.getType()}, ValueRange{}, ValueRange{decoded},
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
            Value raw = extractElement(b, loc, source, coordinates);
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
            Value rawScale = extractElement(b, loc, scales, ValueRange{free, group});
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
      return result.getResult(0);
    };
    Value left = decode(lhs, values.lookup(operation.getLhsScale()), operation.getLhsFormat(), true);
    Value right = decode(rhs, values.lookup(operation.getRhsScale()), operation.getRhsFormat(), false);
    Value output = emptyTensor(outputType, *sizes, loc);
    values.map(operation.getResult(), matrix(left, right, output, loc));
    return success();
  }

  LogicalResult lowerBlock(Block &block) {
    for (Operation &operation : block.without_terminator())
      if (failed(lowerOperation(&operation))) return failure();
    // Only author-declared mutable buffers have lexical lifetime in this
    // construction. Storage for immutable values belongs to bufferization.
    for (auto buffer : block.getOps<BufferOp>())
      builder.create<memref::DeallocOp>(buffer.getLoc(), values.lookup(buffer.getResult()));
    return success();
  }

  LogicalResult orderedControl(Operation *operation) {
    SmallVector<Type> resultTypes;
    appendProductLeafTypes(operation->getResultTypes(), resultTypes);
    for (Type &type : resultTypes) type = valueType(type);
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
      bindValues(operation->getResults(), target.getResults(), loc);
      return success();
    }
    if (auto loop = dyn_cast<ForOp>(operation)) {
      if (!domains.count(loop.getSource())) return operation->emitError("CPU ordered for requires a realized domain");
      Domain domain = domains.lookup(loop.getSource());
      auto target = builder.create<scf::ForOp>(loc, domain.begin, domain.end, domain.step,
                                              flattened(loop.getInitArgs()));
      values.map(loop.getInductionVars().front(), target.getInductionVar());
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(target.getBody());
        bindValues(loop.getRegionIterArgs(), target.getRegionIterArgs(), loc);
      }
      if (failed(body(loop.getBody().front(), target.getBody()))) return failure();
      bindValues(operation->getResults(), target.getResults(), loc);
      return success();
    }
    auto loop = cast<WhileOp>(operation);
    auto initial = flattened(loop.getInitArgs());
    auto target = builder.create<scf::WhileOp>(loc, resultTypes, initial);
    Block &before = target.getBefore().emplaceBlock();
    Block &after = target.getAfter().emplaceBlock();
    for (Type type : TypeRange(initial)) before.addArgument(type, loc);
    for (Type type : resultTypes) after.addArgument(type, loc);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&before);
      bindValues(loop.getBeforeArguments(), before.getArguments(), loc);
      if (failed(lowerBlock(loop.getBefore().front()))) return failure();
      auto condition = cast<ConditionOp>(loop.getBefore().front().getTerminator());
      builder.create<scf::ConditionOp>(loc, values.lookup(condition.getCondition()),
                                       flattened(condition.getArgs()));
    }
    dimensions = savedDimensions;
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&after);
      bindValues(loop.getAfterArguments(), after.getArguments(), loc);
    }
    if (failed(body(loop.getAfter().front(), &after))) return failure();
    bindValues(operation->getResults(), target.getResults(), loc);
    return success();
  }

  bool alwaysValid(Operation *operation) {
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

  LogicalResult lowerOperation(Operation *operation) {
    Location loc = operation->getLoc();
    if (failed(bindShape(operation))) return failure();
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
      auto savedDimensions = dimensions;
      LogicalResult result = lowerBlock(op.getBody().front());
      dimensions = std::move(savedDimensions);
      return result;
    } else if (isa<IfOp, ForOp, WhileOp>(operation)) {
      return orderedControl(operation);
    } else if (auto op = dyn_cast<BufferOp>(operation)) {
      if (!analysis.logicalBuffer(op).isExact())
        return op.emitError("CPU logical buffer requires an exact lexical allocation fact");
      auto tensor = cast<RankedTensorType>(cast<BufferType>(op.getResult().getType()).getTensor());
      auto sizes = extents(tensor, loc);
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
                                     isa<RankedTensorType>(values.lookup(term.operands[0]).getType()));
          })) return indexedWrite(operation);
      auto destination = indexed(operation);
      if (failed(destination)) return failure();
      Value input = values.lookup(cast<IndexedAccessOpInterface>(operation).getStoredValue());
      if (isa<RankedTensorType>(input.getType()))
        builder.create<cpu::WriteOp>(loc, input, *destination);
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
      Value output = emptyTensor(tensor, *sizes, loc);
      auto fill = builder.create<linalg::FillOp>(loc, ValueRange{values.lookup(op.getInputs()[0])}, ValueRange{output});
      values.map(op.getResult(), fill.getResult(0));
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
      Value output = emptyTensor(type, *sizes, loc);
      auto result = builder.create<linalg::GenericOp>(loc, TypeRange{output.getType()}, ValueRange{}, ValueRange{output},
          SmallVector<AffineMap>{builder.getMultiDimIdentityMap(type.getRank())},
          SmallVector<utils::IteratorType>(type.getRank(), utils::IteratorType::parallel),
          [&](OpBuilder &nested, Location location, ValueRange) {
            Value coordinate = nested.create<linalg::IndexOp>(location, axis);
            coordinate = nested.createOrFold<arith::AddIOp>(location, begin,
                nested.createOrFold<arith::MulIOp>(location, coordinate, step));
            if (!type.getElementType().isIndex()) coordinate = nested.create<arith::IndexCastOp>(location, type.getElementType(), coordinate);
            nested.create<linalg::YieldOp>(location, coordinate);
          });
      values.map(op.getResult(), result.getResult(0));
    } else if (auto op = dyn_cast<JoinOp>(operation)) {
      auto tensor = cast<RankedTensorType>(op.getResult().getType());
      auto sizes = extents(tensor, loc);
      if (failed(sizes)) return failure();
      Value output = emptyTensor(tensor, *sizes, loc);
      SmallVector<AffineExpr> prefix;
      for (int64_t axis = 0; axis + 1 < tensor.getRank(); ++axis)
        prefix.push_back(builder.getAffineDimExpr(axis));
      auto inputMap = AffineMap::get(tensor.getRank(), 0, prefix, builder.getContext());
      auto result = builder.create<linalg::GenericOp>(loc, TypeRange{output.getType()},
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
      values.map(op.getResult(), result.getResult(0));
    } else if (isa<TransposeOp, ReshapeOp>(operation)) {
      Value input = values.lookup(operation->getOperand(0));
      auto source = cast<RankedTensorType>(input.getType());
      auto tensor = cast<RankedTensorType>(operation->getResult(0).getType());
      auto sizes = extents(tensor, loc);
      if (failed(sizes)) return failure();
      if (isa<ReshapeOp>(operation)) {
        Value shape = builder.create<tensor::FromElementsOp>(loc,
            RankedTensorType::get({tensor.getRank()}, builder.getIndexType()), *sizes);
        values.map(operation->getResult(0),
            builder.create<tensor::ReshapeOp>(loc, tensorType(tensor), input, shape));
        return success();
      }
      SmallVector<AffineExpr> coordinates(source.getRank());
      for (auto [axis, permuted] : llvm::enumerate(cast<TransposeOp>(operation).getPermutation()))
        coordinates[cast<IntegerAttr>(permuted).getInt()] = builder.getAffineDimExpr(axis);
      Value output = emptyTensor(tensor, *sizes, loc);
      auto result = builder.create<linalg::GenericOp>(loc, TypeRange{output.getType()}, ValueRange{input}, ValueRange{output},
          SmallVector<AffineMap>{AffineMap::get(tensor.getRank(), 0, coordinates, builder.getContext()), builder.getMultiDimIdentityMap(tensor.getRank())},
          SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
          [](OpBuilder &nested, Location location, ValueRange args) { nested.create<linalg::YieldOp>(location, args[0]); });
      values.map(operation->getResult(0), result.getResult(0));
    } else if (auto op = dyn_cast<BroadcastOp>(operation)) {
      auto tensor = cast<RankedTensorType>(op.getResult().getType());
      auto sizes = extents(tensor, loc);
      if (failed(sizes)) return failure();
      Value input = values.lookup(op.getInputs()[0]);
      Value output = emptyTensor(tensor, *sizes, loc);
      auto result = builder.create<linalg::GenericOp>(loc, TypeRange{output.getType()}, ValueRange{input}, ValueRange{output},
          pointwiseMaps(ValueRange{input}, tensor.getRank()),
          SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
          [](OpBuilder &nested, Location loc, ValueRange scalars) {
            nested.create<linalg::YieldOp>(loc, scalars[0]);
          });
      values.map(op.getResult(), result.getResult(0));
    } else if (auto op = dyn_cast<MakeRecordOp>(operation)) {
      bindProduct(op.getResult(), flattened(op.getFields()));
    } else if (auto op = dyn_cast<MakeTupleOp>(operation)) {
      bindProduct(op.getResult(), flattened(op.getComponents()));
    } else if (auto op = dyn_cast<ExtractOp>(operation)) {
      auto range = getProductLeafRange(op.getProduct().getType(),
                                      {static_cast<unsigned>(op.getField())});
      if (failed(range)) return op.emitError("CPU extract has no canonical field schema");
      bindProduct(op.getResult(), ValueRange(products.at(op.getProduct()))
                                      .slice(range->offset, range->size));
    } else if (isa<RegionFoldOp, RegionScanOp>(operation)) {
      return region(operation);
    } else if (auto op = dyn_cast<HistogramOp>(operation)) {
      auto type = cast<RankedTensorType>(op.getResult().getType());
      Value bins = values.lookup(op.getBins());
      Value output = emptyTensor(type, {bins}, loc);
      auto integer = dyn_cast<IntegerType>(getElementTypeOrSelf(op.getValues().getType()));
      auto result = builder.create<cpu::HistogramOp>(loc, tensorType(type), values.lookup(op.getValues()), values.lookup(op.getValid()), output,
          integer && integer.isUnsigned());
      values.map(op.getResult(), result.getResult());
      bindDimensions(type, result.getResult(), loc);
    } else if (auto op = dyn_cast<ScanOp>(operation)) {
      return scan(op);
    } else if (auto op = dyn_cast<ReduceOp>(operation)) {
      return reduce(op);
    } else if (auto op = dyn_cast<QuantizeOp>(operation)) {
      auto tensor = cast<RankedTensorType>(op.getResult().getType());
      auto sizes = extents(tensor, loc);
      if (failed(sizes)) return failure();
      Value output = emptyTensor(tensor, *sizes, loc);
      auto result = builder.create<cpu::QuantizeOp>(loc, tensorType(tensor),
          values.lookup(op.getInput()), output, op.getFormatAttr());
      values.map(op.getResult(), result.getResult());
    } else if (auto op = dyn_cast<QuantizedDotOp>(operation)) {
      Value output = emptyTensor(cast<RankedTensorType>(op.getResult().getType()), {}, loc);
      auto result = builder.create<cpu::QuantizedDotOp>(loc, output.getType(), values.lookup(op.getLhs()), values.lookup(op.getRhs()),
          output, op.getLhsFormatAttr(), op.getRhsFormatAttr());
      values.map(op.getResult(), result.getResult());
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
};

}

LogicalResult lowerCanonicalKIRToCPU(ModuleOp module, CPUEntryLayout entryLayout) {
  CanonicalKernelAnalysis analysis(module);
  if (failed(analysis.verify())) return failure();
  auto physical = OwningOpRef<ModuleOp>(ModuleOp::create(module.getLoc()));
  auto options = readCompileOptions(module);
  if (failed(options)) return failure();
  (*physical)->setAttr(compileOptionsAttr, *options);
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
