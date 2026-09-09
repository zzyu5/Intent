#include "Intent/Target/Weft/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Analysis/AxisRelations.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
namespace {

SmallVector<int64_t> shape(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type)) return llvm::to_vector(value.getShape().asArrayRef());
  if (auto value = dyn_cast<wk::ViewType>(type)) return llvm::to_vector(value.getShape().asArrayRef());
  if (auto value = dyn_cast<wk::SliceType>(type)) return llvm::to_vector(value.getShape().asArrayRef());
  return {};
}

SmallVector<int64_t> axes(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type)) return llvm::to_vector(value.getAxisIds().asArrayRef());
  if (auto value = dyn_cast<wk::ViewType>(type)) return llvm::to_vector(value.getAxisIds().asArrayRef());
  if (auto value = dyn_cast<wk::SliceType>(type)) return llvm::to_vector(value.getAxisIds().asArrayRef());
  return {};
}

Type element(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type)) return value.getElementType();
  return type;
}

struct LocalValue {
  Value value;
  SmallVector<OpFoldResult> sizes;
};

class TaskLowering {
public:
  TaskLowering(func::FuncOp function, ModuleOp output)
      : analysis(function), relations(function), output(output), b(output.getContext()) {}

  LogicalResult lower(cpu::TasksOp tasks, StringRef name) {
    values.clear(); locals.clear();
    Location loc = tasks.getLoc();
    SmallVector<Attribute> names, accesses, symbols;
    SmallVector<int64_t> aliases;
    SmallVector<Type> types{b.getIndexType()};
    auto memoryAccesses = analysis.accesses(tasks);
    names.push_back(b.getStringAttr("coordinate"));
    accesses.push_back(b.getStringAttr("none"));
    aliases.push_back(-1);
    for (auto [number, capture] : llvm::enumerate(tasks.getCaptures())) {
      names.push_back(b.getStringAttr("capture_" + std::to_string(number)));
      if (auto memory = dyn_cast<MemRefType>(capture.getType())) {
        auto ids = relations.axes(capture);
        llvm::SmallSet<int64_t, 8> unique(ids.begin(), ids.end());
        if (unique.size() != ids.size())
          return tasks.emitError("Weft view requires independent logical axes; this reuse needs an explicit axis projection");
        auto abi = analysis.externalView(capture);
        if (!abi) return tasks.emitError("Weft task capture requires an external CPU view; captured scratch is not implemented");
        SmallVector<int64_t> dimensions;
        for (auto [axis, extent] : llvm::enumerate(memory.getShape()))
          dimensions.push_back(ShapedType::isDynamic(extent) ? -ids[axis] : extent);
        types.push_back(wk::ViewType::get(b.getContext(), encoding(), array(dimensions), array(ids)));
        Value storage = analysis.storageRoot(capture);
        bool reads = false, writes = false;
        for (auto access : memoryAccesses)
          if (analysis.storageRoot(access.memory) == storage) {
            reads |= access.read; writes |= access.write;
          }
        accesses.push_back(b.getStringAttr(reads ? (writes ? "readwrite" : "read")
                                               : (writes ? "write" : "none")));
        auto root = cast<BlockArgument>(analysis.storageRoot(capture));
        aliases.push_back(abi.getAccess() == 0 && !abi.getNoalias() ? 0 : root.getArgNumber() + 1);
      } else {
        types.push_back(capture.getType());
        accesses.push_back(b.getStringAttr("none"));
        aliases.push_back(-1);
      }
    }
    for (unsigned axis = 1; axis <= relations.dynamicAxisCount(); ++axis)
      symbols.push_back(b.getStringAttr("axis_" + std::to_string(axis)));
    b.setInsertionPointToEnd(output.getBody());
    auto kernel = b.create<wk::KernelOp>(loc, name, b.getArrayAttr(names),
        b.getArrayAttr(accesses), array(aliases), b.getArrayAttr(symbols), "Intent CPU task");
    Block *body = new Block;
    kernel.getBody().push_back(body);
    for (Type type : types) body->addArgument(type, loc);
    for (auto [source, target] : llvm::zip(tasks.getBody().front().getArguments(), body->getArguments()))
      values.map(source, target);
    b.setInsertionPointToStart(body);
    b.create<wk::RootDomainOp>(loc,
        wk::DomainType::get(b.getContext(), "root", 0, -1, 0, "root", "exact"));
    for (Attribute symbol : symbols)
      b.create<wk::SymbolOp>(loc, b.getIndexType(), cast<StringAttr>(symbol),
          b.getStringAttr("shape"), array({}));
    if (failed(block(tasks.getBody().front()))) return failure();
    b.create<wk::ReturnOp>(loc, ValueRange{});
    return verify(kernel);
  }

private:
  DenseI64ArrayAttr array(ArrayRef<int64_t> entries) { return b.getDenseI64ArrayAttr(entries); }
  wk::EncodingType encoding() { return wk::EncodingType::get(b.getContext(), "f32", "dense", "dense.f32", array({})); }
  Type valueType(Type element, ArrayRef<int64_t> dimensions, ArrayRef<int64_t> ids) {
    if (dimensions.empty()) return element;
    return wk::ValueType::get(b.getContext(), element, array(dimensions), array(ids));
  }
  Value index(Location loc, int64_t value) { return b.create<arith::ConstantIndexOp>(loc, value); }
  Type resultType(Value memory, Type scalar = {}) {
    auto type = cast<MemRefType>(memory.getType());
    auto ids = relations.axes(memory);
    SmallVector<int64_t> dimensions;
    for (auto [axis, extent] : llvm::enumerate(type.getShape()))
      dimensions.push_back(ShapedType::isDynamic(extent) ? -ids[axis] : extent);
    return valueType(scalar ? scalar : type.getElementType(), dimensions, ids);
  }
  bool sameBound(OpFoldResult lhs, OpFoldResult rhs) {
    if (lhs == rhs) return true;
    auto constant = [](OpFoldResult value) -> std::optional<int64_t> {
      if (auto attribute = dyn_cast<Attribute>(value)) return cast<IntegerAttr>(attribute).getInt();
      llvm::APInt integer;
      if (matchPattern(cast<Value>(value), m_ConstantInt(&integer))) return integer.getSExtValue();
      return std::nullopt;
    };
    auto a = constant(lhs), c = constant(rhs);
    return a && c && *a == *c;
  }
  Value localRoot(Value memory) {
    while (true) {
      if (auto view = memory.getDefiningOp<memref::SubViewOp>()) memory = view.getSource();
      else if (auto cast = memory.getDefiningOp<memref::CastOp>()) memory = cast.getSource();
      else return memory;
    }
  }
  bool isLocal(Value memory) {
    return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(localRoot(memory).getDefiningOp());
  }

  FailureOr<Value> view(Value memory) {
    if (values.contains(memory)) return values.lookup(memory);
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) return view(cast.getSource());
    auto operation = memory.getDefiningOp<memref::SubViewOp>();
    if (!operation) return failure();
    auto base = view(operation.getSource());
    if (failed(base)) return failure();
    SmallVector<int64_t> offsets, extents, dimensions;
    SmallVector<Value> dynamic;
    auto append = [&](ArrayRef<OpFoldResult> bounds, SmallVectorImpl<int64_t> &statics) {
      for (OpFoldResult value : bounds) {
        if (auto attr = dyn_cast<Attribute>(value)) statics.push_back(cast<IntegerAttr>(attr).getInt());
        else {
          llvm::APInt integer;
          if (matchPattern(cast<Value>(value), m_ConstantInt(&integer))) statics.push_back(integer.getSExtValue());
          else { statics.push_back(-1); dynamic.push_back(values.lookup(cast<Value>(value))); }
        }
      }
    };
    for (OpFoldResult stride : operation.getMixedStrides())
      if (!sameBound(stride, b.getIndexAttr(1))) return operation.emitError("Weft subview requires unit coordinate steps"), failure();
    append(operation.getMixedOffsets(), offsets);
    append(operation.getMixedSizes(), extents);
    auto ids = axes((*base).getType());
    for (auto [extent, axis] : llvm::zip(extents, ids)) dimensions.push_back(extent < 0 ? -axis : extent);
    Value selected = b.create<wk::SubviewOp>(operation.getLoc(),
        wk::SliceType::get(b.getContext(), encoding(), array(dimensions), array(ids)),
        *base, dynamic, array(offsets), array(extents));
    auto dropped = operation.getDroppedDims();
    if (dropped.any()) {
      SmallVector<Attribute> selectors;
      SmallVector<Value> indices;
      SmallVector<int64_t> keptShape, keptAxes;
      for (unsigned axis = 0; axis < ids.size(); ++axis) {
        if (dropped.test(axis)) {
          selectors.push_back(b.getStringAttr("index"));
          indices.push_back(index(operation.getLoc(), 0));
        } else {
          selectors.push_back(b.getStringAttr("all"));
          keptShape.push_back(dimensions[axis]); keptAxes.push_back(ids[axis]);
        }
      }
      selected = b.create<wk::SliceOp>(operation.getLoc(),
          wk::SliceType::get(b.getContext(), encoding(), array(keptShape), array(keptAxes)),
          selected, indices, b.getArrayAttr(selectors));
    }
    values.map(memory, selected);
    return selected;
  }

  FailureOr<Value> read(Value memory) {
    if (!isa<MemRefType>(memory.getType())) return values.lookup(memory);
    if (!isLocal(memory)) {
      auto previous = operandReads.find(memory);
      if (previous != operandReads.end()) return previous->second;
      auto region = view(memory);
      if (failed(region)) return failure();
      Value loaded = b.create<wk::AdmitOp>(memory.getLoc(),
          valueType(b.getF32Type(), shape((*region).getType()), axes((*region).getType())), *region);
      operandReads[memory] = loaded;
      return loaded;
    }
    Value root = localRoot(memory);
    auto found = locals.find(root);
    if (found == locals.end()) {
      emitError(memory.getLoc(), "Weft local value is read before a dominating complete supply");
      return failure();
    }
    if (memory == root) {
      auto type = cast<MemRefType>(root.getType());
      unsigned dynamic = 0;
      for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
        OpFoldResult full = ShapedType::isDynamic(extent)
            ? OpFoldResult(root.getDefiningOp()->getOperand(dynamic++))
            : OpFoldResult(b.getIndexAttr(extent));
        if (!sameBound(found->second.sizes[axis], full))
          return emitError(memory.getLoc(), "Weft local root read requires a complete initialized region"), failure();
      }
      return found->second.value;
    }
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) return read(cast.getSource());
    auto projection = memory.getDefiningOp<memref::SubViewOp>();
    if (!projection || projection.getSource() != root || projection.getDroppedDims().any()) {
      emitError(memory.getLoc(), "Weft local-value projection requires one explicit rank-preserving subview");
      return failure();
    }
    auto offsets = projection.getMixedOffsets(), sizes = projection.getMixedSizes();
    for (auto [axis, offset, size, stride] : llvm::zip(llvm::seq<unsigned>(0, sizes.size()),
             offsets, sizes, projection.getMixedStrides())) {
      if (!sameBound(stride, b.getIndexAttr(1)) || !sameBound(offset, b.getIndexAttr(0)) ||
          !sameBound(size, found->second.sizes[axis]))
        return projection.emitError("Weft private value must consume its explicit complete active panel; other local windows are not implemented"), failure();
    }
    return found->second.value;
  }

  LogicalResult write(Value memory, Value value, SmallVector<OpFoldResult> sizes = {}) {
    if (!isLocal(memory)) {
      auto region = view(memory);
      if (failed(region)) return failure();
      b.create<wk::CommitOp>(memory.getLoc(), value, *region);
      return success();
    }
    Value root = localRoot(memory);
    if (memory != root) {
      auto projection = memory.getDefiningOp<memref::SubViewOp>();
      if (!projection || projection.getSource() != root || projection.getDroppedDims().any() ||
          llvm::any_of(projection.getMixedOffsets(), [&](OpFoldResult offset) { return !sameBound(offset, b.getIndexAttr(0)); }))
        return emitError(memory.getLoc(), "Weft private supply must define one zero-based complete active region");
      sizes = projection.getMixedSizes();
    }
    if (sizes.empty()) {
      auto type = cast<MemRefType>(root.getType());
      Operation *allocation = root.getDefiningOp();
      unsigned dynamic = 0;
      for (int64_t extent : type.getShape()) {
        if (ShapedType::isDynamic(extent)) sizes.push_back(allocation->getOperand(dynamic++));
        else sizes.push_back(b.getIndexAttr(extent));
      }
    }
    locals[root] = {value, sizes};
    return success();
  }

  FailureOr<Type> pointwiseType(Value lhs, Value rhs) {
    if (element(lhs.getType()) != element(rhs.getType())) return failure();
    auto dimensions = shape(lhs.getType()), ids = axes(lhs.getType());
    auto rightAxes = axes(rhs.getType()), rightShape = shape(rhs.getType());
    for (auto [axis, extent] : llvm::zip(rightAxes, rightShape)) {
      auto it = llvm::find(ids, axis);
      if (it == ids.end()) { ids.push_back(axis); dimensions.push_back(extent); }
      else {
        auto &left = dimensions[it - ids.begin()];
        if (left == 1) left = extent;
        else if (extent != 1 && extent != left) return failure();
      }
    }
    return valueType(element(lhs.getType()), dimensions, ids);
  }

  FailureOr<Value> binary(Location loc, Value lhs, Value rhs, StringRef kind) {
    auto type = pointwiseType(lhs, rhs);
    if (failed(type)) { emitError(loc, "Weft pointwise operands have incompatible axis/extent relations"); return failure(); }
    return Value(b.create<wk::BinaryOp>(loc, *type, lhs, rhs, kind));
  }

  FailureOr<Value> expression(Operation *operation, IRMapping &mapping) {
    Location loc = operation->getLoc();
    if (auto constant = dyn_cast<arith::ConstantOp>(operation))
      return b.clone(*constant, mapping)->getResult(0);
    SmallVector<Value> operands;
    for (Value operand : operation->getOperands()) operands.push_back(mapping.lookup(operand));
    StringRef kind;
    if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
      if (compare.getPredicate() != arith::CmpIPredicate::eq)
        return compare.emitError("this CPU integer comparison has no implemented Weft mapping"), failure();
      return Value(b.create<wk::CompareOp>(loc, b.getI1Type(), operands[0], operands[1], "eq"));
    }
    if (isa<arith::AddFOp, arith::AddIOp>(operation)) kind = "add";
    else if (isa<arith::SubFOp, arith::SubIOp>(operation)) kind = "sub";
    else if (isa<arith::MulFOp, arith::MulIOp>(operation)) kind = "mul";
    else if (isa<arith::DivFOp, arith::DivSIOp>(operation)) kind = "div";
    else if (isa<arith::RemSIOp>(operation)) kind = "mod";
    if (!kind.empty()) return binary(loc, operands[0], operands[1], kind);
    if (isa<arith::MinSIOp, arith::MaxSIOp>(operation))
      return binary(loc, operands[0], operands[1], isa<arith::MinSIOp>(operation) ? "min" : "max");
    if (auto divide = dyn_cast<arith::CeilDivSIOp>(operation)) {
      auto minusOne = binary(loc, operands[1], index(loc, 1), "sub");
      if (failed(minusOne)) return failure();
      auto sum = binary(loc, operands[0], *minusOne, "add");
      if (failed(sum)) return failure();
      return binary(loc, *sum, operands[1], "div");
    }
    if (isa<arith::IndexCastOp>(operation))
      return Value(b.create<wk::CastOp>(loc, operation->getResult(0).getType(), operands[0]));
    if (isa<math::RsqrtOp>(operation)) kind = "rsqrt";
    else if (isa<arith::NegFOp>(operation)) kind = "neg";
    if (!kind.empty()) return Value(b.create<wk::UnaryOp>(loc, operands[0].getType(), operands[0], kind));
    operation->emitError("CPU operation has no Weft numerical representation");
    return failure();
  }

  FailureOr<Value> mappedInput(Value input, AffineMap map) {
    auto loaded = read(input);
    if (failed(loaded)) return failure();
    if (!isa<MemRefType>(input.getType())) return *loaded;
    auto dimensions = shape((*loaded).getType()), ids = axes((*loaded).getType());
    SmallVector<int64_t> keptShape, keptAxes;
    SmallVector<Attribute> selectors;
    SmallVector<Value> indices;
    for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
      if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
        if (constant.getValue() != 0 || dimensions[axis] != 1) return failure();
        selectors.push_back(b.getStringAttr("index")); indices.push_back(index(input.getLoc(), 0));
      } else {
        selectors.push_back(b.getStringAttr("all")); keptShape.push_back(dimensions[axis]); keptAxes.push_back(ids[axis]);
      }
    }
    if (indices.empty()) return *loaded;
    return Value(b.create<wk::ExtractOp>(input.getLoc(), valueType(b.getF32Type(), keptShape, keptAxes),
        *loaded, indices, b.getArrayAttr(selectors)));
  }

  LogicalResult generic(linalg::GenericOp operation) {
    if (operation.getOutputs().size() != 1 || operation.getNumResults())
      return operation.emitError("Weft CPU legalization requires one buffer-semantics result");
    Value destination = operation.getOutputs()[0];
    if (cpu::isMatrixContraction(operation)) {
      auto lhs = read(operation.getInputs()[0]), rhs = read(operation.getInputs()[1]), initial = read(destination);
      if (failed(lhs) || failed(rhs) || failed(initial)) return failure();
      auto definition = (*initial).getDefiningOp<wk::NewOp>();
      if (!definition || !matchPattern(definition->getOperand(0), m_PosZeroFloat()))
        return operation.emitError("Weft contraction requires an explicit zero-initialized partial; nonzero fused accumulation has no equivalent canonical operation");
      int64_t reduction = axes((*lhs).getType())[1];
      Value term = b.create<wk::OuterContractOp>(operation.getLoc(), (*initial).getType(),
          *lhs, *rhs, array({reduction}), TypeAttr::get(b.getF32Type()));
      return write(destination, term);
    }
    if (llvm::any_of(operation.getIteratorTypesArray(), [](utils::IteratorType type) {
          return type != utils::IteratorType::parallel;
        }) || !operation.getIndexingMapsArray().back().isIdentity())
      return operation.emitError("Weft pointwise legalization requires an identity output traversal");
    IRMapping mapping;
    Block &body = operation.getRegion().front();
    for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
      auto value = mappedInput(input, operation.getIndexingMapsArray()[number]);
      if (failed(value)) return failure();
      mapping.map(body.getArgument(number), *value);
    }
    if (!body.getArguments().back().use_empty()) return operation.emitError("Weft pointwise output is not a pure definition");
    for (Operation &nested : body.without_terminator()) {
      auto value = expression(&nested, mapping);
      if (failed(value)) return failure();
      mapping.map(nested.getResult(0), *value);
    }
    return write(destination, mapping.lookup(body.getTerminator()->getOperand(0)));
  }

  LogicalResult reduction(cpu::ReduceOp operation) {
    Block &body = operation.getCombine().front();
    auto add = body.getTerminator()->getOperand(0).getDefiningOp<arith::AddFOp>();
    auto initial = operation.getInitial().getDefiningOp<arith::ConstantOp>();
    auto identity = initial ? dyn_cast<FloatAttr>(initial.getValue()) : FloatAttr();
    if (!operation.getOrder().getAdjacentReassociation() || !add || !identity || !identity.getValue().isZero())
      return operation.emitError("Weft reduction requires the closed additive identity/combine schema");
    IRMapping mapping;
    for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
      auto value = mappedInput(input, cast<AffineMapAttr>(operation.getIndexingMaps()[number]).getValue());
      if (failed(value)) return failure();
      mapping.map(body.getArgument(number + 1), *value);
    }
    for (Operation &nested : body.without_terminator()) {
      if (&nested == add.getOperation()) continue;
      auto value = expression(&nested, mapping);
      if (failed(value)) return failure();
      mapping.map(nested.getResult(0), *value);
    }
    Value contribution = mapping.lookup(add.getLhs() == body.getArgument(0) ? add.getRhs() : add.getLhs());
    if (shape(contribution.getType()).size() != 1)
      return operation.emitError("Weft reduction requires one retained logical input axis");
    Value result = b.create<wk::ReduceOp>(operation.getLoc(), operation.getResult().getType(), contribution, "add", 0);
    values.map(operation.getResult(), result);
    return success();
  }

  LogicalResult block(Block &source) {
    for (Operation &operation : source.without_terminator())
      if (failed(lower(&operation))) return failure();
    return success();
  }

  LogicalResult lower(Operation *operation) {
    operandReads.clear();
    Location loc = operation->getLoc();
    if (isa<memref::AllocOp, memref::AllocaOp, memref::SubViewOp, memref::CastOp>(operation)) return success();
    if (auto dealloc = dyn_cast<memref::DeallocOp>(operation)) { locals.erase(localRoot(dealloc.getMemref())); return success(); }
    if (auto dimension = dyn_cast<memref::DimOp>(operation)) {
      auto axis = dimension.getConstantIndex();
      if (!axis || isLocal(dimension.getSource())) return dimension.emitError("Weft dimension requires an explicit external View axis");
      auto region = view(dimension.getSource());
      if (failed(region)) return failure();
      values.map(dimension.getResult(), b.create<wk::ExtentOp>(loc, b.getIndexType(), *region, *axis));
      return success();
    }
    if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
      auto value = read(copy.getSource());
      if (failed(value)) return failure();
      if (isLocal(copy.getTarget())) {
        Value root = localRoot(copy.getTarget());
        auto scope = copy->getParentOfType<scf::ForOp>();
        for (Operation *user : root.getUsers())
          if (!isa<memref::DeallocOp>(user) && scope && !scope->isProperAncestor(user))
            return copy.emitError("Weft private packing supply escapes its overwrite scope");
        *value = b.create<wk::MaterializeOp>(loc, (*value).getType(), *value);
      }
      return write(copy.getTarget(), *value);
    }
    if (auto fill = dyn_cast<linalg::FillOp>(operation)) {
      Value destination = fill.getOutputs()[0];
      Value initial = values.lookup(fill.getInputs()[0]);
      if (isLocal(destination)) {
        if (!cast<MemRefType>(destination.getType()).hasStaticShape())
          return fill.emitError("Weft private fill requires explicit runtime shape storage, which is not implemented");
        Value result = b.create<wk::NewOp>(loc, resultType(destination), initial, true);
        return write(destination, result);
      }
      auto region = view(destination);
      if (failed(region)) return failure();
      auto dimensions = shape((*region).getType());
      SmallVector<Value> indices;
      std::function<void(unsigned)> fillAxis = [&](unsigned axis) {
        if (axis == dimensions.size()) {
          SmallVector<Attribute> selectors(dimensions.size(), b.getStringAttr("index"));
          Value selected = b.create<wk::SliceOp>(loc,
              wk::SliceType::get(b.getContext(), encoding(), array({}), array({})),
              *region, indices, b.getArrayAttr(selectors));
          b.create<wk::CommitOp>(loc, initial, selected);
          return;
        }
        Value end = b.create<wk::ExtentOp>(loc, b.getIndexType(), *region, axis);
        auto loop = b.create<scf::ForOp>(loc, index(loc, 0), end, index(loc, 1));
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(loop.getBody());
        indices.push_back(loop.getInductionVar());
        fillAxis(axis + 1);
        indices.pop_back();
      };
      fillAxis(0);
      return success();
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(operation)) return generic(genericOp);
    if (auto reduce = dyn_cast<cpu::ReduceOp>(operation)) return reduction(reduce);
    if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
      if (conditional.getNumResults()) return conditional.emitError("Weft conditional SSA results are not implemented");
      for (auto access : analysis.accesses(conditional))
        if (access.write && isLocal(access.memory) &&
            !conditional->isProperAncestor(localRoot(access.memory).getDefiningOp()))
          return conditional.emitError("Weft conditional mutation of an enclosing private value is not implemented");
      auto target = b.create<scf::IfOp>(loc, values.lookup(conditional.getCondition()),
          !conditional.getElseRegion().empty());
      auto savedLocals = locals;
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(target.thenBlock());
      if (failed(block(*conditional.thenBlock()))) return failure();
      locals = savedLocals;
      if (!conditional.getElseRegion().empty()) {
        b.setInsertionPointToStart(target.elseBlock());
        if (failed(block(*conditional.elseBlock()))) return failure();
        locals = savedLocals;
      }
      return success();
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      SmallVector<Value> initial;
      for (Value value : loop.getInitArgs()) initial.push_back(values.lookup(value));
      auto target = b.create<scf::ForOp>(loc, values.lookup(loop.getLowerBound()),
          values.lookup(loop.getUpperBound()), values.lookup(loop.getStep()), initial);
      auto savedLocals = locals;
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(target.getBody());
        values.map(loop.getInductionVar(), target.getInductionVar());
        for (auto [source, destination] : llvm::zip(loop.getRegionIterArgs(), target.getRegionIterArgs())) values.map(source, destination);
        if (failed(block(*loop.getBody()))) return failure();
        SmallVector<Value> results;
        for (Value value : loop.getBody()->getTerminator()->getOperands()) results.push_back(values.lookup(value));
        if (!initial.empty()) b.create<scf::YieldOp>(loc, results);
      }
      locals = std::move(savedLocals);
      for (auto [source, destination] : llvm::zip(loop.getResults(), target.getResults())) values.map(source, destination);
      return success();
    }
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      auto region = view(load.getMemref());
      if (failed(region)) return failure();
      SmallVector<Value> indices;
      SmallVector<Attribute> selectors(load.getIndices().size(), b.getStringAttr("index"));
      for (Value value : load.getIndices()) indices.push_back(values.lookup(value));
      Value selected = b.create<wk::SliceOp>(loc, wk::SliceType::get(b.getContext(), encoding(), array({}), array({})),
          *region, indices, b.getArrayAttr(selectors));
      values.map(load.getResult(), b.create<wk::AdmitOp>(loc, load.getType(), selected));
      return success();
    }
    if (operation->getNumResults() != 1) return operation->emitError("CPU task operation has no Weft representation");
    auto value = expression(operation, values);
    if (failed(value)) return failure();
    values.map(operation->getResult(0), *value);
    return success();
  }

  cpu::PhysicalProgramAnalysis analysis;
  cpu::AxisRelations relations;
  ModuleOp output;
  OpBuilder b;
  IRMapping values;
  llvm::DenseMap<Value, LocalValue> locals;
  llvm::DenseMap<Value, Value> operandReads;
};

}

FailureOr<OwningOpRef<ModuleOp>> legalizeProgram(ModuleOp cpuProgram, std::string &metadata) {
  if (failed(cpu::verifyCPUProgram(cpuProgram, false))) return failure();
  cpuProgram.getContext()->loadDialect<wk::WEFTKernelDialect>();
  OwningOpRef<ModuleOp> output = ModuleOp::create(cpuProgram.getLoc());
  llvm::json::Array interfaces;
  for (func::FuncOp function : cpuProgram.getOps<func::FuncOp>()) {
    TaskLowering lowering(function, *output);
    SmallVector<cpu::TasksOp> tasks;
    function.walk([&](cpu::TasksOp operation) { tasks.push_back(operation); });
    if (tasks.empty()) return function.emitError("Weft generation requires an explicit CPU task interface"), failure();
    for (auto [ordinal, task] : llvm::enumerate(tasks)) {
      std::string name = function.getName().str() + "_task_" + std::to_string(ordinal);
      if (failed(lowering.lower(task, name))) return failure();
      interfaces.push_back(llvm::json::Object{{"cpu_entry", function.getName().str()},
          {"task_ordinal", static_cast<int64_t>(ordinal)}, {"weft_entry", name},
          {"coordinate_argument", 0}, {"capture_count", static_cast<int64_t>(task.getCaptures().size())}});
    }
  }
  if (failed(verify(*output))) return failure();
  llvm::raw_string_ostream stream(metadata);
  stream << llvm::json::Value(llvm::json::Object{{"kind", "weft-generation"},
      {"native", false}, {"tasks", std::move(interfaces)}});
  return std::move(output);
}

}
