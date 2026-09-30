#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace intent::dsa {
namespace {
std::optional<int64_t> integer(Value value) {
  APInt bits;
  if (matchPattern(value, m_ConstantInt(&bits)) && bits.isSignedIntN(64)) return bits.getSExtValue();
  return std::nullopt;
}

// The relation is relative to four consecutive iterations of the current task
// loop: value(group, lane) = value(group, 0) + lane * coefficient. Local memory
// is followed through its actual writers, not through source-language origins.
class GroupRelations {
public:
  explicit GroupRelations(scf::ForOp work)
      : work(work), function(work->getParentOfType<func::FuncOp>()),
        interface(function->getAttrOfType<InterfaceAttr>("intent_dsa.interface")) {}

  bool taskIdentity(Value value) const {
    auto loop = work;
    while (auto divide = value.getDefiningOp<arith::DivSIOp>()) {
      if (!matchPattern(divide.getRhs(), m_One())) break;
      value = divide.getLhs();
    }
    return value == loop.getInductionVar();
  }

  ViewArgumentAttr readonlyView(Value value) const {
    auto owner = function;
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &owner.front()) return {};
    auto view = dyn_cast<ViewArgumentAttr>(interface.getArguments()[argument.getArgNumber()]);
    return view && view.getAccess() == 0 ? view : ViewArgumentAttr();
  }

  // These are the control expressions whose group-uniform form is retained by
  // quotient rewriting below and checked again by the DSA program verifier.
  bool uniformControl(Value value) {
    if (auto known = control.find(value); known != control.end()) return known->second;
    if (!activeControl.insert(value).second) return false;
    auto infer = [&]() {
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        if (argument.getOwner() == &function.front()) return true;
        auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
        return loop && loop != work && argument == loop.getInductionVar() &&
            uniformControl(loop.getLowerBound()) && uniformControl(loop.getUpperBound()) && uniformControl(loop.getStep());
      }
      Operation *definition = value.getDefiningOp();
      if (!definition) return false;
      if (isa<arith::ConstantOp>(definition)) return true;
      if (auto divide = dyn_cast<arith::DivSIOp>(definition)) {
        auto divisor = integer(divide.getRhs());
        if (taskIdentity(divide.getLhs()) && divisor && *divisor > 0 && *divisor % 4 == 0) return true;
      }
      if (auto load = dyn_cast<LoadScalarOp>(definition))
        return bool(readonlyView(load.getSource())) && uniformControl(load.getOffset());
      return (isa<memref::DimOp, StrideOp>(definition) || isa<arith::ArithDialect>(definition->getDialect())) &&
          llvm::all_of(definition->getOperands(), [&](Value input) { return uniformControl(input); });
    };
    bool result = infer();
    activeControl.erase(value); control[value] = result;
    return result;
  }

  bool uniformExecution(Operation *operation) {
    for (Operation *parent = operation->getParentOp(); parent && parent != function; parent = parent->getParentOp()) {
      if (parent == work) return true;
      if (auto loop = dyn_cast<scf::ForOp>(parent)) {
        if (!uniformControl(loop.getLowerBound()) || !uniformControl(loop.getUpperBound()) ||
            !uniformControl(loop.getStep())) return false;
      } else if (auto branch = dyn_cast<scf::IfOp>(parent)) {
        if (!uniformControl(branch.getCondition())) return false;
      } else return false;
    }
    return true;
  }

  std::optional<int64_t> laneCoefficient(Value value) {
    if (auto known = coefficients.find(value); known != coefficients.end()) return known->second;
    if (!activeValues.insert(value).second) return std::nullopt;
    auto infer = [&]() -> std::optional<int64_t> {
      if (uniformControl(value)) return 0;
      if (taskIdentity(value)) return 1;
      Operation *definition = value.getDefiningOp();
      if (!definition) return std::nullopt;
      if (auto remainder = dyn_cast<arith::RemSIOp>(definition)) {
        auto divisor = integer(remainder.getRhs());
        if (taskIdentity(remainder.getLhs()) && divisor && *divisor > 0 && *divisor % 4 == 0) return 1;
      }
      if (auto load = dyn_cast<memref::LoadOp>(definition))
        return uniformBuffer(load.getMemref()) && llvm::all_of(load.getIndices(), [&](Value index) { return uniformValue(index); })
            ? std::optional<int64_t>(0) : std::nullopt;
      if (auto load = dyn_cast<LoadScalarOp>(definition))
        return readonlyView(load.getSource()) && uniformValue(load.getOffset()) ? std::optional<int64_t>(0) : std::nullopt;
      if (isa<arith::IndexCastOp, arith::ExtSIOp>(definition)) {
        Type from = definition->getOperand(0).getType(), to = value.getType();
        unsigned fromWidth = from.isIndex() ? 64 : cast<IntegerType>(from).getWidth();
        unsigned toWidth = to.isIndex() ? 64 : cast<IntegerType>(to).getWidth();
        if (toWidth >= fromWidth) return laneCoefficient(definition->getOperand(0));
      }
      if (!isa<arith::ArithDialect>(definition->getDialect()) || definition->getNumRegions()) return std::nullopt;
      if (llvm::all_of(definition->getOperands(), [&](Value input) { return uniformValue(input); })) return 0;
      if (definition->getNumOperands() != 2) return std::nullopt;
      auto lhs = laneCoefficient(definition->getOperand(0)), rhs = laneCoefficient(definition->getOperand(1));
      if (!lhs || !rhs) return std::nullopt;
      APInt coefficient(128, *lhs, true);
      if (isa<arith::AddIOp>(definition)) coefficient += APInt(128, *rhs, true);
      else if (isa<arith::SubIOp>(definition)) coefficient -= APInt(128, *rhs, true);
      else if (isa<arith::MulIOp>(definition)) {
        if (auto scale = integer(definition->getOperand(0))) coefficient = APInt(128, *rhs, true) * APInt(128, *scale, true);
        else if (auto scale = integer(definition->getOperand(1))) coefficient *= APInt(128, *scale, true);
        else return std::nullopt;
      } else return std::nullopt;
      return coefficient.isSignedIntN(64) ? std::optional<int64_t>(coefficient.getSExtValue()) : std::nullopt;
    };
    auto result = infer();
    // Nonzero address differences are only propagated in the DSA address
    // domain. Extending a wrapped narrow integer would not preserve them.
    if (result && *result != 0 && !value.getType().isIndex() && !value.getType().isInteger(64)) result.reset();
    activeValues.erase(value); coefficients[value] = result;
    return result;
  }

private:
  bool uniformValue(Value value) {
    auto coefficient = laneCoefficient(value);
    return coefficient && *coefficient == 0;
  }

  bool uniformBuffer(Value value) {
    if (readonlyView(value)) return true;
    value = storageRoot(value);
    if (auto known = buffers.find(value); known != buffers.end()) return known->second;
    if (!value.getDefiningOp<memref::AllocaOp>() || !activeBuffers.insert(value).second) return false;
    auto infer = [&]() {
      auto aliases = storageAliases(value);
      for (Value alias : aliases)
        if (auto view = alias.getDefiningOp<memref::ReinterpretCastOp>())
          for (Value parameter : view->getOperands().drop_front())
            if (!uniformValue(parameter)) return false;
      bool written = false;
      for (Value alias : aliases) for (Operation *user : alias.getUsers()) {
        if (isa<memref::ReinterpretCastOp>(user)) continue;
        if (llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); })) return false;
        auto effects = dyn_cast<MemoryEffectOpInterface>(user);
        if (!effects) return false;
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        bool writes = false;
        for (const auto &effect : instances) {
          if (isa<MemoryEffects::Free>(effect.getEffect())) return false;
          if (!isa<MemoryEffects::Write>(effect.getEffect())) continue;
          if (!effect.getValue()) return false;
          writes |= llvm::is_contained(aliases, effect.getValue());
        }
        if (!writes) continue;
        written = true;
        if (!isa<memref::StoreOp, FillOp, IotaOp, LoadTileOp, GatherRowsOp, UnaryOp, BinaryOp,
                 CastOp, SelectOp, CompareOp, CompareRangeOp, CompareRampOp, IndexBinaryOp,
                 IndexLayoutOp, TransposeOp>(user) || !uniformExecution(user)) return false;
        for (Value operand : user->getOperands()) {
          if (!isa<MemRefType>(operand.getType())) {
            if (!uniformValue(operand)) return false;
            continue;
          }
          bool reads = false, writesOperand = false;
          for (const auto &effect : instances) if (effect.getValue() == operand) {
            reads |= isa<MemoryEffects::Read>(effect.getEffect());
            writesOperand |= isa<MemoryEffects::Write>(effect.getEffect());
          }
          if ((!writesOperand || reads) && !uniformBuffer(operand)) return false;
        }
      }
      return written;
    };
    bool result = infer(); activeBuffers.erase(value); buffers[value] = result;
    return result;
  }

  scf::ForOp work;
  func::FuncOp function;
  InterfaceAttr interface;
  DenseMap<Value, bool> control, buffers;
  DenseMap<Value, std::optional<int64_t>> coefficients;
  DenseSet<Value> activeControl, activeBuffers, activeValues;
};

bool fourAligned(Value value) {
  if (auto constant = integer(value)) return *constant % 4 == 0;
  if (auto multiply = value.getDefiningOp<arith::MulIOp>())
    return fourAligned(multiply.getLhs()) || fourAligned(multiply.getRhs());
  return false;
}

bool adjacentIntervals(GatherRowsOp gather, GroupRelations &relations) {
  if (gather.getAsynchronous() || gather.getPlan() || !relations.uniformExecution(gather) ||
      !relations.uniformControl(gather.getRows()) || !relations.uniformControl(gather.getColumns())) return false;
  auto source = cast<MemRefType>(gather.getSource().getType());
  auto output = cast<MemRefType>(gather.getOutput().getType());
  auto view = relations.readonlyView(gather.getSource());
  if (!view || source.getRank() < 2 || !view.getConstraints().getHasStrides() ||
      !output.getLayout().isIdentity() || output.getMemorySpaceAsInt() != nramSpace) return false;
  int64_t rows = output.getDimSize(0), columns = output.getDimSize(1);
  unsigned headAxis = source.getRank() - 2, columnAxis = source.getRank() - 1;
  auto headStride = dyn_cast<IntegerAttr>(view.getConstraints().getStrides()[headAxis]);
  auto columnStride = dyn_cast<IntegerAttr>(view.getConstraints().getStrides()[columnAxis]);
  if (rows < 64 || rows % 64 || rows > 65536 || columns <= 0 || source.getDimSize(columnAxis) != columns ||
      source.getDimSize(headAxis) <= 0 || source.getDimSize(headAxis) % 4 || !headStride || !columnStride ||
      headStride.getInt() != columns || columnStride.getInt() != 1 ||
      integer(gather.getColumnStride()) != std::optional<int64_t>(1) ||
      integer(gather.getColumns()) != std::optional<int64_t>(columns)) return false;
  int64_t elementBytes = llvm::divideCeil(source.getElementType().getIntOrFloatBitWidth(), 8u);
  if (columns > ((3968 * 1024) / rows - 32) / (4 * elementBytes)) return false;

  // Require one complete, current offset snapshot immediately before the read.
  // No other writer or alias can substitute older lane-dependent row addresses.
  Value offsets = gather.getRowOffsets();
  auto allocation = offsets.getDefiningOp<memref::AllocaOp>();
  auto loop = dyn_cast_or_null<scf::ForOp>(gather->getPrevNode());
  if (!allocation || !loop || !loop.getInitArgs().empty() || !matchPattern(loop.getLowerBound(), m_Zero()) ||
      !matchPattern(loop.getStep(), m_One()) || loop.getUpperBound() != gather.getRows() ||
      !relations.uniformExecution(loop)) return false;
  memref::StoreOp store;
  for (Operation *user : offsets.getUsers()) {
    if (auto write = dyn_cast<memref::StoreOp>(user)) {
      if (store || write->getBlock() != loop.getBody() || write.getIndices().size() != 2 ||
          !matchPattern(write.getIndices()[0], m_Zero()) || write.getIndices()[1] != loop.getInductionVar()) return false;
      store = write;
    } else if (user != gather.getOperation()) return false;
  }
  if (!store || !storageRoot(gather.getOutput()).getDefiningOp<memref::AllocaOp>() ||
      storageRoot(gather.getOutput()) == offsets) return false;
  auto coefficient = relations.laneCoefficient(store.getValue());
  return coefficient && *coefficient == columns;
}

void realizeGroup(scf::ForOp work, ArrayRef<GatherRowsOp> gathers, GroupRelations &relations) {
  auto function = work->getParentOfType<func::FuncOp>();
  OpBuilder b(work);
  Location loc = work.getLoc();
  auto index = [&](int64_t value) -> Value { return b.create<arith::ConstantIndexOp>(loc, value); };
  auto allocate = [&](Type element, int64_t rows, int64_t columns, int64_t space = nramSpace) -> Value {
    auto type = MemRefType::get({rows, columns}, element, MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(space));
    auto allocation = b.create<memref::AllocaOp>(loc, type); allocation.setAlignment(128); return allocation;
  };
  Value originalTask = work.getInductionVar();
  Value group = b.create<GroupIdOp>(loc, b.getIndexType()), groups = b.create<GroupCountOp>(loc, b.getIndexType());
  Value local = b.create<LocalIdOp>(loc, b.getIndexType()), memory = b.create<IsMemoryCoreOp>(loc, b.getI1Type());
  Value lane = b.create<arith::SelectOp>(loc, memory, index(0), local);
  Value total = b.create<arith::DivSIOp>(loc, work.getUpperBound(), index(4));
  SmallVector<arith::DivSIOp> quotients;
  work.walk([&](arith::DivSIOp divide) {
    auto divisor = integer(divide.getRhs());
    if (relations.taskIdentity(divide.getLhs()) && divisor && *divisor > 0 && *divisor % 4 == 0) quotients.push_back(divide);
  });
  work.setLowerBound(group); work.setUpperBound(total); work.setStep(groups);
  DenseSet<Operation *> groupOperations;
  for (auto quotient : quotients) {
    b.setInsertionPoint(quotient);
    auto grouped = b.create<arith::DivSIOp>(loc, originalTask, index(*integer(quotient.getRhs()) / 4));
    groupOperations.insert(grouped);
    quotient.replaceAllUsesWith(grouped.getResult()); quotient.erase();
  }
  b.setInsertionPointToStart(work.getBody());
  Value first = b.create<arith::MulIOp>(loc, originalTask, index(4));
  groupOperations.insert(first.getDefiningOp());
  Value task = b.create<arith::AddIOp>(loc, first, lane);
  originalTask.replaceUsesWithIf(task, [&](OpOperand &use) { return !groupOperations.contains(use.getOwner()); });
  function->setAttr("intent_dsa.group_width", b.getI64IntegerAttr(4));
  DenseMap<int64_t, Value> indices;
  for (auto gather : gathers) {
    auto type = cast<MemRefType>(gather.getOutput().getType());
    int64_t rows = type.getDimSize(0), columns = type.getDimSize(1);
    Value ramp = indices.lookup(rows);
    if (!ramp) {
      b.setInsertionPointToStart(&function.front());
      ramp = allocate(b.getF32Type(), 1, rows); b.create<IotaOp>(loc, ramp); indices[rows] = ramp;
    }
    b.setInsertionPoint(gather);
    Value descriptor = allocate(b.getI64Type(), 3, rows);
    b.create<GatherPlanOp>(loc, gather.getRowOffsets(), gather.getRows(), ramp, descriptor);
    Value data = allocate(type.getElementType(), rows, 4 * columns, sharedSpace);
    Value metadata = allocate(b.getI64Type(), 4, rows, sharedSpace);
    b.create<GroupGatherRowsOp>(loc, gather.getSource(), gather.getRowOffsets(), descriptor,
        gather.getOutput(), data, metadata, gather.getRows(), lane);
    gather.erase();
  }
}
} // namespace

LogicalResult realizeCollectiveGatherSupply(func::FuncOp function) {
  auto config = function->getAttrOfType<ConfigurationAttr>("intent_dsa.configuration");
  if (function->hasAttr("intent_dsa.group_width") || config.getTasks() < 4 || config.getTasks() % 4) return success();
  SmallVector<scf::ForOp> worksets;
  function.walk([&](scf::ForOp loop) {
    if (loop.getLowerBound().getDefiningOp<TaskIdOp>()) worksets.push_back(loop);
  });
  if (worksets.size() != 1) return success();
  auto work = worksets.front();
  if (work->getParentOp() != function || !work.getStep().getDefiningOp<TaskCountOp>() ||
      !work.getInitArgs().empty() || !fourAligned(work.getUpperBound()) ||
      !work.getLowerBound().hasOneUse() || !work.getStep().hasOneUse()) return success();
  bool isolated = true;
  function.walk([&](Operation *operation) {
    if (!isa<TaskIdOp, TaskCountOp>(operation)) return;
    Value query = operation->getResult(0);
    if (query != work.getLowerBound() && query != work.getStep() && !query.use_empty()) isolated = false;
  });
  if (!isolated) return success();
  GroupRelations relations(work);
  SmallVector<GatherRowsOp> gathers;
  work.walk([&](GatherRowsOp gather) {
    if (adjacentIntervals(gather, relations)) gathers.push_back(gather);
  });
  if (!gathers.empty()) realizeGroup(work, gathers, relations);
  return success();
}
} // namespace intent::dsa
