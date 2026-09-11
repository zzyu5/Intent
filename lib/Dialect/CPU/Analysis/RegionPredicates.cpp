#include "Intent/Dialect/CPU/Analysis/RegionPredicates.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
namespace intent::cpu {
namespace {

Value stripCast(Value value) {
  while (auto cast = value.getDefiningOp<memref::CastOp>()) value = cast.getSource();
  return value;
}
bool nonnegative(Value value) {
  if (auto constant = getConstantIntValue(value)) return *constant >= 0;
  if (value.getDefiningOp<memref::DimOp>()) return true;
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return (cast.getIn().getType().isIndex() || cast.getIn().getType().isInteger(64)) &&
        (cast.getType().isIndex() || cast.getType().isInteger(64)) && nonnegative(cast.getIn());
  if (auto minimum = value.getDefiningOp<arith::MinSIOp>()) return nonnegative(minimum.getLhs()) && nonnegative(minimum.getRhs());
  if (auto maximum = value.getDefiningOp<arith::MaxSIOp>()) return nonnegative(maximum.getLhs()) || nonnegative(maximum.getRhs());
  auto argument = dyn_cast<BlockArgument>(value);
  auto loop = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp();
  return loop && argument == loop.getInductionVar() &&
      getConstantIntValue(loop.getStep()).value_or(0) > 0 && nonnegative(loop.getLowerBound());
}

Operation *lastWriter(Value value) {
  Operation *last = nullptr;
  for (Operation *user : value.getUsers()) {
    if (isa<memref::CastOp, memref::SubViewOp>(user)) return nullptr;
    bool writes = false;
    if (auto generic = dyn_cast<linalg::LinalgOp>(user)) writes = llvm::is_contained(generic.getDpsInits(), value);
    else if (auto copy = dyn_cast<memref::CopyOp>(user)) writes = copy.getTarget() == value;
    else if (!isa<memref::DimOp, memref::LoadOp, memref::DeallocOp>(user)) return nullptr;
    if (!writes) continue;
    if (last && user->getBlock() != last->getBlock()) return nullptr;
    if (!last || last->isBeforeInBlock(user)) last = user;
  }
  return last;
}
linalg::GenericOp producer(Value value) {
  llvm::SmallDenseSet<Value> seen;
  while (seen.insert(value).second) {
    value = stripCast(value);
    Operation *writer = lastWriter(value);
    if (auto copy = dyn_cast_or_null<memref::CopyOp>(writer)) value = copy.getSource();
    else return dyn_cast_or_null<linalg::GenericOp>(writer);
  }
  return {};
}
bool initializedFalse(Value value, Operation *before) {
  Operation *writer = nullptr;
  for (Operation *user : value.getUsers()) {
    if (isa<memref::CastOp, memref::SubViewOp>(user)) return false;
    bool writes = false;
    if (auto generic = dyn_cast<linalg::LinalgOp>(user)) writes = llvm::is_contained(generic.getDpsInits(), value);
    else if (auto copy = dyn_cast<memref::CopyOp>(user)) writes = copy.getTarget() == value;
    else if (isa<memref::StoreOp>(user)) return false;
    if (!writes || user == before) continue;
    if (user->getBlock() != before->getBlock()) return false;
    if (user->isBeforeInBlock(before) && (!writer || writer->isBeforeInBlock(user))) writer = user;
  }
  auto fill = dyn_cast_or_null<linalg::FillOp>(writer);
  return fill && uniformBoolean(UniformValueAnalysis(describeScalarValue).evaluate(fill.getInputs()[0])) == false;
}

std::optional<unsigned> validityField(RegionProgram program, linalg::GenericOp membership,
                                      unsigned memberAxis) {
  UniformValueAnalysis values(describeScalarValue);
  unsigned count = program.count("identity_count");
  auto &summary = program.summarize().front();
  auto &combine = program.combine().front();
  for (unsigned field = 0; field < count; ++field) {
    Value slot = summary.getArguments().take_back(count)[field];
    auto type = cast<MemRefType>(slot.getType());
    if (!type.getElementType().isInteger(1)) continue;
    auto reduction = producer(slot);
    if (!reduction || reduction.getNumDpsInputs() != 1 || reduction.getNumDpsInits() != 1 ||
        reduction.getNumReductionLoops() != 1 || stripCast(reduction.getInputs()[0]) != membership.getOutputs()[0] ||
        !initializedFalse(reduction.getOutputs()[0], reduction)) continue;
    auto sourceMap = reduction.getIndexingMapsArray()[0];
    auto coordinate = dyn_cast<AffineDimExpr>(sourceMap.getResult(memberAxis));
    if (!coordinate || reduction.getIteratorTypesArray()[coordinate.getPosition()] != utils::IteratorType::reduction) continue;
    Block &body = reduction.getRegion().front();
    if (!isBooleanUnion(values, body.getTerminator()->getOperand(0), body.getArgument(0), body.getArgument(1))) continue;
    Value identity = stripCast(program.identities()[field]);
    if (!initializedFalse(identity, program.getOperation())) continue;
    auto combined = producer(combine.getArguments().take_back(count)[field]);
    if (!combined || combined.getNumReductionLoops() || combined.getNumDpsInits() != 1 ||
        !combined.getIndexingMapsArray().back().isIdentity()) continue;
    Value lhs, rhs;
    for (auto [index, input] : llvm::enumerate(combined.getInputs())) {
      if (!combined.getIndexingMapsArray()[index].isIdentity()) continue;
      if (stripCast(input) == combine.getArgument(field)) lhs = combined.getRegion().front().getArgument(index);
      if (stripCast(input) == combine.getArgument(count + field)) rhs = combined.getRegion().front().getArgument(index);
    }
    Value result = combined.getRegion().front().getTerminator()->getOperand(0);
    if (lhs && rhs && isBooleanUnion(values, result, lhs, rhs) && hasTrueStateInvariant(values, result, lhs)) return field;
  }
  return std::nullopt;
}

}

std::optional<CoordinateSequence> coordinateSequence(Value memory, Operation *at) {
  CoordinateSequence result{{}, true, true};
  while (true) {
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) { memory = cast.getSource(); continue; }
    if (auto view = memory.getDefiningOp<memref::SubViewOp>()) {
      if (view.getSourceType().getRank() != 1 || view.getType().getRank() != 1 ||
          getConstantIntValue(view.getMixedStrides()[0]) != 1) return std::nullopt;
      auto offset = view.getMixedOffsets()[0];
      result.offsets.push_back(offset);
      result.beginsAtZero &= getConstantIntValue(offset) == 0;
      result.nonnegative &= isa<Attribute>(offset) ? cast<IntegerAttr>(cast<Attribute>(offset)).getInt() >= 0
                                                   : nonnegative(cast<Value>(offset));
      memory = view.getSource();
      continue;
    }
    if (auto argument = dyn_cast<BlockArgument>(memory))
      if (auto tasks = dyn_cast<TasksOp>(argument.getOwner()->getParentOp()); tasks && argument.getArgNumber()) {
        memory = tasks.getCaptures()[argument.getArgNumber() - 1];
        continue;
      }
    break;
  }
  auto allocation = memory.getDefiningOp<memref::AllocOp>();
  auto type = dyn_cast<MemRefType>(memory.getType());
  if (!allocation || !type || type.getRank() != 1 ||
      (!type.getElementType().isIndex() && !type.getElementType().isInteger(64))) return std::nullopt;
  linalg::GenericOp writer;
  SmallVector<Value> aliases{memory};
  llvm::SmallDenseSet<Value> visited;
  for (unsigned i = 0; i < aliases.size(); ++i) {
    Value alias = aliases[i];
    if (!visited.insert(alias).second) continue;
    for (OpOperand &use : alias.getUses()) {
      Operation *user = use.getOwner();
      if (auto generic = dyn_cast<linalg::GenericOp>(user)) {
        if (!llvm::is_contained(generic.getOutputs(), alias)) continue;
        if (writer || alias != memory || generic.getOutputs().size() != 1 || generic.getNumReductionLoops()) return std::nullopt;
        writer = generic;
      } else if (auto cast = dyn_cast<memref::CastOp>(user)) aliases.push_back(cast.getResult());
      else if (auto view = dyn_cast<memref::SubViewOp>(user)) aliases.push_back(view.getResult());
      else if (auto tasks = dyn_cast<TasksOp>(user)) {
        if (use.getOperandNumber() >= 1) aliases.push_back(tasks.getBody().front().getArgument(use.getOperandNumber()));
      } else if (isa<RegionFoldOp, RegionScanOp>(user)) {
        if (llvm::is_contained(RegionProgram(user).outputs(), alias)) return std::nullopt;
      } else if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getTarget() == alias) return std::nullopt;
      } else if (!isa<memref::LoadOp, memref::DimOp, memref::DeallocOp>(user)) return std::nullopt;
    }
  }
  if (!writer || !writer.getIndexingMapsArray().back().isIdentity()) return std::nullopt;
  DominanceInfo dominance(at->getParentOfType<func::FuncOp>());
  if (!dominance.properlyDominates(writer, at)) return std::nullopt;
  Value yielded = writer.getRegion().front().getTerminator()->getOperand(0);
  if (auto cast = yielded.getDefiningOp<arith::IndexCastOp>()) yielded = cast.getIn();
  auto index = yielded.getDefiningOp<linalg::IndexOp>();
  if (!index || index.getDim() != 0) return std::nullopt;
  return result;
}

std::optional<RegionPredicatePlan> analyzeRegionPredicate(RegionProgram program) {
  if (program.isScan() || !getEffectsRecursively(program.getOperation())) return std::nullopt;
  Block &helper = program.summarize().front();
  for (auto generic : helper.getOps<linalg::GenericOp>()) {
    if (generic.getNumReductionLoops() || generic.getOutputs().size() != 1) continue;
    auto compare = generic.getRegion().front().getTerminator()->getOperand(0).getDefiningOp<arith::CmpIOp>();
    if (!compare) continue;
    auto operand = [&](Value value) -> std::optional<std::pair<BlockArgument, unsigned>> {
      auto argument = dyn_cast<BlockArgument>(value);
      if (!argument || argument.getOwner() != &generic.getRegion().front() || argument.getArgNumber() >= generic.getNumDpsInputs()) return std::nullopt;
      Value input = stripCast(generic.getInputs()[argument.getArgNumber()]);
      AffineMap coordinates = generic.getIndexingMapsArray()[argument.getArgNumber()];
      Operation *consumer = generic;
      llvm::SmallDenseSet<Value> seen;
      while (!isa<BlockArgument>(input)) {
        if (!seen.insert(input).second) return std::nullopt;
        auto forward = dyn_cast_or_null<linalg::GenericOp>(lastWriter(input));
        if (!forward || forward.getOutputs().size() != 1 || forward.getNumReductionLoops() ||
            forward->getBlock() != consumer->getBlock() || !forward->isBeforeInBlock(consumer) ||
            !forward.getIndexingMapsArray().back().isPermutation()) return std::nullopt;
        Block &body = forward.getRegion().front();
        auto yielded = dyn_cast<BlockArgument>(body.getTerminator()->getOperand(0));
        if (!yielded || yielded.getOwner() != &body || yielded.getArgNumber() >= forward.getNumDpsInputs() ||
            llvm::any_of(body.without_terminator(), [](Operation &op) { return op.getNumRegions() || !isMemoryEffectFree(&op); })) return std::nullopt;
        auto maps = forward.getIndexingMapsArray();
        coordinates = maps[yielded.getArgNumber()].compose(inversePermutation(maps.back())).compose(coordinates);
        input = stripCast(forward.getInputs()[yielded.getArgNumber()]);
        consumer = forward;
      }
      auto formal = dyn_cast<BlockArgument>(input);
      auto type = dyn_cast<MemRefType>(input.getType());
      if (!formal || formal.getOwner() != &helper || !type || type.getRank() != 1 ||
          (!type.getElementType().isIndex() && !type.getElementType().isInteger(64))) return std::nullopt;
      auto axis = dyn_cast<AffineDimExpr>(coordinates.getResult(0));
      if (!axis) return std::nullopt;
      return std::make_pair(formal, axis.getPosition());
    };
    auto left = operand(compare.getLhs()), right = operand(compare.getRhs());
    if (!left || !right || left->second == right->second) continue;
    bool reverse = left->first.getArgNumber() >= program.count("source_count");
    auto source = reverse ? *right : *left, capture = reverse ? *left : *right;
    unsigned sourceCount = program.count("source_count");
    if (source.first.getArgNumber() >= sourceCount || capture.first.getArgNumber() < sourceCount ||
        capture.first.getArgNumber() >= sourceCount + program.count("capture_count")) continue;
    UniformPredicate predicate;
    switch (compare.getPredicate()) {
    case arith::CmpIPredicate::slt: predicate = reverse ? UniformPredicate::Greater : UniformPredicate::Less; break;
    case arith::CmpIPredicate::sle: predicate = reverse ? UniformPredicate::GreaterEqual : UniformPredicate::LessEqual; break;
    case arith::CmpIPredicate::sgt: predicate = reverse ? UniformPredicate::Less : UniformPredicate::Greater; break;
    case arith::CmpIPredicate::sge: predicate = reverse ? UniformPredicate::LessEqual : UniformPredicate::GreaterEqual; break;
    default: continue;
    }
    unsigned memberAxis = generic.getIndexingMapsArray().back().getNumResults();
    for (auto [axis, expression] : llvm::enumerate(generic.getIndexingMapsArray().back().getResults()))
      if (auto dimension = dyn_cast<AffineDimExpr>(expression); dimension && dimension.getPosition() == source.second) memberAxis = axis;
    if (memberAxis == generic.getIndexingMapsArray().back().getNumResults()) continue;
    auto validity = validityField(program, generic, memberAxis);
    auto reduction = validity ? producer(helper.getArguments().take_back(program.count("identity_count"))[*validity])
                              : linalg::GenericOp();
    return RegionPredicatePlan{compare, source.first.getArgNumber(), capture.first.getArgNumber() - sourceCount, predicate, validity, reduction};
  }
  return std::nullopt;
}

}
