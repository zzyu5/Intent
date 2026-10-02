#include "Intent/Dialect/CPU/Analysis/RegionPredicates.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/UniformValues.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;
namespace intent::cpu {
namespace {

bool fullSubview(memref::SubViewOp view) {
  auto type = view.getSourceType();
  if (type.getRank() != view.getType().getRank()) return false;
  for (unsigned axis = 0; axis < type.getRank(); ++axis) {
    if (getConstantIntValue(view.getMixedOffsets()[axis]) != 0 ||
        getConstantIntValue(view.getMixedStrides()[axis]) != 1) return false;
    if (!haveEqualExtents(ValueBoundsConstraintSet::Variable(view.getMixedSizes()[axis]),
                         ValueBoundsConstraintSet::Variable(view.getSource(), axis)))
      return false;
  }
  return true;
}
Value stripIdentityViews(Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<memref::CastOp>()) { value = cast.getSource(); continue; }
    if (auto view = value.getDefiningOp<memref::SubViewOp>(); view && fullSubview(view)) {
      value = view.getSource();
      continue;
    }
    return value;
  }
}
std::optional<SmallVector<Value>> fullAliases(Value value) {
  SmallVector<Value> aliases{stripIdentityViews(value)};
  llvm::SmallDenseSet<Value> seen;
  seen.insert(aliases.front());
  for (unsigned index = 0; index < aliases.size(); ++index) {
    for (Operation *user : aliases[index].getUsers()) {
      Value alias;
      if (auto cast = dyn_cast<memref::CastOp>(user)) alias = cast.getResult();
      else if (auto view = dyn_cast<memref::SubViewOp>(user)) {
        if (!fullSubview(view)) return std::nullopt;
        alias = view.getResult();
      }
      if (alias && seen.insert(alias).second) aliases.push_back(alias);
    }
  }
  return aliases;
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

Value stripIndexWidthCasts(Value value) {
  auto indexWidth = [](Type type) { return type.isIndex() || type.isInteger(64); };
  while (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
    if (!indexWidth(cast.getIn().getType()) || !indexWidth(cast.getType())) break;
    value = cast.getIn();
  }
  return value;
}

using SignedBounds = std::pair<llvm::APInt, llvm::APInt>;

llvm::APInt wideInteger(int64_t value) { return llvm::APInt(65, value, true); }

std::optional<SignedBounds> signedBounds(Value value);

std::optional<SignedBounds> extentBounds(Value memory, unsigned axis) {
  if (auto cast = memory.getDefiningOp<memref::CastOp>()) return extentBounds(cast.getSource(), axis);
  auto type = cast<MemRefType>(memory.getType());
  if (!type.isDynamicDim(axis))
    return SignedBounds{wideInteger(type.getDimSize(axis)), wideInteger(type.getDimSize(axis))};
  if (auto allocation = memory.getDefiningOp<memref::AllocOp>())
    return signedBounds(allocation.getDynamicSizes()[type.getDynamicDimIndex(axis)]);
  if (auto view = memory.getDefiningOp<memref::SubViewOp>()) {
    auto dropped = view.getDroppedDims();
    unsigned resultAxis = 0;
    for (auto [sourceAxis, size] : llvm::enumerate(view.getMixedSizes())) {
      if (dropped.test(sourceAxis)) continue;
      if (resultAxis++ != axis) continue;
      if (auto constant = getConstantIntValue(size))
        return SignedBounds{wideInteger(*constant), wideInteger(*constant)};
      auto bounds = signedBounds(cast<Value>(size));
      if (bounds) bounds->first = llvm::APIntOps::smax(bounds->first, wideInteger(0));
      return bounds;
    }
  }
  return SignedBounds{wideInteger(0), llvm::APInt::getSignedMaxValue(64).sext(65)};
}

std::optional<SignedBounds> signedBounds(Value value) {
  value = stripIndexWidthCasts(value);
  if (auto constant = getConstantIntValue(value))
    return SignedBounds{wideInteger(*constant), wideInteger(*constant)};
  if (auto dimension = value.getDefiningOp<memref::DimOp>())
    if (auto axis = dimension.getConstantIndex()) return extentBounds(dimension.getSource(), *axis);
  auto operation = value.getDefiningOp();
  if ((value.getType().isIndex() || value.getType().isInteger(64)) &&
      isa_and_nonnull<arith::AddIOp, arith::SubIOp, arith::MinSIOp, arith::MaxSIOp>(operation)) {
    auto left = signedBounds(operation->getOperand(0)), right = signedBounds(operation->getOperand(1));
    if (!left || !right) return std::nullopt;
    SignedBounds result = *left;
    if (isa<arith::AddIOp>(operation)) result = {left->first + right->first, left->second + right->second};
    else if (isa<arith::SubIOp>(operation)) result = {left->first - right->second, left->second - right->first};
    else if (isa<arith::MinSIOp>(operation))
      result = {llvm::APIntOps::smin(left->first, right->first), llvm::APIntOps::smin(left->second, right->second)};
    else result = {llvm::APIntOps::smax(left->first, right->first), llvm::APIntOps::smax(left->second, right->second)};
    if (result.first.slt(llvm::APInt::getSignedMinValue(64).sext(65)) ||
        result.second.sgt(llvm::APInt::getSignedMaxValue(64).sext(65))) return std::nullopt;
    return result;
  }
  if (auto argument = dyn_cast<BlockArgument>(value))
    if (auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
        loop && argument == loop.getInductionVar() && getConstantIntValue(loop.getStep()).value_or(0) > 0) {
      auto lower = signedBounds(loop.getLowerBound()), upper = signedBounds(loop.getUpperBound());
      if (lower && upper) return SignedBounds{lower->first, upper->second};
    }
  Value input;
  bool unsignedExtension = false;
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) input = cast.getIn();
  else if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>()) {
    input = cast.getIn();
    unsignedExtension = true;
  } else if (auto extension = value.getDefiningOp<arith::ExtSIOp>()) input = extension.getIn();
  else if (auto extension = value.getDefiningOp<arith::ExtUIOp>()) {
    input = extension.getIn();
    unsignedExtension = true;
  }
  auto integer = input ? dyn_cast<IntegerType>(input.getType()) : IntegerType();
  if (!integer || integer.getWidth() >= 64) return std::nullopt;
  unsigned width = integer.getWidth();
  if (unsignedExtension)
    return SignedBounds{wideInteger(0), llvm::APInt::getMaxValue(width).zext(65)};
  return SignedBounds{llvm::APInt::getSignedMinValue(width).sext(65),
                      llvm::APInt::getSignedMaxValue(width).sext(65)};
}

bool shiftedSequenceDoesNotWrap(memref::AllocOp allocation, Value origin) {
  auto begin = signedBounds(origin);
  if (!begin) return false;
  auto minimum = llvm::APInt::getSignedMinValue(64).sext(65);
  auto maximum = llvm::APInt::getSignedMaxValue(64).sext(65);
  auto type = allocation.getType();
  auto constant = type.isDynamicDim(0) ? getConstantIntValue(allocation.getDynamicSizes()[0])
                                     : std::optional<int64_t>(type.getDimSize(0));
  if (constant)
    return *constant >= 0 && (begin->second + wideInteger(*constant)).sle(maximum);
  auto extent = allocation.getDynamicSizes()[0].getDefiningOp<arith::MaxSIOp>();
  if (!extent) return false;
  Value difference;
  if (getConstantIntValue(extent.getRhs()) == 0) difference = extent.getLhs();
  else if (getConstantIntValue(extent.getLhs()) == 0) difference = extent.getRhs();
  auto subtraction = difference ? difference.getDefiningOp<arith::SubIOp>() : arith::SubIOp();
  if (!subtraction || stripIndexWidthCasts(subtraction.getRhs()) != stripIndexWidthCasts(origin))
    return false;
  auto end = signedBounds(subtraction.getLhs());
  if (!end) return false;
  // A non-wrapping max(end - begin, 0) keeps every populated coordinate
  // between its signed endpoints, even when the common origin is negative.
  return (end->first - begin->second).sge(minimum) &&
         (end->second - begin->first).sle(maximum);
}

Operation *lastWriter(Value value) {
  auto aliases = fullAliases(value);
  if (!aliases) return nullptr;
  Operation *last = nullptr;
  for (Value alias : *aliases) for (Operation *user : alias.getUsers()) {
    if (isa<memref::CastOp, memref::SubViewOp>(user)) continue;
    bool writes = false;
    if (auto generic = dyn_cast<linalg::LinalgOp>(user)) writes = llvm::is_contained(generic.getDpsInits(), alias);
    else if (auto copy = dyn_cast<memref::CopyOp>(user)) writes = copy.getTarget() == alias;
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
    value = stripIdentityViews(value);
    Operation *writer = lastWriter(value);
    if (auto copy = dyn_cast_or_null<memref::CopyOp>(writer)) value = copy.getSource();
    else return dyn_cast_or_null<linalg::GenericOp>(writer);
  }
  return {};
}
bool initializedFalse(Value value, Operation *before) {
  auto aliases = fullAliases(value);
  if (!aliases) return false;
  Operation *writer = nullptr;
  for (Value alias : *aliases) for (Operation *user : alias.getUsers()) {
    if (user == before || isa<memref::CastOp, memref::SubViewOp>(user)) continue;
    bool writes = false;
    if (auto generic = dyn_cast<linalg::LinalgOp>(user)) writes = llvm::is_contained(generic.getDpsInits(), alias);
    else if (auto copy = dyn_cast<memref::CopyOp>(user)) writes = copy.getTarget() == alias;
    else if (!isa<memref::DimOp, memref::LoadOp, memref::DeallocOp>(user)) return false;
    if (!writes) continue;
    if (user->getBlock() != before->getBlock()) return false;
    if (user->isBeforeInBlock(before) && (!writer || writer->isBeforeInBlock(user))) writer = user;
  }
  auto fill = dyn_cast_or_null<linalg::FillOp>(writer);
  return fill && uniformBoolean(UniformValueAnalysis(describeScalarValue).evaluate(fill.getInputs()[0])) == false;
}

std::optional<unsigned> validityField(RegionOpInterface program, linalg::GenericOp membership,
                                      unsigned memberAxis) {
  UniformValueAnalysis values(describeScalarValue);
  auto summaryOutputs = program.getRegionArguments(program.getSummarize(), RegionArgumentKind::Destinations);
  auto combinedOutputs = program.getRegionArguments(program.getCombine(), RegionArgumentKind::Destinations);
  auto left = program.getRegionArguments(program.getCombine(), RegionArgumentKind::LeftSummary);
  auto right = program.getRegionArguments(program.getCombine(), RegionArgumentKind::RightSummary);
  for (auto [field, slot] : llvm::enumerate(summaryOutputs)) {
    auto type = cast<MemRefType>(slot.getType());
    if (!type.getElementType().isInteger(1)) continue;
    auto reduction = producer(slot);
    if (!reduction || reduction.getNumDpsInputs() != 1 || reduction.getNumDpsInits() != 1 ||
        reduction.getNumReductionLoops() != 1 || stripIdentityViews(reduction.getInputs()[0]) != stripIdentityViews(membership.getOutputs()[0]) ||
        !initializedFalse(reduction.getOutputs()[0], reduction)) continue;
    auto sourceMap = reduction.getIndexingMapsArray()[0];
    auto coordinate = dyn_cast<AffineDimExpr>(sourceMap.getResult(memberAxis));
    if (!coordinate || reduction.getIteratorTypesArray()[coordinate.getPosition()] != utils::IteratorType::reduction) continue;
    Block &body = reduction.getRegion().front();
    if (!isBooleanUnion(values, body.getTerminator()->getOperand(0), body.getArgument(0), body.getArgument(1))) continue;
    Value identity = stripIdentityViews(program.getIdentities()[field]);
    if (!initializedFalse(identity, program.getOperation())) continue;
    auto combined = producer(combinedOutputs[field]);
    if (!combined || combined.getNumReductionLoops() || combined.getNumDpsInits() != 1 ||
        !combined.getIndexingMapsArray().back().isIdentity()) continue;
    Value lhs, rhs;
    for (auto [index, input] : llvm::enumerate(combined.getInputs())) {
      if (!combined.getIndexingMapsArray()[index].isIdentity()) continue;
      if (stripIdentityViews(input) == left[field]) lhs = combined.getRegion().front().getArgument(index);
      if (stripIdentityViews(input) == right[field]) rhs = combined.getRegion().front().getArgument(index);
    }
    Value result = combined.getRegion().front().getTerminator()->getOperand(0);
    if (lhs && rhs && isBooleanUnion(values, result, lhs, rhs) && hasTrueStateInvariant(values, result, lhs)) return field;
  }
  return std::nullopt;
}

class ConstantMemory {
public:
  explicit ConstantMemory(PhysicalProgramAnalysis &physical) : physical(physical), values(describeScalarValue) {}

  Attribute read(Value value, const UniformBindings &scalars = UniformBindings()) {
    if (!isa<MemRefType>(value.getType())) return values.evaluate(value, scalars);
    value = stripIdentityViews(value);
    auto found = facts.find(value);
    return found != facts.end() ? found->second : facts.lookup(physical.storageRoot(value));
  }
  void write(Value value, Attribute constant) {
    Value root = physical.storageRoot(value);
    SmallVector<Value> invalidated;
    for (auto &fact : facts)
      if (physical.storageRoot(fact.first) == root) invalidated.push_back(fact.first);
    for (Value alias : invalidated) facts.erase(alias);
    if (constant) facts[stripIdentityViews(value)] = constant;
  }
  void visit(Operation *operation, const UniformBindings &scalars = UniformBindings()) {
    if (auto fill = dyn_cast<linalg::FillOp>(operation)) {
      write(fill.getOutputs()[0], read(fill.getInputs()[0], scalars));
    } else if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
      write(copy.getTarget(), read(copy.getSource(), scalars));
    } else if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
      if (generic.getOutputs().size() != 1) { facts.clear(); return; }
      UniformBindings operands;
      Value output = generic.getOutputs()[0];
      auto maps = generic.getIndexingMapsArray();
      bool unsafeAlias = false;
      for (auto [number, input] : llvm::enumerate(generic.getInputs())) {
        operands[input] = read(input, scalars);
        if (isa<MemRefType>(input.getType()) && physical.storageRoot(input) == physical.storageRoot(output))
          unsafeAlias |= input != output || generic.getNumReductionLoops() ||
              maps[number] != maps.back() || !maps.back().isPermutation();
      }
      operands[output] = read(output, scalars);
      write(output, unsafeAlias ? Attribute() : foldUniformComputation(generic, operands, scalars));
    } else if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      write(store.getMemref(), store.getIndices().empty() ? read(store.getValue(), scalars) : Attribute());
    } else {
      auto effects = getEffectsRecursively(operation);
      if (!effects) { facts.clear(); return; }
      for (auto &effect : *effects) {
        if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) continue;
        if (effect.getValue()) write(effect.getValue(), {});
        else facts.clear();
      }
    }
  }

private:
  PhysicalProgramAnalysis &physical;
  UniformValueAnalysis values;
  UniformBindings facts;
};

bool summaryIsIdentityWhenFalse(RegionOpInterface program, Value predicate) {
  PhysicalProgramAnalysis physical(program.getOperation()->getParentOfType<func::FuncOp>());
  for (Region &region : program.getOperation()->getRegions()) {
    auto destinations = program.getRegionArguments(region, RegionArgumentKind::Destinations);
    for (Operation &operation : region.front().without_terminator()) {
      auto effects = getEffectsRecursively(&operation);
      if (!effects) return false;
      for (auto &effect : *effects) {
        if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) continue;
        if (!effect.getValue()) return false;
        Value root = physical.storageRoot(effect.getValue());
        if (auto argument = dyn_cast<BlockArgument>(root)) {
          if (!llvm::is_contained(destinations, argument) ||
              !isa<MemoryEffects::Write>(effect.getEffect())) return false;
        } else {
          auto allocation = root.getDefiningOp();
          if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(allocation) ||
              !region.isAncestor(allocation->getParentRegion())) return false;
        }
      }
    }
  }
  ConstantMemory outer(physical), summary(physical);
  for (Operation &operation : *program.getOperation()->getBlock()) {
    if (&operation == program.getOperation()) break;
    outer.visit(&operation);
  }
  Block &helper = program.getSummarize().front();
  auto schema = program.getRegionSchema(program.getSummarize());
  if (failed(schema)) return false;
  for (const auto &relation : *schema)
    if (relation.kind != RegionArgumentKind::Destinations)
      summary.write(relation.argument, outer.read(relation.prototype));
  UniformBindings scalars;
  scalars[predicate] = IntegerAttr::get(IntegerType::get(predicate.getContext(), 1), 0);
  for (Operation &operation : helper.without_terminator()) summary.visit(&operation, scalars);
  for (auto [result, identity] : llvm::zip(
           program.getRegionArguments(program.getSummarize(), RegionArgumentKind::Destinations),
           program.getIdentities()))
    if (!equalUniformConstants(summary.read(result), outer.read(identity))) return false;
  return true;
}

}

std::optional<CoordinateSequence> coordinateSequence(Value memory, Operation *at) {
  CoordinateSequence result{{}, true, true, {}};
  Value selectedMemory = memory;
  while (true) {
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) { memory = cast.getSource(); continue; }
    if (auto view = memory.getDefiningOp<memref::SubViewOp>()) {
      if (view.getSourceType().getRank() != 1 || view.getType().getRank() != 1 ||
          getConstantIntValue(view.getMixedStrides()[0]) != 1) return std::nullopt;
      auto offset = view.getMixedOffsets()[0];
      result.offsets.push_back(offset);
      result.startsAtOrigin &= getConstantIntValue(offset) == 0;
      result.nonnegativeOffsets &= isa<Attribute>(offset) ? cast<IntegerAttr>(cast<Attribute>(offset)).getInt() >= 0
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
        if (llvm::is_contained(mlir::cast<RegionOpInterface>(user).getDestinations(), alias)) return std::nullopt;
      } else if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getTarget() == alias) return std::nullopt;
      } else if (!isa<memref::LoadOp, memref::DimOp, memref::DeallocOp>(user)) return std::nullopt;
    }
  }
  if (!writer || !writer.getIndexingMapsArray().back().isIdentity()) return std::nullopt;
  DominanceInfo dominance(at->getParentOfType<func::FuncOp>());
  if (!dominance.properlyDominates(writer, at)) return std::nullopt;
  Value yielded = stripIndexWidthCasts(writer.getRegion().front().getTerminator()->getOperand(0));
  if (auto addition = yielded.getDefiningOp<arith::AddIOp>()) {
    Value left = stripIndexWidthCasts(addition.getLhs());
    Value right = stripIndexWidthCasts(addition.getRhs());
    if (left.getDefiningOp<linalg::IndexOp>()) {
      yielded = left;
      result.origin = right;
    } else if (right.getDefiningOp<linalg::IndexOp>()) {
      yielded = right;
      result.origin = left;
    } else return std::nullopt;
    if (!dominance.properlyDominates(result.origin, writer) ||
        !shiftedSequenceDoesNotWrap(allocation, result.origin)) return std::nullopt;
    // Range construction also adds view offsets and an exclusive endpoint.
    // Prove those relative index operations fit before removing the origin.
    auto extent = extentBounds(selectedMemory, 0);
    if (!extent || extent->first.isNegative()) return std::nullopt;
    llvm::APInt upper = extent->second;
    for (OpFoldResult offset : result.offsets) {
      auto constant = getConstantIntValue(offset);
      auto bounds = constant ? std::optional<SignedBounds>({wideInteger(*constant), wideInteger(*constant)})
                             : signedBounds(cast<Value>(offset));
      if (!bounds || bounds->first.isNegative()) return std::nullopt;
      upper += bounds->second;
      if (upper.sgt(llvm::APInt::getSignedMaxValue(64).sext(65))) return std::nullopt;
    }
  }
  auto index = yielded.getDefiningOp<linalg::IndexOp>();
  if (!index || index.getDim() != 0 || index->getBlock() != &writer.getRegion().front())
    return std::nullopt;
  return result;
}

std::optional<RegionPredicatePlan> analyzeRegionPredicate(RegionOpInterface program) {
  if (program.isScan() || !getEffectsRecursively(program.getOperation())) return std::nullopt;
  Block &helper = program.getSummarize().front();
  for (auto generic : helper.getOps<linalg::GenericOp>()) {
    if (generic.getNumReductionLoops() || generic.getOutputs().size() != 1) continue;
    auto compare = generic.getRegion().front().getTerminator()->getOperand(0).getDefiningOp<arith::CmpIOp>();
    if (!compare) continue;
    auto operand = [&](Value value) -> std::optional<std::pair<BlockArgument, unsigned>> {
      auto argument = dyn_cast<BlockArgument>(value);
      if (!argument || argument.getOwner() != &generic.getRegion().front() || argument.getArgNumber() >= generic.getNumDpsInputs()) return std::nullopt;
      Value input = stripIdentityViews(generic.getInputs()[argument.getArgNumber()]);
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
        input = stripIdentityViews(forward.getInputs()[yielded.getArgNumber()]);
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
    auto sourceArguments = program.getRegionArguments(program.getSummarize(), RegionArgumentKind::Sources);
    auto captureArguments = program.getRegionArguments(program.getSummarize(), RegionArgumentKind::Captures);
    bool reverse = !llvm::is_contained(sourceArguments, left->first);
    auto source = reverse ? *right : *left, capture = reverse ? *left : *right;
    if (!llvm::is_contained(sourceArguments, source.first) ||
        !llvm::is_contained(captureArguments, capture.first)) continue;
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
    auto reduction = validity ? producer(program.getRegionArguments(
        program.getSummarize(), RegionArgumentKind::Destinations)[*validity])
                              : linalg::GenericOp();
    return RegionPredicatePlan{compare,
        static_cast<unsigned>(llvm::find(sourceArguments, source.first) - sourceArguments.begin()),
        static_cast<unsigned>(llvm::find(captureArguments, capture.first) - captureArguments.begin()),
        predicate, validity, reduction, summaryIsIdentityWhenFalse(program, compare)};
  }
  return std::nullopt;
}

}
