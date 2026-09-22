#include "Intent/Dialect/CPU/Analysis/UniformValues.h"
#include "Intent/Dialect/CPU/Analysis/RegionPredicates.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
namespace intent::cpu {
namespace {

class UniformComputations {
public:
  explicit UniformComputations(func::FuncOp function)
      : physical(function), values(describeScalarValue) {}

  void run(Block &block, UniformBindings memory) {
    auto read = [&](Value value) -> Attribute {
      if (!isa<MemRefType>(value.getType())) return values.evaluate(value);
      auto found = memory.find(value);
      return found != memory.end() ? found->second : memory.lookup(physical.storageRoot(value));
    };
    auto forget = [&](Value value) {
      Value root = physical.storageRoot(value);
      SmallVector<Value> aliases;
      for (auto &fact : memory)
        if (physical.storageRoot(fact.first) == root) aliases.push_back(fact.first);
      for (Value alias : aliases) memory.erase(alias);
    };
    auto write = [&](Value value, Attribute constant) {
      forget(value);
      if (constant) memory[value] = constant;
    };
    for (Operation &operation : llvm::make_early_inc_range(block)) {
      if (auto fill = dyn_cast<linalg::FillOp>(operation)) {
        write(fill.getOutputs()[0], values.evaluate(fill.getInputs()[0]));
        continue;
      }
      if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
        Attribute constant = read(copy.getSource());
        write(copy.getTarget(), constant);
        continue;
      }
      if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
        if (generic.getOutputs().size() != 1 || generic.getNumResults()) { memory.clear(); continue; }
        Value output = generic.getOutputs()[0];
        if (llvm::any_of(generic.getRegion().front().without_terminator(), [](Operation &nested) {
              return nested.getNumRegions() || !isMemoryEffectFree(&nested);
            })) { memory.clear(); continue; }
        UniformBindings inputs;
        auto maps = generic.getIndexingMapsArray();
        bool contraction = isMatrixContraction(generic);
        bool unsafeAlias = false;
        for (auto [index, input] : llvm::enumerate(generic.getInputs())) {
          bool aliasesOutput = physical.storageRoot(input) == physical.storageRoot(output);
          bool sameElement = input == output && !generic.getNumReductionLoops() &&
              maps[index] == maps.back() && maps.back().isPermutation();
          unsafeAlias |= aliasesOutput && !sameElement;
          inputs[input] = !aliasesOutput || sameElement ? read(input) : Attribute();
        }
        if (unsafeAlias) { forget(output); continue; }
        if (foldCoordinateReduction(generic)) { forget(output); continue; }
        inputs[output] = read(output);
        Attribute constant = foldUniformComputation(generic, inputs);
        if (!contraction && !constant) {
          OpBuilder builder = OpBuilder::atBlockBegin(&generic.getRegion().front());
          for (auto [argument, input] : llvm::zip(generic.getRegion().front().getArguments(), generic.getInputs())) {
            Attribute known = inputs.lookup(input);
            if (!known || argument.use_empty()) continue;
            Value scalar = builder.create<arith::ConstantOp>(generic.getLoc(), argument.getType(), cast<TypedAttr>(known));
            argument.replaceAllUsesWith(scalar);
          }
        }
        write(output, constant);
        if (!constant) continue;
        OpBuilder builder(generic);
        Value scalar = builder.create<arith::ConstantOp>(generic.getLoc(), cast<MemRefType>(output.getType()).getElementType(), cast<TypedAttr>(constant));
        builder.create<linalg::FillOp>(generic.getLoc(), ValueRange{scalar}, ValueRange{output});
        generic.erase();
        continue;
      }
      if (isa<memref::AllocOp, memref::AllocaOp, memref::CastOp, memref::SubViewOp, memref::DimOp>(operation)) continue;
      if (auto dealloc = dyn_cast<memref::DeallocOp>(operation)) { forget(dealloc.getMemref()); continue; }
      auto accesses = physical.accesses(&operation);
      auto effects = getEffectsRecursively(&operation);
      bool unknownWrite = !effects.has_value();
      SmallVector<Value> written;
      if (effects)
        for (auto &effect : *effects) {
          if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) continue;
          if (Value value = effect.getValue()) written.push_back(value);
          else unknownWrite = true;
        }
      auto invalidate = [&]() {
        if (unknownWrite) { memory.clear(); return; }
        for (Value value : written) forget(value);
        for (auto access : accesses) if (access.write) forget(access.memory);
      };
      // A repeated body cannot inherit a pre-loop constant for mutated storage.
      if (isa<scf::ForOp, scf::WhileOp>(operation)) invalidate();
      for (Region &region : operation.getRegions())
        for (Block &nested : region) run(nested, memory);
      invalidate();
    }
  }

private:
  bool foldCoordinateReduction(linalg::GenericOp reduction) {
    if (reduction.getNumDpsInputs() != 1 || reduction.getNumReductionLoops() != 1 ||
        reduction.getNumLoops() != 1) return false;
    Value input = reduction.getInputs()[0], output = reduction.getOutputs()[0];
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outputType = cast<MemRefType>(output.getType());
    if (!inputType || inputType.getRank() != 1 || !inputType.getElementType().isInteger(1) ||
        outputType.getRank() != 0 || !outputType.getElementType().isInteger(1) ||
        !input.getDefiningOp<memref::AllocOp>()) return false;
    auto maps = reduction.getIndexingMapsArray();
    if (!maps.front().isIdentity() || maps.back().getNumResults() != 0) return false;
    Block &body = reduction.getRegion().front();
    Operation *combine = body.getTerminator()->getOperand(0).getDefiningOp();
    if (!combine || !isa<arith::AndIOp, arith::OrIOp>(combine) ||
        !((combine->getOperand(0) == body.getArgument(0) && combine->getOperand(1) == body.getArgument(1)) ||
          (combine->getOperand(1) == body.getArgument(0) && combine->getOperand(0) == body.getArgument(1)))) return false;
    bool disjunction = isa<arith::OrIOp>(combine);

    linalg::GenericOp predicate;
    for (Operation *user : input.getUsers()) {
      auto writer = dyn_cast<linalg::GenericOp>(user);
      if (!writer || !llvm::is_contained(writer.getOutputs(), input)) continue;
      if (predicate) return false;
      predicate = writer;
    }
    if (!predicate || predicate.getNumResults() || predicate.getOutputs().size() != 1 ||
        predicate.getNumReductionLoops() || predicate.getNumLoops() != 1 ||
        !predicate.getIndexingMapsArray().back().isIdentity() ||
        !physical.mayReadAt(input, predicate, reduction)) return false;
    Block &predicateBody = predicate.getRegion().front();
    if (!predicateBody.getArguments().back().use_empty() ||
        llvm::any_of(predicateBody.without_terminator(), [](Operation &operation) {
          return operation.getNumRegions() || !isMemoryEffectFree(&operation);
        })) return false;
    auto comparison = predicateBody.getTerminator()->getOperand(0).getDefiningOp<arith::CmpIOp>();
    if (!comparison || !matchPattern(comparison.getRhs(), m_Zero())) return false;
    auto coordinate = dyn_cast<BlockArgument>(comparison.getLhs());
    if (!coordinate || coordinate.getOwner() != &predicateBody ||
        coordinate.getArgNumber() >= predicate.getNumDpsInputs() ||
        !predicate.getIndexingMapsArray()[coordinate.getArgNumber()].isIdentity()) return false;
    auto sequence = coordinateSequence(predicate.getInputs()[coordinate.getArgNumber()], predicate);
    if (!sequence || sequence->origin) return false;
    if (comparison.getPredicate() == arith::CmpIPredicate::sge && sequence->nonnegativeOffsets) {
      if (!disjunction) {
        // Every member is true; conjunction leaves the incoming state intact,
        // including an empty domain.
        reduction.erase();
        return true;
      }
    } else if (comparison.getPredicate() != arith::CmpIPredicate::eq || !sequence->startsAtOrigin) {
      return false;
    }
    OpBuilder builder(reduction);
    Location loc = reduction.getLoc();
    Value extent = builder.create<memref::DimOp>(loc, input, 0);
    Value bound = builder.create<arith::ConstantIndexOp>(loc, disjunction ? 0 : 1);
    Value aggregate = builder.create<arith::CmpIOp>(loc,
        disjunction ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::sle, extent, bound);
    Value initial = builder.create<memref::LoadOp>(loc, output, ValueRange{});
    Value result = disjunction ? Value(builder.create<arith::OrIOp>(loc, initial, aggregate))
                               : Value(builder.create<arith::AndIOp>(loc, initial, aggregate));
    builder.create<memref::StoreOp>(loc, result, output, ValueRange{});
    reduction.erase();
    return true;
  }

  PhysicalProgramAnalysis physical;
  UniformValueAnalysis values;
};

}

LogicalResult foldUniformComputations(func::FuncOp function) {
  UniformComputations(function).run(function.front(), UniformBindings());
  return success();
}

}
