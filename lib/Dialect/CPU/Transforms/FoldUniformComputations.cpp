#include "Intent/Analysis/UniformValues.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
        inputs[output] = read(output);
        Attribute constant;
        if (contraction) {
          UniformExpression expression;
          expression.kind = UniformKind::Contract;
          expression.type = cast<MemRefType>(output.getType()).getElementType();
          expression.operands.assign(generic.getInputs().begin(), generic.getInputs().end());
          expression.operands.push_back(output);
          constant = values.fold(expression, inputs);
        } else if (!generic.getNumReductionLoops() && maps.back().isPermutation()) {
          UniformBindings arguments;
          for (auto [argument, operand] : llvm::zip(generic.getRegion().front().getArguments(), generic->getOperands()))
            arguments[argument] = inputs.lookup(operand);
          constant = values.evaluate(generic.getRegion().front().getTerminator()->getOperand(0), arguments);
        } else if (generic.getNumReductionLoops()) {
          UniformExpression expression;
          expression.kind = UniformKind::Fold;
          expression.stateCount = 1;
          expression.type = cast<MemRefType>(output.getType()).getElementType();
          expression.operands.push_back(output);
          llvm::append_range(expression.operands, generic.getInputs());
          auto arguments = generic.getRegion().front().getArguments();
          expression.parameters.push_back(arguments.back());
          llvm::append_range(expression.parameters, arguments.drop_back());
          expression.yields.push_back(generic.getRegion().front().getTerminator()->getOperand(0));
          auto iterators = generic.getIteratorTypesArray();
          SmallVector<bool> positive(iterators.size(), false);
          for (auto [operand, map] : llvm::zip(generic->getOperands(), generic.getIndexingMapsArray())) {
            auto type = dyn_cast<MemRefType>(operand.getType());
            if (!type) continue;
            for (auto [axis, coordinate] : llvm::enumerate(map.getResults()))
              if (auto dimension = dyn_cast<AffineDimExpr>(coordinate))
                positive[dimension.getPosition()] = positive[dimension.getPosition()] || type.getDimSize(axis) > 0;
          }
          expression.nonempty = llvm::all_of(llvm::enumerate(iterators), [&](auto iterator) {
            return iterator.value() != utils::IteratorType::reduction || positive[iterator.index()];
          });
          constant = values.fold(expression, inputs);
        }
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
  PhysicalProgramAnalysis physical;
  UniformValueAnalysis values;
};

}

LogicalResult foldUniformComputations(func::FuncOp function) {
  UniformComputations(function).run(function.front(), UniformBindings());
  return success();
}

}
