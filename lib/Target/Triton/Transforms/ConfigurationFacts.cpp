#include "ConfigurationFacts.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"

using namespace mlir;

namespace intent::triton {
namespace {

// Both recurrence questions follow the same current SSA/loop-carried edges.
// The terminal condition distinguishes a particular carry from a contraction;
// neither walk reinterprets the operation's numeric semantics.
bool dependsOn(Value value, scf::ForOp owner,
               llvm::function_ref<bool(Value)> terminal,
               llvm::SmallDenseSet<Value, 32> &visited) {
  if (!visited.insert(value).second) return false;
  if (terminal(value)) return true;
  Operation *definition = value.getDefiningOp();
  if (!definition || !owner->isProperAncestor(definition)) return false;
  if (auto nested = dyn_cast<scf::ForOp>(definition)) {
    auto result = dyn_cast<OpResult>(value);
    auto yield = dyn_cast<scf::YieldOp>(nested.getBody()->getTerminator());
    if (result && yield && result.getResultNumber() < nested.getInitArgs().size()) {
      unsigned index = result.getResultNumber();
      if (dependsOn(nested.getInitArgs()[index], owner, terminal, visited) ||
          dependsOn(yield.getOperand(index), owner, terminal, visited))
        return true;
    }
  }
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return dependsOn(operand, owner, terminal, visited);
  });
}

bool isDirectContractionAccumulator(Value value, Value carry, scf::ForOp owner,
                                    llvm::SmallDenseSet<Value, 8> &visited) {
  if (!visited.insert(value).second) return false;
  Operation *definition = value.getDefiningOp();
  if (auto contract = dyn_cast_or_null<gpu::ContractOp>(definition)) {
    if (contract.getAccumulator() != carry) return false;
    llvm::SmallDenseSet<Value, 32> dependencies;
    auto isCarry = [&](Value candidate) { return candidate == carry; };
    return !dependsOn(contract.getLhs(), owner, isCarry, dependencies) &&
           !dependsOn(contract.getRhs(), owner, isCarry, dependencies);
  }
  auto nested = dyn_cast_or_null<scf::ForOp>(definition);
  auto result = dyn_cast<OpResult>(value);
  auto yield = nested ? dyn_cast<scf::YieldOp>(nested.getBody()->getTerminator())
                      : scf::YieldOp();
  return nested && result && yield &&
         result.getResultNumber() < nested.getInitArgs().size() &&
         nested.getInitArgs()[result.getResultNumber()] == carry &&
         isDirectContractionAccumulator(yield.getOperand(result.getResultNumber()),
             nested.getRegionIterArgs()[result.getResultNumber()], nested, visited);
}

bool hasRecurrentContraction(scf::ForOp loop) {
  if (loop.getInitArgs().empty()) return false;
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != loop.getRegionIterArgs().size()) return false;
  for (auto [index, next] : llvm::enumerate(yield.getOperands())) {
    llvm::SmallDenseSet<Value, 8> directVisited;
    if (isDirectContractionAccumulator(next, loop.getRegionIterArgs()[index], loop, directVisited))
      continue;
    llvm::SmallDenseSet<Value, 32> contractionVisited;
    if (!dependsOn(next, loop, [&](Value candidate) {
          auto contraction = candidate.getDefiningOp<gpu::ContractOp>();
          return contraction && loop->isProperAncestor(contraction);
        }, contractionVisited))
      continue;
    llvm::SmallDenseSet<Value, 32> carryVisited;
    if (dependsOn(next, loop, [&](Value candidate) {
          return candidate == loop.getRegionIterArgs()[index];
        }, carryVisited))
      return true;
  }
  return false;
}

} // namespace

ProgramConfigurationFacts::ProgramConfigurationFacts(func::FuncOp kernel) {
  bool mayFormDot = false, hasReductionOrScan = false;
  bool hasScaledOrSparse = false, loopMemory = false, straightLine = true;
  for (Attribute attribute : gpu::getParameterDeclarations(kernel)) {
      auto schema = cast<gpu::ParameterAttr>(attribute);
      auto category = schema.getCategory();
      twoAxisPointwise |= category == gpu::ParameterCategory::Pointwise &&
          schema.getRole() == gpu::ParameterRole::OwnershipM;
      if (category != gpu::ParameterCategory::Coverage &&
          category != gpu::ParameterCategory::Provider &&
          !llvm::is_contained(categories, category))
        categories.push_back(category);
  }
  kernel.walk([&](Operation *operation) {
    mayFormDot |= isa<gpu::ReduceOp, ReduceOp>(operation);
    hasReductionOrScan |= isa<gpu::ReduceOp, gpu::ScanOp>(operation);
    if (auto contract = dyn_cast<gpu::ContractOp>(operation)) {
      hasContraction = true;
      allContractionsF32 &= contract.getLhs().getType().getElementType().isF32() &&
          contract.getRhs().getType().getElementType().isF32() &&
          contract.getResult().getType().getElementType().isF32();
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation); loop && !recurrentContraction)
      recurrentContraction = hasRecurrentContraction(loop);
    hasScaledOrSparse |= isa<gpu::ScaledContractOp, gpu::SparseContractOp>(operation);
    loopMemory |= operation->getParentOfType<scf::ForOp>() && !isMemoryEffectFree(operation);
    if (operation->getBlock() == &kernel.front())
      straightLine &= operation->getNumRegions() == 0;
  });
  if (hasReductionOrScan && !llvm::is_contained(categories, gpu::ParameterCategory::Reduction))
    categories.push_back(gpu::ParameterCategory::Reduction);
  if (hasContraction && !llvm::is_contained(categories, gpu::ParameterCategory::Contraction))
    categories.push_back(gpu::ParameterCategory::Contraction);
  straightLinePointwise = straightLine &&
      llvm::is_contained(categories, gpu::ParameterCategory::Pointwise) &&
      llvm::all_of(categories, [](gpu::ParameterCategory category) {
        return category == gpu::ParameterCategory::Pointwise;
      });
  pipelineStagesAffectProgram = (hasContraction && !allContractionsF32) ||
      hasScaledOrSparse || ((hasContraction || mayFormDot) && loopMemory);
}

} // namespace intent::triton
