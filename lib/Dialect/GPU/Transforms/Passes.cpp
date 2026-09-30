#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Predication.h"
#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Transforms/PassManager.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <functional>

using namespace mlir;

namespace intent::gpu {
namespace {

// Complete transformation entries own relation closure; the pipeline schedules
// their semantic dependencies and verifies each finished group.
LogicalResult normalizeStructuredSources(ModuleOp module) {
  if (failed(eliminateCommonValues(module))) return failure();
  return normalizeContractionSources(module);
}

LogicalResult realizeReductionGroup(ModuleOp module) {
  if (failed(fuseIndependentReductions(module))) return failure();
  return realizeReductionBlocking(module);
}

LogicalResult composeRealizedAccesses(ModuleOp module) {
  if (failed(realizeAccessComposition(module)) ||
      failed(materializeRetainedValues(module)) ||
      failed(realizeAccessComposition(module)))
    return failure();
  return orientLoopContractions(module);
}

LogicalResult simplifyValues(ModuleOp module) {
  if (failed(eliminateCommonValues(module)))
    return failure();
  return simplifyMaskedAccessCoordinates(module);
}

LogicalResult closeSharedConfigurations(func::FuncOp kernel,
                                        const TuningProfiles &profiles) {
  eraseUnusedPhysicalParameters(kernel);
  if (failed(materializeSharedConfigTuples(kernel, profiles)))
    return failure();
  return verifySharedConfigTuples(kernel);
}

LogicalResult finishTransformation(ModuleOp module, StringRef name,
                                   LogicalResult result) {
  if (failed(result))
    return module.emitError() << "shared GPU transformation failed: " << name;
  if (failed(verifyGPUProgram(module)))
    return module.emitError() << "shared GPU postcondition failed: " << name;
  return success();
}

class NormalizeStructuredSourcesPass : public PassWrapper<NormalizeStructuredSourcesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeStructuredSourcesPass)
  StringRef getArgument() const final { return "intent-gpu-normalize-structured-sources"; }
  StringRef getDescription() const final { return "Normalize structured GPU sources"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), normalizeStructuredSources(module))))
      signalPassFailure();
  }
};

class PredicateScalarControlPass : public PassWrapper<PredicateScalarControlPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PredicateScalarControlPass)
  StringRef getArgument() const final { return "intent-gpu-predicate-scalar-control"; }
  StringRef getDescription() const final { return "Predicate scalar control in the GPU program"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), predicateScalarControl(module))))
      signalPassFailure();
  }
};

class PointwiseOwnershipPass : public PassWrapper<PointwiseOwnershipPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PointwiseOwnershipPass)
  StringRef getArgument() const final { return "intent-gpu-form-pointwise-ownership"; }
  StringRef getDescription() const final { return "Form pointwise GPU ownership"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizePointwiseOwnership(module))))
      signalPassFailure();
  }
};

class PointwiseBlockingPass : public PassWrapper<PointwiseBlockingPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PointwiseBlockingPass)
  StringRef getArgument() const final { return "intent-gpu-form-pointwise-blocking"; }
  StringRef getDescription() const final { return "Form pointwise GPU blocking and its dependent value graph"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizePointwiseBlocking(module))))
      signalPassFailure();
  }
};

class ScanConsumersPass : public PassWrapper<ScanConsumersPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScanConsumersPass)
  StringRef getArgument() const final { return "intent-gpu-realize-scan-consumers"; }
  StringRef getDescription() const final { return "Realize GPU scan consumer traversals"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizeScanConsumerTraversals(module))))
      signalPassFailure();
  }
};

class RetainedValuesPass : public PassWrapper<RetainedValuesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RetainedValuesPass)
  StringRef getArgument() const final { return "intent-gpu-materialize-retained-values"; }
  StringRef getDescription() const final { return "Materialize retained GPU values"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), materializeRetainedValues(module))))
      signalPassFailure();
  }
};

class OnlineReductionsPass : public PassWrapper<OnlineReductionsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OnlineReductionsPass)
  StringRef getArgument() const final { return "intent-gpu-co-realize-online-reductions"; }
  StringRef getDescription() const final { return "Co-realize GPU online reductions"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizeOnlineReductions(module))))
      signalPassFailure();
  }
};

class RegionFoldsPass : public PassWrapper<RegionFoldsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RegionFoldsPass)
  StringRef getArgument() const final { return "intent-gpu-realize-region-folds"; }
  StringRef getDescription() const final { return "Realize GPU region folds"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizeRegionFolds(module))))
      signalPassFailure();
  }
};

class RegionScansPass : public PassWrapper<RegionScansPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RegionScansPass)
  StringRef getArgument() const final { return "intent-gpu-realize-region-scans"; }
  StringRef getDescription() const final { return "Realize GPU region scans"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizeRegionScans(module))))
      signalPassFailure();
  }
};

class ReductionsPass : public PassWrapper<ReductionsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReductionsPass)
  StringRef getArgument() const final { return "intent-gpu-realize-reductions"; }
  StringRef getDescription() const final { return "Realize GPU reduction programs"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizeReductionGroup(module))))
      signalPassFailure();
  }
};

class ContractionsPass : public PassWrapper<ContractionsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ContractionsPass)
  StringRef getArgument() const final { return "intent-gpu-realize-contractions"; }
  StringRef getDescription() const final { return "Realize GPU contraction programs"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), realizeContractionBlocking(module))))
      signalPassFailure();
  }
};

class AccessCompositionPass : public PassWrapper<AccessCompositionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AccessCompositionPass)
  StringRef getArgument() const final { return "intent-gpu-compose-realized-accesses"; }
  StringRef getDescription() const final { return "Compose realized GPU accesses and values"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), composeRealizedAccesses(module))))
      signalPassFailure();
  }
};

class PrivateStoresPass : public PassWrapper<PrivateStoresPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrivateStoresPass)
  StringRef getArgument() const final { return "intent-gpu-schedule-private-stores"; }
  StringRef getDescription() const final { return "Schedule private GPU stores"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), schedulePrivateStores(module))))
      signalPassFailure();
  }
};

class BufferVectorizationPass : public PassWrapper<BufferVectorizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BufferVectorizationPass)
  StringRef getArgument() const final { return "intent-gpu-vectorize-buffer-loops"; }
  StringRef getDescription() const final { return "Vectorize GPU buffer loops"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), vectorizeBufferLoops(module))))
      signalPassFailure();
  }
};

class BufferPromotionPass : public PassWrapper<BufferPromotionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BufferPromotionPass)
  StringRef getArgument() const final { return "intent-gpu-promote-buffer-values"; }
  StringRef getDescription() const final { return "Promote GPU buffer values"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), promoteBufferValues(module))))
      signalPassFailure();
  }
};

class ProgramMappingPass : public PassWrapper<ProgramMappingPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProgramMappingPass)
  StringRef getArgument() const final { return "intent-gpu-refine-program-mapping"; }
  StringRef getDescription() const final { return "Refine GPU program mapping"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), refineProgramMapping(module))))
      signalPassFailure();
  }
};

class CommonValuesPass : public PassWrapper<CommonValuesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommonValuesPass)
  StringRef getArgument() const final { return "intent-gpu-eliminate-common-values"; }
  StringRef getDescription() const final { return "Eliminate common GPU values and simplify masked coordinates"; }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishTransformation(module, getArgument(), simplifyValues(module))))
      signalPassFailure();
  }
};

void populateTransformations(OpPassManager &manager) {
  manager.addPass(std::make_unique<NormalizeStructuredSourcesPass>());
  manager.addPass(std::make_unique<PredicateScalarControlPass>());
  manager.addPass(std::make_unique<PointwiseOwnershipPass>());
  manager.addPass(std::make_unique<PointwiseBlockingPass>());
  manager.addPass(std::make_unique<ScanConsumersPass>());
  manager.addPass(std::make_unique<RetainedValuesPass>());
  manager.addPass(std::make_unique<OnlineReductionsPass>());
  manager.addPass(std::make_unique<RegionFoldsPass>());
  manager.addPass(std::make_unique<RegionScansPass>());
  manager.addPass(std::make_unique<ReductionsPass>());
  manager.addPass(std::make_unique<ContractionsPass>());
  manager.addPass(std::make_unique<AccessCompositionPass>());
  manager.addPass(std::make_unique<PrivateStoresPass>());
  manager.addPass(std::make_unique<BufferVectorizationPass>());
  manager.addPass(std::make_unique<BufferPromotionPass>());
  manager.addPass(std::make_unique<ProgramMappingPass>());
  manager.addPass(std::make_unique<CommonValuesPass>());
}

scf::IfOp independentUniformBranches(func::FuncOp kernel) {
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!space || space.size() != 1 ||
      constantPhysicalExpression(cast<PhysicalExprAttr>(space[0])) != 1)
    return {};
  auto conditional = dyn_cast_or_null<scf::IfOp>(
      kernel.front().getTerminator()->getPrevNode());
  if (!conditional || conditional.getElseRegion().empty() ||
      !conditional->use_empty() ||
      !isLaunchUniformScalar(conditional.getCondition(), kernel))
    return {};
  // Branch-local tuning parameters are renamed and bound independently.
  // The dispatch condition must retain one shared ABI meaning.
  SmallVector<Value> conditions{conditional.getCondition()};
  llvm::SmallPtrSet<Operation *, 16> visited;
  bool referencesParameter = false;
  AttrTypeWalker expressions;
  expressions.addWalk([&](PhysicalExprAttr expression) {
    referencesParameter |= expression.getKind() ==
                           static_cast<uint32_t>(PhysicalExprKind::Parameter);
  });
  while (!conditions.empty()) {
    Operation *producer = conditions.pop_back_val().getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (isa<ParameterOp>(producer))
      return {};
    expressions.walk(producer->getAttrDictionary());
    if (referencesParameter)
      return {};
    llvm::append_range(conditions, producer->getOperands());
  }
  unsigned mappings = 0;
  bool hasStores[2] = {false, false};
  WalkResult eligible = kernel.walk([&](Operation *operation) {
    if (operation == kernel || operation == conditional)
      return WalkResult::advance();
    if (isa<DelinearizeOp>(operation))
      ++mappings;
    if (auto store = dyn_cast<StoreOp>(operation)) {
      if (!isa<ViewType>(store.getResource().getType()))
        return WalkResult::interrupt();
      for (unsigned index = 0; index < 2; ++index)
        if (conditional->getRegion(index).isAncestor(store->getParentRegion())) {
          hasStores[index] = true;
          return WalkResult::advance();
        }
      return WalkResult::interrupt();
    }
    if (isa<scf::IfOp, scf::ForOp, scf::WhileOp, ContractOp,
            ScaledContractOp, SparseContractOp, RegionFoldOp,
            RegionScanOp>(operation))
      return WalkResult::interrupt();
    if (auto load = dyn_cast<LoadOp>(operation)) {
      auto view = dyn_cast<ViewType>(load.getResource().getType());
      return view && view.getAccess() == 0 ? WalkResult::advance()
                                         : WalkResult::interrupt();
    }
    return isMemoryEffectFree(operation) ? WalkResult::advance()
                                        : WalkResult::interrupt();
  });
  if (eligible.wasInterrupted() || mappings != 1 || !hasStores[0] || !hasStores[1])
    return {};
  SmallVector<PhysicalSourceAxis> reducedAxes[2];
  for (unsigned index = 0; index < 2; ++index)
    conditional->getRegion(index).walk([&](ReduceOp reduction) {
      for (Value source :
           reduction.getInputs().take_front(reduction.getSourceCount())) {
        auto type = cast<FragmentType>(source.getType());
        for (int64_t axis : reduction.getAxes()) {
          auto sourceAxis =
              sourceAxisIdentity(cast<AxisMapAttr>(type.getAxisMaps()[axis]));
          if (!llvm::is_contained(reducedAxes[index], sourceAxis))
            reducedAxes[index].push_back(sourceAxis);
        }
      }
    });
  bool sameAxes =
      reducedAxes[0].size() == reducedAxes[1].size() &&
      llvm::all_of(reducedAxes[0], [&](PhysicalSourceAxis axis) {
        return llvm::is_contained(reducedAxes[1], axis);
      });
  return sameAxes ? scf::IfOp() : conditional;
}

FailureOr<bool> realizeUniformBranches(ModuleOp module, func::FuncOp kernel,
                                       scf::IfOp conditional, PassManager &manager) {
  // Optimize each mutually exclusive region with the existing kernel passes,
  // then rejoin their physical programs under one launch and the original ABI.
  SmallVector<OwningOpRef<ModuleOp>> branches;
  for (unsigned index = 0; index < 2; ++index) {
    IRMapping mapping;
    OwningOpRef<ModuleOp> branch(cast<ModuleOp>(module->clone(mapping)));
    // These temporary modules are separate pass-manager runs. A diagnostic
    // symbol keeps MLIR's native print-tree files distinct for each branch.
    (*branch)->setAttr(SymbolTable::getSymbolAttrName(),
        StringAttr::get(module.getContext(), "intent_uniform_branch_" + std::to_string(index)));
    auto function = cast<func::FuncOp>(mapping.lookup(kernel.getOperation()));
    auto choice = cast<scf::IfOp>(mapping.lookup(conditional.getOperation()));
    Block &body = choice->getRegion(index).front();
    for (Operation &operation :
         llvm::make_early_inc_range(body.without_terminator()))
      operation.moveBefore(choice);
    choice.erase();
    SmallVector<Attribute> effects;
    function.walk([&](StoreOp store) {
      effects.push_back(store->getAttr(originAttr));
    });
    function->setAttr(effectOriginsAttr,
                      ArrayAttr::get(module.getContext(), effects));
    eraseDeadPhysicalValues(function);
    if (failed(manager.run(*branch)))
      return failure();
    eraseUnusedPhysicalParameters(function);
    auto space = function->getAttrOfType<ArrayAttr>(programSpaceAttr);
    bool separateResources =
        function.getFunctionType() != kernel.getFunctionType();
    function.walk([&](BufferOp) { separateResources = true; });
    function.walk([&](DelinearizeOp mapping) {
      separateResources |=
          space.size() != 1 ||
          mapping->getAttr(segmentLengthAttr) != space[0];
    });
    if (separateResources)
      return false;

    // Branch-local tuning symbols remain distinct after the region programs
    // are rejoined. External ABI identities stay shared.
    auto qualify = [&](StringAttr name) {
      return StringAttr::get(
          module.getContext(),
          "G" + std::to_string(index) + "_" + name.getValue().str());
    };
    AttrTypeReplacer replacer;
    replacer.addReplacement([&](ParameterAttr parameter)
                                -> std::optional<Attribute> {
      return ParameterAttr::get(
          module.getContext(), qualify(parameter.getName()), parameter.getRole(),
          parameter.getCategory(), parameter.getElementBitWidth(),
          parameter.getCandidates());
    });
    replacer.addReplacement([&](PhysicalExprAttr expression)
                                -> std::optional<Attribute> {
      if (expression.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Parameter))
        return std::nullopt;
      return PhysicalExprAttr::get(
          module.getContext(), expression.getKind(), expression.getValue(),
          qualify(expression.getSymbol()), expression.getOperands());
    });
    replacer.recursivelyReplaceElementsIn(function, true, true, true);
    branches.push_back(std::move(branch));
  }

  auto combined = cast<func::FuncOp>(kernel->cloneWithoutRegions());
  module.getBody()->push_back(combined);
  Block *entry = combined.addEntryBlock();
  OpBuilder builder(entry, entry->begin());
  IRMapping conditionMapping;
  conditionMapping.map(kernel.getArguments(), combined.getArguments());
  std::function<Value(Value)> cloneCondition = [&](Value value) -> Value {
    if (Value mapped = conditionMapping.lookupOrNull(value))
      return mapped;
    Operation *producer = value.getDefiningOp();
    for (Value operand : producer->getOperands())
      cloneCondition(operand);
    builder.clone(*producer, conditionMapping);
    return conditionMapping.lookup(value);
  };
  Value condition = cloneCondition(conditional.getCondition());
  Location location = conditional.getLoc();
  Value program = builder.create<ProgramIdOp>(location, builder.getIndexType(), 0);
  auto expression = [&](PhysicalExprKind kind, int64_t value,
                        ArrayRef<Attribute> operands = {}) {
    return PhysicalExprAttr::get(module.getContext(), static_cast<uint32_t>(kind),
        value, builder.getStringAttr(""), builder.getArrayAttr(operands));
  };
  PhysicalExprAttr zero = expression(PhysicalExprKind::Constant, 0);
  PhysicalExprAttr offset = zero;
  PhysicalExprAttr launchCondition = queryLaunchExpression(condition);
  for (auto [index, branch] : llvm::enumerate(branches)) {
    auto function = *getPhysicalKernel(*branch);
    auto space = function->getAttrOfType<ArrayAttr>(programSpaceAttr);
    auto length = cast<PhysicalExprAttr>(space[0]);
    if (launchCondition)
      length = expression(PhysicalExprKind::Select, 0,
          {launchCondition, index == 0 ? length : zero,
           index == 0 ? zero : length});
    IRMapping mapping;
    mapping.map(function.getArguments(), combined.getArguments());
    function.walk([&](ParameterOp parameter) {
      builder.clone(*parameter, mapping);
    });
    Value begin = builder.create<PhysicalExprOp>(
        location, builder.getIndexType(), offset);
    Value count = builder.create<PhysicalExprOp>(
        location, builder.getIndexType(), length);
    Value end = builder.create<BinaryOp>(
        location, builder.getIndexType(), begin, count, BinaryOperator::Add);
    Value lower = builder.create<CompareOp>(
        location, builder.getI1Type(), program, begin, ComparePredicate::Ge);
    Value upper = builder.create<CompareOp>(
        location, builder.getI1Type(), program, end, ComparePredicate::Lt);
    Value active = builder.create<BinaryOp>(
        location, builder.getI1Type(), lower, upper, BinaryOperator::LogicalAnd);
    Value selected = condition;
    if (index == 1)
      selected = builder.create<UnaryOp>(
          location, builder.getI1Type(), condition, UnaryOperator::Not);
    active = builder.create<BinaryOp>(
        location, builder.getI1Type(), active, selected, BinaryOperator::LogicalAnd);
    auto dispatch = builder.create<scf::IfOp>(location, active, false);
    dispatch->setAttr(executionGroupAttr, builder.getI64IntegerAttr(index));
    dispatch->setAttr(segmentOffsetAttr, offset);
    dispatch->setAttr(segmentLengthAttr, length);
    OpBuilder nested = OpBuilder::atBlockBegin(&dispatch.getThenRegion().front());
    Value local = nested.create<BinaryOp>(
        location, builder.getIndexType(), program, begin, BinaryOperator::Subtract);
    for (Operation &operation : function.front().without_terminator()) {
      if (isa<ParameterOp>(operation))
        continue;
      if (auto id = dyn_cast<ProgramIdOp>(operation)) {
        mapping.map(id.getResult(), local);
        continue;
      }
      Operation *cloned = nested.clone(operation, mapping);
      cloned->walk([&](Operation *operation) {
        if (operation->hasAttr(executionGroupAttr)) {
          operation->setAttr(executionGroupAttr, builder.getI64IntegerAttr(index));
          operation->setAttr(segmentOffsetAttr, offset);
          operation->setAttr(segmentLengthAttr, length);
        }
      });
    }
    offset = index == 0 ? length
                        : expression(PhysicalExprKind::Add, 0, {offset, length});
  }
  builder.create<func::ReturnOp>(location);
  combined->setAttr(programSpaceAttr, builder.getArrayAttr({offset}));
  combined->setAttr(gridRankAttr, builder.getI64IntegerAttr(1));
  kernel.erase();
  if (failed(verifyGPUProgram(module)))
    return failure();
  return true;
}

} // namespace

void registerGPUPasses() {
  PassRegistration<NormalizeStructuredSourcesPass>();
  PassRegistration<PredicateScalarControlPass>();
  PassRegistration<PointwiseOwnershipPass>();
  PassRegistration<PointwiseBlockingPass>();
  PassRegistration<ScanConsumersPass>();
  PassRegistration<RetainedValuesPass>();
  PassRegistration<OnlineReductionsPass>();
  PassRegistration<RegionFoldsPass>();
  PassRegistration<RegionScansPass>();
  PassRegistration<ReductionsPass>();
  PassRegistration<ContractionsPass>();
  PassRegistration<AccessCompositionPass>();
  PassRegistration<PrivateStoresPass>();
  PassRegistration<BufferVectorizationPass>();
  PassRegistration<BufferPromotionPass>();
  PassRegistration<ProgramMappingPass>();
  PassRegistration<CommonValuesPass>();
}

LogicalResult completeGPUProgramConstruction(ModuleOp module) {
  // Construction closes the initial indexed access graph. Ownership/blocking
  // and structured realization are subsequent transformations of this program.
  if (failed(realizeAccessComposition(module)))
    return failure();
  return verifyGPUProgram(module);
}

LogicalResult runSharedGPUPasses(ModuleOp module, const TuningProfiles &profiles) {
  if (failed(verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  PassManager manager(module.getContext(), ModuleOp::getOperationName());
  populateTransformations(manager);
  if (failed(intent::configurePassManager(manager))) return failure();
  bool realizedBranches = false;
  if (scf::IfOp conditional = independentUniformBranches(*kernel)) {
    auto realized = realizeUniformBranches(module, *kernel, conditional, manager);
    if (failed(realized))
      return failure();
    realizedBranches = *realized;
    kernel = getPhysicalKernel(module);
  }
  if (!realizedBranches && failed(manager.run(module))) {
    return failure();
  }
  if (failed(closeSharedConfigurations(*kernel, profiles)) ||
      failed(fuseIndependentTraversals(module)) ||
      failed(verifySharedConfigTuples(*kernel)))
    return failure();
  return verifyGPUProgram(module);
}

} // namespace intent::gpu
