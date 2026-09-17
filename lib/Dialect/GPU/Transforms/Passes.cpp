#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <functional>

using namespace mlir;

namespace intent::gpu {
namespace {

// Each group owns its relation repairs. Analyses are recreated by its rewrites;
// the executable-program verifier runs only after the complete group.
LogicalResult closeReductionValueRelations(func::FuncOp kernel) {
  if (failed(alignReductionResultRelations(kernel)) ||
      failed(alignReductionIdentityRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignReductionYieldRelations(kernel)) ||
      failed(alignAggregateValueRelations(kernel)))
    return failure();
  return success();
}

LogicalResult normalizeStructuredSources(ModuleOp module, func::FuncOp kernel) {
  if (failed(composeContractResultReshapes(module)) ||
      failed(refreshReshapeRelations(kernel)))
    return failure();
  return success();
}

LogicalResult formPointwiseOwnership(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizePointwiseOwnership(module)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignContractValueRelations(kernel)) ||
      failed(closeReductionValueRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)))
    return failure();
  return alignPointwiseValueRelations(kernel);
}

LogicalResult predicateScalarControlGroup(ModuleOp module, func::FuncOp) {
  return predicateScalarControl(module);
}

LogicalResult formPointwiseBlocking(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizePointwiseBlocking(module)) ||
      failed(alignAccessResultRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(alignAggregateValueRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignAggregateValueRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignContractValueRelations(kernel)))
    return failure();
  // Independent workset axes can lift a vector dot into a matrix contraction.
  if (failed(realizeVectorContractions(module)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)))
    return failure();
  return closeReductionValueRelations(kernel);
}

LogicalResult coRealizeOnlineReductions(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeOnlineReductions(module)) ||
      failed(alignAggregateValueRelations(kernel)) ||
      failed(alignAccessResultRelations(kernel)) ||
      failed(alignReductionResultRelations(kernel)) ||
      failed(alignReductionIdentityRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignReductionYieldRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignContractValueRelations(kernel)))
    return failure();
  return success();
}

LogicalResult closeValueAccessRelations(func::FuncOp kernel) {
  if (failed(alignAccessResultRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignContractValueRelations(kernel)))
    return failure();
  return success();
}

LogicalResult realizeRegionFoldGroup(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeRegionFolds(module)))
    return failure();
  return closeValueAccessRelations(kernel);
}

LogicalResult realizeRegionScanGroup(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeRegionScans(module)))
    return failure();
  return closeValueAccessRelations(kernel);
}

LogicalResult realizeScanConsumerGroup(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeScanConsumerTraversals(module)))
    return failure();
  return closeValueAccessRelations(kernel);
}

LogicalResult materializeRetainedValueGroup(ModuleOp module, func::FuncOp kernel) {
  if (failed(materializeRetainedValues(module)))
    return failure();
  return closeValueAccessRelations(kernel);
}

LogicalResult realizeReductionGroup(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeReductionBlocking(module)) ||
      failed(alignAggregateValueRelations(kernel)) ||
      failed(alignAccessResultRelations(kernel)) ||
      failed(alignReductionResultRelations(kernel)) ||
      failed(alignReductionIdentityRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignReductionYieldRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(alignAggregateValueRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignContractValueRelations(kernel)))
    return failure();
  return success();
}

LogicalResult realizeContractionGroup(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeContractionBlocking(module)) ||
      failed(alignAggregateValueRelations(kernel)) ||
      failed(alignAccessResultRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignContractValueRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)))
    return failure();
  return success();
}

LogicalResult composeRealizedAccesses(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeAccessComposition(module)) ||
      failed(materializeRetainedValues(module)) ||
      failed(realizeAccessComposition(module)) ||
      failed(orientLoopContractions(module)) ||
      failed(alignAggregateValueRelations(kernel)))
    return failure();
  return closeValueAccessRelations(kernel);
}

LogicalResult refineMapping(ModuleOp module, func::FuncOp) {
  return refineProgramMapping(module);
}

LogicalResult vectorizeBufferGroup(ModuleOp module, func::FuncOp kernel) {
  if (failed(vectorizeBufferLoops(module)))
    return failure();
  return closeValueAccessRelations(kernel);
}

LogicalResult simplifyValues(ModuleOp module, func::FuncOp) {
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

struct TransformationGroup {
  StringRef name;
  LogicalResult (*run)(ModuleOp, func::FuncOp);
};

LogicalResult runTransformations(ModuleOp module, func::FuncOp kernel) {
  const TransformationGroup groups[] = {
      {"normalize-structured-sources", normalizeStructuredSources},
      {"predicate-scalar-control", predicateScalarControlGroup},
      {"form-pointwise-ownership", formPointwiseOwnership},
      {"form-pointwise-blocking", formPointwiseBlocking},
      {"realize-scan-consumers", realizeScanConsumerGroup},
      {"materialize-retained-values", materializeRetainedValueGroup},
      {"co-realize-online-reductions", coRealizeOnlineReductions},
      {"realize-region-folds", realizeRegionFoldGroup},
      {"realize-region-scans", realizeRegionScanGroup},
      {"realize-reductions", realizeReductionGroup},
      {"realize-contractions", realizeContractionGroup},
      {"compose-realized-accesses", composeRealizedAccesses},
      {"vectorize-buffer-loops", vectorizeBufferGroup},
      {"refine-program-mapping", refineMapping},
      {"eliminate-common-values", simplifyValues},
  };
  for (const TransformationGroup &group : groups) {
    if (failed(group.run(module, kernel))) {
      return module.emitError()
             << "shared GPU transformation failed: " << group.name;
    }
    if (failed(verifyGPUProgram(module))) {
      return module.emitError()
             << "shared GPU postcondition failed: " << group.name;
    }
  }
  return success();
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
                                       scf::IfOp conditional) {
  // Optimize each mutually exclusive region with the existing kernel passes,
  // then rejoin their physical programs under one launch and the original ABI.
  SmallVector<OwningOpRef<ModuleOp>> branches;
  for (unsigned index = 0; index < 2; ++index) {
    IRMapping mapping;
    OwningOpRef<ModuleOp> branch(cast<ModuleOp>(module->clone(mapping)));
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
    if (failed(runTransformations(*branch, function)))
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
  PhysicalExprAttr offset = expression(PhysicalExprKind::Constant, 0);
  for (auto [index, branch] : llvm::enumerate(branches)) {
    auto function = *getPhysicalKernel(*branch);
    auto space = function->getAttrOfType<ArrayAttr>(programSpaceAttr);
    auto length = cast<PhysicalExprAttr>(space[0]);
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

LogicalResult completeGPUProgramConstruction(ModuleOp module) {
  // Construction closes the initial indexed access graph. Ownership/blocking
  // and structured realization are subsequent transformations of this program.
  if (failed(realizeAccessComposition(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel) || failed(alignContractValueRelations(*kernel)) ||
      failed(alignAggregateValueRelations(*kernel)) ||
      failed(alignPointwiseValueRelations(*kernel)) ||
      failed(alignAccessValueRelations(*kernel)))
    return failure();
  return verifyGPUProgram(module);
}

LogicalResult runSharedGPUPasses(ModuleOp module, const TuningProfiles &profiles) {
  if (failed(verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  bool realizedBranches = false;
  if (scf::IfOp conditional = independentUniformBranches(*kernel)) {
    auto realized = realizeUniformBranches(module, *kernel, conditional);
    if (failed(realized))
      return failure();
    realizedBranches = *realized;
    kernel = getPhysicalKernel(module);
  }
  if (!realizedBranches && failed(runTransformations(module, *kernel))) {
    return failure();
  }
  if (failed(closeSharedConfigurations(*kernel, profiles)))
    return failure();
  return verifyGPUProgram(module);
}

} // namespace intent::gpu
