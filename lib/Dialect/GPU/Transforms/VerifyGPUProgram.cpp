#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

LogicalResult verifyExpressionSymbols(Operation *owner,
                                      PhysicalExprAttr expression,
                                      const llvm::StringSet<> &parameters) {
  auto kind = expression.getKind();
  if (kind == PhysicalExprKind::Parameter &&
      !parameters.contains(expression.getParameterReference().getName().getValue()))
    return owner->emitOpError("launch expression references an undeclared physical parameter");
  for (Attribute operand : expression.getOperands())
    if (failed(verifyExpressionSymbols(owner, cast<PhysicalExprAttr>(operand),
                                       parameters)))
      return failure();
  return success();
}

void collectTypeExpressions(Type type,
                            SmallVectorImpl<PhysicalExprAttr> &expressions) {
  ArrayAttr shape;
  if (auto view = dyn_cast<ViewType>(type))
    shape = view.getLayout().getExtents();
  else if (auto fragment = dyn_cast<FragmentType>(type))
    shape = fragment.getShape();
  else if (auto buffer = dyn_cast<BufferType>(type))
    shape = buffer.getShape();
  if (!shape)
    return;
  for (Attribute extent : shape)
    expressions.push_back(cast<PhysicalExprAttr>(extent));
}

bool hasObservableEffect(Operation *operation) {
  if (auto store = dyn_cast<StoreOp>(operation))
    if (auto buffer = dyn_cast<BufferType>(store.getResource().getType());
        buffer && buffer.getWorkspace() && !store->hasAttr(originAttr))
      return false;
  auto access = dyn_cast<AccessOpInterface>(operation);
  return access && access.writesMemory();
}

bool mutuallyExclusiveEffects(Operation *lhs, Operation *rhs) {
  // A versioned operation may occur once in each exclusive branch, but never
  // twice on the same dynamic control path.
  for (Region *region = lhs->getParentRegion(); region;
       region = region->getParentRegion()) {
    auto branch = dyn_cast_or_null<scf::IfOp>(region->getParentOp());
    if (!branch || branch.getElseRegion().empty())
      continue;
    Region &other = region == &branch.getThenRegion()
                        ? branch.getElseRegion() : branch.getThenRegion();
    if (other.isAncestor(rhs->getParentRegion()))
      return true;
  }
  return false;
}

bool requiresRangeProvenance(Value coordinate) {
  auto fragment = dyn_cast<FragmentType>(coordinate.getType());
  if (!fragment)
    return false;
  return llvm::any_of(fragment.getShape(), [](Attribute extent) {
    auto expression = cast<PhysicalExprAttr>(extent);
    return expression.getKind() !=
               PhysicalExprKind::Constant ||
           expression.getValue() != 1;
  });
}

LogicalResult verifyStructuredSegment(Operation *operation,
                                      func::FuncOp kernel,
                                      ParameterRefAttr segment) {
  FailureOr<ParameterAttr> declaration =
      queryParameterBySymbol(kernel, segment.getName());
  if (failed(declaration))
    return operation->emitOpError(
        "structured segment does not reference one exact physical parameter declaration");
  return declaration->getRole() ==
                 ParameterRole::ScanChunk
             ? success()
             : operation->emitOpError(
                   "structured segment parameter has the wrong physical role");
}

LogicalResult verifyBufferResources(func::FuncOp kernel,
                                    PhysicalProgramAnalysis &analysis) {
  auto verifyUses = [](Value buffer) -> LogicalResult {
    for (OpOperand &use : buffer.getUses()) {
      Operation *user = use.getOwner();
      if (isa<AssumeInBoundsOp, DimOp>(user))
        continue;
      auto access = dyn_cast<AccessOpInterface>(user);
      if (!access || !access.isMemoryAccess() ||
          &use != &access.getAccessResourceOperand())
        return user->emitOpError("physical buffer use is not an explicit resource access");
    }
    return success();
  };
  llvm::DenseSet<uint64_t> instances;
  LogicalResult result = success();
  kernel.walk([&](BufferOp buffer) {
    if (failed(result))
      return WalkResult::interrupt();
    BufferType type = buffer.getResult().getType();
    if (!instances.insert(type.getInstance()).second) {
      buffer.emitOpError("physical buffer instance identity is duplicated");
      result = failure();
      return WalkResult::interrupt();
    }
    bool lifetimeMatches =
        (type.getScope().getValue() == BufferScope::ProgramPrivate &&
         type.getLifetime().getValue() == BufferLifetime::Program) ||
        (type.getScope().getValue() == BufferScope::IterationPrivate &&
         type.getLifetime().getValue() == BufferLifetime::Iteration);
    if (!lifetimeMatches) {
      buffer.emitOpError(
          "physical buffer allocation scope and dynamic lifetime disagree");
      result = failure();
      return WalkResult::interrupt();
    }
    if (failed(verifyUses(buffer.getResult()))) {
      result = failure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (failed(result))
    return failure();
  for (BlockArgument argument : kernel.getArguments()) {
    auto buffer = dyn_cast<BufferType>(argument.getType());
    if (!buffer)
      continue;
    if (!buffer.getWorkspace() ||
        buffer.getScope().getValue() != BufferScope::InvocationWorkspace ||
        buffer.getLifetime().getValue() != BufferLifetime::Invocation ||
        buffer.getInitialization().getValue() != BufferInitialization::FirstWrite ||
        !instances.insert(buffer.getInstance()).second)
      return kernel.emitError(
          "workspace requires a unique invocation allocation and explicit first writes");
    auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
    if (!llvm::all_of(space, [](Attribute extent) {
          auto expression = cast<PhysicalExprAttr>(extent);
          return expression.getKind() ==
                     PhysicalExprKind::Constant &&
                 expression.getValue() == 1;
        }) && !analysis.hasDisjointWorkspaceSlices(argument))
      return kernel.emitError(
          "workspace accesses have no proven disjoint program slices");
    if (failed(verifyUses(argument)))
      return failure();
  }
  return result;
}

} // namespace

LogicalResult verifyGPUProgram(ModuleOp module) {
  if (failed(mlir::verify(module.getOperation())))
    return failure();
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  if (!kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr) ||
      !kernel->getAttrOfType<ArrayAttr>(programSpaceAttr) ||
      !kernel->getAttrOfType<IntegerAttr>(gridRankAttr))
    return kernel.emitError("physical kernel is missing capabilities or launch schema");
  auto programSpace = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (Attribute attribute = kernel->getAttr(configurationsAttr)) {
    auto configurations = dyn_cast<ConfigurationSetAttr>(attribute);
    if (!configurations)
      return kernel.emitError("candidate bindings require a typed configuration set");
    auto space = ParameterSpace::read(kernel);
    if (failed(space) ||
        failed(space->configurations(configurations.getStage())))
      return failure();
  }
  auto expectedEffects = kernel->getAttrOfType<ArrayAttr>(effectOriginsAttr);
  int64_t gridRank = kernel->getAttrOfType<IntegerAttr>(gridRankAttr).getInt();
  if (gridRank <= 0 || programSpace.size() != static_cast<size_t>(gridRank) ||
      !expectedEffects)
    return kernel.emitError("physical program-space rank is invalid");
  for (Attribute extent : programSpace)
    if (!isa<PhysicalExprAttr>(extent))
      return kernel.emitError("program-space extents must be typed physical expressions");

  llvm::StringSet<> parameterNames;
  if (failed(verifyProgramInterface(kernel))) return failure();
  SmallVector<PhysicalExprAttr> abiExpressions;
  for (Type type : kernel.getArgumentTypes())
    collectTypeExpressions(type, abiExpressions);
  auto declarations = ParameterSpace::read(kernel);
  if (failed(declarations))
    return failure();
  for (ParameterAttr parameter : declarations->declarations()) {
    parameterNames.insert(parameter.getName().getValue());
  }
  for (Attribute extent : programSpace)
    if (failed(verifyExpressionSymbols(kernel, cast<PhysicalExprAttr>(extent),
                                       parameterNames)))
      return failure();
  for (PhysicalExprAttr expression : abiExpressions)
    if (failed(verifyExpressionSymbols(kernel, expression, parameterNames)))
      return failure();

  bool hasProgramId = false;
  llvm::DenseSet<int64_t> expectedEffectOrigins;
  for (Attribute origin : expectedEffects) {
    auto node = dyn_cast<IntegerAttr>(origin);
    if (!node || node.getInt() < 0 ||
        !expectedEffectOrigins.insert(node.getInt()).second)
      return kernel.emitError(
          "expected effect origins must be unique non-negative IDs");
  }
  llvm::DenseSet<int64_t> actualEffectOrigins;
  llvm::DenseMap<int64_t, SmallVector<Operation *>> effectDefinitions;
  llvm::DenseSet<int64_t> programAxes;
  llvm::DenseMap<int64_t, std::pair<PhysicalExprAttr, PhysicalExprAttr>>
      executionGroups;
  PhysicalProgramAnalysis physicalAnalysis(kernel);
  WalkResult result = kernel.walk([&](Operation *operation) {
    StringRef dialect = operation->getName().getDialectNamespace();
    if (dialect == "intent") {
      operation->emitOpError("canonical KIR is illegal inside a physical GPU kernel");
      return WalkResult::interrupt();
    }
    if (dialect != "intent_gpu" && dialect != "arith" && dialect != "scf" &&
        dialect != "func" && dialect != "builtin") {
      operation->emitOpError("operation dialect is not legal in shared GPU IR");
      return WalkResult::interrupt();
    }
    if (auto physical = dyn_cast<PhysicalExprOp>(operation))
      if (failed(verifyExpressionSymbols(operation, physical.getExpression(),
                                         parameterNames)))
        return WalkResult::interrupt();
    if (operation->hasAttr(sourceSubregionAttr)) {
      auto parent =
          operation->getAttrOfType<IntegerAttr>(sourceSubregionAttr);
      if (!parent || parent.getInt() <= 0 ||
          !isa<RangeOp, MakeRangeOp>(operation)) {
        operation->emitOpError(
            "physical subregion requires a typed coordinate range and parent identity");
        return WalkResult::interrupt();
      }
    }
    if (operation->hasAttr(sourceSubregionBoundAttr)) {
      auto bound =
          operation->getAttrOfType<IntegerAttr>(sourceSubregionBoundAttr);
      if (!operation->hasAttr(sourceSubregionAttr) || !bound ||
          bound.getInt() <= 0) {
        operation->emitOpError(
            "physical subregion bound requires a positive typed subregion relation");
        return WalkResult::interrupt();
      }
    }
    if (auto program = dyn_cast<ProgramIdOp>(operation)) {
      if (program.getAxis() >= static_cast<uint64_t>(gridRank) ||
          !programAxes.insert(program.getAxis()).second) {
        operation->emitOpError("program coordinate is outside or duplicated in the current grid");
        return WalkResult::interrupt();
      }
      hasProgramId = true;
    }
    Attribute groupAttribute = operation->getAttr(executionGroupAttr);
    Attribute offsetAttribute = operation->getAttr(segmentOffsetAttr);
    Attribute lengthAttribute = operation->getAttr(segmentLengthAttr);
    if (groupAttribute || offsetAttribute || lengthAttribute) {
      auto group = dyn_cast_or_null<IntegerAttr>(groupAttribute);
      auto offset = dyn_cast_or_null<PhysicalExprAttr>(offsetAttribute);
      auto length = dyn_cast_or_null<PhysicalExprAttr>(lengthAttribute);
      if (!group || group.getInt() < 0 || !offset || !length) {
        operation->emitOpError(
            "physical execution segment requires typed group, offset and length together");
        return WalkResult::interrupt();
      }
      if (failed(verifyExpressionSymbols(operation, offset, parameterNames)) ||
          failed(verifyExpressionSymbols(operation, length, parameterNames)))
        return WalkResult::interrupt();
      auto [entry, inserted] = executionGroups.try_emplace(
          group.getInt(), std::make_pair(offset, length));
      if (!inserted && entry->second != std::make_pair(offset, length)) {
        operation->emitOpError(
            "one physical execution group has conflicting segment bounds");
        return WalkResult::interrupt();
      }
    }
    if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
      if (failed(verifyStructuredSegment(operation, kernel, fold.getSegment())))
        return WalkResult::interrupt();
    } else if (auto scan = dyn_cast<RegionScanOp>(operation)) {
      if (failed(verifyStructuredSegment(operation, kernel, scan.getSegment())))
        return WalkResult::interrupt();
    }
    if (operation->hasAttr(independentIterationAttr)) {
      auto loop = dyn_cast<scf::ForOp>(operation);
      if (!loop || loop.getNumRegionIterArgs() ||
          !isa<UnitAttr>(operation->getAttr(independentIterationAttr))) {
        operation->emitOpError(
            "independent iteration requires an explicit loop without carried state");
        return WalkResult::interrupt();
      }
    }
    if (isa<AccessOpInterface>(operation)) {
      PhysicalAccessFootprint footprint = physicalAnalysis.footprint(operation);
      if (footprint.state != PhysicalFactState::Exact) {
        operation->emitOpError(
            "physical access has no exact current-IR footprint");
        return WalkResult::interrupt();
      }
      bool needsRanges = llvm::any_of(footprint.coordinates,
                                      requiresRangeProvenance);
      if (needsRanges && footprint.rangeState != PhysicalFactState::Exact) {
        InFlightDiagnostic diagnostic = operation->emitOpError(
            "non-scalar physical access has unknown range provenance");
        for (Operation *blocker : footprint.blockers)
          diagnostic << "; blocker=" << blocker->getName();
        return WalkResult::interrupt();
      }
      PhysicalAccessBoundsFact bounds =
          physicalAnalysis.accessBounds(operation);
      if (!bounds.isExact()) {
        InFlightDiagnostic diagnostic = operation->emitOpError(
            "physical access is not proven within its resource bounds");
        for (int64_t axis : bounds.unprovenAxes)
          diagnostic << "; unproven_axis=" << axis;
        for (int64_t axis : bounds.missingLowerAxes)
          diagnostic << "; missing_lower_axis=" << axis;
        for (int64_t axis : bounds.missingUpperAxes)
          diagnostic << "; missing_upper_axis=" << axis;
        diagnostic << "; resource=" << footprint.resource.getType();
        for (auto [coordinate, sourceAxis] :
             llvm::zip(footprint.coordinates, footprint.sourceAxes))
          diagnostic << "; coordinate_axis=" << sourceAxis << ":"
                     << coordinate;
        if (footprint.validity)
          diagnostic << "; validity=" << footprint.validity;
        for (Operation *blocker : bounds.blockers)
          diagnostic << "; blocker=" << blocker->getName();
        return WalkResult::interrupt();
      }
    }
    if (hasObservableEffect(operation)) {
      auto origin = operation->getAttrOfType<IntegerAttr>(originAttr);
      if (!origin || !llvm::all_of(effectDefinitions[origin.getInt()],
                                   [&](Operation *previous) {
                                     return mutuallyExclusiveEffects(previous, operation);
                                   })) {
        operation->emitOpError(
            "observable effect requires one canonical origin per control path");
        return WalkResult::interrupt();
      }
      actualEffectOrigins.insert(origin.getInt());
      effectDefinitions[origin.getInt()].push_back(operation);
    }
    SmallVector<PhysicalExprAttr> expressions;
    for (Type type : operation->getOperandTypes())
      collectTypeExpressions(type, expressions);
    for (Type type : operation->getResultTypes())
      collectTypeExpressions(type, expressions);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          collectTypeExpressions(argument.getType(), expressions);
    for (PhysicalExprAttr expression : expressions)
      if (failed(verifyExpressionSymbols(operation, expression, parameterNames)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  for (int64_t group = 0;
       group < static_cast<int64_t>(executionGroups.size()); ++group)
    if (!executionGroups.contains(group))
      return kernel.emitError(
          "physical execution group identities must be dense from zero");
  if (!hasProgramId || programAxes.size() != static_cast<size_t>(gridRank) ||
      executionGroups.empty() ||
      actualEffectOrigins != expectedEffectOrigins) {
    InFlightDiagnostic diagnostic = kernel.emitError(
        "physical kernel program mapping/effect coverage is incomplete");
    diagnostic << "; has_program_id=" << hasProgramId
               << "; program_axes=" << programAxes.size()
               << "; grid_rank=" << gridRank
               << "; execution_groups=" << executionGroups.size()
               << "; expected_effect_origins=";
    for (int64_t origin : expectedEffectOrigins)
      diagnostic << origin << ",";
    diagnostic << "; actual_effect_origins=";
    for (int64_t origin : actualEffectOrigins)
      diagnostic << origin << ",";
    return failure();
  }
  return verifyBufferResources(kernel, physicalAnalysis);
}

} // namespace intent::gpu
