#include "PhysicalProgramDetail.h"
#include "ScalarExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"


using namespace mlir;

namespace intent::gpu {
using namespace detail;

namespace {

bool isExclusiveProgramRange(MakeRangeOp range, func::FuncOp kernel) {
  if (!range || integerConstant(range.getLogicalStart()) != 0 ||
      integerConstant(range.getStep()) != 1)
    return false;
  Value extent = stripScalarIdentity(range.getExtent());
  if (auto expression = extent.getDefiningOp<PhysicalExprOp>();
      expression && expression.getExpression().getKind() ==
          PhysicalExprKind::Parameter) {
    auto parameter = queryParameterBySymbol(kernel, expression.getExpression().getParameterReference().getName());
    if (failed(parameter))
      return false;
  }
  bool positive = integerConstant(extent).value_or(0) > 0;
  if (auto parameter = queryParameter(extent))
    positive = llvm::all_of(parameter.getCandidates().asArrayRef(),
                           [](int64_t value) { return value > 0; });
  auto multiply = stripScalarIdentity(range.getStart()).getDefiningOp<BinaryOp>();
  if (!positive || !multiply ||
      multiply.getOperatorKind() != BinaryOperator::Multiply)
    return false;
  Value coordinate;
  if (sameScalarExpression(multiply.getLhs(), extent))
    coordinate = stripScalarIdentity(multiply.getRhs());
  else if (sameScalarExpression(multiply.getRhs(), extent))
    coordinate = stripScalarIdentity(multiply.getLhs());
  auto argument = dyn_cast_or_null<BlockArgument>(coordinate);
  auto group = argument ? dyn_cast<ExecutionGroupOp>(argument.getOwner()->getParentOp())
                        : ExecutionGroupOp();
  auto decoded = coordinate ? queryDecodedCoordinate(coordinate) : std::nullopt;
  if (!decoded || decoded->extents.size() != 1)
    return false;
  auto program = decoded->linear.getDefiningOp<ProgramIdOp>();
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  auto launchExtent = group ? cast<PhysicalExprAttr>(group.getLaunchExtents()[0])
                           : queryLaunchExpression(decoded->extents[0]);
  if (!program || program.getAxis() != 0 || !space || space.empty() ||
      space[0] != launchExtent)
    return false;
  for (Attribute attribute : space.getValue().drop_front()) {
    auto expression = cast<PhysicalExprAttr>(attribute);
    if (expression.getKind() != PhysicalExprKind::Constant ||
        expression.getValue() != 1)
      return false;
  }
  auto launch = cast<PhysicalExprAttr>(space[0]);
  return launch.getKind() == PhysicalExprKind::CeilDiv &&
         launch.getOperands().size() == 2 &&
         launch.getOperands()[0] == queryLaunchExpression(range.getLogicalStop()) &&
         launch.getOperands()[1] == queryLaunchExpression(extent);
}

Value stripRangeProjection(Value value) {
  Value original = value;
  SmallVector<Operation *> projections;
  while (value) {
    if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
      projections.push_back(broadcast);
      value = broadcast.getValue();
      continue;
    }
    auto reshape = value.getDefiningOp<ReshapeOp>();
    if (!reshape)
      break;
    auto relations = queryFragmentOperandRelations(reshape.getOperation());
    if (failed(relations) || llvm::any_of(
            relations->front().groups, [](const FragmentAxisGroup &group) {
              return group.sourceAxes.size() > 1 || group.resultAxes.size() > 1;
            }))
      break;
    projections.push_back(reshape);
    value = reshape.getValue();
  }
  auto range = value.getDefiningOp<MakeRangeOp>();
  if (!range)
    return original;
  unsigned axis = 0;
  for (Operation *operation : llvm::reverse(projections)) {
    auto input = cast<FragmentType>(operation->getOperand(0).getType());
    auto output = cast<FragmentType>(operation->getResult(0).getType());
    std::optional<unsigned> projected;
    auto relations = queryFragmentOperandRelations(operation);
    if (failed(relations))
      return original;
    for (const FragmentAxisGroup &group : relations->front().groups)
      if (group.sourceAxes.size() == 1 && group.resultAxes.size() == 1 &&
          group.sourceAxes.front() == axis) {
        if (projected)
          return original;
        projected = group.resultAxes.front();
      }
    if (!projected || input.getShape()[axis] != output.getShape()[*projected])
      return original;
    axis = *projected;
  }
  return value;
}

bool isPrivateWorkspaceProgramIndex(Value coordinate, Value buffer,
                                    Operation *access, unsigned axis) {
  auto program = coordinate.getDefiningOp<ProgramIdOp>();
  auto kernel = access->getParentOfType<func::FuncOp>();
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!program || !space || axis >= space.size() || program.getAxis() != axis)
    return false;
  DominanceInfo dominance(kernel);
  return llvm::any_of(buffer.getUsers(), [&](Operation *user) {
    auto assumption = dyn_cast<AssumeInBoundsOp>(user);
    return assumption && assumption.getResource() == buffer &&
           assumption.getIndex() == coordinate &&
           assumption.getAxis() == axis &&
           dominance.properlyDominates(assumption, access);
  });
}

} // namespace

namespace detail {

bool isCoordinateReplayNode(Operation *operation) {
  return isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
             BroadcastOp, SplatOp, ReshapeOp, TransposeOp, JoinOp, DimOp,
             MakeRecordOp, ExtractOp, RandomBitsOp>(operation);
}

bool isValueReplayNode(Operation *operation) {
  return isCoordinateReplayNode(operation) ||
         isa<ContractOp, ScaledContractOp, SparseContractOp, ReduceOp, ScanOp>(
             operation);
}

bool isAccessNode(Operation *operation) {
  return isa<LoadOp, GatherOp>(operation);
}

void appendUnique(SmallVectorImpl<MakeRangeOp> &destination, MakeRangeOp range) {
  if (!llvm::is_contained(destination, range))
    destination.push_back(range);
}

void appendUnique(SmallVectorImpl<Operation *> &destination,
                  Operation *operation) {
  if (operation && !llvm::is_contained(destination, operation))
    destination.push_back(operation);
}

OpOperand *singleControlInput(Value target, ControlFlowEdgeKind kind,
                              Region *sourceRegion) {
  auto incoming = queryControlFlowIncoming(target);
  if (!incoming.complete) return nullptr;
  OpOperand *selected = nullptr;
  for (const ControlFlowEdge &edge : incoming.edges) {
    if (edge.kind != kind || edge.sourceRegion != sourceRegion) continue;
    if (!edge.operand || (selected && selected != edge.operand)) return nullptr;
    selected = edge.operand;
  }
  return selected;
}

} // namespace detail

bool isPhysicalReplayNode(Operation *operation, PhysicalReplayScope scope,
                          bool allowAccesses) {
  if (!operation || operation->getNumResults() == 0)
    return false;
  if (isAccessNode(operation))
    return allowAccesses && operation->getNumRegions() == 0;
  if (operation->getNumRegions() != 0 && !isa<ReduceOp, ScanOp>(operation))
    return false;
  return scope == PhysicalReplayScope::Coordinate
             ? isCoordinateReplayNode(operation)
             : isValueReplayNode(operation);
}

PhysicalProgramAnalysis::PhysicalProgramAnalysis(func::FuncOp kernel)
    : kernel(kernel) {}

bool PhysicalProgramAnalysis::carriesSource(Type type,
                                            PhysicalSourceAxis source) const {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
      auto mapping = cast<AxisMapAttr>(attribute);
      return mapping.getSourceId() == source.sourceId &&
             mapping.getSourceAxis() == source.sourceAxis &&
             mapping.getDerived() == source.derived;
    });
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
      return carriesSource(cast<TypeAttr>(field).getValue(), source);
    });
  return false;
}

SmallVector<Value, 2> PhysicalProgramAnalysis::structuredSourcesForArgument(
    BlockArgument argument) const {
  SmallVector<Value, 2> sources;
  Block *block = argument.getOwner();
  Operation *owner = block ? block->getParentOp() : nullptr;
  if (auto structured = dyn_cast_or_null<StructuredOpInterface>(owner)) {
    bool fold = structured.getStructuredKind() == StructuredOpKind::RegionFold;
    bool scan = structured.getStructuredKind() == StructuredOpKind::RegionScan;
    if (!fold && !scan) return sources;
    for (const auto &relation :
         structured.getRegionArgumentRelations(*block->getParent())) {
      if (relation.to != argument) continue;
      if (block->getParent() == structured.getSummarizeRegion() &&
          (relation.kind == StructuredRelationKind::SourceSlice ||
           (fold && relation.kind == StructuredRelationKind::Capture)))
        sources.push_back(relation.from);
    }
    if (fold && block->getParent() == &structured.getCombine()) {
      for (auto [identity, lhs, rhs, summary] : llvm::zip_equal(
               structured.getIdentities(), structured.getCombineLhs(),
               structured.getCombineRhs(), structured.getSummarizeYields())) {
        if (argument != lhs && argument != rhs) continue;
        sources.push_back(identity);
        sources.push_back(summary);
      }
    }
    return sources;
  }
  auto incoming = queryControlFlowIncoming(argument);
  if (!incoming.complete)
    return sources;
  for (const ControlFlowEdge &edge : incoming.edges)
    if (edge.operand && !llvm::is_contained(sources, edge.operand->get()))
      sources.push_back(edge.operand->get());
  return sources;
}

bool PhysicalProgramAnalysis::isProgramOwnedRange(MakeRangeOp range) const {
  return isExclusiveProgramRange(range, kernel);
}

bool PhysicalProgramAnalysis::hasDisjointWorkspaceSlices(Value buffer) const {
  auto type = dyn_cast<BufferType>(buffer.getType());
  if (!type)
    return false;
  SmallVector<AccessOpInterface> accesses;
  for (Operation *user : buffer.getUsers()) {
    if (isa<DimOp, AssumeInBoundsOp>(user))
      continue;
    auto access = dyn_cast<AccessOpInterface>(user);
    if (!access || (access.getAccessKind() != AccessKind::Load &&
                    access.getAccessKind() != AccessKind::Store))
      return false;
    accesses.push_back(access);
  }
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  // Every access must identify the whole program, including all grid axes.
  bool programPrefix = space && !space.empty() &&
                       type.getShape().size() >= space.size();
  for (AccessOpInterface access : accesses) {
    if (!programPrefix)
      break;
    ValueRange coordinates = access.getAccessCoordinates();
    ArrayRef<int64_t> sourceAxes = access.getAccessSourceAxes();
    for (unsigned axis = 0; axis < space.size(); ++axis) {
      auto position = llvm::find(sourceAxes, axis);
      if (position == sourceAxes.end() ||
          !isPrivateWorkspaceProgramIndex(
              coordinates[position - sourceAxes.begin()], buffer, access, axis)) {
        programPrefix = false;
        break;
      }
    }
  }
  if (programPrefix && !accesses.empty())
    return true;
  for (unsigned axis = 0; axis < type.getShape().size(); ++axis) {
    MakeRangeOp owner;
    bool consistent = true;
    for (AccessOpInterface access : accesses) {
      ValueRange coordinates = access.getAccessCoordinates();
      ArrayRef<int64_t> sourceAxes = access.getAccessSourceAxes();
      auto position = llvm::find(sourceAxes, axis);
      auto range = position == sourceAxes.end() ? MakeRangeOp() :
          stripRangeProjection(coordinates[position - sourceAxes.begin()])
              .getDefiningOp<MakeRangeOp>();
      if (!range || !isExclusiveProgramRange(range, kernel) ||
          queryLaunchExpression(range.getLogicalStop()) != type.getShape()[axis] ||
          (owner && (!sameLogicalRange(owner, range) ||
                     !sameScalarExpression(owner.getStart(), range.getStart()) ||
                     !sameScalarExpression(owner.getExtent(), range.getExtent())))) {
        consistent = false;
        break;
      }
      owner = range;
    }
    if (consistent && owner)
      return true;
  }
  return false;
}

} // namespace intent::gpu
