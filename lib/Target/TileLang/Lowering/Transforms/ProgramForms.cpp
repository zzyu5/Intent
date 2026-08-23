#include "Intent/Target/TileLang/Lowering/Passes.h"

#include "Syntax/Spelling.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/Common/Lowering/ProgramAnalysis.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Pass/Pass.h"

#include <functional>

using namespace mlir;

namespace intent::tilelang::lowering {
namespace {

FailureOr<std::pair<intent::plan::ProgramOp, intent::plan::SearchSpaceOp>>
getProgram(ModuleOp module) {
  SmallVector<intent::plan::ProgramOp> programs(
      module.getOps<intent::plan::ProgramOp>());
  SmallVector<intent::plan::SearchSpaceOp> searchSpaces(
      module.getOps<intent::plan::SearchSpaceOp>());
  if (programs.size() != 1 || searchSpaces.size() > 1) {
    module.emitError(
        "TileLang provider realization requires one physical program and at most one search space");
    return failure();
  }
  return std::make_pair(programs.front(),
                        searchSpaces.empty() ? intent::plan::SearchSpaceOp()
                                             : searchSpaces.front());
}

Operation *definingOperation(Value value) {
  auto result = dyn_cast<OpResult>(value);
  return result ? result.getOwner() : nullptr;
}

bool isSemanticOp(Operation *operation, StringRef name) {
  return operation && ::intent::target::semanticOperationName(*operation) == name;
}

std::optional<int64_t> integerConstant(Value value) {
  Operation *operation = definingOperation(value);
  if (!operation)
    return std::nullopt;
  if (isSemanticOp(operation, "intent.constant")) {
    auto attribute = operation->getAttrOfType<IntegerAttr>("intent.value");
    return attribute ? std::optional<int64_t>(attribute.getInt()) : std::nullopt;
  }
  StringRef name = ::intent::target::semanticOperationName(*operation);
  if ((name == "intent.broadcast" || name == "intent.cast" ||
       name == "intent.reshape") &&
      operation->getNumOperands() == 1)
    return integerConstant(operation->getOperand(0));
  return std::nullopt;
}

bool matchBinary(Value value, StringRef kind, Value &lhs, Value &rhs) {
  Operation *operation = definingOperation(value);
  auto binary = operation
                    ? operation->getAttrOfType<StringAttr>("intent.operator")
                    : StringAttr();
  if (!isSemanticOp(operation, "intent.binary") || !binary ||
      binary.getValue() != kind || operation->getNumOperands() != 2)
    return false;
  lhs = operation->getOperand(0);
  rhs = operation->getOperand(1);
  return true;
}

bool matchBinaryConstant(Value value, StringRef kind, int64_t constant,
                         Value &other) {
  Value lhs;
  Value rhs;
  if (!matchBinary(value, kind, lhs, rhs))
    return false;
  if (integerConstant(rhs) == constant) {
    other = lhs;
    return true;
  }
  if ((kind == "multiply" || kind == "add" || kind == "bitwise_and") &&
      integerConstant(lhs) == constant) {
    other = rhs;
    return true;
  }
  return false;
}

Value stripShapeProjection(Value value) {
  while (Operation *operation = definingOperation(value)) {
    StringRef name = ::intent::target::semanticOperationName(*operation);
    if ((name == "intent.broadcast" || name == "intent.reshape") &&
        operation->getNumOperands() == 1) {
      value = operation->getOperand(0);
      continue;
    }
    if (name == "intent.gather" && operation->getNumOperands() >= 1) {
      value = operation->getOperand(0);
      continue;
    }
    break;
  }
  return value;
}

struct PackedInt2DecodeMatch {
  Operation *load = nullptr;
  SmallVector<Operation *> producers;
};

std::optional<PackedInt2DecodeMatch>
matchPackedInt2Decode(Operation &contract) {
  if (contract.getNumOperands() != 2 || contract.getNumResults() != 1)
    return std::nullopt;
  auto lhsType = dyn_cast<RankedTensorType>(contract.getOperand(0).getType());
  auto rhsType = dyn_cast<RankedTensorType>(contract.getOperand(1).getType());
  auto resultType = dyn_cast<RankedTensorType>(contract.getResult(0).getType());
  if (!lhsType || !rhsType || !resultType || !lhsType.getElementType().isInteger(8) ||
      !rhsType.getElementType().isInteger(8) ||
      !resultType.getElementType().isInteger(32))
    return std::nullopt;

  Operation *decoded = definingOperation(contract.getOperand(1));
  if (!isSemanticOp(decoded, "intent.cast") || decoded->getNumOperands() != 1)
    return std::nullopt;
  Value shifted;
  Value mask;
  if (!matchBinary(decoded->getOperand(0), "bitwise_and", shifted, mask))
    return std::nullopt;
  if (integerConstant(mask) != 3) {
    if (integerConstant(shifted) != 3)
      return std::nullopt;
    std::swap(shifted, mask);
  }
  Value packed;
  Value shifts;
  if (!matchBinary(shifted, "right_shift", packed, shifts))
    return std::nullopt;
  Operation *load = definingOperation(packed);
  if (!isSemanticOp(load, "intent.view_load") || load->getNumOperands() < 2)
    return std::nullopt;
  auto packedView = dyn_cast<intent::ViewType>(load->getOperand(0).getType());
  auto packedTensor =
      packedView ? dyn_cast<RankedTensorType>(packedView.getTensor())
                 : RankedTensorType();
  if (!packedTensor || !packedTensor.getElementType().isUnsignedInteger(8))
    return std::nullopt;

  Value packedIndex;
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(*load);
  if (failed(relation))
    return std::nullopt;
  for (const target::IndexTerm &term : *relation)
    if (term.kind == "value_index" && term.operands.size() == 1 &&
        term.operands.front())
      packedIndex = load->getOperand(*term.operands.front());
  if (!packedIndex)
    return std::nullopt;

  Value packedLhs;
  Value packedRhs;
  if (!matchBinary(packedIndex, "add", packedLhs, packedRhs))
    return std::nullopt;
  auto matchPackedComponents = [&](Value scaled, Value lane,
                                   Value &base) -> bool {
    Value divided;
    Value scaledBase;
    Value laneBase;
    if (!matchBinaryConstant(scaled, "multiply", 4, divided) ||
        !matchBinaryConstant(divided, "floor_divide", 16, scaledBase) ||
        !matchBinaryConstant(lane, "remainder", 4, laneBase) ||
        scaledBase != laneBase)
      return false;
    base = scaledBase;
    return true;
  };
  Value base;
  if (!matchPackedComponents(packedLhs, packedRhs, base) &&
      !matchPackedComponents(packedRhs, packedLhs, base))
    return std::nullopt;

  shifts = stripShapeProjection(shifts);
  Operation *shiftCast = definingOperation(shifts);
  if (!isSemanticOp(shiftCast, "intent.cast") ||
      shiftCast->getNumOperands() != 1)
    return std::nullopt;
  Value shiftRemainder;
  if (!matchBinaryConstant(shiftCast->getOperand(0), "multiply", 2,
                           shiftRemainder))
    return std::nullopt;
  Value shiftDivided;
  Value shiftBase;
  if (!matchBinaryConstant(shiftRemainder, "remainder", 4, shiftDivided) ||
      !matchBinaryConstant(shiftDivided, "floor_divide", 4, shiftBase) ||
      shiftBase != base)
    return std::nullopt;
  Operation *indices = definingOperation(base);
  if (!isSemanticOp(indices, "intent.indices"))
    return std::nullopt;

  llvm::DenseSet<Operation *> producerSet;
  std::function<void(Value)> collect = [&](Value value) {
    Operation *operation = definingOperation(value);
    if (!operation || !producerSet.insert(operation).second)
      return;
    StringRef name = ::intent::target::semanticOperationName(*operation);
    if (name == "intent.view_load" || name == "intent.indices")
      return;
    for (Value operand : operation->getOperands())
      collect(operand);
  };
  collect(contract.getOperand(1));
  collect(packedIndex);
  for (Operation *producer : producerSet)
    for (Value result : producer->getResults())
      for (Operation *user : result.getUsers())
        if (user != &contract && !producerSet.contains(user))
          return std::nullopt;

  PackedInt2DecodeMatch match;
  match.load = load;
  match.producers.assign(producerSet.begin(), producerSet.end());
  llvm::sort(match.producers, [](Operation *lhs, Operation *rhs) {
    auto lhsNode = lhs->getAttrOfType<IntegerAttr>("intent.node");
    auto rhsNode = rhs->getAttrOfType<IntegerAttr>("intent.node");
    return lhsNode && rhsNode && lhsNode.getInt() < rhsNode.getInt();
  });
  return match;
}

intent::plan::AxisOp firstLane(gpu::PhysicalProgramAnalysis &analysis) {
  SmallVector<intent::plan::AxisOp> lanes;
  for (intent::plan::AxisOp axis :
       analysis.getProgram().getBody().getOps<intent::plan::AxisOp>())
    if (analysis.axisHasRole(axis.getNode(), "lane"))
      lanes.push_back(axis);
  llvm::sort(lanes, [](intent::plan::AxisOp lhs, intent::plan::AxisOp rhs) {
    return lhs.getNode() < rhs.getNode();
  });
  return lanes.empty() ? intent::plan::AxisOp() : lanes.front();
}

bool isRaggedBoundAxis(gpu::PhysicalProgramAnalysis &analysis, int64_t node) {
  Operation *domain = analysis.getKernel().nodes.lookup(node);
  if (!domain)
    return false;
  const target::KernelFacts &facts = analysis.getFacts();
  return facts.raggedMembers.count(domain) ||
         facts.raggedOuterRelations.count(domain);
}

struct SelectedContiguousIndexMatch {
  llvm::SmallVector<int64_t> scalarNodes;
};

struct SelectedIndexExpression {
  bool valid = false;
  bool constant = false;
  llvm::DenseSet<int64_t> scalarNodes;
  llvm::DenseSet<int64_t> varyingAxes;
};

bool isPhysicallyScalarTensor(Value value,
                              gpu::PhysicalProgramAnalysis &analysis) {
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  auto axes = analysis.getFacts().valueAxes.find(value);
  if (!tensor || axes == analysis.getFacts().valueAxes.end() ||
      axes->second.size() != static_cast<size_t>(tensor.getRank()))
    return false;
  for (const target::LogicalAxis &axis : axes->second) {
    if (axis.extent == "1")
      continue;
    auto node = axis.domain
                    ? axis.domain->getAttrOfType<IntegerAttr>("intent.node")
                    : IntegerAttr();
    if (!node)
      return false;
    bool singleElement = analysis.isScalarAxis(node.getInt());
    for (StringRef purpose : {StringRef("ownership"), StringRef("traversal"),
                              StringRef("reduction"), StringRef("lane")}) {
      intent::plan::RangeOp range = analysis.getRange(node.getInt(), purpose);
      singleElement |= range && range.getTile() == "fixed_1";
    }
    if (!singleElement)
      return false;
  }
  return true;
}

Operation *matchNativeContractOperandLoad(
    Operation &load, gpu::PhysicalProgramAnalysis &analysis) {
  if (!isSemanticOp(&load, "intent.view_load") || load.getNumResults() != 1 ||
      !llvm::hasSingleElement(load.getResult(0).getUsers()))
    return nullptr;
  Operation *reshape = *load.getResult(0).getUsers().begin();
  if (!isSemanticOp(reshape, "intent.reshape") ||
      reshape->getNumOperands() != 1 || reshape->getNumResults() != 1 ||
      !llvm::hasSingleElement(reshape->getResult(0).getUsers()))
    return nullptr;
  Operation *contract = *reshape->getResult(0).getUsers().begin();
  StringRef contractName = target::semanticOperationName(*contract);
  if (contractName != "intent.contract" &&
      contractName != "intent.scaled_contract" &&
      contractName != "intent.sparse_contract")
    return nullptr;

  auto sourceType = dyn_cast<RankedTensorType>(load.getResult(0).getType());
  auto targetType = dyn_cast<RankedTensorType>(reshape->getResult(0).getType());
  if (!sourceType || !targetType || targetType.getRank() != 2 ||
      sourceType.getRank() <= targetType.getRank() ||
      sourceType.getElementType() != targetType.getElementType())
    return nullptr;

  const target::KernelFacts &facts = analysis.getFacts();
  auto sourceAxes = facts.valueAxes.find(load.getResult(0));
  auto targetAxes = facts.valueAxes.find(reshape->getResult(0));
  if (sourceAxes == facts.valueAxes.end() || targetAxes == facts.valueAxes.end() ||
      sourceAxes->second.size() != static_cast<size_t>(sourceType.getRank()) ||
      targetAxes->second.size() != static_cast<size_t>(targetType.getRank()))
    return nullptr;

  SmallVector<target::LogicalAxis> nonUnitAxes;
  for (const target::LogicalAxis &axis : sourceAxes->second) {
    bool unit = axis.extent == "1";
    auto node = axis.domain
                    ? axis.domain->getAttrOfType<IntegerAttr>("intent.node")
                    : IntegerAttr();
    if (node) {
      unit |= analysis.isScalarAxis(node.getInt());
      for (StringRef purpose : {StringRef("ownership"), StringRef("traversal"),
                                StringRef("reduction"), StringRef("lane")}) {
        intent::plan::RangeOp range = analysis.getRange(node.getInt(), purpose);
        unit |= range && range.getTile() == "fixed_1";
      }
    }
    if (!unit)
      nonUnitAxes.push_back(axis);
  }
  return nonUnitAxes == targetAxes->second ? reshape : nullptr;
}

std::optional<SelectedContiguousIndexMatch>
matchSelectedContiguousIndex(Operation &operation,
                             gpu::PhysicalProgramAnalysis &analysis) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return std::nullopt;

  Value tensorIndex;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis")
      return std::nullopt;
    if (term.kind != "value_index" || term.operands.size() != 1 ||
        !term.operands.front())
      continue;
    Value indexed = operation.getOperand(*term.operands.front());
    if (!isa<RankedTensorType>(indexed.getType()))
      continue;
    if (tensorIndex)
      return std::nullopt;
    tensorIndex = indexed;
  }
  if (!tensorIndex)
    return std::nullopt;

  llvm::DenseSet<Value> active;
  std::function<SelectedIndexExpression(Value)> inspect =
      [&](Value value) -> SelectedIndexExpression {
    if (!active.insert(value).second)
      return {};
    auto finish = [&](SelectedIndexExpression result) {
      active.erase(value);
      return result;
    };
    Operation *definition = definingOperation(value);
    if (!definition)
      return finish({});
    StringRef name = ::intent::target::semanticOperationName(*definition);
    if (name == "intent.constant")
      return finish({true, true, {}, {}});
    if ((name == "intent.broadcast" || name == "intent.reshape" ||
         name == "intent.cast" || name == "intent.gather") &&
        definition->getNumOperands() >= 1)
      return finish(inspect(definition->getOperand(0)));
    if (name == "intent.indices") {
      auto axes = analysis.getFacts().valueAxes.find(value);
      if (axes == analysis.getFacts().valueAxes.end())
        return finish({});
      SelectedIndexExpression result{true, false, {}, {}};
      for (const target::LogicalAxis &axis : axes->second) {
        if (axis.extent == "1")
          continue;
        auto node = axis.domain
                        ? axis.domain->getAttrOfType<IntegerAttr>("intent.node")
                        : IntegerAttr();
        if (!node)
          return finish({});
        if (!analysis.isScalarAxis(node.getInt()))
          result.varyingAxes.insert(node.getInt());
      }
      return finish(result.varyingAxes.size() <= 1 ? std::move(result)
                                                    : SelectedIndexExpression());
    }
    if ((name == "intent.view_load" || name == "intent.members") &&
        isPhysicallyScalarTensor(value, analysis)) {
      auto node = definition->getAttrOfType<IntegerAttr>("intent.node");
      if (!node)
        return finish({});
      SelectedIndexExpression result{true, false, {}, {}};
      result.scalarNodes.insert(node.getInt());
      return finish(std::move(result));
    }
    if (name == "intent.unary" && definition->getNumOperands() == 1) {
      auto kind = definition->getAttrOfType<StringAttr>("intent.operator");
      if (!kind || kind.getValue() != "negate")
        return finish({});
      return finish(inspect(definition->getOperand(0)));
    }
    if (name != "intent.binary" || definition->getNumOperands() != 2)
      return finish({});
    auto kind = definition->getAttrOfType<StringAttr>("intent.operator");
    if (!kind || (kind.getValue() != "add" &&
                  kind.getValue() != "subtract" &&
                  kind.getValue() != "multiply"))
      return finish({});
    SelectedIndexExpression lhs = inspect(definition->getOperand(0));
    SelectedIndexExpression rhs = inspect(definition->getOperand(1));
    if (!lhs.valid || !rhs.valid ||
        (kind.getValue() == "multiply" && !lhs.constant && !rhs.constant))
      return finish({});
    SelectedIndexExpression result{true, lhs.constant && rhs.constant, {}, {}};
    result.scalarNodes.insert(lhs.scalarNodes.begin(), lhs.scalarNodes.end());
    result.scalarNodes.insert(rhs.scalarNodes.begin(), rhs.scalarNodes.end());
    result.varyingAxes.insert(lhs.varyingAxes.begin(), lhs.varyingAxes.end());
    result.varyingAxes.insert(rhs.varyingAxes.begin(), rhs.varyingAxes.end());
    if (result.scalarNodes.size() != 1 || result.varyingAxes.size() > 1)
      return finish({});
    return finish(std::move(result));
  };

  SelectedIndexExpression expression = inspect(tensorIndex);
  if (!expression.valid || expression.scalarNodes.size() != 1 ||
      expression.varyingAxes.size() > 1)
    return std::nullopt;
  SelectedContiguousIndexMatch match;
  match.scalarNodes.assign(expression.scalarNodes.begin(),
                           expression.scalarNodes.end());
  llvm::sort(match.scalarNodes);
  return match;
}

bool feedsAtomicValue(Operation &operation) {
  if (operation.getNumResults() != 1 ||
      !llvm::hasSingleElement(operation.getResult(0).getUsers()))
    return false;
  Operation *user = *operation.getResult(0).user_begin();
  auto valueIndex =
      user->getAttrOfType<IntegerAttr>("intent.value_operand_index");
  return ::intent::target::semanticOperationName(*user) == "intent.atomic_add" && valueIndex &&
         valueIndex.getInt() >= 0 &&
         static_cast<unsigned>(valueIndex.getInt()) < user->getNumOperands() &&
         user->getOperand(valueIndex.getInt()) == operation.getResult(0);
}

bool validityCoveredByPhysicalExtent(intent::plan::ProgramOp program,
                                     intent::plan::TransferOp transfer) {
  if (transfer.getValidityDomainNodes().empty())
    return false;
  llvm::StringSet<> physicalExtents;
  for (intent::plan::BlockExtentOp extent :
       program.getBody().getOps<intent::plan::BlockExtentOp>())
    physicalExtents.insert(extent.getLogicalExtent());
  return llvm::all_of(transfer.getValidityDomainNodes(), [&](int64_t node) {
    return llvm::any_of(program.getBody().getOps<intent::plan::RangeOp>(),
                        [&](intent::plan::RangeOp range) {
                          return static_cast<int64_t>(range.getAxisNode()) ==
                                     node &&
                                 range.getPurpose() != "access" &&
                                 range.getLevel() == 0 &&
                                 physicalExtents.contains(range.getExtent());
                        });
  });
}

bool hasCompactAccessRange(intent::plan::ProgramOp program,
                           intent::plan::TransferOp transfer) {
  SmallVector<intent::plan::RangeOp> ranges;
  for (intent::plan::RangeOp range :
       program.getBody().getOps<intent::plan::RangeOp>())
    if (range.getPurpose() == "access" && range.getTransferNodeAttr() &&
        range.getTransferNodeAttr().getInt() ==
            static_cast<int64_t>(transfer.getNode()))
      ranges.push_back(range);
  return ranges.size() == 1 && ranges.front().getDivisorAttr() &&
         ranges.front().getDivisorAttr().getInt() > 1 &&
         ranges.front().getOffsetAttr();
}

bool isRankReducingReductionChain(
    Operation &operation, const llvm::DenseSet<int64_t> &reductionNodes) {
  if (operation.getNumOperands() == 0 || operation.getNumResults() != 1)
    return false;
  Operation *producer = operation.getOperand(0).getDefiningOp();
  auto producerNode =
      producer ? producer->getAttrOfType<IntegerAttr>("intent.node")
               : IntegerAttr();
  if (!producer || !producerNode ||
      !reductionNodes.contains(producerNode.getInt()) ||
      producer->getNumOperands() == 0)
    return false;
  auto source = dyn_cast<RankedTensorType>(producer->getOperand(0).getType());
  auto intermediate =
      dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  return source && intermediate && result && source.getRank() >= 4 &&
         intermediate.getRank() + 1 == source.getRank() &&
         result.getRank() + 1 == intermediate.getRank();
}

bool hasSubwarpRowContraction(gpu::PhysicalProgramAnalysis &analysis,
                              intent::plan::ContractOp contract,
                              Operation &operation) {
  if (std::optional<int64_t> axis = contract.getLhsResultAxisNode()) {
    intent::plan::AxisOp physicalAxis = analysis.getAxis(*axis);
    if (physicalAxis) {
      for (StringRef purpose : {StringRef("ownership"), StringRef("lane")}) {
        intent::plan::RangeOp range = analysis.getRange(*axis, purpose);
        if (!range)
          continue;
        int64_t extent = 0;
        if (!range.getExtent().getAsInteger(10, extent) && extent > 0)
          return extent < 16;
      }
    }
  }
  if (operation.getNumOperands() != 2 || operation.getNumResults() != 1)
    return false;
  auto lhs = dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
  auto rhs = dyn_cast<RankedTensorType>(operation.getOperand(1).getType());
  auto result = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  auto reduction = operation.getAttrOfType<ArrayAttr>("intent.reduce");
  auto pair = reduction && reduction.size() == 1
                  ? dyn_cast<ArrayAttr>(reduction[0])
                  : ArrayAttr();
  auto lhsReduction = pair && pair.size() == 2
                          ? dyn_cast<IntegerAttr>(pair[0])
                          : IntegerAttr();
  auto rhsReduction = pair && pair.size() == 2
                          ? dyn_cast<IntegerAttr>(pair[1])
                          : IntegerAttr();
  int64_t rows = result ? result.getDimSize(0) : ShapedType::kDynamic;
  return lhs && rhs && result && lhs.getRank() == 2 && rhs.getRank() == 2 &&
         result.getRank() == 2 && rows > 0 && rows < 16 && lhsReduction &&
         rhsReduction && lhsReduction.getInt() == 1 &&
         (rhsReduction.getInt() == 0 || rhsReduction.getInt() == 1);
}

LogicalResult realizeProgram(intent::plan::ProgramOp program,
                             intent::plan::SearchSpaceOp searchSpace) {
  FailureOr<std::unique_ptr<gpu::PhysicalProgramAnalysis>> analysis =
      gpu::PhysicalProgramAnalysis::compute(program);
  if (failed(analysis))
    return failure();
  OpBuilder builder(program.getContext());
  intent::plan::LaunchOp launch = (*analysis)->getLaunch();
  if (!launch)
    return program.emitOpError("has no launch decision for TileLang realization");
  if (launch->hasAttr(rowLaunchAttr) ||
      launch->hasAttr(equalProgramTilesAttr))
    return launch.emitOpError("already has TileLang provider decisions");

  intent::plan::AxisOp lane = firstLane(**analysis);
  intent::plan::RangeOp laneRange =
      lane ? (*analysis)->getRange(lane.getNode(), "lane")
           : intent::plan::RangeOp();
  bool matrixProgram =
      !program.getBody().getOps<intent::plan::ContractOp>().empty() ||
      !program.getBody().getOps<intent::plan::SparseContractOp>().empty();
  bool tuneRows =
      !searchSpace && !(*analysis)->hasWorkerReuse() &&
      (matrixProgram ||
       (laneRange && laneRange.getTile().starts_with("row_vector") &&
        !target::lowering::hasNonReplayableEffect(
            (*analysis)->getKernel().entry.getOperation())));
  launch->setAttr(rowLaunchAttr,
                  builder.getStringAttr(tuneRows ? "delegated" : "fixed"));

  llvm::DenseSet<int64_t> reductionNodes;
  for (intent::plan::ReductionOp reduction :
       program.getBody().getOps<intent::plan::ReductionOp>())
    reductionNodes.insert(reduction.getNode());
  bool equalTiles = false;
  for (int64_t node : reductionNodes) {
    Operation *operation = (*analysis)->getKernel().nodes.lookup(node);
    equalTiles |= operation &&
                  isRankReducingReductionChain(*operation, reductionNodes);
  }
  launch->setAttr(equalProgramTilesAttr, builder.getBoolAttr(equalTiles));

  if (searchSpace) {
    auto declarations = searchSpace.getBody().getOps<intent::plan::AutotuneOp>();
    if (!llvm::hasSingleElement(declarations))
      return searchSpace.emitOpError(
          "TileLang provider realization requires one autotune declaration");
    intent::plan::AutotuneOp autotune = *declarations.begin();
    if (autotune->hasAttr(gemmWarpPolicyAttr))
      return autotune.emitOpError(
          "already has a TileLang GEMM warp-policy decision");
    bool tuneGemm = !program.getBody().getOps<intent::plan::ContractOp>().empty() ||
                    !program.getBody()
                         .getOps<intent::plan::SparseContractOp>()
                         .empty();
    autotune->setAttr(gemmWarpPolicyAttr, builder.getBoolAttr(tuneGemm));
  }

  target::KernelModel &kernel = (*analysis)->getKernel();

  llvm::DenseMap<int64_t, int64_t> nativeContractOperandReshapes;
  WalkResult nativeOperands = kernel.entry.walk([&](Operation *operation) {
    Operation *reshape = matchNativeContractOperandLoad(*operation, **analysis);
    if (!reshape)
      return WalkResult::advance();
    auto loadNode = operation->getAttrOfType<IntegerAttr>("intent.node");
    auto reshapeNode = reshape->getAttrOfType<IntegerAttr>("intent.node");
    if (!loadNode || !reshapeNode) {
      operation->emitOpError(
          "native TileLang contraction operand lacks canonical nodes");
      return WalkResult::interrupt();
    }
    nativeContractOperandReshapes[loadNode.getInt()] = reshapeNode.getInt();
    return WalkResult::advance();
  });
  if (nativeOperands.wasInterrupted())
    return failure();

  FailureOr<func::FuncOp> physicalEntry = intent::plan::getPhysicalEntry(program);
  if (failed(physicalEntry))
    return failure();
  WalkResult loopForms = physicalEntry->walk([&](intent::plan::ExecForOp loop) {
    if (loop->hasAttr(loopCarrierSpacesAttr)) {
      loop.emitOpError("already has a TileLang loop-carrier form");
      return WalkResult::interrupt();
    }
    SmallVector<Attribute> spaces;
    spaces.reserve(loop.getNumResults());
    for (Type type : loop.getResultTypes()) {
      if (type.isIntOrIndexOrFloat())
        spaces.push_back(builder.getStringAttr("local"));
      else if (isa<RankedTensorType>(type))
        spaces.push_back(builder.getStringAttr("shared"));
      else {
        loop.emitOpError("has an unsupported TileLang loop-carrier type");
        return WalkResult::interrupt();
      }
    }
    loop->setAttr(loopCarrierSpacesAttr, builder.getArrayAttr(spaces));
    return WalkResult::advance();
  });
  if (loopForms.wasInterrupted())
    return failure();

  for (const auto &entry : kernel.raggedRelations) {
    Operation *ragged = entry.second.operation;
    if (!ragged || ragged->hasAttr(raggedRouteAttr))
      return program.emitOpError(
          "has an invalid or duplicate TileLang ragged-route decision");
    FailureOr<StringRef> route =
        target::lowering::classifyRaggedProjection(*ragged);
    if (failed(route))
      return failure();
    ragged->setAttr(raggedRouteAttr, builder.getStringAttr(*route));
  }

  for (intent::plan::ContractOp contract :
       program.getBody().getOps<intent::plan::ContractOp>()) {
    if (contract->hasAttr(isolateLhsAttr) || contract->hasAttr(isolateRhsAttr) ||
        contract->hasAttr(contractLoweringAttr) ||
        contract->hasAttr(contractOrientationAttr) ||
        contract->hasAttr(contractBatchedAttr) ||
        contract->hasAttr(contractMmaFormAttr) ||
        contract->hasAttr(packedDecodeLoadNodeAttr) ||
        contract->hasAttr(packedDecodeColumnAxisNodeAttr) ||
        contract->hasAttr(packedDecodeProducerNodesAttr) ||
        contract->hasAttr(scaledContractLayoutAttr))
      return contract.emitOpError(
          "already has TileLang contraction-operand forms");
    Operation *operation =
        kernel.nodes.lookup(contract.getNode());
    if (!operation ||
        (operation->getNumOperands() != 2 &&
         (contract.getForm() != "scaled_direct" ||
          operation->getNumOperands() != 4)))
      return contract.emitOpError("does not bind canonical contraction operands");
    FailureOr<target::lowering::ContractionOrientation> orientation =
        target::lowering::contractionOrientation(*operation);
    if (failed(orientation))
      return contract.emitOpError(
          "does not bind canonical contraction orientation");
    if (orientation->batched)
      return contract.emitOpError(
          "TileLang 0.1.13 has no mechanical batched GEMM projection");
    auto repeatedContractionOperand = [](Value operand) {
      return llvm::count_if(operand.getUsers(), [](Operation *user) {
               return ::intent::target::semanticOperationName(*user) == "intent.contract";
             }) > 1;
    };
    bool subwarpRows =
        hasSubwarpRowContraction(**analysis, contract, *operation);
    if (subwarpRows)
      return contract.emitOpError(
          "the current TileLang provider form cannot pad a logical contraction "
          "row extent smaller than 16 to a native MMA fragment");
    contract->setAttr(
        isolateLhsAttr,
        builder.getBoolAttr(repeatedContractionOperand(
            operation->getOperand(0))));
    contract->setAttr(
        isolateRhsAttr,
        builder.getBoolAttr(repeatedContractionOperand(
            operation->getOperand(1))));
    StringRef lowering = contract.getForm() == "scaled_direct"
                             ? syntax::scaledContraction()
                             : syntax::contraction();
    contract->setAttr(contractLoweringAttr, builder.getStringAttr(lowering));
    contract->setAttr(contractOrientationAttr,
                      builder.getStringAttr(
                          target::lowering::contractionOrientationName(
                              *orientation)));
    contract->setAttr(contractBatchedAttr,
                      builder.getBoolAttr(orientation->batched));
    StringRef mmaForm = "native";
    std::optional<int64_t> reductionAxis = contract.getReductionAxisNode();
    std::optional<int64_t> lhsResultAxis = contract.getLhsResultAxisNode();
    auto operandElementType = [](Value value) -> Type {
      auto tensor = dyn_cast<RankedTensorType>(value.getType());
      return tensor ? tensor.getElementType() : Type();
    };
    Type lhsElement = operandElementType(operation->getOperand(0));
    Type rhsElement = operandElementType(operation->getOperand(1));
    bool fp8Operands =
        lhsElement && rhsElement &&
        isa<Float8E4M3FNType, Float8E5M2Type>(lhsElement) &&
        isa<Float8E4M3FNType, Float8E5M2Type>(rhsElement);
    std::optional<PackedInt2DecodeMatch> packedDecode =
        contract.getForm() == "deferred_one"
            ? matchPackedInt2Decode(*operation)
            : std::nullopt;
    if (packedDecode) {
      auto fact = (*analysis)->getFacts().contractions.find(operation);
      if (fact == (*analysis)->getFacts().contractions.end() ||
          fact->second.rhsAxes.size() != 2 ||
          fact->second.rhsReductionAxes.size() != 1 ||
          fact->second.rhsReductionAxes.front() >= fact->second.rhsAxes.size())
        return contract.emitOpError(
            "packed INT2 provider form requires one rank-two rhs reduction");
      unsigned reductionTensorAxis = fact->second.rhsReductionAxes.front();
      unsigned columnTensorAxis = reductionTensorAxis == 0 ? 1 : 0;
      Operation *columnDomain = fact->second.rhsAxes[columnTensorAxis].domain;
      auto columnNode =
          columnDomain
              ? columnDomain->getAttrOfType<IntegerAttr>("intent.node")
              : IntegerAttr();
      auto loadNode =
          packedDecode->load->getAttrOfType<IntegerAttr>("intent.node");
      SmallVector<int64_t> producerNodes;
      for (Operation *producer : packedDecode->producers) {
        auto producerNode =
            producer->getAttrOfType<IntegerAttr>("intent.node");
        if (!producerNode)
          return producer->emitOpError(
              "packed INT2 provider form requires canonical producer nodes");
        producerNodes.push_back(producerNode.getInt());
      }
      if (!loadNode || !columnNode)
        return packedDecode->load->emitOpError(
            "packed INT2 provider form requires load and column-axis nodes");
      mmaForm = "packed_int2_i8_mma";
      contract->setAttr(packedDecodeLoadNodeAttr,
                        builder.getI64IntegerAttr(loadNode.getInt()));
      contract->setAttr(packedDecodeColumnAxisNodeAttr,
                        builder.getI64IntegerAttr(columnNode.getInt()));
      contract->setAttr(packedDecodeProducerNodesAttr,
                        builder.getDenseI64ArrayAttr(producerNodes));
    } else if (reductionAxis && isRaggedBoundAxis(**analysis, *reductionAxis))
      mmaForm = "ragged_masked";
    else if (fp8Operands && lhsResultAxis &&
             (*analysis)->axisHasRole(*lhsResultAxis, "lane"))
      mmaForm = "fp8_lane";
    contract->setAttr(contractMmaFormAttr, builder.getStringAttr(mmaForm));
    if (contract.getForm() == "scaled_direct") {
      FailureOr<StringRef> layout =
          target::lowering::scaledContractionLayout(*operation);
      if (failed(layout))
        return failure();
      contract->setAttr(scaledContractLayoutAttr,
                        builder.getStringAttr(*layout));
    }
  }

  for (intent::plan::ReductionOp reduction :
       program.getBody().getOps<intent::plan::ReductionOp>()) {
    if (reduction->hasAttr(reductionLoweringAttr) ||
        reduction->hasAttr(reductionAxisAttr))
      return reduction.emitOpError("already has a TileLang reduction spelling");
    Operation *operation = kernel.nodes.lookup(reduction.getNode());
    FailureOr<std::string> role =
        operation ? target::lowering::reductionRole(*operation)
                  : FailureOr<std::string>(failure());
    FailureOr<int64_t> axis =
        operation ? target::lowering::reductionAxis(*operation)
                  : FailureOr<int64_t>(failure());
    if (failed(role) || failed(axis))
      return reduction.emitOpError("does not bind canonical reduction semantics");
    if (*role == "reduce_generic")
      return reduction.emitOpError(
          "TileLang 0.1.13 CUDA codegen cannot lower the tirx.Reduce produced "
          "by comm_reducer; generic reduction combiners are unsupported");
    reduction->setAttr(reductionLoweringAttr,
                       builder.getStringAttr(syntax::reduction(*role)));
    reduction->setAttr(reductionAxisAttr, builder.getI64IntegerAttr(*axis));
  }

  for (intent::plan::ScanOp scan :
       program.getBody().getOps<intent::plan::ScanOp>()) {
    if (scan->hasAttr(scanLoweringAttr))
      return scan.emitOpError("already has a TileLang scan spelling");
    Operation *operation = kernel.nodes.lookup(scan.getNode());
    FailureOr<std::string> role =
        operation ? target::lowering::scanRole(*operation)
                  : FailureOr<std::string>(failure());
    if (succeeded(role) && *role == "scan_generic_inclusive")
      return scan.emitOpError(
          "TileLang 0.1.13 has no mechanically lowerable generic scan "
          "combiner path; generic scan combiners are unsupported");
    if (failed(role) || *role != "scan_inclusive_add")
      return scan.emitOpError("does not bind a supported TileLang scan form");
    scan->setAttr(scanLoweringAttr, builder.getStringAttr(syntax::scan()));
  }

  for (intent::plan::PointwiseOp pointwise :
       program.getBody().getOps<intent::plan::PointwiseOp>()) {
    if (pointwise->hasAttr(pointwiseFormAttr) ||
        pointwise->hasAttr(pointwiseStorageAttr) ||
        pointwise->hasAttr(pointwiseLoweringAttr))
      return pointwise.emitOpError(
          "already has a TileLang pointwise-form decision");
    Operation *operation =
        kernel.nodes.lookup(pointwise.getNode());
    if (!operation)
      return pointwise.emitOpError("does not bind canonical pointwise semantics");
    std::string gatherForm;
    if (target::semanticOperationName(*operation) == "intent.gather") {
      FailureOr<std::string> gather =
          target::lowering::classifyGatherProjection(*operation);
      if (failed(gather))
        return failure();
      if (*gather == "fragment_projection")
        return pointwise.emitOpError(
            "TileLang fragment projection form is not materialized");
      gatherForm = std::move(*gather);
    }
    StringRef form = target::lowering::feedsContraction(*operation)
                         ? StringRef("contract_operand")
                         : StringRef("elementwise");
    FailureOr<std::string> role = target::lowering::pointwiseRole(*operation);
    FailureOr<StringRef> lowering =
        succeeded(role) ? syntax::pointwise(pointwise.getOperation(), *role, form)
                        : FailureOr<StringRef>(failure());
    if (failed(lowering))
      return pointwise.emitOpError("does not bind canonical pointwise semantics");
    pointwise->setAttr(pointwiseFormAttr, builder.getStringAttr(form));
    pointwise->setAttr(
        pointwiseStorageAttr,
        builder.getStringAttr(
            succeeded(role) && *role == "cast" && form == "contract_operand"
                ? "shared"
                : "plan"));
    pointwise->setAttr(pointwiseLoweringAttr,
                       builder.getStringAttr(*lowering));
    if (!gatherForm.empty())
      pointwise->setAttr(gatherFormAttr, builder.getStringAttr(gatherForm));
  }

  for (intent::plan::StreamBindingOp stream :
       program.getBody().getOps<intent::plan::StreamBindingOp>()) {
    if (stream->hasAttr(streamTileAttr))
      return stream.emitOpError("already has a TileLang stream-tile spelling");
    intent::plan::RangeOp range = (*analysis)->getRange(
        stream.getAxisNode(), stream.getPurpose(), stream.getLevel());
    FailureOr<std::string> tile =
        range ? syntax::tile(stream.getOperation(), range.getTile())
              : FailureOr<std::string>(failure());
    if (failed(tile))
      return stream.emitOpError("does not bind one TileLang stream tile");
    stream->setAttr(streamTileAttr, builder.getStringAttr(*tile));
  }

  for (intent::plan::TransferOp transfer :
       program.getBody().getOps<intent::plan::TransferOp>()) {
    if (transfer->hasAttr(accessAttr) || transfer->hasAttr(transferAttr) ||
        transfer->hasAttr(boundsAttr) || transfer->hasAttr(deferredAttr) ||
        transfer->hasAttr(selectedScalarIndexNodesAttr) ||
        transfer->hasAttr(nativeContractReshapeNodeAttr))
      return transfer.emitOpError(
          "already has a TileLang transfer-form decision");
    Operation *operation =
        (*analysis)->getKernel().nodes.lookup(transfer.getNode());
    StringRef name = operation ? ::intent::target::semanticOperationName(*operation) : StringRef();
    bool load = name == "intent.view_load";
    bool store = name == "intent.view_store" || name == "intent.scatter_unique" ||
                 name == "intent.atomic_add" || name == "intent.atomic_cas";
    if (!load && !store)
      return transfer.emitOpError("does not bind a canonical transfer");
    FailureOr<bool> derivedScalar = target::hasDerivedScalarIndex(*operation);
    if (failed(derivedScalar))
      return failure();
    std::optional<SelectedContiguousIndexMatch> selectedContiguous =
        load && transfer.getTensorIndexing() == "data_dependent"
            ? matchSelectedContiguousIndex(*operation, **analysis)
            : std::nullopt;
    bool tensorIndirect = transfer.getTensorIndexing() == "data_dependent" &&
                          !selectedContiguous;
    bool raggedBound = llvm::any_of(
        transfer.getDomainNodes(), [&](int64_t node) {
          return isRaggedBoundAxis(**analysis, node);
        });
    bool plannedValidity = !transfer.getValidityDomainNodes().empty() &&
                           (store || transfer.getFill() != "none");
    bool physicalValidity =
        validityCoveredByPhysicalExtent(program, transfer);
    bool logicalBounds =
        raggedBound ||
        (plannedValidity &&
         (!transfer.getConsumerNeutralized() || !physicalValidity));
    bool packedScalar = llvm::any_of(
        transfer.getDomainNodes(), [&](int64_t node) {
          return (*analysis)->isPackedScalarAxis(node);
        });
    bool compactIndexing =
        load && transfer.getTensorIndexing() == "compact" &&
        transfer.getCoverageSpace() != "none";
    bool compact = compactIndexing && hasCompactAccessRange(program, transfer);
    bool elementwise = *derivedScalar || tensorIndirect || logicalBounds ||
                       (compactIndexing && !compact);
    bool bounds = (*analysis)->hasWorkerReuse() || raggedBound ||
                  (logicalBounds && !physicalValidity) || *derivedScalar ||
                  tensorIndirect || packedScalar;
    bool scalarResult = operation->getNumResults() == 1 &&
                        !isa<RankedTensorType>(operation->getResult(0).getType());
    Type resultElement =
        scalarResult
            ? operation->getResult(0).getType()
            : operation->getNumResults() == 1
                  ? cast<RankedTensorType>(operation->getResult(0).getType())
                        .getElementType()
                  : Type();
    bool guardedF16Bulk =
        load && !scalarResult && resultElement.isF16() && *derivedScalar &&
        !tensorIndirect && !logicalBounds && bounds && transfer.getFill() != "none";
    if (guardedF16Bulk) {
      FailureOr<SmallVector<target::IndexTerm>> relation =
          target::parseIndexRelation(*operation);
      if (failed(relation))
        return failure();
      for (const target::IndexTerm &term : *relation) {
        if (term.kind != "region_index")
          continue;
        if (term.operands.size() != 1 || !term.operands.front())
          return operation->emitOpError(
              "has no guarded float16 region projection");
        Value indexed = operation->getOperand(*term.operands.front());
        auto source = (*analysis)->getFacts().regionArgumentAxes.find(indexed);
        Operation *domain = source == (*analysis)->getFacts().regionArgumentAxes.end()
                                ? nullptr
                                : source->second.domain;
        auto node = domain
                        ? domain->getAttrOfType<IntegerAttr>("intent.node")
                        : IntegerAttr();
        if (!node || !(*analysis)->axisHasRole(node.getInt(), "lane") ||
            (*analysis)->axisHasRole(node.getInt(), "parallel") ||
            (*analysis)->axisHasRole(node.getInt(), "ordered") ||
            (*analysis)->axisHasRole(node.getInt(), "reduction") ||
            (*analysis)->axisHasRole(node.getInt(), "ragged_member")) {
          guardedF16Bulk = false;
          break;
        }
      }
    }
    transfer->setAttr(
        accessAttr,
        builder.getStringAttr((*analysis)->hasWorkerReuse()
                                  ? (load ? "gather" : "scatter")
                                  : (load ? "load" : "store")));
    auto nativeContract = nativeContractOperandReshapes.find(transfer.getNode());
    bool nativeContractOperand =
        nativeContract != nativeContractOperandReshapes.end() && load &&
        transfer.getResultSpace() == "shared" && !elementwise && !bounds;
    StringRef transferForm = nativeContractOperand
                                 ? StringRef("native_contract_operand")
                             : selectedContiguous
                                 ? StringRef("selected_contiguous")
                             : compact
                                 ? StringRef("compact")
                             : guardedF16Bulk
                                 ? StringRef("guarded_f16_bulk")
                             : elementwise ? StringRef("parallel_elements")
                                           : StringRef("bulk_copy");
    transfer->setAttr(transferAttr, builder.getStringAttr(transferForm));
    if (selectedContiguous)
      transfer->setAttr(selectedScalarIndexNodesAttr,
                        builder.getDenseI64ArrayAttr(
                            selectedContiguous->scalarNodes));
    if (nativeContractOperand)
      transfer->setAttr(nativeContractReshapeNodeAttr,
                        builder.getI64IntegerAttr(nativeContract->second));
    if (nativeContractOperand) {
      intent::plan::PointwiseOp reshapeBinding;
      for (intent::plan::PointwiseOp value :
           program.getBody().getOps<intent::plan::PointwiseOp>())
        if (static_cast<int64_t>(value.getNode()) == nativeContract->second) {
          reshapeBinding = value;
          break;
        }
      if (!reshapeBinding)
        return transfer.emitOpError(
            "native TileLang contraction operand has no reshape binding");
      reshapeBinding->setAttr(pointwiseFormAttr,
                              builder.getStringAttr("contract_operand"));
      reshapeBinding->setAttr(pointwiseLoweringAttr,
                              builder.getStringAttr("alias"));
      reshapeBinding->setAttr(pointwiseStorageAttr,
                              builder.getStringAttr("shared"));
    }
    transfer->setAttr(boundsAttr, builder.getBoolAttr(bounds));
    transfer->setAttr(
        deferredAttr,
        builder.getBoolAttr(load && feedsAtomicValue(*operation)));
  }
  return success();
}

class RealizeProviderProgramPass final
    : public PassWrapper<RealizeProviderProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RealizeProviderProgramPass)
  StringRef getArgument() const final {
    return "intent-realize-tilelang-program-forms";
  }
  StringRef getDescription() const final {
    return "Select TileLang-local access, value, and launch forms before terminal translation";
  }
  void runOnOperation() final {
    auto selected = getProgram(getOperation());
    if (failed(selected) ||
        failed(realizeProgram(selected->first, selected->second)))
      signalPassFailure();
  }
};

class VerifyProviderProgramPass final
    : public PassWrapper<VerifyProviderProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyProviderProgramPass)
  StringRef getArgument() const final {
    return "intent-verify-tilelang-program-forms";
  }
  StringRef getDescription() const final {
    return "Verify the complete TileLang provider-program contract";
  }
  void runOnOperation() final {
    auto selected = getProgram(getOperation());
    FailureOr<func::FuncOp> entry =
        succeeded(selected) ? intent::plan::getPhysicalEntry(selected->first)
                            : FailureOr<func::FuncOp>(failure());
    FailureOr<target::KernelModel> kernel =
        succeeded(entry) ? target::analyzeKernel(*entry)
                         : FailureOr<target::KernelModel>(failure());
    if (failed(selected) || failed(entry) || failed(kernel) ||
        failed(verifyProviderProgram(*kernel, selected->first,
                                     selected->second)))
      signalPassFailure();
  }
};

} // namespace

LogicalResult verifyProviderProgram(const target::KernelModel &kernel,
                                    intent::plan::ProgramOp program,
                                    intent::plan::SearchSpaceOp searchSpace) {
  (void)searchSpace;
  FailureOr<func::FuncOp> physicalEntry = intent::plan::getPhysicalEntry(program);
  if (failed(physicalEntry))
    return failure();
  WalkResult loopForms = physicalEntry->walk([&](intent::plan::ExecForOp loop) {
    auto spaces = loop->getAttrOfType<ArrayAttr>(loopCarrierSpacesAttr);
    if (!spaces || spaces.size() != loop.getNumResults()) {
      loop.emitOpError("has no complete TileLang loop-carrier form");
      return WalkResult::interrupt();
    }
    for (auto [type, attribute] : llvm::zip(loop.getResultTypes(), spaces)) {
      auto space = dyn_cast<StringAttr>(attribute);
      StringRef expected = type.isIntOrIndexOrFloat()
                               ? StringRef("local")
                           : isa<RankedTensorType>(type)
                               ? StringRef("shared")
                               : StringRef();
      if (!space || expected.empty() || space.getValue() != expected) {
        loop.emitOpError("has an invalid TileLang loop-carrier residency");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (loopForms.wasInterrupted())
    return failure();
  for (const auto &entry : kernel.raggedRelations) {
    Operation *ragged = entry.second.operation;
    auto route = ragged ? ragged->getAttrOfType<StringAttr>(raggedRouteAttr)
                        : StringAttr();
    if (!route || (route.getValue() != "compact" &&
                   route.getValue() != "indexed"))
      return program.emitOpError(
          "has no complete TileLang ragged-route form");
  }
  auto launches = program.getBody().getOps<intent::plan::LaunchOp>();
  if (!llvm::hasSingleElement(launches))
    return program.emitOpError(
        "TileLang provider program requires one launch form");
  intent::plan::LaunchOp launch = *launches.begin();
  auto row = launch->getAttrOfType<StringAttr>(rowLaunchAttr);
  auto equal = launch->getAttrOfType<BoolAttr>(equalProgramTilesAttr);
  if (!row || (row.getValue() != "fixed" && row.getValue() != "delegated") ||
      !equal)
    return launch.emitOpError(
        "has no complete TileLang launch-form decision");
  if (searchSpace) {
    auto declarations = searchSpace.getBody().getOps<intent::plan::AutotuneOp>();
    if (!llvm::hasSingleElement(declarations) ||
        !(*declarations.begin())->getAttrOfType<BoolAttr>(gemmWarpPolicyAttr))
      return searchSpace.emitOpError(
          "has no TileLang GEMM warp-policy tuning decision");
  }
  FailureOr<std::unique_ptr<gpu::PhysicalProgramAnalysis>> analysis =
      gpu::PhysicalProgramAnalysis::compute(program);
  if (failed(analysis))
    return failure();
  llvm::DenseMap<int64_t, unsigned> nativeContractAliasCounts;
  for (intent::plan::TransferOp transfer :
       program.getBody().getOps<intent::plan::TransferOp>()) {
    auto form = transfer->getAttrOfType<StringAttr>(transferAttr);
    auto bounds = transfer->getAttrOfType<BoolAttr>(boundsAttr);
    auto reshape =
        transfer->getAttrOfType<IntegerAttr>(nativeContractReshapeNodeAttr);
    bool nativeForm = form && form.getValue() == "native_contract_operand";
    if (!nativeForm) {
      if (reshape)
        return transfer.emitOpError(
            "has a native contraction reshape without its transfer form");
      continue;
    }
    Operation *operation = kernel.nodes.lookup(transfer.getNode());
    Operation *matchedReshape =
        operation ? matchNativeContractOperandLoad(*operation, **analysis)
                  : nullptr;
    auto matchedNode = matchedReshape
                           ? matchedReshape->getAttrOfType<IntegerAttr>(
                                 "intent.node")
                           : IntegerAttr();
    if (!operation ||
        target::semanticOperationName(*operation) != "intent.view_load" ||
        transfer.getResultSpace() != "shared" || !bounds || bounds.getValue() ||
        !reshape || !matchedNode || reshape.getInt() != matchedNode.getInt())
      return transfer.emitOpError(
          "has an invalid native TileLang contraction operand form");
    unsigned &count = nativeContractAliasCounts[reshape.getInt()];
    if (++count != 1)
      return transfer.emitOpError(
          "does not uniquely own its native contraction reshape alias");
  }
  for (intent::plan::PointwiseOp pointwise :
       program.getBody().getOps<intent::plan::PointwiseOp>()) {
    auto form = pointwise->getAttrOfType<StringAttr>(pointwiseFormAttr);
    auto storage = pointwise->getAttrOfType<StringAttr>(pointwiseStorageAttr);
    auto lowering = pointwise->getAttrOfType<StringAttr>(pointwiseLoweringAttr);
    Operation *operation = kernel.nodes.lookup(pointwise.getNode());
    auto gather = pointwise->getAttrOfType<StringAttr>(gatherFormAttr);
    StringRef operationName =
        operation ? target::semanticOperationName(*operation) : StringRef();
    bool requiresGather = operationName == "intent.gather";
    bool nativeAlias = nativeContractAliasCounts.contains(pointwise.getNode());
    bool aliasLowering = lowering && lowering.getValue() == "alias";
    bool canonicalBroadcastAlias =
        aliasLowering && operationName == "intent.broadcast";
    if (!form || (form.getValue() != "elementwise" &&
                  form.getValue() != "contract_operand") ||
        !storage || (storage.getValue() != "plan" &&
                     storage.getValue() != "shared") ||
        !lowering || lowering.getValue().empty() || !operation ||
        (aliasLowering && !nativeAlias && !canonicalBroadcastAlias) ||
        (nativeAlias &&
         (nativeContractAliasCounts.lookup(pointwise.getNode()) != 1 ||
          !aliasLowering ||
          form.getValue() != "contract_operand" ||
          storage.getValue() != "shared" ||
          operationName != "intent.reshape")) ||
        (requiresGather && (!gather || gather.getValue().empty())))
      return pointwise.emitOpError(
          "has no complete TileLang pointwise-form decision");
  }
  for (intent::plan::ContractOp contract :
       program.getBody().getOps<intent::plan::ContractOp>()) {
    auto lhs = contract->getAttrOfType<BoolAttr>(isolateLhsAttr);
    auto rhs = contract->getAttrOfType<BoolAttr>(isolateRhsAttr);
    auto lowering = contract->getAttrOfType<StringAttr>(contractLoweringAttr);
    auto orientation =
        contract->getAttrOfType<StringAttr>(contractOrientationAttr);
    auto batched = contract->getAttrOfType<BoolAttr>(contractBatchedAttr);
    auto mmaForm = contract->getAttrOfType<StringAttr>(contractMmaFormAttr);
    auto layout = contract->getAttrOfType<StringAttr>(scaledContractLayoutAttr);
    auto packedColumn = contract->getAttrOfType<IntegerAttr>(
        packedDecodeColumnAxisNodeAttr);
    Operation *operation = kernel.nodes.lookup(contract.getNode());
    if (!lhs || !rhs || !lowering || lowering.getValue().empty() ||
        !orientation || !batched || batched.getValue() || !mmaForm ||
        (mmaForm.getValue() != "native" &&
         mmaForm.getValue() != "ragged_masked" &&
         mmaForm.getValue() != "fp8_lane" &&
         mmaForm.getValue() != "packed_int2_i8_mma") ||
        (orientation.getValue() != "nn" && orientation.getValue() != "nt" &&
         orientation.getValue() != "tn" && orientation.getValue() != "tt") ||
        (contract.getForm() == "scaled_direct" &&
         (!layout || (layout.getValue() != "grouped_rank_two" &&
                      layout.getValue() != "flattened_rank_three"))) ||
        (contract.getForm() != "scaled_direct" && layout) || !operation ||
        ((mmaForm.getValue() == "packed_int2_i8_mma") !=
         static_cast<bool>(contract->getAttrOfType<IntegerAttr>(
             packedDecodeLoadNodeAttr))) ||
        ((mmaForm.getValue() == "packed_int2_i8_mma") !=
         static_cast<bool>(packedColumn)) ||
        (packedColumn && !kernel.nodes.lookup(packedColumn.getInt())) ||
        ((mmaForm.getValue() == "packed_int2_i8_mma") !=
         static_cast<bool>(contract->getAttrOfType<DenseI64ArrayAttr>(
             packedDecodeProducerNodesAttr))) ||
        (operation->getNumOperands() != 2 &&
         (contract.getForm() != "scaled_direct" ||
          operation->getNumOperands() != 4)))
      return contract.emitOpError(
          "has no complete TileLang contraction provider form");
  }
  for (intent::plan::ReductionOp reduction :
       program.getBody().getOps<intent::plan::ReductionOp>()) {
    auto lowering = reduction->getAttrOfType<StringAttr>(reductionLoweringAttr);
    auto axis = reduction->getAttrOfType<IntegerAttr>(reductionAxisAttr);
    if (!lowering || lowering.getValue().empty() || !axis || axis.getInt() < 0 ||
        !kernel.nodes.lookup(reduction.getNode()))
      return reduction.emitOpError(
          "has no complete TileLang reduction spelling");
  }
  for (intent::plan::ScanOp scan :
       program.getBody().getOps<intent::plan::ScanOp>()) {
    auto lowering = scan->getAttrOfType<StringAttr>(scanLoweringAttr);
    if (!lowering || lowering.getValue().empty() ||
        !kernel.nodes.lookup(scan.getNode()))
      return scan.emitOpError("has no complete TileLang scan spelling");
  }
  for (intent::plan::StreamBindingOp stream :
       program.getBody().getOps<intent::plan::StreamBindingOp>()) {
    auto tile = stream->getAttrOfType<StringAttr>(streamTileAttr);
    if (!tile || tile.getValue().empty())
      return stream.emitOpError("has no complete TileLang stream-tile spelling");
  }
  for (intent::plan::TransferOp transfer :
       program.getBody().getOps<intent::plan::TransferOp>()) {
    auto access = transfer->getAttrOfType<StringAttr>(accessAttr);
    auto form = transfer->getAttrOfType<StringAttr>(transferAttr);
    auto bounds = transfer->getAttrOfType<BoolAttr>(boundsAttr);
    auto deferred = transfer->getAttrOfType<BoolAttr>(deferredAttr);
    auto selectedScalars =
        transfer->getAttrOfType<DenseI64ArrayAttr>(selectedScalarIndexNodesAttr);
    auto nativeReshape =
        transfer->getAttrOfType<IntegerAttr>(nativeContractReshapeNodeAttr);
    Operation *operation = kernel.nodes.lookup(transfer.getNode());
    StringRef name = operation ? ::intent::target::semanticOperationName(*operation) : StringRef();
    bool load = name == "intent.view_load";
    bool validAccess = access &&
                       ((load && (access.getValue() == "load" ||
                                  access.getValue() == "gather")) ||
                        (!load && (access.getValue() == "store" ||
                                   access.getValue() == "scatter")));
    bool validTransfer = form &&
                         (form.getValue() == "bulk_copy" ||
                          form.getValue() == "parallel_elements" ||
                          form.getValue() == "compact" ||
                          form.getValue() == "selected_contiguous" ||
                          form.getValue() == "native_contract_operand" ||
                          form.getValue() == "guarded_f16_bulk");
    bool selectedForm = form && form.getValue() == "selected_contiguous";
    bool nativeForm = form && form.getValue() == "native_contract_operand";
    bool validSelectedScalars =
        selectedScalars && !selectedScalars.empty() &&
        llvm::all_of(selectedScalars.asArrayRef(), [&](int64_t node) {
          return kernel.nodes.lookup(node) != nullptr;
        });
    if (!operation || !validAccess || !validTransfer || !bounds || !deferred ||
        (selectedForm != validSelectedScalars) ||
        (nativeForm != static_cast<bool>(nativeReshape)) ||
        (nativeReshape &&
         !isSemanticOp(kernel.nodes.lookup(nativeReshape.getInt()),
                       "intent.reshape")))
      return transfer.emitOpError(
          "has no complete legal TileLang transfer-form decision");
  }
  return success();
}

void addProviderPasses(PassManager &manager) {
  manager.addPass(std::make_unique<RealizeProviderProgramPass>());
  manager.addPass(std::make_unique<VerifyProviderProgramPass>());
}

} // namespace intent::tilelang::lowering
