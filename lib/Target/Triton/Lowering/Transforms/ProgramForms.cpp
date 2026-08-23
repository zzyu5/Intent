#include "Intent/Target/Triton/Lowering/Passes.h"

#include "Syntax/Spelling.h"

#include "Intent/Target/Common/Lowering/ProgramAnalysis.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace intent::triton::lowering {
namespace {

FailureOr<std::pair<intent::plan::ProgramOp, intent::plan::SearchSpaceOp>>
getProgram(ModuleOp module) {
  SmallVector<intent::plan::ProgramOp> programs(
      module.getOps<intent::plan::ProgramOp>());
  SmallVector<intent::plan::SearchSpaceOp> searchSpaces(
      module.getOps<intent::plan::SearchSpaceOp>());
  if (programs.size() != 1 || searchSpaces.size() > 1) {
    module.emitError(
        "Triton provider realization requires one physical program and at most one search space");
    return failure();
  }
  return std::make_pair(programs.front(),
                        searchSpaces.empty() ? intent::plan::SearchSpaceOp()
                                             : searchSpaces.front());
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

struct DescriptorCandidate {
  SmallVector<int64_t> blockAxes;
  std::string layout;
};

struct DescriptorAffineExpression {
  llvm::DenseMap<Operation *, int64_t> coefficients;
  int64_t constant = 0;
};

std::optional<int64_t> descriptorIntegerConstant(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition || target::semanticOperationName(*definition) !=
                         "intent.constant")
    return std::nullopt;
  if (auto boolean = definition->getAttrOfType<BoolAttr>("intent.value"))
    return boolean.getValue() ? 1 : 0;
  auto integer = definition->getAttrOfType<IntegerAttr>("intent.value");
  return integer ? std::optional<int64_t>(integer.getInt()) : std::nullopt;
}

bool isDescriptorShapeProjection(Operation &operation) {
  if (target::semanticOperationName(operation) != "intent.gather" ||
      operation.getNumOperands() < 2 || operation.getNumResults() != 1)
    return false;
  auto input = dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  auto validIndex =
      operation.getAttrOfType<IntegerAttr>("intent.valid_operand_index");
  if (!input || !result || failed(relation) || !validIndex ||
      validIndex.getInt() < 0 ||
      static_cast<unsigned>(validIndex.getInt()) >= operation.getNumOperands() ||
      descriptorIntegerConstant(operation.getOperand(validIndex.getInt())) != 1)
    return false;
  unsigned fullSlices = 0;
  unsigned newAxes = 0;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "full_slice")
      ++fullSlices;
    else if (term.kind == "new_axis")
      ++newAxes;
    else
      return false;
  }
  return fullSlices == static_cast<unsigned>(input.getRank()) &&
         result.getRank() == input.getRank() + static_cast<int64_t>(newAxes);
}

std::optional<DescriptorAffineExpression> descriptorAffineExpression(
    Value value, const target::KernelFacts &facts, Operation &consumer,
    llvm::DenseSet<Value> &active) {
  if (!active.insert(value).second)
    return std::nullopt;
  auto finish = [&](std::optional<DescriptorAffineExpression> expression) {
    active.erase(value);
    return expression;
  };
  if (std::optional<int64_t> literal = descriptorIntegerConstant(value)) {
    DescriptorAffineExpression expression;
    expression.constant = *literal;
    return finish(std::move(expression));
  }

  Operation *definition = value.getDefiningOp();
  if (!definition) {
    FailureOr<target::ScalarIndexSource> source =
        target::traceScalarIndexSource(value, consumer);
    if (failed(source) || !source->domain || source->opaque ||
        source->transformed)
      return finish(std::nullopt);
    DescriptorAffineExpression expression;
    expression.coefficients[source->domain] = 1;
    return finish(std::move(expression));
  }
  StringRef name = target::semanticOperationName(*definition);
  if (name == "intent.indices") {
    auto axes = facts.valueAxes.find(value);
    if (axes == facts.valueAxes.end())
      return finish(std::nullopt);
    DescriptorAffineExpression expression;
    for (const target::LogicalAxis &axis : axes->second)
      if (axis.domain && axis.extent != "1")
        ++expression.coefficients[axis.domain];
    return finish(expression.coefficients.empty()
                      ? std::nullopt
                      : std::optional<DescriptorAffineExpression>(
                            std::move(expression)));
  }
  if (name == "intent.broadcast" || name == "intent.reshape" ||
      name == "intent.transpose" || name == "intent.cast" ||
      (name == "intent.gather" && isDescriptorShapeProjection(*definition))) {
    if (definition->getNumOperands() == 0)
      return finish(std::nullopt);
    return finish(descriptorAffineExpression(definition->getOperand(0), facts,
                                             consumer, active));
  }
  if (name != "intent.binary" || definition->getNumOperands() != 2)
    return finish(std::nullopt);
  auto logical = definition->getAttrOfType<StringAttr>("intent.operator");
  std::optional<DescriptorAffineExpression> lhs = descriptorAffineExpression(
      definition->getOperand(0), facts, consumer, active);
  std::optional<DescriptorAffineExpression> rhs = descriptorAffineExpression(
      definition->getOperand(1), facts, consumer, active);
  if (!logical || !lhs || !rhs)
    return finish(std::nullopt);
  auto scale = [](DescriptorAffineExpression &expression, int64_t factor) {
    expression.constant *= factor;
    for (auto &coefficient : expression.coefficients)
      coefficient.second *= factor;
  };
  if (logical.getValue() == "multiply") {
    if (lhs->coefficients.empty()) {
      scale(*rhs, lhs->constant);
      return finish(std::move(rhs));
    }
    if (rhs->coefficients.empty()) {
      scale(*lhs, rhs->constant);
      return finish(std::move(lhs));
    }
    return finish(std::nullopt);
  }
  int64_t sign = logical.getValue() == "add"
                     ? 1
                     : logical.getValue() == "subtract" ? -1 : 0;
  if (!sign)
    return finish(std::nullopt);
  lhs->constant += sign * rhs->constant;
  for (const auto &coefficient : rhs->coefficients)
    lhs->coefficients[coefficient.first] += sign * coefficient.second;
  return finish(std::move(lhs));
}

std::optional<DescriptorCandidate> descriptorCandidate(
    gpu::PhysicalProgramAnalysis &analysis, intent::plan::TransferOp transfer,
    Operation &operation) {
  StringRef semantic = target::semanticOperationName(operation);
  if (semantic != "intent.view_load" && semantic != "intent.view_store")
    return std::nullopt;
  auto view = dyn_cast<intent::ViewType>(operation.getOperand(0).getType());
  auto tensor = view ? dyn_cast<RankedTensorType>(view.getTensor())
                     : RankedTensorType();
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  bool supportedFill = transfer.getFill() == "none" ||
                       transfer.getFill() == "zero" ||
                       transfer.getFill() == "negative_infinity" ||
                       transfer.getFill() == "positive_infinity";
  if (!tensor || tensor.getRank() < 2 || failed(relation) ||
      relation->size() != static_cast<size_t>(tensor.getRank()) ||
      !supportedFill ||
      (semantic == "intent.view_store" && transfer.getFill() != "none") ||
      (transfer.getFill() != "none" && transfer.getFill() != "zero" &&
       transfer.getMaterialization() != "direct"))
    return std::nullopt;

  llvm::DenseMap<int64_t, intent::plan::RegionBindingOp> regions;
  for (intent::plan::RegionBindingOp binding :
       analysis.getProgram().getBody().getOps<intent::plan::RegionBindingOp>())
    regions[binding.getValue()] = binding;
  DescriptorCandidate candidate;
  llvm::DenseSet<int64_t> usedBlockAxes;
  unsigned blockedAxes = 0;
  for (auto [position, term] : llvm::enumerate(*relation)) {
    if (term.kind == "full_slice") {
      candidate.blockAxes.push_back(-2);
      ++blockedAxes;
      continue;
    }
    if (term.kind == "static_index") {
      candidate.blockAxes.push_back(-1);
      continue;
    }
    if (term.kind == "value_index" && term.operands.size() == 1 &&
        term.operands.front() &&
        !isa<RankedTensorType>(
            operation.getOperand(*term.operands.front()).getType())) {
      candidate.blockAxes.push_back(-1);
      continue;
    }
    if (term.kind == "value_index" && term.operands.size() == 1 &&
        term.operands.front()) {
      Value indexed = operation.getOperand(*term.operands.front());
      auto axes = analysis.getFacts().valueAxes.find(indexed);
      if (axes == analysis.getFacts().valueAxes.end())
        return std::nullopt;
      Operation *blockDomain = nullptr;
      for (const target::LogicalAxis &axis : axes->second) {
        if (!axis.domain || axis.extent == "1")
          continue;
        if (blockDomain && blockDomain != axis.domain)
          return std::nullopt;
        blockDomain = axis.domain;
      }
      if (!blockDomain)
        return std::nullopt;
      llvm::DenseSet<Value> active;
      std::optional<DescriptorAffineExpression> expression =
          descriptorAffineExpression(indexed, analysis.getFacts(), operation,
                                     active);
      if (!expression)
        return std::nullopt;
      auto coefficient = expression->coefficients.find(blockDomain);
      if (coefficient == expression->coefficients.end() ||
          coefficient->second != 1)
        return std::nullopt;
      for (const auto &[domain, value] : expression->coefficients) {
        if (domain == blockDomain || value == 0)
          continue;
        FailureOr<int64_t> node =
            target::getNodeID(*domain, "Triton descriptor affine source");
        intent::plan::AxisOp axis =
            succeeded(node) ? analysis.getAxis(*node) : intent::plan::AxisOp();
        if (axis && !analysis.isScalarAxis(axis.getNode()))
          return std::nullopt;
      }
      FailureOr<int64_t> node =
          target::getNodeID(*blockDomain, "Triton descriptor block axis");
      intent::plan::AxisOp axis =
          succeeded(node) ? analysis.getAxis(*node) : intent::plan::AxisOp();
      if (failed(node) || !axis || analysis.isScalarAxis(axis.getNode()) ||
          (!analysis.getRange(axis.getNode(), "ownership") &&
           !analysis.getRange(axis.getNode(), "traversal") &&
           !analysis.getRange(axis.getNode(), "reduction") &&
           !analysis.getRange(axis.getNode(), "lane")))
        return std::nullopt;
      if (!usedBlockAxes.insert(*node).second)
        return std::nullopt;
      candidate.blockAxes.push_back(*node);
      ++blockedAxes;
      continue;
    }
    if (term.kind != "region_index" || term.operands.size() != 1 ||
        !term.operands.front())
      return std::nullopt;
    Value indexed = operation.getOperand(*term.operands.front());
    FailureOr<int64_t> valueID = target::getValueID(
        indexed, analysis.getKernel(), operation,
        "Triton descriptor region projection");
    auto binding = succeeded(valueID) ? regions.find(*valueID) : regions.end();
    if (failed(valueID) || binding == regions.end())
      return std::nullopt;
    intent::plan::AxisOp axis = analysis.getAxis(binding->second.getAxisNode());
    intent::plan::RangeOp range =
        axis ? analysis.getRange(axis.getNode(), binding->second.getPurpose(),
                                 binding->second.getLevel())
             : intent::plan::RangeOp();
    if (!axis || !range)
      return std::nullopt;
    if (analysis.isScalarAxis(axis.getNode())) {
      candidate.blockAxes.push_back(-1);
    } else {
      if (!usedBlockAxes.insert(axis.getNode()).second)
        return std::nullopt;
      candidate.blockAxes.push_back(axis.getNode());
      ++blockedAxes;
    }
  }
  if (candidate.blockAxes.size() != relation->size() || blockedAxes < 2 ||
      candidate.blockAxes.back() == -1)
    return std::nullopt;
  auto block = llvm::find_if(candidate.blockAxes,
                             [](int64_t axis) { return axis >= 0; });
  bool linear = candidate.blockAxes.back() == -2 &&
                block != candidate.blockAxes.end() &&
                llvm::count_if(candidate.blockAxes,
                               [](int64_t axis) { return axis >= 0; }) == 1 &&
                llvm::all_of(ArrayRef<int64_t>(candidate.blockAxes).drop_back(),
                             [](int64_t axis) { return axis == -1 || axis >= 0; }) &&
                (*relation)[block - candidate.blockAxes.begin()].kind ==
                    "region_index";
  candidate.layout = linear ? "linear" : "strided";
  return std::optional<DescriptorCandidate>(std::move(candidate));
}

bool hasWriteEffect(Operation &operation) {
  bool write = false;
  operation.walk([&](Operation *nested) {
    auto effects = nested->getAttrOfType<ArrayAttr>("intent.effects");
    if (!effects)
      return;
    for (Attribute attribute : effects) {
      auto effect = dyn_cast<DictionaryAttr>(attribute);
      auto kind = effect ? effect.getAs<StringAttr>("kind") : StringAttr();
      write |= kind && kind.getValue() != "read";
    }
  });
  return write;
}

bool supportsStreamPipelineCandidate(
    gpu::PhysicalProgramAnalysis &analysis,
    intent::plan::StreamBindingOp stream) {
  Operation *operation = analysis.getKernel().nodes.lookup(stream.getStreamNode());
  if (!operation || hasWriteEffect(*operation))
    return false;
  bool found = false;
  for (intent::plan::StreamAxisOp streamAxis :
       analysis.getProgram().getBody().getOps<intent::plan::StreamAxisOp>()) {
    if (streamAxis.getStreamNode() != stream.getStreamNode())
      continue;
    intent::plan::AxisOp axis = analysis.getAxis(streamAxis.getAxisNode());
    intent::plan::RangeOp range =
        axis ? analysis.getRange(axis.getNode(), "reduction")
             : intent::plan::RangeOp();
    int64_t logicalExtent = 0;
    if (!axis || !range || !analysis.axisHasRole(axis.getNode(), "contraction_n") ||
        analysis.axisHasRole(axis.getNode(), "contraction_k") ||
        range.getExtent().getAsInteger(10, logicalExtent) ||
        logicalExtent < 128 ||
        !llvm::isPowerOf2_64(static_cast<uint64_t>(logicalExtent)))
      continue;
    StringRef tile = range.getTile();
    int64_t tileExtent = 0;
    if (!tile.consume_front("fixed_") || tile.getAsInteger(10, tileExtent) ||
        tileExtent != logicalExtent || found)
      return false;
    found = true;
  }
  return found;
}

void collectRegionSources(
    Value value, const target::KernelModel &kernel,
    const llvm::DenseMap<int64_t, intent::plan::RegionBindingOp> &regions,
    llvm::DenseSet<int64_t> &result, llvm::DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return;
  auto valueID = kernel.valueIDs.find(value);
  if (valueID != kernel.valueIDs.end() && regions.count(valueID->second)) {
    result.insert(valueID->second);
    return;
  }
  if (Operation *definition = value.getDefiningOp())
    for (Value operand : definition->getOperands())
      collectRegionSources(operand, kernel, regions, result, visited);
}

struct PrefixBoundaryForm {
  int64_t boundaryAxis = -1;
  llvm::SmallVector<int64_t> neutralMasks;
};

std::optional<PrefixBoundaryForm> classifyPrefixBoundaryStream(
    gpu::PhysicalProgramAnalysis &analysis,
    intent::plan::StreamBindingOp stream) {
  target::KernelModel &kernel = analysis.getKernel();
  Operation *operation = kernel.nodes.lookup(stream.getStreamNode());
  if (!operation || target::semanticOperationName(*operation) !=
                        "intent.state_stream")
    return std::nullopt;
  auto stopIndex =
      operation->getAttrOfType<IntegerAttr>("intent.stop_operand_index");
  Operation *stop = stopIndex && stopIndex.getInt() >= 0 &&
                            static_cast<unsigned>(stopIndex.getInt()) <
                                operation->getNumOperands()
                        ? operation->getOperand(stopIndex.getInt()).getDefiningOp()
                        : nullptr;
  if (!stop || target::semanticOperationName(*stop) != "intent.region_end" ||
      stop->getNumOperands() != 1)
    return std::nullopt;

  llvm::DenseMap<int64_t, intent::plan::RegionBindingOp> regions;
  for (intent::plan::RegionBindingOp binding :
       analysis.getProgram().getBody().getOps<intent::plan::RegionBindingOp>())
    regions[binding.getValue()] = binding;
  auto stopValue = kernel.valueIDs.find(stop->getOperand(0));
  auto stopBinding = stopValue != kernel.valueIDs.end()
                         ? regions.find(stopValue->second)
                         : regions.end();
  if (stopBinding == regions.end() ||
      stopBinding->second.getPurpose() != "ownership")
    return std::nullopt;
  int64_t boundaryAxis = stopBinding->second.getAxisNode();

  PrefixBoundaryForm form{boundaryAxis, {}};
  operation->walk([&](Operation *candidate) {
    if (target::semanticOperationName(*candidate) != "intent.compare" ||
        candidate->getNumOperands() != 2 || candidate->getNumResults() != 1)
      return;
    auto predicate = candidate->getAttrOfType<StringAttr>("intent.predicate");
    if (!predicate || predicate.getValue() != "ge")
      return;
    llvm::DenseSet<int64_t> lhsSources;
    llvm::DenseSet<int64_t> rhsSources;
    llvm::DenseSet<Value> visited;
    collectRegionSources(candidate->getOperand(0), kernel, regions, lhsSources,
                         visited);
    visited.clear();
    collectRegionSources(candidate->getOperand(1), kernel, regions, rhsSources,
                         visited);
    if (lhsSources.size() != 1 || rhsSources.size() != 1)
      return;
    intent::plan::RegionBindingOp lhs = regions.lookup(*lhsSources.begin());
    intent::plan::RegionBindingOp rhs = regions.lookup(*rhsSources.begin());
    if (!lhs || !rhs || lhs.getAxisNode() !=
                            static_cast<uint64_t>(boundaryAxis) ||
        lhs.getPurpose() != "ownership" ||
        rhs.getAxisNode() != stream.getAxisNode() ||
        rhs.getPurpose() != "traversal")
      return;
    for (Operation *user : candidate->getResult(0).getUsers()) {
      if (target::semanticOperationName(*user) != "intent.mask" ||
          user->getNumOperands() < 2 || user->getOperand(1) != candidate->getResult(0))
        continue;
      FailureOr<int64_t> node =
          target::getNodeID(*user, "Triton prefix-boundary mask");
      if (succeeded(node))
        form.neutralMasks.push_back(*node);
    }
  });
  if (form.neutralMasks.empty())
    return std::nullopt;
  llvm::sort(form.neutralMasks);
  form.neutralMasks.erase(
      std::unique(form.neutralMasks.begin(), form.neutralMasks.end()),
      form.neutralMasks.end());
  return form;
}

bool isRuntimeABIDimension(const target::KernelModel &kernel,
                           StringRef dimension) {
  for (const target::ABIArgument &argument : kernel.abi.arguments) {
    auto shape = argument.metadata.getAs<ArrayAttr>("shape");
    if (!shape)
      continue;
    for (Attribute extent : shape) {
      auto symbol = dyn_cast<StringAttr>(extent);
      uint64_t constant = 0;
      if (symbol && symbol.getValue() == dimension &&
          symbol.getValue().getAsInteger(10, constant))
        return true;
    }
  }
  return false;
}

LogicalResult realizePointwiseLaneForm(
    gpu::PhysicalProgramAnalysis &analysis,
    intent::plan::SearchSpaceOp searchSpace) {
  if (!searchSpace)
    return success();
  intent::plan::AxisOp pointwiseAxis =
      analysis.getPurePointwiseProgramLane();
  if (!pointwiseAxis)
    return success();
  intent::plan::RangeOp ownership =
      analysis.getRange(pointwiseAxis.getNode(), "ownership");
  if (!ownership || !ownership.getTile().starts_with("program_"))
    return success();
  StringRef ownershipRole = ownership.getTile();
  std::string pointwiseRole = ownershipRole == "program_m"
                                  ? "pointwise_lane"
                              : ownershipRole == "program_n"
                                  ? "pointwise_lane_n"
                                  : "pointwise_lane_" +
                                        ownershipRole.drop_front(8).str();

  auto declarations = searchSpace.getBody().getOps<intent::plan::AutotuneOp>();
  if (!llvm::hasSingleElement(declarations))
    return searchSpace.emitOpError(
        "Triton pointwise-lane form requires one autotune declaration");
  intent::plan::AutotuneOp autotune = *declarations.begin();
  OpBuilder builder(searchSpace.getContext());
  SmallVector<Attribute> parameters(autotune.getParameters().begin(),
                                    autotune.getParameters().end());
  unsigned replaced = 0;
  for (Attribute &parameter : parameters) {
    auto role = dyn_cast<StringAttr>(parameter);
    if (role && role.getValue() == ownership.getTile()) {
      parameter = builder.getStringAttr(pointwiseRole);
      ++replaced;
    }
  }
  if (replaced != 1)
    return autotune.emitOpError(
        "does not declare exactly one innermost pointwise-lane parameter");
  ownership->setAttr("tile", builder.getStringAttr(pointwiseRole));
  autotune->setAttr("parameters", builder.getArrayAttr(parameters));
  return success();
}

LogicalResult realizeProgram(intent::plan::ProgramOp program,
                             intent::plan::SearchSpaceOp searchSpace) {
  FailureOr<std::unique_ptr<gpu::PhysicalProgramAnalysis>> analysis =
      gpu::PhysicalProgramAnalysis::compute(program);
  if (failed(analysis))
    return failure();
  if (failed(realizePointwiseLaneForm(**analysis, searchSpace)))
    return failure();
  analysis = gpu::PhysicalProgramAnalysis::compute(program);
  if (failed(analysis))
    return failure();
  intent::plan::LaunchOp launch = (*analysis)->getLaunch();
  if (!launch)
    return program.emitOpError("has no launch decision for Triton realization");
  if (launch->hasAttr(rowLaunchAttr))
    return launch.emitOpError("already has a Triton row-launch decision");

  intent::plan::AxisOp lane = firstLane(**analysis);
  intent::plan::RangeOp laneRange =
      lane ? (*analysis)->getRange(lane.getNode(), "lane")
           : intent::plan::RangeOp();
  bool hasExternalRead = false;
  bool hasAggregation = false;
  bool hasScan = false;
  (*analysis)->getKernel().entry.walk([&](Operation *operation) {
    StringRef name = ::intent::target::semanticOperationName(*operation);
    hasExternalRead |= name == "intent.view_load";
    hasScan |= name == "intent.scan";
    hasAggregation |= name == "intent.reduce" || name == "intent.arg_reduce" ||
                      name == "intent.scan" || name == "intent.contract" ||
                      name == "intent.sparse_contract" ||
                      name == "intent.state_stream";
  });
  bool configured =
      !searchSpace && !(*analysis)->hasWorkerReuse() && laneRange &&
      laneRange.getTile().starts_with("row_vector") &&
      isRuntimeABIDimension((*analysis)->getKernel(), laneRange.getExtent()) &&
      (hasExternalRead || !hasAggregation) && !hasScan &&
      !target::lowering::hasNonReplayableEffect(
          (*analysis)->getKernel().entry.getOperation());
  OpBuilder builder(program.getContext());
  launch->setAttr(rowLaunchAttr,
                  builder.getStringAttr(configured ? "configured" : "generic"));

  target::KernelModel &kernel = (*analysis)->getKernel();
  for (const auto &entry : kernel.raggedRelations) {
    Operation *ragged = entry.second.operation;
    if (!ragged || ragged->hasAttr(raggedRouteAttr))
      return program.emitOpError(
          "has an invalid or duplicate Triton ragged-route decision");
    FailureOr<StringRef> route =
        target::lowering::classifyRaggedProjection(*ragged);
    if (failed(route))
      return failure();
    ragged->setAttr(raggedRouteAttr, builder.getStringAttr(*route));
  }
  for (intent::plan::ContractOp contract :
       program.getBody().getOps<intent::plan::ContractOp>()) {
    if (contract->hasAttr(contractLoweringAttr) ||
        contract->hasAttr(contractOrientationAttr) ||
        contract->hasAttr(contractBatchedAttr) ||
        contract->hasAttr(scaledContractLayoutAttr))
      return contract.emitOpError("already has a Triton contraction spelling");
    Operation *operation = kernel.nodes.lookup(contract.getNode());
    FailureOr<target::lowering::ContractionOrientation> orientation =
        operation ? target::lowering::contractionOrientation(*operation)
                  : FailureOr<target::lowering::ContractionOrientation>(failure());
    if (failed(orientation))
      return contract.emitOpError(
          "does not bind canonical contraction orientation");
    if (orientation->batched && contract.getForm() != "direct")
      return contract.emitOpError(
          "Triton cannot project the selected contraction form and orientation");
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
      return reduction.emitOpError("already has a Triton reduction spelling");
    Operation *operation = kernel.nodes.lookup(reduction.getNode());
    FailureOr<std::string> role =
        operation ? target::lowering::reductionRole(*operation)
                  : FailureOr<std::string>(failure());
    FailureOr<int64_t> axis =
        operation ? target::lowering::reductionAxis(*operation)
                  : FailureOr<int64_t>(failure());
    if (failed(role) || failed(axis))
      return reduction.emitOpError("does not bind canonical reduction semantics");
    reduction->setAttr(reductionLoweringAttr,
                       builder.getStringAttr(syntax::reduction(*role)));
    reduction->setAttr(reductionAxisAttr, builder.getI64IntegerAttr(*axis));
  }
  for (intent::plan::ScanOp scan :
       program.getBody().getOps<intent::plan::ScanOp>()) {
    if (scan->hasAttr(scanLoweringAttr))
      return scan.emitOpError("already has a Triton scan spelling");
    Operation *operation = kernel.nodes.lookup(scan.getNode());
    FailureOr<std::string> role =
        operation ? target::lowering::scanRole(*operation)
                  : FailureOr<std::string>(failure());
    if (failed(role))
      return scan.emitOpError("does not bind canonical scan semantics");
    scan->setAttr(scanLoweringAttr,
                  builder.getStringAttr(syntax::scan(*role)));
  }
  for (intent::plan::PointwiseOp pointwise :
       program.getBody().getOps<intent::plan::PointwiseOp>()) {
    if (pointwise->hasAttr(pointwiseLoweringAttr))
      return pointwise.emitOpError("already has a Triton pointwise spelling");
    Operation *operation = kernel.nodes.lookup(pointwise.getNode());
    std::string gatherForm;
    if (operation && target::semanticOperationName(*operation) ==
                         "intent.gather") {
      FailureOr<std::string> gather =
          target::lowering::classifyGatherProjection(*operation);
      if (failed(gather))
        return failure();
      if (*gather == "fragment_projection")
        return pointwise.emitOpError(
            "Triton fragment projection form is not materialized");
      gatherForm = std::move(*gather);
    }
    FailureOr<std::string> role =
        operation ? target::lowering::pointwiseRole(*operation)
                  : FailureOr<std::string>(failure());
    FailureOr<StringRef> lowering =
        succeeded(role) ? syntax::pointwise(operation, *role)
                        : FailureOr<StringRef>(failure());
    if (failed(lowering))
      return pointwise.emitOpError("does not bind canonical pointwise semantics");
    pointwise->setAttr(pointwiseLoweringAttr,
                       builder.getStringAttr(*lowering));
    if (!gatherForm.empty())
      pointwise->setAttr(gatherFormAttr, builder.getStringAttr(gatherForm));
  }
  for (intent::plan::TransferOp transfer :
       program.getBody().getOps<intent::plan::TransferOp>()) {
    if (transfer->hasAttr(transferAccessAttr) ||
        transfer->hasAttr(transferFormAttr))
      return transfer.emitOpError("already has a Triton transfer spelling");
    Operation *operation = kernel.nodes.lookup(transfer.getNode());
    StringRef name = operation ? target::semanticOperationName(*operation)
                               : StringRef();
    bool load = name == "intent.view_load";
    bool store = name == "intent.view_store" ||
                 name == "intent.scatter_unique" ||
                 name == "intent.atomic_add" || name == "intent.atomic_cas";
    if (!load && !store)
      return transfer.emitOpError("does not bind canonical transfer semantics");
    transfer->setAttr(transferAccessAttr,
                      builder.getStringAttr(load ? "load" : "store"));
    std::optional<DescriptorCandidate> descriptor =
        operation && searchSpace
            ? descriptorCandidate(**analysis, transfer, *operation)
            : std::nullopt;
    transfer->setAttr(
        transferFormAttr,
        builder.getStringAttr(descriptor ? "pointer_or_descriptor" : "pointer"));
    if (descriptor) {
      transfer->setAttr(descriptorBlockAxesAttr,
                        builder.getDenseI64ArrayAttr(descriptor->blockAxes));
      transfer->setAttr(descriptorLayoutAttr,
                        builder.getStringAttr(descriptor->layout));
    }
  }
  for (intent::plan::StreamBindingOp stream :
       program.getBody().getOps<intent::plan::StreamBindingOp>()) {
    if (stream->hasAttr(streamTileAttr) || stream->hasAttr(streamFormAttr))
      return stream.emitOpError("already has a Triton stream-tile spelling");
    intent::plan::RangeOp range = (*analysis)->getRange(
        stream.getAxisNode(), stream.getPurpose(), stream.getLevel());
    FailureOr<std::string> tile =
        range ? syntax::tile(stream.getOperation(), range.getTile())
              : FailureOr<std::string>(failure());
    if (failed(tile))
      return stream.emitOpError("does not bind one Triton stream tile");
    stream->setAttr(streamTileAttr, builder.getStringAttr(*tile));
    std::optional<PrefixBoundaryForm> prefix =
        classifyPrefixBoundaryStream(**analysis, stream);
    bool pipelineCandidates =
        !prefix && supportsStreamPipelineCandidate(**analysis, stream);
    stream->setAttr(streamFormAttr,
                    builder.getStringAttr(prefix ? "prefix_boundary"
                                                 : pipelineCandidates
                                                       ? "pipeline_candidates"
                                                       : "single"));
    if (prefix) {
      stream->setAttr(streamBoundaryAxisAttr,
                      builder.getI64IntegerAttr(prefix->boundaryAxis));
      stream->setAttr(streamNeutralMasksAttr,
                      builder.getDenseI64ArrayAttr(prefix->neutralMasks));
    }
  }

  return success();
}

class RealizeProviderProgramPass final
    : public PassWrapper<RealizeProviderProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RealizeProviderProgramPass)

  StringRef getArgument() const final {
    return "intent-realize-triton-program-forms";
  }
  StringRef getDescription() const final {
    return "Select Triton-local program forms before terminal source translation";
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
    return "intent-verify-triton-program-forms";
  }
  StringRef getDescription() const final {
    return "Verify the complete Triton provider-program contract";
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
  for (const auto &entry : kernel.raggedRelations) {
    Operation *ragged = entry.second.operation;
    auto route = ragged ? ragged->getAttrOfType<StringAttr>(raggedRouteAttr)
                        : StringAttr();
    if (!route || (route.getValue() != "compact" &&
                   route.getValue() != "indexed"))
      return program.emitOpError(
          "has no complete Triton ragged-route form");
  }
  auto launches = program.getBody().getOps<intent::plan::LaunchOp>();
  if (!llvm::hasSingleElement(launches))
    return program.emitOpError(
        "Triton provider program requires one launch form");
  intent::plan::LaunchOp launch = *launches.begin();
  auto form = launch->getAttrOfType<StringAttr>(rowLaunchAttr);
  if (!form || (form.getValue() != "generic" &&
                form.getValue() != "configured"))
    return launch.emitOpError(
        "requires one realized Triton row-launch form");
  if (form.getValue() == "configured") {
    bool hasReuse = llvm::any_of(
        program.getBody().getOps<intent::plan::AxisOp>(),
        [](intent::plan::AxisOp axis) { return axis.getReuseWorker(); });
    if (searchSpace || hasReuse)
      return launch.emitOpError(
          "configured Triton row launch is incompatible with search or worker reuse");
  }
  for (intent::plan::ContractOp contract :
       program.getBody().getOps<intent::plan::ContractOp>()) {
    auto lowering = contract->getAttrOfType<StringAttr>(contractLoweringAttr);
    auto orientation =
        contract->getAttrOfType<StringAttr>(contractOrientationAttr);
    auto batched = contract->getAttrOfType<BoolAttr>(contractBatchedAttr);
    auto layout = contract->getAttrOfType<StringAttr>(scaledContractLayoutAttr);
    if (!lowering || lowering.getValue().empty() || !orientation || !batched ||
        (orientation.getValue() != "nn" && orientation.getValue() != "nt" &&
         orientation.getValue() != "tn" && orientation.getValue() != "tt") ||
        (contract.getForm() == "scaled_direct" &&
         (!layout || (layout.getValue() != "grouped_rank_two" &&
                      layout.getValue() != "flattened_rank_three"))) ||
        (contract.getForm() != "scaled_direct" && layout))
      return contract.emitOpError(
          "has no complete Triton contraction provider form");
  }
  for (intent::plan::ReductionOp reduction :
       program.getBody().getOps<intent::plan::ReductionOp>()) {
    auto lowering = reduction->getAttrOfType<StringAttr>(reductionLoweringAttr);
    auto axis = reduction->getAttrOfType<IntegerAttr>(reductionAxisAttr);
    if (!lowering || lowering.getValue().empty() || !axis || axis.getInt() < 0 ||
        !kernel.nodes.lookup(reduction.getNode()))
      return reduction.emitOpError("has no complete Triton reduction spelling");
  }
  for (intent::plan::ScanOp scan :
       program.getBody().getOps<intent::plan::ScanOp>()) {
    auto lowering = scan->getAttrOfType<StringAttr>(scanLoweringAttr);
    if (!lowering || lowering.getValue().empty() ||
        !kernel.nodes.lookup(scan.getNode()))
      return scan.emitOpError("has no complete Triton scan spelling");
  }
  for (intent::plan::PointwiseOp pointwise :
       program.getBody().getOps<intent::plan::PointwiseOp>()) {
    auto lowering = pointwise->getAttrOfType<StringAttr>(pointwiseLoweringAttr);
    Operation *operation = kernel.nodes.lookup(pointwise.getNode());
    auto gather = pointwise->getAttrOfType<StringAttr>(gatherFormAttr);
    bool requiresGather =
        operation && target::semanticOperationName(*operation) == "intent.gather";
    if (!lowering || lowering.getValue().empty() || !operation ||
        (requiresGather && (!gather || gather.getValue().empty())))
      return pointwise.emitOpError("has no complete Triton pointwise spelling");
  }
  for (intent::plan::TransferOp transfer :
       program.getBody().getOps<intent::plan::TransferOp>()) {
    auto access = transfer->getAttrOfType<StringAttr>(transferAccessAttr);
    auto transferForm = transfer->getAttrOfType<StringAttr>(transferFormAttr);
    auto descriptorAxes =
        transfer->getAttrOfType<DenseI64ArrayAttr>(descriptorBlockAxesAttr);
    auto descriptorLayout =
        transfer->getAttrOfType<StringAttr>(descriptorLayoutAttr);
    Operation *operation = kernel.nodes.lookup(transfer.getNode());
    auto view = operation && operation->getNumOperands() > 0
                    ? dyn_cast<intent::ViewType>(operation->getOperand(0).getType())
                    : intent::ViewType();
    auto tensor = view ? dyn_cast<RankedTensorType>(view.getTensor())
                       : RankedTensorType();
    bool linearLayout =
        descriptorAxes && descriptorLayout &&
        descriptorLayout.getValue() == "linear" && !descriptorAxes.empty() &&
        descriptorAxes.asArrayRef().back() == -2 &&
        llvm::count_if(descriptorAxes.asArrayRef(),
                       [](int64_t axis) { return axis >= 0; }) == 1 &&
        llvm::all_of(descriptorAxes.asArrayRef().drop_back(),
                     [](int64_t axis) { return axis == -1 || axis >= 0; });
    bool validDescriptor =
        (!descriptorAxes && !descriptorLayout) ||
        (transferForm && transferForm.getValue() == "pointer_or_descriptor" &&
         descriptorAxes && descriptorLayout &&
         (linearLayout || descriptorLayout.getValue() == "strided") &&
         tensor && static_cast<int64_t>(descriptorAxes.size()) == tensor.getRank() &&
         llvm::count_if(descriptorAxes.asArrayRef(),
                        [](int64_t axis) { return axis >= 0 || axis == -2; }) >= 2 &&
         llvm::all_of(descriptorAxes.asArrayRef(), [&](int64_t axis) {
           return axis == -1 || axis == -2 || llvm::any_of(
                                                  program.getBody().getOps<
                                                      intent::plan::AxisOp>(),
                                                  [&](intent::plan::AxisOp value) {
                                                    return value.getNode() ==
                                                           static_cast<uint64_t>(axis);
                                                  });
         }));
    if (!access || (access.getValue() != "load" &&
                    access.getValue() != "store") ||
        !transferForm ||
        (transferForm.getValue() != "pointer" &&
         transferForm.getValue() != "pointer_or_descriptor") ||
        !operation || !validDescriptor ||
        (transferForm.getValue() == "pointer_or_descriptor" &&
         (!descriptorAxes || !descriptorLayout)) ||
        (transferForm.getValue() == "pointer" &&
         (descriptorAxes || descriptorLayout)))
      return transfer.emitOpError("has no complete Triton transfer spelling");
  }
  for (intent::plan::StreamBindingOp stream :
       program.getBody().getOps<intent::plan::StreamBindingOp>()) {
    auto tile = stream->getAttrOfType<StringAttr>(streamTileAttr);
    auto form = stream->getAttrOfType<StringAttr>(streamFormAttr);
    auto boundary = stream->getAttrOfType<IntegerAttr>(streamBoundaryAxisAttr);
    auto masks = stream->getAttrOfType<DenseI64ArrayAttr>(streamNeutralMasksAttr);
    bool validBoundary = !boundary || llvm::any_of(
                                          program.getBody().getOps<
                                              intent::plan::AxisOp>(),
                                          [&](intent::plan::AxisOp axis) {
                                            return axis.getNode() ==
                                                   static_cast<uint64_t>(
                                                       boundary.getInt());
                                          });
    bool validMasks = !masks || llvm::all_of(
                                    masks.asArrayRef(), [&](int64_t node) {
                                      Operation *operation = kernel.nodes.lookup(node);
                                      return operation &&
                                             target::semanticOperationName(*operation) ==
                                                 "intent.mask";
                                    });
    if (!tile || tile.getValue().empty() || !form ||
        (form.getValue() != "single" && form.getValue() != "prefix_boundary" &&
         form.getValue() != "pipeline_candidates") ||
        (form.getValue() == "prefix_boundary" &&
         (!boundary || !masks || masks.empty() || !validBoundary ||
          !validMasks)) ||
        (form.getValue() != "prefix_boundary" && (boundary || masks)))
      return stream.emitOpError("has no complete Triton stream-tile spelling");
  }
  return success();
}

void addProviderPasses(PassManager &manager) {
  manager.addPass(std::make_unique<RealizeProviderProgramPass>());
  manager.addPass(std::make_unique<VerifyProviderProgramPass>());
}

} // namespace intent::triton::lowering
