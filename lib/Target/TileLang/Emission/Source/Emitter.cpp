#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::tilelang::emission {
namespace {

std::string dimensionSpelling(StringRef symbol) {
  return symbol == "T" ? "DIM_T" : symbol.str();
}

FailureOr<std::string> tileSpelling(Operation *operation, StringRef role) {
  if (role == "one")
    return std::string("1");
  if (role.starts_with("fixed_"))
    return role.drop_front(6).str();
  if (role == "row_vector")
    return std::string("TILE_SIZE");
  if (role.starts_with("row_vector_"))
    return "TILE_SIZE_V" + role.drop_front(11).str();
  if (role == "lane_pack")
    return std::string("TILE_SIZE_L");
  if (role.starts_with("lane_pack_"))
    return "TILE_SIZE_L" + role.drop_front(10).str();
  if (role == "program_m" || role == "ragged_member")
    return std::string("TILE_SIZE_M");
  if (role.starts_with("ragged_member_"))
    return "TILE_SIZE_R" + role.drop_front(14).str();
  if (role.starts_with("query_"))
    return "TILE_SIZE_Q" + role.drop_front(6).str();
  if (role == "query")
    return std::string("TILE_SIZE_Q");
  if (role == "program_n" || role == "stream")
    return std::string("TILE_SIZE_N");
  if (role == "scan")
    return std::string("TILE_SIZE_SCAN");
  if (role.starts_with("scan_"))
    return "TILE_SIZE_SCAN" + role.drop_front(5).str();
  if (role == "stream_contract")
    return std::string("TILE_SIZE_K");
  if (role.starts_with("stream_contract_"))
    return "TILE_SIZE_C" + role.drop_front(16).str();
  if (role.starts_with("stream_"))
    return "TILE_SIZE_S" + role.drop_front(7).str();
  if (role.starts_with("program_"))
    return "TILE_SIZE_P" + role.drop_front(8).str();
  if (role == "reduction")
    return std::string("TILE_SIZE_K");
  if (role.starts_with("reduction_"))
    return "TILE_SIZE_K" + role.drop_front(10).str();
  operation->emitOpError("has no TileLang tile spelling for role ") << role;
  return failure();
}

StringRef bufferSpace(StringRef space) {
  if (space == "external" || space == "workspace")
    return "global";
  if (space == "shared")
    return "shared";
  if (space == "private_fragment")
    return "fragment";
  if (space == "private_scalar")
    return "local";
  if (space == "none")
    return "none";
  return {};
}

FailureOr<StringRef> pointwiseSpelling(Operation *operation, StringRef role,
                                       StringRef materialization) {
  if (role == "indices")
    return StringRef("logical_indices");
  if (role == "counter_random_f32")
    return StringRef("counter_xorshift32");
  if (role == "broadcast")
    return StringRef("alias");
  if (role == "cast")
    return materialization == "contract_operand" ? StringRef("T.copy_cast")
                                                   : StringRef("T.cast");
  if (role == "reshape")
    return StringRef("T.reshape");
  if (role == "transpose")
    return StringRef("fragment_permute");
  if (role == "unary_exp")
    return StringRef("T.exp");
  if (role == "unary_exp2")
    return StringRef("T.exp2");
  if (role == "unary_log")
    return StringRef("T.log");
  if (role == "unary_rsqrt")
    return StringRef("T.rsqrt");
  if (role == "unary_sigmoid")
    return StringRef("T.sigmoid");
  if (role == "unary_negate")
    return StringRef("python_negate");
  if (role == "unary_not")
    return StringRef("python_not");
  if (role == "binary_add")
    return StringRef("python_add");
  if (role == "binary_subtract")
    return StringRef("python_subtract");
  if (role == "binary_multiply")
    return StringRef("python_multiply");
  if (role == "binary_true_divide")
    return StringRef("python_true_divide");
  if (role == "binary_floor_divide")
    return StringRef("python_floor_divide");
  if (role == "binary_remainder")
    return StringRef("python_remainder");
  if (role == "binary_bitwise_and")
    return StringRef("T.bitwise_and");
  if (role == "binary_bitwise_or")
    return StringRef("T.bitwise_or");
  if (role == "binary_bitwise_xor")
    return StringRef("T.bitwise_xor");
  if (role == "binary_left_shift")
    return StringRef("T.shift_left");
  if (role == "binary_right_shift")
    return StringRef("T.shift_right");
  if (role == "binary_maximum")
    return StringRef("T.max");
  if (role == "binary_minimum")
    return StringRef("T.min");
  if (role == "compare_equal")
    return StringRef("python_equal");
  if (role == "compare_not_equal")
    return StringRef("python_not_equal");
  if (role == "compare_less")
    return StringRef("python_less");
  if (role == "compare_less_equal")
    return StringRef("python_less_equal");
  if (role == "compare_greater")
    return StringRef("python_greater");
  if (role == "compare_greater_equal")
    return StringRef("python_greater_equal");
  if (role == "mask")
    return StringRef("T.if_then_else");
  if (role == "select")
    return StringRef("T.if_then_else");
  if (role == "full")
    return StringRef("T.fill");
  if (role == "zeros")
    return StringRef("T.clear");
  if (role == "members")
    return StringRef("T.members");
  if (role == "expand_dims")
    return StringRef("expand_dims");
  if (role == "indirect_gather")
    return StringRef("T.indirect_gather");
  operation->emitOpError("has no TileLang pointwise spelling for role ") << role;
  return failure();
}

FailureOr<std::string> parameterSpelling(Operation *operation, StringRef role) {
  if (role == "lane_pack")
    return std::string("TILE_SIZE_L");
  if (role.starts_with("lane_pack_"))
    return "TILE_SIZE_L" + role.drop_front(10).str();
  if (role == "program_m" || role == "ragged_member")
    return std::string("TILE_SIZE_M");
  if (role.starts_with("ragged_member_"))
    return "TILE_SIZE_R" + role.drop_front(14).str();
  if (role.starts_with("query_"))
    return "TILE_SIZE_Q" + role.drop_front(6).str();
  if (role == "query")
    return std::string("TILE_SIZE_Q");
  if (role == "program_n" || role == "feature" || role == "stream")
    return std::string("TILE_SIZE_N");
  if (role == "scan")
    return std::string("TILE_SIZE_SCAN");
  if (role.starts_with("scan_"))
    return "TILE_SIZE_SCAN" + role.drop_front(5).str();
  if (role == "stream_contract")
    return std::string("TILE_SIZE_K");
  if (role.starts_with("stream_contract_"))
    return "TILE_SIZE_C" + role.drop_front(16).str();
  if (role.starts_with("stream_"))
    return "TILE_SIZE_S" + role.drop_front(7).str();
  if (role.starts_with("program_"))
    return "TILE_SIZE_P" + role.drop_front(8).str();
  if (role == "reduction")
    return std::string("TILE_SIZE_K");
  if (role.starts_with("reduction_"))
    return "TILE_SIZE_K" + role.drop_front(10).str();
  if (role == "group_m")
    return std::string("GROUP_SIZE_M");
  if (role.starts_with("group_"))
    return "GROUP_SIZE_G" + role.drop_front(6).str();
  operation->emitOpError("has no TileLang tuner parameter for role ") << role;
  return failure();
}

bool workerReuse(const RealizationIndex &index) {
  return llvm::any_of(index.axesByRole, [](const auto &binding) {
    return binding.getValue().getReuseWorker();
  });
}

bool feedsAtomicValue(Operation &operation) {
  if (operation.getNumResults() != 1 ||
      !llvm::hasSingleElement(operation.getResult(0).getUsers()))
    return false;
  Operation *user = *operation.getResult(0).user_begin();
  auto valueIndex =
      user->getAttrOfType<IntegerAttr>("intent.value_operand_index");
  return user->getName().getStringRef() == "intent.atomic_add" && valueIndex &&
         valueIndex.getInt() >= 0 &&
         static_cast<unsigned>(valueIndex.getInt()) < user->getNumOperands() &&
         user->getOperand(valueIndex.getInt()) == operation.getResult(0);
}

} // namespace

FailureOr<RealizationIndex>
indexRealization(intent::plan::RealizationOp realization,
                 const target::KernelModel &kernel) {
  RealizationIndex index;
  SmallVector<intent::plan::ReductionOp> reductions;
  SmallVector<intent::plan::ScanOp> scans;
  SmallVector<intent::plan::PointwiseOp> pointwise;
  SmallVector<intent::plan::TransferOp> transfers;
  SmallVector<intent::plan::RangeOp> ranges;
  for (Operation &operation : realization.getBody().front()) {
    if (isa<intent::plan::YieldOp>(operation))
      continue;
    if (auto value = dyn_cast<intent::plan::DeviceOp>(operation)) {
      index.target.operation = value;
    } else if (auto value = dyn_cast<intent::plan::AxisOp>(operation)) {
      plan::AxisOp binding;
      binding.operation = value;
      if (std::optional<StringRef> group = value.getGroup()) {
        FailureOr<std::string> spelling = parameterSpelling(value, *group);
        if (failed(spelling))
          return failure();
        binding.group = *spelling;
      }
      index.axes.try_emplace(value.getNode(), binding);
    } else if (auto value = dyn_cast<intent::plan::RangeOp>(operation)) {
      ranges.push_back(value);
    } else if (auto value = dyn_cast<intent::plan::ProgramOp>(operation)) {
      index.program.operation = value;
    } else if (auto value = dyn_cast<intent::plan::BlockExtentOp>(operation)) {
      plan::BlockExtentOp binding;
      binding.operation = value;
      index.blockExtents[value.getLogicalExtent()] = binding;
    } else if (auto value = dyn_cast<intent::plan::BufferOp>(operation)) {
      Operation *buffer = kernel.nodes.lookup(value.getNode());
      if (!buffer || buffer->getName().getStringRef() != "intent.buffer" ||
          (value.getSpace() != "private_scalar_array" &&
           value.getSpace() != "private_vector" &&
           value.getSpace() != "private_workspace"))
        return value.emitOpError("does not bind a logical buffer residency");
      plan::BufferOp binding;
      binding.operation = value;
      index.buffers[value.getNode()] = binding;
    } else if (auto value = dyn_cast<intent::plan::PaddingOp>(operation)) {
      plan::PaddingOp binding;
      binding.operation = value;
      index.paddings[value.getValue()] = binding;
    } else if (auto value = dyn_cast<intent::plan::ReductionOp>(operation)) {
      reductions.push_back(value);
    } else if (auto value = dyn_cast<intent::plan::ScanOp>(operation)) {
      scans.push_back(value);
    } else if (auto value = dyn_cast<intent::plan::PointwiseOp>(operation)) {
      pointwise.push_back(value);
    } else if (auto value = dyn_cast<intent::plan::ContractOp>(operation)) {
      plan::ContractOp binding;
      binding.operation = value;
      binding.lowering = "T.gemm";
      binding.lhsSpace = bufferSpace(value.getLhsSpace()).str();
      binding.rhsSpace = bufferSpace(value.getRhsSpace()).str();
      binding.accumulatorSpace =
          bufferSpace(value.getAccumulatorSpace()).str();
      if (binding.lhsSpace.empty() || binding.rhsSpace.empty() ||
          binding.accumulatorSpace.empty()) {
        value.emitOpError("has no TileLang contraction residency spelling");
        return failure();
      }
      index.contracts[value.getNode()] = binding;
    } else if (auto value = dyn_cast<intent::plan::TransferOp>(operation)) {
      transfers.push_back(value);
    } else if (auto value = dyn_cast<intent::plan::StageOp>(operation)) {
      plan::StageOp binding;
      binding.operation = value;
      binding.position = index.stages.size();
      index.stages.push_back(binding);
    } else if (auto value = dyn_cast<intent::plan::StageAxisOp>(operation)) {
      plan::StageAxisOp binding;
      binding.operation = value;
      index.stageAxes[value.getStageNode()][value.getRole()] = binding;
    }
  }
  if (failed(target::emission::indexAxisRanges(index, ranges, tileSpelling)) ||
      failed(target::emission::indexCanonicalStructure(index, kernel)))
    return failure();
  target::emission::indexAxisRoles(index);
  for (auto &entry : index.streams) {
    plan::StreamOp &binding = entry.second;
    plan::AxisOp axis = index.axes.lookup(binding.getAxisNode());
    const target::emission::RangeBinding *traversal =
        axis ? axis.getRange("traversal", 0) : nullptr;
    FailureOr<std::string> tile =
        traversal ? tileSpelling(binding.operation, traversal->getTileRole())
                  : FailureOr<std::string>(failure());
    if (failed(tile))
      return binding.emitOpError("does not bind an ordered physical axis");
    binding.tile = *tile;
  }
  index.components = target::emission::indexPhysicalComponents(index);
  bool rowStrided = workerReuse(index);
  for (intent::plan::ReductionOp value : reductions) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    FailureOr<std::string> role =
        operation ? target::emission::reductionRole(*operation)
                  : FailureOr<std::string>(failure());
    FailureOr<int64_t> axis =
        operation ? target::emission::reductionAxis(*operation)
                  : FailureOr<int64_t>(failure());
    if (failed(role) || failed(axis))
      return value.emitOpError("does not bind a canonical reduction");
    plan::ReductionOp binding;
    binding.operation = value;
    binding.lowering = *role == "reduce_argmax"
                           ? "T.reduce_max_with_index"
                       : *role == "reduce_maximum" ? "T.reduce_max"
                       : *role == "reduce_any"     ? "T.reduce_any_i32"
                       : *role == "reduce_all"     ? "T.reduce_all_i32"
                                                    : "T.reduce_sum";
    binding.resultSpace = bufferSpace(value.getResultSpace()).str();
    binding.axis = *axis;
    if (binding.resultSpace.empty())
      return value.emitOpError("has no TileLang reduction residency spelling");
    index.reductions[value.getNode()] = binding;
  }
  for (intent::plan::ScanOp value : scans) {
    if (value.getSemantics() != "scan_inclusive_add" ||
        !index.axes.count(value.getAxisNode()))
      return value.emitOpError("does not bind a canonical scan");
    plan::ScanOp binding;
    binding.operation = value;
    binding.lowering = "T.cumsum";
    binding.resultSpace = value.getResultSpace() == "private_workspace"
                              ? "global"
                              : bufferSpace(value.getResultSpace()).str();
    binding.axis = value.getTensorAxis();
    binding.axisNode = value.getAxisNode();
    if (binding.resultSpace.empty())
      return value.emitOpError("has no TileLang scan residency spelling");
    index.scans[value.getNode()] = binding;
  }
  for (intent::plan::PointwiseOp value : pointwise) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    FailureOr<std::string> role =
        operation ? target::emission::pointwiseRole(*operation)
                  : FailureOr<std::string>(failure());
    std::string materialization =
        operation && target::emission::feedsContraction(*operation)
            ? "contract_operand"
            : "elementwise";
    FailureOr<StringRef> lowering =
        succeeded(role)
            ? pointwiseSpelling(value, *role, materialization)
            : FailureOr<StringRef>(failure());
    plan::PointwiseOp binding;
    binding.operation = value;
    binding.resultSpace = bufferSpace(value.getResultSpace()).str();
    if (failed(lowering) || binding.resultSpace.empty())
      return value.emitOpError("does not bind canonical pointwise semantics");
    binding.lowering = lowering->str();
    binding.defer = target::emission::feedsStagedContraction(index, *operation);
    index.pointwise[value.getNode()] = binding;
  }
  for (intent::plan::TransferOp value : transfers) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    bool load = operation &&
                operation->getName().getStringRef() == "intent.view_load";
    bool store = operation &&
                 (operation->getName().getStringRef() == "intent.view_store" ||
                  operation->getName().getStringRef() == "intent.scatter_unique" ||
                  operation->getName().getStringRef() == "intent.atomic_add" ||
                  operation->getName().getStringRef() == "intent.atomic_cas");
    if (!load && !store)
      return value.emitOpError("does not bind a canonical transfer");
    FailureOr<bool> derivedScalar =
        target::hasDerivedScalarIndex(*operation);
    FailureOr<bool> tensorIndirect =
        target::hasTensorIndirectIndex(*operation);
    if (failed(derivedScalar) || failed(tensorIndirect))
      return failure();
    plan::BoundaryOp binding;
    binding.operation = value;
    binding.access = rowStrided ? (load ? "gather" : "scatter")
                                : (load ? "load" : "store");
    bool raggedBound = llvm::any_of(value.getDomainNodes(), [&](int64_t axis) {
      return target::emission::isRaggedBoundAxis(index.components, axis);
    });
    bool materializeLogicalBounds =
        raggedBound && !value.getConsumerNeutralized();
    bool packedScalar =
        target::emission::hasPackedScalarDomain(index, binding);
    binding.transfer = *derivedScalar || *tensorIndirect ||
                               materializeLogicalBounds
                           ? "parallel_elements"
                           : "bulk_copy";
    binding.resultSpace = bufferSpace(value.getResultSpace()).str();
    binding.defer =
        load && (target::emission::deferSharedContractionTransfer(
                     index, *operation, value.getResultSpace()) ||
                 (index.stages.empty() && feedsAtomicValue(*operation)));
    binding.explicitBounds =
        rowStrided || materializeLogicalBounds ||
        (!index.stages.empty() && store) || *derivedScalar || *tensorIndirect ||
        packedScalar;
    if (binding.resultSpace.empty()) {
      value.emitOpError("has no TileLang transfer residency spelling");
      return failure();
    }
    index.boundaries[value.getNode()] = binding;
  }
  if (!index.target || !index.program) {
    realization.emitOpError("lacks TileLang target or program choices");
    return failure();
  }
  return index;
}

FailureOr<SearchIndex>
indexSearchSpace(intent::plan::SearchSpaceOp searchSpace) {
  SearchIndex index;
  if (!searchSpace)
    return index;
  for (Operation &operation : searchSpace.getBody().front()) {
    if (auto autotune = dyn_cast<intent::plan::AutotuneOp>(operation)) {
      SmallVector<NamedAttribute> mappings;
      OpBuilder builder(searchSpace.getContext());
      for (Attribute attribute : autotune.getParameters()) {
        StringRef role = cast<StringAttr>(attribute).getValue();
        FailureOr<std::string> spelling = parameterSpelling(autotune, role);
        if (failed(spelling))
          return failure();
        mappings.push_back(builder.getNamedAttr(*spelling,
                                               builder.getStringAttr(role)));
      }
      index.autotune.operation = autotune;
      index.autotune.parameterMap = builder.getDictionaryAttr(mappings);
    }
  }
  return index;
}

SourceEmitter::SourceEmitter(target::KernelModel kernel,
                             intent::plan::RealizationOp realization,
                             intent::plan::SearchSpaceOp searchSpace,
                             RealizationIndex planIndex,
                             SearchIndex searchIndex, raw_ostream &output)
    : kernel(std::move(kernel)), realization(realization),
      searchSpace(searchSpace), planIndex(std::move(planIndex)),
      searchIndex(std::move(searchIndex)), output(output) {}

LogicalResult SourceEmitter::emit() {
  return target::emitSource(*this);
}

LogicalResult SourceEmitter::prepare() {
  if (failed(indexABI()))
    return failure();
  if (!planIndex.ragged.empty() && failed(prepareRaggedMetadata()))
    return failure();
  if (failed(resolvePhysicalBindings()))
    return failure();
  if (failed(target::emission::indexScanProducerOperations(
          kernel, planIndex, scanProducerOwners)))
    return failure();
  for (const auto &entry : planIndex.scans)
    if (failed(target::emission::verifyScanMaterializedValues(kernel,
                                                              entry.second)))
      return failure();
  if (failed(preparePrivateWorkspaces()))
    return failure();
  if (!planIndex.stages.empty())
    return prepareRaggedStages();
  return success();
}

LogicalResult SourceEmitter::preparePrivateWorkspaces() {
  SmallVector<int64_t> nodes;
  for (const auto &entry : planIndex.buffers)
    if (entry.second.getSpace() == "private_workspace")
      nodes.push_back(entry.first);
  llvm::sort(nodes);
  for (int64_t node : nodes) {
    plan::BufferOp binding = planIndex.buffers.lookup(node);
    Operation *buffer = kernel.nodes.lookup(node);
    FailureOr<target::LogicalBufferInfo> info =
        buffer ? target::getLogicalBufferInfo(*buffer)
               : FailureOr<target::LogicalBufferInfo>(failure());
    if (!buffer || failed(info) || info->shape.empty() ||
        buffer->getNumResults() != 1)
      return binding.emitOpError(
          "does not bind a logical private workspace");
    workspaceNames[buffer->getResult(0)] =
        "workspace_" + std::to_string(node);
    privateWorkspaceBuffers.push_back(buffer);
  }
  for (const auto &entry : planIndex.scans) {
    plan::ScanOp binding = entry.second;
    if (binding.getResultSpace() != "global")
      continue;
    Operation *scan = kernel.nodes.lookup(entry.first);
    Operation *axis = kernel.nodes.lookup(binding.getAxisNode());
    FailureOr<std::string> extent =
        axis ? dimensionName(*axis) : FailureOr<std::string>(failure());
    if (!scan || scan->getName().getStringRef() != "intent.scan" ||
        scan->getNumResults() != 1 || failed(extent))
      return binding.emitOpError(
          "does not bind a workspace-backed TileLang scan tensor");
    workspaceNames[scan->getResult(0)] =
        "scan_workspace_" + std::to_string(entry.first);
    scanResults[scan->getResult(0)] = binding;
    scanExtents[entry.first] = *extent;
    for (int64_t valueID : binding.getMaterializedValues()) {
      FailureOr<Value> value = target::emission::lookupScanMaterializedValue(
          kernel, binding, valueID);
      if (failed(value))
        return failure();
      workspaceNames[*value] =
          "scan_materialized_" + std::to_string(valueID);
      scanMaterializedValues[*value] = binding;
    }
  }
  return success();
}

LogicalResult SourceEmitter::registerOperationHandlers(
    target::OperationHandlerRegistry &registry) {
  return registerEmissionHandlers(registry, *this);
}

LogicalResult SourceEmitter::indexABI() {
  views.reserve(kernel.abi.arguments.size());
  for (const target::ABIArgument &argument : kernel.abi.arguments) {
    auto view = dyn_cast<intent::ViewType>(argument.type);
    if (!view) {
      auto kind = argument.metadata.getAs<StringAttr>("kind");
      if (!kind)
        return kernel.entry.emitOpError()
               << "ABI value " << argument.valueID << " has no parameter kind";
      if (kind.getValue() == "constexpr")
        continue;
      if (kind.getValue() != "runtime_scalar" ||
          !isa<FloatType, IntegerType, IndexType>(argument.type))
        return kernel.entry.emitOpError()
               << "ABI value " << argument.valueID
               << " has no TileLang runtime-scalar binding";
      scalars.push_back(ABIScalar{&argument, argument.type, argument.name});
      valueNames[argument.value] = argument.name;
      continue;
    }
    auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
    if (!tensor)
      return kernel.entry.emitOpError("TileLang emitter requires ranked views");
    ABIView emitted{&argument, view, tensor, {}};
    auto shape = argument.metadata.getAs<ArrayAttr>("shape");
    if (!shape || shape.size() != static_cast<size_t>(tensor.getRank()))
      return kernel.entry.emitOpError()
             << "view " << argument.name << " has incomplete shape metadata";
    for (auto [axis, extent] : llvm::enumerate(shape)) {
      if (auto symbol = dyn_cast<StringAttr>(extent)) {
        uint64_t staticExtent = 0;
        if (!symbol.getValue().getAsInteger(10, staticExtent)) {
          emitted.shape.push_back(symbol.getValue().str());
          continue;
        }
        std::string spelling = dimensionSpelling(symbol.getValue());
        emitted.shape.push_back(spelling);
        if (!dimensionOwners.count(spelling)) {
          dimensionOwners[spelling] =
              argument.name + ".shape[" + std::to_string(axis) + "]";
          dimensionOrder.push_back(std::move(spelling));
        }
      } else if (auto integer = dyn_cast<IntegerAttr>(extent)) {
        emitted.shape.push_back(std::to_string(integer.getInt()));
      } else {
        return kernel.entry.emitOpError("contains an unsupported shape extent");
      }
    }
    unsigned position = views.size();
    views.push_back(std::move(emitted));
    viewPositions[argument.value] = position;
    valueNames[argument.value] = argument.name;
  }
  if (views.empty())
    return kernel.entry.emitOpError("TileLang emitter requires external views");
  return success();
}

LogicalResult SourceEmitter::resolvePhysicalBindings() {
  programRoot = kernel.nodes.lookup(planIndex.program.getLoopNode());
  if (!programRoot || programRoot->getName().getStringRef() != "intent.parallel")
    return planIndex.program.emitOpError("does not bind an intent.parallel op");
  for (auto &entry : planIndex.axesByRole) {
    Operation *domain = kernel.nodes.lookup(entry.getValue().getNode());
    if (!domain ||
        (domain->getName().getStringRef() != "intent.domain" &&
         domain->getName().getStringRef() != "intent.ragged_outer" &&
         domain->getName().getStringRef() != "intent.ragged_member"))
      return entry.getValue().emitOpError("does not bind a logical domain op");
    FailureOr<std::string> dimension = dimensionName(*domain);
    if (failed(dimension))
      return entry.getValue().emitOpError("cannot resolve its source dimension");
    roleDimensions[entry.getValue().getRole()] = *dimension;
  }
  for (auto &entry : planIndex.axes) {
    Operation *domain = kernel.nodes.lookup(entry.first);
    if (!domain)
      return entry.second.emitOpError("does not bind a logical domain op");
    if (domain->getNumResults() != 1)
      return entry.second.emitOpError("binds a domain without one canonical value");
    FailureOr<int64_t> valueID = target::getValueID(
        domain->getResult(0), kernel, *domain, "TileLang domain tile binding");
    if (failed(valueID))
      return failure();
    std::string tile = entry.second.getTile().str();
    if (!entry.second.getReuseWorker() &&
        entry.second.getTileRole().starts_with("row_vector")) {
      FailureOr<std::string> dimension = dimensionName(*domain);
      if (failed(dimension))
        return entry.second.emitOpError("cannot resolve its row-vector extent");
      tile = physicalExtent(*dimension);
    }
    regionTiles["?region_" + std::to_string(*valueID) + "_0"] =
        tile;
  }
  for (auto &entry : planIndex.paddings) {
    Value value = kernel.values.lookup(entry.first);
    if (!value || !isa<RankedTensorType>(value.getType()))
      return entry.second.emitOpError("does not bind a ranked tensor value");
  }
  for (const target::RegionNode &region : kernel.regions.nodes) {
    Operation *operation = region.operation;
    StringRef name = operation->getName().getStringRef();
    if (name != "intent.parallel" && name != "intent.state_stream")
      continue;
    FailureOr<SmallVector<Operation *>> domains =
        operation->getNumOperands() > 0
            ? target::expandDomainSource(operation->getOperand(0), *operation)
            : FailureOr<SmallVector<Operation *>>(failure());
    auto regions = operation->getAttrOfType<ArrayAttr>(
        "intent.region_argument_nodes");
    auto blocks = regions && !regions.empty() ? dyn_cast<ArrayAttr>(regions[0])
                                              : ArrayAttr();
    auto arguments = blocks && !blocks.empty() ? dyn_cast<ArrayAttr>(blocks[0])
                                               : ArrayAttr();
    if (failed(domains) || !arguments || arguments.size() < domains->size())
      return operation->emitOpError(
          "cannot index its region shape against the TileLang plan");
    for (auto [index, domain] : llvm::enumerate(*domains)) {
      FailureOr<int64_t> domainNode =
          target::getNodeID(*domain, "region tile indexing");
      plan::AxisOp axis = succeeded(domainNode)
                              ? planIndex.axes.lookup(*domainNode)
                              : plan::AxisOp();
      auto argument = dyn_cast<IntegerAttr>(arguments[index]);
      StringRef purpose = name == "intent.parallel" ? "ownership" : "traversal";
      const target::emission::RangeBinding *range =
          axis ? axis.getRange(purpose, 0) : nullptr;
      if (failed(domainNode) || !axis || !argument || !range)
        return operation->emitOpError(
            "cannot index its region axis against the TileLang plan");
      regionTiles["?region_" + std::to_string(argument.getInt()) + "_0"] =
          range->getTile().str();
    }
  }

  if (!planIndex.components.reusedAxes.empty()) {
    if (searchSpace)
      return realization.emitOpError(
          "fixed TileLang rows cannot consume a search space");
    auto lane = planIndex.axesByRole.find("lane_0");
    if (!planIndex.axesByRole.count("program_0") ||
        lane == planIndex.axesByRole.end())
      return realization.emitOpError(
          "fixed TileLang rows require program_0 and lane_0 axes");
    vectorDomain = kernel.nodes.lookup(lane->second.getNode());
    for (ABIView &view : views) {
      if ((view.view.getAccess() == "out" ||
           view.view.getAccess() == "inout") &&
          !fixedOutput)
        fixedOutput = &view;
    }
    if (!fixedOutput || fixedOutput->tensor.getRank() < 1)
      return kernel.entry.emitOpError(
          "fixed-row TileLang program requires one ranked output view");
  }
  if (target::emission::requiresDelegatedTuning(planIndex) &&
      (!searchSpace || !searchIndex.autotune))
    return realization.emitOpError(
        "tiled physical components require a delegated TileLang tuner");
  SmallVector<StringRef> logicalBlockExtents;
  logicalBlockExtents.reserve(planIndex.blockExtents.size());
  for (const auto &entry : planIndex.blockExtents)
    logicalBlockExtents.push_back(entry.getKey());
  llvm::sort(logicalBlockExtents);
  for (StringRef logicalExtent : logicalBlockExtents)
    blockExtentConstants.emplace_back(physicalExtent(logicalExtent),
                                      dimensionSpelling(logicalExtent));
  return success();
}

LogicalResult SourceEmitter::prepareRaggedMetadata() {
  raggedRuntimes.reserve(planIndex.ragged.size());
  for (plan::RaggedOp ragged : planIndex.ragged) {
    RaggedRuntime runtime;
    runtime.binding = ragged;
    runtime.relation = kernel.nodes.lookup(ragged.getNode());
    runtime.outer = kernel.nodes.lookup(ragged.getOuterNode());
    StringRef outerName = runtime.outer
                              ? runtime.outer->getName().getStringRef()
                              : StringRef();
    if (!runtime.relation ||
        runtime.relation->getName().getStringRef() != "intent.ragged" ||
        (outerName != "intent.ragged_outer" &&
         outerName != "intent.ragged_member") ||
        ragged.getMemberNodes().empty())
      return ragged.emitOpError(
          "does not resolve to canonical ragged ownership operations");
    for (int64_t memberNode : ragged.getMemberNodes()) {
      Operation *member = kernel.nodes.lookup(memberNode);
      plan::AxisOp axis = planIndex.axes.lookup(memberNode);
      if (!member ||
          member->getName().getStringRef() != "intent.ragged_member" || !axis)
        return ragged.emitOpError(
            "references a non-canonical ragged member domain");
      runtime.members.push_back(member);
      if (axis.hasRole("parallel"))
        runtime.ownedMembers.push_back(member);
    }
    Operation *offsetsLoad = runtime.relation->getNumOperands() >= 3
                                 ? runtime.relation->getOperand(2).getDefiningOp()
                                 : nullptr;
    FailureOr<ABIView *> offsets =
        offsetsLoad &&
                offsetsLoad->getName().getStringRef() == "intent.view_load" &&
                offsetsLoad->getNumOperands() == 1
            ? lookupView(offsetsLoad->getOperand(0), *runtime.relation)
            : FailureOr<ABIView *>(failure());
    if (failed(offsets) || (*offsets)->tensor.getRank() != 1)
      return runtime.relation->emitOpError(
          "requires a canonical rank-one offsets view");
    runtime.offsets = *offsets;
    if (runtime.relation->getNumOperands() == 4) {
      Operation *indicesLoad = runtime.relation->getOperand(3).getDefiningOp();
      FailureOr<ABIView *> indices =
          indicesLoad &&
                  indicesLoad->getName().getStringRef() == "intent.view_load" &&
                  indicesLoad->getNumOperands() == 1
              ? lookupView(indicesLoad->getOperand(0), *runtime.relation)
              : FailureOr<ABIView *>(failure());
      if (failed(indices) || (*indices)->tensor.getRank() != 1)
        return runtime.relation->emitOpError(
            "requires a canonical rank-one member-index view");
      runtime.indices = *indices;
    }
    unsigned position = raggedRuntimes.size();
    raggedRuntimeByRelation[ragged.getNode()] = position;
    raggedRuntimesByAxis[ragged.getOuterNode()].push_back(position);
    for (int64_t member : ragged.getMemberNodes())
      raggedRuntimesByAxis[member].push_back(position);
    raggedRuntimes.push_back(std::move(runtime));
  }
  return success();
}

LogicalResult SourceEmitter::prepareRaggedStages() {
  if (failed(target::emission::indexStageOperations(
          kernel, planIndex, operationStages)))
    return failure();
  stageBodies.resize(planIndex.stages.size());
  for (auto [position, stage] : llvm::enumerate(planIndex.stages)) {
    Operation *contract = kernel.nodes.lookup(stage.getNode());
    if (!contract || contract->getName().getStringRef() != "intent.contract" ||
        contract->getNumResults() != 1)
      return stage.emitOpError("does not bind one canonical contraction");
    auto axes = planIndex.stageAxes.find(stage.getNode());
    plan::StageAxisOp feature =
        axes == planIndex.stageAxes.end() ? plan::StageAxisOp()
                                         : axes->second.lookup("feature");
    plan::StageAxisOp reduction =
        axes == planIndex.stageAxes.end() ? plan::StageAxisOp()
                                         : axes->second.lookup("reduction");
    plan::StageAxisOp member =
        axes == planIndex.stageAxes.end() ? plan::StageAxisOp()
                                         : axes->second.lookup("member");
    if (!feature || !reduction || !member || !feature.getWorkerAxisAttr() ||
        !member.getWorkerAxisAttr() || !member.getAxisNodeAttr())
      return stage.emitOpError("has incomplete physical stage-axis decisions");
    auto runtimes =
        raggedRuntimesByAxis.find(member.getAxisNodeAttr().getInt());
    if (runtimes == raggedRuntimesByAxis.end() ||
        runtimes->second.size() != 1)
      return stage.emitOpError(
          "does not resolve one ragged relation for its member axis");
    unsigned runtimeIndex = runtimes->second.front();
    RaggedRuntime &runtime = raggedRuntimes[runtimeIndex];
    bool owned = llvm::any_of(runtime.ownedMembers, [&](Operation *candidate) {
      auto node = candidate->getAttrOfType<IntegerAttr>("intent.node");
      return node && node.getInt() == member.getAxisNodeAttr().getInt();
    });
    if (!owned)
      return stage.emitOpError("uses a ragged member without program ownership");
    if (runtime.memberDimension.empty())
      runtime.memberDimension = member.getExtent().str();
    else if (runtime.memberDimension != member.getExtent())
      return stage.emitOpError("changes its staged ragged member extent");
    Operation *members = nullptr;
    for (int64_t operationNode : stage.getOperations()) {
      Operation *candidate = kernel.nodes.lookup(operationNode);
      if (!candidate ||
          candidate->getName().getStringRef() != "intent.members")
        continue;
      if (members)
        return stage.emitOpError("contains multiple member enumeration ops");
      members = candidate;
    }
    if (!members)
      return stage.emitOpError("has no member enumeration operation");
    stageRaggedRuntime[position] = runtimeIndex;
    stageFeatureDimensions[position] = feature.getExtent().str();
    stageMemberDimensions[position] = member.getExtent().str();
    stageReductionDimensions[position] = reduction.getExtent().str();
    stageFeatureWorkers[position] = feature.getWorkerAxisAttr().getInt();
    stageMemberWorkers[position] = member.getWorkerAxisAttr().getInt();
    for (int64_t valueID : stage.getInputs())
      if (!kernel.values.count(valueID))
        return stage.emitOpError("references an unknown stage input value");
    for (int64_t valueID : stage.getOutputs()) {
      auto found = kernel.values.find(valueID);
      if (found == kernel.values.end() ||
          !stageOutputOwners.try_emplace(found->second, position).second)
        return stage.emitOpError("has an invalid stage output value");
      workspaceNames[found->second] =
          "workspace_" + std::to_string(valueID);
    }
    if (!llvm::is_contained(operationStages.lookup(contract), position))
      return stage.emitOpError("stage roots do not depend on its contraction");
  }
  return success();
}

bool SourceEmitter::selectOperation(Operation &operation) {
  auto scanProducer = scanProducerOwners.find(&operation);
  if (scanProducer != scanProducerOwners.end())
    return activeScanReplay == scanProducer->second;
  if (planIndex.stages.empty())
    return true;
  activeStages = operationStages.lookup(&operation);
  return !activeStages.empty() &&
         !target::emission::isAbsorbedStagedAccessMetadata(planIndex,
                                                           operation);
}

void SourceEmitter::stageLine(unsigned stage, StringRef text, unsigned indent) {
  stageBodies[stage].append(indent * 4, ' ');
  stageBodies[stage] += text.str();
  stageBodies[stage] += "\n";
}

void SourceEmitter::emitImports() {
  output << "import torch\n";
  output << "import tilelang\n";
  output << "import tilelang.language as T\n";
  output << "from intent.runtime.tuning.tilelang import DEFAULT_NUM_STAGES, DEFAULT_THREADS, row_configuration\n";
  if (planIndex.program.getPersistent())
    output << "_NUM_SMS = torch.cuda.get_device_properties("
           << planIndex.target.getDevice() << ").multi_processor_count\n";
  if (searchSpace) {
    output << "from tilelang.autotuner import autotune\n";
    output << "from intent.runtime.tuning.tilelang import autotune_configurations\n";
    output << "\n_PARAMETER_MAP = {";
    for (auto [index, mapping] :
         llvm::enumerate(searchIndex.autotune.getParameterMap())) {
      if (index)
        output << ", ";
      output << "'" << mapping.getName().getValue() << "': '"
             << cast<StringAttr>(mapping.getValue()).getValue() << "'";
    }
    output << "}\n_CONFIGS = autotune_configurations(_PARAMETER_MAP)\n";
    output << "_AUTOTUNE_INPUTS = None\n\n";
    output << "def _fresh_autotune_inputs(_):\n";
    output << "    if _AUTOTUNE_INPUTS is None:\n";
    output << "        raise RuntimeError('TileLang autotuning has no captured launch inputs')\n";
    output << "    return [value.clone() if isinstance(value, torch.Tensor) else value for value in _AUTOTUNE_INPUTS]\n";
  }
  output << "\n\n";
}

LogicalResult SourceEmitter::emitKernelHeader() {
  kernelName = (kernel.entry.getName() + "_kernel").str();
  auto emitBuilderParameters = [&](raw_ostream &stream, int64_t stage) {
    bool first = true;
    auto parameter = [&](StringRef value) {
      if (!first)
        stream << ", ";
      stream << value;
      first = false;
    };
    for (const std::string &dimension : dimensionOrder)
      parameter(dimension);
    if (stage >= 0) {
      RaggedRuntime &ragged =
          raggedRuntimes[stageRaggedRuntime.lookup(stage)];
      parameter(ragged.indices ? "MAX_ROUTES" : "ROUTE_LENGTHS");
    }
    for (int64_t axis : planIndex.components.orderedRaggedProgramAxes)
      parameter("MAX_SEQUENCE_LENGTH_" + std::to_string(axis));
    if (!planIndex.components.reusedAxes.empty()) {
      parameter("TILE_SIZE");
      parameter("num_stages=DEFAULT_NUM_STAGES");
      parameter("threads=DEFAULT_THREADS");
    } else {
      if (searchIndex.autotune)
        for (NamedAttribute config : searchIndex.autotune.getParameterMap())
          parameter(config.getName().getValue().str() + "=1");
      parameter("num_stages=DEFAULT_NUM_STAGES");
      parameter("threads=DEFAULT_THREADS");
    }
  };
  auto emitMainParameters = [&](raw_ostream &stream) -> LogicalResult {
    bool first = true;
    auto parameter = [&](StringRef value) {
      if (!first)
        stream << ", ";
      stream << value;
      first = false;
    };
    for (ABIView &view : views) {
      std::string dtype =
          dtypeName(view.tensor.getElementType(), *kernel.entry.getOperation());
      if (dtype.empty())
        return failure();
      std::string shape = "(";
      for (auto [index, extent] : llvm::enumerate(view.shape)) {
        if (index)
          shape += ", ";
        shape += extent;
      }
      if (view.shape.size() == 1)
        shape += ",";
      shape += ")";
      parameter(view.argument->name + ": T.Tensor(" + shape + ", " + dtype +
                ")");
    }
    for (ABIScalar &scalar : scalars) {
      std::string dtype = dtypeName(scalar.type, *kernel.entry.getOperation());
      if (dtype.empty())
        return failure();
      parameter(scalar.name + ": " + dtype);
    }
    for (Operation *buffer : privateWorkspaceBuffers) {
      FailureOr<std::string> size =
          target::emission::privateWorkspaceElementCount(
              *buffer, planIndex, roleDimensions);
      FailureOr<target::LogicalBufferInfo> info =
          target::getLogicalBufferInfo(*buffer);
      std::string dtype = succeeded(info)
                              ? dtypeName(info->elementType, *buffer)
                              : std::string();
      if (failed(size) || failed(info) || dtype.empty())
        return buffer->emitOpError(
            "has no supported TileLang private-workspace parameter");
      parameter(workspaceNames.lookup(buffer->getResult(0)) +
                ": T.Tensor((" + *size + ",), " + dtype + ")");
    }
    for (const auto &entry : planIndex.scans) {
      if (entry.second.getResultSpace() != "global")
        continue;
      Operation *scan = kernel.nodes.lookup(entry.first);
      auto tensor = scan && scan->getNumResults() == 1
                        ? dyn_cast<RankedTensorType>(scan->getResult(0).getType())
                        : RankedTensorType();
      FailureOr<std::string> size = target::emission::scanWorkspaceElementCount(
          entry.second, scanExtents.lookup(entry.first), planIndex,
          roleDimensions);
      std::string dtype =
          tensor ? dtypeName(tensor.getElementType(), *scan) : std::string();
      if (!tensor || failed(size) || dtype.empty())
        return entry.second.emitOpError(
            "has no supported TileLang scan-workspace parameter");
      parameter(workspaceNames.lookup(scan->getResult(0)) +
                ": T.Tensor((" + *size + ",), " + dtype + ")");
      for (int64_t valueID : entry.second.getMaterializedValues()) {
        Value value = kernel.values.lookup(valueID);
        auto materialized = dyn_cast<RankedTensorType>(value.getType());
        std::string valueDtype = materialized
                                     ? dtypeName(materialized.getElementType(),
                                                 *scan)
                                     : std::string();
        if (!materialized || materialized.getRank() != 1 || valueDtype.empty())
          return entry.second.emitOpError(
              "requires rank-one materialized TileLang scan producer values");
        parameter(workspaceNames.lookup(value) + ": T.Tensor((" + *size +
                  ",), " + valueDtype + ")");
      }
    }
    for (plan::StageOp stage : planIndex.stages)
      for (int64_t valueID : stage.getOutputs()) {
        Value value = kernel.values.lookup(valueID);
        auto tensor = dyn_cast<RankedTensorType>(value.getType());
        std::string dtype = tensor
                                ? dtypeName(tensor.getElementType(),
                                            *kernel.entry.getOperation())
                                   : std::string();
        if (!tensor || tensor.getRank() != 2 || dtype.empty())
          return stage.emitOpError("has an unsupported TileLang workspace");
        parameter(workspaceNames.lookup(value) + ": T.Tensor((" +
                  stageMemberDimensions.lookup(stage.getOrdinal()) + ", " +
                  stageFeatureDimensions.lookup(stage.getOrdinal()) + "), " +
                  dtype + ")");
      }
    return success();
  };

  auto emitDecorator = [&](raw_ostream &stream) {
    if (searchSpace)
      stream << "@autotune(configs=_CONFIGS, warmup=3, rep=10, "
                "timeout=100, skip_check=True, "
                "supply_prog=_fresh_autotune_inputs)\n";
    stream << "@tilelang.jit\n";
  };
  auto emitBlockExtentConstants = [&](raw_ostream &stream) {
    for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
      stream << "    " << physicalExtent << " = 1 << (" << logicalExtent
             << " - 1).bit_length()\n";
  };

  if (!planIndex.stages.empty()) {
    for (unsigned stage = 0; stage < planIndex.stages.size(); ++stage) {
      RaggedRuntime &ragged =
          raggedRuntimes[stageRaggedRuntime.lookup(stage)];
      bool compact = !ragged.indices;
      llvm::raw_string_ostream source(stageBodies[stage]);
      emitDecorator(source);
      source << "def " << kernelName << "_stage_" << stage << "(";
      emitBuilderParameters(source, stage);
      source << "):\n";
      emitBlockExtentConstants(source);
      if (compact)
        source << "    total_route_tiles = sum((length + TILE_SIZE_M - 1) "
                  "// TILE_SIZE_M for length in ROUTE_LENGTHS)\n";
      source << "    @T.prim_func\n    def main(";
      if (failed(emitMainParameters(source)))
        return failure();
      source << "):\n";
      std::string feature = stageFeatureDimensions.lookup(stage);
      int64_t featureWorker = stageFeatureWorkers.lookup(stage);
      int64_t memberWorker = stageMemberWorkers.lookup(stage);
      if (featureWorker < 0 || featureWorker > 1 || memberWorker < 0 ||
          memberWorker > 1 || featureWorker == memberWorker)
        return planIndex.stages[stage].emitOpError(
            "has invalid TileLang stage worker-axis decisions");
      SmallVector<std::string> grids(2);
      SmallVector<std::string> names(2);
      grids[featureWorker] = "T.ceildiv(" + feature + ", TILE_SIZE_N)";
      plan::AxisOp outerAxis =
          planIndex.axes.lookup(ragged.binding.getOuterNode());
      std::string experts =
          outerAxis ? roleDimensions.lookup(
                          "program_" + std::to_string(outerAxis.getProgramOrder()))
                    : std::string();
      if (experts.empty())
        return ragged.binding.emitOpError(
            "has no program-owned outer-axis dimension");
      grids[memberWorker] = compact
                                ? "total_route_tiles"
                                : experts +
                                      " * T.ceildiv(MAX_ROUTES, TILE_SIZE_M)";
      names[featureWorker] = "bid_feature";
      names[memberWorker] = "bid_expert_route";
      source << "        with T.Kernel(" << grids[0] << ", " << grids[1]
             << ", threads=threads) as (" << names[0] << ", " << names[1]
             << "):\n";
      source.flush();
      if (compact) {
        stageLine(stage, "expert = T.alloc_var(dtype=T.int32)");
        stageLine(stage, "route_tile = T.alloc_var(dtype=T.int32)");
        stageLine(stage, "tile_cursor = T.alloc_var(dtype=T.int32)");
        stageLine(stage, "expert = 0");
        stageLine(stage, "route_tile = 0");
        stageLine(stage, "tile_cursor = 0");
        stageLine(stage,
                  "for candidate in range(" + experts + "):");
        stageLine(stage, "candidate_begin = " +
                             ragged.offsets->argument->name + "[" +
                             addressIndex("candidate") + "]",
                  4);
        stageLine(stage, "candidate_end = " +
                             ragged.offsets->argument->name +
                             "[" + addressIndex("candidate + 1") + "]",
                  4);
        stageLine(stage,
                  "candidate_tiles = T.ceildiv(candidate_end - "
                  "candidate_begin, TILE_SIZE_M)",
                  4);
        stageLine(stage,
                  "owns_tile = bid_expert_route >= tile_cursor and "
                  "bid_expert_route < tile_cursor + candidate_tiles",
                  4);
        stageLine(stage,
                  "expert = T.if_then_else(owns_tile, candidate, expert)", 4);
        stageLine(stage,
                  "route_tile = T.if_then_else(owns_tile, bid_expert_route - "
                  "tile_cursor, route_tile)",
                  4);
        stageLine(stage, "tile_cursor = tile_cursor + candidate_tiles", 4);
      } else {
        stageLine(stage,
                  "num_route_tiles = T.ceildiv(MAX_ROUTES, TILE_SIZE_M)");
        stageLine(stage, "expert = bid_expert_route // num_route_tiles");
        stageLine(stage, "route_tile = bid_expert_route % num_route_tiles");
      }
      stageLine(stage, "route_begin = " + ragged.offsets->argument->name +
                           "[" + addressIndex("expert") + "]");
      stageLine(stage, "route_end = " + ragged.offsets->argument->name +
                           "[" + addressIndex("expert + 1") + "]");
      stageLine(stage,
                "member_start = " + addressIndex("route_begin") + " + " +
                    addressIndex("route_tile") + " * TILE_SIZE_M");
    }
    return success();
  }

  emitDecorator(output);
  output << "def " << kernelName << "(";
  emitBuilderParameters(output, -1);
  output << "):\n";
  emitBlockExtentConstants(output);
  output << "    @T.prim_func\n    def main(";
  if (failed(emitMainParameters(output)))
    return failure();
  output << "):\n";
  if (!planIndex.components.reusedAxes.empty()) {
    output << "        with T.Kernel("
           << roleDimensions.lookup("program_0")
           << ", threads=threads) as program_index:\n";
    if (planIndex.components.reusedAxes.size() != 1)
      return realization.emitOpError(
          "TileLang worker reuse requires one explicit program axis");
    plan::AxisOp reused = planIndex.components.reusedAxes.front();
    programBlocks[reused.getNode()] = addressIndex("program_index");
    axisIndices[reused.getNode()] = addressIndex("program_index");
    plan::AxisOp lane = planIndex.axesByRole.lookup("lane_0");
    if (lane)
      axisIndices[lane.getNode()] = "0";
    return success();
  }

  SmallVector<plan::AxisOp> programAxes =
      target::emission::orderedProgramAxes(planIndex);
  auto axisExtent = [&](plan::AxisOp axis) {
    std::string role = "program_" + std::to_string(axis.getProgramOrder());
    std::string extent =
        planIndex.components.orderedRaggedProgramAxes.contains(axis.getNode())
            ? "MAX_SEQUENCE_LENGTH_" + std::to_string(axis.getNode())
            : roleDimensions.lookup(role);
    return axis.isScalar()
               ? extent
               : "T.ceildiv(" + extent + ", " + axis.getTile().str() + ")";
  };
  std::array<std::string, 3> grid =
      target::emission::projectProgramGrid(planIndex, axisExtent);
  bool persistent = planIndex.program.getPersistent();
  std::string linear = "persistent_program";
  unsigned workers = 0;
  for (plan::AxisOp axis : programAxes)
    workers = std::max(workers,
                       static_cast<unsigned>(axis.getWorkerAxis() + 1));
  if (persistent) {
    output << "        total_program_tiles = "
           << target::emission::projectProgramVolume(planIndex, axisExtent)
           << "\n";
    output << "        persistent_programs = T.min(_NUM_SMS, total_program_tiles)\n";
    workers = 1;
  }
  output << "        with T.Kernel(";
  if (persistent) {
    output << "persistent_programs";
  } else {
    for (unsigned worker = 0; worker < workers; ++worker) {
      if (worker)
        output << ", ";
      output << grid[worker];
    }
  }
  bool addressableLocalBuffer = llvm::any_of(
      planIndex.buffers, [](const auto &entry) {
        return entry.second.getSpace() == "private_vector" ||
               entry.second.getSpace() == "private_workspace";
      });
  output << ", threads=" << (addressableLocalBuffer ? "1" : "threads") << ") as ";
  if (workers == 1)
    output << "pid_worker_0:\n";
  else {
    output << "(";
    for (unsigned worker = 0; worker < workers; ++worker) {
      if (worker)
        output << ", ";
      output << "pid_worker_" << worker;
    }
    output << "):\n";
  }
  std::string programIndent = "            ";
  if (persistent) {
    output << "            for persistent_wave in T.serial(T.ceildiv(total_program_tiles, persistent_programs)):\n";
    output << "                " << linear
           << " = " << addressIndex("persistent_wave") << " * "
           << addressIndex("persistent_programs") << " + "
           << addressIndex("pid_worker_0") << "\n";
    output << "                if " << linear << " < total_program_tiles:\n";
    programIndent = "                    ";
    indentation = 5;
  }

  for (const auto &entry : planIndex.components.groups) {
    SmallVector<plan::AxisOp> axes(entry.getValue().begin(), entry.getValue().end());
    llvm::sort(axes, [](plan::AxisOp lhs, plan::AxisOp rhs) {
      return lhs.getProgramOrder() < rhs.getProgramOrder();
    });
    if (axes.size() != 2 ||
        axes.front().getWorkerAxis() != axes.back().getWorkerAxis())
      return axes.front().emitOpError(
          "TileLang supports two-axis program grouping on one worker axis");
    StringRef group = axes.front().getGroupSpelling();
    if (group.empty())
      return axes.front().emitOpError("has no TileLang group spelling");
    plan::AxisOp lhs = axes[0];
    plan::AxisOp rhs = axes[1];
    std::string lhsRole =
        "program_" + std::to_string(lhs.getProgramOrder());
    std::string rhsRole =
        "program_" + std::to_string(rhs.getProgramOrder());
    std::string pid = "group_bid_" + std::to_string(lhs.getNode());
    std::string lhsCount = "group_count_" + std::to_string(lhs.getNode());
    std::string rhsCount = "group_count_" + std::to_string(rhs.getNode());
    std::string span = "group_span_" + std::to_string(lhs.getNode());
    std::string id = "group_id_" + std::to_string(lhs.getNode());
    std::string first = "group_first_" + std::to_string(lhs.getNode());
    std::string size = "group_size_" + std::to_string(lhs.getNode());
    std::string lhsBlock = "block_axis_" + std::to_string(lhs.getNode());
    std::string rhsBlock = "block_axis_" + std::to_string(rhs.getNode());
    if (persistent) {
      FailureOr<std::string> groupIndex =
          target::emission::projectLinearGroupIndex(
              planIndex, axes, axisExtent, linear,
              *lhs.operation.getOperation());
      if (failed(groupIndex))
        return failure();
      output << programIndent << pid << " = " << *groupIndex << "\n";
    } else {
      output << programIndent << pid << " = "
             << addressIndex("pid_worker_" +
                             std::to_string(lhs.getWorkerAxis()))
             << "\n";
    }
    output << programIndent << lhsCount << " = "
           << addressIndex("T.ceildiv(" + roleDimensions.lookup(lhsRole) +
                           ", " + lhs.getTile().str() + ")")
           << "\n";
    output << programIndent << rhsCount << " = "
           << addressIndex("T.ceildiv(" + roleDimensions.lookup(rhsRole) +
                           ", " + rhs.getTile().str() + ")")
           << "\n";
    output << programIndent << span << " = " << group << " * " << rhsCount
           << "\n";
    output << programIndent << id << " = " << pid << " // " << span << "\n";
    output << programIndent << first << " = " << id << " * " << group << "\n";
    output << programIndent << size << " = T.min(" << lhsCount << " - "
           << first << ", " << group << ")\n";
    output << programIndent << lhsBlock << " = " << first << " + (" << pid
           << " % " << size << ")\n";
    output << programIndent << rhsBlock << " = (" << pid << " % " << span
           << ") // " << size << "\n";
    programBlocks[lhs.getNode()] = lhsBlock;
    programBlocks[rhs.getNode()] = rhsBlock;
    axisIndices[lhs.getNode()] = lhsBlock + " * " + lhs.getTile().str();
    axisIndices[rhs.getNode()] = rhsBlock + " * " + rhs.getTile().str();
  }
  SmallVector<target::emission::ProgramIndexProjection> projections =
      persistent
          ? target::emission::projectLinearProgramIndices(
                planIndex, axisExtent, linear)
          : target::emission::projectProgramIndices(
                planIndex, axisExtent, [&](unsigned worker) {
                  return addressIndex("pid_worker_" + std::to_string(worker));
                });
  for (const target::emission::ProgramIndexProjection &projection : projections) {
    plan::AxisOp axis = projection.axis;
    std::string block = "block_axis_" + std::to_string(axis.getNode());
    output << programIndent << block << " = "
           << addressIndex(projection.expression) << "\n";
    programBlocks[axis.getNode()] = block;
    axisIndices[axis.getNode()] =
        axis.isScalar() ? block : block + " * " + axis.getTile().str();
  }
  return success();
}

LogicalResult SourceEmitter::emitWrapper() {
  auto torchDtype = [&](Type type) -> StringRef {
    if (type.isInteger(1))
      return "torch.bool";
    if (type.isF16())
      return "torch.float16";
    if (type.isF32())
      return "torch.float32";
    if (type.isBF16())
      return "torch.bfloat16";
    if (auto integer = dyn_cast<IntegerType>(type);
        integer && integer.getWidth() == 8)
      return integer.isUnsigned() ? "torch.uint8" : "torch.int8";
    if (auto integer = dyn_cast<IntegerType>(type);
        integer && integer.getWidth() == 32)
      return "torch.int32";
    return {};
  };
  auto emitValidation = [&]() -> LogicalResult {
    for (ABIView &view : views) {
      StringRef dtype = torchDtype(view.tensor.getElementType());
      if (dtype.empty())
        return kernel.entry.emitOpError("has an unsupported TileLang ABI dtype");
      output << "    if " << view.argument->name << ".device != _DEVICE:\n";
      output << "        raise ValueError('" << view.argument->name
             << " must reside on the realized CUDA device')\n";
      output << "    if " << view.argument->name << ".ndim != "
             << view.tensor.getRank() << ":\n";
      output << "        raise ValueError('" << view.argument->name
             << " has the wrong rank')\n";
      output << "    if " << view.argument->name << ".dtype != " << dtype
             << ":\n";
      output << "        raise ValueError('" << view.argument->name
             << " has the wrong dtype')\n";
      output << "    if any(extent == 0 for extent in "
             << view.argument->name << ".shape):\n";
      output << "        raise NotImplementedError('TileLang does not support "
                "zero-extent external views in this compiler')\n";
      output << "    if sum(max(0, extent - 1) * abs(stride) for extent, stride "
                "in zip("
             << view.argument->name << ".shape, " << view.argument->name
             << ".stride())) > 2147483647:\n";
      output << "        raise NotImplementedError('TileLang cannot project this "
                "64-bit external-buffer address through its current bulk-copy "
                "lowering')\n";
    }
    for (const std::string &dimension : dimensionOrder)
      output << "    " << dimension << " = "
             << dimensionOwners.lookup(dimension) << "\n";
    llvm::SmallVector<StringRef> exactExtents;
    exactExtents.reserve(exactBulkExtents.size());
    for (const auto &extent : exactBulkExtents)
      exactExtents.push_back(extent.getKey());
    llvm::sort(exactExtents);
    for (StringRef extent : exactExtents) {
      std::string logical = dimensionSpelling(extent);
      output << "    if " << logical << " != 1 << (" << logical
             << " - 1).bit_length():\n";
      output << "        raise NotImplementedError('TileLang guarded float16 "
                "bulk transfer requires an exact power-of-two extent')\n";
    }
    for (ABIView &view : views) {
      output << "    if tuple(" << view.argument->name << ".shape) != (";
      for (auto [axis, extent] : llvm::enumerate(view.shape)) {
        if (axis)
          output << ", ";
        output << extent;
      }
      if (view.shape.size() == 1)
        output << ",";
      output << "):\n";
      output << "        raise ValueError('" << view.argument->name
             << " shape violates the kernel symbols')\n";
    }
    return success();
  };
  auto emitPrivateWorkspaceAllocations = [&]() -> LogicalResult {
    for (Operation *buffer : privateWorkspaceBuffers) {
      FailureOr<std::string> size =
          target::emission::privateWorkspaceElementCount(
              *buffer, planIndex, roleDimensions);
      FailureOr<std::string> initializer =
          target::emission::logicalBufferPythonInitializer(*buffer);
      FailureOr<target::LogicalBufferInfo> info =
          target::getLogicalBufferInfo(*buffer);
      StringRef dtype =
          succeeded(info) ? torchDtype(info->elementType) : StringRef();
      if (failed(size) || failed(initializer) || failed(info) || dtype.empty())
        return buffer->emitOpError(
            "has no supported TileLang private-workspace allocation");
      output << "    " << workspaceNames.lookup(buffer->getResult(0))
             << " = torch.full((" << *size << ",), " << *initializer
             << ", device=_DEVICE, dtype=" << dtype << ")\n";
    }
    for (const auto &entry : planIndex.scans) {
      if (entry.second.getResultSpace() != "global")
        continue;
      Operation *scan = kernel.nodes.lookup(entry.first);
      auto tensor = scan && scan->getNumResults() == 1
                        ? dyn_cast<RankedTensorType>(scan->getResult(0).getType())
                        : RankedTensorType();
      FailureOr<std::string> size = target::emission::scanWorkspaceElementCount(
          entry.second, scanExtents.lookup(entry.first), planIndex,
          roleDimensions);
      StringRef dtype = tensor ? torchDtype(tensor.getElementType()) : StringRef();
      if (!tensor || failed(size) || dtype.empty())
        return entry.second.emitOpError(
            "has no supported TileLang scan-workspace allocation");
      output << "    " << workspaceNames.lookup(scan->getResult(0))
             << " = torch.empty((" << *size
             << ",), device=_DEVICE, dtype=" << dtype << ")\n";
      for (int64_t valueID : entry.second.getMaterializedValues()) {
        Value value = kernel.values.lookup(valueID);
        auto materialized = dyn_cast<RankedTensorType>(value.getType());
        StringRef valueDtype =
            materialized ? torchDtype(materialized.getElementType()) : StringRef();
        if (!materialized || materialized.getRank() != 1 || valueDtype.empty())
          return entry.second.emitOpError(
              "requires rank-one materialized TileLang scan producer values");
        output << "    " << workspaceNames.lookup(value) << " = torch.empty(("
               << *size << ",), device=_DEVICE, dtype=" << valueDtype << ")\n";
      }
    }
    return success();
  };
  auto emitLaunchSignature = [&]() {
    bool first = true;
    for (ABIView &view : views) {
      if (!first)
        output << ", ";
      output << view.argument->name;
      first = false;
    }
    for (ABIScalar &scalar : scalars) {
      if (!first)
        output << ", ";
      output << scalar.name;
      first = false;
    }
  };
  auto emitKernelArguments = [&](ABIView *substitute = nullptr) {
    bool first = true;
    for (ABIView &view : views) {
      if (!first)
        output << ", ";
      if (&view == substitute)
        output << "torch.zeros_like(" << view.argument->name << ")";
      else
        output << view.argument->name;
      first = false;
    }
    for (ABIScalar &scalar : scalars) {
      if (!first)
        output << ", ";
      output << scalar.name;
      first = false;
    }
    for (Operation *buffer : privateWorkspaceBuffers) {
      if (!first)
        output << ", ";
      output << workspaceNames.lookup(buffer->getResult(0));
      first = false;
    }
    for (const auto &entry : planIndex.scans) {
      if (entry.second.getResultSpace() != "global")
        continue;
      if (!first)
        output << ", ";
      Operation *scan = kernel.nodes.lookup(entry.first);
      output << workspaceNames.lookup(scan->getResult(0));
      first = false;
      for (int64_t valueID : entry.second.getMaterializedValues()) {
        output << ", " << workspaceNames.lookup(kernel.values.lookup(valueID));
      }
    }
    for (plan::StageOp stage : planIndex.stages)
      for (int64_t valueID : stage.getOutputs()) {
        if (!first)
          output << ", ";
        output << workspaceNames.lookup(kernel.values.lookup(valueID));
        first = false;
      }
  };
  auto emitBuilderArguments = [&](int64_t stage) {
    bool first = true;
    for (const std::string &dimension : dimensionOrder) {
      if (!first)
        output << ", ";
      output << dimension;
      first = false;
    }
    if (stage >= 0) {
      RaggedRuntime &ragged =
          raggedRuntimes[stageRaggedRuntime.lookup(stage)];
      if (!first)
        output << ", ";
      std::string suffix = std::to_string(ragged.binding.getNode());
      output << (ragged.indices ? "max_routes_" : "route_lengths_")
             << suffix;
    }
    for (int64_t axis : planIndex.components.orderedRaggedProgramAxes) {
      if (!first)
        output << ", ";
      output << "max_sequence_length_" << axis;
      first = false;
    }
  };

  if (!planIndex.stages.empty()) {
    for (const std::string &body : stageBodies)
      output << body << "    return main\n\n\n";
  } else
    output << "    return main\n\n\n";
  output << "_DEVICE = torch.device('cuda', " << planIndex.target.getDevice()
         << ")\n_KERNEL_CACHE = {}\n\n\n";
  output << "def launch(";
  emitLaunchSignature();
  output << "):\n";
  if (searchSpace)
    output << "    global _AUTOTUNE_INPUTS\n";
  if (failed(emitValidation()))
    return failure();
  if (failed(emitPrivateWorkspaceAllocations()))
    return failure();

  if (!planIndex.stages.empty()) {
    SmallVector<ABIView *> inputs;
    ABIView *merge = nullptr;
    for (ABIView &view : views) {
      if (view.view.getAccess() == "in")
        inputs.push_back(&view);
      else if ((view.view.getAccess() == "out" ||
                view.view.getAccess() == "inout") &&
               !merge)
        merge = &view;
      else
        return kernel.entry.emitOpError(
            "ragged TileLang stages require inputs and one inout merge view");
    }
    if (!merge)
      return kernel.entry.emitOpError("ragged TileLang stages have no merge view");
    llvm::DenseSet<unsigned> preparedRagged;
    for (const auto &entry : stageRaggedRuntime) {
      if (!preparedRagged.insert(entry.second).second)
        continue;
      RaggedRuntime &ragged = raggedRuntimes[entry.second];
      std::string suffix = std::to_string(ragged.binding.getNode());
      if (!ragged.indices)
        output << "    route_lengths_" << suffix
               << " = tuple(int(length) for length in ("
               << ragged.offsets->argument->name << "[1:] - "
               << ragged.offsets->argument->name << "[:-1]).tolist())\n";
      else
        output << "    max_routes_" << suffix << " = int(("
               << ragged.offsets->argument->name << "[1:] - "
               << ragged.offsets->argument->name
               << "[:-1]).max().item())\n";
    }
    for (plan::StageOp stage : planIndex.stages)
      for (int64_t valueID : stage.getOutputs()) {
        Value value = kernel.values.lookup(valueID);
        auto tensor = dyn_cast<RankedTensorType>(value.getType());
        StringRef dtype =
            tensor ? torchDtype(tensor.getElementType()) : StringRef();
        if (!tensor || tensor.getRank() != 2 || dtype.empty())
          return stage.emitOpError("has an unsupported TileLang workspace");
        output << "    " << workspaceNames.lookup(value) << " = torch.empty(("
               << stageMemberDimensions.lookup(stage.getOrdinal()) << ", "
               << stageFeatureDimensions.lookup(stage.getOrdinal())
               << "), device=_DEVICE, dtype=" << dtype << ")\n";
      }
    for (unsigned stage = 0; stage < planIndex.stages.size(); ++stage) {
      RaggedRuntime &ragged =
          raggedRuntimes[stageRaggedRuntime.lookup(stage)];
      bool compact = !ragged.indices;
      std::string suffix = std::to_string(ragged.binding.getNode());
      output << "    cache_key = (" << stage;
      for (const std::string &dimension : dimensionOrder)
        output << ", " << dimension;
      output << ", " << (compact ? "route_lengths_" : "max_routes_")
             << suffix;
      for (ABIView *input : inputs)
        output << ", " << input->argument->name << ".dtype";
      output << ", str(_DEVICE))\n";
      output << "    if cache_key not in _KERNEL_CACHE:\n";
      output << "        _AUTOTUNE_INPUTS = [";
      bool first = true;
      for (ABIView &view : views) {
        if (!first)
          output << ", ";
        if (planIndex.stages[stage].getOutputs().empty() && &view == merge)
          output << (target::emission::stageUsesScatterReduction(
                         planIndex.stages[stage], kernel)
                         ? "torch.zeros_like("
                         : "torch.empty_like(")
                 << view.argument->name << ")";
        else
          output << view.argument->name;
        first = false;
      }
      for (ABIScalar &scalar : scalars) {
        if (!first)
          output << ", ";
        output << scalar.name;
        first = false;
      }
      for (plan::StageOp binding : planIndex.stages)
        for (int64_t valueID : binding.getOutputs()) {
          if (!first)
            output << ", ";
          output << workspaceNames.lookup(kernel.values.lookup(valueID));
          first = false;
        }
      output << "]\n        try:\n            compiled = " << kernelName << "_stage_" << stage
             << "(";
      emitBuilderArguments(stage);
      output << ")\n        finally:\n            _AUTOTUNE_INPUTS = None\n";
      output << "        _KERNEL_CACHE[cache_key] = compiled\n";
      output << "    compiled = _KERNEL_CACHE[cache_key]\n    compiled(";
      emitKernelArguments();
      output << ")\n";
    }
    output << "    return " << merge->argument->name << "\n\n\n";
    output << "def run(";
    bool first = true;
    for (ABIView *view : inputs) {
      if (!first)
        output << ", ";
      output << view->argument->name;
      first = false;
    }
    for (ABIScalar &scalar : scalars) {
      if (!first)
        output << ", ";
      output << scalar.name;
      first = false;
    }
    output << "):\n";
    for (const std::string &dimension : dimensionOrder)
      output << "    " << dimension << " = "
             << dimensionOwners.lookup(dimension) << "\n";
    output << "    " << merge->argument->name << " = torch."
           << (target::emission::planUsesScatterReduction(planIndex, kernel)
                   ? "zeros"
                   : "empty")
           << "((";
    for (auto [axis, extent] : llvm::enumerate(merge->shape)) {
      if (axis)
        output << ", ";
      output << extent;
    }
    output << "), device=_DEVICE, dtype="
           << torchDtype(merge->tensor.getElementType()) << ")\n    launch(";
    first = true;
    for (ABIView &view : views) {
      if (!first)
        output << ", ";
      output << view.argument->name;
      first = false;
    }
    for (ABIScalar &scalar : scalars)
      output << ", " << scalar.name;
    output << ")\n    return " << merge->argument->name << "\n";
    return success();
  }

  SmallVector<ABIView *> inputs;
  SmallVector<ABIView *> outputs;
  for (ABIView &view : views) {
    if (view.view.getAccess() == "in" || view.view.getAccess() == "inout")
      inputs.push_back(&view);
    else if (view.view.getAccess() == "out")
      outputs.push_back(&view);
    else
      return kernel.entry.emitOpError(
          "TileLang wrapper supports input and output views");
  }
  bool hasInOut = llvm::any_of(views, [](const ABIView &view) {
    return view.view.getAccess() == "inout";
  });
  if (outputs.empty() && !hasInOut)
    return kernel.entry.emitOpError("TileLang wrapper has no writable views");
  for (int64_t axis : planIndex.components.orderedRaggedProgramAxes) {
    FailureOr<plan::RaggedOp> relation =
        target::emission::uniqueRaggedRelation(
            planIndex, axis, *kernel.entry.getOperation());
    auto runtime = succeeded(relation)
                       ? raggedRuntimeByRelation.find(relation->getNode())
                       : raggedRuntimeByRelation.end();
    if (failed(relation) || runtime == raggedRuntimeByRelation.end())
      return kernel.entry.emitOpError(
          "has no ordered ragged runtime metadata");
    ABIView *offsets = raggedRuntimes[runtime->second].offsets;
    output << "    max_sequence_length_" << axis << " = int(("
           << offsets->argument->name << "[1:] - "
           << offsets->argument->name << "[:-1]).max().item())\n";
  }
  output << "    cache_key = (";
  bool firstCacheKey = true;
  auto emitCacheKey = [&](const std::string &value) {
    if (!firstCacheKey)
      output << ", ";
    output << value;
    firstCacheKey = false;
  };
  for (const std::string &dimension : dimensionOrder)
    emitCacheKey(dimension);
  for (int64_t axis : planIndex.components.orderedRaggedProgramAxes)
    emitCacheKey("max_sequence_length_" + std::to_string(axis));
  for (ABIView *input : inputs)
    emitCacheKey(input->argument->name + ".dtype");
  emitCacheKey("str(_DEVICE)");
  output << ")\n";
  output << "    if cache_key not in _KERNEL_CACHE:\n";
  if (searchSpace) {
    output << "        _AUTOTUNE_INPUTS = [";
    emitKernelArguments();
    output << "]\n        try:\n            compiled = " << kernelName << "(";
    emitBuilderArguments(-1);
    output << ")\n        finally:\n            _AUTOTUNE_INPUTS = None\n";
  } else if (!planIndex.components.reusedAxes.empty()) {
    std::string rowDimension = roleDimensions.lookup("lane_0");
    if (rowDimension.empty())
      return kernel.entry.emitOpError(
          "TileLang row configuration requires a lane axis");
    output << "        compiled = " << kernelName << "(";
    emitBuilderArguments(-1);
    if (!dimensionOrder.empty())
      output << ", ";
    output << "row_configuration(" << rowDimension << ").tile_size, "
           << "num_stages=row_configuration(" << rowDimension
           << ").num_stages, threads=row_configuration(" << rowDimension
           << ").threads)\n";
  } else {
    output << "        compiled = " << kernelName << "(";
    emitBuilderArguments(-1);
    output << ")\n";
  }
  output << "        _KERNEL_CACHE[cache_key] = compiled\n";
  output << "    compiled = _KERNEL_CACHE[cache_key]\n    compiled(";
  emitKernelArguments();
  output << ")\n";
  if (privateWorkspaceBuffers.empty() && scanResults.empty()) {
    output << "    return compiled\n\n\n";
  } else {
    output << "    return lambda: compiled(";
    emitKernelArguments();
    output << ")\n\n\n";
  }
  output << "def run(";
  bool first = true;
  for (ABIView *view : inputs) {
    if (!first)
      output << ", ";
    output << view->argument->name;
    first = false;
  }
  for (ABIScalar &scalar : scalars) {
    if (!first)
      output << ", ";
    output << scalar.name;
    first = false;
  }
  output << "):\n";
  for (const std::string &dimension : dimensionOrder)
    output << "    " << dimension << " = "
           << dimensionOwners.lookup(dimension) << "\n";
  for (ABIView *result : outputs) {
    StringRef outputDtype = torchDtype(result->tensor.getElementType());
    if (outputDtype.empty())
      return kernel.entry.emitOpError("has an unsupported TileLang output dtype");
    output << "    " << result->argument->name << " = torch.empty((";
    for (auto [axis, extent] : llvm::enumerate(result->shape)) {
      if (axis)
        output << ", ";
      output << extent;
    }
    if (result->shape.size() == 1)
      output << ",";
    output << "), device=_DEVICE, dtype=" << outputDtype << ")\n";
  }
  output << "    launch(";
  first = true;
  for (ABIView &view : views) {
    if (!first)
      output << ", ";
    output << view.argument->name;
    first = false;
  }
  for (ABIScalar &scalar : scalars)
    output << ", " << scalar.name;
  output << ")\n    return ";
  if (outputs.size() == 1) {
    output << outputs.front()->argument->name;
  } else {
    output << "(";
    for (auto [index, result] : llvm::enumerate(outputs)) {
      if (index)
        output << ", ";
      output << result->argument->name;
    }
    output << ")";
  }
  output << "\n";
  return success();
}

FailureOr<StringRef> SourceEmitter::lookupValue(Operation &consumer,
                                                 unsigned operandIndex) {
  if (operandIndex >= consumer.getNumOperands()) {
    consumer.emitOpError("references a missing operand during TileLang emission");
    return failure();
  }
  auto found = valueNames.find(consumer.getOperand(operandIndex));
  if (found == valueNames.end()) {
    consumer.emitOpError() << "operand " << operandIndex
                           << " has no emitted TileLang value";
    return failure();
  }
  return StringRef(found->second);
}

FailureOr<ABIView *> SourceEmitter::lookupView(Value value,
                                               Operation &consumer) {
  auto found = viewPositions.find(value);
  if (found == viewPositions.end()) {
    consumer.emitOpError("references a non-ABI external view");
    return failure();
  }
  return &views[found->second];
}

FailureOr<Operation *> SourceEmitter::resolveDomain(Value indexedValue,
                                                    Operation &consumer) {
  FailureOr<target::ScalarIndexSource> scalarSource =
      target::traceScalarIndexSource(indexedValue, consumer);
  if (failed(scalarSource))
    return failure();
  if (scalarSource->domain)
    return scalarSource->domain;
  if (Operation *definition = indexedValue.getDefiningOp())
    if (definition->getName().getStringRef() == "intent.domain" ||
        definition->getName().getStringRef() == "intent.ragged_outer" ||
        definition->getName().getStringRef() == "intent.ragged_member")
      return definition;
  auto argument = dyn_cast<BlockArgument>(indexedValue);
  Operation *owner = argument ? argument.getOwner()->getParentOp() : nullptr;
  if (owner && owner->getName().getStringRef() == "intent.state_stream" &&
      argument.getArgNumber() == 0 && owner->getNumOperands() > 0) {
    Operation *domain = owner->getOperand(0).getDefiningOp();
    if (domain &&
        (domain->getName().getStringRef() == "intent.domain" ||
         domain->getName().getStringRef() == "intent.ragged_member"))
      return domain;
    consumer.emitOpError("cannot resolve a TileLang stream source domain");
    return failure();
  }
  if (!owner || owner->getName().getStringRef() != "intent.parallel" ||
      owner->getNumOperands() != 1) {
    consumer.emitOpError("cannot resolve index ownership during TileLang emission");
    return failure();
  }
  Operation *source = owner->getOperand(0).getDefiningOp();
  if (source &&
      (source->getName().getStringRef() == "intent.domain" ||
       source->getName().getStringRef() == "intent.ragged_outer" ||
       source->getName().getStringRef() == "intent.ragged_member"))
    return source;
  if (source && source->getName().getStringRef() == "intent.partition" &&
      source->getNumOperands() == 1) {
    Operation *domain = source->getOperand(0).getDefiningOp();
    if (domain &&
        (domain->getName().getStringRef() == "intent.domain" ||
         domain->getName().getStringRef() == "intent.ragged_member"))
      return domain;
  }
  consumer.emitOpError("cannot resolve a partition to its source domain");
  return failure();
}

FailureOr<plan::AxisOp> SourceEmitter::resolveAxis(Value indexedValue,
                                                  Operation &consumer) {
  FailureOr<Operation *> domain = resolveDomain(indexedValue, consumer);
  if (failed(domain))
    return failure();
  FailureOr<int64_t> node = target::getNodeID(**domain, "axis lookup");
  plan::AxisOp axis =
      succeeded(node) ? planIndex.axes.lookup(*node) : plan::AxisOp();
  if (failed(node) || !axis) {
    consumer.emitOpError("indexes a domain without a TileLang axis binding");
    return failure();
  }
  return axis;
}

FailureOr<std::string> SourceEmitter::dimensionName(Operation &domain) {
  StringRef name = domain.getName().getStringRef();
  if (name == "intent.ragged_outer") {
    Operation *relation = domain.getOperand(0).getDefiningOp();
    Operation *source = relation &&
                                (relation->getNumOperands() == 3 ||
                                 relation->getNumOperands() == 4)
                            ? relation->getOperand(0).getDefiningOp()
                            : nullptr;
    if (!source)
      return failure();
    return dimensionName(*source);
  }
  if (name == "intent.ragged_member") {
    Operation *relation = domain.getOperand(0).getDefiningOp();
    Operation *source = relation &&
                                (relation->getNumOperands() == 3 ||
                                 relation->getNumOperands() == 4)
                            ? relation->getOperand(1).getDefiningOp()
                            : nullptr;
    if (!source)
      return failure();
    return dimensionName(*source);
  }
  if (domain.getNumOperands() < 2)
    return failure();
  Operation *dim = domain.getOperand(1).getDefiningOp();
  auto constant = dim ? dim->getAttrOfType<IntegerAttr>("intent.value")
                      : IntegerAttr();
  if (dim && dim->getName().getStringRef() == "intent.constant" && constant &&
      constant.getInt() > 0)
    return std::to_string(constant.getInt());
  auto axis = dim ? dim->getAttrOfType<IntegerAttr>("intent.axis")
                  : IntegerAttr();
  if (!dim || dim->getName().getStringRef() != "intent.dim" || !axis ||
      dim->getNumOperands() != 1)
    return failure();
  FailureOr<ABIView *> view = lookupView(dim->getOperand(0), domain);
  if (failed(view) || axis.getInt() < 0 ||
      static_cast<size_t>(axis.getInt()) >= (*view)->shape.size())
    return failure();
  return (*view)->shape[axis.getInt()];
}

std::string SourceEmitter::addressIndex(StringRef expression) const {
  return "(" + expression.str() + ")";
}

std::string SourceEmitter::physicalExtent(StringRef logicalExtent) const {
  auto extent = planIndex.blockExtents.find(logicalExtent);
  if (extent == planIndex.blockExtents.end())
    return dimensionSpelling(logicalExtent);
  return "PHYSICAL_" + dimensionSpelling(logicalExtent);
}

FailureOr<std::string>
SourceEmitter::transferPhysicalExtentFill(Operation &operation) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  if (failed(relation) || failed(view))
    return failure();
  return target::emission::transferPhysicalExtentFill(
      planIndex, *relation, (*view)->shape, operation);
}

FailureOr<std::string> SourceEmitter::accessIndices(Operation &operation) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  SmallVector<std::string> indices;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis")
      continue;
    if (term.kind == "full_slice") {
      indices.push_back(":");
      continue;
    }
    if (term.kind == "static_index") {
      if (term.staticValues.size() != 1 || !term.staticValues.front())
        return operation.emitOpError(
            "static TileLang index has no canonical value");
      indices.push_back(std::to_string(*term.staticValues.front()));
      continue;
    }
    if ((term.kind != "region_index" && term.kind != "value_index") ||
        term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "TileLang access has no mechanical index relation");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index" &&
        isa<RankedTensorType>(indexed.getType())) {
      auto tensor = cast<RankedTensorType>(indexed.getType());
      auto result = dyn_cast<OpResult>(indexed);
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      FailureOr<SmallVector<std::string>> extents =
          result ? tensorExtents(*result.getOwner(), result.getResultNumber())
                 : FailureOr<SmallVector<std::string>>(failure());
      if (tensor.getRank() != 1 || failed(exact) || failed(extents) ||
          extents->size() != 1 || extents->front() != "1")
        return operation.emitOpError(
            "TileLang bulk indirect access requires one singleton index tile");
      auto assumed = assumedIndexNames.find(indexed);
      indices.push_back(assumed == assumedIndexNames.end()
                            ? addressIndex(exact->str() + "[0]")
                            : addressIndex(assumed->second));
    } else if (term.kind == "value_index" ||
               target::emission::isSequentialIterator(indexed)) {
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (failed(exact))
        return failure();
      indices.push_back(addressIndex(*exact));
    } else {
      FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
      if (failed(axis))
        return failure();
      std::string base = axisIndices.lookup(axis->getNode());
      if (base.empty()) {
        operation.emitOpError()
            << "has no active TileLang index for logical axis "
            << axis->getNode();
        return failure();
      }
      std::string wideBase = addressIndex(base);
      if (target::emission::isPackedScalarAxis(*axis)) {
        indices.push_back(wideBase);
        continue;
      }
      std::string extent = axis->getTile().str();
      if (!axis->getReuseWorker() &&
          axis->getTileRole().starts_with("row_vector")) {
        FailureOr<Operation *> domain = resolveDomain(indexed, operation);
        FailureOr<std::string> logical =
            succeeded(domain) ? dimensionName(**domain)
                              : FailureOr<std::string>(failure());
        if (failed(domain) || failed(logical))
          return failure();
        extent = physicalExtent(*logical);
      }
      indices.push_back(axis->isScalar()
                            ? wideBase
                            : wideBase + " : " + wideBase + " + " +
                                  extent);
    }
  }
  std::string result;
  for (auto [index, value] : llvm::enumerate(indices)) {
    if (index)
      result += ", ";
    result += value;
  }
  return result;
}

FailureOr<std::string>
SourceEmitter::elementAccessIndices(Operation &operation,
                                    ArrayRef<std::string> tileIndices) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  unsigned tensorIndexCount = llvm::count_if(
      *relation, [&](const target::IndexTerm &term) {
        return term.kind == "value_index" && term.operands.size() == 1 &&
               term.operands.front() &&
               isa<RankedTensorType>(
                   operation.getOperand(*term.operands.front()).getType());
      });
  SmallVector<std::string> indices;
  unsigned tileAxis = 0;
  bool advancedTensorAxesCovered = false;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis") {
      if (tileAxis >= tileIndices.size())
        return operation.emitOpError(
            "parallel TileLang transfer has too few new-axis indices");
      ++tileAxis;
      continue;
    }
    if (term.kind == "full_slice") {
      if (tileAxis >= tileIndices.size())
        return operation.emitOpError(
            "parallel TileLang transfer has too few tile indices");
      indices.push_back(tileIndices[tileAxis++]);
      continue;
    }
    if (term.kind == "static_index") {
      if (term.staticValues.size() != 1 || !term.staticValues.front())
        return operation.emitOpError(
            "parallel TileLang transfer has an invalid static index");
      indices.push_back(std::to_string(*term.staticValues.front()));
      continue;
    }
    if ((term.kind != "region_index" && term.kind != "value_index") ||
        term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "parallel TileLang transfer has no mechanical index relation");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index" &&
        isa<RankedTensorType>(indexed.getType())) {
      auto tensor = cast<RankedTensorType>(indexed.getType());
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (failed(exact))
        return failure();
      if (tensorIndexCount > 1 || tensor.getRank() > 1) {
        if (tensor.getRank() != static_cast<int64_t>(tileIndices.size()) ||
            (tileAxis != 0 && !advancedTensorAxesCovered))
          return operation.emitOpError(
              "TileLang broadcasted tensor indices must jointly cover the transfer rank");
        std::string element = exact->str() + "[";
        for (auto [axis, index] : llvm::enumerate(tileIndices)) {
          if (axis)
            element += ", ";
          element += index;
        }
        element += "]";
        indices.push_back(addressIndex(element));
        if (!advancedTensorAxesCovered) {
          tileAxis = tileIndices.size();
          advancedTensorAxesCovered = true;
        }
      } else {
        if (tensor.getRank() != 1 || tileAxis >= tileIndices.size())
          return operation.emitOpError(
              "TileLang indirect element access requires one index-tile axis");
        auto assumed = assumedIndexNames.find(indexed);
        indices.push_back(assumed == assumedIndexNames.end()
                              ? addressIndex(exact->str() + "[" +
                                             tileIndices[tileAxis] + "]")
                              : addressIndex(assumed->second));
        ++tileAxis;
      }
    } else if (term.kind == "value_index" ||
               target::emission::isSequentialIterator(indexed)) {
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (failed(exact))
        return failure();
      indices.push_back(addressIndex(*exact));
    } else {
      FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
      if (failed(axis))
        return failure();
      std::string base = axisIndices.lookup(axis->getNode());
      if (base.empty())
        return operation.emitOpError(
            "has no active TileLang element index for its logical axis");
      if (target::emission::isPackedScalarAxis(*axis)) {
        indices.push_back(addressIndex(base));
        continue;
      }
      if (!axis->isScalar()) {
        if (tileAxis >= tileIndices.size())
          return operation.emitOpError(
              "parallel TileLang transfer has too few tile indices");
        indices.push_back(addressIndex(base) + " + " +
                          addressIndex(tileIndices[tileAxis++]));
      } else {
        indices.push_back(addressIndex(base));
      }
    }
  }
  if (tileAxis != tileIndices.size())
    return operation.emitOpError(
        "parallel TileLang transfer has unused tile indices");
  std::string result;
  for (auto [index, value] : llvm::enumerate(indices)) {
    if (index)
      result += ", ";
    result += value;
  }
  return result;
}

FailureOr<std::string>
SourceEmitter::elementBoundsPredicate(Operation &operation,
                                      ArrayRef<std::string> tileIndices,
                                      bool physicalOnly,
                                      bool includePhysical) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  if (failed(view) ||
      static_cast<size_t>(llvm::count_if(
          *relation, [](const target::IndexTerm &term) {
            return term.kind != "new_axis";
          })) != (*view)->shape.size())
    return failure();
  unsigned tensorIndexCount = llvm::count_if(
      *relation, [&](const target::IndexTerm &term) {
        return term.kind == "value_index" && term.operands.size() == 1 &&
               term.operands.front() &&
               isa<RankedTensorType>(
                   operation.getOperand(*term.operands.front()).getType());
      });
  SmallVector<std::string> predicates;
  unsigned tileAxis = 0;
  unsigned sourceAxis = 0;
  bool advancedTensorAxesCovered = false;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis") {
      if (tileAxis >= tileIndices.size())
        return operation.emitOpError(
            "bounded TileLang transfer has too few new-axis indices");
      ++tileAxis;
      continue;
    }
    unsigned axisNumber = sourceAxis++;
    if (term.kind == "full_slice") {
      if (tileAxis >= tileIndices.size())
        return operation.emitOpError(
            "bounded TileLang transfer has too few tile indices");
      if (includePhysical &&
          planIndex.blockExtents.count((*view)->shape[axisNumber]))
        predicates.push_back(tileIndices[tileAxis] + " < " +
                             (*view)->shape[axisNumber]);
      ++tileAxis;
      continue;
    }
    if (term.kind == "static_index")
      continue;
    if ((term.kind != "region_index" && term.kind != "value_index") ||
        term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "bounded TileLang transfer has no mechanical index relation");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index" &&
        isa<RankedTensorType>(indexed.getType())) {
      auto tensor = cast<RankedTensorType>(indexed.getType());
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (failed(exact))
        return failure();
      std::string index;
      if (tensorIndexCount > 1 || tensor.getRank() > 1) {
        if (tensor.getRank() != static_cast<int64_t>(tileIndices.size()) ||
            (tileAxis != 0 && !advancedTensorAxesCovered))
          return operation.emitOpError(
              "TileLang broadcasted tensor bounds must jointly cover the transfer rank");
        index = exact->str() + "[";
        for (auto [axis, tileIndex] : llvm::enumerate(tileIndices)) {
          if (axis)
            index += ", ";
          index += tileIndex;
        }
        index += "]";
        if (!advancedTensorAxesCovered) {
          tileAxis = tileIndices.size();
          advancedTensorAxesCovered = true;
        }
      } else {
        if (tensor.getRank() != 1 || tileAxis >= tileIndices.size())
          return operation.emitOpError(
              "TileLang indirect bounds require one index-tile axis");
        index = exact->str() + "[" + tileIndices[tileAxis++] + "]";
      }
      if (!physicalOnly) {
        predicates.push_back("0 <= " + index);
        predicates.push_back(index + " < " + (*view)->shape[axisNumber]);
      }
      continue;
    }
    if (term.kind == "value_index" &&
        !isa<RankedTensorType>(indexed.getType())) {
      FailureOr<target::ScalarIndexSource> source =
          target::traceScalarIndexSource(indexed, operation);
      if (failed(source))
        return failure();
      bool packedScalar = false;
      if (source->domain) {
        FailureOr<int64_t> node =
            target::getNodeID(*source->domain, "packed scalar bounds");
        auto axis = succeeded(node) ? planIndex.axes.find(*node)
                                    : planIndex.axes.end();
        if (failed(node))
          return failure();
        packedScalar =
            axis != planIndex.axes.end() &&
            target::emission::isPackedScalarAxis(axis->second);
      }
      if (!physicalOnly && source->domain &&
          (source->transformed || packedScalar)) {
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (failed(exact))
          return failure();
        std::string extent = (*view)->shape[axisNumber];
        predicates.push_back("0 <= (" + exact->str() + ")");
        predicates.push_back("(" + exact->str() + ") < " + extent);
      }
      continue;
    }
    FailureOr<plan::AxisOp> axis =
        resolveAxis(indexed, operation);
    if (failed(axis))
      return failure();
    if (target::emission::isPackedScalarAxis(*axis)) {
      if (!physicalOnly) {
        std::string exact = axisIndices.lookup(axis->getNode());
        Operation *domain = kernel.nodes.lookup(axis->getNode());
        FailureOr<std::string> extent =
            domain ? dimensionName(*domain)
                   : FailureOr<std::string>(failure());
        if (exact.empty() || failed(extent))
          return operation.emitOpError(
              "bounded TileLang packed scalar has no exact interval");
        predicates.push_back("0 <= " + exact);
        predicates.push_back(exact + " < " + *extent);
      }
      continue;
    }
    if (axis->isScalar())
      continue;
    if (tileAxis >= tileIndices.size())
      return operation.emitOpError(
          "bounded TileLang transfer has too few tile indices");
    std::string base = axisIndices.lookup(axis->getNode());
    Operation *domain = kernel.nodes.lookup(axis->getNode());
    FailureOr<std::string> extent = failure();
    if (target::emission::isRaggedBoundAxis(planIndex.components,
                                            axis->getNode())) {
      FailureOr<int64_t> orderedAxis =
          target::emission::representativeOrderedAxis(planIndex,
                                                      axis->getNode(), operation);
      if (failed(orderedAxis))
        return failure();
      extent = "sequence_end_" + std::to_string(*orderedAxis);
    } else if (domain) {
      extent = dimensionName(*domain);
    }
    if (base.empty() || failed(extent))
      return operation.emitOpError(
          "bounded TileLang transfer has no active axis interval");
    if (physicalOnly && !planIndex.blockExtents.count(*extent)) {
      ++tileAxis;
      continue;
    }
    predicates.push_back(base + " + " + tileIndices[tileAxis++] + " < " +
                         *extent);
  }
  if (tileAxis != tileIndices.size())
    return operation.emitOpError(
        "bounded TileLang transfer has unused tile indices");
  if (predicates.empty())
    return operation.emitOpError(
        "bounded TileLang transfer has no dynamic boundary predicate");
  std::string result;
  for (auto [index, predicate] : llvm::enumerate(predicates)) {
    if (index)
      result += " and ";
    result += predicate;
  }
  return result;
}

FailureOr<std::string> SourceEmitter::scalarTransferPredicate(
    Operation &operation, const plan::BoundaryOp &boundary) {
  SmallVector<std::string> predicates;
  FailureOr<bool> derivedScalar = target::hasDerivedScalarIndex(operation);
  if (failed(derivedScalar))
    return failure();
  if (*derivedScalar) {
    FailureOr<std::string> address = elementBoundsPredicate(operation, {});
    if (failed(address))
      return failure();
    if (!address->empty())
      predicates.push_back(*address);
  }
  for (int64_t node : boundary.getDomainNodes()) {
    auto found = planIndex.axes.find(node);
    if (found == planIndex.axes.end() ||
        !target::emission::isPackedScalarAxis(found->second))
      continue;
    std::string exact = axisIndices.lookup(node);
    Operation *domain = kernel.nodes.lookup(node);
    FailureOr<std::string> extent =
        domain ? dimensionName(*domain)
               : FailureOr<std::string>(failure());
    if (exact.empty() || failed(extent))
      return operation.emitOpError(
          "packed scalar transfer has no exact TileLang interval");
    predicates.push_back("0 <= " + exact);
    predicates.push_back(exact + " < " + *extent);
  }
  std::string result;
  for (StringRef predicate : predicates) {
    if (!result.empty())
      result += " and ";
    result += predicate;
  }
  return result;
}

FailureOr<std::string> SourceEmitter::wholeTileBoundsPredicate(
    Operation &operation, ArrayRef<std::string> tileExtents) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  if (failed(relation) || failed(view) ||
      static_cast<size_t>(llvm::count_if(
          *relation, [](const target::IndexTerm &term) {
            return term.kind != "new_axis";
          })) != (*view)->shape.size())
    return failure();
  SmallVector<std::string> predicates;
  unsigned tileAxis = 0;
  unsigned sourceAxis = 0;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis") {
      if (tileAxis >= tileExtents.size())
        return operation.emitOpError(
            "whole-tile TileLang transfer has too few new-axis extents");
      ++tileAxis;
      continue;
    }
    unsigned axisNumber = sourceAxis++;
    if (term.kind == "full_slice") {
      if (tileAxis >= tileExtents.size())
        return operation.emitOpError(
            "whole-tile TileLang transfer has too few tile extents");
      ++tileAxis;
      continue;
    }
    if (term.kind == "static_index")
      continue;
    if ((term.kind != "region_index" && term.kind != "value_index") ||
        term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "whole-tile TileLang transfer has no mechanical index relation");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index" &&
        isa<RankedTensorType>(indexed.getType())) {
      auto tensor = cast<RankedTensorType>(indexed.getType());
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (tensor.getRank() != 1 || failed(exact) ||
          tileAxis >= tileExtents.size() || tileExtents[tileAxis] != "1")
        return operation.emitOpError(
            "TileLang bulk bounds require one singleton indirect index tile");
      ++tileAxis;
      std::string index = exact->str() + "[0]";
      predicates.push_back("0 <= " + index);
      predicates.push_back(index + " < " + (*view)->shape[axisNumber]);
      continue;
    }
    if (term.kind == "value_index" &&
        !isa<RankedTensorType>(indexed.getType())) {
      FailureOr<target::ScalarIndexSource> source =
          target::traceScalarIndexSource(indexed, operation);
      if (failed(source))
        return failure();
      if (source->domain && source->transformed) {
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (failed(exact))
          return failure();
        predicates.push_back("0 <= (" + exact->str() + ")");
        predicates.push_back("(" + exact->str() + ") < " +
                             (*view)->shape[axisNumber]);
      }
      continue;
    }
    FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
    if (failed(axis))
      return failure();
    if (axis->isScalar())
      continue;
    if (tileAxis >= tileExtents.size())
      return operation.emitOpError(
          "whole-tile TileLang transfer has too few tile extents");
    std::string base = axisIndices.lookup(axis->getNode());
    FailureOr<std::string> extent = failure();
    if (target::emission::isRaggedBoundAxis(planIndex.components,
                                            axis->getNode())) {
      FailureOr<int64_t> orderedAxis =
          target::emission::representativeOrderedAxis(planIndex,
                                                      axis->getNode(), operation);
      if (failed(orderedAxis))
        return failure();
      extent = "sequence_end_" + std::to_string(*orderedAxis);
    } else {
      Operation *domain = kernel.nodes.lookup(axis->getNode());
      if (domain)
        extent = dimensionName(*domain);
    }
    if (base.empty() || failed(extent))
      return operation.emitOpError(
          "whole-tile TileLang transfer has no active axis interval");
    predicates.push_back(base + " + " + tileExtents[tileAxis++] + " <= " +
                         *extent);
  }
  if (tileAxis != tileExtents.size())
    return operation.emitOpError(
        "whole-tile TileLang transfer has unused tile extents");
  if (predicates.empty())
    return std::string();
  std::string result = predicates.front();
  for (StringRef predicate : llvm::drop_begin(predicates))
    result += " and " + predicate.str();
  return result;
}

FailureOr<std::string> SourceEmitter::elementValidityPredicate(
    ArrayRef<int64_t> tensorAxes, ArrayRef<int64_t> domainNodes,
    ArrayRef<std::string> elementIndices, Operation &consumer) {
  if (tensorAxes.size() != domainNodes.size())
    return consumer.emitOpError(
        "has no TileLang value-validity binding");
  SmallVector<std::string> predicates;
  for (auto [tensorAxis, domainNode] : llvm::zip(tensorAxes, domainNodes)) {
    auto domain = kernel.nodes.find(domainNode);
    auto physical = planIndex.axes.find(domainNode);
    if (tensorAxis < 0 ||
        static_cast<size_t>(tensorAxis) >= elementIndices.size() ||
        domain == kernel.nodes.end() || physical == planIndex.axes.end())
      return consumer.emitOpError(
          "references an unresolved TileLang validity axis");
    plan::AxisOp axis = physical->second;
    if (axis.isScalar())
      continue;
    std::string base = axisIndices.lookup(axis.getNode());
    if (base.empty())
      return consumer.emitOpError(
          "has no active TileLang index for its planned validity axis");
    FailureOr<std::string> extent = failure();
    if (target::emission::isRaggedBoundAxis(planIndex.components,
                                            axis.getNode())) {
      FailureOr<int64_t> orderedAxis =
          target::emission::representativeOrderedAxis(planIndex, axis.getNode(),
                                                      consumer);
      if (failed(orderedAxis))
        return failure();
      extent = "sequence_end_" + std::to_string(*orderedAxis);
    } else {
      extent = dimensionName(*domain->second);
    }
    if (failed(extent))
      return failure();
    predicates.push_back(base + " + " + elementIndices[tensorAxis] + " < " +
                         *extent);
  }
  if (predicates.empty())
    return std::string("True");
  std::string result = predicates.front();
  for (StringRef predicate : llvm::drop_begin(predicates))
    result += " and " + predicate.str();
  return result;
}

FailureOr<std::string> SourceEmitter::padElementExpression(
    Value value, StringRef expression, ArrayRef<std::string> elementIndices,
    Operation &consumer) {
  FailureOr<int64_t> valueID =
      target::getValueID(value, kernel, consumer, "TileLang padding lookup");
  if (failed(valueID))
    return failure();
  plan::PaddingOp padding = planIndex.paddings.lookup(*valueID);
  if (!padding)
    return expression.str();
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor)
    return consumer.emitOpError(
        "cannot apply TileLang padding to a non-tensor value");
  std::string dtype = dtypeName(tensor.getElementType(), consumer);
  if (dtype.empty())
    return failure();
  FailureOr<std::string> predicate = elementValidityPredicate(
      padding.getTensorAxes(), padding.getDomainNodes(), elementIndices,
      consumer);
  if (failed(predicate))
    return failure();
  std::string fill = padding.getFill() == "negative_infinity"
                         ? "-T.infinity(" + dtype + ")"
                     : padding.getFill() == "true" ? "True"
                     : padding.getFill() == "false" ? "False"
                     : isa<IntegerType, IndexType>(tensor.getElementType())
                         ? "0"
                         : "0.0";
  return "T.if_then_else(" + *predicate + ", " + expression.str() + ", " +
         fill + ")";
}

FailureOr<SmallVector<std::string>>
SourceEmitter::tensorExtents(Operation &operation, unsigned resultIndex,
                             bool physical) {
  auto shapes = operation.getAttrOfType<ArrayAttr>("intent.result_shapes");
  auto shape = shapes && resultIndex < shapes.size()
                   ? dyn_cast<ArrayAttr>(shapes[resultIndex])
                   : ArrayAttr();
  if (!shape)
    return operation.emitOpError("has no canonical tensor shape metadata");
  SmallVector<std::string> extents;
  for (Attribute attribute : shape) {
    auto label = dyn_cast<StringAttr>(attribute);
    if (!label)
      return operation.emitOpError(
          "tensor shape contains a non-symbolic extent");
    auto tile = regionTiles.find(label.getValue());
    if (tile != regionTiles.end())
      extents.push_back(tile->getValue());
    else if (!planIndex.stages.empty() && activeStages.size() == 1 &&
             label.getValue() ==
                 stageFeatureDimensions.lookup(activeStages.front()))
      extents.push_back("TILE_SIZE_N");
    else if (label.getValue().starts_with("?region_"))
      return operation.emitOpError(
          "tensor shape region has no TileLang tile binding");
    else
      extents.push_back(physical ? physicalExtent(label.getValue())
                                 : dimensionSpelling(label.getValue()));
  }
  return extents;
}

FailureOr<std::string> SourceEmitter::tensorShape(Operation &operation,
                                                  unsigned resultIndex) {
  FailureOr<SmallVector<std::string>> extents =
      tensorExtents(operation, resultIndex);
  if (failed(extents))
    return failure();
  std::string shape = "(";
  for (auto [index, extent] : llvm::enumerate(*extents)) {
    if (index)
      shape += ", ";
    shape += extent;
  }
  if (extents->size() == 1)
    shape += ",";
  return shape + ")";
}

FailureOr<std::string>
SourceEmitter::tensorElement(Value value, ArrayRef<std::string> indices,
                             Operation &consumer) {
  auto found = valueNames.find(value);
  if (found == valueNames.end())
    return consumer.emitOpError("references an unemitted TileLang value");
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor)
    return found->second;
  if (tensor.getRank() > static_cast<int64_t>(indices.size()))
    return consumer.emitOpError("cannot broadcast a higher-rank TileLang value");
  unsigned offset = indices.size() - tensor.getRank();
  std::string result = found->second + "[";
  for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
    if (axis)
      result += ", ";
    int64_t extent = tensor.getDimSize(axis);
    result += extent == 1 ? "0" : indices[offset + axis];
  }
  return result + "]";
}

FailureOr<std::string> SourceEmitter::allocateResult(Operation &operation,
                                                     unsigned resultIndex,
                                                     StringRef space) {
  if (resultIndex >= operation.getNumResults())
    return operation.emitOpError("references a missing result allocation");
  auto tensor = dyn_cast<RankedTensorType>(operation.getResult(resultIndex).getType());
  FailureOr<std::string> shape = tensorShape(operation, resultIndex);
  if (!tensor || failed(shape))
    return operation.emitOpError("requires a ranked TileLang result buffer");
  std::string dtype = dtypeName(tensor.getElementType(), operation);
  if (dtype.empty())
    return failure();
  std::string result = makeResultName(operation, resultIndex);
  StringRef allocator = space == "shared"     ? "T.alloc_shared"
                        : space == "local"    ? "T.alloc_local"
                        : space == "fragment" ? "T.alloc_fragment"
                                               : StringRef();
  if (allocator.empty())
    return operation.emitOpError("has an unsupported TileLang allocation space");
  line(result + " = " + allocator.str() + "(" + *shape + ", " + dtype + ")");
  return result;
}

std::string SourceEmitter::dtypeName(Type type, Operation &consumer) {
  if (type.isF16())
    return "T.float16";
  if (type.isF32())
    return "T.float32";
  if (type.isF64())
    return "T.float64";
  if (type.isBF16())
    return "T.bfloat16";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 8)
    return integer.isUnsigned() ? "T.uint8" : "T.int8";
  if (type.isInteger(1))
    return "T.bool";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 32)
    return "T.int32";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 64 && !integer.isUnsigned())
    return "T.int64";
  if (isa<IndexType>(type))
    return "T.int32";
  consumer.emitOpError("uses an unsupported TileLang dtype");
  return {};
}

void SourceEmitter::bindResult(Operation &operation, unsigned index,
                               StringRef name) {
  Value value = operation.getResult(index);
  valueNames[value] = name.str();
  auto owner = stageOutputOwners.find(value);
  if (planIndex.stages.empty() || owner == stageOutputOwners.end())
    return;
  std::string feature = stageFeatureDimensions.lookup(owner->second);
  line("for store_i, store_j in T.Parallel(TILE_SIZE_M, TILE_SIZE_N):");
  ++indentation;
  line("if member_start + store_i < route_end and "
       "bid_feature * TILE_SIZE_N + store_j < " + feature + ":");
  ++indentation;
  line(workspaceNames.lookup(value) +
       "[" + addressIndex("member_start + store_i") + ", " +
       addressIndex("bid_feature * TILE_SIZE_N + store_j") + "] = " +
       name.str() + "[store_i, store_j]");
  --indentation;
  --indentation;
}

std::string SourceEmitter::uniqueName(StringRef candidate, int64_t node) {
  std::string result = candidate.str();
  if (!usedNames.insert(result).second) {
    result += "_" + std::to_string(node);
    usedNames.insert(result);
  }
  return result;
}

std::string SourceEmitter::makeResultName(Operation &operation, unsigned index) {
  auto names = operation.getAttrOfType<ArrayAttr>("intent.result_names");
  auto name = names && index < names.size() ? dyn_cast<StringAttr>(names[index])
                                           : StringAttr();
  FailureOr<int64_t> node = target::getNodeID(operation, "result naming");
  return uniqueName(name ? name.getValue() : "value", *node);
}

std::string SourceEmitter::makeRegionArgumentName(Operation &operation,
                                                  unsigned argumentIndex) {
  auto regions =
      operation.getAttrOfType<ArrayAttr>("intent.region_argument_names");
  auto blocks = regions && !regions.empty() ? dyn_cast<ArrayAttr>(regions[0])
                                            : ArrayAttr();
  auto arguments = blocks && !blocks.empty() ? dyn_cast<ArrayAttr>(blocks[0])
                                             : ArrayAttr();
  auto name = arguments && argumentIndex < arguments.size()
                  ? dyn_cast<StringAttr>(arguments[argumentIndex])
                  : StringAttr();
  FailureOr<int64_t> node = target::getNodeID(operation, "region naming");
  return uniqueName(name ? name.getValue() : "region", *node);
}

void SourceEmitter::line(StringRef text) {
  if (!planIndex.stages.empty()) {
    for (unsigned stage : activeStages)
      stageLine(stage, text, indentation);
    return;
  }
  output.indent(indentation * 4) << text << "\n";
}

LogicalResult emitRealizedKernelSource(
    target::KernelModel kernel, intent::plan::RealizationOp realization,
    intent::plan::SearchSpaceOp searchSpace, raw_ostream &output) {
  FailureOr<RealizationIndex> indexed = indexRealization(realization, kernel);
  FailureOr<SearchIndex> indexedSearch = indexSearchSpace(searchSpace);
  if (failed(indexed) || failed(indexedSearch))
    return failure();
  return SourceEmitter(std::move(kernel), realization, searchSpace,
                       std::move(*indexed), std::move(*indexedSearch), output)
      .emit();
}

} // namespace intent::tilelang::emission
