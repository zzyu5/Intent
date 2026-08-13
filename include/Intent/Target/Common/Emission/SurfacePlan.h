#ifndef INTENT_TARGET_COMMON_EMISSION_SURFACEPLAN_H
#define INTENT_TARGET_COMMON_EMISSION_SURFACEPLAN_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "Intent/Target/Common/Analysis/LogicalBuffer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringMap.h"

#include <optional>
#include <array>
#include <functional>
#include <string>
#include <utility>

namespace intent::target::emission {

inline mlir::FailureOr<std::string>
reductionRole(mlir::Operation &operation) {
  auto combine = operation.getAttrOfType<mlir::StringAttr>("intent.combine");
  auto tie = operation.getAttrOfType<mlir::StringAttr>("intent.tie");
  if (operation.getName().getStringRef() == "intent.arg_reduce" && combine &&
      combine.getValue() == "maximum" && tie &&
      tie.getValue() == "lowest_index")
    return std::string("reduce_argmax");
  if (combine && combine.getValue() == "maximum")
    return std::string("reduce_maximum");
  if (combine && combine.getValue() == "add")
    return std::string("reduce_add");
  if (combine && combine.getValue() == "logical_or")
    return std::string("reduce_any");
  if (combine && combine.getValue() == "logical_and")
    return std::string("reduce_all");
  return operation.emitOpError("has no supported reduction semantics");
}

inline mlir::FailureOr<int64_t>
reductionAxis(mlir::Operation &operation) {
  auto axes = operation.getAttrOfType<mlir::ArrayAttr>("intent.axes");
  auto axis = axes && axes.size() == 1
                  ? mlir::dyn_cast<mlir::IntegerAttr>(axes[0])
                  : mlir::IntegerAttr();
  if (!axis || axis.getInt() < 0)
    return operation.emitOpError("has no canonical single reduction axis");
  return axis.getInt();
}

inline mlir::FailureOr<llvm::SmallVector<int64_t>>
transposePermutation(mlir::Operation &operation) {
  auto result = operation.getNumResults() == 1
                    ? mlir::dyn_cast<mlir::RankedTensorType>(
                          operation.getResult(0).getType())
                    : mlir::RankedTensorType();
  auto permutation =
      operation.getAttrOfType<mlir::ArrayAttr>("intent.permutation");
  if (!result || !permutation ||
      permutation.size() != static_cast<size_t>(result.getRank()))
    return operation.emitOpError("has no canonical transpose permutation");
  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<bool> covered(result.getRank(), false);
  for (mlir::Attribute attribute : permutation) {
    auto axis = mlir::dyn_cast<mlir::IntegerAttr>(attribute);
    if (!axis || axis.getInt() < 0 || axis.getInt() >= result.getRank() ||
        covered[axis.getInt()])
      return operation.emitOpError(
          "transpose permutation must cover every tensor axis");
    axes.push_back(axis.getInt());
    covered[axis.getInt()] = true;
  }
  return axes;
}

inline mlir::FailureOr<std::string> scanRole(mlir::Operation &operation) {
  auto combine = operation.getAttrOfType<mlir::StringAttr>("intent.combine");
  auto inclusive =
      operation.getAttrOfType<mlir::BoolAttr>("intent.inclusive");
  if (combine && combine.getValue() == "add" && inclusive &&
      inclusive.getValue())
    return std::string("scan_inclusive_add");
  return operation.emitOpError("has no supported scan semantics");
}

inline mlir::FailureOr<int64_t> scanAxis(mlir::Operation &operation) {
  auto axis = operation.getAttrOfType<mlir::IntegerAttr>("intent.axis");
  if (!axis || axis.getInt() < 0)
    return operation.emitOpError("has no canonical scan axis");
  return axis.getInt();
}

inline mlir::FailureOr<std::string>
pointwiseRole(mlir::Operation &operation) {
  llvm::StringRef name = operation.getName().getStringRef();
  if (name == "intent.indices")
    return std::string("indices");
  if (name == "intent.random") {
    auto algorithm =
        operation.getAttrOfType<mlir::StringAttr>("intent.algorithm");
    if (algorithm && algorithm.getValue() == "counter_xorshift32")
      return std::string("counter_random_f32");
    return operation.emitOpError("has no supported counter RNG semantics");
  }
  if (name == "intent.broadcast")
    return std::string("broadcast");
  if (name == "intent.cast")
    return std::string("cast");
  if (name == "intent.reshape")
    return std::string("reshape");
  if (name == "intent.transpose")
    return std::string("transpose");
  if (name == "intent.mask")
    return std::string("mask");
  if (name == "intent.select")
    return std::string("select");
  if (name == "intent.full")
    return std::string("full");
  if (name == "intent.zeros")
    return std::string("zeros");
  if (name == "intent.members")
    return std::string("members");
  if (name == "intent.compare") {
    auto predicate =
        operation.getAttrOfType<mlir::StringAttr>("intent.predicate");
    if (!predicate)
      return operation.emitOpError("has no comparison predicate");
    if (predicate.getValue() == "eq")
      return std::string("compare_equal");
    if (predicate.getValue() == "ne")
      return std::string("compare_not_equal");
    if (predicate.getValue() == "lt")
      return std::string("compare_less");
    if (predicate.getValue() == "le")
      return std::string("compare_less_equal");
    if (predicate.getValue() == "gt")
      return std::string("compare_greater");
    if (predicate.getValue() == "ge")
      return std::string("compare_greater_equal");
    return operation.emitOpError("has no supported comparison semantics");
  }
  if (name == "intent.gather") {
    mlir::FailureOr<llvm::SmallVector<target::IndexTerm>> relation =
        target::parseIndexRelation(operation);
    if (mlir::failed(relation))
      return mlir::failure();
    bool hasNewAxis = llvm::any_of(*relation, [](const target::IndexTerm &term) {
      return term.kind == "new_axis";
    });
    bool expand = hasNewAxis && llvm::all_of(
                                    *relation, [](const target::IndexTerm &term) {
                                      return term.kind == "full_slice" ||
                                             term.kind == "new_axis";
                                    });
    bool indirect = llvm::any_of(*relation, [](const target::IndexTerm &term) {
      return term.kind == "value_index";
    });
    if (expand)
      return std::string("expand_dims");
    if (indirect)
      return std::string("indirect_gather");
    return operation.emitOpError("has no supported gather relation");
  }
  auto logical = operation.getAttrOfType<mlir::StringAttr>("intent.operator");
  if (name == "intent.unary" && logical)
    return ("unary_" + logical.getValue()).str();
  if (name == "intent.binary" && logical)
    return ("binary_" + logical.getValue()).str();
  return operation.emitOpError("has no supported pointwise semantics");
}

inline bool feedsContraction(mlir::Operation &operation) {
  return operation.getNumResults() == 1 &&
         llvm::any_of(operation.getResult(0).getUsers(), [](mlir::Operation *user) {
           return user->getName().getStringRef() == "intent.contract";
         });
}

inline bool isNestedInStateStream(mlir::Operation *operation) {
  for (mlir::Operation *parent = operation; parent; parent = parent->getParentOp())
    if (parent->getName().getStringRef() == "intent.state_stream")
      return true;
  return false;
}

inline bool isSequentialIterator(mlir::Value value) {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  mlir::Operation *owner =
      argument ? argument.getOwner()->getParentOp() : nullptr;
  return owner && owner->getName().getStringRef() == "intent.for" &&
         argument.getArgNumber() == 0;
}

inline bool dependsOnStateStream(mlir::Value value,
                                 llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  if (mlir::Operation *definition = value.getDefiningOp()) {
    if (definition->getName().getStringRef() == "intent.state_stream" ||
        isNestedInStateStream(definition))
      return true;
    return llvm::any_of(definition->getOperands(), [&](mlir::Value operand) {
      return dependsOnStateStream(operand, visited);
    });
  }
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  mlir::Operation *owner =
      argument ? argument.getOwner()->getParentOp() : nullptr;
  return owner && owner->getName().getStringRef() == "intent.state_stream";
}

inline bool feedsStateStream(mlir::Value value,
                             llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  for (mlir::Operation *user : value.getUsers()) {
    if (user->getName().getStringRef() == "intent.state_stream" ||
        isNestedInStateStream(user))
      return true;
    for (mlir::Value result : user->getResults())
      if (feedsStateStream(result, visited))
        return true;
  }
  return false;
}

inline bool touchesStateStream(mlir::Operation &operation) {
  if (isNestedInStateStream(operation.getParentOp()))
    return true;
  llvm::DenseSet<mlir::Value> visited;
  for (mlir::Value operand : operation.getOperands()) {
    if (dependsOnStateStream(operand, visited))
      return true;
  }
  visited.clear();
  for (mlir::Value result : operation.getResults())
    if (feedsStateStream(result, visited))
      return true;
  return false;
}

template <typename OperationType>
struct Binding {
  mutable OperationType operation;

  explicit operator bool() const { return static_cast<bool>(operation); }
  mlir::Location getLoc() const { return operation.getLoc(); }
  mlir::InFlightDiagnostic emitOpError() const {
    return operation.emitOpError();
  }
  mlir::InFlightDiagnostic emitOpError(const llvm::Twine &message) const {
    return operation.emitOpError(message);
  }
};

struct TargetBinding : Binding<intent::plan::DeviceOp> {
  int64_t getDevice() const { return operation.getDevice(); }
};

struct RangeBinding : Binding<intent::plan::RangeOp> {
  std::string tile;

  int64_t getAxisNode() const { return operation.getAxisNode(); }
  llvm::StringRef getPurpose() const { return operation.getPurpose(); }
  int64_t getLevel() const { return operation.getLevel(); }
  llvm::StringRef getTile() const { return tile; }
  llvm::StringRef getTileRole() const { return operation.getTile(); }
  int64_t getTransferNode() const {
    return operation.getTransferNodeAttr().getInt();
  }
  int64_t getSourceAxis() const {
    return operation.getSourceAxisAttr().getInt();
  }
};

struct AxisBinding : Binding<intent::plan::AxisOp> {
  std::string role;
  std::string group;
  llvm::SmallVector<RangeBinding> ranges;

  int64_t getNode() const { return operation.getNode(); }
  mlir::IntegerAttr getNodeAttr() const { return operation.getNodeAttr(); }
  llvm::ArrayRef<mlir::Attribute> getRoles() const {
    return operation.getRoles().getValue();
  }
  bool hasRole(llvm::StringRef expected) const {
    for (mlir::Attribute attribute : operation.getRoles()) {
      auto value = mlir::dyn_cast<mlir::StringAttr>(attribute);
      if (value && value.getValue() == expected)
        return true;
    }
    return false;
  }
  llvm::StringRef getRole() const { return role; }
  const RangeBinding *getRange(llvm::StringRef purpose,
                               int64_t level = 0) const {
    auto found = llvm::find_if(ranges, [&](const RangeBinding &range) {
      return range.getPurpose() == purpose && range.getLevel() == level;
    });
    if (found == ranges.end())
      return nullptr;
    return &*found;
  }
  const RangeBinding *roleRange() const {
    llvm::StringRef activeRole(role);
    if (activeRole.starts_with("program_") ||
        activeRole.starts_with("ragged_member_"))
      return getRange("ownership");
    if (activeRole.starts_with("stream_"))
      return getRange("traversal");
    if (activeRole.starts_with("reduction_"))
      return getRange("reduction");
    if (activeRole.starts_with("lane_"))
      return getRange("lane");
    return nullptr;
  }
  llvm::StringRef getTile() const {
    const RangeBinding *range = roleRange();
    return range ? range->getTile() : llvm::StringRef();
  }
  llvm::StringRef getTileRole() const {
    const RangeBinding *range = roleRange();
    return range ? range->getTileRole() : llvm::StringRef();
  }
  bool isScalar() const {
    const RangeBinding *range = roleRange();
    return range && range->getTileRole() == "one";
  }
  mlir::IntegerAttr getProgramOrderAttr() const {
    return operation.getProgramOrderAttr();
  }
  int64_t getProgramOrder() const {
    return operation.getProgramOrderAttr().getInt();
  }
  mlir::IntegerAttr getWorkerAxisAttr() const {
    return operation.getWorkerAxisAttr();
  }
  int64_t getWorkerAxis() const { return operation.getWorkerAxisAttr().getInt(); }
  mlir::IntegerAttr getFoldOrderAttr() const {
    return operation.getFoldOrderAttr();
  }
  int64_t getFoldOrder() const { return operation.getFoldOrderAttr().getInt(); }
  bool getReuseWorker() const { return operation.getReuseWorker(); }
  mlir::StringAttr getGroupAttr() const { return operation.getGroupAttr(); }
  std::optional<llvm::StringRef> getGroup() const {
    return operation.getGroup();
  }
  llvm::StringRef getGroupSpelling() const { return group; }
};

inline bool isPackedScalarAxis(const AxisBinding &axis) {
  return axis.hasRole("parallel") && axis.hasRole("lane") &&
         axis.hasRole("packed_lane");
}

template <typename PlanIndex, typename TileSpelling>
mlir::LogicalResult indexAxisRanges(
    PlanIndex &index, llvm::ArrayRef<intent::plan::RangeOp> ranges,
    TileSpelling spelling) {
  for (intent::plan::RangeOp range : ranges) {
    auto axis = index.axes.find(range.getAxisNode());
    if (axis == index.axes.end())
      return range.emitOpError("references an unbound target axis");
    mlir::FailureOr<std::string> tile = spelling(range, range.getTile());
    if (mlir::failed(tile))
      return mlir::failure();
    RangeBinding binding;
    binding.operation = range;
    binding.tile = std::move(*tile);
    axis->second.ranges.push_back(std::move(binding));
  }
  for (auto &entry : index.axes)
    llvm::sort(entry.second.ranges,
               [](const RangeBinding &lhs, const RangeBinding &rhs) {
                 if (lhs.getPurpose() != rhs.getPurpose())
                   return lhs.getPurpose() < rhs.getPurpose();
                 return lhs.getLevel() < rhs.getLevel();
               });
  return mlir::success();
}

template <typename PlanIndex>
llvm::SmallVector<RangeBinding>
accessRangesForTransfer(const PlanIndex &index, int64_t transferNode) {
  llvm::SmallVector<RangeBinding> result;
  for (const auto &entry : index.axes)
    for (const RangeBinding &range : entry.second.ranges)
      if (range.getPurpose() == "access" &&
          range.getTransferNode() == transferNode)
        result.push_back(range);
  llvm::sort(result, [](const RangeBinding &lhs, const RangeBinding &rhs) {
    return lhs.getSourceAxis() < rhs.getSourceAxis();
  });
  return result;
}

struct ProgramBinding : Binding<intent::plan::ProgramOp> {
  int64_t getLoopNode() const { return operation.getLoopNode(); }
  bool getPersistent() const { return operation.getPersistent(); }
  mlir::IntegerAttr getLoopNodeAttr() const {
    return operation.getLoopNodeAttr();
  }
};

struct BlockExtentBinding : Binding<intent::plan::BlockExtentOp> {
  llvm::StringRef getLogicalExtent() const {
    return operation.getLogicalExtent();
  }
  llvm::StringRef getRounding() const { return operation.getRounding(); }
  llvm::StringRef getFill() const { return operation.getFill(); }
};

template <typename PlanIndex>
mlir::FailureOr<std::string> transferPhysicalExtentFill(
    const PlanIndex &index, llvm::ArrayRef<target::IndexTerm> relation,
    llvm::ArrayRef<std::string> viewShape, mlir::Operation &operation) {
  if (static_cast<size_t>(llvm::count_if(
          relation, [](const target::IndexTerm &term) {
            return term.kind != "new_axis";
          })) != viewShape.size())
    return operation.emitOpError(
        "physical block-extent projection does not match the external view rank");
  std::string fill;
  unsigned viewAxis = 0;
  for (const target::IndexTerm &term : relation) {
    if (term.kind == "new_axis")
      continue;
    llvm::StringRef viewExtent = viewShape[viewAxis++];
    bool vectorAccess = term.kind == "full_slice";
    if ((term.kind == "region_index" || term.kind == "value_index") &&
        term.operands.size() == 1 && term.operands.front()) {
      mlir::Value indexed = operation.getOperand(*term.operands.front());
      vectorAccess = term.kind == "region_index"
                         ? !isSequentialIterator(indexed)
                         : mlir::isa<mlir::RankedTensorType>(indexed.getType());
    }
    if (!vectorAccess)
      continue;
    auto extent = index.blockExtents.find(viewExtent);
    if (extent == index.blockExtents.end())
      continue;
    if (!fill.empty() && fill != extent->second.getFill())
      return operation.emitOpError(
          "uses incompatible physical block-extent fill rules");
    fill = extent->second.getFill().str();
  }
  return fill;
}

struct BufferBinding : Binding<intent::plan::BufferOp> {
  int64_t getNode() const { return operation.getNode(); }
  llvm::StringRef getSpace() const { return operation.getSpace(); }
  llvm::ArrayRef<int64_t> getOwnerNodes() const {
    return operation.getOwnerNodes();
  }
};

template <typename PlanIndex, typename LogicalIndexSpelling>
inline mlir::FailureOr<std::string> projectPrivateWorkspaceOffset(
    const BufferBinding &binding, const target::LogicalBufferInfo &info,
    llvm::ArrayRef<target::LogicalBufferIndex> logicalIndices,
    const PlanIndex &index,
    const llvm::DenseMap<int64_t, std::string> &axisIndices,
    const llvm::StringMap<std::string> &roleDimensions,
    LogicalIndexSpelling spellLogicalIndex, mlir::Operation &operation) {
  if (logicalIndices.size() != info.shape.size())
    return operation.emitOpError(
        "private-workspace projection does not match its logical buffer rank");

  std::string offset;
  auto append = [&](llvm::StringRef projectedIndex, llvm::StringRef extent) {
    offset = offset.empty()
                 ? projectedIndex.str()
                 : "(" + offset + ") * (" + extent.str() + ") + (" +
                       projectedIndex.str() + ")";
  };
  for (int64_t owner : binding.getOwnerNodes()) {
    auto axis = index.axes.find(owner);
    std::string projectedIndex = axisIndices.lookup(owner);
    std::string extent =
        axis == index.axes.end()
            ? std::string()
            : roleDimensions.lookup(
                  "program_" + std::to_string(axis->second.getProgramOrder()));
    if (axis == index.axes.end() || projectedIndex.empty() || extent.empty())
      return operation.emitOpError(
          "has no active private-workspace owner projection");
    append(projectedIndex, extent);
  }
  for (auto [axis, logicalIndex] : llvm::enumerate(logicalIndices)) {
    mlir::FailureOr<std::string> projectedIndex =
        spellLogicalIndex(logicalIndex);
    if (mlir::failed(projectedIndex))
      return mlir::failure();
    append(*projectedIndex, std::to_string(info.shape[axis]));
  }
  if (offset.empty())
    return operation.emitOpError("has an empty private-workspace projection");
  return offset;
}

template <typename PlanIndex>
inline mlir::FailureOr<std::string> privateWorkspaceElementCount(
    mlir::Operation &buffer, const PlanIndex &index,
    const llvm::StringMap<std::string> &roleDimensions) {
  mlir::FailureOr<int64_t> node =
      target::getNodeID(buffer, "private-workspace allocation");
  auto binding = mlir::succeeded(node) ? index.buffers.find(*node)
                                       : index.buffers.end();
  mlir::FailureOr<target::LogicalBufferInfo> info =
      target::getLogicalBufferInfo(buffer);
  if (mlir::failed(node) || binding == index.buffers.end() ||
      binding->second.getSpace() != "private_workspace" || mlir::failed(info))
    return buffer.emitOpError("has no planned private-workspace allocation");

  std::string size;
  auto append = [&](llvm::StringRef extent) {
    size = size.empty() ? extent.str() : size + " * " + extent.str();
  };
  for (int64_t owner : binding->second.getOwnerNodes()) {
    auto axis = index.axes.find(owner);
    std::string extent =
        axis == index.axes.end()
            ? std::string()
            : roleDimensions.lookup(
                  "program_" + std::to_string(axis->second.getProgramOrder()));
    if (axis == index.axes.end() || extent.empty())
      return buffer.emitOpError("has no private-workspace owner extent");
    append(extent);
  }
  for (int64_t extent : info->shape)
    append(std::to_string(extent));
  return size;
}

inline mlir::FailureOr<std::string>
logicalBufferPythonInitializer(mlir::Operation &buffer) {
  mlir::Operation *constant = buffer.getNumOperands() == 1
                                  ? buffer.getOperand(0).getDefiningOp()
                                  : nullptr;
  if (!constant ||
      constant->getName().getStringRef() != "intent.constant")
    return buffer.emitOpError(
        "private workspace initializer must be a scalar constant");
  if (auto integer =
          constant->getAttrOfType<mlir::IntegerAttr>("intent.value"))
    return std::to_string(integer.getInt());
  auto floating =
      constant->getAttrOfType<mlir::FloatAttr>("intent.value");
  if (!floating)
    return buffer.emitOpError(
        "private workspace initializer has no scalar constant value");
  const llvm::APFloat &value = floating.getValue();
  if (value.isNaN())
    return std::string("float('nan')");
  if (value.isInfinity())
    return value.isNegative() ? std::string("-float('inf')")
                              : std::string("float('inf')");
  llvm::SmallString<32> spelling;
  value.toString(spelling);
  return spelling.str().str();
}

struct PaddingBinding : Binding<intent::plan::PaddingOp> {
  int64_t getValue() const { return operation.getValue(); }
  llvm::ArrayRef<int64_t> getTensorAxes() const {
    return operation.getTensorAxes();
  }
  llvm::ArrayRef<int64_t> getDomainNodes() const {
    return operation.getDomainNodes();
  }
  llvm::StringRef getFill() const { return operation.getFill(); }
};

struct ReductionBinding : Binding<intent::plan::ReductionOp> {
  std::string lowering;
  std::string resultSpace;
  int64_t axis = -1;
  bool keepDims = false;

  int64_t getNode() const { return operation.getNode(); }
  llvm::StringRef getLowering() const { return lowering; }
  int64_t getAxis() const { return axis; }
  bool getKeepDims() const { return keepDims; }
  llvm::StringRef getResultSpace() const { return resultSpace; }
};

struct ScanBinding : Binding<intent::plan::ScanOp> {
  std::string lowering;
  std::string resultSpace;
  int64_t axis = -1;
  int64_t axisNode = -1;

  int64_t getNode() const { return operation.getNode(); }
  llvm::StringRef getLowering() const { return lowering; }
  int64_t getAxis() const { return axis; }
  int64_t getAxisNode() const { return axisNode; }
  llvm::StringRef getResultSpace() const { return resultSpace; }
  llvm::StringRef getCarrySpace() const { return operation.getCarrySpace(); }
  llvm::StringRef getMaterialization() const {
    return operation.getMaterialization();
  }
  llvm::ArrayRef<int64_t> getOwnerNodes() const {
    return operation.getOwnerNodes();
  }
  llvm::ArrayRef<int64_t> getProducers() const {
    return operation.getProducers();
  }
  llvm::ArrayRef<int64_t> getMaterializedValues() const {
    return operation.getMaterializedValues();
  }
};

template <typename PlanIndex, typename OperationSlices>
mlir::LogicalResult indexScanProducerOperations(
    const target::KernelModel &kernel, const PlanIndex &index,
    OperationSlices &operationSlices) {
  for (const auto &entry : index.scans) {
    const ScanBinding &scan = entry.second;
    if (scan.getMaterialization() != "scalar_access")
      continue;
    for (int64_t node : scan.getProducers()) {
      mlir::Operation *operation = kernel.nodes.lookup(node);
      if (!operation)
        return scan.emitOpError("references an unknown scan producer node");
      auto existing = operationSlices.find(operation);
      if (existing != operationSlices.end() && existing->second != scan.getNode())
        return scan.emitOpError(
            "shares a producer with another physical scan slice");
      operationSlices[operation] = scan.getNode();
    }
  }
  return mlir::success();
}

inline mlir::LogicalResult verifyScanMaterializedValues(
    const target::KernelModel &kernel, const ScanBinding &scan) {
  llvm::DenseSet<int64_t> producedValues;
  for (int64_t node : scan.getProducers()) {
    mlir::Operation *producer = kernel.nodes.lookup(node);
    if (!producer)
      return scan.emitOpError("references an unknown scan producer node");
    for (mlir::Value result : producer->getResults()) {
      mlir::FailureOr<int64_t> value = target::getValueID(
          result, kernel, *producer, "scan materialized value verification");
      if (mlir::failed(value))
        return mlir::failure();
      producedValues.insert(*value);
    }
  }
  for (int64_t value : scan.getMaterializedValues())
    if (!producedValues.contains(value))
      return scan.emitOpError(
          "materializes a value outside its physical producer slice");
  return mlir::success();
}

inline mlir::FailureOr<mlir::Value>
lookupScanMaterializedValue(const target::KernelModel &kernel,
                            const ScanBinding &scan, int64_t valueID) {
  if (!llvm::is_contained(scan.getMaterializedValues(), valueID))
    return mlir::failure();
  auto value = kernel.values.find(valueID);
  if (value == kernel.values.end() ||
      !mlir::isa<mlir::RankedTensorType>(value->second.getType()))
    return scan.emitOpError("references an invalid materialized tensor value");
  return value->second;
}

template <typename PlanIndex>
inline mlir::FailureOr<std::string> scanWorkspaceElementCount(
    const ScanBinding &binding, llvm::StringRef logicalExtent,
    const PlanIndex &index,
    const llvm::StringMap<std::string> &roleDimensions) {
  std::string size;
  auto append = [&](llvm::StringRef extent) {
    size = size.empty() ? extent.str() : size + " * " + extent.str();
  };
  for (int64_t owner : binding.getOwnerNodes()) {
    auto axis = index.axes.find(owner);
    std::string extent =
        axis == index.axes.end()
            ? std::string()
            : roleDimensions.lookup(
                  "program_" + std::to_string(axis->second.getProgramOrder()));
    if (axis == index.axes.end() || extent.empty())
      return binding.emitOpError("has no scan workspace owner extent");
    append(extent);
  }
  if (logicalExtent.empty())
    return binding.emitOpError("has no scan workspace logical extent");
  append(logicalExtent);
  return size;
}

template <typename PlanIndex>
inline mlir::FailureOr<std::string> projectScanWorkspaceOffset(
    const ScanBinding &binding, llvm::StringRef logicalExtent,
    llvm::StringRef logicalIndex, const PlanIndex &index,
    const llvm::DenseMap<int64_t, std::string> &axisIndices,
    const llvm::StringMap<std::string> &roleDimensions,
    mlir::Operation &operation) {
  std::string offset;
  auto append = [&](llvm::StringRef projectedIndex, llvm::StringRef extent) {
    offset = offset.empty()
                 ? projectedIndex.str()
                 : "(" + offset + ") * (" + extent.str() + ") + (" +
                       projectedIndex.str() + ")";
  };
  for (int64_t owner : binding.getOwnerNodes()) {
    auto axis = index.axes.find(owner);
    std::string projected = axisIndices.lookup(owner);
    std::string extent =
        axis == index.axes.end()
            ? std::string()
            : roleDimensions.lookup(
                  "program_" + std::to_string(axis->second.getProgramOrder()));
    if (axis == index.axes.end() || projected.empty() || extent.empty())
      return operation.emitOpError("has no active scan workspace owner projection");
    append(projected, extent);
  }
  if (logicalExtent.empty() || logicalIndex.empty())
    return operation.emitOpError("has no scan workspace logical projection");
  append(logicalIndex, logicalExtent);
  return offset;
}

struct PointwiseBinding : Binding<intent::plan::PointwiseOp> {
  std::string lowering;
  std::string resultSpace;
  bool defer = false;

  int64_t getNode() const { return operation.getNode(); }
  llvm::StringRef getLowering() const { return lowering; }
  llvm::StringRef getResultSpace() const { return resultSpace; }
  llvm::StringRef getSpace() const { return resultSpace; }
  mlir::IntegerAttr getReuseOperandAttr() const {
    return operation.getReuseOperandAttr();
  }
  int64_t getReuseOperand() const { return operation.getReuseOperand(); }
  bool getNonnegativeOperands() const {
    return operation.getNonnegativeOperands();
  }
  llvm::ArrayRef<int64_t> getAxisNodes() const {
    return operation.getAxisNodes();
  }
  bool getDefer() const { return defer; }
};

struct ContractBinding : Binding<intent::plan::ContractOp> {
  std::string lowering;
  std::string lhsSpace;
  std::string rhsSpace;
  std::string accumulatorSpace;

  int64_t getNode() const { return operation.getNode(); }
  llvm::StringRef getLowering() const { return lowering; }
  llvm::StringRef getLhsSpace() const { return lhsSpace; }
  llvm::StringRef getRhsSpace() const { return rhsSpace; }
  llvm::StringRef getAccumulatorSpace() const { return accumulatorSpace; }
};

struct ContractionOrientation {
  bool lhsTranspose;
  bool rhsTranspose;
};

inline mlir::FailureOr<ContractionOrientation>
contractionOrientation(mlir::Operation &operation) {
  auto lhsType = operation.getNumOperands() == 2
                     ? mlir::dyn_cast<mlir::RankedTensorType>(
                           operation.getOperand(0).getType())
                     : mlir::RankedTensorType();
  auto rhsType = operation.getNumOperands() == 2
                     ? mlir::dyn_cast<mlir::RankedTensorType>(
                           operation.getOperand(1).getType())
                     : mlir::RankedTensorType();
  auto reduce = operation.getAttrOfType<mlir::ArrayAttr>("intent.reduce");
  auto pair = reduce && reduce.size() == 1
                  ? mlir::dyn_cast<mlir::ArrayAttr>(reduce[0])
                  : mlir::ArrayAttr();
  auto lhs = pair && pair.size() == 2
                 ? mlir::dyn_cast<mlir::IntegerAttr>(pair[0])
                 : mlir::IntegerAttr();
  auto rhs = pair && pair.size() == 2
                 ? mlir::dyn_cast<mlir::IntegerAttr>(pair[1])
                 : mlir::IntegerAttr();
  if (!lhsType || lhsType.getRank() != 2 || !rhsType ||
      rhsType.getRank() != 2 || !lhs || !rhs ||
      (lhs.getInt() != 0 && lhs.getInt() != 1) ||
      (rhs.getInt() != 0 && rhs.getInt() != 1))
    return operation.emitOpError(
        "has no rank-two contraction orientation for target emission");
  return ContractionOrientation{lhs.getInt() == 0, rhs.getInt() == 1};
}

struct CanonicalBinding {
  mlir::Operation *operation = nullptr;

  explicit operator bool() const { return operation != nullptr; }
  mlir::Location getLoc() const { return operation->getLoc(); }
  mlir::InFlightDiagnostic emitOpError() const {
    return operation->emitOpError();
  }
  mlir::InFlightDiagnostic emitOpError(const llvm::Twine &message) const {
    return operation->emitOpError(message);
  }
};

struct StreamBinding : CanonicalBinding {
  int64_t node = -1;
  int64_t axisNode = -1;
  mlir::IntegerAttr stopNode;
  std::string tile;
  llvm::SmallVector<int64_t> innerReductionAxes;

  int64_t getNode() const { return node; }
  int64_t getAxisNode() const { return axisNode; }
  mlir::IntegerAttr getStopNodeAttr() const { return stopNode; }
  llvm::StringRef getTile() const { return tile; }
  llvm::ArrayRef<int64_t> getInnerReductionAxes() const {
    return innerReductionAxes;
  }
};

struct RaggedBinding : CanonicalBinding {
  int64_t node = -1;
  int64_t outerNode = -1;
  llvm::SmallVector<int64_t> memberNodes;

  int64_t getNode() const { return node; }
  int64_t getOuterNode() const { return outerNode; }
  llvm::ArrayRef<int64_t> getMemberNodes() const { return memberNodes; }
};

struct StageBinding : Binding<intent::plan::StageOp> {
  unsigned position = 0;

  unsigned getOrdinal() const { return position; }
  int64_t getNode() const { return operation.getNode(); }
  llvm::ArrayRef<int64_t> getInputs() const { return operation.getInputs(); }
  llvm::ArrayRef<int64_t> getOutputs() const { return operation.getOutputs(); }
  llvm::ArrayRef<int64_t> getOperations() const {
    return operation.getOperations();
  }
  llvm::ArrayRef<int64_t> getTerminals() const {
    return operation.getTerminals();
  }
};

struct StageAxisBinding : Binding<intent::plan::StageAxisOp> {
  int64_t getStageNode() const { return operation.getStageNode(); }
  llvm::StringRef getRole() const { return operation.getRole(); }
  mlir::IntegerAttr getAxisNodeAttr() const {
    return operation.getAxisNodeAttr();
  }
  llvm::StringRef getExtent() const { return operation.getExtent(); }
  llvm::StringRef getTile() const { return operation.getTile(); }
  mlir::IntegerAttr getWorkerAxisAttr() const {
    return operation.getWorkerAxisAttr();
  }
};

inline bool stageUsesScatterReduction(const StageBinding &stage,
                                      const target::KernelModel &kernel) {
  return llvm::any_of(stage.getTerminals(), [&](int64_t node) {
    mlir::Operation *terminal = kernel.nodes.lookup(node);
    return terminal &&
           terminal->getName().getStringRef() == "intent.scatter_reduce";
  });
}

template <typename PlanIndex>
bool planUsesScatterReduction(const PlanIndex &index,
                              const target::KernelModel &kernel) {
  return llvm::any_of(index.stages, [&](const StageBinding &stage) {
    return stageUsesScatterReduction(stage, kernel);
  });
}

struct BoundaryBinding : Binding<intent::plan::TransferOp> {
  std::string access;
  std::string transfer;
  std::string resultSpace;
  bool explicitBounds = false;
  bool defer = false;

  int64_t getNode() const { return operation.getNode(); }
  llvm::ArrayRef<int64_t> getDomainNodes() const {
    return operation.getDomainNodes();
  }
  llvm::StringRef getAccess() const { return access; }
  llvm::StringRef getLoadFill() const { return operation.getFill(); }
  llvm::StringRef getPadding() const { return operation.getFill(); }
  llvm::StringRef getTensorIndexing() const {
    return operation.getTensorIndexing();
  }
  bool hasStructuredTensorIndex() const {
    return operation.getTensorIndexing() == "structured";
  }
  bool hasDataDependentTensorIndex() const {
    return operation.getTensorIndexing() == "data_dependent";
  }
  llvm::StringRef getStoreMask() const { return "predicate"; }
  llvm::StringRef getTransfer() const { return transfer; }
  llvm::StringRef getResultSpace() const { return resultSpace; }
  bool getConsumerNeutralized() const {
    return operation.getConsumerNeutralized();
  }
  bool getCheckBounds() const { return explicitBounds; }
  bool getDefer() const { return defer; }
};

template <typename PlanIndex>
inline bool hasPackedScalarDomain(const PlanIndex &index,
                                  const BoundaryBinding &binding) {
  return llvm::any_of(binding.getDomainNodes(), [&](int64_t node) {
    auto axis = index.axes.find(node);
    return axis != index.axes.end() && isPackedScalarAxis(axis->second);
  });
}

struct AutotuneBinding : Binding<intent::plan::AutotuneOp> {
  mlir::DictionaryAttr parameterMap;

  mlir::ArrayAttr getKey() const { return operation.getKey(); }
  mlir::DictionaryAttr getParameterMap() const { return parameterMap; }
};

struct PhysicalComponents {
  llvm::SmallVector<AxisBinding> programAxes;
  llvm::SmallVector<AxisBinding> reusedAxes;
  llvm::StringMap<llvm::SmallVector<AxisBinding>> groups;
  llvm::DenseMap<int64_t, llvm::SmallVector<StreamBinding>> streamsByAxis;
  llvm::DenseMap<int64_t, llvm::SmallVector<RaggedBinding>> raggedByAxis;
  llvm::DenseSet<int64_t> orderedRaggedAxes;
  llvm::DenseSet<int64_t> orderedRaggedProgramAxes;
  llvm::DenseMap<int64_t, llvm::SmallVector<int64_t>> orderedAxesByRelation;
  llvm::DenseMap<int64_t, llvm::SmallVector<int64_t>> programAxesByRelation;
};

template <typename PlanIndex>
bool requiresDelegatedTuning(const PlanIndex &index) {
  return llvm::any_of(index.axes, [&](const auto &entry) {
    return llvm::any_of(entry.second.ranges, [](const RangeBinding &range) {
      llvm::StringRef tile = range.getTileRole();
      return tile != "one" && !tile.starts_with("row_vector") &&
             !tile.starts_with("fixed_");
    });
  });
}

template <typename PlanIndex>
bool isStagedContraction(const PlanIndex &index, mlir::Operation *operation) {
  if (!operation || operation->getName().getStringRef() != "intent.contract")
    return false;
  auto node = operation->getAttrOfType<mlir::IntegerAttr>("intent.node");
  return node && llvm::any_of(index.stages, [&](auto stage) {
           return stage.getNode() == node.getInt();
         });
}

template <typename PlanIndex>
bool feedsStagedContraction(const PlanIndex &index, mlir::Operation &operation) {
  return operation.getNumResults() == 1 &&
         llvm::any_of(operation.getResult(0).getUsers(),
                      [&](mlir::Operation *user) {
                        return isStagedContraction(index, user);
                      });
}

template <typename PlanIndex>
bool deferSharedContractionTransfer(const PlanIndex &index,
                                    mlir::Operation &operation,
                                    llvm::StringRef resultSpace) {
  if (resultSpace != "shared" || operation.getNumResults() != 1 ||
      !llvm::hasSingleElement(operation.getResult(0).getUsers()))
    return false;
  for (mlir::Operation *user : operation.getResult(0).getUsers()) {
    if (user->getName().getStringRef() != "intent.contract")
      continue;
    if (isStagedContraction(index, user))
      return true;
    auto node = user->getAttrOfType<mlir::IntegerAttr>("intent.node");
    auto contract = node ? index.contracts.find(node.getInt())
                         : index.contracts.end();
    if (contract == index.contracts.end() ||
        contract->second.getLhsSpace() != "shared" ||
        contract->second.getRhsSpace() != "shared")
      continue;
    auto boundary = [&](mlir::Operation &load) -> const BoundaryBinding * {
      auto loadNode = load.getAttrOfType<mlir::IntegerAttr>("intent.node");
      auto found = loadNode ? index.boundaries.find(loadNode.getInt())
                            : index.boundaries.end();
      return found == index.boundaries.end() ? nullptr : &found->second;
    };
    mlir::Operation *lhs = user->getOperand(0).getDefiningOp();
    mlir::Operation *rhs = user->getOperand(1).getDefiningOp();
    const BoundaryBinding *lhsBoundary = lhs ? boundary(*lhs) : nullptr;
    const BoundaryBinding *rhsBoundary = rhs ? boundary(*rhs) : nullptr;
    if (!lhsBoundary || !rhsBoundary ||
        lhsBoundary->getResultSpace() != "shared" ||
        rhsBoundary->getResultSpace() != "shared")
      continue;
    auto directLoad = [](mlir::Value operand) {
      mlir::Operation *definition = operand.getDefiningOp();
      return definition &&
             definition->getName().getStringRef() == "intent.view_load";
    };
    if (!llvm::all_of(user->getOperands(), directLoad))
      continue;
    unsigned reductionAxes = 0;
    for (const auto &entry : index.axesByRole) {
      const AxisBinding &axis = entry.getValue();
      if (!entry.getKey().starts_with("reduction_") ||
          !llvm::is_contained(lhsBoundary->getDomainNodes(), axis.getNode()) ||
          !llvm::is_contained(rhsBoundary->getDomainNodes(), axis.getNode()))
        continue;
      ++reductionAxes;
    }
    if (reductionAxes != 1)
      return false;
    return true;
  }
  return false;
}

template <typename PlanIndex>
bool isAbsorbedStagedAccessMetadata(const PlanIndex &index,
                                    mlir::Operation &operation) {
  if (index.stages.empty() || operation.getNumResults() == 0)
    return false;
  llvm::DenseSet<mlir::Value> visited;
  std::function<bool(mlir::Value)> absorbed = [&](mlir::Value value) {
    if (!visited.insert(value).second || value.use_empty())
      return false;
    for (mlir::Operation *user : value.getUsers()) {
      bool consumed = false;
      if (user->getName().getStringRef() == "intent.gather" &&
          feedsStagedContraction(index, *user)) {
        auto valid =
            user->getAttrOfType<mlir::IntegerAttr>("intent.valid_operand_index");
        auto fill =
            user->getAttrOfType<mlir::IntegerAttr>("intent.fill_operand_index");
        for (auto [operand, candidate] : llvm::enumerate(user->getOperands()))
          if (candidate == value &&
              ((valid && valid.getInt() == static_cast<int64_t>(operand)) ||
               (fill && fill.getInt() == static_cast<int64_t>(operand))))
            consumed = true;
      } else if (user->getNumResults() == 1) {
        consumed = absorbed(user->getResult(0));
      }
      if (!consumed)
        return false;
    }
    return true;
  };
  return llvm::all_of(operation.getResults(), absorbed);
}

template <typename PlanIndex>
mlir::FailureOr<RaggedBinding>
uniqueRaggedRelation(const PlanIndex &index, int64_t axis,
                     mlir::Operation &consumer) {
  auto found = index.components.raggedByAxis.find(axis);
  if (found == index.components.raggedByAxis.end())
    return consumer.emitOpError(
        "does not resolve one ragged relation for its logical axis");
  if (found->second.size() == 1)
    return found->second.front();
  llvm::SmallVector<RaggedBinding> memberRelations;
  for (RaggedBinding relation : found->second)
    if (llvm::is_contained(relation.getMemberNodes(), axis))
      memberRelations.push_back(relation);
  if (memberRelations.size() != 1)
    return consumer.emitOpError(
        "does not resolve one member relation for its logical axis");
  return memberRelations.front();
}

inline bool isRaggedBoundAxis(const PhysicalComponents &components,
                              int64_t axis) {
  return components.orderedRaggedAxes.contains(axis) ||
         components.orderedRaggedProgramAxes.contains(axis);
}

template <typename PlanIndex>
mlir::FailureOr<int64_t>
representativeOrderedAxis(const PlanIndex &index, int64_t axis,
                          mlir::Operation &consumer) {
  if (index.components.orderedRaggedAxes.contains(axis))
    return axis;
  mlir::FailureOr<RaggedBinding> relation =
      uniqueRaggedRelation(index, axis, consumer);
  if (mlir::failed(relation))
    return mlir::failure();
  auto ordered =
      index.components.orderedAxesByRelation.find(relation->getNode());
  if (ordered == index.components.orderedAxesByRelation.end() ||
      ordered->second.empty())
    return consumer.emitOpError(
        "has no ordered axis for its ragged relation");
  return ordered->second.front();
}

template <typename PlanIndex>
llvm::SmallVector<AxisBinding> orderedProgramAxes(const PlanIndex &index);

template <typename PlanIndex>
mlir::LogicalResult indexCanonicalStructure(
    PlanIndex &index, const target::KernelModel &kernel) {
  llvm::SmallVector<mlir::Operation *> relations;
  llvm::SmallVector<mlir::Operation *> streams;
  for (const auto &entry : kernel.nodes) {
    mlir::Operation *operation = entry.second;
    llvm::StringRef name = operation->getName().getStringRef();
    if (name == "intent.ragged")
      relations.push_back(operation);
    else if (name == "intent.state_stream")
      streams.push_back(operation);
  }
  auto byNode = [](mlir::Operation *lhs, mlir::Operation *rhs) {
    return lhs->getAttrOfType<mlir::IntegerAttr>("intent.node").getInt() <
           rhs->getAttrOfType<mlir::IntegerAttr>("intent.node").getInt();
  };
  llvm::sort(relations, byNode);
  llvm::sort(streams, byNode);

  for (mlir::Operation *operation : relations) {
    mlir::FailureOr<int64_t> relationNode =
        target::getNodeID(*operation, "ragged emission index");
    if (mlir::failed(relationNode) || operation->getNumResults() != 1)
      return operation->emitOpError("has no canonical ragged result");
    RaggedBinding binding;
    binding.operation = operation;
    binding.node = *relationNode;
    for (mlir::Operation *user : operation->getResult(0).getUsers()) {
      llvm::StringRef name = user->getName().getStringRef();
      if (name != "intent.ragged_outer" && name != "intent.ragged_member")
        continue;
      mlir::FailureOr<int64_t> userNode =
          target::getNodeID(*user, "ragged axis emission index");
      if (mlir::failed(userNode))
        return mlir::failure();
      if (name == "intent.ragged_outer") {
        if (binding.outerNode >= 0)
          return operation->emitOpError("has multiple canonical outer domains");
        binding.outerNode = *userNode;
      } else {
        binding.memberNodes.push_back(*userNode);
        auto selector = mlir::dyn_cast<mlir::BlockArgument>(user->getOperand(1));
        mlir::Operation *owner =
            selector ? selector.getOwner()->getParentOp() : nullptr;
        mlir::Operation *outer =
            owner && selector.getArgNumber() == 0 && owner->getNumOperands() > 0
                ? owner->getOperand(0).getDefiningOp()
                : nullptr;
        if (binding.outerNode < 0 && outer &&
            outer->getName().getStringRef() == "intent.ragged_member") {
          mlir::FailureOr<int64_t> outerNode =
              target::getNodeID(*outer, "nested ragged outer emission index");
          if (mlir::failed(outerNode))
            return mlir::failure();
          binding.outerNode = *outerNode;
        }
      }
    }
    llvm::sort(binding.memberNodes);
    if (binding.outerNode < 0 || binding.memberNodes.empty())
      return operation->emitOpError(
          "has incomplete canonical ragged ownership domains");
    index.ragged.push_back(std::move(binding));
  }

  for (mlir::Operation *operation : streams) {
    mlir::FailureOr<int64_t> streamNode =
        target::getNodeID(*operation, "stream emission index");
    mlir::Operation *axis = operation->getNumOperands() > 0
                                ? operation->getOperand(0).getDefiningOp()
                                : nullptr;
    mlir::FailureOr<int64_t> axisNode =
        axis ? target::getNodeID(*axis, "ordered-axis emission index")
             : mlir::FailureOr<int64_t>(mlir::failure());
    if (mlir::failed(streamNode) || mlir::failed(axisNode) ||
        !index.axes.count(*axisNode))
      return operation->emitOpError(
          "does not resolve an ordered physical axis");
    StreamBinding binding;
    binding.operation = operation;
    binding.node = *streamNode;
    binding.axisNode = *axisNode;
    auto stopIndex =
        operation->getAttrOfType<mlir::IntegerAttr>("intent.stop_operand_index");
    if (stopIndex) {
      int64_t operand = stopIndex.getInt();
      mlir::Operation *stop =
          operand >= 0 && static_cast<unsigned>(operand) < operation->getNumOperands()
              ? operation->getOperand(operand).getDefiningOp()
              : nullptr;
      binding.stopNode =
          stop ? stop->getAttrOfType<mlir::IntegerAttr>("intent.node")
               : mlir::IntegerAttr();
      if (!binding.stopNode)
        return operation->emitOpError("has no canonical logical stream stop");
    }
    index.streams[binding.node] = std::move(binding);
  }
  for (auto relation : index.streamAxes) {
    auto stream = index.streams.find(relation.getStreamNode());
    auto axis = index.axes.find(relation.getAxisNode());
    if (stream == index.streams.end() || axis == index.axes.end() ||
        relation.getRole() != "inner_reduction")
      return relation.emitOpError(
          "does not resolve a canonical stream-axis relation");
    stream->second.innerReductionAxes.push_back(relation.getAxisNode());
  }
  for (auto &entry : index.streams) {
    llvm::sort(entry.second.innerReductionAxes);
    if (std::adjacent_find(entry.second.innerReductionAxes.begin(),
                           entry.second.innerReductionAxes.end()) !=
        entry.second.innerReductionAxes.end())
      return entry.second.emitOpError(
          "contains a duplicate inner stream axis");
  }
  return mlir::success();
}

template <typename PlanIndex>
void indexAxisRoles(PlanIndex &index) {
  auto bind = [&](llvm::StringRef role, AxisBinding axis) {
    axis.role = role.str();
    index.axesByRole[role] = axis;
    AxisBinding &canonical = index.axes[axis.getNode()];
    if (canonical.role.empty())
      canonical.role = role.str();
  };

  llvm::SmallVector<AxisBinding> axes;
  axes.reserve(index.axes.size());
  for (const auto &entry : index.axes)
    axes.push_back(entry.second);
  llvm::sort(axes, [](AxisBinding lhs, AxisBinding rhs) {
    return lhs.getNode() < rhs.getNode();
  });
  for (AxisBinding axis : axes)
    if (axis.hasRole("parallel"))
      bind("program_" + std::to_string(axis.getProgramOrder()), axis);

  llvm::SmallVector<StreamBinding> streams;
  streams.reserve(index.streams.size());
  for (const auto &entry : index.streams)
    streams.push_back(entry.second);
  llvm::sort(streams, [](StreamBinding lhs, StreamBinding rhs) {
    return lhs.getNode() < rhs.getNode();
  });
  for (auto [ordinal, stream] : llvm::enumerate(streams)) {
    auto axis = index.axes.find(stream.getAxisNode());
    if (axis != index.axes.end())
      bind("stream_" + std::to_string(ordinal), axis->second);
  }

  for (llvm::StringRef role : {"reduction", "ragged_member", "lane"}) {
    unsigned ordinal = 0;
    for (AxisBinding axis : axes)
      if (axis.hasRole(role))
        bind(role.str() + "_" + std::to_string(ordinal++), axis);
  }
}

template <typename PlanIndex>
mlir::FailureOr<AxisBinding>
contractionReductionAxis(const PlanIndex &index, mlir::Operation &lhsLoad,
                         mlir::Operation &rhsLoad,
                         mlir::Operation &consumer) {
  mlir::FailureOr<int64_t> lhsNode =
      target::getNodeID(lhsLoad, "contraction lhs transfer");
  mlir::FailureOr<int64_t> rhsNode =
      target::getNodeID(rhsLoad, "contraction rhs transfer");
  if (mlir::failed(lhsNode) || mlir::failed(rhsNode))
    return mlir::failure();
  auto lhs = index.boundaries.find(*lhsNode);
  auto rhs = index.boundaries.find(*rhsNode);
  if (lhs == index.boundaries.end() || rhs == index.boundaries.end())
    return consumer.emitOpError(
        "has no physical transfer binding for its contraction operands");

  llvm::SmallVector<AxisBinding> candidates;
  for (const auto &entry : index.axesByRole) {
    AxisBinding axis = entry.getValue();
    if (!entry.getKey().starts_with("reduction_") ||
        !llvm::is_contained(lhs->second.getDomainNodes(), axis.getNode()) ||
        !llvm::is_contained(rhs->second.getDomainNodes(), axis.getNode()))
      continue;
    candidates.push_back(axis);
  }
  if (candidates.size() != 1)
    return consumer.emitOpError(
        "does not resolve exactly one shared physical reduction axis");
  return candidates.front();
}

struct ContractionAxes {
  AxisBinding lhsResult;
  AxisBinding rhsResult;
  AxisBinding reduction;
};

template <typename PlanIndex>
mlir::FailureOr<ContractionAxes>
contractionAxes(const PlanIndex &index, mlir::Operation &lhsLoad,
                mlir::Operation &rhsLoad, mlir::Operation &consumer) {
  mlir::FailureOr<AxisBinding> reduction =
      contractionReductionAxis(index, lhsLoad, rhsLoad, consumer);
  mlir::FailureOr<int64_t> lhsNode =
      target::getNodeID(lhsLoad, "contraction lhs transfer");
  mlir::FailureOr<int64_t> rhsNode =
      target::getNodeID(rhsLoad, "contraction rhs transfer");
  if (mlir::failed(reduction) || mlir::failed(lhsNode) ||
      mlir::failed(rhsNode))
    return mlir::failure();
  auto lhs = index.boundaries.find(*lhsNode);
  auto rhs = index.boundaries.find(*rhsNode);
  if (lhs == index.boundaries.end() || rhs == index.boundaries.end())
    return consumer.emitOpError(
        "has no physical transfer binding for its contraction operands");
  auto resultAxis = [&](llvm::ArrayRef<int64_t> domains,
                        llvm::StringRef side) -> mlir::FailureOr<AxisBinding> {
    llvm::SmallVector<AxisBinding> candidates;
    for (int64_t node : domains) {
      auto found = index.axes.find(node);
      if (found == index.axes.end() || node == reduction->getNode() ||
          found->second.isScalar())
        continue;
      if (!llvm::any_of(candidates, [&](const AxisBinding &candidate) {
            return candidate.getNode() == found->second.getNode();
          }))
        candidates.push_back(found->second);
    }
    if (candidates.size() != 1)
      return consumer.emitOpError()
             << "does not resolve exactly one physical " << side
             << " result axis";
    return candidates.front();
  };
  mlir::FailureOr<AxisBinding> lhsResult =
      resultAxis(lhs->second.getDomainNodes(), "lhs");
  mlir::FailureOr<AxisBinding> rhsResult =
      resultAxis(rhs->second.getDomainNodes(), "rhs");
  if (mlir::failed(lhsResult) || mlir::failed(rhsResult))
    return mlir::failure();
  return ContractionAxes{*lhsResult, *rhsResult, *reduction};
}

template <typename PlanIndex>
PhysicalComponents indexPhysicalComponents(const PlanIndex &index) {
  PhysicalComponents result;
  result.programAxes = orderedProgramAxes(index);
  for (AxisBinding axis : result.programAxes) {
    if (axis.getReuseWorker())
      result.reusedAxes.push_back(axis);
    if (std::optional<llvm::StringRef> group = axis.getGroup())
      result.groups[*group].push_back(axis);
  }
  for (const auto &entry : index.streams)
    result.streamsByAxis[entry.second.getAxisNode()].push_back(entry.second);
  for (RaggedBinding relation : index.ragged) {
    result.raggedByAxis[relation.getOuterNode()].push_back(relation);
    for (int64_t member : relation.getMemberNodes())
      result.raggedByAxis[member].push_back(relation);
  }
  for (const auto &entry : result.streamsByAxis) {
    auto relations = result.raggedByAxis.find(entry.first);
    if (relations == result.raggedByAxis.end())
      continue;
    for (RaggedBinding relation : relations->second) {
      result.orderedRaggedAxes.insert(entry.first);
      result.orderedAxesByRelation[relation.getNode()].push_back(entry.first);
    }
  }
  for (RaggedBinding relation : index.ragged) {
    if (!result.orderedAxesByRelation.count(relation.getNode()))
      continue;
    for (int64_t member : relation.getMemberNodes()) {
      auto axis = index.axes.find(member);
      if (axis == index.axes.end() || !axis->second.hasRole("parallel"))
        continue;
      result.orderedRaggedProgramAxes.insert(member);
      result.programAxesByRelation[relation.getNode()].push_back(member);
    }
  }
  return result;
}

template <typename PlanIndex>
llvm::SmallVector<AxisBinding> orderedProgramAxes(const PlanIndex &index) {
  llvm::SmallVector<AxisBinding> result;
  for (const auto &entry : index.axes)
    if (entry.second.hasRole("parallel"))
      result.push_back(entry.second);
  llvm::sort(result, [](AxisBinding lhs, AxisBinding rhs) {
    return lhs.getProgramOrder() < rhs.getProgramOrder();
  });
  return result;
}

template <typename PlanIndex, typename AxisExpression>
std::array<std::string, 3>
projectProgramGrid(const PlanIndex &index, AxisExpression expression) {
  std::array<std::string, 3> grid = {"1", "1", "1"};
  for (AxisBinding axis : orderedProgramAxes(index)) {
    unsigned worker = static_cast<unsigned>(axis.getWorkerAxis());
    std::string extent = expression(axis);
    if (grid[worker] == "1")
      grid[worker] = std::move(extent);
    else
      grid[worker] += " * " + extent;
  }
  return grid;
}

struct ProgramIndexProjection {
  AxisBinding axis;
  std::string expression;
};

template <typename PlanIndex, typename AxisExpression,
          typename WorkerExpression>
llvm::SmallVector<ProgramIndexProjection>
projectProgramIndices(const PlanIndex &index, AxisExpression axisExtent,
                      WorkerExpression workerExpression) {
  llvm::SmallVector<ProgramIndexProjection> result;
  for (unsigned worker = 0; worker < 3; ++worker) {
    llvm::SmallVector<AxisBinding> folded;
    for (AxisBinding axis : index.components.programAxes)
      if (axis.getWorkerAxis() == worker && !axis.getGroupAttr())
        folded.push_back(axis);
    llvm::sort(folded, [](AxisBinding lhs, AxisBinding rhs) {
      return lhs.getFoldOrder() < rhs.getFoldOrder();
    });
    for (auto [position, axis] : llvm::enumerate(folded)) {
      std::string expression = workerExpression(worker);
      std::string divisor;
      for (AxisBinding later : llvm::drop_begin(folded, position + 1)) {
        if (!divisor.empty())
          divisor += " * ";
        divisor += axisExtent(later);
      }
      if (!divisor.empty())
        expression += " // (" + divisor + ")";
      if (position > 0 || !divisor.empty())
        expression = "(" + expression + ") % " + axisExtent(axis);
      result.push_back(ProgramIndexProjection{axis, std::move(expression)});
    }
  }
  llvm::sort(result, [](const ProgramIndexProjection &lhs,
                        const ProgramIndexProjection &rhs) {
    return lhs.axis.getProgramOrder() < rhs.axis.getProgramOrder();
  });
  return result;
}

template <typename PlanIndex, typename AxisExpression>
std::string projectProgramVolume(const PlanIndex &index,
                                 AxisExpression axisExtent) {
  std::string result;
  for (AxisBinding axis : orderedProgramAxes(index)) {
    if (!result.empty())
      result += " * ";
    result += axisExtent(axis);
  }
  return result.empty() ? std::string("1") : result;
}

template <typename PlanIndex, typename AxisExpression>
llvm::SmallVector<ProgramIndexProjection>
projectLinearProgramIndices(const PlanIndex &index, AxisExpression axisExtent,
                            llvm::StringRef linearExpression) {
  llvm::SmallVector<ProgramIndexProjection> result;
  llvm::SmallVector<AxisBinding> axes = orderedProgramAxes(index);
  for (auto [position, axis] : llvm::enumerate(axes)) {
    if (axis.getGroupAttr())
      continue;
    std::string expression = linearExpression.str();
    std::string divisor;
    for (AxisBinding later : llvm::drop_begin(axes, position + 1)) {
      if (!divisor.empty())
        divisor += " * ";
      divisor += axisExtent(later);
    }
    if (!divisor.empty())
      expression += " // (" + divisor + ")";
    if (position > 0)
      expression = "(" + expression + ") % " + axisExtent(axis);
    result.push_back(ProgramIndexProjection{axis, std::move(expression)});
  }
  return result;
}

template <typename PlanIndex, typename AxisExpression>
mlir::FailureOr<std::string>
projectLinearGroupIndex(const PlanIndex &index, llvm::ArrayRef<AxisBinding> group,
                        AxisExpression axisExtent,
                        llvm::StringRef linearExpression,
                        mlir::Operation &consumer) {
  llvm::SmallVector<AxisBinding> axes = orderedProgramAxes(index);
  llvm::SmallVector<unsigned> positions;
  for (AxisBinding member : group) {
    auto found = llvm::find_if(axes, [&](AxisBinding axis) {
      return axis.getNode() == member.getNode();
    });
    if (found == axes.end())
      return consumer.emitOpError("program group references an unbound axis");
    positions.push_back(static_cast<unsigned>(std::distance(axes.begin(), found)));
  }
  llvm::sort(positions);
  if (positions.empty() ||
      positions.back() - positions.front() + 1 != positions.size())
    return consumer.emitOpError("persistent program groups must be contiguous");
  std::string expression = linearExpression.str();
  std::string laterVolume;
  for (AxisBinding later : llvm::drop_begin(axes, positions.back() + 1)) {
    if (!laterVolume.empty())
      laterVolume += " * ";
    laterVolume += axisExtent(later);
  }
  if (!laterVolume.empty())
    expression += " // (" + laterVolume + ")";
  if (positions.front() > 0) {
    std::string groupVolume;
    for (unsigned position : positions) {
      if (!groupVolume.empty())
        groupVolume += " * ";
      groupVolume += axisExtent(axes[position]);
    }
    expression = "(" + expression + ") % (" + groupVolume + ")";
  }
  return expression;
}

template <typename PlanIndex, typename OperationStages>
mlir::LogicalResult indexStageOperations(const target::KernelModel &kernel,
                                         const PlanIndex &index,
                                         OperationStages &operationStages) {
  for (auto [position, stage] : llvm::enumerate(index.stages)) {
    for (int64_t node : stage.getOperations()) {
      mlir::Operation *operation = kernel.nodes.lookup(node);
      if (!operation)
        return stage.emitOpError("references an unknown operation node");
      auto &stages = operationStages[operation];
      if (!llvm::is_contained(stages, position))
        stages.push_back(position);
    }
    for (int64_t node : stage.getTerminals()) {
      mlir::Operation *terminal = kernel.nodes.lookup(node);
      if (!terminal ||
          !llvm::is_contained(operationStages.lookup(terminal), position))
        return stage.emitOpError(
            "references a terminal outside its physical operation slice");
    }
  }
  if (index.stages.empty())
    return mlir::success();

  mlir::Operation *program =
      kernel.nodes.lookup(index.program.getLoopNode());
  if (!program)
    return index.program.emitOpError("references an unknown program root");
  auto nestedInProgram = [&](mlir::Operation *operation) {
    for (mlir::Operation *parent = operation; parent;
         parent = parent->getParentOp())
      if (parent == program)
        return true;
    return false;
  };
  auto requireCovered = [&](const auto &bindings) -> mlir::LogicalResult {
    for (const auto &entry : bindings) {
      mlir::Operation *operation = kernel.nodes.lookup(entry.first);
      if (operation && nestedInProgram(operation) &&
          operationStages.lookup(operation).empty())
        return entry.second.emitOpError(
            "is inside a staged program but absent from every physical stage");
    }
    return mlir::success();
  };
  if (mlir::failed(requireCovered(index.boundaries)) ||
      mlir::failed(requireCovered(index.reductions)) ||
      mlir::failed(requireCovered(index.scans)) ||
      mlir::failed(requireCovered(index.pointwise)) ||
      mlir::failed(requireCovered(index.contracts)))
    return mlir::failure();
  return mlir::success();
}

} // namespace intent::target::emission

#endif
