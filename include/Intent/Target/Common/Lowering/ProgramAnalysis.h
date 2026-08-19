#ifndef INTENT_TARGET_COMMON_LOWERING_PROGRAMANALYSIS_H
#define INTENT_TARGET_COMMON_LOWERING_PROGRAMANALYSIS_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/ContractReplay.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "Intent/Target/Common/Analysis/LogicalBuffer.h"
#include "Intent/Target/Common/Lowering/Combiner.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <optional>
#include <array>
#include <functional>
#include <string>
#include <utility>

namespace intent::target::lowering {

inline std::optional<int64_t> staticViewStride(
    const target::ABIArgument &argument, unsigned axis) {
  auto constraints = argument.metadata.getAs<mlir::DictionaryAttr>("constraints");
  auto strides = constraints
                     ? mlir::dyn_cast_or_null<mlir::ArrayAttr>(
                           constraints.get("strides"))
                     : mlir::ArrayAttr();
  if (!strides || axis >= strides.size())
    return std::nullopt;
  auto value = mlir::dyn_cast<mlir::IntegerAttr>(strides[axis]);
  return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
}

inline bool hasNonReplayableEffect(mlir::Operation *root) {
  bool found = false;
  root->walk([&](mlir::Operation *operation) {
    llvm::StringRef name = operation->getName().getStringRef();
    found |= name == "intent.scatter_reduce" ||
             name == "intent.atomic_add" || name == "intent.atomic_cas";
  });
  return found;
}

template <typename LookupShape>
inline mlir::FailureOr<std::string>
logicalDomainExtent(mlir::Operation &domain, LookupShape lookupShape) {
  llvm::StringRef name = domain.getName().getStringRef();
  if (name == "intent.ragged_outer" || name == "intent.ragged_member") {
    mlir::Operation *relation =
        domain.getNumOperands() >= 1
            ? domain.getOperand(0).getDefiningOp()
            : nullptr;
    unsigned sourceOperand = name == "intent.ragged_outer" ? 0 : 1;
    mlir::Operation *source =
        relation && (relation->getNumOperands() == 3 ||
                     relation->getNumOperands() == 4)
            ? relation->getOperand(sourceOperand).getDefiningOp()
            : nullptr;
    if (!source)
      return domain.emitOpError(
          "has no canonical ragged source-domain extent");
    return logicalDomainExtent(*source, lookupShape);
  }
  if (domain.getNumOperands() < 2)
    return domain.emitOpError("has no canonical extent operand");
  mlir::Operation *dimension = domain.getOperand(1).getDefiningOp();
  auto constant =
      dimension
          ? dimension->getAttrOfType<mlir::IntegerAttr>("intent.value")
          : mlir::IntegerAttr();
  if (dimension &&
      dimension->getName().getStringRef() == "intent.constant" && constant &&
      constant.getInt() > 0)
    return std::to_string(constant.getInt());
  auto axis =
      dimension
          ? dimension->getAttrOfType<mlir::IntegerAttr>("intent.axis")
          : mlir::IntegerAttr();
  if (!dimension ||
      dimension->getName().getStringRef() != "intent.dim" || !axis ||
      dimension->getNumOperands() != 1)
    return domain.emitOpError("has no canonical ABI dimension source");
  mlir::FailureOr<llvm::ArrayRef<std::string>> shape =
      lookupShape(dimension->getOperand(0), domain);
  if (mlir::failed(shape) || axis.getInt() < 0 ||
      static_cast<size_t>(axis.getInt()) >= shape->size())
    return domain.emitOpError("references an invalid ABI dimension axis");
  return (*shape)[axis.getInt()];
}

inline mlir::FailureOr<std::string>
reductionRole(mlir::Operation &operation) {
  auto builtin =
      operation.getAttrOfType<mlir::StringAttr>("intent.combine_builtin");
  if (hasGenericCombiner(operation) && builtin &&
      builtin.getValue() == "argmax_lowest")
    return std::string("reduce_argmax");
  if (hasGenericCombiner(operation))
    return std::string("reduce_generic");
  auto combine = operation.getAttrOfType<mlir::StringAttr>("intent.combine");
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
  if (hasGenericCombiner(operation)) {
    auto inclusive =
        operation.getAttrOfType<mlir::BoolAttr>("intent.inclusive");
    if (inclusive && inclusive.getValue())
      return std::string("scan_generic_inclusive");
    return operation.emitOpError(
        "generic scan currently requires inclusive prefix semantics");
  }
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
    auto sourceType = operation.getNumOperands() > 0
                          ? mlir::dyn_cast<mlir::RankedTensorType>(
                                operation.getOperand(0).getType())
                          : mlir::RankedTensorType();
    bool extractFirstScalar =
        sourceType && sourceType.getRank() == 1 &&
        operation.getNumResults() == 1 &&
        !mlir::isa<mlir::RankedTensorType>(operation.getResult(0).getType()) &&
        relation->size() == 1 && relation->front().kind == "static_index" &&
        relation->front().staticValues.size() == 1 &&
        relation->front().staticValues.front() &&
        *relation->front().staticValues.front() == 0;
    if (expand)
      return std::string("expand_dims");
    if (indirect)
      return std::string("indirect_gather");
    if (extractFirstScalar)
      return std::string("extract_first_scalar");
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
           llvm::StringRef name = user->getName().getStringRef();
           return name == "intent.contract" || name == "intent.sparse_contract";
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
  llvm::StringRef getExtent() const { return operation.getExtent(); }
  int64_t getTransferNode() const {
    return operation.getTransferNodeAttr().getInt();
  }
  int64_t getSourceAxis() const {
    return operation.getSourceAxisAttr().getInt();
  }
  int64_t getDivisor() const {
    mlir::IntegerAttr value = operation.getDivisorAttr();
    return value ? value.getInt() : 1;
  }
  int64_t getOffset() const {
    mlir::IntegerAttr value = operation.getOffsetAttr();
    return value ? value.getInt() : 0;
  }
  bool isCompact() const { return getDivisor() > 1; }
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

struct RegionRangeBinding {
  AxisBinding axis;
  RangeBinding range;
};

template <typename PlanIndex>
mlir::FailureOr<RegionRangeBinding>
selectedRegionArgumentRange(const PlanIndex &index,
                            const target::KernelModel &kernel,
                            mlir::Value value, mlir::Operation &consumer) {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  mlir::FailureOr<int64_t> valueID =
      argument ? target::getValueID(value, kernel, consumer,
                                    "selected region-argument range")
               : mlir::FailureOr<int64_t>(mlir::failure());
  auto binding = mlir::succeeded(valueID)
                     ? index.regionBindings.find(*valueID)
                     : index.regionBindings.end();
  intent::plan::RegionBindingOp region =
      binding != index.regionBindings.end() ? binding->second
                                            : intent::plan::RegionBindingOp();
  auto axis = binding != index.regionBindings.end()
                  ? index.axes.find(region.getAxisNode())
                  : index.axes.end();
  const RangeBinding *range =
      axis != index.axes.end()
          ? axis->second.getRange(region.getPurpose(), region.getLevel())
          : nullptr;
  if (!argument || mlir::failed(valueID) ||
      binding == index.regionBindings.end() || axis == index.axes.end() ||
      !range)
    return consumer.emitOpError(
        "has no selected physical range for its region argument");
  return RegionRangeBinding{axis->second, *range};
}

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

struct ProgramBinding : Binding<intent::plan::LaunchOp> {
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
    llvm::ArrayRef<std::string> viewShape, mlir::Operation &operation,
    llvm::ArrayRef<int64_t> neutralizedTensorAxes = {}) {
  if (static_cast<size_t>(llvm::count_if(
          relation, [](const target::IndexTerm &term) {
            return term.kind != "new_axis";
          })) != viewShape.size())
    return operation.emitOpError(
        "physical block-extent projection does not match the external view rank");
  std::string fill;
  unsigned viewAxis = 0;
  unsigned tensorAxis = 0;
  for (const target::IndexTerm &term : relation) {
    if (term.kind == "new_axis") {
      ++tensorAxis;
      continue;
    }
    llvm::StringRef viewExtent = viewShape[viewAxis++];
    bool vectorAccess = term.kind == "full_slice";
    unsigned tensorAxes = vectorAccess ? 1 : 0;
    if ((term.kind == "region_index" || term.kind == "value_index") &&
        term.operands.size() == 1 && term.operands.front()) {
      mlir::Value indexed = operation.getOperand(*term.operands.front());
      vectorAccess = term.kind == "region_index"
                         ? !isSequentialIterator(indexed)
                         : mlir::isa<mlir::RankedTensorType>(indexed.getType());
      if (vectorAccess) {
        auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(indexed.getType());
        tensorAxes = tensor ? tensor.getRank() : 1;
      }
    }
    bool neutralized = false;
    for (unsigned axis = tensorAxis; axis < tensorAxis + tensorAxes; ++axis)
      neutralized |= llvm::is_contained(neutralizedTensorAxes,
                                        static_cast<int64_t>(axis));
    tensorAxis += tensorAxes;
    if (!vectorAccess || neutralized)
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
    const llvm::DenseMap<int64_t, std::string> &axisDimensions,
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
    std::string projectedIndex = axisIndices.lookup(owner);
    std::string extent = axisDimensions.lookup(owner);
    if (!index.axes.count(owner) || projectedIndex.empty() || extent.empty())
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
    const llvm::DenseMap<int64_t, std::string> &axisDimensions) {
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
    std::string extent = axisDimensions.lookup(owner);
    if (!index.axes.count(owner) || extent.empty())
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
    const llvm::DenseMap<int64_t, std::string> &axisDimensions) {
  std::string size;
  auto append = [&](llvm::StringRef extent) {
    size = size.empty() ? extent.str() : size + " * " + extent.str();
  };
  for (int64_t owner : binding.getOwnerNodes()) {
    std::string extent = axisDimensions.lookup(owner);
    if (!index.axes.count(owner) || extent.empty())
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
    const llvm::DenseMap<int64_t, std::string> &axisDimensions,
    mlir::Operation &operation) {
  std::string offset;
  auto append = [&](llvm::StringRef projectedIndex, llvm::StringRef extent) {
    offset = offset.empty()
                 ? projectedIndex.str()
                 : "(" + offset + ") * (" + extent.str() + ") + (" +
                       projectedIndex.str() + ")";
  };
  for (int64_t owner : binding.getOwnerNodes()) {
    std::string projected = axisIndices.lookup(owner);
    std::string extent = axisDimensions.lookup(owner);
    if (!index.axes.count(owner) || projected.empty() || extent.empty())
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
  bool getProducerReplay() const { return operation.getProducerReplay(); }
};

struct ContractionOrientation {
  bool lhsTranspose;
  bool rhsTranspose;
  bool batched;
};

inline mlir::FailureOr<ContractionOrientation>
contractionOrientation(mlir::Operation &operation) {
  auto lhsType = operation.getNumOperands() >= 2
                     ? mlir::dyn_cast<mlir::RankedTensorType>(
                           operation.getOperand(0).getType())
                     : mlir::RankedTensorType();
  auto rhsType = operation.getNumOperands() >= 2
                     ? mlir::dyn_cast<mlir::RankedTensorType>(
                           operation.getOperand(1).getType())
                     : mlir::RankedTensorType();
  auto reduce = operation.getAttrOfType<mlir::ArrayAttr>("intent.reduce");
  auto batch = operation.getAttrOfType<mlir::ArrayAttr>("intent.batch");
  auto pair = reduce && reduce.size() == 1
                  ? mlir::dyn_cast<mlir::ArrayAttr>(reduce[0])
                  : mlir::ArrayAttr();
  auto lhs = pair && pair.size() == 2
                 ? mlir::dyn_cast<mlir::IntegerAttr>(pair[0])
                 : mlir::IntegerAttr();
  auto rhs = pair && pair.size() == 2
                 ? mlir::dyn_cast<mlir::IntegerAttr>(pair[1])
                 : mlir::IntegerAttr();
  auto batchPair = batch && batch.size() == 1
                       ? mlir::dyn_cast<mlir::ArrayAttr>(batch[0])
                       : mlir::ArrayAttr();
  auto lhsBatch = batchPair && batchPair.size() == 2
                      ? mlir::dyn_cast<mlir::IntegerAttr>(batchPair[0])
                      : mlir::IntegerAttr();
  auto rhsBatch = batchPair && batchPair.size() == 2
                      ? mlir::dyn_cast<mlir::IntegerAttr>(batchPair[1])
                      : mlir::IntegerAttr();
  bool ordinary = batch && batch.empty() && lhsType && rhsType &&
                  lhsType.getRank() == 2 && rhsType.getRank() == 2;
  bool batched = batch && batch.size() == 1 && lhsBatch && rhsBatch &&
                 lhsType && rhsType && lhsType.getRank() == 3 &&
                 rhsType.getRank() == 3 && lhsBatch.getInt() == 0 &&
                 rhsBatch.getInt() == 0 && lhs && rhs &&
                 (lhs.getInt() == 1 || lhs.getInt() == 2) &&
                 (rhs.getInt() == 1 || rhs.getInt() == 2);
  if ((!ordinary && !batched) || !lhs || !rhs ||
      (lhs.getInt() != 0 && lhs.getInt() != 1 && lhs.getInt() != 2) ||
      (rhs.getInt() != 0 && rhs.getInt() != 1 && rhs.getInt() != 2))
    return operation.emitOpError(
        "has no supported contraction orientation for target emission");
  return ContractionOrientation{batched ? lhs.getInt() == 1 : lhs.getInt() == 0,
                                batched ? rhs.getInt() == 2 : rhs.getInt() == 1,
                                batched};
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
  int64_t relationNode = -1;
  std::string rangePurpose;
  int64_t rangeLevel = 0;
  std::string extent;
  std::string tile;
  llvm::SmallVector<int64_t> innerReductionAxes;

  int64_t getNode() const { return node; }
  int64_t getAxisNode() const { return axisNode; }
  mlir::IntegerAttr getStopNodeAttr() const { return stopNode; }
  int64_t getRelationNode() const { return relationNode; }
  llvm::StringRef getRangePurpose() const { return rangePurpose; }
  int64_t getRangeLevel() const { return rangeLevel; }
  llvm::StringRef getExtent() const { return extent; }
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
  llvm::SmallVector<int64_t> dependencies;
  llvm::SmallVector<int64_t> inputs;
  llvm::SmallVector<int64_t> outputs;
  llvm::SmallVector<int64_t> terminals;

  unsigned getOrdinal() const { return position; }
  int64_t getNode() const { return operation.getNode(); }
  llvm::ArrayRef<int64_t> getDependencies() const { return dependencies; }
  llvm::ArrayRef<int64_t> getInputs() const { return inputs; }
  llvm::ArrayRef<int64_t> getOutputs() const { return outputs; }
  llvm::ArrayRef<int64_t> getOperations() const {
    return operation.getOperations();
  }
  llvm::ArrayRef<int64_t> getTerminals() const { return terminals; }
  llvm::StringRef getSynchronization() const {
    return operation.getSynchronization();
  }
};

struct StageAxisBinding : Binding<intent::plan::StageAxisOp> {
  std::string extent;

  int64_t getStageNode() const { return operation.getStageNode(); }
  llvm::StringRef getRole() const { return operation.getRole(); }
  mlir::IntegerAttr getAxisNodeAttr() const {
    return operation.getAxisNodeAttr();
  }
  mlir::IntegerAttr getSourceValueAttr() const {
    return operation.getSourceValueAttr();
  }
  mlir::IntegerAttr getTensorAxisAttr() const {
    return operation.getTensorAxisAttr();
  }
  llvm::StringRef getExtent() const { return extent; }
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
  llvm::ArrayRef<int64_t> getValidityTensorAxes() const {
    return operation.getValidityTensorAxes();
  }
  llvm::ArrayRef<int64_t> getValidityDomainNodes() const {
    return operation.getValidityDomainNodes();
  }
  llvm::StringRef getAccess() const { return access; }
  llvm::StringRef getLoadFill() const { return operation.getFill(); }
  llvm::StringRef getPadding() const { return operation.getFill(); }
  llvm::StringRef getTensorIndexing() const {
    return operation.getTensorIndexing();
  }
  llvm::StringRef getMaterialization() const {
    return operation.getMaterialization();
  }
  bool hasStructuredTensorIndex() const {
    return operation.getTensorIndexing() == "structured";
  }
  bool hasCompactTensorIndex() const {
    return operation.getTensorIndexing() == "compact";
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
  bool getDefer() const {
    return defer || getMaterialization() == "deferred_to_contract";
  }
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
bool isPlannedStageNode(const PlanIndex &index, mlir::Operation *operation) {
  if (!operation)
    return false;
  auto node = operation->getAttrOfType<mlir::IntegerAttr>("intent.node");
  return node && llvm::any_of(index.stages, [&](auto stage) {
           return stage.getNode() == node.getInt();
         });
}

template <typename PlanIndex>
bool isStagedContraction(const PlanIndex &index, mlir::Operation *operation) {
  return operation &&
         operation->getName().getStringRef() == "intent.contract" &&
         isPlannedStageNode(index, operation);
}

inline mlir::FailureOr<llvm::SmallVector<int64_t>>
exactContractionReductionArguments(const target::KernelModel &kernel,
                                   mlir::Operation &consumer) {
  auto reduce = consumer.getAttrOfType<mlir::ArrayAttr>("intent.reduce");
  auto pair = reduce && reduce.size() == 1
                  ? mlir::dyn_cast<mlir::ArrayAttr>(reduce[0])
                  : mlir::ArrayAttr();
  auto lhsAxis = pair && pair.size() == 2
                     ? mlir::dyn_cast<mlir::IntegerAttr>(pair[0])
                     : mlir::IntegerAttr();
  auto rhsAxis = pair && pair.size() == 2
                     ? mlir::dyn_cast<mlir::IntegerAttr>(pair[1])
                     : mlir::IntegerAttr();
  if (!lhsAxis || !rhsAxis || consumer.getNumOperands() != 2)
    return consumer.emitOpError(
        "does not declare one exact contraction reduction pair");
  auto operandArgument = [&](mlir::Value operand,
                             int64_t tensorAxis) -> std::optional<int64_t> {
    mlir::FailureOr<llvm::SmallVector<std::string>> shape =
        target::getLogicalShape(operand, kernel, consumer,
                                "contraction reduction-axis projection");
    if (mlir::succeeded(shape) && tensorAxis >= 0 &&
        static_cast<size_t>(tensorAxis) < shape->size()) {
      llvm::StringRef label = (*shape)[tensorAxis];
      if (label.consume_front("?region_")) {
        llvm::StringRef argumentText = label.split('_').first;
        int64_t argument = -1;
        if (!argumentText.getAsInteger(10, argument))
          return argument;
      }
    }
    return std::nullopt;
  };
  std::optional<int64_t> lhs =
      operandArgument(consumer.getOperand(0), lhsAxis.getInt());
  std::optional<int64_t> rhs =
      operandArgument(consumer.getOperand(1), rhsAxis.getInt());
  llvm::SmallVector<int64_t> arguments;
  if (lhs)
    arguments.push_back(*lhs);
  if (rhs && (!lhs || *rhs != *lhs))
    arguments.push_back(*rhs);
  if (arguments.empty())
    return consumer.emitOpError(
        "does not preserve a region identity for its reduction pair");
  return arguments;
}

template <typename PlanIndex>
mlir::FailureOr<AxisBinding>
exactContractionReductionAxis(const PlanIndex &index,
                              const target::KernelModel &kernel,
                              mlir::Operation &consumer) {
  mlir::FailureOr<llvm::SmallVector<int64_t>> arguments =
      exactContractionReductionArguments(kernel, consumer);
  if (mlir::failed(arguments))
    return mlir::failure();
  std::optional<int64_t> axisNode;
  for (int64_t argument : *arguments) {
    auto binding = index.regionBindings.find(argument);
    std::optional<int64_t> candidate;
    if (binding != index.regionBindings.end()) {
      intent::plan::RegionBindingOp region = binding->second;
      candidate = static_cast<int64_t>(region.getAxisNode());
    } else {
      mlir::Value value = kernel.values.lookup(argument);
      mlir::Operation *definition = value ? value.getDefiningOp() : nullptr;
      mlir::FailureOr<int64_t> node =
          definition
              ? target::getNodeID(*definition,
                                  "contraction reduction-region projection")
              : mlir::FailureOr<int64_t>(mlir::failure());
      if (mlir::succeeded(node) && index.axes.count(*node))
        candidate = *node;
    }
    if (!candidate)
      return consumer.emitOpError(
          "has no selected range for a reduction-region identity");
    if (axisNode && *axisNode != *candidate)
      return consumer.emitOpError(
          "maps its reduction pair to different physical axes");
    axisNode = *candidate;
  }
  auto axis = axisNode ? index.axes.find(*axisNode) : index.axes.end();
  if (!axisNode || axis == index.axes.end() ||
      !axis->second.hasRole("reduction"))
    return consumer.emitOpError(
        "does not resolve its exact reduction pair to one physical axis");
  return axis->second;
}

template <typename PlanIndex>
mlir::FailureOr<AxisBinding>
contractionReductionAxis(const PlanIndex &index,
                         llvm::ArrayRef<mlir::Operation *> lhsTransfers,
                         llvm::ArrayRef<mlir::Operation *> rhsTransfers,
                         mlir::Operation &consumer);

struct DeferredContractReplay {
  target::ContractOperandReplay lhs;
  target::ContractOperandReplay rhs;
  llvm::SmallVector<mlir::Operation *> lhsReplayProducers;
  llvm::SmallVector<mlir::Operation *> rhsReplayProducers;
};

template <typename PlanIndex>
mlir::LogicalResult indexDeferredContractReplays(
    const target::KernelModel &kernel, const PlanIndex &index,
    llvm::DenseMap<mlir::Operation *, DeferredContractReplay> &replays,
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
        &producerOwners) {
  llvm::DenseSet<mlir::Operation *> globallyDeferred;
  for (const auto &entry : index.contracts) {
    if (!entry.second.getProducerReplay())
      continue;
    mlir::Operation *contract = kernel.nodes.lookup(entry.first);
    if (!contract || contract->getName().getStringRef() != "intent.contract" ||
        contract->getNumOperands() != 2 ||
        isPlannedStageNode(index, contract))
      return entry.second.emitOpError(
          "producer replay does not bind one unstaged canonical contraction");
    std::optional<target::ContractOperandReplay> lhs =
        target::analyzeContractOperandReplay(contract->getOperand(0), *contract);
    std::optional<target::ContractOperandReplay> rhs =
        target::analyzeContractOperandReplay(contract->getOperand(1), *contract);
    if (!lhs || !rhs)
      return entry.second.emitOpError(
          "selected producer replay has no canonical producer chain");
    bool anyDeferred = false;
    bool allDeferred = true;
    for (mlir::Operation *transfer :
         llvm::concat<mlir::Operation *>(lhs->transfers, rhs->transfers)) {
      mlir::FailureOr<int64_t> node =
          target::getNodeID(*transfer, "deferred contract transfer");
      auto binding = mlir::succeeded(node) ? index.boundaries.find(*node)
                                           : index.boundaries.end();
      if (mlir::failed(node) || binding == index.boundaries.end())
        return transfer->emitOpError(
            "lacks a contract-transfer materialization decision");
      bool deferred =
          binding->second.getMaterialization() == "deferred_to_contract";
      anyDeferred |= deferred;
      allDeferred &= deferred;
    }
    if (!anyDeferred || !allDeferred)
      return entry.second.emitOpError(
          "producer replay requires every source transfer to be deferred");
    if (entry.second.getLhsSpace() != "shared" ||
        entry.second.getRhsSpace() != "shared")
      return entry.second.emitOpError(
          "producer replay requires shared contraction operand bindings");
    mlir::FailureOr<AxisBinding> reduction =
        exactContractionReductionAxis(index, kernel, *contract);
    if (mlir::failed(reduction))
      return mlir::failure();
    mlir::FailureOr<llvm::SmallVector<int64_t>> reductionArgumentIDs =
        exactContractionReductionArguments(kernel, *contract);
    if (mlir::failed(reductionArgumentIDs))
      return mlir::failure();
    llvm::DenseSet<mlir::Value> reductionArguments;
    for (int64_t argumentID : *reductionArgumentIDs) {
      mlir::Value argument = kernel.values.lookup(argumentID);
      if (argument)
        reductionArguments.insert(argument);
    }
    llvm::DenseMap<mlir::Value, bool> reductionDependencies;
    llvm::DenseSet<mlir::Value> activeDependencies;
    std::function<bool(mlir::Value)> dependsOnReduction =
        [&](mlir::Value value) -> bool {
      if (reductionArguments.contains(value))
        return true;
      auto cached = reductionDependencies.find(value);
      if (cached != reductionDependencies.end())
        return cached->second;
      if (!activeDependencies.insert(value).second)
        return false;
      mlir::Operation *definition = value.getDefiningOp();
      bool depends = definition && llvm::any_of(
                                       definition->getOperands(),
                                       [&](mlir::Value input) {
                                         return dependsOnReduction(input);
                                       });
      activeDependencies.erase(value);
      reductionDependencies[value] = depends;
      return depends;
    };
    llvm::DenseSet<mlir::Operation *> deferred;
    auto collectDeferred = [&](const target::ContractOperandReplay &operand) {
      deferred.insert(operand.loadDependentProducers.begin(),
                      operand.loadDependentProducers.end());
      for (mlir::Operation *producer : operand.exclusiveProducers) {
        if (dependsOnReduction(producer->getResult(0)))
          deferred.insert(producer);
      }
    };
    collectDeferred(*lhs);
    collectDeferred(*rhs);
    globallyDeferred.insert(deferred.begin(), deferred.end());
    auto replayProducers = [&](const target::ContractOperandReplay &operand) {
      llvm::SmallVector<mlir::Operation *> producers;
      for (mlir::Operation *producer : operand.producers)
        if (deferred.contains(producer))
          producers.push_back(producer);
      return producers;
    };
    llvm::SmallVector<mlir::Operation *> lhsProducers = replayProducers(*lhs);
    llvm::SmallVector<mlir::Operation *> rhsProducers = replayProducers(*rhs);
    replays[contract] = DeferredContractReplay{
        std::move(*lhs), std::move(*rhs), std::move(lhsProducers),
        std::move(rhsProducers)};
  }
  for (const auto &entry : replays) {
    auto bindOwners = [&](llvm::ArrayRef<mlir::Operation *> producers) {
      for (mlir::Operation *producer : producers)
        if (globallyDeferred.contains(producer)) {
          auto &owners = producerOwners[producer];
          if (!llvm::is_contained(owners, entry.first))
            owners.push_back(entry.first);
        }
    };
    bindOwners(entry.second.lhsReplayProducers);
    bindOwners(entry.second.rhsReplayProducers);
  }
  return mlir::success();
}

template <typename PlanIndex>
bool isEnclosingStreamReductionAxis(const PlanIndex &index,
                                    mlir::Operation &operation,
                                    int64_t axisNode) {
  for (mlir::Operation *parent = operation.getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->getName().getStringRef() != "intent.state_stream")
      continue;
    auto node = parent->getAttrOfType<mlir::IntegerAttr>("intent.node");
    if (!node)
      continue;
    auto stream = index.streams.find(node.getInt());
    if (stream != index.streams.end() &&
        llvm::is_contained(stream->second.getInnerReductionAxes(), axisNode))
      return true;
  }
  return false;
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
  for (const auto &entry : index.regionBindings) {
    intent::plan::RegionBindingOp binding = entry.second;
    mlir::Value value = kernel.values.lookup(entry.first);
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    auto axis = index.axes.find(binding.getAxisNode());
    const RangeBinding *range =
        axis == index.axes.end()
            ? nullptr
            : axis->second.getRange(binding.getPurpose(), binding.getLevel());
    mlir::Operation *owner =
        argument ? argument.getOwner()->getParentOp() : nullptr;
    llvm::StringRef expected =
        owner && owner->getName().getStringRef() == "intent.parallel"
            ? llvm::StringRef("ownership")
        : owner && owner->getName().getStringRef() == "intent.state_stream"
            ? llvm::StringRef("traversal")
            : llvm::StringRef();
    mlir::FailureOr<target::ScalarIndexSource> source =
        argument && owner
            ? target::traceScalarIndexSource(argument, *owner)
            : mlir::FailureOr<target::ScalarIndexSource>(mlir::failure());
    mlir::FailureOr<int64_t> sourceNode =
        mlir::succeeded(source) && source->domain
            ? target::getNodeID(*source->domain,
                                "region-argument source-axis verification")
            : mlir::FailureOr<int64_t>(mlir::failure());
    if (!argument || !owner || expected.empty() ||
        binding.getPurpose() != expected || !range || mlir::failed(sourceNode) ||
        *sourceNode != static_cast<int64_t>(binding.getAxisNode()))
      return binding.emitOpError(
          "does not bind the exact region argument source to its selected range");
  }
  llvm::SmallVector<int64_t> relationNodes;
  for (const auto &entry : kernel.raggedRelations)
    relationNodes.push_back(entry.first);
  llvm::sort(relationNodes);
  for (int64_t relationNode : relationNodes) {
    const target::RaggedStructure &relation =
        kernel.raggedRelations.lookup(relationNode);
    RaggedBinding binding;
    binding.operation = relation.operation;
    binding.node = relation.node;
    binding.outerNode = relation.outerNode;
    binding.memberNodes = relation.memberNodes;
    index.ragged.push_back(std::move(binding));
  }

  llvm::SmallVector<int64_t> streamNodes;
  for (const auto &entry : kernel.stateStreams)
    streamNodes.push_back(entry.first);
  llvm::sort(streamNodes);
  for (int64_t streamNode : streamNodes) {
    const target::StateStreamStructure &stream =
        kernel.stateStreams.lookup(streamNode);
    auto selected = index.streamBindings.find(streamNode);
    if (selected == index.streamBindings.end())
      return stream.operation->emitOpError(
          "has no selected physical stream binding");
    intent::plan::StreamBindingOp physical = selected->second;
    if (static_cast<int64_t>(physical.getAxisNode()) != stream.axisNode ||
        !index.axes.count(physical.getAxisNode()))
      return physical.emitOpError(
          "does not bind the canonical state-stream axis");
    const AxisBinding &axis = index.axes.lookup(physical.getAxisNode());
    const RangeBinding *range =
        axis.getRange(physical.getPurpose(), physical.getLevel());
    if (!range)
      return physical.emitOpError(
          "does not select a physical range for the state-stream axis");
    StreamBinding binding;
    binding.operation = stream.operation;
    binding.node = stream.node;
    binding.axisNode = physical.getAxisNode();
    binding.rangePurpose = physical.getPurpose().str();
    binding.rangeLevel = physical.getLevel();
    binding.extent = range->getExtent().str();
    if (stream.stopNode >= 0)
      binding.stopNode = mlir::IntegerAttr::get(
          mlir::IntegerType::get(stream.operation->getContext(), 64),
          stream.stopNode);
    if (physical.getRelationNodeAttr()) {
      binding.relationNode = physical.getRelationNodeAttr().getInt();
      auto relation = kernel.raggedRelations.find(binding.relationNode);
      if (relation == kernel.raggedRelations.end() ||
          (relation->second.outerNode != binding.axisNode &&
           !llvm::is_contained(relation->second.memberNodes,
                               binding.axisNode)))
        return physical.emitOpError(
            "does not bind a canonical ragged relation for its stream axis");
    }
    index.streams[binding.node] = std::move(binding);
  }
  for (auto relation : index.streamAxes) {
    auto stream = index.streams.find(relation.getStreamNode());
    auto axis = index.axes.find(relation.getAxisNode());
    if (stream == index.streams.end() || axis == index.axes.end())
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
contractionReductionAxis(const PlanIndex &index,
                         llvm::ArrayRef<mlir::Operation *> lhsTransfers,
                         llvm::ArrayRef<mlir::Operation *> rhsTransfers,
                         mlir::Operation &consumer) {
  auto contains = [&](llvm::ArrayRef<mlir::Operation *> transfers,
                      int64_t axisNode) {
    return llvm::any_of(transfers, [&](mlir::Operation *transfer) {
      mlir::FailureOr<int64_t> node =
          target::getNodeID(*transfer, "contraction transfer");
      auto binding = mlir::succeeded(node) ? index.boundaries.find(*node)
                                           : index.boundaries.end();
      return mlir::succeeded(node) && binding != index.boundaries.end() &&
             llvm::is_contained(binding->second.getDomainNodes(), axisNode);
    });
  };

  llvm::SmallVector<AxisBinding> candidates;
  for (const auto &entry : index.axesByRole) {
    AxisBinding axis = entry.getValue();
    if (!entry.getKey().starts_with("reduction_") ||
        !contains(lhsTransfers, axis.getNode()) ||
        !contains(rhsTransfers, axis.getNode()))
      continue;
    candidates.push_back(axis);
  }
  if (candidates.size() != 1)
    return consumer.emitOpError(
        "does not resolve exactly one shared physical reduction axis");
  return candidates.front();
}

template <typename PlanIndex>
mlir::FailureOr<AxisBinding>
contractionReductionAxis(const PlanIndex &index, mlir::Operation &lhsLoad,
                         mlir::Operation &rhsLoad,
                         mlir::Operation &consumer) {
  mlir::Operation *lhs[] = {&lhsLoad};
  mlir::Operation *rhs[] = {&rhsLoad};
  return contractionReductionAxis(index, lhs, rhs, consumer);
}

struct ContractionAxes {
  AxisBinding lhsResult;
  AxisBinding rhsResult;
  AxisBinding reduction;
};

template <typename PlanIndex>
mlir::FailureOr<ContractionAxes>
contractionAxes(const PlanIndex &index,
                llvm::ArrayRef<mlir::Operation *> lhsTransfers,
                llvm::ArrayRef<mlir::Operation *> rhsTransfers,
                mlir::Operation &consumer) {
  mlir::FailureOr<AxisBinding> reduction =
      contractionReductionAxis(index, lhsTransfers, rhsTransfers, consumer);
  if (mlir::failed(reduction))
    return mlir::failure();
  auto resultAxis = [&](llvm::ArrayRef<mlir::Operation *> transfers,
                        llvm::StringRef side) -> mlir::FailureOr<AxisBinding> {
    llvm::SmallVector<AxisBinding> candidates;
    for (mlir::Operation *transfer : transfers) {
      mlir::FailureOr<int64_t> node =
          target::getNodeID(*transfer, "contraction transfer");
      auto binding = mlir::succeeded(node) ? index.boundaries.find(*node)
                                           : index.boundaries.end();
      if (mlir::failed(node) || binding == index.boundaries.end())
        return mlir::failure();
      for (int64_t domainNode : binding->second.getDomainNodes()) {
        auto found = index.axes.find(domainNode);
        if (found == index.axes.end() || domainNode == reduction->getNode() ||
            found->second.isScalar())
          continue;
        if (!llvm::any_of(candidates, [&](const AxisBinding &candidate) {
              return candidate.getNode() == found->second.getNode();
            }))
          candidates.push_back(found->second);
      }
    }
    if (candidates.size() != 1)
      return consumer.emitOpError()
             << "does not resolve exactly one physical " << side
             << " result axis";
    return candidates.front();
  };
  mlir::FailureOr<AxisBinding> lhsResult =
      resultAxis(lhsTransfers, "lhs");
  mlir::FailureOr<AxisBinding> rhsResult =
      resultAxis(rhsTransfers, "rhs");
  if (mlir::failed(lhsResult) || mlir::failed(rhsResult))
    return mlir::failure();
  return ContractionAxes{*lhsResult, *rhsResult, *reduction};
}

template <typename PlanIndex>
mlir::FailureOr<ContractionAxes>
contractionAxes(const PlanIndex &index, mlir::Operation &lhsLoad,
                mlir::Operation &rhsLoad, mlir::Operation &consumer) {
  mlir::Operation *lhs[] = {&lhsLoad};
  mlir::Operation *rhs[] = {&rhsLoad};
  return contractionAxes(index, lhs, rhs, consumer);
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
    for (StreamBinding stream : entry.second) {
      if (stream.getRelationNode() < 0)
        continue;
      result.orderedRaggedAxes.insert(entry.first);
      result.orderedAxesByRelation[stream.getRelationNode()].push_back(entry.first);
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
                                         PlanIndex &index,
                                         OperationStages &operationStages) {
  for (auto &stageAxes : index.stageAxes) {
    for (auto &roleAndAxis : stageAxes.second) {
      StageAxisBinding &binding = roleAndAxis.getValue();
      if (binding.getSourceValueAttr()) {
        mlir::Value value =
            kernel.values.lookup(binding.getSourceValueAttr().getInt());
        mlir::FailureOr<llvm::SmallVector<std::string>> shape =
            value ? target::getLogicalShape(value, kernel,
                                            *binding.operation.getOperation(),
                                            "physical stage-axis binding")
                  : mlir::FailureOr<llvm::SmallVector<std::string>>(
                        mlir::failure());
        int64_t axis = binding.getTensorAxisAttr().getInt();
        if (!value || mlir::failed(shape) || axis < 0 ||
            static_cast<size_t>(axis) >= shape->size())
          return binding.emitOpError(
              "does not resolve its canonical source tensor dimension");
        binding.extent = (*shape)[axis];
        continue;
      }
      auto physical = index.axes.find(binding.getAxisNodeAttr().getInt());
      const RangeBinding *range =
          physical == index.axes.end()
              ? nullptr
              : physical->second.getRange("ownership", 0);
      if (!range)
        return binding.emitOpError(
            "does not resolve its canonical domain-axis extent");
      binding.extent = range->getExtent().str();
    }
  }
  llvm::DenseMap<int64_t, unsigned> stagePositions;
  for (auto [position, stage] : llvm::enumerate(index.stages))
    stagePositions[stage.getNode()] = position;
  for (auto [position, stage] : llvm::enumerate(index.stages)) {
    if (stage.getSynchronization() != "same_stream")
      return stage.emitOpError(
          "target emitter requires ordered same-stream physical stages");
    for (int64_t node : stage.getOperations()) {
      mlir::Operation *operation = kernel.nodes.lookup(node);
      if (!operation)
        return stage.emitOpError("references an unknown operation node");
      auto &stages = operationStages[operation];
      if (!llvm::is_contained(stages, position))
        stages.push_back(position);
    }
  }
  if (index.stages.empty())
    return mlir::success();

  auto appendUnique = [](auto &values, int64_t value) {
    if (!llvm::is_contained(values, value))
      values.push_back(value);
  };
  llvm::DenseMap<mlir::Operation *, unsigned> terminalOwners;
  for (unsigned position = 0; position < index.stages.size(); ++position) {
    StageBinding &stage = index.stages[position];
    for (int64_t node : stage.getOperations()) {
      mlir::Operation *operation = kernel.nodes.lookup(node);
      if (!operation)
        return stage.emitOpError("references an unknown operation node");
      if (operation->getNumRegions() == 0 &&
          mlir::hasEffect<mlir::MemoryEffects::Write>(operation)) {
        auto owner = terminalOwners.try_emplace(operation, position);
        if (!owner.second && owner.first->second != position)
          return operation->emitOpError(
              "is assigned as an effectful terminal to multiple physical stages");
        appendUnique(stage.terminals, node);
      }

      for (mlir::Value operand : operation->getOperands()) {
        mlir::Operation *definition = operand.getDefiningOp();
        if (!definition)
          continue;
        llvm::ArrayRef<unsigned> definitionStages =
            operationStages.lookup(definition);
        if (definitionStages.empty() ||
            llvm::is_contained(definitionStages, position))
          continue;
        llvm::SmallVector<unsigned> producers;
        for (unsigned producer : definitionStages)
          if (producer < position)
            producers.push_back(producer);
        if (producers.size() != 1)
          return operation->emitOpError(
              "does not resolve one earlier physical stage for a cross-stage value");
        mlir::FailureOr<int64_t> valueID = target::getValueID(
            operand, kernel, *operation, "derived physical stage boundary");
        if (mlir::failed(valueID))
          return mlir::failure();
        StageBinding &producer = index.stages[producers.front()];
        appendUnique(stage.inputs, *valueID);
        appendUnique(producer.outputs, *valueID);
        appendUnique(stage.dependencies, producer.getNode());
      }
    }
  }
  for (StageBinding &stage : index.stages) {
    llvm::sort(stage.dependencies);
    llvm::sort(stage.inputs);
    llvm::sort(stage.outputs);
    llvm::sort(stage.terminals);
    if (stage.outputs.empty() == stage.terminals.empty())
      return stage.emitOpError(
          "derived stage boundary must have either intermediate outputs or effectful terminals");
    for (int64_t output : stage.outputs) {
      mlir::Value value = kernel.values.lookup(output);
      if (!value || !mlir::isa<mlir::RankedTensorType>(value.getType()))
        return stage.emitOpError(
            "derives a non-tensor physical stage intermediate");
    }
    for (int64_t dependency : stage.dependencies) {
      auto found = stagePositions.find(dependency);
      if (found == stagePositions.end() || found->second >= stage.position)
        return stage.emitOpError(
            "derives a non-predecessor physical stage dependency");
    }
  }

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

} // namespace intent::target::lowering

#endif
