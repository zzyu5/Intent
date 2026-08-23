#include "Support/Decisions.h"

#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/GPU/Realization/PhysicalProgram.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <array>
#include <numeric>
#include <tuple>

using namespace mlir;

namespace intent::gpu::realization {
namespace {

IntegerAttr i64(OpBuilder &builder, int64_t value) {
  return builder.getI64IntegerAttr(value);
}

StringAttr string(OpBuilder &builder, StringRef value) {
  return builder.getStringAttr(value);
}

FailureOr<int64_t> node(Operation &operation, StringRef purpose) {
  return target::getNodeID(operation, purpose);
}

FailureOr<int64_t> valueID(Value value, const target::KernelModel &kernel,
                           Operation &consumer, StringRef purpose) {
  return target::getValueID(value, kernel, consumer, purpose);
}

FailureOr<ArrayRef<Operation *>> ownedDomains(
    Operation &parallel, const target::KernelFacts &facts) {
  auto found = facts.parallelDomains.find(&parallel);
  if (found == facts.parallelDomains.end() || found->second.empty()) {
    parallel.emitOpError("has no source domains for physical role assignment");
    return failure();
  }
  return ArrayRef<Operation *>(found->second);
}

FailureOr<Operation *> programRoot(const target::KernelFacts &facts) {
  SmallVector<Operation *> roots;
  for (const target::RegionNode &region : facts.kernel.regions.nodes) {
    if (::intent::target::semanticOperationName(*region.operation) !=
            "intent.parallel" ||
        region.parent)
      continue;
    roots.push_back(region.operation);
  }
  if (roots.size() > 1) {
    facts.kernel.entry.emitOpError(
        "one GPU kernel function cannot contain multiple outer program regions; "
        "multiple launches require separate kernel functions");
    return failure();
  }
  return roots.empty() ? nullptr : roots.front();
}

Operation *nearestParallel(Operation *operation) {
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp())
    if (::intent::target::semanticOperationName(*parent) == "intent.parallel")
      return parent;
  return nullptr;
}

FailureOr<std::string> sourceDimensionSymbol(Operation &domain,
                                             const target::KernelFacts &facts) {
  auto staticExtent = facts.staticDomainExtents.find(&domain);
  if (staticExtent != facts.staticDomainExtents.end())
    return std::to_string(staticExtent->second);
  auto staticBounds = facts.staticDomainBounds.find(&domain);
  if (staticBounds != facts.staticDomainBounds.end())
    return std::to_string(staticBounds->second.second -
                          staticBounds->second.first);
  Value source = facts.domainSources.lookup(&domain);
  int64_t axis = facts.domainSourceAxes.lookup(&domain);
  auto argument = llvm::find_if(facts.kernel.abi.arguments,
                                [&](const target::ABIArgument &candidate) {
                                  return candidate.value == source;
                                });
  if (argument == facts.kernel.abi.arguments.end()) {
    domain.emitOpError("does not refer to a canonical ABI argument");
    return failure();
  }
  auto shape = argument->metadata.getAs<ArrayAttr>("shape");
  auto symbol = shape && axis >= 0 && static_cast<size_t>(axis) < shape.size()
                    ? dyn_cast<StringAttr>(shape[axis])
                    : StringAttr();
  if (!symbol || symbol.getValue().empty()) {
    domain.emitOpError("does not expose a symbolic specialization dimension");
    return failure();
  }
  return symbol.getValue().str();
}

FailureOr<std::string>
sourceViewDimensionSymbol(Operation &transfer, unsigned sourceAxis,
                          const target::KernelFacts &facts) {
  if (transfer.getNumOperands() == 0)
    return transfer.emitOpError("has no source view for its access range");
  Value source = transfer.getOperand(0);
  auto argument = llvm::find_if(facts.kernel.abi.arguments,
                                [&](const target::ABIArgument &candidate) {
                                  return candidate.value == source;
                                });
  if (argument == facts.kernel.abi.arguments.end())
    return transfer.emitOpError(
        "does not refer to a canonical source-view ABI argument");
  auto shape = argument->metadata.getAs<ArrayAttr>("shape");
  auto symbol = shape && sourceAxis < shape.size()
                    ? dyn_cast<StringAttr>(shape[sourceAxis])
                    : StringAttr();
  if (!symbol || symbol.getValue().empty())
    return transfer.emitOpError(
        "does not expose its access-range source dimension");
  return symbol.getValue().str();
}

void appendUnique(SmallVectorImpl<std::string> &values, StringRef value) {
  if (!value.empty() && !llvm::is_contained(values, value))
    values.push_back(value.str());
}

void appendRole(SmallVectorImpl<std::string> &roles, StringRef role) {
  if (!llvm::is_contained(roles, role))
    roles.push_back(role.str());
}

bool hasRole(ArrayRef<std::string> roles, StringRef role) {
  return llvm::is_contained(roles, role);
}

struct AxisChoice {
  struct RangeChoice {
    std::string purpose;
    int64_t level = 0;
    std::string tile;
  };

  Operation *domain = nullptr;
  SmallVector<Operation *> parallels;
  SmallVector<std::string> roles;
  bool tiled = false;
  std::optional<int64_t> fixedOwnershipExtent;
  Operation *countPartition = nullptr;
  bool packedLane = false;
  SmallVector<RangeChoice> ranges;
  std::optional<int64_t> programOrder;
  std::optional<int64_t> worker;
  std::optional<int64_t> fold;
  bool reuse = false;
  std::string group;
};

bool canPackScalarParallel(Operation *parallel, Operation *domain,
                           const target::KernelFacts &facts) {
  auto domains = facts.parallelDomains.find(parallel);
  if (domains == facts.parallelDomains.end() || domains->second.empty() ||
      domains->second.back() != domain ||
      parallel->getNumRegions() != 1 ||
      !llvm::hasSingleElement(parallel->getRegion(0)))
    return false;

  Block &body = parallel->getRegion(0).front();
  for (Operation &operation : body) {
    StringRef name = ::intent::target::semanticOperationName(operation);
    bool scalarPointwise =
        name == "intent.constant" || name == "intent.dim" ||
        name == "intent.assume_in_bounds" || name == "intent.view_load" ||
        name == "intent.view_store" || name == "intent.gather" ||
        name == "intent.indices" || name == "intent.broadcast" ||
        name == "intent.make_record" || name == "intent.extract" ||
        name == "intent.unary" || name == "intent.binary" ||
        name == "intent.compare" || name == "intent.select" ||
        name == "intent.cast" || name == "intent.bitcast" ||
        name == "intent.random" ||
        name == "intent.yield";
    if (operation.getNumRegions() != 0 || !scalarPointwise)
      return false;
  }
  return true;
}

bool canDistributePointwiseLane(Operation *domain,
                                const target::KernelFacts &facts) {
  if (!domain || domain->getNumResults() != 1 || domain->getResult(0).use_empty())
    return false;
  Operation *owner = nullptr;
  for (Operation *user : domain->getResult(0).getUsers()) {
    StringRef name = ::intent::target::semanticOperationName(*user);
    if (name != "intent.view_load" && name != "intent.view_store" &&
        name != "intent.gather")
      return false;
    Operation *parallel = nearestParallel(user);
    if (!parallel || (owner && owner != parallel))
      return false;
    owner = parallel;
  }
  Operation *ownerSource =
      owner && owner->getNumOperands() == 1
          ? owner->getOperand(0).getDefiningOp()
          : nullptr;
  if (!ownerSource || !facts.partitionDomains.contains(ownerSource))
    return false;
  return llvm::none_of(facts.vectorDomains, [&](Operation *other) {
    if (other == domain || other->getNumResults() != 1)
      return false;
    return llvm::any_of(other->getResult(0).getUsers(), [&](Operation *user) {
      return nearestParallel(user) == owner;
    });
  });
}

void addRange(AxisChoice &choice, StringRef purpose, int64_t level,
              std::string tile) {
  choice.ranges.push_back(
      AxisChoice::RangeChoice{purpose.str(), level, std::move(tile)});
}

const AxisChoice::RangeChoice *findRange(const AxisChoice &choice,
                                         StringRef purpose,
                                         int64_t level = 0) {
  auto found = llvm::find_if(choice.ranges, [&](const auto &range) {
    return range.purpose == purpose && range.level == level;
  });
  return found == choice.ranges.end() ? nullptr : &*found;
}

struct AxisAssignments {
  SmallVector<AxisChoice, 0> axes;
  bool persistent = false;
};

SmallVector<Operation *>
independentLanes(const AxisChoice &choice,
                 const target::KernelFacts &facts) {
  if (choice.parallels.empty())
    return {};
  auto isOwnedByChoice = [&](Operation *operation) {
    return llvm::is_contained(choice.parallels, nearestParallel(operation));
  };
  auto usable = [&](Operation *domain) {
    return domain != choice.domain && facts.vectorDomains.contains(domain) &&
           !facts.orderedDomains.contains(domain) &&
           !facts.contractionDomains.contains(domain);
  };
  SmallVector<Operation *> lanes;
  auto collect = [&](Operation *domain) {
    if (usable(domain) && !llvm::is_contained(lanes, domain))
      lanes.push_back(domain);
  };
  for (const auto &entry : facts.boundaryDomains) {
    if (!isOwnedByChoice(entry.first))
      continue;
    for (Operation *domain : entry.second)
      collect(domain);
  }
  for (const auto &entry : facts.valueAxes) {
    Operation *definition = entry.first.getDefiningOp();
    if (!definition || !isOwnedByChoice(definition))
      continue;
    for (const target::LogicalAxis &axis : entry.second)
      collect(axis.domain);
  }
  return lanes;
}

unsigned independentLaneCount(const AxisChoice &choice,
                              const target::KernelFacts &facts) {
  return independentLanes(choice, facts).size();
}

bool hasPackableIndependentLanes(const AxisChoice &choice,
                                 const target::KernelFacts &facts) {
  SmallVector<Operation *> lanes = independentLanes(choice, facts);
  if (lanes.empty())
    return true;
  if (lanes.size() != 1)
    return false;
  Operation *lane = lanes.front();
  std::optional<int64_t> extent;
  if (auto found = facts.staticDomainExtents.find(lane);
      found != facts.staticDomainExtents.end())
    extent = found->second;
  else if (auto found = facts.staticDomainBounds.find(lane);
           found != facts.staticDomainBounds.end())
    extent = found->second.second - found->second.first;
  constexpr int64_t maximumNarrowLaneExtent = 128;
  return extent && *extent > 0 && *extent <= maximumNarrowLaneExtent;
}

bool ownsOrderedStream(const AxisChoice &choice,
                       const target::KernelFacts &facts) {
  return llvm::any_of(facts.stateStreams, [&](const auto &entry) {
    return llvm::is_contained(choice.parallels, nearestParallel(entry.first));
  });
}

std::optional<int64_t>
innerStreamContractionExtent(Operation *domain,
                             const target::KernelFacts &facts) {
  for (const auto &entry : facts.contractions) {
    const target::ContractionFact &contract = entry.second;
    const target::LogicalAxis *reduction = nullptr;
    for (unsigned axis : contract.lhsReductionAxes)
      if (axis < contract.lhsAxes.size() &&
          contract.lhsAxes[axis].domain == domain) {
        reduction = &contract.lhsAxes[axis];
        break;
      }
    if (!reduction)
      continue;
    bool nested = false;
    for (Operation *parent = entry.first->getParentOp(); parent;
         parent = parent->getParentOp())
      if (::intent::target::semanticOperationName(*parent) == "intent.state_stream") {
        nested = true;
        break;
      }
    if (!nested)
      continue;
    int64_t extent = 0;
    if (!StringRef(reduction->extent).getAsInteger(10, extent) && extent > 0)
      return extent;
  }
  return std::nullopt;
}

SmallVector<unsigned> contractionProgramAxes(
    const target::ContractionFact &contraction,
    const llvm::DenseMap<Operation *, unsigned> &positions,
    ArrayRef<AxisChoice> choices) {
  SmallVector<unsigned> result;
  for (const target::LogicalAxis &axis : contraction.resultAxes) {
    auto found = positions.find(axis.domain);
    if (found == positions.end())
      continue;
    const AxisChoice &choice = choices[found->second];
    if (choice.programOrder && choice.tiled)
      result.push_back(found->second);
  }
  llvm::sort(result, [&](unsigned lhs, unsigned rhs) {
    return *choices[lhs].programOrder < *choices[rhs].programOrder;
  });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

FailureOr<AxisAssignments>
assignAxes(const target::KernelFacts &facts) {
  SmallVector<AxisChoice, 0> choices;
  llvm::DenseMap<Operation *, unsigned> positions;
  auto ensure = [&](Operation *domain) -> AxisChoice & {
    auto found = positions.find(domain);
    if (found != positions.end())
      return choices[found->second];
    unsigned position = choices.size();
    positions[domain] = position;
    choices.push_back(AxisChoice{});
    choices.back().domain = domain;
    return choices.back();
  };

  int64_t programOrder = 0;
  for (Operation *parallel : facts.parallels) {
    FailureOr<ArrayRef<Operation *>> domains = ownedDomains(*parallel, facts);
    if (failed(domains))
      return failure();
    Operation *source = parallel->getOperand(0).getDefiningOp();
    for (Operation *domain : *domains) {
      AxisChoice &choice = ensure(domain);
      if (!llvm::is_contained(choice.parallels, parallel))
        choice.parallels.push_back(parallel);
      bool tiled = domains->size() == 1 &&
                   facts.partitionDomains.count(source) != 0;
      auto fixed = facts.partitionFixedExtents.find(source);
      if (tiled && fixed != facts.partitionFixedExtents.end()) {
        if (choice.fixedOwnershipExtent &&
            *choice.fixedOwnershipExtent != fixed->second) {
          parallel->emitOpError(
              "maps one logical axis to conflicting fixed partition extents");
          return failure();
        }
        choice.fixedOwnershipExtent = fixed->second;
      }
      if (tiled && facts.partitionCounts.count(source)) {
        if (choice.countPartition && choice.countPartition != source) {
          parallel->emitOpError(
              "maps one logical axis to conflicting count partitions");
          return failure();
        }
        if (choice.fixedOwnershipExtent) {
          parallel->emitOpError(
              "maps one logical axis to fixed-extent and count partitions");
          return failure();
        }
        choice.countPartition = source;
      }
      if (choice.programOrder)
        continue;
      choice.programOrder = programOrder++;
      choice.tiled = tiled;
      appendRole(choice.roles, "parallel");
    }
  }
  for (Operation *domain : facts.orderedDomains)
    if (!facts.serialLoopDomains.contains(domain) &&
        !facts.runtimeBoundedDomains.count(domain))
      appendRole(ensure(domain).roles, "ordered");
  for (const auto &entry : facts.scans)
    if (entry.second.scalarConsumers)
      appendRole(ensure(entry.second.axis).roles, "ordered");
  for (Operation *domain : facts.contractionDomains)
    appendRole(ensure(domain).roles, "reduction");
  for (Operation *domain : facts.reductionDomains)
    appendRole(ensure(domain).roles, "reduction");
  for (Operation *domain : facts.vectorDomains)
    appendRole(ensure(domain).roles, "lane");
  for (const auto &entry : facts.raggedRelations)
    for (Operation *member : entry.second.memberDomains)
      appendRole(ensure(member).roles, "ragged_member");

  for (const auto &entry : facts.raggedRelations) {
    bool ordered = llvm::any_of(entry.second.memberDomains, [&](Operation *member) {
      return facts.orderedDomains.contains(member);
    });
    if (ordered)
      for (Operation *member : entry.second.memberDomains)
        appendRole(ensure(member).roles, "ordered");
  }

  SmallVector<std::pair<Operation *, Operation *>> matrixAxes;
  auto matrixAxis = [](ArrayRef<target::LogicalAxis> axes,
                       ArrayRef<unsigned> reduced,
                       ArrayRef<unsigned> batched) -> Operation * {
    for (size_t position = axes.size(); position > 0; --position) {
      size_t axis = position - 1;
      if (!llvm::is_contained(reduced, axis) &&
          !llvm::is_contained(batched, axis) && axes[axis].domain)
        return axes[axis].domain;
    }
    return nullptr;
  };
  auto assignMatrixRole = [&](Operation *domain, StringRef role,
                              Operation &contract) -> LogicalResult {
    if (!domain)
      return success();
    AxisChoice &choice = ensure(domain);
    appendRole(choice.roles, role);
    return success();
  };
  for (const auto &entry : facts.contractions) {
    const target::ContractionFact &contract = entry.second;
    Operation *m = matrixAxis(contract.lhsAxes, contract.lhsReductionAxes,
                              contract.lhsBatchAxes);
    Operation *n = matrixAxis(contract.rhsAxes, contract.rhsReductionAxes,
                              contract.rhsBatchAxes);
    if (failed(assignMatrixRole(m, "contraction_m", *entry.first)) ||
        failed(assignMatrixRole(n, "contraction_n", *entry.first)))
      return failure();
    if (m && n && m != n)
      matrixAxes.emplace_back(m, n);
  }
  for (const auto &entry : facts.sparseContractions) {
    const target::SparseContractionFact &contract = entry.second;
    if (!contract.rowDomain || !contract.columnDomain ||
        !contract.reductionDomain)
      return entry.first->emitOpError(
          "has incomplete canonical sparse-contraction axes");
    if (failed(assignMatrixRole(contract.rowDomain, "contraction_m",
                                *entry.first)) ||
        failed(assignMatrixRole(contract.columnDomain, "contraction_n",
                                *entry.first)))
      return failure();
    appendRole(ensure(contract.reductionDomain).roles, "reduction");
    if (contract.rowDomain && contract.columnDomain &&
        contract.rowDomain != contract.columnDomain)
      matrixAxes.emplace_back(contract.rowDomain, contract.columnDomain);
  }
  for (auto [m, n] : matrixAxes) {
    AxisChoice &mChoice = ensure(m);
    AxisChoice &nChoice = ensure(n);
    if (mChoice.programOrder && nChoice.programOrder &&
        *mChoice.programOrder > *nChoice.programOrder)
      std::swap(mChoice.programOrder, nChoice.programOrder);
  }

  // Full-domain tensor programs do not expose an author-written blocking
  // skeleton.  Their result axes are nevertheless independent ownership axes:
  // each physical program writes one disjoint result tile.  Introduce those
  // ownership decisions from value provenance, while preserving explicit
  // scalar parallel loops exactly as authored.
  facts.kernel.entry.walk([&](Operation *operation) {
    StringRef name = ::intent::target::semanticOperationName(*operation);
    if (name != "intent.view_store" && name != "intent.scatter_unique" &&
        name != "intent.scatter_reduce" && name != "intent.atomic_add" &&
        name != "intent.atomic_cas")
      return;
    auto valueIndex =
        operation->getAttrOfType<IntegerAttr>("intent.value_operand_index");
    unsigned index = valueIndex && valueIndex.getInt() >= 0
                         ? static_cast<unsigned>(valueIndex.getInt())
                         : name == "intent.view_store" &&
                                   operation->getNumOperands() > 1
                               ? 1u
                               : operation->getNumOperands() > 0
                                   ? operation->getNumOperands() - 1
                                   : 0u;
    if (index >= operation->getNumOperands())
      return;
    auto axes = facts.valueAxes.find(operation->getOperand(index));
    if (axes == facts.valueAxes.end())
      return;
    for (const target::LogicalAxis &axis : axes->second) {
      bool reductionRole =
          axis.domain && (facts.contractionDomains.contains(axis.domain) ||
                          facts.reductionDomains.contains(axis.domain));
      bool scanRole = axis.domain && llvm::any_of(
                                         facts.scans, [&](const auto &entry) {
                                           return entry.second.axis == axis.domain;
                                         });
      bool orderedRole =
          axis.domain && facts.orderedDomains.contains(axis.domain);
      bool independentContractionResult =
          axis.domain && hasRole(ensure(axis.domain).roles, "contraction_m");
      bool independentOrderedResult =
          orderedRole && independentContractionResult;
      if (!axis.domain || scanRole ||
          (reductionRole && !independentContractionResult) ||
          (orderedRole && !independentOrderedResult))
        continue;
      AxisChoice &choice = ensure(axis.domain);
      if (choice.programOrder)
        continue;
      choice.programOrder = programOrder++;
      choice.tiled = true;
      appendRole(choice.roles, "parallel");
    }
  });

  // Once a contraction result axis owns program space, its ownership range is
  // also the local matrix-fragment extent.  Keeping the earlier derived lane
  // role would describe a second, runtime-sized fragment range for the same
  // logical axis and lets provider lowering observe two conflicting physical
  // answers.  Intermediate contraction axes that do not own programs retain
  // their lane role.
  for (AxisChoice &choice : choices) {
    bool matrixOwnership =
        choice.programOrder && choice.tiled &&
        (hasRole(choice.roles, "contraction_m") ||
         hasRole(choice.roles, "contraction_n"));
    if (matrixOwnership)
      llvm::erase_if(choice.roles,
                     [](const std::string &role) { return role == "lane"; });
  }

  SmallVector<unsigned> canonicalOrder(choices.size());
  SmallVector<int64_t> canonicalNodes(choices.size());
  std::iota(canonicalOrder.begin(), canonicalOrder.end(), 0);
  for (unsigned position : canonicalOrder) {
    auto node =
        choices[position].domain->getAttrOfType<IntegerAttr>("intent.node");
    if (!node) {
      choices[position].domain->emitOpError(
          "has no canonical Kernel IR node for physical axis ordering");
      return failure();
    }
    canonicalNodes[position] = node.getInt();
  }
  llvm::sort(canonicalOrder, [&](unsigned lhs, unsigned rhs) {
    return canonicalNodes[lhs] < canonicalNodes[rhs];
  });

  if (programOrder == 0) {
    facts.kernel.entry.emitOpError(
        "has no independent output axis for GPU program ownership");
    return failure();
  }

  for (unsigned position : canonicalOrder) {
    AxisChoice &choice = choices[position];
    if (choice.programOrder || choice.roles.size() != 1 ||
        !hasRole(choice.roles, "lane") ||
        !canDistributePointwiseLane(choice.domain, facts))
      continue;
    choice.programOrder = programOrder++;
    choice.tiled = true;
    appendRole(choice.roles, "parallel");
  }

  for (AxisChoice &choice : choices) {
    bool scalarParallel = choice.programOrder && !choice.tiled &&
                          choice.roles.size() == 1 &&
                          hasRole(choice.roles, "parallel") &&
                          hasPackableIndependentLanes(choice, facts);
    choice.packedLane =
        scalarParallel && !choice.parallels.empty() &&
        llvm::all_of(choice.parallels, [&](Operation *parallel) {
          return canPackScalarParallel(parallel, choice.domain, facts);
        });
    if (choice.packedLane) {
      appendRole(choice.roles, "lane");
      appendRole(choice.roles, "packed_lane");
    }
  }

  bool hasMatrixProgramAxes = llvm::any_of(choices, [](const AxisChoice &choice) {
    return choice.programOrder && choice.tiled &&
           (hasRole(choice.roles, "contraction_m") ||
            hasRole(choice.roles, "contraction_n"));
  });
  unsigned ordinaryTile = hasMatrixProgramAxes ? 2 : 0;
  unsigned queryTile = 0;
  unsigned raggedTile = 0;
  unsigned streamTile = 0;
  unsigned streamContractionTile = 0;
  unsigned scaledStreamTile = 0;
  unsigned scanTile = 0;
  unsigned reductionTile = 0;
  unsigned laneTile = 0;
  auto indexedTile = [](StringRef base, unsigned &ordinal) {
    unsigned current = ordinal++;
    return current == 0 ? base.str()
                        : base.str() + "_" + std::to_string(current);
  };
  SmallVector<unsigned> physicalOrder(canonicalOrder);
  llvm::sort(physicalOrder, [&](unsigned lhs, unsigned rhs) {
    const std::optional<int64_t> &lhsOrder = choices[lhs].programOrder;
    const std::optional<int64_t> &rhsOrder = choices[rhs].programOrder;
    if (lhsOrder && rhsOrder)
      return *lhsOrder < *rhsOrder;
    if (lhsOrder || rhsOrder)
      return lhsOrder.has_value();
    return canonicalNodes[lhs] < canonicalNodes[rhs];
  });
  for (unsigned position : physicalOrder) {
    AxisChoice &choice = choices[position];
    std::string packedTile = choice.packedLane
                                 ? indexedTile("lane_pack", laneTile)
                                 : std::string();
    auto nextLaneTile = [&]() {
      auto extent = facts.staticDomainExtents.find(choice.domain);
      auto bounds = facts.staticDomainBounds.find(choice.domain);
      std::optional<int64_t> staticExtent =
          extent != facts.staticDomainExtents.end()
              ? std::optional<int64_t>(extent->second)
          : bounds != facts.staticDomainBounds.end()
              ? std::optional<int64_t>(bounds->second.second -
                                       bounds->second.first)
              : std::nullopt;
      if (!staticExtent)
        return indexedTile("row_vector", laneTile);
      int64_t physical = 1;
      while (physical < *staticExtent)
        physical *= 2;
      return "fixed_" + std::to_string(physical);
    };
    std::optional<std::string> explicitReductionLaneTile;
    if (!choice.packedLane && hasRole(choice.roles, "lane") &&
        facts.reductionDomains.contains(choice.domain) &&
        !hasRole(choice.roles, "ordered"))
      explicitReductionLaneTile = nextLaneTile();
    if (choice.programOrder) {
      std::string tile;
      if (!choice.tiled) {
        tile = choice.packedLane ? packedTile : "one";
      } else if (choice.countPartition) {
        FailureOr<int64_t> partitionNode =
            node(*choice.countPartition, "count-partition extent binding");
        if (failed(partitionNode))
          return failure();
        tile = "partition_extent_" + std::to_string(*partitionNode);
      } else if (choice.fixedOwnershipExtent) {
        tile = "fixed_" + std::to_string(*choice.fixedOwnershipExtent);
      } else if (ownsOrderedStream(choice, facts)) {
        tile = indexedTile("query", queryTile);
      } else if (hasRole(choice.roles, "ragged_member")) {
        tile = indexedTile("ragged_member", raggedTile);
      } else if (hasRole(choice.roles, "contraction_m")) {
        tile = "program_m";
      } else if (hasRole(choice.roles, "contraction_n")) {
        tile = "program_n";
      } else if (ordinaryTile == 0) {
        tile = "program_m";
        ++ordinaryTile;
      } else if (ordinaryTile == 1) {
        tile = "program_n";
        ++ordinaryTile;
      } else {
        tile = "program_" + std::to_string(ordinaryTile++);
      }
      addRange(choice, "ownership", 0, std::move(tile));
    }
    if (hasRole(choice.roles, "ordered")) {
      auto fixed = facts.orderedStreamFixedExtents.find(choice.domain);
      bool scanAxis = llvm::any_of(facts.scans, [&](const auto &entry) {
        return entry.second.scalarConsumers && entry.second.axis == choice.domain;
      });
      std::string tile =
          scanAxis ? indexedTile("scan", scanTile)
          : fixed != facts.orderedStreamFixedExtents.end()
              ? "fixed_" + std::to_string(fixed->second)
          : facts.scaledStreamDomains.contains(choice.domain)
              ? indexedTile("stream_scaled", scaledStreamTile)
          : facts.contractionDomains.contains(choice.domain)
              ? indexedTile("stream_contract", streamContractionTile)
          : hasRole(choice.roles, "ragged_member")
              ? indexedTile("ragged_member", raggedTile)
              : indexedTile("stream", streamTile);
      addRange(choice, "traversal", 0, tile);
      if (facts.serialLoopDomains.contains(choice.domain))
        addRange(choice, "traversal", 1, "one");
    }
    if (hasRole(choice.roles, "reduction")) {
      const AxisChoice::RangeChoice *traversal =
          findRange(choice, "traversal");
      std::optional<int64_t> innerExtent =
          innerStreamContractionExtent(choice.domain, facts);
      addRange(choice, "reduction", 0,
               innerExtent
                   ? "fixed_" + std::to_string(*innerExtent)
                   : traversal ? traversal->tile
                   : explicitReductionLaneTile ? *explicitReductionLaneTile
                               : indexedTile("reduction", reductionTile));
    }
    if (choice.packedLane) {
      addRange(choice, "lane", 0, packedTile);
    } else if (hasRole(choice.roles, "lane")) {
      addRange(choice, "lane", 0,
               explicitReductionLaneTile ? *explicitReductionLaneTile
                                         : nextLaneTile());
    }
    if (choice.ranges.empty())
      addRange(choice, "traversal", 0, "one");
  }

  SmallVector<unsigned> parents(choices.size());
  std::iota(parents.begin(), parents.end(), 0);
  auto root = [&](unsigned value) {
    while (parents[value] != value) {
      parents[value] = parents[parents[value]];
      value = parents[value];
    }
    return value;
  };
  auto unite = [&](unsigned lhs, unsigned rhs) {
    lhs = root(lhs);
    rhs = root(rhs);
    if (lhs != rhs)
      parents[rhs] = lhs;
  };
  for (const auto &entry : facts.contractions) {
    SmallVector<unsigned> axes =
        contractionProgramAxes(entry.second, positions, choices);
    for (unsigned index = 1; index < axes.size(); ++index)
      unite(axes.front(), axes[index]);
  }

  llvm::DenseMap<unsigned, SmallVector<unsigned>> components;
  for (auto [position, choice] : llvm::enumerate(choices))
    if (choice.programOrder && choice.tiled)
      components[root(position)].push_back(position);
  SmallVector<SmallVector<unsigned>> groups;
  for (auto &entry : components)
    if (entry.second.size() > 1) {
      llvm::sort(entry.second, [&](unsigned lhs, unsigned rhs) {
        return *choices[lhs].programOrder < *choices[rhs].programOrder;
      });
      groups.push_back(std::move(entry.second));
    }
  llvm::sort(groups, [&](const auto &lhs, const auto &rhs) {
    return *choices[lhs.front()].programOrder <
           *choices[rhs.front()].programOrder;
  });

  std::array<int64_t, 3> nextFold = {0, 0, 0};
  int64_t nextWorker = 0;
  llvm::DenseSet<unsigned> assigned;
  for (auto [groupIndex, group] : llvm::enumerate(groups)) {
    int64_t worker = std::min<int64_t>(nextWorker++, 2);
    std::string groupName =
        groupIndex == 0 ? "group_m" : "group_" + std::to_string(groupIndex);
    for (unsigned position : group) {
      choices[position].worker = worker;
      choices[position].fold = nextFold[worker]++;
      choices[position].group = groupName;
      assigned.insert(position);
    }
  }
  SmallVector<unsigned> tiled;
  SmallVector<unsigned> scalar;
  for (unsigned position : physicalOrder) {
    const AxisChoice &choice = choices[position];
    if (!choice.programOrder || assigned.contains(position))
      continue;
    (choice.tiled ? tiled : scalar).push_back(position);
  }
  auto assignWorker = [&](unsigned position) {
    int64_t worker = std::min<int64_t>(nextWorker, 2);
    choices[position].worker = worker;
    choices[position].fold = nextFold[worker]++;
    if (nextWorker < 3)
      ++nextWorker;
  };
  for (unsigned position : tiled)
    assignWorker(position);
  for (unsigned position : llvm::reverse(scalar))
    assignWorker(position);

  auto ownsOneDomain = [&](Operation *parallel) {
    auto found = facts.parallelDomains.find(parallel);
    return found != facts.parallelDomains.end() && found->second.size() == 1;
  };
  bool hasDelegatedTileChoice = llvm::any_of(choices, [](const AxisChoice &choice) {
    return llvm::any_of(choice.ranges, [](const AxisChoice::RangeChoice &range) {
      return range.tile != "one" && !StringRef(range.tile).starts_with("fixed_");
    });
  });
  Operation *outerProgram = nullptr;
  for (const target::RegionNode &region : facts.kernel.regions.nodes)
    if (::intent::target::semanticOperationName(*region.operation) ==
            "intent.parallel" &&
        !region.parent) {
      outerProgram = region.operation;
      break;
    }
  bool hasMaterializedScan = !facts.scans.empty();
  for (AxisChoice &choice : choices)
    choice.reuse = choice.programOrder && !choice.tiled && !choice.packedLane &&
                   !hasMaterializedScan &&
                   !hasDelegatedTileChoice &&
                   llvm::all_of(choice.parallels, [&](Operation *parallel) {
                     return parallel == outerProgram;
                   }) &&
                   llvm::all_of(choice.parallels, ownsOneDomain) &&
                   independentLaneCount(choice, facts) == 1;

  return AxisAssignments{std::move(choices), false};
}

} // namespace

FailureOr<PhysicalDecisions>
emitPhysicalDecisions(OpBuilder &builder, const KernelFacts &facts,
                      bool conservativeAutomaticBlocking) {
  PhysicalDecisions decisions;
  FailureOr<Operation *> root = programRoot(facts);
  FailureOr<AxisAssignments> assignments = assignAxes(facts);
  if (failed(root) || failed(assignments))
    return failure();

  // Construction establishes a complete executable baseline, not the final
  // GPU schedule.  Every compiler-owned range begins as one logical element,
  // all program axes share one worker, and no reuse/grouping is assumed.  A
  // source-visible fixed partition remains fixed because changing it would
  // change what the authored body observes.  AutomaticBlocking replaces this
  // conservative mapping with the selected GPU structure in a later pass.
  if (conservativeAutomaticBlocking) {
    for (AxisChoice &choice : assignments->axes) {
      if (choice.programOrder) {
        choice.worker = 0;
        choice.fold = *choice.programOrder;
      }
      choice.reuse = false;
      choice.group.clear();
      for (AxisChoice::RangeChoice &range : choice.ranges) {
        bool sourceVisibleFixedOwnership =
            range.purpose == "ownership" && range.level == 0 &&
            (choice.fixedOwnershipExtent.has_value() || choice.countPartition);
        bool sourceVisibleFixedTraversal =
            range.purpose == "traversal" && range.level == 0 &&
            facts.orderedStreamFixedExtents.contains(choice.domain);
        if (!sourceVisibleFixedOwnership && !sourceVisibleFixedTraversal)
          range.tile = "one";
      }
    }
    assignments->persistent = false;
  }

  IntegerAttr rootNode;
  if (*root) {
    FailureOr<int64_t> resolved = node(**root, "program mapping");
    if (failed(resolved))
      return failure();
    rootNode = i64(builder, *resolved);
  }
  decisions.program = builder.create<intent::plan::LaunchOp>(
      *root ? (*root)->getLoc() : facts.kernel.entry.getLoc(), rootNode,
      builder.getBoolAttr(assignments->persistent));

  llvm::StringSet<> roundedBlockExtents;
  auto collectImplicitExtent = [&](const target::LogicalAxis &axis) {
    if (!axis.domain && axis.extent != "1")
      roundedBlockExtents.insert(axis.extent);
  };
  for (const auto &entry : facts.contractions) {
    const target::ContractionFact &contraction = entry.second;
    for (const target::LogicalAxis &axis : contraction.lhsAxes)
      collectImplicitExtent(axis);
    for (const target::LogicalAxis &axis : contraction.rhsAxes)
      collectImplicitExtent(axis);
    for (const target::LogicalAxis &axis : contraction.resultAxes)
      collectImplicitExtent(axis);
  }
  for (const AxisChoice &choice : assignments->axes) {
    const AxisChoice::RangeChoice *lane = findRange(choice, "lane");
    if (choice.reuse || !lane || !StringRef(lane->tile).starts_with("row_vector"))
      continue;
    FailureOr<std::string> extent = sourceDimensionSymbol(*choice.domain, facts);
    if (failed(extent))
      return failure();
    roundedBlockExtents.insert(*extent);
  }
  SmallVector<std::string> orderedImplicitExtents;
  for (const auto &extent : roundedBlockExtents)
    orderedImplicitExtents.push_back(extent.getKey().str());
  llvm::sort(orderedImplicitExtents);
  for (const std::string &extent : orderedImplicitExtents)
    decisions.blockExtents.push_back(
        builder.create<intent::plan::BlockExtentOp>(
            facts.kernel.entry.getLoc(), string(builder, extent),
            string(builder, "power_of_two"), string(builder, "zero")));

  for (const AxisChoice &choice : assignments->axes) {
    FailureOr<int64_t> domainNode = node(*choice.domain, "axis binding");
    FailureOr<std::string> logicalExtent =
        sourceDimensionSymbol(*choice.domain, facts);
    if (failed(domainNode) || failed(logicalExtent))
      return failure();
    SmallVector<Attribute> roles;
    for (const std::string &role : choice.roles)
      roles.push_back(string(builder, role));
    IntegerAttr programOrder = choice.programOrder
                                   ? i64(builder, *choice.programOrder)
                                   : IntegerAttr();
    IntegerAttr worker =
        choice.worker ? i64(builder, *choice.worker) : IntegerAttr();
    IntegerAttr fold = choice.fold ? i64(builder, *choice.fold) : IntegerAttr();
    StringAttr group =
        choice.group.empty() ? StringAttr() : string(builder, choice.group);
    decisions.axes.push_back(builder.create<intent::plan::AxisOp>(
        choice.domain->getLoc(), i64(builder, *domainNode),
        builder.getArrayAttr(roles), programOrder, worker, fold,
        builder.getBoolAttr(choice.reuse), group));
    for (const AxisChoice::RangeChoice &range : choice.ranges)
      decisions.ranges.push_back(builder.create<intent::plan::RangeOp>(
          choice.domain->getLoc(), i64(builder, *domainNode),
          string(builder, range.purpose), i64(builder, range.level),
          string(builder, range.tile), string(builder, *logicalExtent),
          IntegerAttr(), IntegerAttr(), IntegerAttr(), IntegerAttr()));
  }

  SmallVector<std::tuple<int64_t, int64_t, StringRef>> regionBindings;
  for (const AxisChoice &choice : assignments->axes) {
    if (!hasRole(choice.roles, "parallel"))
      continue;
    if (!choice.domain || choice.domain->getNumResults() != 1 ||
        !isa<intent::DomainType>(choice.domain->getResult(0).getType()))
      return facts.kernel.entry.emitOpError(
          "has a parallel axis without one canonical logical region value");
    FailureOr<int64_t> regionValue =
        valueID(choice.domain->getResult(0), facts.kernel, *choice.domain,
                "region-value ownership binding");
    FailureOr<int64_t> axisNode =
        node(*choice.domain, "region-value physical axis binding");
    if (failed(regionValue) || failed(axisNode))
      return failure();
    regionBindings.emplace_back(*regionValue, *axisNode,
                                StringRef("ownership"));
  }
  for (const auto &entry : facts.regionArgumentAxes) {
    auto argument = dyn_cast<BlockArgument>(entry.first);
    if (!argument || !entry.second.domain) {
      facts.kernel.entry.emitOpError(
          "has a region argument without an exact logical-axis provenance");
      return failure();
    }
    Operation *owner = argument.getOwner()->getParentOp();
    if (!owner) {
      facts.kernel.entry.emitOpError(
          "has a region argument without a canonical region owner");
      return failure();
    }
    StringRef name = ::intent::target::semanticOperationName(*owner);
    if (name == "intent.for")
      continue;
    if (name != "intent.parallel" && name != "intent.state_stream")
      return owner->emitOpError(
          "does not define a supported physical region-argument binding");
    FailureOr<int64_t> argumentID = valueID(
        argument, facts.kernel, *owner, "region-argument range binding");
    FailureOr<int64_t> axisNode = node(
        *entry.second.domain, "region-argument physical axis binding");
    if (failed(argumentID) || failed(axisNode))
      return failure();
    regionBindings.emplace_back(
        *argumentID, *axisNode,
        name == "intent.parallel" ? StringRef("ownership")
                                  : StringRef("traversal"));
  }
  llvm::sort(regionBindings, [](const auto &lhs, const auto &rhs) {
    return std::get<0>(lhs) < std::get<0>(rhs);
  });
  for (const auto &[value, axis, purpose] : regionBindings)
    decisions.regionBindings.push_back(
        builder.create<intent::plan::RegionBindingOp>(
            facts.kernel.entry.getLoc(), i64(builder, value),
            i64(builder, axis), string(builder, purpose), i64(builder, 0)));

  for (const target::CountPartitionFact &partition : facts.countPartitions) {
    FailureOr<int64_t> partitionNode =
        node(*partition.partition, "count-partition binding");
    FailureOr<int64_t> axisNode =
        node(*partition.domain, "count-partition axis binding");
    FailureOr<int64_t> countValue =
        valueID(partition.count, facts.kernel, *partition.partition,
                "count-partition count binding");
    FailureOr<int64_t> partArgument =
        valueID(partition.partArgument, facts.kernel, *partition.iteration,
                "count-partition part binding");
    FailureOr<int64_t> regionArgument =
        valueID(partition.regionArgument, facts.kernel, *partition.iteration,
                "count-partition region binding");
    if (failed(partitionNode) || failed(axisNode) || failed(countValue) ||
        failed(partArgument) || failed(regionArgument))
      return failure();
    std::string segmentExtent =
        "partition_extent_" + std::to_string(*partitionNode);
    decisions.partitionBindings.push_back(
        builder.create<intent::plan::PartitionBindingOp>(
            partition.partition->getLoc(), i64(builder, *partitionNode),
            i64(builder, *axisNode), i64(builder, *countValue),
            i64(builder, *partArgument), i64(builder, *regionArgument),
            string(builder, segmentExtent)));
  }

  SmallVector<int64_t> streamNodes;
  for (const auto &entry : facts.kernel.stateStreams)
    streamNodes.push_back(entry.first);
  llvm::sort(streamNodes);
  for (int64_t streamNode : streamNodes) {
    const target::StateStreamStructure &stream =
        facts.kernel.stateStreams.lookup(streamNode);
    Operation *axis = facts.kernel.nodes.lookup(stream.axisNode);
    Operation *relation = nullptr;
    auto member = axis ? facts.raggedMembers.find(axis) : facts.raggedMembers.end();
    if (member != facts.raggedMembers.end())
      relation = member->second.relation;
    else if (axis) {
      auto outer = facts.raggedOuterRelations.find(axis);
      if (outer != facts.raggedOuterRelations.end())
        relation = outer->second;
    }
    IntegerAttr relationNode;
    if (relation) {
      FailureOr<int64_t> id =
          node(*relation, "state-stream selected ragged relation");
      if (failed(id))
        return failure();
      relationNode = i64(builder, *id);
    }
    IntegerAttr stopValue;
    if (stream.stopValue >= 0)
      stopValue = i64(builder, stream.stopValue);
    IntegerAttr partitionNode;
    auto partition =
        facts.partitionRegionArguments.find(stream.operation->getOperand(0));
    if (partition != facts.partitionRegionArguments.end()) {
      FailureOr<int64_t> id =
          node(*partition->second, "state-stream count-partition binding");
      if (failed(id))
        return failure();
      partitionNode = i64(builder, *id);
    }
    decisions.streamBindings.push_back(
        builder.create<intent::plan::StreamBindingOp>(
            stream.operation->getLoc(), i64(builder, stream.node),
            i64(builder, stream.axisNode), string(builder, "traversal"),
            i64(builder, 0), stopValue, relationNode, partitionNode));
  }

  if (conservativeAutomaticBlocking)
    for (const target::AccessRangeFact &access : facts.accessRanges) {
    auto choice = llvm::find_if(assignments->axes, [&](const AxisChoice &candidate) {
      return candidate.domain == access.axis;
    });
    const AxisChoice::RangeChoice *ownership = nullptr;
    if (choice != assignments->axes.end()) {
      ownership = findRange(*choice, "ownership");
      if (!ownership)
        ownership = findRange(*choice, "traversal");
      if (!ownership)
        ownership = findRange(*choice, "reduction");
      if (!ownership)
        ownership = findRange(*choice, "lane");
    }
    FailureOr<int64_t> axisNode =
        node(*access.axis, "access-range axis binding");
    FailureOr<int64_t> transferNode =
        node(*access.transfer, "access-range transfer binding");
    FailureOr<std::string> sourceExtent =
        sourceViewDimensionSymbol(*access.transfer, access.sourceAxis, facts);
    if (!ownership || failed(axisNode) || failed(transferNode) ||
        failed(sourceExtent))
      return access.transfer->emitOpError(
          "has no ownership range for its affine access footprint");
    IntegerAttr divisor = access.divisor > 1
                              ? i64(builder, access.divisor)
                              : IntegerAttr();
    IntegerAttr offset = access.divisor > 1 || access.offset != 0
                             ? i64(builder, access.offset)
                             : IntegerAttr();
    decisions.ranges.push_back(builder.create<intent::plan::RangeOp>(
        access.transfer->getLoc(), i64(builder, *axisNode),
        string(builder, "access"), i64(builder, 0),
        string(builder, ownership->tile),
        string(builder, *sourceExtent),
        i64(builder, *transferNode),
        i64(builder, access.sourceAxis), divisor, offset));
    }
  return decisions;
}

LogicalResult emitSearchSpace(ModuleOp module, const KernelFacts &facts,
                              const PhysicalDecisions &decisions) {
  SmallVector<std::string> keys;
  SmallVector<std::string> parameters;
  auto tunable = [](StringRef tile) {
    return tile != "one" && !tile.starts_with("row_vector") &&
           !tile.starts_with("fixed_") &&
           !tile.starts_with("partition_extent_");
  };
  for (intent::plan::RangeOp range : decisions.ranges) {
    if (tunable(range.getTile())) {
      Operation *domain = facts.kernel.nodes.lookup(range.getAxisNode());
      FailureOr<std::string> key =
          domain ? sourceDimensionSymbol(*domain, facts)
                 : FailureOr<std::string>(failure());
      if (failed(key))
        return failure();
      appendUnique(keys, *key);
      appendUnique(parameters, range.getTile());
    }
  }
  for (intent::plan::AxisOp axis : decisions.axes) {
    if (axis.getGroupAttr())
      appendUnique(parameters, *axis.getGroup());
  }
  for (intent::plan::BlockExtentOp extent : decisions.blockExtents)
    appendUnique(keys, extent.getLogicalExtent());
  if (parameters.empty())
    return success();

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  func::FuncOp entry = facts.kernel.entry;
  auto search = builder.create<intent::plan::SearchSpaceOp>(
      entry.getLoc(), FlatSymbolRefAttr::get(module.getContext(), entry.getName()),
      string(builder, "gpu"));
  Block &body = search.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  SmallVector<Attribute> keyAttrs;
  for (const std::string &key : keys)
    keyAttrs.push_back(string(builder, key));
  SmallVector<Attribute> parameterAttrs;
  for (const std::string &parameter : parameters)
    parameterAttrs.push_back(string(builder, parameter));
  builder.create<intent::plan::AutotuneOp>(
      entry.getLoc(), builder.getArrayAttr(keyAttrs),
      builder.getArrayAttr(parameterAttrs));
  builder.create<intent::plan::YieldOp>(entry.getLoc());
  return intent::plan::verifyGpuSearchSpace(search);
}

} // namespace intent::gpu::realization

mlir::LogicalResult intent::gpu::materializeSearchSpace(
    mlir::ModuleOp module, const intent::target::KernelFacts &facts,
    intent::plan::ProgramOp program) {
  realization::PhysicalDecisions decisions;
  for (mlir::Operation &operation : program.getBody().front()) {
    if (auto axis = mlir::dyn_cast<intent::plan::AxisOp>(operation))
      decisions.axes.push_back(axis);
    else if (auto range = mlir::dyn_cast<intent::plan::RangeOp>(operation))
      decisions.ranges.push_back(range);
    else if (auto extent =
                 mlir::dyn_cast<intent::plan::BlockExtentOp>(operation))
      decisions.blockExtents.push_back(extent);
  }
  return realization::emitSearchSpace(module, facts, decisions);
}

mlir::LogicalResult intent::gpu::formAutomaticBlocking(
    intent::plan::ProgramOp program,
    const intent::target::KernelFacts &facts) {
  mlir::SmallVector<mlir::Operation *> previous;
  for (mlir::Operation &operation : program.getBody().front())
    if (auto range = mlir::dyn_cast<intent::plan::RangeOp>(operation)) {
      if (range.getPurpose() != "access")
        previous.push_back(&operation);
    } else if (mlir::isa<intent::plan::LaunchOp, intent::plan::BlockExtentOp,
                  intent::plan::AxisOp, intent::plan::RangeOp,
                  intent::plan::RegionBindingOp,
                  intent::plan::PartitionBindingOp,
                  intent::plan::StreamBindingOp>(operation))
      previous.push_back(&operation);

  mlir::OpBuilder builder(program.getContext());
  // Replace the conservative declarations in place.  Appending the refined
  // declarations after operation bindings leaves consumers such as sparse
  // contraction ahead of the AxisOps they reference, so the program is no
  // longer a declaration-before-use physical program once the old decisions
  // are erased.
  if (previous.empty())
    builder.setInsertionPoint(program.getBody().front().getTerminator());
  else
    builder.setInsertionPoint(previous.front());
  mlir::FailureOr<realization::PhysicalDecisions> decisions =
      realization::emitPhysicalDecisions(builder, facts, false);
  if (mlir::failed(decisions))
    return mlir::failure();
  for (mlir::Operation *operation : previous)
    operation->erase();
  return mlir::success();
}

mlir::LogicalResult intent::gpu::reconcileAccessRanges(
    intent::plan::ProgramOp program,
    const intent::target::KernelFacts &facts) {
  mlir::SmallVector<intent::plan::RangeOp> old;
  for (intent::plan::RangeOp range :
       program.getBody().getOps<intent::plan::RangeOp>())
    if (range.getPurpose() == "access")
      old.push_back(range);

  llvm::DenseMap<int64_t, intent::plan::AxisOp> axes;
  llvm::DenseMap<int64_t, llvm::SmallVector<intent::plan::RangeOp>> ranges;
  for (mlir::Operation &operation : program.getBody().front()) {
    if (auto axis = mlir::dyn_cast<intent::plan::AxisOp>(operation))
      axes[axis.getNode()] = axis;
    else if (auto range = mlir::dyn_cast<intent::plan::RangeOp>(operation);
             range && range.getPurpose() != "access")
      ranges[range.getAxisNode()].push_back(range);
  }
  auto selectedRange = [&](int64_t node) -> intent::plan::RangeOp {
    auto found = ranges.find(node);
    if (found == ranges.end())
      return {};
    for (llvm::StringRef purpose : {llvm::StringRef("ownership"),
                                    llvm::StringRef("traversal"),
                                    llvm::StringRef("reduction"),
                                    llvm::StringRef("lane")}) {
      auto selected = llvm::find_if(found->second, [&](intent::plan::RangeOp range) {
        return range.getPurpose() == purpose && range.getLevel() == 0;
      });
      if (selected != found->second.end())
        return *selected;
    }
    return {};
  };

  mlir::OpBuilder builder(program.getContext());
  builder.setInsertionPoint(program.getBody().front().getTerminator());
  mlir::SmallVector<intent::plan::RangeOp> created;
  for (const target::AccessRangeFact &access : facts.accessRanges) {
    mlir::FailureOr<int64_t> axisNode =
        realization::node(*access.axis, "access-range axis realization");
    mlir::FailureOr<int64_t> transferNode =
        realization::node(*access.transfer, "access-range transfer realization");
    mlir::FailureOr<std::string> extent =
        realization::sourceViewDimensionSymbol(*access.transfer,
                                               access.sourceAxis, facts);
    intent::plan::RangeOp source =
        succeeded(axisNode) ? selectedRange(*axisNode) : intent::plan::RangeOp();
    if (failed(axisNode) || failed(transferNode) || failed(extent) || !source)
      return access.transfer->emitOpError(
          "has no selected physical range for its affine access footprint");
    mlir::IntegerAttr divisor =
        access.divisor > 1 ? builder.getI64IntegerAttr(access.divisor)
                           : mlir::IntegerAttr();
    mlir::IntegerAttr offset =
        access.divisor > 1 || access.offset != 0
            ? builder.getI64IntegerAttr(access.offset)
            : mlir::IntegerAttr();
    created.push_back(builder.create<intent::plan::RangeOp>(
        access.transfer->getLoc(), builder.getI64IntegerAttr(*axisNode),
        builder.getStringAttr("access"), builder.getI64IntegerAttr(0),
        builder.getStringAttr(source.getTile()), builder.getStringAttr(*extent),
        builder.getI64IntegerAttr(*transferNode),
        builder.getI64IntegerAttr(access.sourceAxis), divisor, offset));
  }
  for (intent::plan::RangeOp range : old)
    range.erase();
  return mlir::success();
}
