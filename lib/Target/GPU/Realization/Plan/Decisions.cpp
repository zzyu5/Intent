#include "Support/Decisions.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <array>
#include <numeric>

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
    if (region.operation->getName().getStringRef() != "intent.parallel" ||
        region.parent)
      continue;
    roots.push_back(region.operation);
  }
  if (roots.size() != 1) {
    facts.kernel.entry.emitOpError(
        "one GPU kernel function must contain exactly one outer program region; "
        "multiple launches require separate kernel functions");
    return failure();
  }
  return roots.front();
}

Operation *nearestParallel(Operation *operation) {
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent->getName().getStringRef() == "intent.parallel")
      return parent;
  return nullptr;
}

FailureOr<std::string> sourceDimensionSymbol(Operation &domain,
                                             const target::KernelFacts &facts) {
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

bool dependsOn(Value value, Operation &producer,
               llvm::DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  if (value.getDefiningOp() == &producer)
    return true;
  Operation *definition = value.getDefiningOp();
  return definition && llvm::any_of(definition->getOperands(), [&](Value operand) {
           return dependsOn(operand, producer, visited);
         });
}

bool reachesTerminal(Operation &operation,
                     const llvm::DenseSet<Operation *> &terminals,
                     llvm::DenseSet<Operation *> &visited) {
  if (!visited.insert(&operation).second)
    return false;
  if (terminals.contains(&operation))
    return true;
  for (Value result : operation.getResults())
    for (Operation *user : result.getUsers())
      if (reachesTerminal(*user, terminals, visited))
        return true;
  return false;
}

void appendUnique(SmallVectorImpl<std::string> &values, StringRef value) {
  if (!value.empty() && !llvm::is_contained(values, value))
    values.push_back(value.str());
}

void appendUnique(SmallVectorImpl<Value> &values, Value value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

void appendRole(SmallVectorImpl<std::string> &roles, StringRef role) {
  if (!llvm::is_contained(roles, role))
    roles.push_back(role.str());
}

bool hasRole(ArrayRef<std::string> roles, StringRef role) {
  return llvm::is_contained(roles, role);
}

struct AxisChoice {
  Operation *domain = nullptr;
  SmallVector<Operation *> parallels;
  SmallVector<std::string> roles;
  bool tiled = false;
  std::string tile;
  std::optional<int64_t> programOrder;
  std::optional<int64_t> worker;
  std::optional<int64_t> fold;
  bool reuse = false;
  std::string group;
};

struct AxisAssignments {
  SmallVector<AxisChoice> axes;
  bool persistent = false;
};

bool hasIndependentLane(const AxisChoice &choice,
                        const target::KernelFacts &facts) {
  if (choice.parallels.empty())
    return false;
  auto isOwnedByChoice = [&](Operation *operation) {
    return llvm::is_contained(choice.parallels, nearestParallel(operation));
  };
  auto usable = [&](Operation *domain) {
    return domain != choice.domain && facts.vectorDomains.contains(domain) &&
           !facts.orderedStreamDomains.contains(domain) &&
           !facts.contractionDomains.contains(domain);
  };
  for (const auto &entry : facts.boundaryDomains) {
    if (!isOwnedByChoice(entry.first))
      continue;
    if (llvm::any_of(entry.second, usable))
      return true;
  }
  for (const auto &entry : facts.valueAxes) {
    Operation *definition = entry.first.getDefiningOp();
    if (!definition || !isOwnedByChoice(definition))
      continue;
    if (llvm::any_of(entry.second, [&](const target::LogicalAxis &axis) {
          return usable(axis.domain);
        }))
      return true;
  }
  return false;
}

bool ownsOrderedStream(const AxisChoice &choice,
                       const target::KernelFacts &facts) {
  return llvm::any_of(facts.stateStreams, [&](const auto &entry) {
    return llvm::is_contained(choice.parallels, nearestParallel(entry.first));
  });
}

bool hasIndirectRaggedMembership(Operation *domain,
                                 const target::KernelFacts &facts) {
  auto member = facts.raggedMembers.find(domain);
  if (member == facts.raggedMembers.end())
    return false;
  auto relation = facts.raggedRelations.find(member->second.relation);
  return relation != facts.raggedRelations.end() &&
         static_cast<bool>(relation->second.indices);
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
  SmallVector<AxisChoice> choices;
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
      if (choice.programOrder)
        continue;
      choice.programOrder = programOrder++;
      choice.tiled = domains->size() == 1 &&
                     facts.partitionDomains.count(source) != 0;
      appendRole(choice.roles, "parallel");
    }
  }
  if (programOrder == 0) {
    facts.kernel.entry.emitOpError("has no parallel axis to assign");
    return failure();
  }

  for (Operation *domain : facts.orderedStreamDomains)
    appendRole(ensure(domain).roles, "ordered");
  for (Operation *domain : facts.contractionDomains)
    appendRole(ensure(domain).roles, "reduction");
  for (Operation *domain : facts.vectorDomains)
    appendRole(ensure(domain).roles, "lane");
  for (const auto &entry : facts.raggedRelations)
    for (Operation *member : entry.second.memberDomains)
      appendRole(ensure(member).roles, "ragged_member");

  unsigned ordinaryTile = 0;
  unsigned queryTile = 0;
  unsigned raggedTile = 0;
  unsigned streamTile = 0;
  unsigned streamContractionTile = 0;
  unsigned reductionTile = 0;
  unsigned laneTile = 0;
  auto indexedTile = [](StringRef base, unsigned &ordinal) {
    unsigned current = ordinal++;
    return current == 0 ? base.str()
                        : base.str() + "_" + std::to_string(current);
  };
  for (AxisChoice &choice : choices) {
    if (choice.programOrder) {
      if (!choice.tiled) {
        choice.tile = "one";
      } else if (ownsOrderedStream(choice, facts)) {
        choice.tile = indexedTile("query", queryTile);
      } else if (hasRole(choice.roles, "ragged_member")) {
        choice.tile = indexedTile("ragged_member", raggedTile);
      } else if (ordinaryTile == 0) {
        choice.tile = "program_m";
        ++ordinaryTile;
      } else if (ordinaryTile == 1) {
        choice.tile = "program_n";
        ++ordinaryTile;
      } else {
        choice.tile = "program_" + std::to_string(ordinaryTile++);
      }
    } else if (hasRole(choice.roles, "ordered")) {
      auto fixed = facts.orderedStreamFixedExtents.find(choice.domain);
      choice.tile = fixed != facts.orderedStreamFixedExtents.end()
                        ? "fixed_" + std::to_string(fixed->second)
                    : hasIndirectRaggedMembership(choice.domain, facts)
                        ? "one"
                    : hasRole(choice.roles, "reduction")
                        ? indexedTile("stream_contract",
                                      streamContractionTile)
                        : indexedTile("stream", streamTile);
    } else if (hasRole(choice.roles, "reduction")) {
      choice.tile = indexedTile("reduction", reductionTile);
    } else if (hasRole(choice.roles, "lane")) {
      auto extent = facts.staticDomainExtents.find(choice.domain);
      if (extent != facts.staticDomainExtents.end()) {
        int64_t physical = 1;
        while (physical < extent->second)
          physical *= 2;
        choice.tile = "fixed_" + std::to_string(physical);
      } else {
        choice.tile = indexedTile("row_vector", laneTile);
      }
    } else {
      choice.tile = "one";
    }
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
  for (auto [position, choice] : llvm::enumerate(choices)) {
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
  for (AxisChoice &choice : choices)
    choice.reuse = choice.programOrder && !choice.tiled &&
                   llvm::all_of(choice.parallels, ownsOneDomain) &&
                   hasIndependentLane(choice, facts);

  bool persistent = llvm::any_of(facts.contractions, [&](const auto &entry) {
    llvm::DenseSet<unsigned> owned;
    unsigned parallel = 0;
    unsigned tiled = 0;
    bool ragged = false;
    for (Operation *parent = entry.first->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (parent->getName().getStringRef() != "intent.parallel" ||
          parent->getNumOperands() != 1)
        continue;
      auto domains = facts.parallelDomains.find(parent);
      if (domains == facts.parallelDomains.end())
        continue;
      for (Operation *domain : domains->second) {
        auto found = positions.find(domain);
        if (found == positions.end() ||
            !choices[found->second].programOrder ||
            !owned.insert(found->second).second)
          continue;
        ragged |= facts.raggedMembers.count(domain);
        ++parallel;
        tiled += choices[found->second].tiled;
      }
    }
    return !ragged && parallel >= 3 && tiled >= 2;
  });
  if (persistent) {
    SmallVector<unsigned> programAxes;
    for (auto [position, choice] : llvm::enumerate(choices))
      if (choice.programOrder)
        programAxes.push_back(position);
    llvm::sort(programAxes, [&](unsigned lhs, unsigned rhs) {
      return *choices[lhs].programOrder < *choices[rhs].programOrder;
    });
    for (auto [fold, position] : llvm::enumerate(programAxes)) {
      choices[position].worker = 0;
      choices[position].fold = fold;
      choices[position].reuse = false;
    }
  }
  return AxisAssignments{std::move(choices), persistent};
}

struct StageDecision {
  Operation *contraction = nullptr;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  SmallVector<Operation *> terminals;
  SmallVector<int64_t> operations;
};

bool hasRaggedAxis(const target::ContractionFact &contraction,
                   const target::KernelFacts &facts) {
  auto contains = [&](ArrayRef<target::LogicalAxis> axes) {
    return llvm::any_of(axes, [&](const target::LogicalAxis &axis) {
      return axis.domain && facts.raggedMembers.count(axis.domain);
    });
  };
  return contains(contraction.lhsAxes) || contains(contraction.rhsAxes) ||
         contains(contraction.resultAxes);
}

void collectSlice(Value value, const llvm::DenseSet<Value> &inputs,
                  llvm::DenseSet<Value> &visited,
                  llvm::DenseSet<int64_t> &operations) {
  if (inputs.contains(value) || !visited.insert(value).second)
    return;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return;
  auto node = definition->getAttrOfType<IntegerAttr>("intent.node");
  if (node)
    operations.insert(node.getInt());
  for (Value operand : definition->getOperands())
    collectSlice(operand, inputs, visited, operations);
}

FailureOr<SmallVector<StageDecision, 0>>
contractionStages(const target::KernelFacts &facts) {
  llvm::DenseSet<Operation *> terminals = facts.scatterWrites;
  SmallVector<Operation *> contractions;
  for (const auto &entry : facts.contractions) {
    llvm::DenseSet<Operation *> visited;
    if (hasRaggedAxis(entry.second, facts) &&
        reachesTerminal(*entry.first, terminals, visited))
      contractions.push_back(entry.first);
  }
  llvm::sort(contractions, [](Operation *lhs, Operation *rhs) {
    return lhs->getAttrOfType<IntegerAttr>("intent.node").getInt() <
           rhs->getAttrOfType<IntegerAttr>("intent.node").getInt();
  });
  if (contractions.empty())
    return SmallVector<StageDecision, 0>();

  SmallVector<StageDecision, 0> stages;
  llvm::DenseMap<Operation *, unsigned> stagePositions;
  for (Operation *contraction : contractions) {
    stagePositions[contraction] = stages.size();
    stages.push_back(StageDecision{contraction});
  }
  for (Operation *consumer : contractions) {
    StageDecision &consumerStage = stages[stagePositions.lookup(consumer)];
    for (Value operand : consumer->getOperands()) {
      for (Operation *producer : contractions) {
        if (producer == consumer)
          continue;
        llvm::DenseSet<Value> visited;
        if (!dependsOn(operand, *producer, visited))
          continue;
        appendUnique(consumerStage.inputs, operand);
        appendUnique(stages[stagePositions.lookup(producer)].outputs, operand);
      }
    }
  }
  for (StageDecision &stage : stages) {
    if (stage.outputs.empty())
      for (Operation *terminal : terminals) {
        bool dependent = llvm::any_of(terminal->getOperands(), [&](Value operand) {
          llvm::DenseSet<Value> visited;
          return dependsOn(operand, *stage.contraction, visited);
        });
        if (dependent)
          stage.terminals.push_back(terminal);
      }
    if (stage.outputs.empty() && stage.terminals.empty())
      return stage.contraction->emitOpError(
          "has no physical stage output or terminal");

    llvm::DenseSet<Value> inputs(stage.inputs.begin(), stage.inputs.end());
    llvm::DenseSet<Value> visited;
    llvm::DenseSet<int64_t> operationNodes;
    for (Value output : stage.outputs)
      collectSlice(output, inputs, visited, operationNodes);
    for (Operation *terminal : stage.terminals) {
      FailureOr<int64_t> terminalNode = node(*terminal, "stage terminal");
      if (failed(terminalNode))
        return failure();
      operationNodes.insert(*terminalNode);
      for (Value operand : terminal->getOperands())
        collectSlice(operand, inputs, visited, operationNodes);
    }
    FailureOr<int64_t> rootNode = node(*stage.contraction, "stage root");
    if (failed(rootNode))
      return failure();
    operationNodes.insert(*rootNode);
    SmallVector<int64_t> dataflowNodes(operationNodes.begin(),
                                      operationNodes.end());
    for (int64_t operationNode : dataflowNodes) {
      Operation *operation = facts.kernel.nodes.lookup(operationNode);
      for (Operation *parent = operation ? operation->getParentOp() : nullptr;
           parent && parent != facts.kernel.entry.getOperation();
           parent = parent->getParentOp()) {
        auto parentNode = parent->getAttrOfType<IntegerAttr>("intent.node");
        if (parentNode)
          operationNodes.insert(parentNode.getInt());
      }
    }
    stage.operations.assign(operationNodes.begin(), operationNodes.end());
    llvm::sort(stage.operations);
  }
  return stages;
}

FailureOr<Operation *>
stageMemberDomain(const StageDecision &stage, const target::KernelFacts &facts) {
  const target::ContractionFact &contract =
      facts.contractions.lookup(stage.contraction);
  auto ownedMember = [&](const target::LogicalAxis &axis) {
    return axis.domain && facts.raggedMembers.count(axis.domain) &&
           llvm::any_of(facts.parallels, [&](Operation *parallel) {
             FailureOr<ArrayRef<Operation *>> owned =
                 ownedDomains(*parallel, facts);
             return succeeded(owned) && llvm::is_contained(*owned, axis.domain);
           });
  };
  for (const target::LogicalAxis &axis : contract.resultAxes)
    if (ownedMember(axis))
      return axis.domain;
  for (const target::LogicalAxis &axis : contract.lhsAxes)
    if (ownedMember(axis))
      return axis.domain;
  stage.contraction->emitOpError(
      "has no program-owned ragged member axis for physical staging");
  return failure();
}

FailureOr<std::string>
stageFeatureExtent(const StageDecision &stage,
                   const target::KernelFacts &facts, Operation *member) {
  const target::ContractionFact &contract =
      facts.contractions.lookup(stage.contraction);
  for (const target::LogicalAxis &axis : llvm::reverse(contract.resultAxes))
    if (axis.domain != member && axis.extent != "1" && !axis.extent.empty())
      return axis.extent;
  stage.contraction->emitOpError("has no staged feature extent");
  return failure();
}

FailureOr<std::string>
stageReductionExtent(const StageDecision &stage,
                     const target::KernelFacts &facts) {
  const target::ContractionFact &contract =
      facts.contractions.lookup(stage.contraction);
  if (contract.lhsReductionAxes.empty() ||
      contract.lhsReductionAxes.front() >= contract.lhsAxes.size() ||
      contract.lhsAxes[contract.lhsReductionAxes.front()].extent.empty()) {
    stage.contraction->emitOpError("has no staged reduction extent");
    return failure();
  }
  return contract.lhsAxes[contract.lhsReductionAxes.front()].extent;
}

} // namespace

FailureOr<PhysicalDecisions>
emitPhysicalDecisions(OpBuilder &builder, const KernelFacts &facts) {
  PhysicalDecisions decisions;
  FailureOr<Operation *> root = programRoot(facts);
  FailureOr<AxisAssignments> assignments = assignAxes(facts);
  if (failed(root) || failed(assignments))
    return failure();

  FailureOr<int64_t> rootNode = node(**root, "program mapping");
  if (failed(rootNode))
    return failure();
  decisions.program = builder.create<intent::plan::ProgramOp>(
      (*root)->getLoc(), i64(builder, *rootNode),
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
  bool hasWorkerReuse = llvm::any_of(
      assignments->axes, [](const AxisChoice &choice) { return choice.reuse; });
  for (const AxisChoice &choice : assignments->axes) {
    if (hasWorkerReuse || !StringRef(choice.tile).starts_with("row_vector"))
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
    if (failed(domainNode))
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
        builder.getArrayAttr(roles), string(builder, choice.tile), programOrder,
        worker, fold, builder.getBoolAttr(choice.reuse), group));
  }

  FailureOr<SmallVector<StageDecision, 0>> stages =
      contractionStages(facts);
  if (failed(stages))
    return failure();
  for (const StageDecision &stage : *stages) {
    FailureOr<int64_t> stageNode = node(*stage.contraction, "stage binding");
    FailureOr<Operation *> member =
        stageMemberDomain(stage, facts);
    FailureOr<std::string> feature =
        succeeded(member)
            ? stageFeatureExtent(stage, facts, *member)
            : FailureOr<std::string>(failure());
    FailureOr<std::string> reduction =
        stageReductionExtent(stage, facts);
    FailureOr<std::string> memberExtent =
        succeeded(member)
            ? sourceDimensionSymbol(**member, facts)
            : FailureOr<std::string>(failure());
    if (failed(stageNode) || failed(member) || failed(feature) ||
        failed(reduction) || failed(memberExtent))
      return failure();
    SmallVector<int64_t> inputs;
    SmallVector<int64_t> outputs;
    SmallVector<int64_t> terminals;
    for (Value value : stage.inputs) {
      FailureOr<int64_t> id = valueID(value, facts.kernel,
                                      *stage.contraction, "stage input binding");
      if (failed(id))
        return failure();
      inputs.push_back(*id);
    }
    for (Value value : stage.outputs) {
      FailureOr<int64_t> id = valueID(value, facts.kernel,
                                      *stage.contraction, "stage output binding");
      if (failed(id))
        return failure();
      outputs.push_back(*id);
    }
    for (Operation *terminal : stage.terminals) {
      FailureOr<int64_t> id = node(*terminal, "stage terminal binding");
      if (failed(id))
        return failure();
      terminals.push_back(*id);
    }
    decisions.stages.push_back(builder.create<intent::plan::StageOp>(
        stage.contraction->getLoc(), i64(builder, *stageNode),
        builder.getDenseI64ArrayAttr(inputs),
        builder.getDenseI64ArrayAttr(outputs),
        builder.getDenseI64ArrayAttr(stage.operations),
        builder.getDenseI64ArrayAttr(terminals)));
    FailureOr<int64_t> memberNode = node(**member, "stage member axis");
    if (failed(memberNode))
      return failure();
    decisions.stageAxes.push_back(builder.create<intent::plan::StageAxisOp>(
        stage.contraction->getLoc(), i64(builder, *stageNode),
        string(builder, "member"), i64(builder, *memberNode),
        string(builder, *memberExtent), string(builder, "ragged_member"),
        i64(builder, 1)));
    decisions.stageAxes.push_back(builder.create<intent::plan::StageAxisOp>(
        stage.contraction->getLoc(), i64(builder, *stageNode),
        string(builder, "feature"), IntegerAttr(), string(builder, *feature),
        string(builder, "feature"), i64(builder, 0)));
    decisions.stageAxes.push_back(builder.create<intent::plan::StageAxisOp>(
        stage.contraction->getLoc(), i64(builder, *stageNode),
        string(builder, "reduction"), IntegerAttr(),
        string(builder, *reduction), string(builder, "reduction"),
        IntegerAttr()));
  }
  return decisions;
}

LogicalResult emitSearchSpace(ModuleOp module, const KernelFacts &facts,
                              const PhysicalDecisions &decisions) {
  SmallVector<std::string> keys;
  SmallVector<std::string> parameters;
  auto tunable = [](StringRef tile) {
    return tile != "one" && !tile.starts_with("row_vector") &&
           !tile.starts_with("fixed_");
  };
  for (intent::plan::AxisOp axis : decisions.axes) {
    if (tunable(axis.getTile())) {
      Operation *domain = facts.kernel.nodes.lookup(axis.getNode());
      FailureOr<std::string> key =
          domain ? sourceDimensionSymbol(*domain, facts)
                 : FailureOr<std::string>(failure());
      if (failed(key))
        return failure();
      appendUnique(keys, *key);
      appendUnique(parameters, axis.getTile());
    }
    if (axis.getGroupAttr())
      appendUnique(parameters, *axis.getGroup());
  }
  for (intent::plan::BlockExtentOp extent : decisions.blockExtents)
    appendUnique(keys, extent.getLogicalExtent());
  for (intent::plan::StageAxisOp axis : decisions.stageAxes) {
    appendUnique(keys, axis.getExtent());
    appendUnique(parameters, axis.getTile());
  }
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
