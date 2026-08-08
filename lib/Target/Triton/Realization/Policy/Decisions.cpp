#include "Support/Model.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::triton::realization {
namespace {

FailureOr<Operation *> ownedDomain(Operation &parallel,
                                   const target::KernelFacts &semantics) {
  Operation *source = parallel.getOperand(0).getDefiningOp();
  if (semantics.domainSourceAxes.count(source))
    return source;
  auto partition = semantics.partitionDomains.find(source);
  if (partition != semantics.partitionDomains.end())
    return partition->second;
  parallel.emitOpError("has no source domain for policy selection");
  return failure();
}

FailureOr<std::string> sourceDimension(Operation &domain,
                                       const target::KernelFacts &semantics) {
  Value source = semantics.domainSources.lookup(&domain);
  int64_t axis = semantics.domainSourceAxes.lookup(&domain);
  auto argument = llvm::find_if(semantics.kernel.abi.arguments,
                                [&](const target::ABIArgument &candidate) {
                                  return candidate.value == source;
                                });
  if (argument == semantics.kernel.abi.arguments.end()) {
    domain.emitOpError("does not refer to a canonical ABI argument");
    return failure();
  }
  auto shape = argument->metadata.getAs<ArrayAttr>("shape");
  auto symbol = shape && axis >= 0 && static_cast<size_t>(axis) < shape.size()
                    ? dyn_cast<StringAttr>(shape[axis])
                    : StringAttr();
  if (!symbol || symbol.getValue().empty()) {
    domain.emitOpError("does not expose a symbolic autotune dimension");
    return failure();
  }
  return symbol.getValue().str();
}

} // namespace

FailureOr<PolicyDecision> decidePolicy(const OperationFacts &facts) {
  const target::KernelFacts &semantics = facts.semantics;
  if (semantics.parallels.empty()) {
    semantics.kernel.entry.emitOpError("has no parallel ownership to realize");
    return failure();
  }

  SmallVector<Operation *> programDomains;
  llvm::DenseSet<Operation *> programDomainSet;
  llvm::DenseSet<Operation *> tiledProgramDomains;
  for (Operation *parallel : semantics.parallels) {
    FailureOr<Operation *> domain = ownedDomain(*parallel, semantics);
    if (failed(domain))
      return failure();
    if (programDomainSet.insert(*domain).second)
      programDomains.push_back(*domain);
    Operation *source = parallel->getOperand(0).getDefiningOp();
    if (semantics.partitionDomains.count(source))
      tiledProgramDomains.insert(*domain);
  }

  auto ordered = [](const llvm::DenseSet<Operation *> &domains) {
    SmallVector<Operation *> result(domains.begin(), domains.end());
    llvm::sort(result, [](Operation *lhs, Operation *rhs) {
      return lhs->getAttrOfType<IntegerAttr>("intent.node").getInt() <
             rhs->getAttrOfType<IntegerAttr>("intent.node").getInt();
    });
    return result;
  };
  SmallVector<Operation *> vectorDomains = ordered(semantics.vectorDomains);
  SmallVector<Operation *> streamedDomains =
      ordered(semantics.streamedReductionDomains);

  llvm::DenseSet<Operation *> classified = programDomainSet;
  classified.insert(semantics.vectorDomains.begin(), semantics.vectorDomains.end());
  classified.insert(semantics.streamedReductionDomains.begin(),
                    semantics.streamedReductionDomains.end());
  for (const auto &binding : semantics.boundaryDomains)
    for (Operation *domain : binding.second)
      if (!classified.contains(domain)) {
        domain->emitOpError(
            "has no program, vector, or streamed-reduction mechanism");
        return failure();
      }

  PolicyDecision policy;
  policy.workerAxes = {0};
  llvm::DenseSet<Operation *> rootCandidates;
  for (const target::RegionNode &region : semantics.kernel.regions.nodes) {
    if (region.operation->getName().getStringRef() != "intent.parallel")
      continue;
    bool hasParallelAncestor = false;
    Operation *ancestor = region.parent;
    while (ancestor) {
      if (ancestor->getName().getStringRef() == "intent.parallel") {
        hasParallelAncestor = true;
        break;
      }
      auto position = semantics.kernel.regions.positions.find(ancestor);
      ancestor = position == semantics.kernel.regions.positions.end()
                     ? nullptr
                     : semantics.kernel.regions.nodes[position->second].parent;
    }
    if (!hasParallelAncestor)
      rootCandidates.insert(region.operation);
  }
  if (rootCandidates.size() != 1) {
    semantics.kernel.entry.emitOpError(
        "Triton policy requires exactly one root parallel ownership region");
    return failure();
  }
  policy.programRoot = *rootCandidates.begin();

  static constexpr StringLiteral programTiles[] = {"BLOCK_SIZE_M",
                                                    "BLOCK_SIZE_N"};
  for (auto [index, domain] : llvm::enumerate(programDomains)) {
    if (index >= std::size(programTiles) && tiledProgramDomains.contains(domain)) {
      domain->emitOpError("exceeds the supported Triton tiled program rank");
      return failure();
    }
    policy.axes.push_back(AxisDecision{
        domain, semantics.domainSourceAxes.lookup(domain),
        "program_" + std::to_string(index),
        tiledProgramDomains.contains(domain) ? programTiles[index].str() : "one"});
  }
  for (auto [index, domain] : llvm::enumerate(streamedDomains))
    policy.axes.push_back(AxisDecision{
        domain, semantics.domainSourceAxes.lookup(domain),
        "reduction_" + std::to_string(index),
        index == 0 ? "BLOCK_SIZE_K" : "BLOCK_SIZE_REDUCTION"});
  for (Operation *domain : vectorDomains) {
    if (semantics.streamedReductionDomains.contains(domain))
      continue;
    unsigned index = llvm::count_if(policy.axes, [](const AxisDecision &axis) {
      return StringRef(axis.role).starts_with("lane_");
    });
    policy.axes.push_back(AxisDecision{
        domain, semantics.domainSourceAxes.lookup(domain),
        "lane_" + std::to_string(index), "next_power_of_two"});
  }

  if (tiledProgramDomains.empty()) {
    if (programDomains.size() != 1 || vectorDomains.size() != 1 ||
        !streamedDomains.empty()) {
      semantics.kernel.entry.emitOpError(
          "persistent grid-stride requires one program and one vector domain");
      return failure();
    }
    policy.traversal = "persistent";
    policy.mapping = "grid_stride";
    policy.usesAutotuner = false;
    return policy;
  }

  if (programDomains.size() != 2 || tiledProgramDomains.size() != 2 ||
      streamedDomains.size() != 1) {
    semantics.kernel.entry.emitOpError(
        "grouped tiled scheduling requires two tiled program axes and one "
        "streamed reduction axis");
    return failure();
  }
  policy.traversal = "grouped";
  policy.mapping = "grouped_2d_tiles";
  for (Operation *domain :
       {programDomains[0], programDomains[1], streamedDomains[0]}) {
    FailureOr<std::string> key = sourceDimension(*domain, semantics);
    if (failed(key))
      return failure();
    policy.autotuneKeys.push_back(*key);
  }
  policy.usesAutotuner = true;
  return policy;
}

} // namespace intent::triton::realization
