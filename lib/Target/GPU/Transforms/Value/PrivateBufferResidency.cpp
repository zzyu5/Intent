#include "Intent/Target/GPU/Transforms/Passes.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Pass/Pass.h"

#include <limits>

using namespace mlir;

namespace intent::gpu {
namespace {

struct PrivateBufferCandidate {
  Operation *operation = nullptr;
  int64_t registerUnits = 0;
  bool scalarized = false;
};

FailureOr<int64_t> privateBufferRegisterUnits(
    Operation &operation, const target::LogicalBufferInfo &info,
    bool roundToVectorExtent) {
  int64_t elements = 1;
  for (int64_t extent : info.shape) {
    if (elements > std::numeric_limits<int64_t>::max() / extent) {
      operation.emitOpError(
          "private logical-buffer capacity overflows the machine plan");
      return failure();
    }
    elements *= extent;
  }
  if (roundToVectorExtent) {
    int64_t physicalExtent = 1;
    while (physicalExtent < elements) {
      if (physicalExtent > std::numeric_limits<int64_t>::max() / 2) {
        operation.emitOpError(
            "private logical-buffer vector extent overflows the machine plan");
        return failure();
      }
      physicalExtent *= 2;
    }
    elements = physicalExtent;
  }
  int64_t elementBits = info.elementType.isIndex()
                            ? 64
                            : info.elementType.getIntOrFloatBitWidth();
  int64_t unitsPerElement = std::max<int64_t>(1, (elementBits + 31) / 32);
  if (elements > std::numeric_limits<int64_t>::max() / unitsPerElement) {
    operation.emitOpError(
        "private logical-buffer capacity overflows the machine plan");
    return failure();
  }
  return elements * unitsPerElement;
}

FailureOr<llvm::DenseMap<Operation *, std::string>>
choosePrivateBufferResidencies(const target::KernelFacts &facts,
                               const DeviceCapabilities &device) {
  llvm::DenseMap<Operation *, std::string> spaces;
  llvm::DenseMap<Operation *, SmallVector<PrivateBufferCandidate>> byOwner;
  int64_t scalarizationBudget =
      std::max<int64_t>(1, device.registersPerUnit / 1024);
  int64_t localRegisterBudget =
      std::max<int64_t>(1, device.registersPerUnit / 128);

  for (const auto &entry : facts.logicalBuffers) {
    Operation *operation = entry.first;
    const target::LogicalBufferFact &buffer = entry.second;
    if (!buffer.owner) {
      operation->emitOpError("private logical buffer has no parallel owner");
      return failure();
    }
    FailureOr<int64_t> logicalUnits =
        privateBufferRegisterUnits(*operation, buffer.info, false);
    if (failed(logicalUnits))
      return failure();
    bool scalarized = buffer.info.shape.size() == 1 &&
                      *logicalUnits <= scalarizationBudget;
    if (buffer.hasUnstructuredDynamicAccess) {
      spaces[operation] = "private_workspace";
      continue;
    }
    FailureOr<int64_t> physicalUnits =
        scalarized ? logicalUnits
                   : privateBufferRegisterUnits(*operation, buffer.info, true);
    if (failed(physicalUnits))
      return failure();
    byOwner[buffer.owner].push_back(
        PrivateBufferCandidate{operation, *physicalUnits, scalarized});
  }

  for (auto &entry : byOwner) {
    llvm::sort(entry.second, [](const PrivateBufferCandidate &lhs,
                                const PrivateBufferCandidate &rhs) {
      if (lhs.registerUnits != rhs.registerUnits)
        return lhs.registerUnits < rhs.registerUnits;
      auto lhsNode = lhs.operation->getAttrOfType<IntegerAttr>("intent.node");
      auto rhsNode = rhs.operation->getAttrOfType<IntegerAttr>("intent.node");
      return lhsNode && rhsNode && lhsNode.getInt() < rhsNode.getInt();
    });
    int64_t used = 0;
    for (const PrivateBufferCandidate &candidate : entry.second) {
      bool fits = candidate.registerUnits <= localRegisterBudget - used;
      spaces[candidate.operation] =
          fits ? candidate.scalarized ? "private_scalar_array"
                                      : "private_vector"
               : "private_workspace";
      if (fits)
        used += candidate.registerUnits;
    }
  }
  return spaces;
}

class RefinePrivateBufferResidencyPass final
    : public PassWrapper<RefinePrivateBufferResidencyPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RefinePrivateBufferResidencyPass)

  explicit RefinePrivateBufferResidencyPass(const DeviceCapabilities &device)
      : device(device) {}

  StringRef getArgument() const final {
    return "intent-refine-gpu-private-buffer-residency";
  }
  StringRef getDescription() const final {
    return "Refine conservative private buffers using lifetime and device capacity facts";
  }

  void runOnOperation() final {
    SmallVector<plan::ProgramOp> programs(
        getOperation().getOps<plan::ProgramOp>());
    if (programs.size() != 1) {
      getOperation().emitError(
          "private-buffer refinement requires one physical program");
      signalPassFailure();
      return;
    }
    plan::ProgramOp program = programs.front();
    FailureOr<std::unique_ptr<PhysicalProgramAnalysis>> analysis =
        PhysicalProgramAnalysis::compute(program);
    if (failed(analysis) || failed(refine(program, **analysis, device)))
      signalPassFailure();
  }

private:
  static LogicalResult refine(plan::ProgramOp program,
                              PhysicalProgramAnalysis &analysis,
                              const DeviceCapabilities &device) {
    FailureOr<llvm::DenseMap<Operation *, std::string>> spaces =
        choosePrivateBufferResidencies(analysis.getFacts(), device);
    if (failed(spaces))
      return failure();
    target::KernelModel &kernel = analysis.getKernel();
    OpBuilder builder(program.getContext());
    unsigned bound = 0;
    for (plan::BufferOp binding : program.getBody().getOps<plan::BufferOp>()) {
      Operation *operation = kernel.nodes.lookup(binding.getNode());
      auto placement = operation ? spaces->find(operation) : spaces->end();
      auto fact = operation ? analysis.getFacts().logicalBuffers.find(operation)
                            : analysis.getFacts().logicalBuffers.end();
      if (!operation || placement == spaces->end() ||
          fact == analysis.getFacts().logicalBuffers.end())
        return binding.emitOpError(
            "does not resolve a private logical-buffer realization");
      SmallVector<int64_t> ownerNodes;
      if (placement->second == "private_workspace") {
        auto domains =
            analysis.getFacts().parallelDomains.find(fact->second.owner);
        if (domains == analysis.getFacts().parallelDomains.end() ||
            domains->second.empty())
          return binding.emitOpError(
              "private workspace has no physical program owners");
        for (Operation *domain : domains->second) {
          FailureOr<int64_t> owner =
              target::getNodeID(*domain, "private-workspace owner");
          if (failed(owner))
            return failure();
          ownerNodes.push_back(*owner);
        }
      }
      binding->setAttr("space", builder.getStringAttr(placement->second));
      binding->setAttr("owner_nodes",
                       builder.getDenseI64ArrayAttr(ownerNodes));
      ++bound;
    }
    if (bound != spaces->size())
      return program.emitOpError(
          "does not bind every private logical-buffer realization");
    return success();
  }

  DeviceCapabilities device;
};

} // namespace

std::unique_ptr<Pass>
createRefinePrivateBufferResidencyPass(const DeviceCapabilities &device) {
  return std::make_unique<RefinePrivateBufferResidencyPass>(device);
}

} // namespace intent::gpu
