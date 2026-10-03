#include "NativeRewrite.h"
#include "Access/Accesses.h"
#include "Compute/ComputeForms.h"
#include "Intent/Target/CuTile/Analysis/Program.h"
#include "Configuration/Configurations.h"
#include "Program.h"
#include "mlir/IR/Verifier.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cassert>

using namespace mlir;
namespace intent::cutile {
bool containsFragment(Type type) {
  if (isa<gpu::FragmentType>(type))
    return true;
  auto record = dyn_cast<gpu::RecordType>(type);
  return record && llvm::any_of(record.getFieldTypes(), [](Attribute field) {
           return containsFragment(cast<TypeAttr>(field).getValue());
         });
}

bool hasLoopCarriedFragment(func::FuncOp kernel) {
  bool found = false;
  kernel.walk([&](scf::ForOp loop) {
    found |= llvm::any_of(loop.getInitArgs(), [](Value value) {
      return containsFragment(value.getType());
    });
  });
  kernel.walk([&](scf::WhileOp loop) {
    found |= llvm::any_of(loop.getInits(), [](Value value) {
      return containsFragment(value.getType());
    });
  });
  return found;
}

bool hasResidentWorkerTraversal(func::FuncOp kernel) {
  for (Attribute attribute : gpu::getParameterDeclarations(kernel))
    if (cast<gpu::ParameterAttr>(attribute).getRole() ==
        gpu::ParameterRole::ResidentWorkers)
      return true;
  return false;
}

StringRef occupancyProfileFamily(func::FuncOp kernel,
                                  gpu::CapabilitiesAttr capabilities) {
  if (hasResidentWorkerTraversal(kernel))
    return "occupancy_persistent";
  if (hasLoopCarriedFragment(kernel))
    return "occupancy_loop";
  return capabilities.getComputeCapabilityMajor() < 9
             ? "occupancy_legacy" : "occupancy_modern";
}


LogicalResult formNativeProgram(ModuleOp module) {
  auto profiles = gpu::TuningProfiles::from(module);
  if (failed(profiles)) return failure();
  auto physicalKernel = gpu::getPhysicalKernel(module);
  if (failed(physicalKernel)) return failure();
  func::FuncOp kernel = *physicalKernel;
  NativeAccessInputs accesses;
  NativeComputeInputs computations;
  SmallVector<gpu::AssumeInBoundsOp> assumptions;
  NativeProgramFeatures features;
  kernel.walk([&](Operation *operation) {
    features.observe(operation);
    if (auto op = dyn_cast<gpu::LoadOp>(operation)) accesses.loads.push_back(op);
    else if (auto op = dyn_cast<gpu::GatherOp>(operation)) accesses.gathers.push_back(op);
    else if (auto op = dyn_cast<gpu::StoreOp>(operation)) accesses.stores.push_back(op);
    else if (auto op = dyn_cast<gpu::AtomicRMWOp>(operation)) accesses.atomics.push_back(op);
    else if (auto op = dyn_cast<gpu::ContractOp>(operation)) computations.contracts.push_back(op);
    else if (auto op = dyn_cast<gpu::ScaledContractOp>(operation)) computations.scaledContracts.push_back(op);
    else if (auto op = dyn_cast<gpu::ReduceOp>(operation)) computations.reductions.push_back(op);
    else if (auto op = dyn_cast<gpu::HistogramOp>(operation)) computations.histograms.push_back(op);
    else if (auto op = dyn_cast<gpu::ScanOp>(operation)) computations.scans.push_back(op);
    else if (auto op = dyn_cast<gpu::AssumeInBoundsOp>(operation)) assumptions.push_back(op);
  });
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!computations.scaledContracts.empty() && !supportsE8M0ScaledMMA(capabilities))
    return computations.scaledContracts.front().emitOpError(
        "cuTile E8M0 scaled MMA requires compute capability 10.0 or newer");
  if (features.occupancySensitive &&
      failed(declareProviderParameter(
          kernel, *profiles, occupancyProfileFamily(kernel, capabilities),
          occupancyParameter, gpu::ParameterRole::ProviderOccupancy,
          isLegalOccupancy)))
    return failure();
  if (features.matrixCompute &&
      failed(declareProviderParameter(
          kernel, *profiles, "ctas", ctasParameter,
          gpu::ParameterRole::ProviderCTAs, isLegalCTAs)))
    return failure();
  if (features.occupancySensitive &&
      failed(declareProviderParameter(
          kernel, *profiles, "worker_warps", workerWarpsParameter,
          gpu::ParameterRole::ProviderWarps, isLegalWorkerWarps)))
    return failure();

  NativeFormRewriter rewriter;
  if (failed(formNativeAccesses(kernel, *profiles, accesses, features.matrixCompute,
                                rewriter)) ||
      failed(formComputePrimitives(kernel, computations, rewriter)))
    return failure();
  rewriter.commit();
  for (gpu::AssumeInBoundsOp assumption : assumptions)
    assumption.erase();
  gpu::eraseDeadPhysicalValues(kernel);
  return mlir::verify(module);
}
} // namespace intent::cutile
