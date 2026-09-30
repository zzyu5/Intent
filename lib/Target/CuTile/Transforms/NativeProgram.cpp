#include "NativeAccess.h"
#include "Configurations.h"
#include "Legalize.h"
#include "mlir/IR/Verifier.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cassert>

using namespace mlir;
namespace intent::cutile {
bool supportsE8M0ScaledMMA(gpu::CapabilitiesAttr capabilities) {
  return capabilities && capabilities.getComputeCapabilityMajor() >= 10;
}

Type withElementType(Type type, Type elementType) {
  auto fragment = dyn_cast<gpu::FragmentType>(type);
  if (!fragment)
    return elementType;
  return gpu::FragmentType::get(
      fragment.getContext(), elementType, fragment.getShape(),
      fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
}

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
  bool found = false;
  kernel.walk([&](gpu::ParameterOp parameter) {
    found |= parameter.getParameter().getRole() ==
             static_cast<uint32_t>(gpu::ParameterRole::ResidentWorkers);
  });
  return found;
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


void NativeFormRewriter::replace(Operation *operation, ValueRange values) {
  assert(operation->getNumResults() == values.size());
  replacements.push_back({operation, llvm::to_vector(values)});
}
void NativeFormRewriter::erase(Operation *operation) {
  assert(operation->use_empty());
  replacements.push_back({operation, {}});
}
void NativeFormRewriter::commit() {
  for (Replacement &replacement : replacements)
    if (!replacement.values.empty())
      replacement.operation->getResults().replaceAllUsesWith(replacement.values);
  // Destroy inner operations before their owners even if a future native form
  // consumes nested input operations in a different construction order.
  auto depth = [](Operation *operation) {
    unsigned result = 0;
    while ((operation = operation->getParentOp())) ++result;
    return result;
  };
  llvm::stable_sort(replacements, [&](const Replacement &a, const Replacement &b) {
    return depth(a.operation) > depth(b.operation);
  });
  for (Replacement &replacement : replacements)
    replacement.operation->erase();
  replacements.clear();
}

void NativeProgramFeatures::observe(Operation *operation) {
  bool matrix = isa<gpu::ContractOp, gpu::ScaledContractOp, MMAOp,
                    ScaledMMAOp>(operation);
  matrixCompute |= matrix;
  occupancySensitive |= matrix ||
      isa<gpu::ReduceOp, gpu::HistogramOp, gpu::ScanOp, ReduceOp, ScanOp>(operation);
}

NativeProgramFeatures queryNativeProgramFeatures(func::FuncOp kernel) {
  NativeProgramFeatures features;
  kernel.walk([&](Operation *operation) { features.observe(operation); });
  return features;
}

LogicalResult formNativeProgram(ModuleOp module, const gpu::TuningProfiles &profiles) {
  auto physicalKernel = gpu::getPhysicalKernel(module);
  if (failed(physicalKernel)) return failure();
  func::FuncOp kernel = *physicalKernel;
  NativeProgramInputs inputs;
  NativeProgramFeatures features;
  kernel.walk([&](Operation *operation) {
    features.observe(operation);
    if (auto op = dyn_cast<gpu::LoadOp>(operation)) inputs.loads.push_back(op);
    else if (auto op = dyn_cast<gpu::GatherOp>(operation)) inputs.gathers.push_back(op);
    else if (auto op = dyn_cast<gpu::StoreOp>(operation)) inputs.stores.push_back(op);
    else if (auto op = dyn_cast<gpu::AtomicRMWOp>(operation)) inputs.atomics.push_back(op);
    else if (auto op = dyn_cast<gpu::ContractOp>(operation)) inputs.contracts.push_back(op);
    else if (auto op = dyn_cast<gpu::ScaledContractOp>(operation)) inputs.scaledContracts.push_back(op);
    else if (auto op = dyn_cast<gpu::ReduceOp>(operation)) inputs.reductions.push_back(op);
    else if (auto op = dyn_cast<gpu::HistogramOp>(operation)) inputs.histograms.push_back(op);
    else if (auto op = dyn_cast<gpu::ScanOp>(operation)) inputs.scans.push_back(op);
    else if (auto op = dyn_cast<gpu::AssumeInBoundsOp>(operation)) inputs.assumptions.push_back(op);
  });
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!inputs.scaledContracts.empty() && !supportsE8M0ScaledMMA(capabilities))
    return inputs.scaledContracts.front().emitOpError(
        "cuTile E8M0 scaled MMA requires compute capability 10.0 or newer");
  if (features.occupancySensitive &&
      failed(declareProviderParameter(
          kernel, profiles, occupancyProfileFamily(kernel, capabilities),
          occupancyParameter, gpu::ParameterRole::ProviderOccupancy,
          isLegalOccupancy)))
    return failure();
  if (features.matrixCompute &&
      failed(declareProviderParameter(
          kernel, profiles, "ctas", ctasParameter,
          gpu::ParameterRole::ProviderCTAs, isLegalCTAs)))
    return failure();
  if (features.occupancySensitive &&
      failed(declareProviderParameter(
          kernel, profiles, "worker_warps", workerWarpsParameter,
          gpu::ParameterRole::ProviderWarps, isLegalWorkerWarps)))
    return failure();

  NativeFormRewriter rewriter;
  if (failed(formNativeAccesses(kernel, profiles, inputs, features.matrixCompute,
                                rewriter)) ||
      failed(formComputePrimitives(kernel, inputs, rewriter)))
    return failure();
  rewriter.commit();
  for (gpu::AssumeInBoundsOp assumption : inputs.assumptions)
    assumption.erase();
  gpu::eraseDeadPhysicalValues(kernel);
  return mlir::verify(module);
}
} // namespace intent::cutile
