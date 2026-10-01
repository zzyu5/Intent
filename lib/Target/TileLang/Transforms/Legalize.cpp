#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Transforms/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "Intent/Target/TileLang/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"
#include "PassDetail.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <limits>
#include <optional>

using namespace mlir;

namespace intent::tilelang {

const gpu::TuningProfileSchema &tuningProfileSchema() {
  static const StringRef columns[] = {"value"};
  static const gpu::TuningProfileSchema schema{"tilelang", columns};
  return schema;
}

bool isLegalPipelineStageCount(int64_t stages) {
  return stages > 0 && stages <= std::numeric_limits<int32_t>::max();
}

bool isLegalMmaWarpPartition(int64_t m, int64_t n, int64_t threads) {
  if (threads <= 0 || threads % 32 != 0 || m % 16 != 0 || n % 8 != 0)
    return false;
  int64_t warps = threads / 32;
  for (int64_t mWarps = 1; mWarps <= warps; ++mWarps) {
    if (warps % mWarps != 0)
      continue;
    int64_t nWarps = warps / mWarps;
    if (m % (mWarps * 16) == 0 && n % (nWarps * 8) == 0)
      return true;
  }
  return false;
}

namespace {

constexpr llvm::StringLiteral legalizedAttr = "intent_tilelang.legalized";

bool isLegalThreadCount(int64_t threads, gpu::CapabilitiesAttr capabilities) {
  return threads > 0 && threads % 32 == 0 &&
         threads <= capabilities.getMaxThreadsPerBlock();
}

FailureOr<gpu::ParameterSpace> readConfigurationDomains(func::FuncOp kernel) {
  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities)
    return kernel.emitError("TileLang configuration requires GPU capabilities"), failure();
  auto space = gpu::ParameterSpace::read(kernel);
  if (failed(space)) return failure();
  bool sawThreads = false;
  for (gpu::ParameterAttr domain : space->declarations()) {
    if (!domain.isExtent())
      return kernel.emitError("TileLang parameters must have index type"), failure();
    if (domain.getPhase() != gpu::ConfigurationBindingPhase::Provider) continue;
    auto role = domain.getRole();
    if (role != gpu::ParameterRole::ProviderThreads &&
        role != gpu::ParameterRole::ProviderStages)
      return kernel.emitError("TileLang program contains a foreign provider parameter"), failure();
    sawThreads |= role == gpu::ParameterRole::ProviderThreads;
    for (int64_t value : domain.getCandidates().asArrayRef())
      if (role == gpu::ParameterRole::ProviderThreads
              ? !isLegalThreadCount(value, capabilities)
              : !isLegalPipelineStageCount(value))
        return kernel.emitError("TileLang parameter is outside the native option domain")
                   << "; parameter=" << domain.getName() << "; value=" << value,
               failure();
  }
  if (!sawThreads)
    return kernel.emitError("TileLang provider parameter domains have no thread binding"), failure();
  return *space;
}

std::optional<int64_t>
evaluate(gpu::PhysicalExprAttr expression, DictionaryAttr config) {
  return gpu::evaluatePhysicalExpression(expression,
      [&](gpu::PhysicalExprAttr leaf) -> std::optional<int64_t> {
        if (leaf.getKind() != gpu::PhysicalExprKind::Parameter)
          return std::nullopt;
        auto value = config.getAs<IntegerAttr>(leaf.getParameterReference().getName());
        return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
      }, [](gpu::PhysicalExprAttr operation, ArrayRef<int64_t> operands) {
        switch (operation.getKind()) {
        case gpu::PhysicalExprKind::Add:
        case gpu::PhysicalExprKind::Subtract:
        case gpu::PhysicalExprKind::Multiply:
        case gpu::PhysicalExprKind::Minimum:
        case gpu::PhysicalExprKind::Maximum: return true;
        case gpu::PhysicalExprKind::CeilDiv:
        case gpu::PhysicalExprKind::FloorDiv:
          return operands.size() == 2 && operands[0] >= 0 && operands[1] > 0;
        default: return false;
        }
      });
}

enum CandidateFailure : unsigned {
  MissingThreads = 1u << 0,
  ThreadLimit = 1u << 1,
  MmaPartition = 1u << 2,
  SharedMemory = 1u << 3,
  SparseShape = 1u << 4,
};

struct FlatSharedLifetime {
  Block *block;
  Operation *first;
  Operation *last;
  uint64_t bytes;
};

std::optional<uint64_t> allocationStorageBytes(
    AllocOp allocation, DictionaryAttr config) {
  BufferType buffer = allocation.getResult().getType();
  unsigned bitWidth = 0;
  if (auto integer = dyn_cast<IntegerType>(buffer.getElementType()))
    bitWidth = integer.getWidth();
  else if (auto floating = dyn_cast<FloatType>(buffer.getElementType()))
    bitWidth = floating.getWidth();
  else
    return std::nullopt;
  if (bitWidth == 0)
    return std::nullopt;

  uint64_t elements = 1;
  for (Attribute attribute : buffer.getShape()) {
    std::optional<int64_t> extent =
        evaluate(cast<gpu::PhysicalExprAttr>(attribute), config);
    if (!extent || *extent <= 0)
      return std::nullopt;
    uint64_t value = static_cast<uint64_t>(*extent);
    if (elements > std::numeric_limits<uint64_t>::max() / value)
      return std::nullopt;
    elements *= value;
  }
  if (elements > std::numeric_limits<uint64_t>::max() / bitWidth)
    return std::nullopt;
  uint64_t bits = elements * bitWidth;
  return bits / 8 + static_cast<uint64_t>(bits % 8 != 0);
}

std::optional<FlatSharedLifetime>
exactFlatLifetime(AllocOp allocation, uint64_t bytes) {
  Block *block = allocation->getBlock();
  Operation *first = nullptr;
  Operation *last = nullptr;
  for (OpOperand &use : allocation.getResult().getUses()) {
    Operation *user = use.getOwner();
    if (user->getBlock() != block)
      return std::nullopt;
    if (!first || user->isBeforeInBlock(first))
      first = user;
    if (!last || last->isBeforeInBlock(user))
      last = user;
  }
  if (!first)
    return std::nullopt;
  return FlatSharedLifetime{block, first, last, bytes};
}

bool isLiveAt(const FlatSharedLifetime &lifetime, Operation *operation) {
  return lifetime.first == operation || lifetime.last == operation ||
         (lifetime.first->isBeforeInBlock(operation) &&
          operation->isBeforeInBlock(lifetime.last));
}

bool exactSharedMemoryIsLegal(func::FuncOp kernel,
                              DictionaryAttr config,
                              uint64_t capacity) {
  SmallVector<FlatSharedLifetime> exactLifetimes;
  bool legal = true;
  kernel.walk([&](AllocOp allocation) {
    BufferType buffer = allocation.getResult().getType();
    if (!legal || buffer.getSpace().getValue() != BufferSpace::Shared ||
        allocation.getResult().use_empty())
      return;
    std::optional<uint64_t> bytes = allocationStorageBytes(allocation, config);
    if (!bytes)
      return;
    if (*bytes > capacity) {
      legal = false;
      return;
    }
    if (std::optional<FlatSharedLifetime> lifetime =
            exactFlatLifetime(allocation, *bytes))
      exactLifetimes.push_back(*lifetime);
  });
  if (!legal)
    return false;

  // A flat same-block interval is an exact interference fact: distinct allocs
  // whose first/last uses overlap must hold independent values concurrently.
  // Nested-region, symbolic-size and cross-block cases stay unknown and are
  // deliberately left to TileLang's own allocation planner.
  for (const FlatSharedLifetime &anchor : exactLifetimes) {
    uint64_t remaining = capacity;
    for (const FlatSharedLifetime &candidate : exactLifetimes) {
      if (candidate.block != anchor.block ||
          !isLiveAt(candidate, anchor.first))
        continue;
      if (candidate.bytes > remaining)
        return false;
      remaining -= candidate.bytes;
    }
  }
  return true;
}

unsigned configurationFailures(func::FuncOp kernel,
                               ArrayRef<gpu::ParameterAttr> domains,
                               DictionaryAttr config,
                               gpu::CapabilitiesAttr capabilities) {
  std::optional<int64_t> threads;
  for (const auto &domain : domains) {
    if (domain.getRole() != gpu::ParameterRole::ProviderThreads)
      continue;
    if (auto value = config.getAs<IntegerAttr>(domain.getName()))
      threads = value.getInt();
  }
  if (!threads)
    return MissingThreads;
  unsigned failures = 0;
  if (!isLegalThreadCount(*threads, capabilities))
    failures |= ThreadLimit;
  kernel.walk([&](GemmOp gemm) {
    auto lhs = gemm.getLhs().getType();
    auto rhs = gemm.getRhs().getType();
    std::optional<int64_t> m = evaluate(
        cast<gpu::PhysicalExprAttr>(
            lhs.getShape()[gemm.getTransposeLhs() ? 1 : 0]),
        config);
    std::optional<int64_t> n = evaluate(
        cast<gpu::PhysicalExprAttr>(
            rhs.getShape()[gemm.getTransposeRhs() ? 0 : 1]),
        config);
    if (m && n && !isLegalMmaWarpPartition(*m, *n, *threads))
      failures |= MmaPartition;
  });
  kernel.walk([&](SparseGemmOp gemm) {
    auto compressed = gemm.getCompressed().getType();
    auto metadata = gemm.getMetadata().getType();
    auto rhs = gemm.getRhs().getType();
    std::optional<int64_t> m = evaluate(
        cast<gpu::PhysicalExprAttr>(compressed.getShape()[
            gemm.getTransposeCompressed() ? 1 : 0]),
        config);
    std::optional<int64_t> n = evaluate(
        cast<gpu::PhysicalExprAttr>(
            rhs.getShape()[gemm.getTransposeRhs() ? 0 : 1]),
        config);
    std::optional<int64_t> compressedK = evaluate(
        cast<gpu::PhysicalExprAttr>(compressed.getShape()[
            gemm.getTransposeCompressed() ? 0 : 1]),
        config);
    std::optional<int64_t> metadataK = evaluate(
        cast<gpu::PhysicalExprAttr>(
            metadata.getShape()[gemm.getTransposeMetadata() ? 0 : 1]),
        config);
    std::optional<int64_t> rhsK = evaluate(
        cast<gpu::PhysicalExprAttr>(
            rhs.getShape()[gemm.getTransposeRhs() ? 1 : 0]),
        config);
    if (m && n && !isLegalMmaWarpPartition(*m, *n, *threads))
      failures |= MmaPartition;
    if (compressedK && metadataK && rhsK &&
        (*rhsK <= 0 || *rhsK % 16 != 0 || *compressedK * 2 != *rhsK ||
         *metadataK * 16 != *rhsK))
      failures |= SparseShape;
  });
  if (!exactSharedMemoryIsLegal(
          kernel, config,
          static_cast<uint64_t>(
              capabilities.getMaxDynamicSharedMemoryPerBlock())))
    failures |= SharedMemory;
  return failures;
}

LogicalResult diagnoseConfigurationFailures(func::FuncOp kernel, unsigned failures,
                                             StringRef message, DictionaryAttr row = {}) {
  auto diagnostic = kernel.emitError(message);
  if (row) diagnostic << "; configuration=" << row;
  diagnostic << "; typed reasons=";
  bool first = true;
  auto append = [&](StringRef reason) {
    if (!first) diagnostic << ",";
    diagnostic << reason;
    first = false;
  };
  if (failures & MissingThreads) append("missing_threads");
  if (failures & ThreadLimit) append("threads_per_block");
  if (failures & MmaPartition) append("mma_warp_partition");
  if (failures & SharedMemory) append("exact_shared_memory");
  if (failures & SparseShape) append("two_of_four_tile_shape");
  return failure();
}

SmallVector<int64_t> extentCandidates(func::FuncOp kernel, Attribute attribute) {
  auto extent = dyn_cast<gpu::PhysicalExprAttr>(attribute);
  if (!extent)
    return {};
  auto kind = extent.getKind();
  if (kind == gpu::PhysicalExprKind::Constant)
    return {extent.getValue()};
  if (kind != gpu::PhysicalExprKind::Parameter)
    return {};
  auto parameter = gpu::lookupParameter(kernel, extent.getParameterReference());
  return parameter ? SmallVector<int64_t>(parameter.getCandidates().asArrayRef())
                   : SmallVector<int64_t>();
}

LogicalResult materializeLegalConfigurations(func::FuncOp kernel) {
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities)
    return kernel.emitError(
        "TileLang legalization requires typed GPU capabilities");
  auto space = readConfigurationDomains(kernel);
  if (failed(space)) return failure();

  auto shared = space->configurations(gpu::ConfigurationStage::Shared);
  if (failed(shared)) return failure();
  SmallVector<DictionaryAttr> configs = std::move(*shared);
  Builder builder(kernel.getContext());
  for (gpu::ParameterAttr domain : space->declarations()) {
    if (domain.getPhase() != gpu::ConfigurationBindingPhase::Provider)
      continue;
    SmallVector<DictionaryAttr> expanded;
    for (DictionaryAttr base : configs)
      for (int64_t candidate : domain.getCandidates().asArrayRef()) {
        NamedAttrList bindings(base);
        bindings.set(domain.getName(), builder.getI64IntegerAttr(candidate));
        DictionaryAttr config = bindings.getDictionary(kernel.getContext());
        if (!llvm::is_contained(expanded, config))
          expanded.push_back(config);
      }
    configs = std::move(expanded);
  }

  SmallVector<DictionaryAttr> accepted;
  unsigned rejected = 0;
  for (DictionaryAttr config : configs) {
    unsigned failures =
        configurationFailures(kernel, space->declarations(), config, capabilities);
    if (failures) {
      rejected |= failures;
      continue;
    }
    accepted.push_back(config);
  }
  if (accepted.empty())
    return diagnoseConfigurationFailures(kernel, rejected,
        "all TileLang parameter candidates are provably illegal");
  return gpu::writeConfigurations(kernel, accepted, gpu::ConfigurationStage::Complete);
}

bool supportsThreads(func::FuncOp kernel, int64_t threads) {
  bool sawGemm = false;
  bool supported = true;
  kernel.walk([&](GemmOp gemm) {
    sawGemm = true;
    auto lhs = gemm.getLhs().getType();
    auto rhs = gemm.getRhs().getType();
    Attribute mExtent = lhs.getShape()[gemm.getTransposeLhs() ? 1 : 0];
    Attribute nExtent = rhs.getShape()[gemm.getTransposeRhs() ? 0 : 1];
    SmallVector<int64_t> mCandidates = extentCandidates(kernel, mExtent);
    SmallVector<int64_t> nCandidates = extentCandidates(kernel, nExtent);
    if (mCandidates.empty() || nCandidates.empty()) {
      supported = false;
      return;
    }
    bool hasLegalShape = false;
    for (int64_t m : mCandidates)
      for (int64_t n : nCandidates)
        hasLegalShape |= isLegalMmaWarpPartition(m, n, threads);
    if (!hasLegalShape)
      supported = false;
  });
  kernel.walk([&](SparseGemmOp gemm) {
    sawGemm = true;
    auto compressed = gemm.getCompressed().getType();
    auto rhs = gemm.getRhs().getType();
    Attribute mExtent = compressed.getShape()[
        gemm.getTransposeCompressed() ? 1 : 0];
    Attribute nExtent =
        rhs.getShape()[gemm.getTransposeRhs() ? 0 : 1];
    SmallVector<int64_t> mCandidates = extentCandidates(kernel, mExtent);
    SmallVector<int64_t> nCandidates = extentCandidates(kernel, nExtent);
    if (mCandidates.empty() || nCandidates.empty()) {
      supported = false;
      return;
    }
    bool hasLegalShape = false;
    for (int64_t m : mCandidates)
      for (int64_t n : nCandidates)
        hasLegalShape |= isLegalMmaWarpPartition(m, n, threads);
    if (!hasLegalShape)
      supported = false;
  });
  return !sawGemm || supported;
}

} // namespace

LogicalResult materializeLaunchConfiguration(
    func::FuncOp kernel, const gpu::TuningProfiles &profiles) {
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities)
    return kernel.emitError(
        "TileLang launch configuration requires typed GPU capabilities");
  SmallVector<int64_t> candidates;
  auto legalThreads = [&](int64_t threads) {
    return isLegalThreadCount(threads, capabilities) &&
           supportsThreads(kernel, threads);
  };
  bool smallPartition = !legalThreads(128) && !legalThreads(256);
  auto rows = profiles.get(tuningProfileSchema(), smallPartition ? "small_threads" : "threads",
                           kernel.getLoc());
  if (failed(rows))
    return failure();
  for (const auto &row : *rows)
    if (legalThreads(row[0]))
      candidates.push_back(row[0]);
  if (candidates.empty())
    return kernel.emitError(
        "TileLang provider found no legal thread count for every native GEMM shape");
  auto threads = gpu::getOrCreatePhysicalParameter(
      kernel, "THREADS", gpu::ParameterRole::ProviderThreads,
      gpu::ParameterCategory::Provider, 0, candidates);
  if (failed(threads))
    return failure();
  SmallVector<LaunchConfigOp> configs;
  kernel.getBody().front().walk(
      [&](LaunchConfigOp config) { configs.push_back(config); });
  if (configs.size() > 1)
    return kernel.emitError(
        "TileLang provider program has duplicate launch configurations");
  if (!configs.empty()) {
    auto declaration = gpu::queryParameter(configs.front().getThreads());
    return declaration && declaration.getReference() == *threads
               ? success()
               : configs.front().emitOpError(
                     "uses a conflicting TileLang thread parameter");
  }
  OpBuilder builder(&kernel.front(), kernel.front().begin());
  auto value = gpu::materializeParameter(builder, kernel.getLoc(), *threads);
  builder.create<LaunchConfigOp>(kernel.getLoc(), value.getResult());
  return success();
}

LogicalResult verifyTileLangProgram(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel) || failed(mlir::verify(module))) return failure();
  auto domains = readConfigurationDomains(*kernel);
  auto configurations = gpu::ParameterSpace::read(*kernel);
  if (failed(domains) || failed(configurations)) return failure();
  auto rows = configurations->configurations(gpu::ConfigurationStage::Complete);
  if (failed(rows) || failed(verifyTileLangKernel(*kernel))) return failure();
  auto capabilities = (*kernel)->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  for (DictionaryAttr row : *rows)
    if (unsigned failures = configurationFailures(*kernel, domains->declarations(), row, capabilities))
      return diagnoseConfigurationFailures(*kernel, failures,
          "final TileLang candidate violates the current native program", row);
  return success();
}

LogicalResult formNativeMemory(ModuleOp module) {
  if (failed(gpu::verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  return failed(kernel) ? failure() : bufferizeGPUProgram(*kernel);
}

LogicalResult configureNativeProgram(ModuleOp module) {
  auto profiles = gpu::TuningProfiles::from(module);
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(profiles) || failed(kernel) ||
      failed(materializeLaunchConfiguration(*kernel, *profiles)) ||
      failed(formPipelines(*kernel, *profiles)))
    return failure();
  return materializeLegalConfigurations(*kernel);
}

LogicalResult finalizeNativeProgram(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  (*kernel)->setAttr(lowerPredicatedLoadStoreAttr,
                     BoolAttr::get(module.getContext(), true));
  if (failed(gpu::eliminateCommonValues(module)) ||
      failed(verifyTileLangProgram(module)))
    return failure();
  (*kernel)->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::tilelang
