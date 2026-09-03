#include "Intent/Target/TileLang/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"
#include "PassDetail.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>

using namespace mlir;

namespace intent::tilelang {

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

struct ParameterDomain {
  std::string name;
  gpu::ParameterRole role;
  SmallVector<int64_t> candidates;
  bool provider = false;
  bool coverage = false;
};

using TileLangConfig = std::map<std::string, int64_t>;

std::optional<int64_t>
evaluate(gpu::PhysicalExprAttr expression, const TileLangConfig &config) {
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  if (kind == gpu::PhysicalExprKind::Constant)
    return expression.getValue();
  if (kind == gpu::PhysicalExprKind::Parameter) {
    auto found = config.find(expression.getSymbol().getValue().str());
    return found == config.end() ? std::nullopt
                                 : std::optional<int64_t>(found->second);
  }
  SmallVector<int64_t> operands;
  for (Attribute operand : expression.getOperands()) {
    std::optional<int64_t> value =
        evaluate(cast<gpu::PhysicalExprAttr>(operand), config);
    if (!value)
      return std::nullopt;
    operands.push_back(*value);
  }
  int64_t result = 0;
  if (kind == gpu::PhysicalExprKind::Add)
    return __builtin_add_overflow(operands[0], operands[1], &result)
               ? std::nullopt
               : std::optional<int64_t>(result);
  if (kind == gpu::PhysicalExprKind::Subtract)
    return __builtin_sub_overflow(operands[0], operands[1], &result)
               ? std::nullopt
               : std::optional<int64_t>(result);
  if (kind == gpu::PhysicalExprKind::Multiply)
    return __builtin_mul_overflow(operands[0], operands[1], &result)
               ? std::nullopt
               : std::optional<int64_t>(result);
  if (kind == gpu::PhysicalExprKind::CeilDiv && operands[0] >= 0 &&
      operands[1] > 0)
    return operands[0] / operands[1] +
           static_cast<int64_t>(operands[0] % operands[1] != 0);
  if (kind == gpu::PhysicalExprKind::FloorDiv && operands[0] >= 0 &&
      operands[1] > 0)
    return operands[0] / operands[1];
  if (kind == gpu::PhysicalExprKind::Minimum)
    return std::min(operands[0], operands[1]);
  if (kind == gpu::PhysicalExprKind::Maximum)
    return std::max(operands[0], operands[1]);
  return std::nullopt;
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
    AllocOp allocation, const TileLangConfig &config) {
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
                              const TileLangConfig &config,
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
                               ArrayRef<ParameterDomain> domains,
                               const TileLangConfig &config,
                               gpu::CapabilitiesAttr capabilities) {
  std::optional<int64_t> threads;
  for (const ParameterDomain &domain : domains) {
    if (domain.role != gpu::ParameterRole::ProviderThreads)
      continue;
    auto found = config.find(domain.name);
    if (found != config.end())
      threads = found->second;
  }
  if (!threads)
    return MissingThreads;
  unsigned failures = 0;
  if (*threads <= 0 || *threads > capabilities.getMaxThreadsPerBlock())
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

gpu::ParameterOp getOrCreateParameter(func::FuncOp kernel, StringRef name,
                                      gpu::ParameterRole role,
                                      ArrayRef<int64_t> candidates) {
  gpu::ParameterOp existing;
  bool duplicate = false;
  kernel.walk([&](gpu::ParameterOp parameter) {
    if (parameter.getParameter().getName().getValue() != name)
      return;
    if (existing)
      duplicate = true;
    else
      existing = parameter;
  });
  if (duplicate) {
    kernel.emitError("duplicates a TileLang physical parameter name");
    return {};
  }
  auto expectedCandidates =
      DenseI64ArrayAttr::get(kernel.getContext(), candidates);
  if (existing) {
    gpu::ParameterAttr schema = existing.getParameter();
    if (schema.getRole() != static_cast<uint32_t>(role) ||
        schema.getCandidates() != expectedCandidates) {
      existing.emitOpError(
          "physical parameter name is reused with a different role or candidate domain");
      return {};
    }
    return existing;
  }
  OpBuilder builder(&kernel.getBody().front(), kernel.getBody().front().begin());
  auto schema = gpu::ParameterAttr::get(
      kernel.getContext(), builder.getStringAttr(name),
      static_cast<uint32_t>(role),
      static_cast<uint32_t>(gpu::ParameterCategory::Provider),
      /*elementBitWidth=*/0, expectedCandidates);
  return builder.create<gpu::ParameterOp>(kernel.getLoc(), builder.getIndexType(),
                                          schema);
}

SmallVector<int64_t> extentCandidates(func::FuncOp kernel, Attribute attribute) {
  auto extent = dyn_cast<gpu::PhysicalExprAttr>(attribute);
  if (!extent)
    return {};
  auto kind = static_cast<gpu::PhysicalExprKind>(extent.getKind());
  if (kind == gpu::PhysicalExprKind::Constant)
    return {extent.getValue()};
  if (kind != gpu::PhysicalExprKind::Parameter)
    return {};
  SmallVector<int64_t> result;
  kernel.walk([&](gpu::ParameterOp parameter) {
    if (parameter.getParameter().getName() == extent.getSymbol()) {
      ArrayRef<int64_t> candidates =
          parameter.getParameter().getCandidates().asArrayRef();
      result.assign(candidates.begin(), candidates.end());
    }
  });
  return result;
}

LogicalResult materializeLegalConfigurations(func::FuncOp kernel) {
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities)
    return kernel.emitError(
        "TileLang legalization requires typed GPU capabilities");
  SmallVector<ParameterDomain> domains;
  llvm::StringSet<> names;
  bool invalid = false;
  bool sawThreads = false;
  kernel.walk([&](gpu::ParameterOp parameter) {
    gpu::ParameterAttr schema = parameter.getParameter();
    StringRef name = schema.getName().getValue();
    if (!names.insert(name).second) {
      parameter.emitOpError("duplicates a TileLang physical parameter name");
      invalid = true;
      return;
    }
    auto role = static_cast<gpu::ParameterRole>(schema.getRole());
    auto category = static_cast<gpu::ParameterCategory>(schema.getCategory());
    bool provider = category == gpu::ParameterCategory::Provider;
    bool coverage = parameter->hasAttr(gpu::coverageDimensionAttr);
    if (provider && role != gpu::ParameterRole::ProviderThreads &&
        role != gpu::ParameterRole::ProviderStages) {
      parameter.emitOpError(
          "TileLang program contains a foreign provider parameter");
      invalid = true;
      return;
    }
    sawThreads |= role == gpu::ParameterRole::ProviderThreads;
    domains.push_back({name.str(), role,
                       SmallVector<int64_t>(
                           schema.getCandidates().asArrayRef()),
                       provider, coverage});
  });
  if (invalid)
    return failure();
  if (!sawThreads)
    return kernel.emitError(
        "TileLang provider parameter domains have no thread binding");

  auto shared =
      kernel->getAttrOfType<ArrayAttr>(gpu::sharedConfigTuplesAttr);
  if (!shared || shared.empty())
    return kernel.emitError(
        "TileLang legalization requires shared config tuples");
  SmallVector<TileLangConfig> configs;
  for (Attribute attribute : shared) {
    auto tuple = dyn_cast<DictionaryAttr>(attribute);
    if (!tuple)
      return kernel.emitError("shared config tuple is malformed");
    TileLangConfig config;
    unsigned sharedParameters = 0;
    for (const ParameterDomain &domain : domains) {
      if (domain.provider || domain.coverage)
        continue;
      ++sharedParameters;
      auto value = tuple.getAs<IntegerAttr>(domain.name);
      if (!value ||
          !llvm::is_contained(domain.candidates, value.getInt()))
        return kernel.emitError(
                   "shared config tuple does not bind a TileLang kernel parameter: ")
               << domain.name;
      config[domain.name] = value.getInt();
    }
    if (tuple.size() != sharedParameters)
      return kernel.emitError(
          "shared config tuple contains a non-kernel binding");
    configs.push_back(std::move(config));
  }
  for (const ParameterDomain &domain : domains) {
    if (!domain.provider)
      continue;
    SmallVector<TileLangConfig> expanded;
    for (const TileLangConfig &base : configs)
      for (int64_t candidate : domain.candidates) {
        TileLangConfig config = base;
        config[domain.name] = candidate;
        if (!llvm::is_contained(expanded, config))
          expanded.push_back(std::move(config));
      }
    configs = std::move(expanded);
  }

  Builder builder(kernel.getContext());
  SmallVector<Attribute> encoded;
  unsigned rejected = 0;
  for (const TileLangConfig &config : configs) {
    unsigned failures =
        configurationFailures(kernel, domains, config, capabilities);
    if (failures) {
      rejected |= failures;
      continue;
    }
    SmallVector<NamedAttribute> bindings;
    for (const auto &[name, value] : config)
      bindings.push_back(
          builder.getNamedAttr(name, builder.getI64IntegerAttr(value)));
    encoded.push_back(builder.getDictionaryAttr(bindings));
  }
  if (encoded.empty()) {
    InFlightDiagnostic diagnostic =
        kernel.emitError("all TileLang parameter candidates are provably illegal");
    diagnostic << "; typed reasons=";
    bool first = true;
    auto appendReason = [&](StringRef reason) {
      if (!first)
        diagnostic << ",";
      first = false;
      diagnostic << reason;
    };
    if (rejected & MissingThreads)
      appendReason("missing_threads");
    if (rejected & ThreadLimit)
      appendReason("threads_per_block");
    if (rejected & MmaPartition)
      appendReason("mma_warp_partition");
    if (rejected & SharedMemory)
      appendReason("exact_shared_memory");
    if (rejected & SparseShape)
      appendReason("two_of_four_tile_shape");
    return failure();
  }
  kernel->setAttr(gpu::tileLangConfigsAttr, builder.getArrayAttr(encoded));
  return success();
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

LogicalResult materializeLaunchConfiguration(func::FuncOp kernel) {
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities)
    return kernel.emitError(
        "TileLang launch configuration requires typed GPU capabilities");
  SmallVector<int64_t> candidates;
  for (int64_t threads : {128, 256})
    if (threads <= capabilities.getMaxThreadsPerBlock() &&
        supportsThreads(kernel, threads))
      candidates.push_back(threads);
  if (candidates.empty())
    for (int64_t threads : {64, 32})
      if (threads <= capabilities.getMaxThreadsPerBlock() &&
          supportsThreads(kernel, threads)) {
        candidates.push_back(threads);
        break;
      }
  if (candidates.empty())
    return kernel.emitError(
        "TileLang provider found no legal thread count for every native GEMM shape");
  gpu::ParameterOp threads = getOrCreateParameter(
      kernel, "THREADS", gpu::ParameterRole::ProviderThreads, candidates);
  if (!threads)
    return failure();
  SmallVector<LaunchConfigOp> configs;
  kernel.getBody().front().walk(
      [&](LaunchConfigOp config) { configs.push_back(config); });
  if (configs.size() > 1)
    return kernel.emitError(
        "TileLang provider program has duplicate launch configurations");
  if (!configs.empty())
    return configs.front().getThreads() == threads.getResult()
               ? success()
               : configs.front().emitOpError(
                     "uses a conflicting TileLang thread parameter");
  OpBuilder builder(threads);
  builder.setInsertionPointAfter(threads);
  builder.create<LaunchConfigOp>(kernel.getLoc(), threads.getResult());
  return success();
}

LogicalResult verifyTileLangProgram(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  return failed(kernel) || failed(mlir::verify(module))
             ? failure()
             : verifyTileLangKernel(*kernel);
}

LogicalResult legalizeGPUProgram(ModuleOp module) {
  if (failed(gpu::verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel) || failed(bufferizeGPUProgram(*kernel)) ||
      failed(materializeLaunchConfiguration(*kernel)) ||
      failed(formPipelines(*kernel)) ||
      failed(materializeLegalConfigurations(*kernel)))
    return failure();
  (*kernel)->setAttr(lowerPredicatedLoadStoreAttr,
                     BoolAttr::get(module.getContext(), true));
  if (failed(verifyTileLangProgram(module)))
    return failure();
  (*kernel)->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::tilelang
