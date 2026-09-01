#include "Intent/Target/TileLang/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"
#include "PassDetail.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

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
  if (kind == gpu::PhysicalExprKind::Add)
    return operands[0] + operands[1];
  if (kind == gpu::PhysicalExprKind::Subtract)
    return operands[0] - operands[1];
  if (kind == gpu::PhysicalExprKind::Multiply)
    return operands[0] * operands[1];
  if (kind == gpu::PhysicalExprKind::CeilDiv && operands[1] > 0)
    return (operands[0] + operands[1] - 1) / operands[1];
  if (kind == gpu::PhysicalExprKind::FloorDiv && operands[1] > 0)
    return operands[0] / operands[1];
  if (kind == gpu::PhysicalExprKind::Minimum)
    return std::min(operands[0], operands[1]);
  if (kind == gpu::PhysicalExprKind::Maximum)
    return std::max(operands[0], operands[1]);
  return std::nullopt;
}

bool configurationIsLegal(func::FuncOp kernel,
                          ArrayRef<ParameterDomain> domains,
                          const TileLangConfig &config) {
  std::optional<int64_t> threads;
  for (const ParameterDomain &domain : domains) {
    if (domain.role != gpu::ParameterRole::ProviderThreads)
      continue;
    auto found = config.find(domain.name);
    if (found != config.end())
      threads = found->second;
  }
  if (!threads)
    return false;
  bool legal = true;
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
    if (m && n)
      legal &= isLegalMmaWarpPartition(*m, *n, *threads);
  });
  return legal;
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
  for (const TileLangConfig &config : configs) {
    if (!configurationIsLegal(kernel, domains, config))
      continue;
    SmallVector<NamedAttribute> bindings;
    for (const auto &[name, value] : config)
      bindings.push_back(
          builder.getNamedAttr(name, builder.getI64IntegerAttr(value)));
    encoded.push_back(builder.getDictionaryAttr(bindings));
  }
  if (encoded.empty())
    return kernel.emitError(
        "all TileLang parameter candidates violate typed MMA warp partition legality");
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
  return !sawGemm || supported;
}

} // namespace

LogicalResult materializeLaunchConfiguration(func::FuncOp kernel) {
  SmallVector<int64_t> candidates;
  for (int64_t threads : {128, 256})
    if (supportsThreads(kernel, threads))
      candidates.push_back(threads);
  if (candidates.empty())
    for (int64_t threads : {64, 32})
      if (supportsThreads(kernel, threads)) {
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
      failed(materializeLegalConfigurations(*kernel)) ||
      failed(verifyTileLangProgram(module)))
    return failure();
  (*kernel)->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::tilelang
