#include "Configurations.h"
#include "ConfigurationFacts.h"
#include "ConfigurationRequirements.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/Resources.h"
#include "Intent/Target/Triton/IR/Configuration.h"
#include "Intent/Target/Triton/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
namespace intent::triton {
const gpu::TuningProfileSchema &tuningProfileSchema() {
  static const StringRef columns[] = {"warps", "stages", "ctas"};
  static const gpu::TuningProfileSchema schema{"triton", columns};
  return schema;
}

namespace {
StringRef localOptionsFamily(ArrayRef<gpu::ParameterCategory> categories,
                             bool twoAxisPointwise,
                             StringRef recurrentContractionFamily,
                             bool fp32Contractions) {
  if (llvm::is_contained(categories,
                         gpu::ParameterCategory::RegionReduction))
    return "region_reduction";
  if (llvm::is_contained(categories,
                         gpu::ParameterCategory::RegionContraction))
    return "region_contraction";
  if (llvm::is_contained(categories,
                         gpu::ParameterCategory::PersistentContraction)) {
    return recurrentContractionFamily.empty() ? "persistent_contraction"
                                               : recurrentContractionFamily;
  }
  if (llvm::is_contained(categories, gpu::ParameterCategory::Contraction)) {
    if (fp32Contractions)
      return "contraction_f32";
    return recurrentContractionFamily.empty() ? "contraction"
                                               : recurrentContractionFamily;
  }
  if (llvm::is_contained(categories, gpu::ParameterCategory::Histogram))
    return "histogram";
  if (llvm::is_contained(categories, gpu::ParameterCategory::Reduction) ||
      llvm::is_contained(categories, gpu::ParameterCategory::Scan))
    return "reduction_scan";
  if (twoAxisPointwise)
    return "pointwise_two_axis";
  return "pointwise";
}

} // namespace

std::optional<int64_t> evaluateCompileTimeExpression(
    gpu::PhysicalExprAttr expression, DictionaryAttr bindings) {
  return gpu::evaluatePhysicalExpression(expression, [&](gpu::PhysicalExprAttr leaf)
      -> std::optional<int64_t> {
    if (leaf.getKind() != gpu::PhysicalExprKind::Parameter)
      return std::nullopt;
    auto value = bindings ? bindings.getAs<IntegerAttr>(leaf.getParameterReference().getName()) : IntegerAttr();
    return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
  });
}

bool isTritonFragmentExtent(Attribute attribute) {
  auto expression = dyn_cast<gpu::PhysicalExprAttr>(attribute);
  if (!expression)
    return false;
  auto kind = expression.getKind();
  return kind != gpu::PhysicalExprKind::ScalarABI &&
         isTritonExpression(expression) &&
         llvm::all_of(expression.getOperands(), isTritonFragmentExtent);
}

bool isTritonExpression(gpu::PhysicalExprAttr expression) {
  auto kind = expression.getKind();
  switch (kind) {
  case gpu::PhysicalExprKind::Constant:
  case gpu::PhysicalExprKind::Parameter:
  case gpu::PhysicalExprKind::Dimension:
  case gpu::PhysicalExprKind::ScalarABI:
    return expression.getOperands().empty();
  case gpu::PhysicalExprKind::Add:
  case gpu::PhysicalExprKind::Multiply:
  case gpu::PhysicalExprKind::CeilDiv:
  case gpu::PhysicalExprKind::Minimum:
  case gpu::PhysicalExprKind::Subtract:
  case gpu::PhysicalExprKind::FloorDiv:
  case gpu::PhysicalExprKind::Maximum:
    if (expression.getOperands().size() != 2)
      return false;
    break;
  case gpu::PhysicalExprKind::Select:
    if (expression.getOperands().size() != 3)
      return false;
    break;
  case gpu::PhysicalExprKind::NextPowerOfTwo:
    if (expression.getOperands().size() != 1)
      return false;
    break;
  default:
    return false;
  }
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isTritonExpression(cast<gpu::PhysicalExprAttr>(operand));
  });
}

std::optional<int64_t> descriptorElementBytes(Type type) {
  unsigned bitWidth = type.getIntOrFloatBitWidth();
  if (bitWidth < 8 || bitWidth % 8 != 0)
    return std::nullopt;
  return bitWidth / 8;
}

LogicalResult materializeLegalConfigs(func::FuncOp kernel,
                                      TensorDescriptorChoiceOp descriptorChoice,
                                      ArrayRef<TritonLocalOptions> localOptions) {
  auto space = gpu::ParameterSpace::read(kernel);
  if (failed(space))
    return failure();
  auto schema = ConfigurationSchema::read(kernel);
  if (failed(schema)) return failure();
  auto shared = space->configurations(gpu::ConfigurationStage::Shared);
  if (failed(shared))
    return failure();
  auto warps = space->find(gpu::ParameterRole::ProviderWarps);
  auto stages = space->find(gpu::ParameterRole::ProviderStages);
  auto ctas = space->find(gpu::ParameterRole::ProviderCTAs);
  if (!warps || !stages || !ctas)
    return kernel.emitError("Triton provider parameter domains are incomplete");
  SmallVector<DictionaryAttr> configs;
  Builder builder(kernel.getContext());
  for (DictionaryAttr tuple : *shared) {
    for (const TritonLocalOptions &options : localOptions) {
      int64_t formCount = descriptorChoice ? 2 : 1;
      for (int64_t form = 0; form < formCount; ++form) {
        NamedAttrList bindings(tuple);
        if (descriptorChoice)
          bindings.set(descriptorChoice.getConfigParameter().getName(), builder.getI64IntegerAttr(form));
        bindings.set(schema->warps, builder.getI64IntegerAttr(options.warps));
        bindings.set(schema->stages, builder.getI64IntegerAttr(options.stages));
        bindings.set(schema->ctas, builder.getI64IntegerAttr(options.ctas));
        DictionaryAttr config = bindings.getDictionary(kernel.getContext());
        if (!llvm::is_contained(configs, config)) configs.push_back(config);
      }
    }
  }

  auto requirements = collectConfigurationRequirements(kernel);
  if (failed(requirements)) return failure();
  auto filtered = gpu::filterConfigurationRequirements(kernel, configs, *requirements);
  if (failed(filtered)) return failure();
  return gpu::writeConfigurations(kernel, *filtered, gpu::ConfigurationStage::Complete, *requirements);
}

FailureOr<SmallVector<TritonLocalOptions>> declareProviderOptions(
    func::FuncOp kernel, const gpu::TuningProfiles &profiles,
    bool requiresCtaSynchronization, ArrayRef<scf::ForOp> loadPipelineLoops) {
  const ProgramConfigurationFacts facts(kernel);
  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  bool blackwell = capabilities.getComputeCapabilityMajor() == 10 ||
                   capabilities.getComputeCapabilityMajor() == 12;
  StringRef recurrentContractionFamily;
  if (facts.recurrentContraction) {
    if (blackwell)
      recurrentContractionFamily = "blackwell_recurrent_contraction";
    else if (capabilities.getComputeCapabilityMajor() == 9)
      recurrentContractionFamily = "hopper_recurrent_contraction";
  }
  auto rows = profiles.get(tuningProfileSchema(), localOptionsFamily(
      facts.categories, facts.twoAxisPointwise, recurrentContractionFamily,
      facts.hasContraction && facts.allContractionsF32), kernel.getLoc());
  if (failed(rows))
    return failure();
  bool pipelineStagesAffectProgram = facts.pipelineStagesAffectProgram || !loadPipelineLoops.empty();
  SmallVector<TritonLocalOptions> localOptions;
  SmallVector<int64_t> warpDomain, stageDomain, ctaDomain;
  auto isDeviceOption = [&](int64_t warps, int64_t stages, int64_t ctas) {
    return isLegalDeviceOption(gpu::ParameterRole::ProviderWarps, warps,
                               capabilities, requiresCtaSynchronization) &&
           isLegalDeviceOption(gpu::ParameterRole::ProviderStages, stages,
                               capabilities, requiresCtaSynchronization) &&
           isLegalDeviceOption(gpu::ParameterRole::ProviderCTAs, ctas,
                               capabilities, requiresCtaSynchronization);
  };
  for (const auto &row : *rows) {
    int64_t warps = row[0], stages = row[1], ctas = row[2];
    if (facts.straightLinePointwise)
      stages = 1;
    if (!isDeviceOption(warps, stages, ctas))
      continue;
    if (llvm::any_of(localOptions, [&](const TritonLocalOptions &option) {
          // Kernel-level stages pipeline dot producers. Ordinary loads need
          // an explicit tl.range stage binding; keep one supplied setting
          // when the current program has no such pipeline producer.
          return option.warps == warps &&
                 (!pipelineStagesAffectProgram || option.stages == stages) &&
                 option.ctas == ctas;
        }))
      continue;
    localOptions.push_back({warps, stages, ctas});
    if (!llvm::is_contained(warpDomain, warps)) warpDomain.push_back(warps);
    if (!llvm::is_contained(stageDomain, stages)) stageDomain.push_back(stages);
    if (!llvm::is_contained(ctaDomain, ctas)) ctaDomain.push_back(ctas);
  }
  if (localOptions.empty())
    return kernel.emitError("Triton tuning profile has no legal provider options");
  Builder builder(kernel.getContext());
  auto declareProviderParameter = [&](StringRef name, gpu::ParameterRole role,
                                      ArrayRef<int64_t> candidates) -> LogicalResult {
    return success(succeeded(gpu::getOrCreatePhysicalParameter(
        kernel, name, role, gpu::ParameterCategory::Provider, 0, candidates)));
  };
  if (failed(declareProviderParameter("NUM_WARPS", gpu::ParameterRole::ProviderWarps,
                                      warpDomain)) ||
      failed(declareProviderParameter("NUM_STAGES", gpu::ParameterRole::ProviderStages,
                                      stageDomain)) ||
      failed(declareProviderParameter("NUM_CTAS", gpu::ParameterRole::ProviderCTAs,
                                      ctaDomain)))
    return kernel.emitError("Triton provider parameter name is already owned");
  if (!loadPipelineLoops.empty()) {
    auto stages =
        gpu::queryParameterBySymbol(kernel, builder.getStringAttr("NUM_STAGES"));
    if (failed(stages))
      return failure();
    for (scf::ForOp loop : loadPipelineLoops)
      loop->setAttr(loopStagesAttr, stages->getReference());
  }
  return localOptions;
}

} // namespace intent::triton
