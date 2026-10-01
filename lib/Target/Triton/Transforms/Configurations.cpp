#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "Configurations.h"
#include "ConfigurationFacts.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/Configurations.h"
#include "Intent/Target/Triton/IR/Configuration.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Resources.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "Intent/Target/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <functional>
#include <limits>

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
    if (leaf.getKind() != static_cast<uint32_t>(gpu::PhysicalExprKind::Parameter))
      return std::nullopt;
    auto value = bindings ? bindings.getAs<IntegerAttr>(leaf.getSymbol()) : IntegerAttr();
    return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
  });
}

bool isTritonFragmentExtent(Attribute attribute) {
  auto expression = dyn_cast<gpu::PhysicalExprAttr>(attribute);
  if (!expression)
    return false;
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  return kind != gpu::PhysicalExprKind::ScalarABI &&
         isTritonExpression(expression) &&
         llvm::all_of(expression.getOperands(), isTritonFragmentExtent);
}

bool isTritonExpression(gpu::PhysicalExprAttr expression) {
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
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

bool fragmentFitsTritonTensor(gpu::FragmentType fragment,
                              DictionaryAttr config) {
  auto bound = gpu::checkFragmentFootprint(
      fragment, maxTritonTensorElements, 1, [&](gpu::PhysicalExprAttr leaf) {
        return evaluateCompileTimeExpression(leaf, config);
      });
  // ABI-dependent extents retain the specialization assertions below.
  return bound != gpu::FootprintBound::Exceeds &&
         bound != gpu::FootprintBound::Invalid;
}

SmallVector<gpu::FragmentType> collectiveFragments(func::FuncOp kernel) {
  SmallVector<gpu::FragmentType> fragments;
  kernel.walk([&](Operation *operation) {
    auto collect = [&](Value value) {
      if (auto fragment = dyn_cast<gpu::FragmentType>(value.getType());
          fragment && !llvm::is_contained(fragments, fragment))
        fragments.push_back(fragment);
    };
    if (isa<gpu::ContractOp, gpu::ScaledContractOp, gpu::SparseContractOp>(operation)) {
      for (Value result : operation->getResults())
        collect(result);
      return;
    }
    auto store = dyn_cast<gpu::StoreOp>(operation);
    bool workspaceStore = store && isa<gpu::BufferType>(store.getResource().getType());
    if (store)
      if (auto argument = dyn_cast<BlockArgument>(store.getResource());
          argument && argument.getOwner() == &kernel.front()) {
        auto kind = kernel.getArgAttrOfType<StringAttr>(
            argument.getArgNumber(), gpu::abiKindAttr);
        workspaceStore |= kind && kind.getValue() == "workspace";
      }
    if (!isa<gpu::ReduceOp, gpu::ScanOp, ReduceOp, ScanOp>(operation) &&
        !workspaceStore)
      return;
    if (workspaceStore)
      collect(store.getValue());
    else
      for (Value operand : operation->getOperands())
        collect(operand);
  });
  return fragments;
}

LogicalResult materializeDeferredResourceBounds(func::FuncOp kernel) {
  NamedAttrList knownBindings;
  Builder attributes(kernel.getContext());
  gpu::ParameterAttr warpParameter;
  kernel.walk([&](gpu::ParameterOp parameter) {
    auto schema = parameter.getParameter();
    if (schema.getRole() ==
        static_cast<uint32_t>(gpu::ParameterRole::ProviderWarps))
      warpParameter = schema;
    if (schema.getCategory() !=
            static_cast<uint32_t>(gpu::ParameterCategory::Coverage) &&
        !parameter->hasAttr(gpu::coverageDimensionAttr))
      knownBindings.set(schema.getName(),
          attributes.getI64IntegerAttr(schema.getCandidates().asArrayRef().front()));
  });
  DictionaryAttr known = knownBindings.getDictionary(kernel.getContext());
  const gpu::FragmentResourceAnalysis resources(kernel);
  SmallVector<gpu::PhysicalExprAttr> bounds;
  for (gpu::FragmentType fragment : resources.valueTypes()) {
    gpu::PhysicalExprAttr elements;
    for (Attribute attribute : fragment.getShape()) {
      auto extent = cast<gpu::PhysicalExprAttr>(attribute);
      if (extent.getKind() ==
              static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) &&
          extent.getValue() == 1)
        continue;
      elements = !elements ? extent : gpu::PhysicalExprAttr::get(
          kernel.getContext(),
          static_cast<uint32_t>(gpu::PhysicalExprKind::Multiply), 0,
          StringAttr::get(kernel.getContext(), ""),
          ArrayAttr::get(kernel.getContext(), {elements, extent}));
    }
    if (!elements || evaluateCompileTimeExpression(elements, known))
      continue;
    if (!isTritonFragmentExtent(elements))
      return kernel.emitError("Triton tensor bounds require constexpr fragment extents");
    if (!llvm::is_contained(bounds, elements))
      bounds.push_back(elements);
  }
  kernel.getContext()->getOrLoadDialect<cf::ControlFlowDialect>();
  OpBuilder builder = OpBuilder::atBlockBegin(&kernel.front());
  auto assertBound = [&](gpu::PhysicalExprAttr expression, int64_t limit,
                         StringRef message) {
    Value maximum = builder.create<arith::ConstantIndexOp>(kernel.getLoc(), limit);
    Value count = builder.create<gpu::PhysicalExprOp>(
        kernel.getLoc(), builder.getIndexType(), expression);
    Value valid = builder.create<gpu::CompareOp>(
        kernel.getLoc(), builder.getI1Type(), count, maximum, ComparePredicate::Le);
    builder.create<cf::AssertOp>(kernel.getLoc(), valid, message);
  };
  for (gpu::PhysicalExprAttr elements : bounds)
    assertBound(elements, maxTritonTensorElements,
                "Triton block tensor exceeds the maximum element count");
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  SmallVector<ValueRange> reductionSources;
  kernel.walk([&](ReduceOp reduce) {
    reductionSources.push_back(reduce.getSources());
  });
  gpu::materializeDeferredReductionBounds(kernel, reductionSources);
  if (capabilities && capabilities.getRegistersPerUnit() > 0) {
    if (warpParameter && warpParameter.getCandidates().size() > 1) {
      auto expression = [&](gpu::PhysicalExprKind kind, int64_t value,
                            ArrayRef<Attribute> operands) {
        return gpu::PhysicalExprAttr::get(
            kernel.getContext(), static_cast<uint32_t>(kind), value,
            builder.getStringAttr(""), builder.getArrayAttr(operands));
      };
      auto constant = [&](int64_t value) {
        return expression(gpu::PhysicalExprKind::Constant, value, {});
      };
      auto warps = gpu::PhysicalExprAttr::get(
          kernel.getContext(),
          static_cast<uint32_t>(gpu::PhysicalExprKind::Parameter), 0,
          warpParameter.getName(), builder.getArrayAttr({}));
      int64_t maximumWarps =
          *llvm::max_element(warpParameter.getCandidates().asArrayRef());
      auto belowMaximum = expression(gpu::PhysicalExprKind::Subtract, 0,
                                     {warps, constant(maximumWarps)});
      auto nominalBudget = expression(gpu::PhysicalExprKind::Multiply, 0,
                                      {warps, constant(32 * 255)});
      // Apply the same per-fragment policy after specialization binds the
      // physical extents. The widest supplied option may spill.
      auto budget = expression(gpu::PhysicalExprKind::Select, 0,
                               {belowMaximum, nominalBudget,
                                constant(std::numeric_limits<int64_t>::max())});
      for (gpu::FragmentType fragment : collectiveFragments(kernel)) {
        auto footprint = gpu::fragmentRegisterFootprint(fragment);
        if (evaluateCompileTimeExpression(footprint))
          continue;
        Value count = builder.create<gpu::PhysicalExprOp>(
            kernel.getLoc(), builder.getIndexType(), footprint);
        Value maximum = builder.create<gpu::PhysicalExprOp>(
            kernel.getLoc(), builder.getIndexType(), budget);
        Value valid = builder.create<gpu::CompareOp>(
            kernel.getLoc(), builder.getI1Type(), count, maximum,
            ComparePredicate::Le);
        builder.create<cf::AssertOp>(
            kernel.getLoc(), valid,
            "Triton collective fragment exceeds the nominal per-thread register budget");
      }
    }
  }
  return success();
}

bool descriptorFragmentFits(gpu::FragmentType fragment,
                            DictionaryAttr config, int64_t stages,
                            int64_t pipelineBlockAlignment,
                            const llvm::StringMap<SmallVector<int64_t>>
                                &parameterDomains,
                            const llvm::StringSet<> &coverageParameters) {
  std::optional<int64_t> elementBytes =
      descriptorElementBytes(fragment.getElementType());
  if (!elementBytes)
    return false;
  __int128 elements = 1;
  __int128 concreteElements = 1;
  bool concrete = true;
  int64_t minimumLastExtent = std::numeric_limits<int64_t>::max();
  bool runtimeGuardsLastExtent = false;
  bool runtimeGuardsElementCount = false;
  for (auto [axis, extent] : llvm::enumerate(fragment.getShape())) {
    auto expression = cast<gpu::PhysicalExprAttr>(extent);
    SmallVector<int64_t> values;
    if (std::optional<int64_t> value =
            evaluateCompileTimeExpression(expression, config)) {
      values.push_back(*value);
      concreteElements *= *value;
    } else if (static_cast<gpu::PhysicalExprKind>(expression.getKind()) ==
               gpu::PhysicalExprKind::Parameter) {
      auto domain = parameterDomains.find(expression.getSymbol().getValue());
      if (domain == parameterDomains.end())
        return false;
      concrete = false;
      values.append(domain->second.begin(), domain->second.end());
      if (axis + 1 == fragment.getShape().size())
        runtimeGuardsLastExtent =
            coverageParameters.contains(expression.getSymbol().getValue());
      runtimeGuardsElementCount |=
          coverageParameters.contains(expression.getSymbol().getValue());
    } else {
      return false;
    }
    int64_t maximum = 0;
    int64_t minimum = std::numeric_limits<int64_t>::max();
    for (int64_t value : values) {
      if (value <= 0 || !llvm::isPowerOf2_64(value))
        return false;
      maximum = std::max(maximum, value);
      minimum = std::min(minimum, value);
    }
    elements *= maximum;
    if (axis + 1 == fragment.getShape().size())
      minimumLastExtent = minimum;
    if (elements > maxTritonTensorElements && !runtimeGuardsElementCount)
      return false;
  }
  // Triton's canPipelineTMALoad requires each shared stage to begin at a
  // 128-byte boundary. Small legal descriptor tiles can otherwise produce a
  // misaligned second buffer in a multistage pipeline.
  if (stages > 1 && concrete &&
      (concreteElements * *elementBytes) % pipelineBlockAlignment != 0)
    return false;
  return minimumLastExtent != std::numeric_limits<int64_t>::max() &&
         (runtimeGuardsLastExtent ||
          minimumLastExtent * *elementBytes >= 16);
}

LogicalResult materializeLegalConfigs(func::FuncOp kernel,
                                      TensorDescriptorChoiceOp descriptorChoice,
                                      ArrayRef<TritonLocalOptions> localOptions) {
  auto space = gpu::PhysicalParameterSpace::read(kernel);
  if (failed(space))
    return failure();
  auto schema = ConfigurationSchema::read(kernel);
  if (failed(schema)) return failure();
  auto shared = space->sharedConfigurations();
  if (failed(shared))
    return failure();
  llvm::StringMap<SmallVector<int64_t>> parameterDomains;
  llvm::StringSet<> coverageParameters;
  for (const auto &domain : space->domains()) {
    parameterDomains[domain.name().getValue()] =
        SmallVector<int64_t>(domain.candidates());
    if (domain.coverage)
      coverageParameters.insert(domain.name().getValue());
  }
  const auto *warps = space->find(gpu::ParameterRole::ProviderWarps);
  const auto *stages = space->find(gpu::ParameterRole::ProviderStages);
  const auto *ctas = space->find(gpu::ParameterRole::ProviderCTAs);
  if (!warps || !stages || !ctas)
    return kernel.emitError("Triton provider parameter domains are incomplete");
  SmallVector<DictionaryAttr> configs;
  Builder builder(kernel.getContext());
  int64_t maximumWarps = *llvm::max_element(warps->candidates());
  SmallVector<gpu::PhysicalExprAttr> collectiveFootprints;
  for (gpu::FragmentType fragment : collectiveFragments(kernel)) {
    auto footprint = gpu::fragmentRegisterFootprint(fragment);
    if (!llvm::is_contained(collectiveFootprints, footprint))
      collectiveFootprints.push_back(footprint);
  }
  for (DictionaryAttr tuple : *shared) {
    for (const TritonLocalOptions &options : localOptions) {
      int64_t formCount = descriptorChoice ? 2 : 1;
      for (int64_t form = 0; form < formCount; ++form) {
        NamedAttrList bindings(tuple);
        if (descriptorChoice)
          bindings.set(descriptorChoice.getConfigParameter(), builder.getI64IntegerAttr(form));
        bindings.set(schema->warps, builder.getI64IntegerAttr(options.warps));
        bindings.set(schema->stages, builder.getI64IntegerAttr(options.stages));
        bindings.set(schema->ctas, builder.getI64IntegerAttr(options.ctas));
        DictionaryAttr config = bindings.getDictionary(kernel.getContext());
        if (!llvm::is_contained(configs, config)) configs.push_back(config);
      }
    }
  }

  const gpu::FragmentResourceAnalysis resources(kernel);
  SmallVector<gpu::PhysicalExprAttr> rangeExtents;
  SmallVector<SmallVector<gpu::PhysicalExprAttr>> reductionFootprints;
  struct DescriptorConstraint {
    gpu::FragmentType fragment;
    uint64_t alignment;
  };
  SmallVector<DescriptorConstraint> descriptors;
  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  bool validRanges = true;
  kernel.walk([&](Operation *operation) {
    if (auto range = dyn_cast<gpu::MakeRangeOp>(operation)) {
      auto fragment = dyn_cast<gpu::FragmentType>(range.getResult().getType());
      if (!fragment || fragment.getShape().size() != 1)
        validRanges = false;
      else {
        auto extent = cast<gpu::PhysicalExprAttr>(fragment.getShape()[0]);
        if (!llvm::is_contained(rangeExtents, extent))
          rangeExtents.push_back(extent);
      }
    }
    if (auto reduce = dyn_cast<gpu::ReduceOp>(operation);
        reduce && capabilities && capabilities.getRegistersPerUnit() > 0) {
      SmallVector<gpu::PhysicalExprAttr> footprints;
      for (Value source : reduce.getSources())
        if (auto footprint = gpu::reductionRegisterFootprint(ValueRange{source}, kernel))
          footprints.push_back(footprint);
      reductionFootprints.push_back(std::move(footprints));
    }
    auto descriptorConstraint = [&](Value descriptor, gpu::FragmentType fragment) {
      descriptors.push_back({fragment, descriptor.getDefiningOp<TensorDescriptorOp>()
                                          .getPipelineBlockAlignment()});
    };
    if (auto load = dyn_cast<DescriptorLoadOp>(operation))
      descriptorConstraint(load.getDescriptor(), load.getResult().getType());
    else if (auto store = dyn_cast<DescriptorStoreOp>(operation))
      descriptorConstraint(store.getDescriptor(), store.getValue().getType());
  });
  if (!validRanges)
    return kernel.emitError("Triton ranges require one-dimensional fragments");
  auto fitsReductionBudget = [&](ArrayRef<gpu::PhysicalExprAttr> footprints,
                                DictionaryAttr config) {
    __int128 registers = 0;
    for (auto footprint : footprints) {
      auto size = evaluateCompileTimeExpression(footprint, config);
      if (!size)
        return true;
      registers += *size;
      if (registers > capabilities.getRegistersPerUnit())
        return false;
    }
    return true;
  };

  SmallVector<DictionaryAttr> accepted;
  for (DictionaryAttr config : configs) {
    int64_t configWarps = config.getAs<IntegerAttr>(schema->warps).getInt();
    int64_t configStages = config.getAs<IntegerAttr>(schema->stages).getInt();
    int64_t configCTAs = config.getAs<IntegerAttr>(schema->ctas).getInt();
    if (configWarps <= 0 || configStages <= 0 || configCTAs <= 0)
      return kernel.emitError("Triton provider parameter domains are incomplete");
    // Apply the existing per-fragment policy after binding the shared tuple.
    // ABI-dependent extents retain their deferred specialization assertion.
    if (configWarps < maximumWarps &&
        llvm::any_of(collectiveFootprints, [&](gpu::PhysicalExprAttr footprint) {
          auto words = evaluateCompileTimeExpression(footprint, config);
          return words && *words > configWarps * 32 * 255;
        }))
      continue;
    if (llvm::any_of(reductionFootprints, [&](const auto &footprints) {
          return !fitsReductionBudget(footprints, config);
        }) || llvm::any_of(rangeExtents, [&](gpu::PhysicalExprAttr expression) {
          auto extent = evaluateCompileTimeExpression(expression, config);
          return extent && (*extent <= 0 || !llvm::isPowerOf2_64(*extent));
        }) || llvm::any_of(resources.valueTypes(), [&](gpu::FragmentType fragment) {
          return !fragmentFitsTritonTensor(fragment, config);
        }))
      continue;
    bool descriptorConfig = descriptorChoice &&
        config.getAs<IntegerAttr>(descriptorChoice.getConfigParameter()).getInt() != 0;
    if (descriptorConfig && llvm::any_of(descriptors, [&](const auto &constraint) {
          return !descriptorFragmentFits(constraint.fragment, config, configStages,
              constraint.alignment, parameterDomains, coverageParameters);
        }))
      continue;
    accepted.push_back(config);
  }
  if (accepted.empty())
    return kernel.emitError(
        "all Triton parameter candidates violate typed fragment legality or the reduction register budget");
  return gpu::writeConfigurations(kernel, accepted, gpu::ConfigurationStage::Complete);
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
  OpBuilder builder(&kernel.getBody().front(), kernel.getBody().front().begin());
  llvm::StringSet<> names;
  kernel.walk([&](gpu::ParameterOp parameter) {
    names.insert(parameter.getParameter().getName().getValue());
  });
  auto declareProviderParameter = [&](StringRef name, gpu::ParameterRole role,
                                      ArrayRef<int64_t> candidates) -> LogicalResult {
    if (names.contains(name))
      return failure();
    auto schema = gpu::ParameterAttr::get(
        kernel.getContext(), builder.getStringAttr(name),
        static_cast<uint32_t>(role),
        static_cast<uint32_t>(gpu::ParameterCategory::Provider),
        /*elementBitWidth=*/0,
        DenseI64ArrayAttr::get(kernel.getContext(), candidates));
    builder.create<gpu::ParameterOp>(kernel.getLoc(), builder.getIndexType(),
                                     schema);
    names.insert(name);
    return success();
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
      loop->setAttr(loopStagesAttr, stages->getParameter());
  }
  return localOptions;
}

} // namespace intent::triton
