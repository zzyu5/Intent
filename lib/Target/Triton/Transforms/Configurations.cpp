#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "Configurations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
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

bool valueDependsOn(Value value, Value root, scf::ForOp owner,
                    llvm::SmallDenseSet<Value, 32> &visited) {
  if (value == root)
    return true;
  if (!visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition || !owner->isProperAncestor(definition))
    return false;
  if (auto nested = dyn_cast<scf::ForOp>(definition)) {
    auto result = dyn_cast<OpResult>(value);
    auto yield = dyn_cast<scf::YieldOp>(nested.getBody()->getTerminator());
    if (result && yield && result.getResultNumber() < nested.getInitArgs().size()) {
      unsigned index = result.getResultNumber();
      if (valueDependsOn(nested.getInitArgs()[index], root, owner, visited) ||
          valueDependsOn(yield.getOperand(index), root, owner, visited))
        return true;
    }
  }
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueDependsOn(operand, root, owner, visited);
  });
}

namespace {

bool valueDependsOnNestedContract(Value value, scf::ForOp owner,
                                  llvm::SmallDenseSet<Value, 32> &visited) {
  if (!visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition || !owner->isProperAncestor(definition))
    return false;
  if (isa<gpu::ContractOp>(definition))
    return true;
  if (auto nested = dyn_cast<scf::ForOp>(definition)) {
    auto result = dyn_cast<OpResult>(value);
    auto yield = dyn_cast<scf::YieldOp>(nested.getBody()->getTerminator());
    if (result && yield && result.getResultNumber() < nested.getInitArgs().size()) {
      unsigned index = result.getResultNumber();
      if (valueDependsOnNestedContract(nested.getInitArgs()[index], owner,
                                       visited) ||
          valueDependsOnNestedContract(yield.getOperand(index), owner, visited))
        return true;
    }
  }
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueDependsOnNestedContract(operand, owner, visited);
  });
}

bool isDirectContractionAccumulator(Value value, Value carry, scf::ForOp owner,
                                    llvm::SmallDenseSet<Value, 8> &visited) {
  if (!visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (auto contract = dyn_cast_or_null<gpu::ContractOp>(definition)) {
    if (contract.getAccumulator() != carry)
      return false;
    llvm::SmallDenseSet<Value, 32> dependencies;
    return !valueDependsOn(contract.getLhs(), carry, owner, dependencies) &&
           !valueDependsOn(contract.getRhs(), carry, owner, dependencies);
  }
  auto nested = dyn_cast_or_null<scf::ForOp>(definition);
  auto result = dyn_cast<OpResult>(value);
  auto yield = nested
                   ? dyn_cast<scf::YieldOp>(nested.getBody()->getTerminator())
                   : scf::YieldOp();
  return nested && result && yield &&
         result.getResultNumber() < nested.getInitArgs().size() &&
         nested.getInitArgs()[result.getResultNumber()] == carry &&
         isDirectContractionAccumulator(
             yield.getOperand(result.getResultNumber()),
             nested.getRegionIterArgs()[result.getResultNumber()], nested, visited);
}

bool hasRecurrentContraction(func::FuncOp kernel) {
  bool found = false;
  kernel.walk([&](scf::ForOp loop) {
    if (found || loop.getInitArgs().empty())
      return;
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != loop.getRegionIterArgs().size())
      return;
    for (auto [index, next] : llvm::enumerate(yield.getOperands())) {
      llvm::SmallDenseSet<Value, 8> directVisited;
      if (isDirectContractionAccumulator(next, loop.getRegionIterArgs()[index],
                                         loop, directVisited))
        continue;
      llvm::SmallDenseSet<Value, 32> contractionVisited;
      if (!valueDependsOnNestedContract(next, loop, contractionVisited))
        continue;
      llvm::SmallDenseSet<Value, 32> carryVisited;
      if (valueDependsOn(next, loop.getRegionIterArgs()[index], loop,
                         carryVisited)) {
        found = true;
        return;
      }
    }
  });
  return found;
}

} // namespace

std::optional<int64_t> evaluateCompileTimeExpression(
    gpu::PhysicalExprAttr expression, const TritonConfig &config) {
  return gpu::evaluatePhysicalExpression(expression, [&](gpu::PhysicalExprAttr leaf)
      -> std::optional<int64_t> {
    if (leaf.getKind() != static_cast<uint32_t>(gpu::PhysicalExprKind::Parameter))
      return std::nullopt;
    auto found = config.kernelParameters.find(leaf.getSymbol().getValue().str());
    return found == config.kernelParameters.end() ? std::nullopt
        : std::optional<int64_t>(found->second);
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
                              const TritonConfig &config) {
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
  TritonConfig known;
  gpu::ParameterAttr warpParameter;
  kernel.walk([&](gpu::ParameterOp parameter) {
    auto schema = parameter.getParameter();
    if (schema.getRole() ==
        static_cast<uint32_t>(gpu::ParameterRole::ProviderWarps))
      warpParameter = schema;
    if (schema.getCategory() !=
            static_cast<uint32_t>(gpu::ParameterCategory::Coverage) &&
        !parameter->hasAttr(gpu::coverageDimensionAttr))
      known.kernelParameters[schema.getName().getValue().str()] =
          schema.getCandidates().asArrayRef().front();
  });
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
    reductionSources.push_back(reduce.getInputs().take_front(reduce.getSourceCount()));
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
        if (evaluateCompileTimeExpression(footprint, TritonConfig{}))
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
                            const TritonConfig &config,
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
  if (config.stages > 1 && concrete &&
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
    if (domain.provider && domain.role() != gpu::ParameterRole::ProviderWarps &&
        domain.role() != gpu::ParameterRole::ProviderStages &&
        domain.role() != gpu::ParameterRole::ProviderCTAs)
      return kernel.emitError("Triton program contains a foreign provider parameter");
  }
  const auto *warps = space->find(gpu::ParameterRole::ProviderWarps);
  const auto *stages = space->find(gpu::ParameterRole::ProviderStages);
  const auto *ctas = space->find(gpu::ParameterRole::ProviderCTAs);
  if (!warps || !stages || !ctas)
    return kernel.emitError("Triton provider parameter domains are incomplete");
  SmallVector<TritonConfig> configs;
  bool stagesInKernel = false;
  kernel.walk([&](scf::ForOp loop) {
    stagesInKernel |= loop->hasAttr(loopStagesAttr);
  });
  int64_t maximumWarps = *llvm::max_element(warps->candidates());
  SmallVector<gpu::PhysicalExprAttr> collectiveFootprints;
  for (gpu::FragmentType fragment : collectiveFragments(kernel)) {
    auto footprint = gpu::fragmentRegisterFootprint(fragment);
    if (!llvm::is_contained(collectiveFootprints, footprint))
      collectiveFootprints.push_back(footprint);
  }
  for (DictionaryAttr tuple : *shared) {
    TritonConfig sharedConfig;
    for (NamedAttribute binding : tuple)
      sharedConfig.kernelParameters[binding.getName().getValue().str()] =
          cast<IntegerAttr>(binding.getValue()).getInt();
    for (const TritonLocalOptions &options : localOptions) {
      int64_t formCount = descriptorChoice ? 2 : 1;
      for (int64_t form = 0; form < formCount; ++form) {
        TritonConfig config = sharedConfig;
        if (descriptorChoice)
          config.kernelParameters[descriptorChoice.getConfigParameter().str()] =
              form;
        config.warps = options.warps;
        config.stages = options.stages;
        config.ctas = options.ctas;
        if (stagesInKernel)
          config.kernelParameters[stages->name().getValue().str()] = options.stages;
        if (llvm::none_of(configs, [&](const TritonConfig &existing) {
              return existing.kernelParameters == config.kernelParameters &&
                     existing.warps == config.warps &&
                     existing.stages == config.stages &&
                     existing.ctas == config.ctas;
            }))
          configs.push_back(std::move(config));
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
      for (Value source : reduce.getInputs().take_front(reduce.getSourceCount()))
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
                                const TritonConfig &config) {
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

  SmallVector<Attribute> encoded;
  Builder builder(kernel.getContext());
  for (const TritonConfig &config : configs) {
    if (config.warps <= 0 || config.stages <= 0 || config.ctas <= 0)
      return kernel.emitError("Triton provider parameter domains are incomplete");
    // Apply the existing per-fragment policy after binding the shared tuple.
    // ABI-dependent extents retain their deferred specialization assertion.
    if (config.warps < maximumWarps &&
        llvm::any_of(collectiveFootprints, [&](gpu::PhysicalExprAttr footprint) {
          auto words = evaluateCompileTimeExpression(footprint, config);
          return words && *words > config.warps * 32 * 255;
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
        config.kernelParameters.at(descriptorChoice.getConfigParameter().str()) != 0;
    if (descriptorConfig && llvm::any_of(descriptors, [&](const auto &constraint) {
          return !descriptorFragmentFits(constraint.fragment, config,
              constraint.alignment, parameterDomains, coverageParameters);
        }))
      continue;
    SmallVector<NamedAttribute> parameters;
    for (const auto &[name, value] : config.kernelParameters)
      parameters.push_back(builder.getNamedAttr(name, builder.getI64IntegerAttr(value)));
    encoded.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("parameters", builder.getDictionaryAttr(parameters)),
        builder.getNamedAttr("num_warps", builder.getI64IntegerAttr(config.warps)),
        builder.getNamedAttr("num_stages", builder.getI64IntegerAttr(config.stages)),
        builder.getNamedAttr("num_ctas", builder.getI64IntegerAttr(config.ctas)),
    }));
  }
  if (encoded.empty())
    return kernel.emitError(
        "all Triton parameter candidates violate typed fragment legality or the reduction register budget");
  kernel->setAttr(gpu::tritonConfigsAttr, builder.getArrayAttr(encoded));
  return success();
}

FailureOr<SmallVector<TritonLocalOptions>> declareProviderOptions(
    func::FuncOp kernel, const gpu::TuningProfiles &profiles,
    bool requiresCtaSynchronization, ArrayRef<scf::ForOp> loadPipelineLoops) {
  SmallVector<gpu::ParameterCategory> categories;
  bool twoAxisPointwise = false;
  kernel.walk([&](gpu::ParameterOp parameter) {
    auto schema = parameter.getParameter();
    auto category = static_cast<gpu::ParameterCategory>(schema.getCategory());
    twoAxisPointwise |= category == gpu::ParameterCategory::Pointwise &&
                       schema.getRole() == static_cast<uint32_t>(gpu::ParameterRole::OwnershipM);
    if (category != gpu::ParameterCategory::Coverage &&
        category != gpu::ParameterCategory::Provider &&
        !llvm::is_contained(categories, category))
      categories.push_back(category);
  });
  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  bool mayFormDot = false;
  kernel.walk([&](Operation *operation) {
    // Triton can combine a broadcast-multiply-reduce into a dot later.
    mayFormDot |= isa<gpu::ReduceOp, ReduceOp>(operation);
    if (isa<gpu::ReduceOp, gpu::ScanOp>(operation) &&
        !llvm::is_contained(categories, gpu::ParameterCategory::Reduction))
      categories.push_back(gpu::ParameterCategory::Reduction);
  });
  bool blackwell = capabilities.getComputeCapabilityMajor() == 10 ||
                   capabilities.getComputeCapabilityMajor() == 12;
  bool hasContraction = false;
  bool allFp32 = true;
  kernel.walk([&](gpu::ContractOp contract) {
    hasContraction = true;
    allFp32 &= contract.getLhs().getType().getElementType().isF32() &&
               contract.getRhs().getType().getElementType().isF32() &&
               contract.getResult().getType().getElementType().isF32();
  });
  if (hasContraction &&
      !llvm::is_contained(categories, gpu::ParameterCategory::Contraction))
    categories.push_back(gpu::ParameterCategory::Contraction);
  StringRef recurrentContractionFamily;
  if (hasRecurrentContraction(kernel)) {
    if (blackwell)
      recurrentContractionFamily = "blackwell_recurrent_contraction";
    else if (capabilities.getComputeCapabilityMajor() == 9)
      recurrentContractionFamily = "hopper_recurrent_contraction";
  }
  auto rows = profiles.get("triton", localOptionsFamily(
      categories, twoAxisPointwise, recurrentContractionFamily,
      hasContraction && allFp32), kernel.getLoc());
  if (failed(rows))
    return failure();
  bool straightLinePointwise =
      llvm::is_contained(categories, gpu::ParameterCategory::Pointwise) &&
      llvm::all_of(categories, [](gpu::ParameterCategory category) {
        return category == gpu::ParameterCategory::Pointwise;
      }) && llvm::all_of(kernel.front(), [](Operation &operation) {
        return operation.getNumRegions() == 0;
      });
  bool pipelineStagesAffectProgram = hasContraction && !allFp32;
  kernel.walk([&](Operation *operation) {
    pipelineStagesAffectProgram |=
        isa<gpu::ScaledContractOp, gpu::SparseContractOp>(operation) ||
        ((hasContraction || mayFormDot) &&
         operation->getParentOfType<scf::ForOp>() &&
         !isMemoryEffectFree(operation));
  });
  pipelineStagesAffectProgram |= !loadPipelineLoops.empty();
  SmallVector<TritonLocalOptions> localOptions;
  SmallVector<int64_t> warpDomain, stageDomain, ctaDomain;
  // Consumer Blackwell (SM12x) does not support CTA cluster operations.
  bool supportsCtaClusters = capabilities.getComputeCapabilityMajor() >= 9 &&
                             capabilities.getComputeCapabilityMajor() != 12;
  auto isDeviceOption = [&](int64_t warps, int64_t stages, int64_t ctas) {
    return (!requiresCtaSynchronization || ctas == 1) &&
           (warps & (warps - 1)) == 0 &&
           warps <= capabilities.getMaxThreadsPerBlock() / 32 &&
           stages <= std::numeric_limits<int32_t>::max() &&
           (ctas & (ctas - 1)) == 0 && ctas <= 16 &&
           (ctas == 1 || supportsCtaClusters);
  };
  for (const auto &row : *rows) {
    int64_t warps = row[0], stages = row[1], ctas = row[2];
    if (straightLinePointwise)
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
