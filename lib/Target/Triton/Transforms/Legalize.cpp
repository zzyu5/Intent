#include "Intent/Target/Triton/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringSet.h"

#include <limits>
#include <map>
#include <optional>

using namespace mlir;

namespace intent::triton {
namespace {

constexpr llvm::StringLiteral legalizedAttr = "intent_gpu.triton.legalized";
constexpr llvm::StringLiteral reduceFormAttr = "intent_gpu.triton.reduce_form";
constexpr int64_t maxTritonTensorElements = 1048576;

struct TritonConfig {
  std::map<std::string, int64_t> kernelParameters;
  int64_t warps = 0;
  int64_t stages = 0;
  int64_t ctas = 0;
};

int64_t staticDefault(gpu::ParameterRole role) {
  switch (role) {
  case gpu::ParameterRole::OwnershipM:
  case gpu::ParameterRole::OwnershipN:
    return 64;
  case gpu::ParameterRole::Reduction:
    return 32;
  case gpu::ParameterRole::ScanChunk:
    return 128;
  case gpu::ParameterRole::TraversalWorkers:
    return 1;
  case gpu::ParameterRole::TraversalGroup:
    return 8;
  case gpu::ParameterRole::ResidentWorkers:
    return std::numeric_limits<int64_t>::max();
  case gpu::ParameterRole::ProviderWarps:
    return 4;
  case gpu::ParameterRole::ProviderStages:
    return 2;
  case gpu::ParameterRole::ProviderCTAs:
    return 1;
  case gpu::ParameterRole::ProviderThreads:
    return 128;
  case gpu::ParameterRole::FullCoverage:
    llvm_unreachable("full-coverage parameters are bound by runtime extents");
  }
  llvm_unreachable("unknown physical parameter role");
}

int64_t selectStaticDefault(gpu::ParameterRole role,
                            ArrayRef<int64_t> candidates) {
  int64_t requested = staticDefault(role);
  int64_t selected = candidates.front();
  for (int64_t candidate : candidates) {
    if (candidate == requested)
      return candidate;
    if (candidate <= requested && candidate > selected)
      selected = candidate;
  }
  return selected;
}

std::optional<int64_t>
evaluateCompileTimeExpression(gpu::PhysicalExprAttr expression,
                              const TritonConfig &config) {
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  if (kind == gpu::PhysicalExprKind::Constant)
    return expression.getValue();
  if (kind == gpu::PhysicalExprKind::Parameter) {
    auto found = config.kernelParameters.find(expression.getSymbol().getValue().str());
    return found == config.kernelParameters.end()
               ? std::nullopt
               : std::optional<int64_t>(found->second);
  }
  if (kind == gpu::PhysicalExprKind::Dimension ||
      kind == gpu::PhysicalExprKind::ScalarABI)
    return std::nullopt;

  SmallVector<int64_t> operands;
  for (Attribute operand : expression.getOperands()) {
    std::optional<int64_t> value = evaluateCompileTimeExpression(
        cast<gpu::PhysicalExprAttr>(operand), config);
    if (!value)
      return std::nullopt;
    operands.push_back(*value);
  }
  auto checked = [](const __int128 value) -> std::optional<int64_t> {
    if (value < std::numeric_limits<int64_t>::min() ||
        value > std::numeric_limits<int64_t>::max())
      return std::nullopt;
    return static_cast<int64_t>(value);
  };
  if (kind == gpu::PhysicalExprKind::Add)
    return checked(static_cast<__int128>(operands[0]) + operands[1]);
  if (kind == gpu::PhysicalExprKind::Subtract)
    return checked(static_cast<__int128>(operands[0]) - operands[1]);
  if (kind == gpu::PhysicalExprKind::Multiply)
    return checked(static_cast<__int128>(operands[0]) * operands[1]);
  if (kind == gpu::PhysicalExprKind::Minimum)
    return std::min(operands[0], operands[1]);
  if (kind == gpu::PhysicalExprKind::Maximum)
    return std::max(operands[0], operands[1]);
  if (kind == gpu::PhysicalExprKind::Select)
    return operands[0] ? operands[1] : operands[2];
  if (kind == gpu::PhysicalExprKind::FloorDiv) {
    if (operands[1] <= 0 || operands[0] < 0)
      return std::nullopt;
    return operands[0] / operands[1];
  }
  if (kind == gpu::PhysicalExprKind::CeilDiv) {
    if (operands[1] <= 0 || operands[0] < 0)
      return std::nullopt;
    return checked((static_cast<__int128>(operands[0]) + operands[1] - 1) /
                   operands[1]);
  }
  if (kind == gpu::PhysicalExprKind::NextPowerOfTwo) {
    if (operands[0] <= 0)
      return std::nullopt;
    uint64_t value = static_cast<uint64_t>(operands[0] - 1);
    for (unsigned shift = 1; shift < 64; shift <<= 1)
      value |= value >> shift;
    if (value == std::numeric_limits<uint64_t>::max() ||
        value + 1 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return std::nullopt;
    return static_cast<int64_t>(value + 1);
  }
  return std::nullopt;
}

bool isCompileTimeExpression(gpu::PhysicalExprAttr expression) {
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  if (kind == gpu::PhysicalExprKind::Constant ||
      kind == gpu::PhysicalExprKind::Parameter)
    return true;
  if (kind == gpu::PhysicalExprKind::Dimension ||
      kind == gpu::PhysicalExprKind::ScalarABI)
    return false;
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isCompileTimeExpression(cast<gpu::PhysicalExprAttr>(operand));
  });
}

bool fragmentFitsTritonTensor(gpu::FragmentType fragment,
                              const TritonConfig &config) {
  __int128 elements = 1;
  for (Attribute extent : fragment.getShape()) {
    std::optional<int64_t> value = evaluateCompileTimeExpression(
        cast<gpu::PhysicalExprAttr>(extent), config);
    // Runtime dimensions and full-coverage heuristics are checked by Triton at
    // specialization time.  This pass rejects only candidates whose typed
    // compile-time shape is already known to be illegal.
    if (!value)
      return true;
    if (*value <= 0)
      return false;
    elements *= *value;
    if (elements > maxTritonTensorElements)
      return false;
  }
  return true;
}

bool typeFitsTritonTensor(Type type, const TritonConfig &config) {
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return fragmentFitsTritonTensor(fragment, config);
  if (auto record = dyn_cast<gpu::RecordType>(type))
    return llvm::all_of(record.getFieldTypes(), [&](Attribute field) {
      return typeFitsTritonTensor(cast<TypeAttr>(field).getValue(), config);
    });
  return true;
}

LogicalResult materializeLegalConfigs(func::FuncOp kernel) {
  struct Domain {
    StringRef name;
    gpu::ParameterRole role;
    ArrayRef<int64_t> candidates;
    bool heuristic;
  };
  SmallVector<Domain> domains;
  llvm::StringSet<> names;
  WalkResult schema = kernel.walk([&](gpu::ParameterOp parameter) {
    auto definition = parameter.getParameter();
    StringRef name = definition.getName().getValue();
    if (!names.insert(name).second) {
      parameter.emitOpError("duplicates a Triton parameter name");
      return WalkResult::interrupt();
    }
    domains.push_back({name,
                       static_cast<gpu::ParameterRole>(definition.getRole()),
                       definition.getCandidates().asArrayRef(),
                       parameter->hasAttr(gpu::coverageDimensionAttr)});
    return WalkResult::advance();
  });
  if (schema.wasInterrupted())
    return failure();

  TritonConfig defaultConfig;
  for (const Domain &domain : domains) {
    if (domain.heuristic)
      continue;
    int64_t binding = selectStaticDefault(domain.role, domain.candidates);
    switch (domain.role) {
    case gpu::ParameterRole::ProviderWarps:
      defaultConfig.warps = binding;
      break;
    case gpu::ParameterRole::ProviderStages:
      defaultConfig.stages = binding;
      break;
    case gpu::ParameterRole::ProviderCTAs:
      defaultConfig.ctas = binding;
      break;
    default:
      defaultConfig.kernelParameters[domain.name.str()] = binding;
      break;
    }
  }

  SmallVector<Attribute> encoded;
  Builder builder(kernel.getContext());
  for (const TritonConfig &config : ArrayRef<TritonConfig>(defaultConfig)) {
    if (config.warps <= 0 || config.stages <= 0 || config.ctas <= 0)
      return kernel.emitError("Triton provider parameter domains are incomplete");
    bool legal = true;
    kernel.walk([&](Operation *operation) {
      if (!legal)
        return WalkResult::interrupt();
      if (auto range = dyn_cast<gpu::MakeRangeOp>(operation)) {
        auto fragment = dyn_cast<gpu::FragmentType>(range.getResult().getType());
        if (!fragment || fragment.getShape().size() != 1) {
          legal = false;
          return WalkResult::interrupt();
        }
        std::optional<int64_t> extent = evaluateCompileTimeExpression(
            cast<gpu::PhysicalExprAttr>(fragment.getShape()[0]), config);
        if (extent && (*extent <= 0 ||
                       (static_cast<uint64_t>(*extent) &
                        (static_cast<uint64_t>(*extent) - 1)) != 0)) {
          legal = false;
          return WalkResult::interrupt();
        }
      }
      for (Type type : operation->getResultTypes())
        if (!typeFitsTritonTensor(type, config)) {
          legal = false;
          return WalkResult::interrupt();
        }
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            if (!typeFitsTritonTensor(argument.getType(), config)) {
              legal = false;
              return WalkResult::interrupt();
            }
      return WalkResult::advance();
    });
    if (!legal)
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
        "all Triton parameter candidates violate typed fragment legality");
  kernel->setAttr(gpu::tritonConfigsAttr, builder.getArrayAttr(encoded));
  return success();
}

bool isTritonScalarType(Type type) {
  if (type.isIndex() || isa<Float16Type, BFloat16Type, Float32Type,
                            Float64Type, Float8E4M3FNType,
                            Float8E5M2Type>(type))
    return true;
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() == 1 || integer.getWidth() == 8 ||
           integer.getWidth() == 16 || integer.getWidth() == 32 ||
           integer.getWidth() == 64;
  return false;
}

bool isTritonDataType(Type type) {
  if (isTritonScalarType(type))
    return true;
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return isTritonScalarType(fragment.getElementType());
  if (auto record = dyn_cast<gpu::RecordType>(type)) {
    for (Attribute field : record.getFieldTypes())
      if (!isTritonDataType(cast<TypeAttr>(field).getValue()))
        return false;
    return true;
  }
  return false;
}

bool isTritonExpression(gpu::PhysicalExprAttr expression);

Type elementType(Type type) {
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return fragment.getElementType();
  return type;
}

bool isTritonAtomicAddType(Type type) {
  type = elementType(type);
  if (isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(type))
    return true;
  auto integer = dyn_cast<IntegerType>(type);
  return integer && (integer.getWidth() == 32 || integer.getWidth() == 64);
}

bool samePhysicalShape(Type lhs, Type rhs) {
  auto left = dyn_cast<gpu::FragmentType>(lhs);
  auto right = dyn_cast<gpu::FragmentType>(rhs);
  if (static_cast<bool>(left) != static_cast<bool>(right))
    return false;
  return !left ||
         (left.getShape() == right.getShape() &&
          left.getAxisMaps() == right.getAxisMaps() &&
          left.getValidity() == right.getValidity() &&
          left.getOwner() == right.getOwner());
}

std::optional<unsigned>
expandedGatherAxis(gpu::FragmentType source, gpu::FragmentType result,
                   unsigned selectedSourceAxis) {
  if (source.getOwner() != result.getOwner() ||
      source.getShape().size() >= result.getShape().size() ||
      selectedSourceAxis >= source.getShape().size())
    return std::nullopt;

  SmallVector<std::optional<unsigned>> sourceToResult(source.getShape().size());
  SmallVector<bool> resultUsed(result.getShape().size(), false);
  for (auto [sourceIndex, sourceMapping] :
       llvm::enumerate(source.getAxisMaps())) {
    auto sourceAxis = cast<gpu::AxisMapAttr>(sourceMapping);
    for (auto [resultIndex, resultMapping] :
         llvm::enumerate(result.getAxisMaps())) {
      auto resultAxis = cast<gpu::AxisMapAttr>(resultMapping);
      if (sourceAxis.getSourceId() != resultAxis.getSourceId() ||
          sourceAxis.getSourceAxis() != resultAxis.getSourceAxis())
        continue;
      if (sourceToResult[sourceIndex] || resultUsed[resultIndex] ||
          source.getShape()[sourceIndex] != result.getShape()[resultIndex])
        return std::nullopt;
      sourceToResult[sourceIndex] = resultIndex;
      resultUsed[resultIndex] = true;
    }
    if (!sourceToResult[sourceIndex])
      return std::nullopt;
  }

  for (auto [resultIndex, used] : llvm::enumerate(resultUsed)) {
    if (used)
      continue;
    auto extent = dyn_cast<gpu::PhysicalExprAttr>(result.getShape()[resultIndex]);
    if (!extent ||
        extent.getKind() !=
            static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) ||
        extent.getValue() != 1)
      return std::nullopt;
  }
  return sourceToResult[selectedSourceAxis];
}

FailureOr<Value> zeroLike(OpBuilder &builder, Location location, Type type) {
  Type scalarType = elementType(type);
  Value zero;
  if (scalarType.isIndex())
    zero = builder.create<arith::ConstantIndexOp>(location, 0);
  else if (auto integer = dyn_cast<IntegerType>(scalarType))
    zero = builder.create<arith::ConstantOp>(
        location, integer, builder.getIntegerAttr(integer, 0));
  else
    return failure();
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return Value(builder.create<gpu::SplatOp>(location, fragment, zero));
  return zero;
}

LogicalResult legalizeMaskedGather(func::FuncOp kernel) {
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  for (gpu::GatherOp gather : gathers) {
    if (!gather.getValid())
      continue;
    auto scalarConstant = [](Value value) -> arith::ConstantOp {
      while (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>())
        value = broadcast.getValue();
      while (auto splat = value.getDefiningOp<gpu::SplatOp>())
        value = splat.getValue();
      while (auto cast = value.getDefiningOp<gpu::CastOp>())
        value = cast.getValue();
      return value.getDefiningOp<arith::ConstantOp>();
    };
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    if (gather.getCoordinates().size() == 1 &&
        gather.getSourceAxes().size() == 1 && source && result &&
        source.getShape().size() < result.getShape().size()) {
      std::optional<unsigned> selectedAxis = expandedGatherAxis(
          source, result, gather.getSourceAxes().front());
      if (selectedAxis) {
        OpBuilder builder(gather);
        auto expandedSourceType = gpu::FragmentType::get(
            gather.getContext(), source.getElementType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        Value expandedSource = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), expandedSourceType, gather.getSource());
        Value coordinate = gather.getCoordinates().front();
        auto coordinateType = gpu::FragmentType::get(
            gather.getContext(), coordinate.getType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        if (auto coordinateFragment =
                dyn_cast<gpu::FragmentType>(coordinate.getType())) {
          coordinateType = gpu::FragmentType::get(
              gather.getContext(), coordinateFragment.getElementType(),
              result.getShape(), result.getAxisMaps(), result.getValidity(),
              result.getOwner());
        }
        coordinate = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), coordinateType, coordinate);
        FailureOr<Value> zero =
            zeroLike(builder, gather.getLoc(), coordinateType);
        if (succeeded(zero)) {
          Value safeIndex = builder.create<gpu::SelectOp>(
              gather.getLoc(), coordinateType, gather.getValid(), coordinate,
              *zero);
          auto safeGather = builder.create<gpu::GatherOp>(
              gather.getLoc(), result, expandedSource, ValueRange{safeIndex},
              Value(), Value(), ArrayRef<int64_t>{static_cast<int64_t>(*selectedAxis)});
          auto selected = builder.create<gpu::SelectOp>(
              gather.getLoc(), result, gather.getValid(), safeGather.getResult(),
              gather.getFill());
          if (Attribute origin = gather->getAttr(gpu::originAttr))
            selected->setAttr(gpu::originAttr, origin);
          gather.getResult().replaceAllUsesWith(selected.getResult());
          gather.erase();
          continue;
        }
      }
    }
    if (gather.getCoordinates().size() == 1 &&
        gather.getSourceAxes().size() == 1 && source && result &&
        source.getShape().size() == result.getShape().size() + 1) {
      unsigned selectedAxis = gather.getSourceAxes().front();
      auto selectedExtent =
          selectedAxis < source.getShape().size()
              ? dyn_cast<gpu::PhysicalExprAttr>(source.getShape()[selectedAxis])
              : gpu::PhysicalExprAttr();
      bool compatible = selectedExtent &&
                        selectedExtent.getKind() == static_cast<uint32_t>(
                                                        gpu::PhysicalExprKind::Constant) &&
                        selectedExtent.getValue() > 0;
      unsigned resultAxis = 0;
      for (unsigned sourceAxis = 0;
           compatible && sourceAxis < source.getShape().size(); ++sourceAxis) {
        if (sourceAxis == selectedAxis)
          continue;
        compatible = resultAxis < result.getShape().size() &&
                     source.getShape()[sourceAxis] == result.getShape()[resultAxis];
        ++resultAxis;
      }
      if (compatible) {
        OpBuilder builder(gather);
        SmallVector<Attribute> selectedShape(source.getShape().begin(),
                                             source.getShape().end());
        selectedShape[selectedAxis] = gpu::PhysicalExprAttr::get(
            gather.getContext(),
            static_cast<uint32_t>(gpu::PhysicalExprKind::Constant), 1,
            builder.getStringAttr(""), builder.getArrayAttr({}));
        auto coordinateType = gpu::FragmentType::get(
            gather.getContext(), gather.getCoordinates().front().getType(),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            source.getValidity(), source.getOwner());
        auto predicateType = gpu::FragmentType::get(
            gather.getContext(), builder.getI1Type(),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            source.getValidity(), source.getOwner());
        auto selectedResultType = gpu::FragmentType::get(
            gather.getContext(), result.getElementType(),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            result.getValidity(), result.getOwner());
        Value coordinate = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), coordinateType, gather.getCoordinates().front());
        Value valid = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), predicateType, gather.getValid());
        Value fill = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), selectedResultType, gather.getFill());
        FailureOr<Value> zero = zeroLike(builder, gather.getLoc(), coordinateType);
        if (succeeded(zero)) {
          Value safeIndex = builder.create<gpu::SelectOp>(
              gather.getLoc(), coordinateType, valid, coordinate, *zero);
          auto safeGather = builder.create<gpu::GatherOp>(
              gather.getLoc(), selectedResultType, gather.getSource(),
              ValueRange{safeIndex}, Value(), Value(), gather.getSourceAxes());
          Value selected = builder.create<gpu::SelectOp>(
              gather.getLoc(), selectedResultType, valid, safeGather.getResult(),
              fill);
          FailureOr<ArrayAttr> reassociation =
              gpu::inferReshapeReassociation(selectedResultType, result);
          if (failed(reassociation))
            return gather.emitOpError(
                "gather fallback has no exact row-major reassociation");
          auto reshaped = builder.create<gpu::ReshapeOp>(
              gather.getLoc(), result, selected, *reassociation);
          if (Attribute origin = gather->getAttr(gpu::originAttr))
            reshaped->setAttr(gpu::originAttr, origin);
          gather.getResult().replaceAllUsesWith(reshaped.getResult());
          gather.erase();
          continue;
        }
      }
    }
    if (gather.getCoordinates().size() == 1 &&
        isa<gpu::FragmentType>(gather.getCoordinates().front().getType()) &&
        gather.getSourceAxes().size() == 1 && source) {
      unsigned axis = gather.getSourceAxes().front();
      auto coordinate = scalarConstant(gather.getCoordinates().front());
      auto predicate = scalarConstant(gather.getValid());
      auto extent = axis < source.getShape().size()
                        ? dyn_cast<gpu::PhysicalExprAttr>(source.getShape()[axis])
                        : gpu::PhysicalExprAttr();
      auto coordinateValue =
          coordinate ? dyn_cast<IntegerAttr>(coordinate.getValue()) : IntegerAttr();
      auto predicateValue = predicate ? dyn_cast<IntegerAttr>(predicate.getValue())
                                      : IntegerAttr();
      if (extent &&
          extent.getKind() == static_cast<uint32_t>(
                                  gpu::PhysicalExprKind::Constant) &&
          coordinateValue && coordinateValue.getInt() >= 0 &&
          coordinateValue.getInt() < extent.getValue() && predicateValue &&
          predicateValue.getValue().isOne()) {
        OpBuilder builder(gather);
        auto replacement = builder.create<gpu::GatherOp>(
            gather.getLoc(), gather.getResult().getType(), gather.getSource(),
            gather.getCoordinates(), Value(), Value(), gather.getSourceAxes());
        if (Attribute origin = gather->getAttr(gpu::originAttr))
          replacement->setAttr(gpu::originAttr, origin);
        gather.getResult().replaceAllUsesWith(replacement.getResult());
        gather.erase();
        continue;
      }
    }
    if (gather.getCoordinates().size() != 1)
      continue;
    OpBuilder builder(gather);
    Value coordinate = gather.getCoordinates().front();
    if (!isa<gpu::FragmentType>(coordinate.getType())) {
      auto predicateType = dyn_cast<gpu::FragmentType>(gather.getValid().getType());
      if (!predicateType ||
          !isa<IntegerType, IndexType>(coordinate.getType()))
        continue;
      auto coordinateType = gpu::FragmentType::get(
          gather.getContext(), coordinate.getType(), predicateType.getShape(),
          predicateType.getAxisMaps(), predicateType.getValidity(),
          predicateType.getOwner());
      coordinate = builder.create<gpu::BroadcastOp>(gather.getLoc(),
                                                     coordinateType, coordinate);
    }
    if (!samePhysicalShape(gather.getValid().getType(), coordinate.getType()) ||
        !samePhysicalShape(gather.getValid().getType(),
                           gather.getResult().getType()))
      continue;
    FailureOr<Value> zero =
        zeroLike(builder, gather.getLoc(), coordinate.getType());
    if (failed(zero))
      continue;
    Value safeIndex = builder.create<gpu::SelectOp>(
        gather.getLoc(), coordinate.getType(), gather.getValid(), coordinate,
        *zero);
    auto safeGather = builder.create<gpu::GatherOp>(
        gather.getLoc(), gather.getResult().getType(), gather.getSource(),
        ValueRange{safeIndex}, Value(), Value(), gather.getSourceAxes());
    auto selected = builder.create<gpu::SelectOp>(
        gather.getLoc(), gather.getResult().getType(), gather.getValid(),
        safeGather.getResult(), gather.getFill());
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      selected->setAttr(gpu::originAttr, origin);
    gather.getResult().replaceAllUsesWith(selected.getResult());
    gather.erase();
  }
  return success();
}

bool isAddCombine(gpu::ScatterReduceOp scatter) {
  return gpu::queryBinaryCombineKind(scatter.getCombine()) ==
         BinaryOperator::Add;
}

bool isNativeAddReduce(gpu::ReduceOp reduce) {
  if (reduce.getSourceCount() != 1 || reduce.getIdentityCount() != 1 ||
      reduce.getCaptureCount() != 0 || reduce.getResults().size() != 1 ||
      reduce.getCombine().empty() || reduce.getCombine().getBlocks().size() != 1)
    return false;
  return gpu::queryBinaryCombineKind(reduce.getCombine()) ==
         BinaryOperator::Add;
}

void selectNativeReduceForms(func::FuncOp kernel) {
  kernel.walk([&](gpu::ReduceOp reduce) {
    if (isNativeAddReduce(reduce))
      reduce->setAttr(reduceFormAttr,
                      StringAttr::get(kernel.getContext(), "sum"));
  });
}

LogicalResult legalizeScatterAdd(func::FuncOp kernel) {
  SmallVector<gpu::ScatterReduceOp> scatters;
  kernel.walk([&](gpu::ScatterReduceOp scatter) { scatters.push_back(scatter); });
  for (gpu::ScatterReduceOp scatter : scatters) {
    if (!isAddCombine(scatter) || !isTritonAtomicAddType(scatter.getValue().getType()))
      continue;
    OpBuilder builder(scatter);
    auto atomic = builder.create<gpu::AtomicRMWOp>(
        scatter.getLoc(), scatter.getValue().getType(), scatter.getResource(),
        scatter.getCoordinates(), scatter.getValue(), scatter.getValid(),
        AtomicRMWKind::Add, AtomicOrdering::Relaxed, scatter.getSharing(),
        scatter.getSourceAxes());
    if (Attribute origin = scatter->getAttr(gpu::originAttr))
      atomic->setAttr(gpu::originAttr, origin);
    scatter.erase();
  }
  return success();
}

bool isTritonFragmentExtent(Attribute attribute) {
  auto expression = dyn_cast<gpu::PhysicalExprAttr>(attribute);
  if (!expression)
    return false;
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  return kind != gpu::PhysicalExprKind::ScalarABI &&
         isTritonExpression(expression);
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

LogicalResult verifyAccess(Operation *operation, Value resource,
                           ValueRange coordinates,
                           ArrayRef<int64_t> sourceAxes) {
  auto view = dyn_cast<gpu::ViewType>(resource.getType());
  if (!view)
    return operation->emitOpError(
        "Triton pointer access requires a legalized external-view resource");
  if (coordinates.size() != sourceAxes.size())
    return operation->emitOpError(
        "Triton access coordinates and source axes are not bijective");
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t axis : sourceAxes)
    if (axis < 0 || axis >= view.getRank() || !seen.insert(axis).second)
      return operation->emitOpError(
          "Triton access contains an invalid or duplicate source axis");
  return success();
}

LogicalResult verifyKernel(func::FuncOp kernel) {
  auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
  if (!space || space.empty() || space.size() > 3)
    return kernel.emitError(
        "Triton source surface requires a one-to-three dimensional grid");
  for (Attribute extent : space)
    if (!isTritonExpression(cast<gpu::PhysicalExprAttr>(extent)))
      return kernel.emitError(
          "Triton launch expression has no deterministic terminal spelling");

  for (Type type : kernel.getArgumentTypes()) {
    if (auto view = dyn_cast<gpu::ViewType>(type)) {
      if (!isTritonScalarType(view.getElementType()))
        return kernel.emitError("Triton view element type is unsupported");
      continue;
    }
    if (!isTritonScalarType(type))
      return kernel.emitError("Triton physical ABI type is unsupported");
  }

  llvm::SmallDenseSet<uint32_t> providerRoles;
  LogicalResult parameterSchema = success();
  kernel.walk([&](gpu::ParameterOp parameter) {
    uint32_t role = parameter.getParameter().getRole();
    bool providerRole =
        role == static_cast<uint32_t>(gpu::ParameterRole::ProviderWarps) ||
        role == static_cast<uint32_t>(gpu::ParameterRole::ProviderStages) ||
        role == static_cast<uint32_t>(gpu::ParameterRole::ProviderCTAs);
    if (!providerRole)
      return;
    if (!providerRoles.insert(role).second) {
      parameter.emitOpError("duplicates a Triton provider-parameter role");
      parameterSchema = failure();
      return;
    }
    if (role == static_cast<uint32_t>(gpu::ParameterRole::ProviderWarps))
      for (int64_t candidate :
           parameter.getParameter().getCandidates().asArrayRef())
        if (!llvm::isPowerOf2_64(candidate)) {
          parameter.emitOpError(
              "declares a non-power-of-two Triton num_warps candidate");
          parameterSchema = failure();
          return;
        }
  });
  if (failed(parameterSchema))
    return failure();

  WalkResult result = kernel.walk([&](Operation *operation) {
    if (isa<gpu::AtomicLoadOp>(operation)) {
      operation->emitOpError(
          "has no Triton atomic-load primitive with preserved memory-order semantics");
      return WalkResult::interrupt();
    }
    if (auto scatter = dyn_cast<gpu::ScatterReduceOp>(operation)) {
      scatter.emitOpError(
          "requires an add combine and Triton-native atomic-add dtype legalization");
      return WalkResult::interrupt();
    }
    if (isa<gpu::BufferOp>(operation)) {
      operation->emitOpError(
          "requires provider-local mutable-buffer realization before Triton serialization");
      return WalkResult::interrupt();
    }
    if (isa<gpu::SparseContractOp>(operation)) {
      operation->emitOpError(
          "has no native Triton structured-sparse contraction surface");
      return WalkResult::interrupt();
    }
    for (Type type : operation->getResultTypes()) {
      if (auto fragment = dyn_cast<gpu::FragmentType>(type)) {
        if (!isTritonScalarType(fragment.getElementType()) ||
            !llvm::all_of(fragment.getShape(), isTritonFragmentExtent)) {
          operation->emitOpError(
              "has no statically blocked Triton fragment representation");
          return WalkResult::interrupt();
        }
      } else if (isa<gpu::RecordType>(type)) {
        if (!isTritonDataType(type)) {
          operation->emitOpError("contains a record field outside Triton data types");
          return WalkResult::interrupt();
        }
      } else if (!isa<gpu::ViewType, gpu::RangeType>(type) &&
                 !isTritonScalarType(type)) {
        operation->emitOpError("has a result type outside the Triton surface");
        return WalkResult::interrupt();
      }
    }
    if (auto load = dyn_cast<gpu::LoadOp>(operation)) {
      if (failed(verifyAccess(operation, load.getResource(),
                              load.getCoordinates(), load.getSourceAxes())))
        return WalkResult::interrupt();
    } else if (auto store = dyn_cast<gpu::StoreOp>(operation)) {
      if (failed(verifyAccess(operation, store.getResource(),
                              store.getCoordinates(), store.getSourceAxes()))) {
        return WalkResult::interrupt();
      }
    } else if (auto atomic = dyn_cast<gpu::AtomicStoreOp>(operation)) {
      if (failed(verifyAccess(operation, atomic.getResource(),
                              atomic.getCoordinates(), atomic.getSourceAxes())))
        return WalkResult::interrupt();
    } else if (auto atomic = dyn_cast<gpu::AtomicRMWOp>(operation)) {
      if (failed(verifyAccess(operation, atomic.getResource(),
                              atomic.getCoordinates(), atomic.getSourceAxes())))
        return WalkResult::interrupt();
    } else if (auto atomic =
                   dyn_cast<gpu::AtomicCompareExchangeOp>(operation)) {
      if (atomic.getValid()) {
        atomic.emitOpError(
            "Triton tl.atomic_cas has no mask and cannot preserve physical validity");
        return WalkResult::interrupt();
      }
      if (failed(verifyAccess(operation, atomic.getResource(),
                              atomic.getCoordinates(), atomic.getSourceAxes())))
        return WalkResult::interrupt();
    }
    if (auto contract = dyn_cast<gpu::ContractOp>(operation)) {
      unsigned lhsRank = contract.getLhs().getType().getShape().size();
      unsigned rhsRank = contract.getRhs().getType().getShape().size();
      SmallVector<int64_t> lhsBatch;
      SmallVector<int64_t> rhsBatch;
      for (unsigned axis = 0; axis + 2 < lhsRank; ++axis) {
        lhsBatch.push_back(axis);
        rhsBatch.push_back(axis);
      }
      if (lhsRank < 2 || rhsRank != lhsRank ||
          contract.getLhsReductionAxes() !=
              ArrayRef<int64_t>{static_cast<int64_t>(lhsRank - 1)} ||
          contract.getRhsReductionAxes() !=
              ArrayRef<int64_t>{static_cast<int64_t>(rhsRank - 2)} ||
          contract.getLhsBatchAxes() != ArrayRef<int64_t>(lhsBatch) ||
          contract.getRhsBatchAxes() != ArrayRef<int64_t>(rhsBatch)) {
        contract.emitOpError(
            "requires provider legalization to [...,M,K] x [...,K,N] tl.dot form");
        return WalkResult::interrupt();
      }
    } else if (auto range = dyn_cast<gpu::MakeRangeOp>(operation)) {
      auto fragment = dyn_cast<gpu::FragmentType>(range.getResult().getType());
      auto physicalExtent =
          fragment && fragment.getShape().size() == 1
              ? dyn_cast<gpu::PhysicalExprAttr>(fragment.getShape()[0])
              : gpu::PhysicalExprAttr();
      if (!physicalExtent || !isCompileTimeExpression(physicalExtent)) {
        InFlightDiagnostic diagnostic = range.emitOpError(
            "Triton tl.arange physical extent must be a compile-time physical expression");
        diagnostic << "; source_id=" << range.getSourceId();
        if (fragment && fragment.getAxisMaps().size() == 1)
          diagnostic << ", source_dimension="
                     << cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[0])
                            .getDimensionId();
        unsigned loopDepth = 0;
        for (Operation *parent = range->getParentOp(); parent;
             parent = parent->getParentOp())
          loopDepth += isa<scf::ForOp>(parent);
        diagnostic << ", loop_depth=" << loopDepth
                   << ", physical_extent=" << physicalExtent;
        diagnostic << (range.getResult().use_empty() ? " and is dead"
                                                      : " and still has live uses");
        for (Operation *user : range.getResult().getUsers()) {
          diagnostic << " " << user->getName();
          if (auto load = dyn_cast<gpu::LoadOp>(user)) {
            if (auto view = dyn_cast<gpu::ViewType>(load.getResource().getType()))
              diagnostic << "(abi=" << view.getAbiIndex() << ",source_axes=["
                         << load.getSourceAxes() << "])";
            for (Operation *loadUser : load.getResult().getUsers()) {
              diagnostic << "->" << loadUser->getName();
              for (Value result : loadUser->getResults())
                for (Operation *resultUser : result.getUsers()) {
                  diagnostic << "->" << resultUser->getName();
                  for (Value nextResult : resultUser->getResults())
                    for (Operation *nextUser : nextResult.getUsers()) {
                      diagnostic << "->" << nextUser->getName();
                      if (auto reduction = dyn_cast<gpu::ReduceOp>(nextUser))
                        diagnostic << "(axes=[" << reduction.getAxes()
                                   << "],sources="
                                   << reduction.getSourceCount() << ")";
                    }
                }
            }
          }
        }
        return WalkResult::interrupt();
      }
    } else if (auto gather = dyn_cast<gpu::GatherOp>(operation)) {
      if (gather.getValid()) {
        gather.emitOpError(
            "requires safe-index legalization before Triton tl.gather: source=")
            << gather.getSource().getType()
            << ", coordinate=" << gather.getCoordinates().front().getType()
            << ", valid=" << gather.getValid().getType()
            << ", fill=" << gather.getFill().getType()
            << ", result=" << gather.getResult().getType();
        return WalkResult::interrupt();
      }
      if (gather.getCoordinates().size() != 1 ||
          gather.getSourceAxes().size() != 1) {
        gather.emitOpError(
            "requires provider legalization to one-axis tl.gather");
        return WalkResult::interrupt();
      }
    } else if (auto reduce = dyn_cast<gpu::ReduceOp>(operation)) {
      if (reduce.getAxes().size() != 1 || reduce.getCaptureCount() != 0) {
        reduce.emitOpError(
            "Triton reduce requires one physical axis and capture-free helper");
        return WalkResult::interrupt();
      }
    } else if (auto scan = dyn_cast<gpu::ScanOp>(operation)) {
      if (!scan.getInclusive() || scan.getCaptureCount() != 0) {
        scan.emitOpError(
            "Triton associative_scan requires inclusive capture-free physical form");
        return WalkResult::interrupt();
      }
    } else if (auto contract = dyn_cast<gpu::ScaledContractOp>(operation)) {
      unsigned lhsRank = contract.getLhs().getType().getShape().size();
      unsigned rhsRank = contract.getRhs().getType().getShape().size();
      Type lhsScaleElement =
          contract.getLhsScale().getType().getElementType();
      Type rhsScaleElement =
          contract.getRhsScale().getType().getElementType();
      if ((lhsRank != 2 && lhsRank != 3) || rhsRank != lhsRank ||
          contract.getLhsFormat() == ScaledFormat::E8M0 ||
          contract.getRhsFormat() == ScaledFormat::E8M0 ||
          contract.getLhsGroupSize() != 32 ||
          contract.getRhsGroupSize() != 32 ||
          !lhsScaleElement.isUnsignedInteger(8) ||
          !rhsScaleElement.isUnsignedInteger(8) ||
          !contract.getResult().getType().getElementType().isF32() ||
          contract.getLhsReductionAxes() !=
              ArrayRef<int64_t>{static_cast<int64_t>(lhsRank - 1)} ||
          contract.getRhsReductionAxes() !=
              ArrayRef<int64_t>{static_cast<int64_t>(rhsRank - 2)}) {
        contract.emitOpError(
            "is outside Triton tl.dot_scaled rank/format/e8m0-scale/group/axis legality");
        return WalkResult::interrupt();
      }
    } else if (auto histogram = dyn_cast<gpu::HistogramOp>(operation)) {
      if (!histogram.getBins().getDefiningOp<arith::ConstantOp>() &&
          !histogram.getBins().getDefiningOp<gpu::ParameterOp>()) {
        histogram.emitOpError(
            "Triton histogram bin count must be compile-time bound");
        return WalkResult::interrupt();
      }
    }

    if (isa<gpu::ParameterOp, gpu::PhysicalExprOp, gpu::ProgramIdOp,
            gpu::WorksetCoordinateOp, gpu::DelinearizeOp,
            gpu::DimOp, gpu::RangeOp, gpu::RangeBoundOp, gpu::MakeRangeOp,
            gpu::SplatOp, gpu::BroadcastOp, gpu::UnaryOp, gpu::BinaryOp,
            gpu::CompareOp, gpu::SelectOp, gpu::CastOp, gpu::BitcastOp,
            gpu::ReshapeOp, gpu::TransposeOp, gpu::JoinOp, gpu::MakeRecordOp,
            gpu::ExtractOp, gpu::LoadOp, gpu::GatherOp,
            gpu::AssumeInBoundsOp, gpu::StoreOp,
            gpu::ContractOp, gpu::ReduceOp, gpu::ScanOp,
            gpu::ScaledContractOp, gpu::HistogramOp, gpu::AtomicStoreOp,
            gpu::AtomicRMWOp, gpu::AtomicCompareExchangeOp,
            gpu::RandomBitsOp, gpu::YieldOp, arith::ConstantOp, scf::ForOp,
            scf::IfOp, scf::WhileOp, scf::ConditionOp, scf::YieldOp,
            func::FuncOp, func::ReturnOp>(operation))
      return WalkResult::advance();
    operation->emitOpError("is outside the closed Triton provider surface");
    return WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

} // namespace

LogicalResult verifyTritonProgram(ModuleOp module) {
  SmallVector<func::FuncOp> kernels;
  module.walk([&](func::FuncOp function) {
    if (function->hasAttr(gpu::kernelAttr))
      kernels.push_back(function);
  });
  if (kernels.size() != 1)
    return module.emitError("Triton provider program requires one physical kernel");
  return verifyKernel(kernels.front());
}

LogicalResult legalizeGPUProgram(ModuleOp module) {
  if (failed(gpu::verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> physicalKernel = gpu::getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  if (failed(legalizeProgramGrid(module)) ||
      failed(gpu::verifyGPUProgram(module)))
    return failure();
  if (failed(legalizeMaskedGather(kernel)) ||
      failed(legalizeScatterAdd(kernel)) ||
      failed(gpu::verifyGPUProgram(module)))
    return failure();
  selectNativeReduceForms(kernel);
  OpBuilder builder(&kernel.getBody().front(), kernel.getBody().front().begin());
  llvm::StringSet<> names;
  kernel.walk([&](gpu::ParameterOp parameter) {
    names.insert(parameter.getParameter().getName().getValue());
  });
  auto declareProviderParameter = [&](StringRef name, gpu::ParameterRole role,
                                      ArrayRef<int64_t> candidates) {
    if (names.contains(name))
      return;
    auto schema = gpu::ParameterAttr::get(
        module.getContext(), builder.getStringAttr(name),
        static_cast<uint32_t>(role),
        DenseI64ArrayAttr::get(module.getContext(), candidates));
    builder.create<gpu::ParameterOp>(kernel.getLoc(), builder.getIndexType(),
                                     schema);
    names.insert(name);
  };
  declareProviderParameter("NUM_WARPS", gpu::ParameterRole::ProviderWarps,
                           ArrayRef<int64_t>{1, 2, 4, 8, 16, 32});
  declareProviderParameter("NUM_STAGES", gpu::ParameterRole::ProviderStages,
                           ArrayRef<int64_t>{1, 2, 3, 4, 5, 6});
  declareProviderParameter("NUM_CTAS", gpu::ParameterRole::ProviderCTAs,
                           ArrayRef<int64_t>{1});
  if (failed(materializeLegalConfigs(kernel)) ||
      failed(gpu::verifyGPUProgram(module)))
    return failure();
  if (failed(verifyTritonProgram(module)))
    return failure();
  kernel->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::triton
