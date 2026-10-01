#include "Intent/Dialect/GPU/Serialization/Interface.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace intent::gpu {
namespace {

std::string typeName(Type type) {
  std::string name;
  llvm::raw_string_ostream stream(name);
  type.print(stream);
  return llvm::StringRef(name).lower();
}

llvm::json::Array expressions(ArrayAttr attributes) {
  llvm::json::Array result;
  for (Attribute attribute : attributes)
    result.push_back(serializeExpression(cast<PhysicalExprAttr>(attribute)));
  return result;
}

llvm::json::Array integers(ArrayRef<int64_t> values) {
  llvm::json::Array result;
  for (int64_t value : values)
    result.push_back(value);
  return result;
}

llvm::json::Array strides(ArrayAttr attributes) {
  llvm::json::Array result;
  for (Attribute attribute : attributes) {
    if (auto value = dyn_cast<IntegerAttr>(attribute))
      result.push_back(llvm::json::Object{{"kind", "constant"},
                                         {"value", value.getInt()}});
    else
      result.push_back(llvm::json::Object{
          {"kind", "scalar"},
          {"symbol", cast<StringAttr>(attribute).getValue()}});
  }
  return result;
}

bool isLaunchSpecialization(PhysicalExprAttr expression) {
  if (expression.getKind() ==
      PhysicalExprKind::ScalarABI)
    return false;
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isLaunchSpecialization(cast<PhysicalExprAttr>(operand));
  });
}

std::optional<llvm::json::Value> resourceExpression(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return llvm::json::Object{{"kind", "constant"}, {"value", constant.value()}};
  if (auto physical = value.getDefiningOp<PhysicalExprOp>())
    if (isLaunchSpecialization(physical.getExpression()))
      return serializeExpression(physical.getExpression());
  return std::nullopt;
}

FailureOr<llvm::json::Array> configurations(const ParameterSpace &space) {
  auto rows = space.configurations(ConfigurationStage::Complete);
  if (failed(rows))
    return failure();
  llvm::json::Array result;
  for (DictionaryAttr bindings : *rows) {
    llvm::json::Object encoded;
    for (NamedAttribute binding : bindings)
      encoded[binding.getName().getValue()] =
          cast<IntegerAttr>(binding.getValue()).getInt();
    result.push_back(std::move(encoded));
  }
  return result;
}

} // namespace

FailureOr<KernelInterface> readInterface(func::FuncOp kernel) {
  KernelInterface result;
  for (auto [index, argument] : llvm::enumerate(kernel.getArguments())) {
    auto attrs = kernel.getArgAttrDict(index);
    auto kindAttr = attrs.getAs<StringAttr>(abiKindAttr);
    auto nameAttr = attrs.getAs<StringAttr>(abiNameAttr);
    if (!kindAttr || !nameAttr) {
      kernel.emitError("GPU argument has no complete typed ABI: ") << index;
      return failure();
    }
    StringRef kind = kindAttr.getValue();
    unsigned abi = index;
    std::string name = nameAttr.getValue().str();
    if (kind == "constexpr") {
      if (!argument.use_empty()) {
        kernel.emitError("live constexpr reached GPU runtime ABI after specialization");
        return failure();
      }
      continue;
    }
    if (kind == "view" || kind == "workspace") {
      auto view = dyn_cast<ViewType>(argument.getType());
      if (!view) {
        kernel.emitError("GPU view ABI argument has no view type: ") << index;
        return failure();
      }
      bool workspace = kind == "workspace";
      result.views.push_back({abi, std::move(name), view, workspace});
      if (!workspace)
        result.publicArguments.push_back(abi);
      continue;
    }
    if (kind == "scalar" || kind == "value") {
      result.scalars.push_back(
          {abi, std::move(name), kind.str(), argument.getType()});
      result.publicArguments.push_back(abi);
      continue;
    }
    auto source = attrs.getAs<IntegerAttr>(sourceABIAttr);
    auto axis = attrs.getAs<IntegerAttr>(sourceAxisAttr);
    auto dimension = attrs.getAs<IntegerAttr>(dimensionAttr);
    if ((kind != "dimension" && kind != "stride") || !source || !axis ||
        (kind == "dimension" && !dimension) || source.getInt() < 0 ||
        source.getInt() >= kernel.getNumArguments() || axis.getInt() < 0) {
      kernel.emitError("GPU metadata argument has no complete typed source: ")
          << index;
      return failure();
    }
    auto sourceView = dyn_cast<ViewType>(
        kernel.getArgument(source.getInt()).getType());
    if (!sourceView || axis.getInt() >= sourceView.getRank()) {
      kernel.emitError("GPU metadata argument does not reference a view axis: ")
          << index;
      return failure();
    }
    result.metadata.push_back(
        {abi, std::move(name), kind.str(),
         static_cast<unsigned>(source.getInt()),
         static_cast<unsigned>(axis.getInt()),
         dimension ? std::optional<int64_t>(dimension.getInt()) : std::nullopt});
  }
  return result;
}

llvm::json::Value serializeExpression(PhysicalExprAttr expression) {
  StringRef kind;
  switch (expression.getKind()) {
  case PhysicalExprKind::Constant:
    return llvm::json::Object{{"kind", "constant"},
                              {"value", expression.getValue()}};
  case PhysicalExprKind::Parameter: kind = "parameter"; break;
  case PhysicalExprKind::Dimension: kind = "dimension"; break;
  case PhysicalExprKind::ScalarABI: kind = "scalar"; break;
  case PhysicalExprKind::Add: kind = "add"; break;
  case PhysicalExprKind::Subtract: kind = "subtract"; break;
  case PhysicalExprKind::Multiply: kind = "multiply"; break;
  case PhysicalExprKind::CeilDiv: kind = "ceil_div"; break;
  case PhysicalExprKind::FloorDiv: kind = "floor_div"; break;
  case PhysicalExprKind::Minimum: kind = "min"; break;
  case PhysicalExprKind::Maximum: kind = "max"; break;
  case PhysicalExprKind::Select: kind = "select"; break;
  case PhysicalExprKind::NextPowerOfTwo: kind = "next_power_of_two"; break;
  default: llvm_unreachable("unverified physical expression kind");
  }
  if (kind == "parameter" || kind == "dimension" || kind == "scalar")
    return llvm::json::Object{{"kind", kind},
                              {"symbol", expression.getSymbolName().getValue()}};
  return llvm::json::Object{{"kind", kind},
                            {"operands", expressions(expression.getOperands())}};
}

FailureOr<llvm::json::Object> serializeInterface(
    func::FuncOp kernel, StringRef provider,
    llvm::function_ref<std::string(Value)> kernelName) {
  auto facts = readInterface(kernel);
  if (failed(facts))
    return failure();
  auto configurationSpace = ParameterSpace::read(kernel);
  if (failed(configurationSpace))
    return failure();
  llvm::json::Array views, scalars, metadata, publicArguments, parameters;
  for (const auto &view : facts->views) {
    auto layout = view.type.getLayout();
    views.push_back(llvm::json::Object{
        {"abi", view.abi}, {"name", view.name},
        {"kernel_name", kernelName(kernel.getArgument(view.abi))},
        {"dtype", typeName(view.type.getElementType())},
        {"access", view.type.getAccess()}, {"workspace", view.workspace},
        {"source_id", view.type.getSourceId()},
        {"shape", expressions(layout.getExtents())},
        {"dimensions", integers(layout.getDimensionIds().asArrayRef())},
        {"has_strides", layout.getHasStrides()},
        {"strides", strides(layout.getStrides())},
        {"alias", layout.getAlias().getValue()}, {"noalias", layout.getNoalias()}});
  }
  for (const auto &scalar : facts->scalars)
    scalars.push_back(llvm::json::Object{
        {"abi", scalar.abi}, {"name", scalar.name}, {"kind", scalar.kind},
        {"kernel_name", kernelName(kernel.getArgument(scalar.abi))},
        {"dtype", typeName(scalar.type)}});
  for (const auto &argument : facts->metadata) {
    llvm::json::Object entry{
        {"abi", argument.abi}, {"name", argument.name}, {"kind", argument.kind},
        {"kernel_name", kernelName(kernel.getArgument(argument.abi))},
        {"source_abi", argument.sourceABI}, {"source_axis", argument.sourceAxis}};
    if (argument.dimension)
      entry["dimension"] = *argument.dimension;
    metadata.push_back(std::move(entry));
  }
  for (unsigned abi : facts->publicArguments)
    publicArguments.push_back(abi);

  for (ParameterAttr declaration : configurationSpace->declarations()) {
    PhysicalParameterBinding binding = queryParameterBinding(declaration);
    llvm::json::Object entry{
        {"name", declaration.getName().getValue()},
        {"role", static_cast<uint32_t>(declaration.getRole())},
        {"category", static_cast<uint32_t>(declaration.getCategory())},
        {"element_bits", declaration.getElementBitWidth()},
        {"candidates", integers(declaration.getCandidates().asArrayRef())}};
    if (declaration.isDeferred()) {
      auto bound = declaration.getBinding().getCoverageBound();
      if (!bound) {
        kernel.emitError("full-coverage declaration has no typed bound expression");
        return failure();
      }
      entry["coverage"] = serializeExpression(bound);
    }
    if (binding.dimension) {
      entry["dimension"] = *binding.dimension;
      for (const auto &argument : facts->metadata) {
        if (argument.kind != "dimension" ||
            argument.dimension != binding.dimension)
          continue;
        auto found = llvm::find(facts->publicArguments, argument.sourceABI);
        if (found != facts->publicArguments.end()) {
          entry["argument_axis"] = llvm::json::Array{
              static_cast<int64_t>(found - facts->publicArguments.begin()),
              argument.sourceAxis};
          break;
        }
      }
    }
    if (binding.source)
      entry["source"] = llvm::json::Array{binding.source->sourceId,
                                         binding.source->sourceAxis,
                                         binding.source->derived};
    parameters.push_back(std::move(entry));
  }

  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!space) {
    kernel.emitError("final GPU program has no launch grid");
    return failure();
  }
  llvm::json::Array overlaps;
  bool valid = true;
  kernel.walk([&](ViewOverlapOp overlap) {
    auto lhs = dyn_cast<BlockArgument>(overlap.getLhs());
    auto rhs = dyn_cast<BlockArgument>(overlap.getRhs());
    if (!lhs || !rhs || lhs.getOwner() != &kernel.front() ||
        rhs.getOwner() != &kernel.front()) {
      overlap.emitError("launch overlap must refer to physical ABI arguments");
      valid = false;
      return;
    }
    overlaps.push_back(llvm::json::Object{
        {"name", kernelName(overlap.getResult())},
        {"lhs", lhs.getArgNumber()}, {"rhs", rhs.getArgNumber()}});
  });
  if (!valid)
    return failure();

  llvm::json::Array bounds;
  for (auto assertion : kernel.front().getOps<cf::AssertOp>()) {
    auto comparison = assertion.getArg().getDefiningOp<CompareOp>();
    if (!comparison || comparison.getPredicate() != ComparePredicate::Le)
      continue;
    auto lhs = resourceExpression(comparison.getLhs());
    auto rhs = resourceExpression(comparison.getRhs());
    if (lhs && rhs)
      bounds.push_back(llvm::json::Object{{"lhs", std::move(*lhs)},
                                         {"rhs", std::move(*rhs)}});
  }
  auto configs = configurations(*configurationSpace);
  if (failed(configs))
    return failure();
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities) {
    kernel.emitError("generated GPU program has no compilation target");
    return failure();
  }
  llvm::json::Object target{
      {"family", "gpu"},
      {"capabilities", llvm::json::Object{
          {"compute_units", capabilities.getComputeUnits()},
          {"shared_memory_per_unit", capabilities.getSharedMemoryPerUnit()},
          {"max_dynamic_shared_memory_per_block", capabilities.getMaxDynamicSharedMemoryPerBlock()},
          {"registers_per_unit", capabilities.getRegistersPerUnit()},
          {"max_threads_per_block", capabilities.getMaxThreadsPerBlock()},
          {"compute_capability_major", capabilities.getComputeCapabilityMajor()},
          {"compute_capability_minor", capabilities.getComputeCapabilityMinor()},
          {"single_to_double_precision_perf_ratio", capabilities.getSingleToDoublePrecisionPerfRatio()},
          {"matrix_units", capabilities.getMatrixUnits()},
          {"dynamic_vector_width", capabilities.getDynamicVectorWidth()}}}};
  llvm::json::Object interface{
      {"views", std::move(views)}, {"scalars", std::move(scalars)},
      {"metadata", std::move(metadata)},
      {"public_arguments", std::move(publicArguments)},
      {"parameters", std::move(parameters)}, {"grid", expressions(space)},
      {"overlaps", std::move(overlaps)},
      {"configurations", std::move(*configs)},
      {"resource_bounds", std::move(bounds)}};
  return llvm::json::Object{{"provider", provider},
                            {"entry_name", kernel.getName()},
                            {"target", std::move(target)},
                            {"interface", std::move(interface)}};
}

} // namespace intent::gpu
