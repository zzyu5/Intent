#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

LogicalResult verifyExpressionSymbols(Operation *owner,
                                      PhysicalExprAttr expression,
                                      const llvm::StringSet<> &parameters,
                                      const llvm::StringSet<> &launchABI) {
  auto kind = static_cast<PhysicalExprKind>(expression.getKind());
  if (kind == PhysicalExprKind::Parameter &&
      !parameters.contains(expression.getSymbol().getValue()))
    return owner->emitOpError("launch expression references an undeclared physical parameter");
  if ((kind == PhysicalExprKind::Dimension ||
       kind == PhysicalExprKind::ScalarABI) &&
      !launchABI.contains(expression.getSymbol().getValue()))
    return owner->emitOpError(
               "launch expression references unavailable host metadata: ")
           << expression.getSymbol();
  for (Attribute operand : expression.getOperands())
    if (failed(verifyExpressionSymbols(owner, cast<PhysicalExprAttr>(operand),
                                       parameters, launchABI)))
      return failure();
  return success();
}

void collectTypeExpressions(Type type,
                            SmallVectorImpl<PhysicalExprAttr> &expressions) {
  ArrayAttr shape;
  if (auto view = dyn_cast<ViewType>(type))
    shape = view.getLayout().getExtents();
  else if (auto fragment = dyn_cast<FragmentType>(type))
    shape = fragment.getShape();
  else if (auto buffer = dyn_cast<BufferType>(type))
    shape = buffer.getShape();
  if (!shape)
    return;
  for (Attribute extent : shape)
    expressions.push_back(cast<PhysicalExprAttr>(extent));
}

bool hasObservableEffect(Operation *operation) {
  return isa<StoreOp, ScatterReduceOp, AtomicStoreOp, AtomicRMWOp,
             AtomicCompareExchangeOp>(operation);
}

} // namespace

FailureOr<func::FuncOp> getPhysicalKernel(ModuleOp module) {
  SmallVector<func::FuncOp> kernels;
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (function->hasAttr(kernelAttr))
      kernels.push_back(function);
  if (kernels.size() != 1) {
    module.emitError("shared GPU module requires exactly one physical kernel");
    return failure();
  }
  return kernels.front();
}

LogicalResult verifyGPUProgram(ModuleOp module) {
  if (failed(mlir::verify(module.getOperation())))
    return failure();
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  if (!kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr) ||
      !kernel->getAttrOfType<ArrayAttr>(programSpaceAttr) ||
      !kernel->getAttrOfType<IntegerAttr>(gridRankAttr))
    return kernel.emitError("physical kernel is missing capabilities or launch schema");
  auto programSpace = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  auto expectedEffects = kernel->getAttrOfType<ArrayAttr>(effectOriginsAttr);
  int64_t gridRank = kernel->getAttrOfType<IntegerAttr>(gridRankAttr).getInt();
  if (gridRank <= 0 || programSpace.size() != static_cast<size_t>(gridRank) ||
      !expectedEffects)
    return kernel.emitError("physical program-space rank is invalid");
  for (Attribute extent : programSpace)
    if (!isa<PhysicalExprAttr>(extent))
      return kernel.emitError("program-space extents must be typed physical expressions");

  llvm::StringSet<> parameterNames;
  llvm::StringSet<> launchABI;
  llvm::StringSet<> abiNames;
  for (auto [index, type] : llvm::enumerate(kernel.getArgumentTypes())) {
    DictionaryAttr attrs = kernel.getArgAttrDict(index);
    auto kind = attrs.getAs<StringAttr>(abiKindAttr);
    auto name = attrs.getAs<StringAttr>(abiNameAttr);
    if (!kind || !name || name.empty())
      return kernel.emitError("every physical ABI argument requires a typed role and name");
    if (!abiNames.insert(name.getValue()).second)
      return kernel.emitError("physical ABI argument names must be unique");
    if (kind.getValue() == "view") {
      auto view = dyn_cast<ViewType>(type);
      if (!view || view.getAbiIndex() != index)
        return kernel.emitError("view ABI argument has a non-view physical type");
    } else if (kind.getValue() == "dimension" ||
               kind.getValue() == "stride") {
      auto sourceABI = attrs.getAs<IntegerAttr>(sourceABIAttr);
      auto sourceAxis = attrs.getAs<IntegerAttr>(sourceAxisAttr);
      if (!type.isIndex() || !sourceABI || !sourceAxis)
        return kernel.emitError("metadata ABI argument lacks its source binding");
      int64_t source = sourceABI.getInt();
      int64_t axis = sourceAxis.getInt();
      if (source < 0 || source >= static_cast<int64_t>(kernel.getNumArguments()))
        return kernel.emitError("metadata ABI source is outside the physical signature");
      auto sourceView = dyn_cast<ViewType>(kernel.getArgumentTypes()[source]);
      if (!sourceView || sourceView.getAbiIndex() != source || axis < 0 ||
          axis >= static_cast<int64_t>(sourceView.getRank()))
        return kernel.emitError(
            "metadata ABI source binding does not name an axis of its physical view");
      launchABI.insert(name.getValue());
    } else if (kind.getValue() == "scalar" ||
               kind.getValue() == "constexpr" ||
               kind.getValue() == "value") {
      if (!isa<IntegerType, IndexType, FloatType>(type))
        return kernel.emitError("physical scalar ABI has a non-scalar type");
      if (isa<IntegerType, IndexType>(type))
        launchABI.insert(name.getValue());
    } else if (kind.getValue() == "workspace") {
      auto buffer = dyn_cast<BufferType>(type);
      if (!buffer || !buffer.getWorkspace())
        return kernel.emitError("workspace ABI argument has a non-workspace type");
    } else {
      return kernel.emitError("unknown physical ABI argument kind");
    }
  }
  kernel.walk([&](ParameterOp parameter) {
    parameterNames.insert(parameter.getParameter().getName().getValue());
  });
  for (Attribute extent : programSpace)
    if (failed(verifyExpressionSymbols(kernel, cast<PhysicalExprAttr>(extent),
                                       parameterNames, launchABI)))
      return failure();

  bool hasProgramId = false;
  llvm::DenseSet<int64_t> expectedEffectOrigins;
  for (Attribute origin : expectedEffects) {
    auto node = dyn_cast<IntegerAttr>(origin);
    if (!node || node.getInt() < 0 ||
        !expectedEffectOrigins.insert(node.getInt()).second)
      return kernel.emitError(
          "expected effect origins must be unique non-negative IDs");
  }
  llvm::DenseSet<int64_t> actualEffectOrigins;
  llvm::DenseSet<int64_t> programAxes;
  WalkResult result = kernel.walk([&](Operation *operation) {
    StringRef dialect = operation->getName().getDialectNamespace();
    if (dialect == "intent") {
      operation->emitOpError("canonical KIR is illegal inside a physical GPU kernel");
      return WalkResult::interrupt();
    }
    if (dialect != "intent_gpu" && dialect != "arith" && dialect != "scf" &&
        dialect != "func" && dialect != "builtin") {
      operation->emitOpError("operation dialect is not legal in shared GPU IR");
      return WalkResult::interrupt();
    }
    if (auto parameter = dyn_cast<ParameterOp>(operation)) {
      StringRef name = parameter.getParameter().getName().getValue();
      unsigned count = 0;
      kernel.walk([&](ParameterOp candidate) {
        count += candidate.getParameter().getName().getValue() == name;
      });
      if (count != 1) {
        operation->emitOpError("physical parameter name is duplicated: ")
            << name;
        return WalkResult::interrupt();
      }
    }
    if (auto physical = dyn_cast<PhysicalExprOp>(operation))
      if (failed(verifyExpressionSymbols(operation, physical.getExpression(),
                                         parameterNames, launchABI)))
        return WalkResult::interrupt();
    if (auto program = dyn_cast<ProgramIdOp>(operation)) {
      if (program.getAxis() >= static_cast<uint64_t>(gridRank) ||
          !programAxes.insert(program.getAxis()).second) {
        operation->emitOpError("program coordinate is outside or duplicated in the current grid");
        return WalkResult::interrupt();
      }
      hasProgramId = true;
    }
    if (hasObservableEffect(operation)) {
      auto origin = operation->getAttrOfType<IntegerAttr>(originAttr);
      if (!origin || !actualEffectOrigins.insert(origin.getInt()).second) {
        operation->emitOpError(
            "observable effect requires one unique canonical origin");
        return WalkResult::interrupt();
      }
    }
    SmallVector<PhysicalExprAttr> expressions;
    for (Type type : operation->getResultTypes())
      collectTypeExpressions(type, expressions);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          collectTypeExpressions(argument.getType(), expressions);
    for (PhysicalExprAttr expression : expressions)
      if (failed(verifyExpressionSymbols(operation, expression, parameterNames,
                                         launchABI)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  if (!hasProgramId || programAxes.size() != static_cast<size_t>(gridRank) ||
      actualEffectOrigins != expectedEffectOrigins)
    return kernel.emitError(
        "physical kernel program mapping/effect coverage is incomplete");
  return success();
}

} // namespace intent::gpu
