#include "Intent/Target/Triton/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::triton {
namespace {

constexpr llvm::StringLiteral legalizedAttr = "intent_gpu.triton.legalized";

bool isTritonScalarType(Type type) {
  if (type.isIndex() || isa<Float16Type, BFloat16Type, Float32Type,
                            Float64Type>(type))
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
    if (gather.getCoordinates().size() != 1 ||
        !samePhysicalShape(gather.getValid().getType(),
                           gather.getCoordinates().front().getType()) ||
        !samePhysicalShape(gather.getValid().getType(),
                           gather.getResult().getType()))
      continue;
    OpBuilder builder(gather);
    FailureOr<Value> zero =
        zeroLike(builder, gather.getLoc(),
                 gather.getCoordinates().front().getType());
    if (failed(zero))
      continue;
    Value safeIndex = builder.create<gpu::SelectOp>(
        gather.getLoc(), gather.getCoordinates().front().getType(),
        gather.getValid(), gather.getCoordinates().front(), *zero);
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
  Block &block = scatter.getCombine().front();
  if (block.getNumArguments() != 2 ||
      std::distance(block.begin(), block.end()) != 2)
    return false;
  auto binary = dyn_cast<gpu::BinaryOp>(block.front());
  auto yield = dyn_cast<gpu::YieldOp>(block.back());
  if (!binary || !yield || binary.getOperatorKind() != 0 ||
      yield.getValues().size() != 1 ||
      yield.getValues().front() != binary.getResult())
    return false;
  Value lhs = binary.getLhs();
  Value rhs = binary.getRhs();
  return (lhs == block.getArgument(0) && rhs == block.getArgument(1)) ||
         (lhs == block.getArgument(1) && rhs == block.getArgument(0));
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
        /*kind=*/1, /*ordering=*/0, scatter.getSharing(),
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
    if (role < static_cast<uint32_t>(gpu::ParameterRole::ProviderWarps))
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
      if (store.getCollision() != 0 ||
          failed(verifyAccess(operation, store.getResource(),
                              store.getCoordinates(), store.getSourceAxes()))) {
        if (store.getCollision() != 0)
          store.emitOpError(
              "requires collision legalization before Triton serialization");
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
      if (!range.getExtent().getDefiningOp<arith::ConstantOp>() &&
          !range.getExtent().getDefiningOp<gpu::ParameterOp>() &&
          !range.getExtent().getDefiningOp<gpu::PhysicalExprOp>()) {
        InFlightDiagnostic diagnostic = range.emitOpError(
            "Triton tl.arange extent must be compile-time bound; extent is defined by ");
        if (Operation *definition = range.getExtent().getDefiningOp())
          diagnostic << definition->getName();
        else
          diagnostic << "a block argument";
        diagnostic << (range.getResult().use_empty() ? " and is dead"
                                                      : " and still has live uses");
        for (Operation *user : range.getResult().getUsers()) {
          diagnostic << " " << user->getName();
          if (auto load = dyn_cast<gpu::LoadOp>(user)) {
            if (auto view = dyn_cast<gpu::ViewType>(load.getResource().getType()))
              diagnostic << "(abi=" << view.getAbiIndex() << ",source_axes=["
                         << load.getSourceAxes() << "])";
          }
        }
        return WalkResult::interrupt();
      }
    } else if (auto gather = dyn_cast<gpu::GatherOp>(operation)) {
      if (gather.getValid()) {
        gather.emitOpError(
            "requires safe-index legalization before Triton tl.gather");
        return WalkResult::interrupt();
      }
      if (gather.getCoordinates().size() != 1 ||
          gather.getSourceAxes().size() != 1 ||
          gather.getSourceAxes().front() != 0) {
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
          contract.getLhsFormat() > 1 || contract.getRhsFormat() > 1 ||
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
            gpu::DelinearizeOp,
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
                           ArrayRef<int64_t>{4, 8});
  declareProviderParameter("NUM_STAGES", gpu::ParameterRole::ProviderStages,
                           ArrayRef<int64_t>{3, 4});
  declareProviderParameter("NUM_CTAS", gpu::ParameterRole::ProviderCTAs,
                           ArrayRef<int64_t>{1});
  if (failed(verifyTritonProgram(module)))
    return failure();
  kernel->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::triton
