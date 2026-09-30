#include "Legalization.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "llvm/ADT/DenseSet.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringSet.h"
#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;

namespace intent::triton::detail {
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
    if (!isTritonScalarType(type) ||
        isa<Float8E4M3FNType, Float8E5M2Type>(type))
      return kernel.emitError("Triton physical ABI type is unsupported");
  }

  llvm::SmallDenseSet<uint32_t> providerRoles;
  gpu::ParameterAttr stageParameter;
  LogicalResult parameterSchema = success();
  kernel.walk([&](gpu::ParameterOp parameter) {
    uint32_t role = parameter.getParameter().getRole();
    auto category = static_cast<gpu::ParameterCategory>(
        parameter.getParameter().getCategory());
    bool providerRole =
        role == static_cast<uint32_t>(gpu::ParameterRole::ProviderWarps) ||
        role == static_cast<uint32_t>(gpu::ParameterRole::ProviderStages) ||
        role == static_cast<uint32_t>(gpu::ParameterRole::ProviderCTAs);
    if ((category == gpu::ParameterCategory::Provider) != providerRole) {
      parameter.emitOpError(
          "Triton program contains a foreign provider parameter");
      parameterSchema = failure();
      return;
    }
    if (!providerRole)
      return;
    if (role == static_cast<uint32_t>(gpu::ParameterRole::ProviderStages))
      stageParameter = parameter.getParameter();
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
    if (operation->hasAttr(loopStagesAttr)) {
      auto binding = operation->getAttrOfType<gpu::ParameterAttr>(loopStagesAttr);
      if (!isa<scf::ForOp>(operation) || !binding || binding != stageParameter) {
        operation->emitOpError(
            "loop stages must bind the declared Triton stage parameter");
        return WalkResult::interrupt();
      }
    }
    if (auto assertion = dyn_cast<cf::AssertOp>(operation)) {
      auto compare = assertion.getArg().getDefiningOp<gpu::CompareOp>();
      if (!compare || !llvm::all_of(compare->getOperands(), [&](Value operand) {
            if (auto expression = operand.getDefiningOp<gpu::PhysicalExprOp>())
              return isTritonFragmentExtent(expression.getExpression());
            return operand.getDefiningOp<arith::ConstantOp>() != nullptr;
          })) {
        assertion.emitOpError("Triton assertions require a constexpr condition");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (auto unary = dyn_cast<gpu::UnaryOp>(operation);
        unary && unary.getApproximate() &&
        unary.getOperatorKind() == UnaryOperator::Tanh) {
      auto capabilities =
          kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
      if (!capabilities ||
          10 * capabilities.getComputeCapabilityMajor() +
                  capabilities.getComputeCapabilityMinor() < 75) {
        unary.emitOpError(
            "native approximate tanh requires compute capability 7.5 or newer");
        return WalkResult::interrupt();
      }
    }
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
      std::string reason;
      auto axes = gpu::queryContractionAxes(contract, &reason);
      if (!axes) {
        contract.emitOpError("invalid Triton contraction axis schema: ") << reason;
        return WalkResult::interrupt();
      }
      if (!axes->hasCanonicalMatrixAxes()) {
        contract.emitOpError(
            "requires provider legalization to [...,M,K] x [...,K,N] tl.dot form");
        return WalkResult::interrupt();
      }
      auto form = contract->getAttrOfType<StringAttr>(contractFormAttr);
      if (form && form.getValue() != "multiply_sum" &&
          form.getValue() != "fma") {
        contract.emitOpError("has an unknown Triton contraction form");
        return WalkResult::interrupt();
      }
      if (form && form.getValue() == "fma" &&
          !contract.getAccumulator().getType().getElementType().isF32()) {
        contract.emitOpError(
            "Triton explicit FMA contraction requires f32 accumulation");
        return WalkResult::interrupt();
      }
      if ((!form || form.getValue() != "multiply_sum") &&
          isa<BFloat16Type>(contract.getResult().getType().getElementType())) {
        contract.emitOpError(
            "Triton tl.dot does not support a bfloat16 accumulator; requires precision-preserving provider legalization");
        return WalkResult::interrupt();
      }
    } else if (auto range = dyn_cast<gpu::MakeRangeOp>(operation)) {
      auto fragment = dyn_cast<gpu::FragmentType>(range.getResult().getType());
      auto physicalExtent =
          fragment && fragment.getShape().size() == 1
              ? dyn_cast<gpu::PhysicalExprAttr>(fragment.getShape()[0])
              : gpu::PhysicalExprAttr();
      if (!physicalExtent || !isTritonFragmentExtent(physicalExtent)) {
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
      if (!isa<gpu::FragmentType>(gather.getResult().getType()) &&
          (cast<gpu::FragmentType>(gather.getSource().getType()).getShape().size() != 1 ||
           gather.getSourceAxes().front() != 0 ||
           !isa<IndexType, IntegerType>(gather.getCoordinates().front().getType()))) {
        gather.emitOpError("Triton scalar gather requires one scalar index into a vector");
        return WalkResult::interrupt();
      }
    } else if (isa<gpu::ReduceOp, gpu::ScanOp>(operation)) {
      operation->emitOpError(
          "Triton collective requires an explicitly scalarized callback");
      return WalkResult::interrupt();
    } else if (auto contract = dyn_cast<gpu::ScaledContractOp>(operation)) {
      unsigned lhsRank = contract.getLhs().getType().getShape().size();
      unsigned rhsRank = contract.getRhs().getType().getShape().size();
      Type lhsScaleElement =
          contract.getLhsScale().getType().getElementType();
      Type rhsScaleElement =
          contract.getRhsScale().getType().getElementType();
      if (lhsRank != 3 || rhsRank != 3 ||
          contract.getLhsFormat() == ScaledFormat::E8M0 ||
          contract.getRhsFormat() == ScaledFormat::E8M0 ||
          contract.getLhsGroupSize() != 32 ||
          contract.getRhsGroupSize() != 32 ||
          !lhsScaleElement.isUnsignedInteger(8) ||
          !rhsScaleElement.isUnsignedInteger(8) ||
          !contract.getResult().getType().getElementType().isF32() ||
          contract.getLhsReductionAxes() !=
              ArrayRef<int64_t>{1, 2} ||
          contract.getRhsReductionAxes() !=
              ArrayRef<int64_t>{0, 1}) {
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

    if (isa<CtaBarrierOp, gpu::ViewOverlapOp, TensorDescriptorChoiceOp, TensorDescriptorAllocatorOp,
            TensorDescriptorOp, BlockLoadOp, BlockStoreOp, DescriptorLoadOp,
            DescriptorStoreOp, SplitOp, ReduceOp, ScanOp, MapElementwiseOp, gpu::ParameterOp,
            gpu::PhysicalExprOp,
            gpu::ProgramIdOp,
            gpu::WorksetCoordinateOp, gpu::DelinearizeOp,
            gpu::DimOp, gpu::RangeOp, gpu::RangeBoundOp, gpu::MakeRangeOp,
            gpu::SplatOp, gpu::BroadcastOp, gpu::UnaryOp, gpu::BinaryOp,
            gpu::CompareOp, gpu::SelectOp, gpu::CastOp, gpu::BitcastOp,
            gpu::ReshapeOp, gpu::TransposeOp, gpu::JoinOp, gpu::MakeRecordOp,
            gpu::ExtractOp, gpu::LoadOp, gpu::GatherOp, gpu::StoreOp,
            gpu::ContractOp,
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


} // namespace intent::triton::detail

namespace intent::triton {
LogicalResult verifyTritonProgram(ModuleOp module) {
  if (failed(mlir::verify(module.getOperation())))
    return failure();
  SmallVector<func::FuncOp> kernels;
  module.walk([&](func::FuncOp function) {
    if (function->hasAttr(gpu::kernelAttr))
      kernels.push_back(function);
  });
  if (kernels.size() != 1)
    return module.emitError("Triton provider program requires one physical kernel");
  return detail::verifyKernel(kernels.front());
}

} // namespace intent::triton
