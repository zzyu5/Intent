#include "Intent/Target/CuTile/Transforms/Passes.h"
#include "Configurations.h"
#include "Legalize.h"
#include "NativeAccess.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Resources.h"
#include "Intent/Dialect/GPU/Transforms/Contraction.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MathExtras.h"

#include <functional>
#include <optional>

using namespace mlir;

namespace intent::cutile {
namespace {

constexpr llvm::StringLiteral legalizedAttr = "intent_cutile.legalized";
bool isCuTileScalarType(Type type) {
  if (type.isIndex() ||
      isa<Float16Type, BFloat16Type, Float32Type, Float64Type, Float8E4M3FNType,
          Float8E5M2Type>(type))
    return true;
  auto integer = dyn_cast<IntegerType>(type);
  return integer && (integer.getWidth() == 1 || integer.getWidth() == 8 ||
                     integer.getWidth() == 16 || integer.getWidth() == 32 ||
                     integer.getWidth() == 64);
}

gpu::PhysicalExprAttr arrayIndexTileBound(gpu::PhysicalExprAttr expression,
                                         func::FuncOp kernel) {
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  if (kind == gpu::PhysicalExprKind::Constant)
    return expression.getValue() > 0 ? expression : gpu::PhysicalExprAttr();
  if (kind == gpu::PhysicalExprKind::Parameter) {
    auto parameter = gpu::queryParameterBySymbol(kernel, expression.getSymbol());
    if (failed(parameter))
      return {};
    if ((*parameter)->hasAttr(gpu::coverageDimensionAttr))
      return expression;
    auto configurations = kernel->getAttrOfType<gpu::ConfigurationSetAttr>(gpu::configurationsAttr);
    if (!configurations || configurations.getStage() != gpu::ConfigurationStage::Complete)
      return {};
    int64_t maximum = 0;
    for (Attribute configuration : configurations.getRows()) {
      auto tuple = dyn_cast<DictionaryAttr>(configuration);
      auto value = tuple ? tuple.getAs<IntegerAttr>(expression.getSymbol()) : IntegerAttr();
      if (!value || value.getInt() <= 0)
        return {};
      maximum = std::max(maximum, value.getInt());
    }
    return gpu::PhysicalExprAttr::get(
        kernel.getContext(), static_cast<uint32_t>(gpu::PhysicalExprKind::Constant),
        maximum, StringAttr::get(kernel.getContext()),
        ArrayAttr::get(kernel.getContext(), {}));
  }
  if (kind != gpu::PhysicalExprKind::Add &&
      kind != gpu::PhysicalExprKind::Multiply &&
      kind != gpu::PhysicalExprKind::Minimum &&
      kind != gpu::PhysicalExprKind::Maximum)
    return {};
  SmallVector<Attribute> operands;
  for (Attribute operand : expression.getOperands()) {
    auto bound = arrayIndexTileBound(cast<gpu::PhysicalExprAttr>(operand), kernel);
    if (!bound && kind != gpu::PhysicalExprKind::Minimum)
      return {};
    if (bound)
      operands.push_back(bound);
  }
  // A bounded tile min(chunk, runtime_shape) is no wider than chunk. The
  // unknown shape operand must not discard that already proven upper bound.
  if (operands.empty())
    return {};
  if (operands.size() == 1)
    return cast<gpu::PhysicalExprAttr>(operands.front());
  return gpu::PhysicalExprAttr::get(
      kernel.getContext(), expression.getKind(), expression.getValue(),
      expression.getSymbol(), ArrayAttr::get(kernel.getContext(), operands));
}

ArrayAttr arrayIndexTileBounds(func::FuncOp kernel) {
  MLIRContext *context = kernel.getContext();
  auto one = gpu::PhysicalExprAttr::get(
      context, static_cast<uint32_t>(gpu::PhysicalExprKind::Constant), 1,
      StringAttr::get(context), ArrayAttr::get(context, {}));
  SmallVector<SmallVector<Attribute>> bounds(kernel.getNumArguments());
  for (BlockArgument argument : kernel.getArguments())
    if (auto view = dyn_cast<gpu::ViewType>(argument.getType()))
      bounds[argument.getArgNumber()].assign(view.getRank(), one);
  bool hasArrayAccess = false;
  auto result = kernel.walk([&](Operation *operation) {
    Value resource;
    gpu::FragmentType tile;
    if (auto load = dyn_cast<TileLoadOp>(operation)) {
      if (load.getResource().getDefiningOp<ArrayViewOp>()) {
        auto original = unfoldedArrayLoad(load);
        if (failed(original) || failed(load.verify()))
          return WalkResult::interrupt();
        // The original rectangular padding also bounds its contiguous alias.
        load = *original;
      }
      resource = load.getResource();
      tile = load.getResult().getType();
    } else if (auto store = dyn_cast<TileStoreOp>(operation)) {
      resource = store.getResource();
      tile = store.getValue().getType();
    } else if (auto atomic = dyn_cast<TileAtomicAddOp>(operation)) {
      resource = atomic.getResource();
      tile = atomic.getValue().getType();
    } else if (isa<ScalarLoadOp, ScalarStoreOp, GatherLoadOp, ScatterStoreOp,
                   AtomicRMWOp>(operation)) {
      // Gather/scatter use element coordinates.  Their active coordinates must
      // already be in bounds before narrowing the array's index arithmetic.
      if (!operation->hasAttr("in_bounds"))
        return WalkResult::interrupt();
      resource = operation->getOperand(0);
    } else {
      return WalkResult::advance();
    }
    hasArrayAccess = true;
    auto argument = dyn_cast<BlockArgument>(resource);
    if (!argument || argument.getOwner() != &kernel.getBody().front())
      return WalkResult::interrupt();
    if (!tile)
      return WalkResult::advance();
    auto &viewBounds = bounds[argument.getArgNumber()];
    if (tile.getShape().size() != viewBounds.size())
      return WalkResult::interrupt();
    for (auto [axis, extent] : llvm::enumerate(tile.getShape())) {
      auto bound = arrayIndexTileBound(cast<gpu::PhysicalExprAttr>(extent), kernel);
      if (!bound)
        return WalkResult::interrupt();
      if (viewBounds[axis] == one || viewBounds[axis] == bound)
        viewBounds[axis] = bound;
      else
        viewBounds[axis] = gpu::PhysicalExprAttr::get(
            context, static_cast<uint32_t>(gpu::PhysicalExprKind::Maximum), 0,
            StringAttr::get(context), ArrayAttr::get(context, {viewBounds[axis], bound}));
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted() || !hasArrayAccess)
    return {};
  SmallVector<Attribute> encoded;
  for (const auto &shape : bounds)
    encoded.push_back(ArrayAttr::get(context, shape));
  return ArrayAttr::get(context, encoded);
}

void preserveNativeIndexValues(func::FuncOp kernel) {
  kernel.walk([](Operation *operation) {
    if (!isa<TileLoadOp, TileStoreOp, TileAtomicAddOp>(operation))
      return;
    for (OpOperand &operand : operation->getOpOperands()) {
      auto cast = operand.get().getDefiningOp<gpu::CastOp>();
      if (cast && cast.getResult().getType().isIndex() &&
          cast.getValue().getType().isSignlessInteger(32))
        operand.set(cast.getValue());
    }
  });
}

LogicalResult verifyKernel(func::FuncOp kernel) {
  if (Attribute bounds = kernel->getAttr(arrayIndexTileBoundsAttr))
    if (bounds != arrayIndexTileBounds(kernel))
      return kernel.emitError("cuTile array-index bounds do not cover the current native accesses");
  auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
  if (!space || space.size() != 1)
    return kernel.emitError(
        "cuTile provider currently requires one explicit linear program space");
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities)
    return kernel.emitError("cuTile provider requires selected GPU capabilities");
  gpu::ParameterOp accessForm;
  gpu::ParameterOp occupancy;
  gpu::ParameterOp ctas;
  gpu::ParameterOp workerWarps;
  gpu::ParameterOp loadPolicy;
  LogicalResult parameterSchema = success();
  kernel.walk([&](gpu::ParameterOp parameter) {
    auto schema = parameter.getParameter();
    auto role = static_cast<gpu::ParameterRole>(schema.getRole());
    auto category =
        static_cast<gpu::ParameterCategory>(schema.getCategory());
    bool provider = category == gpu::ParameterCategory::Provider;
    bool cuTileProvider = isCuTileProviderRole(role);
    if (provider != cuTileProvider) {
      parameter.emitOpError(
          "cuTile program contains a foreign provider parameter");
      parameterSchema = failure();
      return;
    }
    if (!provider)
      return;
    ArrayRef<int64_t> candidates = schema.getCandidates().asArrayRef();
    if (role == gpu::ParameterRole::ProviderAccessForm) {
      if (accessForm) {
        parameter.emitOpError("duplicates the cuTile access-form parameter");
        parameterSchema = failure();
        return;
      }
      if (schema.getName().getValue() != accessFormParameter ||
          candidates.empty() || !llvm::all_of(candidates, isLegalAccessForm)) {
        parameter.emitOpError("has an invalid cuTile access-form domain");
        parameterSchema = failure();
        return;
      }
      accessForm = parameter;
      return;
    }
    if (role == gpu::ParameterRole::ProviderCTAs) {
      if (ctas || schema.getName().getValue() != ctasParameter ||
          candidates.empty() || !llvm::all_of(candidates, isLegalCTAs) ||
          !parameter.getResult().use_empty()) {
        parameter.emitOpError("has an invalid or duplicate cuTile CTA hint schema");
        parameterSchema = failure();
        return;
      }
      ctas = parameter;
      return;
    }
    if (role == gpu::ParameterRole::ProviderWarps) {
      if (workerWarps || schema.getName().getValue() != workerWarpsParameter ||
          candidates.empty() || !llvm::all_of(candidates, isLegalWorkerWarps) ||
          !parameter.getResult().use_empty()) {
        parameter.emitOpError(
            "has an invalid or duplicate cuTile worker-warp hint schema");
        parameterSchema = failure();
        return;
      }
      workerWarps = parameter;
      return;
    }
    if (role == gpu::ParameterRole::ProviderLoadPolicy) {
      if (loadPolicy || schema.getName().getValue() != loadPolicyParameter ||
          candidates.empty() || !llvm::all_of(candidates, isLegalLoadPolicy) ||
          parameter.getResult().use_empty()) {
        parameter.emitOpError("has an invalid cuTile load-latency domain");
        parameterSchema = failure();
        return;
      }
      for (OpOperand &use : parameter.getResult().getUses()) {
        auto load = dyn_cast<TileLoadOp>(use.getOwner());
        auto gather = dyn_cast<GatherLoadOp>(use.getOwner());
        if ((!load || load.getLatencyPolicy() != parameter.getResult()) &&
            (!gather || gather.getLatencyPolicy() != parameter.getResult())) {
          parameter.emitOpError(
              "cuTile load latency must bind a tile or gather load");
          parameterSchema = failure();
          return;
        }
      }
      loadPolicy = parameter;
      return;
    }
    if (occupancy) {
      parameter.emitOpError("duplicates the cuTile occupancy parameter");
      parameterSchema = failure();
      return;
    }
    if (schema.getName().getValue() != occupancyParameter ||
        candidates.empty() || !llvm::all_of(candidates, isLegalOccupancy) ||
        !parameter.getResult().use_empty()) {
      parameter.emitOpError(
          "has an invalid cuTile occupancy hint schema");
      parameterSchema = failure();
      return;
    }
    occupancy = parameter;
  });
  if (failed(parameterSchema))
    return failure();
  if (failed(verifyClosedConfigs(kernel)))
    return failure();
  const NativeProgramFeatures features = queryNativeProgramFeatures(kernel);
  if (features.occupancySensitive != static_cast<bool>(occupancy))
    return kernel.emitError(
        "cuTile occupancy hint does not match occupancy-sensitive tile compute");
  if (features.matrixCompute != static_cast<bool>(ctas))
    return kernel.emitError("cuTile CTA hint does not match matrix tile compute");
  auto isNativeTMACondition = [&](Value value) {
    auto compare = value.getDefiningOp<gpu::CompareOp>();
    return accessForm && compare &&
           compare.getPredicate() == ComparePredicate::Ne &&
           compare.getLhs() == accessForm.getResult() &&
           gpu::IndexRelations().constant(compare.getRhs()) == nativeNoTMAForm;
  };
  WalkResult result = kernel.walk([&](Operation *operation) {
    if (auto unary = dyn_cast<gpu::UnaryOp>(operation);
        unary && (unary.getOperatorKind() == UnaryOperator::Asin ||
                  unary.getOperatorKind() == UnaryOperator::Erf ||
                  unary.getOperatorKind() == UnaryOperator::Erfc ||
                  unary.getOperatorKind() == UnaryOperator::I0 ||
                  unary.getOperatorKind() == UnaryOperator::Lgamma ||
                  unary.getOperatorKind() == UnaryOperator::Log1p)) {
      Type type = unary.getInput().getType();
      if (auto fragment = dyn_cast<gpu::FragmentType>(type))
        type = fragment.getElementType();
      if (type.isF64()) {
        unary.emitOpError("cuTile library lowering currently requires f32 or narrower inputs");
        return WalkResult::interrupt();
      }
    }
    if (auto unary = dyn_cast<gpu::UnaryOp>(operation);
        unary && unary.getApproximate() &&
        unary.getOperatorKind() == UnaryOperator::Tanh &&
        10 * capabilities.getComputeCapabilityMajor() +
                capabilities.getComputeCapabilityMinor() < 75) {
      unary.emitOpError(
          "native approximate tanh requires compute capability 7.5 or newer");
      return WalkResult::interrupt();
    }
    if (isa<gpu::LoadOp, gpu::StoreOp, gpu::ContractOp>(operation)) {
      operation->emitOpError(
          "was not converted to an explicit cuTile tile/MMA form");
      return WalkResult::interrupt();
    }
    if (auto load = dyn_cast<TileLoadOp>(operation)) {
      if (!isNativeTMACondition(load.getAllowTma())) {
        load.emitOpError(
            "allow_tma is not the typed cuTile access-form decision");
        return WalkResult::interrupt();
      }
    }
    if (auto store = dyn_cast<TileStoreOp>(operation)) {
      if (!isNativeTMACondition(store.getAllowTma())) {
        store.emitOpError(
            "allow_tma is not the typed cuTile access-form decision");
        return WalkResult::interrupt();
      }
    }
    if (auto scaled = dyn_cast<ScaledMMAOp>(operation)) {
      if (!supportsE8M0ScaledMMA(capabilities)) {
        scaled.emitOpError(
            "requires compute capability 10.0 or newer for E8M0 scaled MMA");
        return WalkResult::interrupt();
      }
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation))
      if (!loop.getInductionVar().getType().isSignlessInteger(32)) {
        loop.emitOpError("cuTile native for requires an i32 induction variable");
        return WalkResult::interrupt();
      }
    for (Type type : operation->getResultTypes()) {
      if (auto fragment = dyn_cast<gpu::FragmentType>(type)) {
        if (!isCuTileScalarType(fragment.getElementType())) {
          operation->emitOpError("contains a fragment dtype outside cuTile");
          return WalkResult::interrupt();
        }
      } else if (!isa<gpu::ViewType, gpu::RangeType, gpu::RecordType>(type) &&
                 !isCuTileScalarType(type)) {
        operation->emitOpError("has a result type outside the cuTile surface");
        return WalkResult::interrupt();
      }
    }
    if (auto assertion = dyn_cast<cf::AssertOp>(operation)) {
      auto comparison = assertion.getArg().getDefiningOp<gpu::CompareOp>();
      if (assertion->getBlock() != &kernel.front() || !comparison ||
          comparison.getPredicate() != ComparePredicate::Le ||
          !comparison.getLhs().getDefiningOp<gpu::PhysicalExprOp>() ||
          !comparison.getRhs().getDefiningOp<arith::ConstantIndexOp>()) {
        assertion.emitOpError("cuTile resource assertion requires a constexpr physical bound");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (isa<ArrayViewOp, TileLoadOp, TileStoreOp, TileAtomicAddOp,
            ScalarLoadOp, ScalarStoreOp, GatherLoadOp,
            ScatterStoreOp, AtomicRMWOp, ExtractOp, MMAOp, ScaledMMAOp,
            ReduceOp, ScanOp, gpu::ParameterOp,
            gpu::PhysicalExprOp, gpu::ProgramIdOp, gpu::WorksetCoordinateOp,
            gpu::DelinearizeOp, gpu::ViewOverlapOp,
            gpu::DimOp, gpu::RangeOp, gpu::RangeBoundOp, gpu::MakeRangeOp,
            gpu::SplatOp, gpu::BroadcastOp, gpu::UnaryOp, gpu::BinaryOp,
            gpu::CompareOp, gpu::SelectOp, gpu::CastOp, gpu::BitcastOp,
            gpu::ReshapeOp, gpu::TransposeOp, gpu::JoinOp, gpu::MakeRecordOp,
            gpu::ExtractOp, gpu::YieldOp, arith::ConstantOp,
            scf::ForOp, scf::WhileOp, scf::ConditionOp, scf::IfOp, scf::YieldOp, func::FuncOp,
            func::ReturnOp>(operation))
      return WalkResult::advance();
    operation->emitOpError("is outside the closed cuTile provider surface");
    return WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

bool fitsNativeLoopBound(Value value, unsigned depth = 0) {
  if (depth >= 32)
    return false;
  value = stripIndexIdentities(value);
  if (value.getDefiningOp<gpu::ProgramIdOp>())
    return true;
  if (auto expression = value.getDefiningOp<gpu::PhysicalExprOp>()) {
    auto bounds = gpu::queryPositiveExtentBounds(
        expression.getExpression(), expression->getParentOfType<func::FuncOp>());
    return bounds && llvm::isInt<32>(bounds->second);
  }
  auto integer = dyn_cast<IntegerType>(value.getType());
  if (integer && (integer.getWidth() < 32 ||
                  (integer.getWidth() == 32 && !integer.isUnsigned())))
    return true;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto attribute = dyn_cast<IntegerAttr>(constant.getValue());
    if (!attribute)
      return false;
    return integer && integer.isUnsigned()
               ? attribute.getValue().getActiveBits() < 32
               : attribute.getValue().isSignedIntN(32);
  }
  if (auto upper = gpu::queryNonNegativeIndexUpperBound(value)) {
    auto bounds = gpu::queryPositiveExtentBounds(
        upper, value.getParentRegion()->getParentOfType<func::FuncOp>());
    if (bounds && llvm::isInt<32>(bounds->second))
      return true;
  }
  if (auto parameter = value.getDefiningOp<gpu::ParameterOp>())
    return llvm::all_of(parameter.getParameter().getCandidates().asArrayRef(),
                        [](int64_t candidate) { return llvm::isInt<32>(candidate); });
  if (auto bound = value.getDefiningOp<gpu::RangeBoundOp>())
    if (auto range = bound.getRange().getDefiningOp<gpu::RangeOp>())
      return fitsNativeLoopBound(bound.getBound() == 0 ? range.getStart()
                                 : bound.getBound() == 1 ? range.getStop()
                                                        : range.getStep(),
                                 depth + 1);
  if (auto cast = value.getDefiningOp<gpu::CastOp>())
    if (value.getType().isIndex() ||
        (integer && integer.getWidth() == 64 && !integer.isUnsigned()))
      return fitsNativeLoopBound(cast.getValue(), depth + 1);
  if (auto clamp = value.getDefiningOp<gpu::BinaryOp>()) {
    auto kind = clamp.getOperatorKind();
    if (kind != BinaryOperator::Minimum && kind != BinaryOperator::Maximum)
      return false;
    BinaryOperator opposite = kind == BinaryOperator::Minimum
                                  ? BinaryOperator::Maximum
                                  : BinaryOperator::Minimum;
    for (auto [limit, nestedValue] :
         {std::pair{clamp.getLhs(), clamp.getRhs()},
          std::pair{clamp.getRhs(), clamp.getLhs()}}) {
      auto nested = nestedValue.getDefiningOp<gpu::BinaryOp>();
      if (fitsNativeLoopBound(limit, depth + 1) && nested &&
          nested.getOperatorKind() == opposite &&
          (fitsNativeLoopBound(nested.getLhs(), depth + 1) ||
           fitsNativeLoopBound(nested.getRhs(), depth + 1)))
        return true;
    }
  }
  return false;
}

std::optional<int64_t> maximumNativeLoopStep(Value value) {
  value = stripIndexIdentities(value);
  if (auto expression = value.getDefiningOp<gpu::PhysicalExprOp>()) {
    auto bounds = gpu::queryPositiveExtentBounds(
        expression.getExpression(), expression->getParentOfType<func::FuncOp>());
    if (bounds && llvm::isInt<32>(bounds->second))
      return bounds->second;
  }
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto attribute = dyn_cast<IntegerAttr>(constant.getValue());
    if (attribute && attribute.getValue().isSignedIntN(32) &&
        attribute.getInt() > 0)
      return attribute.getInt();
  }
  if (auto parameter = value.getDefiningOp<gpu::ParameterOp>()) {
    ArrayRef<int64_t> candidates =
        parameter.getParameter().getCandidates().asArrayRef();
    if (!candidates.empty() && llvm::all_of(candidates, [](int64_t candidate) {
          return candidate > 0 && llvm::isInt<32>(candidate);
        }))
      return *llvm::max_element(candidates);
  }
  return std::nullopt;
}

void realizeWideLoops(func::FuncOp kernel) {
  SmallVector<scf::ForOp> loops;
  kernel.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
    Type type = loop.getInductionVar().getType();
    if (type.isIndex() || type.isInteger(64))
      loops.push_back(loop);
  });
  for (scf::ForOp loop : loops) {
    OpBuilder builder(loop);
    Location location = loop.getLoc();
    std::optional<int64_t> maximumStep = maximumNativeLoopStep(loop.getStep());
    auto isInductionQuotient = [&](Operation &operation) {
      auto binary = dyn_cast<gpu::BinaryOp>(operation);
      return binary && binary.getOperatorKind() == BinaryOperator::FloorDivide &&
             binary.getLhs() == loop.getInductionVar() &&
             gpu::samePhysicalScalarExpression(binary.getRhs(), loop.getStep());
    };
    bool useTileCounter = maximumStep && *maximumStep > 1 &&
                          gpu::IndexRelations().nonnegative(loop.getLowerBound()) &&
                          gpu::IndexRelations().multipleOf(loop.getLowerBound(), loop.getStep()) &&
                          llvm::any_of(loop.getBody()->without_terminator(),
                                       isInductionQuotient);
    auto nativeLoop = [&](OpBuilder &nested) {
      Value lowerBound = loop.getLowerBound();
      Value upperBound = loop.getUpperBound();
      if (useTileCounter) {
        Type wideType = loop.getInductionVar().getType();
        Value one = nested.create<arith::ConstantOp>(
            location, wideType, nested.getIntegerAttr(wideType, 1));
        Value positive = nested.create<gpu::CompareOp>(
            location, nested.getI1Type(), upperBound, loop.getLowerBound(),
            ComparePredicate::Gt);
        Value distance = nested.create<gpu::SelectOp>(
            location, wideType, positive, upperBound, loop.getLowerBound());
        Value adjustment = nested.create<gpu::BinaryOp>(
            location, wideType, loop.getStep(), one, BinaryOperator::Subtract);
        Value rounded = nested.create<gpu::BinaryOp>(
            location, wideType, distance, adjustment, BinaryOperator::Add);
        upperBound = nested.create<gpu::BinaryOp>(
            location, wideType, rounded, loop.getStep(), BinaryOperator::FloorDivide);
        lowerBound = nested.create<gpu::BinaryOp>(
            location, wideType, lowerBound, loop.getStep(), BinaryOperator::FloorDivide);
      }
      Value lower = nested.create<gpu::CastOp>(
          location, nested.getI32Type(), lowerBound);
      Value upper = nested.create<gpu::CastOp>(
          location, nested.getI32Type(), upperBound);
      Value step = nested.create<gpu::CastOp>(
          location, nested.getI32Type(), loop.getStep());
      if (useTileCounter)
        step = nested.create<arith::ConstantIntOp>(location, 1, 32);
      auto replacement = nested.create<scf::ForOp>(
          location, lower, upper, step, loop.getInitArgs(),
          [&](OpBuilder &body, Location bodyLoc, Value induction,
              ValueRange carried) {
            IRMapping mapping;
            Value counter = body.create<gpu::CastOp>(
                bodyLoc, loop.getInductionVar().getType(), induction);
            Value wide = counter;
            if (useTileCounter)
              wide = body.create<gpu::BinaryOp>(
                  bodyLoc, wide.getType(), counter, loop.getStep(), BinaryOperator::Multiply);
            mapping.map(loop.getInductionVar(), wide);
            mapping.map(loop.getRegionIterArgs(), carried);
            for (Operation &operation : loop.getBody()->without_terminator()) {
              // IV = counter * step, so its exact tile quotient is the counter.
              if (useTileCounter && isInductionQuotient(operation)) {
                Value quotient = counter;
                Type resultType = operation.getResult(0).getType();
                if (quotient.getType() != resultType)
                  quotient = body.create<gpu::CastOp>(bodyLoc, resultType, quotient);
                mapping.map(operation.getResult(0), quotient);
                continue;
              }
              body.clone(operation, mapping);
            }
            SmallVector<Value> yielded;
            for (Value value :
                 cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperands())
              yielded.push_back(mapping.lookupOrDefault(value));
            body.create<scf::YieldOp>(bodyLoc, yielded);
          });
      replacement->setAttrs(loop->getAttrs());
      return replacement;
    };
    auto wideLoop = [&](OpBuilder &nested) {
      OpBuilder::InsertionGuard guard(nested);
      SmallVector<Value> initial{loop.getLowerBound()};
      llvm::append_range(initial, loop.getInitArgs());
      SmallVector<Type> types;
      for (Value value : initial)
        types.push_back(value.getType());
      SmallVector<Location> locations(types.size(), location);
      auto replacement = nested.create<scf::WhileOp>(location, types, initial);
      if (Attribute origin = loop->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      Block *before = nested.createBlock(&replacement.getBefore(), {}, types, locations);
      Value condition = nested.create<gpu::CompareOp>(
          location, nested.getI1Type(), before->getArgument(0),
          loop.getUpperBound(), ComparePredicate::Lt);
      nested.create<scf::ConditionOp>(location, condition, before->getArguments());
      Block *after = nested.createBlock(&replacement.getAfter(), {}, types, locations);
      IRMapping mapping;
      mapping.map(loop.getInductionVar(), after->getArgument(0));
      mapping.map(loop.getRegionIterArgs(), after->getArguments().drop_front());
      for (Operation &operation : loop.getBody()->without_terminator())
        nested.clone(operation, mapping);
      Value next = nested.create<gpu::BinaryOp>(
          location, types.front(), after->getArgument(0), loop.getStep(),
          BinaryOperator::Add);
      SmallVector<Value> yielded{next};
      for (Value value : cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperands())
        yielded.push_back(mapping.lookupOrDefault(value));
      nested.create<scf::YieldOp>(location, yielded);
      return replacement;
    };
    // Unit steps keep the terminating increment in range as well as the body IV.
    bool unitStep = maximumStep && *maximumStep == 1;
    bool boundedStep = unitStep;
    if (!boundedStep && maximumStep)
      if (auto upper = gpu::queryNonNegativeIndexUpperBound(loop.getUpperBound())) {
        auto bounds = gpu::queryPositiveExtentBounds(upper, kernel);
        boundedStep = bounds &&
                      bounds->second <= int64_t{INT32_MAX} - *maximumStep + 1;
      }
    if (boundedStep && fitsNativeLoopBound(loop.getLowerBound()) &&
        fitsNativeLoopBound(loop.getUpperBound())) {
      auto replacement = nativeLoop(builder);
      loop.replaceAllUsesWith(replacement.getResults());
      loop.erase();
      continue;
    }
    auto integer = dyn_cast<IntegerType>(loop.getInductionVar().getType());
    // Runtime scalar bounds can use the same checked native loop as specialized
    // bounds.  The predicate proves the i32 range before narrowing; genuinely
    // wide domains retain the original explicit i64 induction state.
    if (maximumStep && (!integer || !integer.isUnsigned())) {
      Value minimum = builder.create<arith::ConstantOp>(
          location, loop.getInductionVar().getType(),
          builder.getIntegerAttr(loop.getInductionVar().getType(), INT32_MIN));
      Value maximum = builder.create<arith::ConstantOp>(
          location, loop.getInductionVar().getType(),
          builder.getIntegerAttr(loop.getInductionVar().getType(), INT32_MAX));
      // The last body IV is at most upper - 1; its increment must also fit.
      Value maximumUpper = builder.create<arith::ConstantOp>(
          location, loop.getInductionVar().getType(),
          builder.getIntegerAttr(loop.getInductionVar().getType(),
                                 int64_t{INT32_MAX} - *maximumStep + 1));
      Value condition;
      for (Value bound : {loop.getLowerBound(), loop.getUpperBound()}) {
        if (bound != loop.getUpperBound() && fitsNativeLoopBound(bound))
          continue;
        Value limit = bound == loop.getUpperBound() ? maximumUpper : maximum;
        Value fits;
        if (gpu::PhysicalExprAttr upperBound =
                gpu::queryNonNegativeIndexUpperBound(bound)) {
          Value symbolic = builder.create<gpu::PhysicalExprOp>(
              location, builder.getIndexType(), upperBound);
          fits = builder.create<gpu::CompareOp>(
              location, builder.getI1Type(), symbolic, limit, ComparePredicate::Le);
        } else {
          Value lower = builder.create<gpu::CompareOp>(
              location, builder.getI1Type(), bound, minimum, ComparePredicate::Ge);
          Value upper = builder.create<gpu::CompareOp>(
              location, builder.getI1Type(), bound, limit, ComparePredicate::Le);
          fits = builder.create<gpu::BinaryOp>(
              location, builder.getI1Type(), lower, upper, BinaryOperator::LogicalAnd);
        }
        condition = condition ? Value(builder.create<gpu::BinaryOp>(
                                    location, builder.getI1Type(), condition, fits,
                                    BinaryOperator::LogicalAnd))
                              : fits;
      }
      auto replacement = builder.create<scf::IfOp>(
          location, condition,
          [&](OpBuilder &nested, Location nestedLocation) {
            auto native = nativeLoop(nested);
            nested.create<scf::YieldOp>(nestedLocation, native.getResults());
          },
          [&](OpBuilder &nested, Location nestedLocation) {
            auto wide = wideLoop(nested);
            nested.create<scf::YieldOp>(nestedLocation, wide.getResults().drop_front());
          });
      loop.replaceAllUsesWith(replacement.getResults());
      loop.erase();
      continue;
    }
    // cuTile range always has an i32 IV. Unproven bounds retain explicit wide state.
    auto replacement = wideLoop(builder);
    loop.replaceAllUsesWith(replacement.getResults().drop_front());
    loop.erase();
  }
}

} // namespace

LogicalResult verifyCuTileProgram(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  return failed(kernel) || failed(mlir::verify(module)) ? failure()
                                                       : verifyKernel(*kernel);
}

LogicalResult prepareProgram(ModuleOp module) {
  if (failed(gpu::verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  if (failed(gpu::contraction::normalizeMatrixContractShapes(*kernel)) ||
      failed(gpu::verifyGPUProgram(module)))
    return failure();
  if (failed(gpu::materializeProgramBuffers(module)) ||
      failed(gpu::lowerInvocationWorkspaces(module)))
    return failure();
  gpu::foldExactConstantDivisions(*kernel);
  return gpu::verifyGPUProgram(module);
}

LogicalResult finalizeProgram(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel)) return failure();
  if (failed(refineMMALoops(module)))
    return failure();
  if (failed(materializeClosedConfigs(*kernel)))
    return failure();
  gpu::foldScalarIntegerValues(*kernel);
  realizeWideLoops(*kernel);
  preserveNativeIndexValues(*kernel);
  if (failed(collapseArrayViews(module)))
    return failure();
  if (failed(gpu::eliminateCommonValues(module)))
    return failure();
  if (ArrayAttr bounds = arrayIndexTileBounds(*kernel))
    (*kernel)->setAttr(arrayIndexTileBoundsAttr, bounds);
  SmallVector<ValueRange> reductionSources;
  kernel->walk([&](ReduceOp reduce) {
    reductionSources.push_back(reduce.getSources());
  });
  gpu::materializeDeferredReductionBounds(*kernel, reductionSources);
  if (failed(verifyCuTileProgram(module)))
    return failure();
  (*kernel)->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}
} // namespace intent::cutile
