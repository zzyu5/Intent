#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool foldConstantDivision(Operation &operation) {
  auto binary = dyn_cast<BinaryOp>(operation);
  if (!binary || binary.getOperatorKind() != BinaryOperator::TrueDivide ||
      binary.getApproximate() || binary.getFlushToZero())
    return false;
  auto type = dyn_cast<FloatType>(binary.getResult().getType());
  auto lhs = binary.getLhs().getDefiningOp<arith::ConstantOp>();
  auto rhs = binary.getRhs().getDefiningOp<arith::ConstantOp>();
  if (!type || !lhs || !rhs)
    return false;
  auto left = dyn_cast<FloatAttr>(lhs.getValue());
  auto right = dyn_cast<FloatAttr>(rhs.getValue());
  if (!left || !right || left.getType() != type || right.getType() != type)
    return false;
  auto quotient = left.getValue();
  quotient.divide(right.getValue(), llvm::APFloat::rmNearestTiesToEven);
  OpBuilder builder(binary);
  auto constant = builder.create<arith::ConstantOp>(
      binary.getLoc(), type, FloatAttr::get(type, quotient));
  if (Attribute origin = binary->getAttr(originAttr))
    constant->setAttr(originAttr, origin);
  binary.getResult().replaceAllUsesWith(constant);
  binary.erase();
  return true;
}

std::optional<llvm::APFloat>
widenedConstantReciprocal(const llvm::APFloat &divisor) {
  constexpr auto rounding = llvm::APFloat::rmNearestTiesToEven;
  bool losesInfo = false;
  llvm::APFloat denominator(divisor);
  denominator.convert(llvm::APFloat::IEEEdouble(), rounding, &losesInfo);
  llvm::APFloat inverse(1.0);
  inverse.divide(denominator, rounding);

  // The 53-bit reciprocal times the 24-bit divisor is exact in binary128.
  llvm::APFloat error(inverse);
  error.convert(llvm::APFloat::IEEEquad(), rounding, &losesInfo);
  denominator.convert(llvm::APFloat::IEEEquad(), rounding, &losesInfo);
  error.multiply(denominator, rounding);
  llvm::APFloat one(1.0);
  one.convert(llvm::APFloat::IEEEquad(), rounding, &losesInfo);
  error.subtract(one, rounding);
  error.clearSign();
  llvm::APFloat bound(0x1p-54);
  bound.convert(llvm::APFloat::IEEEquad(), rounding, &losesInfo);
  if (error.compare(bound) == llvm::APFloat::cmpGreaterThan)
    return std::nullopt;
  // This bound keeps exact f32 midpoints (including subnormal ones) in their
  // f64 RN interval. Other f32 quotients are farther from any midpoint than
  // the reciprocal error plus f64 multiplication rounding can reach.
  return inverse;
}

void eliminateInBlock(Block &block) {
  llvm::DenseMap<OperationName, SmallVector<Operation *>> available;
  for (Operation &operation : llvm::make_early_inc_range(block)) {
    for (Region &region : operation.getRegions())
      for (Block &nested : region)
        eliminateInBlock(nested);
    if (foldConstantDivision(operation))
      continue;
    // Parameter declarations also have symbolic type/attribute users. They are
    // retained and cleaned up by eraseUnusedPhysicalParameters, not SSA DCE.
    if (isa<ParameterOp, DelinearizeOp>(operation) ||
        operation.getNumRegions() != 0 ||
        operation.getNumResults() == 0 || !isMemoryEffectFree(&operation) ||
        !isPhysicalReplayNode(&operation, PhysicalReplayScope::ValueGraph,
                              /*allowAccesses=*/false))
      continue;
    auto &candidates = available[operation.getName()];
    Operation *equivalent = nullptr;
    for (Operation *candidate : candidates)
      if (OperationEquivalence::isEquivalentTo(
              candidate, &operation, OperationEquivalence::exactValueMatch,
              nullptr, OperationEquivalence::IgnoreLocations)) {
        equivalent = candidate;
        break;
      }
    if (!equivalent) {
      candidates.push_back(&operation);
      continue;
    }
    operation.replaceAllUsesWith(equivalent->getResults());
    operation.erase();
  }
}

} // namespace

void foldExactConstantDivisions(func::FuncOp kernel) {
  auto capabilities =
      kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  bool fastFloat64 =
      capabilities.getSingleToDoublePrecisionPerfRatio() <= 2;
  kernel.walk([&](BinaryOp binary) {
    Type element = uniformElementType(binary.getResult().getType());
    if ((binary.getOperatorKind() == BinaryOperator::FloorDivide ||
         binary.getOperatorKind() == BinaryOperator::Remainder) &&
        isa<IndexType, IntegerType>(element)) {
      auto constant = dyn_cast_or_null<IntegerAttr>(
          UniformValueAnalysis(describeUniformValue).evaluate(binary.getRhs()));
      if (!constant || !constant.getValue().isStrictlyPositive() ||
          !constant.getValue().isPowerOf2())
        return;
      bool remainder = binary.getOperatorKind() == BinaryOperator::Remainder;
      int64_t value = remainder ? constant.getInt() - 1
                                : constant.getValue().logBase2();
      OpBuilder builder(binary);
      Value operand = builder.create<arith::ConstantOp>(
          binary.getLoc(), IntegerAttr::get(element, value));
      if (auto fragment = dyn_cast<FragmentType>(binary.getRhs().getType()))
        operand = builder.create<SplatOp>(binary.getLoc(), fragment, operand);
      // Arithmetic right shift and a low-bit mask implement floor division
      // and remainder by positive powers of two, including negative inputs.
      // Expose these before native layout analysis; a truncation correction
      // otherwise obscures the contiguous groups in reshaped addresses.
      binary->setOperand(1, operand);
      binary.setOperatorKind(remainder ? BinaryOperator::BitwiseAnd
                                       : BinaryOperator::RightShift);
      return;
    }
    if (binary.getOperatorKind() != BinaryOperator::TrueDivide ||
        binary.getApproximate() || binary.getFlushToZero() ||
        (!element.isF32() && !element.isF64()))
      return;
    auto constant = dyn_cast_or_null<FloatAttr>(
        UniformValueAnalysis(describeUniformValue).evaluate(binary.getRhs()));
    if (!constant || !constant.getValue().isNormal())
      return;
    llvm::APFloat inverse(constant.getValue().getSemantics());
    bool widened = false;
    if (!constant.getValue().getExactInverse(&inverse) || !inverse.isNormal()) {
      if (!element.isF32() || !fastFloat64)
        return;
      auto reciprocal = widenedConstantReciprocal(constant.getValue());
      if (!reciprocal)
        return;
      inverse = std::move(*reciprocal);
      widened = true;
    }
    OpBuilder builder(binary);
    Type computationElement = widened ? builder.getF64Type() : element;
    auto computationType = [&](Type type) -> Type {
      if (auto fragment = dyn_cast<FragmentType>(type))
        return FragmentType::get(
            kernel.getContext(), computationElement, fragment.getShape(),
            fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
      return computationElement;
    };
    Value reciprocal = builder.create<arith::ConstantOp>(
        binary.getLoc(), FloatAttr::get(computationElement, inverse));
    if (auto fragment = dyn_cast<FragmentType>(binary.getRhs().getType()))
      reciprocal = builder.create<SplatOp>(
          binary.getLoc(), cast<FragmentType>(computationType(fragment)),
          reciprocal);
    Value lhs = binary.getLhs();
    if (widened)
      lhs = builder.create<CastOp>(binary.getLoc(), computationType(lhs.getType()),
                                   lhs);
    auto product = builder.create<BinaryOp>(
        binary.getLoc(), computationType(binary.getType()), lhs, reciprocal,
        BinaryOperator::Multiply);
    Value result = product.getResult();
    if (widened)
      result = builder.create<CastOp>(binary.getLoc(), binary.getType(), result);
    result.getDefiningOp()->setDiscardableAttrs(
        llvm::to_vector(binary->getDiscardableAttrs()));
    binary.getResult().replaceAllUsesWith(result);
    binary.erase();
  });
}

LogicalResult eliminateCommonValues(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  for (Block &block : kernel->getBody())
    eliminateInBlock(block);
  return success();
}

} // namespace intent::gpu
