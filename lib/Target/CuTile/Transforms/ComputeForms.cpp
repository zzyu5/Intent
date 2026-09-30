#include "NativeAccess.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;
namespace intent::cutile {
gpu::FragmentType transposeRankTwo(gpu::FragmentType source) {
  SmallVector<Attribute> shape = {source.getShape()[1], source.getShape()[0]};
  SmallVector<Attribute> mappings;
  mappings.reserve(2);
  for (unsigned resultAxis = 0; resultAxis < 2; ++resultAxis) {
    const unsigned sourceAxis = 1 - resultAxis;
    auto mapping = cast<gpu::AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    mappings.push_back(gpu::AxisMapAttr::get(
        source.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), resultAxis, mapping.getDerived()));
  }
  return gpu::FragmentType::get(
      source.getContext(), source.getElementType(),
      ArrayAttr::get(source.getContext(), shape),
      ArrayAttr::get(source.getContext(), mappings), source.getValidity(),
      source.getOwner());
}

std::optional<BinaryOperator> nativeCombineKind(Region &region) {
  std::optional<BinaryOperator> kind = gpu::queryBinaryCombineKind(region);
  if (!kind)
    return std::nullopt;
  if (*kind == BinaryOperator::Add || *kind == BinaryOperator::MaximumNum ||
      *kind == BinaryOperator::MinimumNum)
    return kind;
  Type element = region.front().getArgument(0).getType();
  if (auto fragment = dyn_cast<gpu::FragmentType>(element))
    element = fragment.getElementType();
  if (element.isIntOrIndex()) {
    if (*kind == BinaryOperator::Maximum)
      return BinaryOperator::MaximumNum;
    if (*kind == BinaryOperator::Minimum)
      return BinaryOperator::MinimumNum;
  }
  if (element.isInteger(1)) {
    if (*kind == BinaryOperator::LogicalOr)
      return BinaryOperator::LogicalOr;
    if (*kind == BinaryOperator::LogicalAnd)
      return BinaryOperator::LogicalAnd;
  }
  return std::nullopt;
}

bool isNativeReductionIdentity(BinaryOperator kind, Value identity) {
  Attribute constant = getCompileTimeScalar(identity);
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant)) {
    switch (kind) {
    case BinaryOperator::Add:
    case BinaryOperator::LogicalOr:
      return integer.getValue().isZero();
    case BinaryOperator::LogicalAnd:
      return integer.getValue().isAllOnes();
    default:
      return false;
    }
  }
  auto floating = dyn_cast<FloatAttr>(constant);
  if (!floating)
    return false;
  const llvm::APFloat &value = floating.getValue();
  if (kind == BinaryOperator::Add)
    return value.isZero();
  if (!value.isInfinity())
    return false;
  if (kind == BinaryOperator::MaximumNum || kind == BinaryOperator::Maximum)
    return value.isNegative();
  if (kind == BinaryOperator::MinimumNum || kind == BinaryOperator::Minimum)
    return !value.isNegative();
  return false;
}

LogicalResult verifyScan(gpu::ScanOp scan) {
  if (scan.getSourceCount() == 0 ||
      scan.getSourceCount() != scan.getIdentityCount() ||
      scan.getSourceCount() != scan.getNumResults() ||
      scan.getCaptureCount() != 0)
    return scan.emitOpError(
        "cuTile scan lowering requires matching source/identity/result schemas and no captures");
  auto source = dyn_cast<gpu::FragmentType>(scan.getInputs().front().getType());
  if (!source)
    return scan.emitOpError("cuTile scan source must be a tile");
  for (auto [input, identity] : llvm::zip(
           scan.getInputs().take_front(scan.getSourceCount()),
           scan.getInputs().slice(scan.getSourceCount(), scan.getIdentityCount()))) {
    auto fragment = dyn_cast<gpu::FragmentType>(input.getType());
    if (!fragment || fragment.getShape() != source.getShape())
      return scan.emitOpError(
          "cuTile scan lowering requires source components with the same physical shape");
    if (!getCompileTimeScalar(identity) ||
        gpu::uniformElementType(identity.getType()) != fragment.getElementType())
      return scan.emitOpError(
          "cuTile scan identity must be a compile-time scalar of the source dtype");
  }
  return success();
}

LogicalResult formComputePrimitives(func::FuncOp kernel,
    const NativeProgramInputs &inputs, NativeFormRewriter &rewriter) {
  for (gpu::HistogramOp histogram : inputs.histograms) {
    auto source = cast<gpu::FragmentType>(histogram.getValues().getType());
    auto result = cast<gpu::FragmentType>(histogram.getResult().getType());
    if (source.getShape().size() != 1)
      return histogram.emitOpError(
          "cuTile histogram requires a realized one-axis input fragment");
    OpBuilder builder(histogram);
    Location location = histogram.getLoc();
    auto sourceInteger = dyn_cast<IntegerType>(source.getElementType());
    Type comparisonElement = sourceInteger && sourceInteger.isUnsigned()
                                 ? IntegerType::get(kernel.getContext(), 64,
                                                    IntegerType::Unsigned)
                                 : builder.getI64Type();
    auto binMap = cast<gpu::AxisMapAttr>(result.getAxisMaps()[0]);
    auto inputMap = cast<gpu::AxisMapAttr>(source.getAxisMaps()[0]);
    auto inputAxis = gpu::AxisMapAttr::get(
        kernel.getContext(), inputMap.getSourceId(), inputMap.getSourceAxis(),
        inputMap.getDimensionId(), 1, inputMap.getDerived());
    auto shape = builder.getArrayAttr(
        {result.getShape()[0], source.getShape()[0]});
    auto maps = builder.getArrayAttr({binMap, inputAxis});
    auto relation = [&](Type element) {
      return gpu::FragmentType::get(kernel.getContext(), element, shape, maps,
                                    result.getValidity(), result.getOwner());
    };
    auto binType = gpu::FragmentType::get(
        kernel.getContext(), builder.getIndexType(), result.getShape(),
        result.getAxisMaps(), result.getValidity(), result.getOwner());
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    Value bins = builder.create<gpu::MakeRangeOp>(
        location, binType, zero, histogram.getBins(), one, zero,
        histogram.getBins(), binMap.getSourceId(), binMap.getSourceAxis(),
        binMap.getDerived());
    bins = builder.create<gpu::CastOp>(
        location, withElementType(binType, comparisonElement), bins);
    Value input = builder.create<gpu::CastOp>(
        location, withElementType(source, comparisonElement),
        histogram.getValues());
    Value values = builder.create<gpu::BroadcastOp>(
        location, relation(comparisonElement), input);
    Value selectedBins = builder.create<gpu::BroadcastOp>(
        location, relation(comparisonElement), bins);
    Value members = builder.create<gpu::CompareOp>(
        location, relation(builder.getI1Type()), values, selectedBins,
        ComparePredicate::Eq);
    if (histogram.getValid()) {
      Value valid = builder.create<gpu::BroadcastOp>(
          location, relation(builder.getI1Type()), histogram.getValid());
      members = builder.create<gpu::BinaryOp>(
          location, relation(builder.getI1Type()), members, valid,
          BinaryOperator::LogicalAnd);
    }
    Value counts = builder.create<gpu::CastOp>(
        location, relation(result.getElementType()), members);
    auto replacement = builder.create<ReduceOp>(
        location, TypeRange{result}, ValueRange{counts}, 1, 1, false,
        BinaryOperatorAttr::get(kernel.getContext(), BinaryOperator::Add));
    if (Attribute origin = histogram->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    rewriter.replace(histogram, ValueRange{replacement.getResult(0)});
  }

  for (gpu::ReduceOp reduce : inputs.reductions) {
    std::optional<BinaryOperator> kind = nativeCombineKind(reduce.getCombine());
    bool native = reduce.getSourceCount() == 1 &&
                  reduce.getIdentityCount() == 1 &&
                  reduce.getCaptureCount() == 0 &&
                  reduce.getAxes().size() == 1 &&
                  reduce.getNumResults() == 1 && kind.has_value();
    if (native) {
      // Shared realization has already filled inactive lanes with the declared
      // identity. As with tl.reduce, the native tree consumes those values;
      // it does not require that identity to be a compile-time literal.
      auto source =
          dyn_cast<gpu::FragmentType>(reduce.getInputs().front().getType());
      if (!source)
        return reduce.emitOpError("cuTile native reduce source must be a tile");
      OpBuilder builder(reduce);
      auto replacement = builder.create<ReduceOp>(
          reduce.getLoc(), reduce.getResultTypes(),
          ValueRange{reduce.getInputs().front()}, 1, reduce.getAxes().front(),
          false, BinaryOperatorAttr::get(kernel.getContext(), *kind));
      if (Attribute origin = reduce->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      rewriter.replace(reduce, replacement.getResults());
      continue;
    }

    std::optional<BinaryOperator> canonicalKind =
        gpu::queryBinaryCombineKind(reduce.getCombine());
    bool propagatingMaximum =
        reduce.getSourceCount() == 1 && reduce.getIdentityCount() == 1 &&
        reduce.getCaptureCount() == 0 && reduce.getAxes().size() == 1 &&
        reduce.getNumResults() == 1 && canonicalKind == BinaryOperator::Maximum &&
        isNativeReductionIdentity(
            BinaryOperator::Maximum,
            reduce.getInputs()[reduce.getSourceCount()]);
    if (propagatingMaximum) {
      auto sourceType =
          dyn_cast<gpu::FragmentType>(reduce.getInputs().front().getType());
      if (!sourceType || !isa<FloatType>(sourceType.getElementType()))
        return reduce.emitOpError(
            "cuTile propagating maximum reduction requires a floating tile");
      OpBuilder builder(reduce);
      Location location = reduce.getLoc();
      Type resultType = reduce.getResultTypes().front();
      Type sourcePredicateType =
          withElementType(sourceType, builder.getI1Type());
      Type resultPredicateType =
          withElementType(resultType, builder.getI1Type());
      auto isNan = builder.create<gpu::CompareOp>(
          location, sourcePredicateType, reduce.getInputs().front(),
          reduce.getInputs().front(), ComparePredicate::Ne);
      auto anyNan = builder.create<ReduceOp>(
          location, TypeRange{resultPredicateType}, ValueRange{isNan.getResult()},
          1, reduce.getAxes().front(), false,
          BinaryOperatorAttr::get(kernel.getContext(), BinaryOperator::LogicalOr));
      auto numericMaximum = builder.create<ReduceOp>(
          location, TypeRange{resultType}, ValueRange{reduce.getInputs().front()},
          1, reduce.getAxes().front(), false,
          BinaryOperatorAttr::get(kernel.getContext(), BinaryOperator::MaximumNum));
      auto floatType = cast<FloatType>(sourceType.getElementType());
      auto nan = builder.create<arith::ConstantOp>(
          location, floatType,
          builder.getFloatAttr(
              floatType,
              llvm::APFloat::getNaN(floatType.getFloatSemantics())));
      Value nanValue = nan.getResult();
      if (isa<gpu::FragmentType>(resultType))
        nanValue = builder.create<gpu::SplatOp>(location, resultType, nanValue);
      auto replacement = builder.create<gpu::SelectOp>(
          location, resultType, anyNan.getResult(0), nanValue,
          numericMaximum.getResult(0));
      if (Attribute origin = reduce->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      rewriter.replace(reduce, ValueRange{replacement.getResult()});
      continue;
    }

    if (reduce.getSourceCount() == 0 ||
        reduce.getSourceCount() != reduce.getIdentityCount() ||
        reduce.getSourceCount() != reduce.getNumResults() ||
        reduce.getCaptureCount() != 0 || reduce.getAxes().size() != 1)
      return reduce.emitOpError(
          "cuTile custom reduce requires matching non-empty source/identity/result schemas, one axis, and no captures");
    for (Value source : reduce.getInputs().take_front(reduce.getSourceCount()))
      if (!isa<gpu::FragmentType>(source.getType()))
        return reduce.emitOpError("cuTile custom reduce source must be a tile");
    for (Value identity : reduce.getInputs().slice(
             reduce.getSourceCount(), reduce.getIdentityCount()))
      if (!getCompileTimeScalar(identity)) {
        InFlightDiagnostic diagnostic = reduce.emitOpError(
            "cuTile custom reduce identity must be a compile-time scalar");
        diagnostic << "; identity type=" << identity.getType();
        if (Operation *producer = identity.getDefiningOp())
          diagnostic << ", producer=" << producer->getName();
        return failure();
      }
    OpBuilder builder(reduce);
    auto replacement = builder.create<ReduceOp>(
        reduce.getLoc(), reduce.getResultTypes(), reduce.getInputs(),
        reduce.getSourceCount(), reduce.getAxes().front(), false,
        BinaryOperatorAttr());
    if (failed(gpu::scalarizeElementwiseCallback(reduce.getCombine(),
                                                 replacement.getCombine())))
      return failure();
    if (Attribute origin = reduce->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    rewriter.replace(reduce, replacement.getResults());
  }

  for (gpu::ScanOp scan : inputs.scans) {
    if (failed(verifyScan(scan)))
      return failure();
    std::optional<BinaryOperator> kind = nativeCombineKind(scan.getCombine());
    bool native = scan.getSourceCount() == 1 && kind &&
                  *kind == BinaryOperator::Add &&
                  isNativeReductionIdentity(*kind, scan.getInputs()[1]);
    Type element = gpu::uniformElementType(scan.getInputs().front().getType());
    bool exclusiveSum = !scan.getInclusive() && native &&
                        isa<IntegerType, IndexType>(element) &&
                        !element.isInteger(1);
    if (!scan.getInclusive() && !exclusiveSum)
      return scan.emitOpError(
          "cuTile exclusive scan requires an additive integer prefix");
    OpBuilder builder(scan);
    auto replacement = builder.create<ScanOp>(
        scan.getLoc(), scan.getResultTypes(),
        native ? scan.getInputs().take_front(1) : scan.getInputs(),
        scan.getSourceCount(), scan.getAxis(), scan.getReverse(),
        native ? BinaryOperatorAttr::get(kernel.getContext(), *kind)
               : BinaryOperatorAttr());
    if (!native &&
        failed(gpu::scalarizeElementwiseCallback(scan.getCombine(),
                                                replacement.getCombine())))
      return failure();
    if (Attribute origin = scan->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    SmallVector<Value> results(replacement.getResults());
    if (exclusiveSum)
      // Modular integer addition has an exact inverse. This also preserves
      // reverse prefixes; floating subtraction is not an equivalent rewrite.
      results[0] = builder.create<gpu::BinaryOp>(
          scan.getLoc(), results[0].getType(), results[0], scan.getInputs()[0],
          BinaryOperator::Subtract);
    rewriter.replace(scan, results);
  }

  for (gpu::ContractOp contract : inputs.contracts) {
    std::string reason;
    auto axes = gpu::queryContractionAxes(contract, &reason);
    if (!axes)
      return contract.emitOpError("invalid cuTile contraction axis schema: ") << reason;
    if (!axes->hasCanonicalMatrixAxes() || axes->lhsResultAxes.size() > 3)
      return contract.emitOpError(
          "cuTile native MMA requires [M,K] x [K,N] or [B,M,K] x [B,K,N] "
          "physical axes");
    OpBuilder builder(contract);
    Attribute origin = contract->getAttr(gpu::originAttr);
    auto inheritOrigin = [&](Operation *operation) {
      if (origin)
        operation->setAttr(gpu::originAttr, origin);
    };
    auto replacement = builder.create<MMAOp>(
        contract.getLoc(), contract.getResult().getType(), contract.getLhs(),
        contract.getRhs(), contract.getAccumulator(), IntegerAttr());
    inheritOrigin(replacement);
    rewriter.replace(contract, ValueRange{replacement.getResult()});
  }

  for (gpu::ScaledContractOp contract : inputs.scaledContracts) {
    if (contract.getLhsReductionAxes() != ArrayRef<int64_t>{1, 2} ||
        contract.getRhsReductionAxes() != ArrayRef<int64_t>{0, 1} ||
        !contract.getLhsBatchAxes().empty() ||
        !contract.getRhsBatchAxes().empty() ||
        contract.getLhsFormat() != ScaledFormat::E4M3 ||
        contract.getRhsFormat() != ScaledFormat::E4M3 ||
        contract.getLhsGroupSize() != 32 ||
        contract.getRhsGroupSize() != 32)
      return contract.emitOpError(
          "cuTile scaled MMA requires E4M3/E8M0 group-32 adjacent reduction axes");
    OpBuilder builder(contract);
    auto rhsScale = builder.create<gpu::TransposeOp>(
        contract.getLoc(),
        transposeRankTwo(
            cast<gpu::FragmentType>(contract.getRhsScale().getType())),
        contract.getRhsScale(), ArrayRef<int64_t>{1, 0});
    if (Attribute origin = contract->getAttr(gpu::originAttr))
      rhsScale->setAttr(gpu::originAttr, origin);
    auto replacement = builder.create<ScaledMMAOp>(
        contract.getLoc(), contract.getResult().getType(), contract.getLhs(),
        contract.getLhsScale(), contract.getRhs(), rhsScale,
        contract.getAccumulator(), contract.getLhsFormat(),
        contract.getRhsFormat(), contract.getLhsGroupSize(),
        contract.getRhsGroupSize());
    if (Attribute origin = contract->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    rewriter.replace(contract, ValueRange{replacement.getResult()});
  }

  return success();
}
} // namespace intent::cutile
