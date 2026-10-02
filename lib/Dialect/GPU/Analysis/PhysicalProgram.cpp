#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/IndexPredicates.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <limits>

using namespace mlir;

namespace intent::gpu {

std::optional<ContractionAxes>
queryContractionAxes(Operation *operation, std::string *failureReason) {
  auto read = [&](auto contract, Value lhs, Value rhs)
      -> std::optional<ContractionAxes> {
    auto left = dyn_cast<FragmentType>(lhs.getType());
    auto right = dyn_cast<FragmentType>(rhs.getType());
    auto result = dyn_cast<FragmentType>(contract.getResult().getType());
    if (!left || !right || !result) {
      if (failureReason) *failureReason = "requires fragment operands and result";
      return std::nullopt;
    }
    auto axes = ContractionAxes::get(
        left.getShape().size(), right.getShape().size(),
        contract.getLhsReductionAxes(), contract.getRhsReductionAxes(),
        contract.getLhsBatchAxes(), contract.getRhsBatchAxes(), failureReason);
    if (axes && axes->results.size() != result.getShape().size()) {
      if (failureReason) *failureReason = "result rank disagrees with batch/free axes";
      return std::nullopt;
    }
    return axes;
  };
  if (auto contract = dyn_cast_or_null<ContractOp>(operation))
    return read(contract, contract.getLhs(), contract.getRhs());
  if (auto contract = dyn_cast_or_null<ScaledContractOp>(operation))
    return read(contract, contract.getLhs(), contract.getRhs());
  if (auto contract = dyn_cast_or_null<SparseContractOp>(operation))
    return read(contract, contract.getCompressed(), contract.getRhs());
  if (failureReason) *failureReason = "expected a contraction operation";
  return std::nullopt;
}

namespace {

bool isCoordinateReplayNode(Operation *operation) {
  return isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
             BroadcastOp, SplatOp, ReshapeOp, TransposeOp, JoinOp, DimOp,
             MakeRecordOp, ExtractOp, RandomBitsOp>(operation);
}

bool isValueReplayNode(Operation *operation) {
  return isCoordinateReplayNode(operation) ||
         isa<ContractOp, ScaledContractOp, SparseContractOp, ReduceOp, ScanOp>(
             operation);
}

bool isAccessNode(Operation *operation) {
  return isa<LoadOp, GatherOp>(operation);
}

void collectStructuredPrograms(Value value,
                               SmallPtrSetImpl<Operation *> &visited,
                               SmallVectorImpl<Operation *> &programs) {
  Operation *operation = value.getDefiningOp();
  if (!operation || !visited.insert(operation).second)
    return;
  if (isa<RegionFoldOp, RegionScanOp>(operation)) {
    if (!llvm::is_contained(programs, operation))
      programs.push_back(operation);
    return;
  }
  if (isAccessNode(operation) || operation->getNumRegions() != 0)
    return;
  for (Value operand : operation->getOperands())
    collectStructuredPrograms(operand, visited, programs);
}

std::optional<int64_t> integerConstant(Value value, unsigned depth = 0) {
  if (!value || depth >= 32)
    return std::nullopt;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    return integerConstant(broadcast.getValue(), depth + 1);
  if (auto splat = value.getDefiningOp<SplatOp>())
    return integerConstant(splat.getValue(), depth + 1);
  if (auto cast = value.getDefiningOp<CastOp>()) {
    if (cast.getValue().getType() == cast.getResult().getType())
      return integerConstant(cast.getValue(), depth + 1);
    if (!cast.getResult().getType().isIndex() ||
        !isa<IntegerType, IndexType>(cast.getValue().getType()))
      return std::nullopt;
    auto constant = dyn_cast_or_null<IntegerAttr>(
        UniformValueAnalysis(describeUniformValue).evaluate(value));
    return constant && constant.getType().isIndex()
               ? std::optional<int64_t>(constant.getInt())
               : std::nullopt;
  }
  if (auto bound = value.getDefiningOp<RangeBoundOp>()) {
    auto range = bound.getRange().getDefiningOp<RangeOp>();
    if (!range)
      return std::nullopt;
    if (bound.getBound() == 0)
      return integerConstant(range.getStart(), depth + 1);
    if (bound.getBound() == 1)
      return integerConstant(range.getStop(), depth + 1);
    return integerConstant(range.getStep(), depth + 1);
  }
  auto binary = value.getDefiningOp<BinaryOp>();
  if (!binary)
    return std::nullopt;
  std::optional<int64_t> lhs = integerConstant(binary.getLhs(), depth + 1);
  std::optional<int64_t> rhs = integerConstant(binary.getRhs(), depth + 1);
  if (!lhs || !rhs)
    return std::nullopt;
  __int128 evaluated;
  switch (binary.getOperatorKind()) {
  case BinaryOperator::Add:
    evaluated = static_cast<__int128>(*lhs) + *rhs;
    break;
  case BinaryOperator::Subtract:
    evaluated = static_cast<__int128>(*lhs) - *rhs;
    break;
  case BinaryOperator::Multiply:
    evaluated = static_cast<__int128>(*lhs) * *rhs;
    break;
  case BinaryOperator::FloorDivide:
    if (*rhs != 1)
      return std::nullopt;
    evaluated = *lhs;
    break;
  default:
    return std::nullopt;
  }
  if (evaluated < std::numeric_limits<int64_t>::min() ||
      evaluated > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(evaluated);
}

Value stripBroadcast(Value value) {
  while (auto broadcast = value.getDefiningOp<BroadcastOp>())
    value = broadcast.getValue();
  while (auto splat = value.getDefiningOp<SplatOp>())
    value = splat.getValue();
  return value;
}

Value stripCombineProjection(Value value) {
  while (auto broadcast = value.getDefiningOp<BroadcastOp>())
    value = broadcast.getValue();
  return value;
}

Value stripScalarIdentity(Value value) {
  value = stripBroadcast(value);
  while (true) {
    if (auto cast = value.getDefiningOp<CastOp>()) {
      if (cast.getValue().getType() == cast.getResult().getType()) {
        value = stripBroadcast(cast.getValue());
        continue;
      }
      return value;
    }
    auto binary = value.getDefiningOp<BinaryOp>();
    if (!binary)
      return value;
    auto isInteger = [](Value operand, int64_t expected) {
      std::optional<int64_t> value = integerConstant(operand);
      return value && *value == expected;
    };
    if (binary.getOperatorKind() == BinaryOperator::Add) {
      if (isInteger(binary.getLhs(), 0)) {
        value = stripBroadcast(binary.getRhs());
        continue;
      }
      if (isInteger(binary.getRhs(), 0)) {
        value = stripBroadcast(binary.getLhs());
        continue;
      }
    }
    if (binary.getOperatorKind() == BinaryOperator::Multiply) {
      if (isInteger(binary.getLhs(), 0))
        return binary.getLhs();
      if (isInteger(binary.getRhs(), 0))
        return binary.getRhs();
      if (isInteger(binary.getLhs(), 1)) {
        value = stripBroadcast(binary.getRhs());
        continue;
      }
      if (isInteger(binary.getRhs(), 1)) {
        value = stripBroadcast(binary.getLhs());
        continue;
      }
    }
    if (binary.getOperatorKind() == BinaryOperator::Subtract &&
        isInteger(binary.getRhs(), 0)) {
      value = stripBroadcast(binary.getLhs());
      continue;
    }
    if (binary.getOperatorKind() == BinaryOperator::FloorDivide &&
        isInteger(binary.getRhs(), 1)) {
      value = stripBroadcast(binary.getLhs());
      continue;
    }
    return value;
  }
}

void appendUnique(SmallVectorImpl<MakeRangeOp> &destination, MakeRangeOp range) {
  if (!llvm::is_contained(destination, range))
    destination.push_back(range);
}

void appendUnique(SmallVectorImpl<Operation *> &destination,
                  Operation *operation) {
  if (operation && !llvm::is_contained(destination, operation))
    destination.push_back(operation);
}

PhysicalExprAttr resourceExtentExpression(Value resource, unsigned axis);

bool sameScalarExpression(Value lhs, Value rhs, unsigned depth = 0) {
  if (lhs == rhs)
    return true;
  if (depth >= 32 || lhs.getType() != rhs.getType())
    return false;
  Value left = stripScalarIdentity(lhs);
  Value right = stripScalarIdentity(rhs);
  if (left != lhs || right != rhs)
    return sameScalarExpression(left, right, depth + 1);
  auto leftConstant = lhs.getDefiningOp<arith::ConstantOp>();
  auto rightConstant = rhs.getDefiningOp<arith::ConstantOp>();
  if (leftConstant || rightConstant)
    return leftConstant && rightConstant &&
           leftConstant.getValue() == rightConstant.getValue();
  auto leftParameter = queryParameter(lhs), rightParameter = queryParameter(rhs);
  if (leftParameter || rightParameter)
    return leftParameter && rightParameter &&
           leftParameter.getReference() == rightParameter.getReference() &&
           lhs.getDefiningOp()->getParentOfType<func::FuncOp>() ==
               rhs.getDefiningOp()->getParentOfType<func::FuncOp>();
  auto leftExpression = lhs.getDefiningOp<PhysicalExprOp>();
  auto rightExpression = rhs.getDefiningOp<PhysicalExprOp>();
  if (leftExpression || rightExpression)
    return leftExpression && rightExpression &&
           leftExpression->getParentOfType<func::FuncOp>() ==
               rightExpression->getParentOfType<func::FuncOp>() &&
           leftExpression.getExpression() == rightExpression.getExpression();
  auto leftBinary = lhs.getDefiningOp<BinaryOp>();
  auto rightBinary = rhs.getDefiningOp<BinaryOp>();
  if (leftBinary || rightBinary)
    return leftBinary && rightBinary &&
           leftBinary.getOperatorKind() == rightBinary.getOperatorKind() &&
           leftBinary.getApproximate() == rightBinary.getApproximate() &&
           leftBinary.getFlushToZero() == rightBinary.getFlushToZero() &&
           sameScalarExpression(leftBinary.getLhs(), rightBinary.getLhs(),
                                depth + 1) &&
           sameScalarExpression(leftBinary.getRhs(), rightBinary.getRhs(),
                                depth + 1);
  auto leftCompare = lhs.getDefiningOp<CompareOp>();
  auto rightCompare = rhs.getDefiningOp<CompareOp>();
  if (leftCompare || rightCompare)
    return leftCompare && rightCompare &&
           leftCompare.getPredicate() == rightCompare.getPredicate() &&
           sameScalarExpression(leftCompare.getLhs(), rightCompare.getLhs(),
                                depth + 1) &&
           sameScalarExpression(leftCompare.getRhs(), rightCompare.getRhs(),
                                depth + 1);
  auto leftDim = lhs.getDefiningOp<DimOp>();
  auto rightDim = rhs.getDefiningOp<DimOp>();
  if (leftDim || rightDim) {
    if (!leftDim || !rightDim ||
        leftDim->getParentOfType<func::FuncOp>() !=
            rightDim->getParentOfType<func::FuncOp>())
      return false;
    PhysicalExprAttr leftExtent =
        resourceExtentExpression(leftDim.getView(), leftDim.getAxis());
    PhysicalExprAttr rightExtent =
        resourceExtentExpression(rightDim.getView(), rightDim.getAxis());
    return leftExtent && rightExtent && leftExtent == rightExtent;
  }
  auto leftCast = lhs.getDefiningOp<CastOp>();
  auto rightCast = rhs.getDefiningOp<CastOp>();
  return leftCast && rightCast &&
         sameScalarExpression(leftCast.getValue(), rightCast.getValue(),
                              depth + 1);
}

bool isUnitStepValue(Value value) {
  if (std::optional<int64_t> constant = integerConstant(value))
    return *constant == 1;
  if (auto cast = value.getDefiningOp<CastOp>())
    return isUnitStepValue(cast.getValue());
  if (auto bound = value.getDefiningOp<RangeBoundOp>()) {
    auto range = bound.getRange().getDefiningOp<RangeOp>();
    return range && bound.getBound() == 2 && isUnitStepValue(range.getStep());
  }
  return false;
}

PhysicalExprAttr resourceExtentExpression(Value resource, unsigned axis) {
  if (auto view = dyn_cast<ViewType>(resource.getType())) {
    if (axis < view.getRank())
      return cast<PhysicalExprAttr>(view.getLayout().getExtents()[axis]);
    return {};
  }
  if (auto buffer = dyn_cast<BufferType>(resource.getType())) {
    if (axis < buffer.getShape().size())
      return cast<PhysicalExprAttr>(buffer.getShape()[axis]);
    return {};
  }
  if (auto fragment = dyn_cast<FragmentType>(resource.getType())) {
    if (axis < fragment.getShape().size())
      return cast<PhysicalExprAttr>(fragment.getShape()[axis]);
  }
  return {};
}

bool matchesResourceExtent(Value value, Value resource, unsigned axis) {
  PhysicalExprAttr extent = resourceExtentExpression(resource, axis);
  if (!extent)
    return false;
  Value stripped = stripScalarIdentity(value);
  if (stripped != value)
    return matchesResourceExtent(stripped, resource, axis);
  if (std::optional<int64_t> constant = integerConstant(value))
    return extent.getKind() ==
               PhysicalExprKind::Constant &&
           *constant == extent.getValue();
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    return queryArgumentExpression(argument) == extent;
  }
  if (auto dim = value.getDefiningOp<DimOp>()) {
    auto view = dyn_cast<ViewType>(resource.getType());
    return view && dim.getView() == resource && dim.getAxis() == axis;
  }
  if (auto expression = value.getDefiningOp<PhysicalExprOp>())
    return expression.getExpression() == extent;
  if (auto parameter = queryParameter(value))
    return extent.getKind() ==
               PhysicalExprKind::Parameter &&
           parameter.getName() == extent.getParameterReference().getName();
  if (auto bound = value.getDefiningOp<RangeBoundOp>()) {
    auto range = bound.getRange().getDefiningOp<RangeOp>();
    if (!range)
      return false;
    if (bound.getBound() == 0)
      return matchesResourceExtent(range.getStart(), resource, axis);
    if (bound.getBound() == 1)
      return matchesResourceExtent(range.getStop(), resource, axis);
    return matchesResourceExtent(range.getStep(), resource, axis);
  }
  return false;
}

bool capacityCoversResourceExtent(Value value, Value resource, unsigned axis) {
  value = stripScalarIdentity(value);
  ParameterAttr parameter = queryParameter(value);
  if (auto expression = value.getDefiningOp<PhysicalExprOp>();
      expression && expression.getExpression().getKind() ==
                        PhysicalExprKind::Parameter) {
    auto resolved = queryParameterBySymbol(
        expression->getParentOfType<func::FuncOp>(),
        expression.getExpression().getParameterReference().getName());
    if (succeeded(resolved))
      parameter = *resolved;
  }
  if (!parameter || parameter.getCategory() !=
                        ParameterCategory::Coverage)
    return false;
  auto covered = parameter.getBinding().getCoverageBound();
  return covered && covered == resourceExtentExpression(resource, axis);
}

bool expressionAtMost(PhysicalExprAttr lhs, PhysicalExprAttr rhs,
                      unsigned depth = 0) {
  if (!lhs || !rhs || depth >= 32)
    return false;
  if (lhs == rhs)
    return true;
  auto left = constantPhysicalExpression(lhs);
  auto right = constantPhysicalExpression(rhs);
  if (left && right)
    return *left <= *right;
  if (lhs.getOperands().size() == 2) {
    auto first = cast<PhysicalExprAttr>(lhs.getOperands()[0]);
    auto second = cast<PhysicalExprAttr>(lhs.getOperands()[1]);
    if (lhs.getKind() == PhysicalExprKind::Minimum)
      return expressionAtMost(first, rhs, depth + 1) ||
             expressionAtMost(second, rhs, depth + 1);
    if (lhs.getKind() == PhysicalExprKind::Maximum)
      return expressionAtMost(first, rhs, depth + 1) &&
             expressionAtMost(second, rhs, depth + 1);
  }
  if (rhs.getOperands().size() == 2) {
    auto first = cast<PhysicalExprAttr>(rhs.getOperands()[0]);
    auto second = cast<PhysicalExprAttr>(rhs.getOperands()[1]);
    if (rhs.getKind() == PhysicalExprKind::Maximum)
      return expressionAtMost(lhs, first, depth + 1) ||
             expressionAtMost(lhs, second, depth + 1);
    if (rhs.getKind() == PhysicalExprKind::Minimum)
      return expressionAtMost(lhs, first, depth + 1) &&
             expressionAtMost(lhs, second, depth + 1);
  }
  return false;
}

bool upperBoundWithinResource(Value value, Value resource, unsigned axis) {
  if (matchesResourceExtent(value, resource, axis))
    return true;
  std::optional<int64_t> bound = integerConstant(value);
  PhysicalExprAttr provenBound =
      queryNonNegativeIndexUpperBound(stripScalarIdentity(value));
  if (!bound && provenBound && provenBound.getKind() ==
                                  PhysicalExprKind::Constant)
    bound = provenBound.getValue();
  PhysicalExprAttr extent = resourceExtentExpression(resource, axis);
  if (!extent)
    return false;
  if (provenBound && expressionAtMost(provenBound, extent))
    return true;
  auto kind = extent.getKind();
  if (kind == PhysicalExprKind::Constant)
    return bound && *bound >= 0 && *bound <= extent.getValue();
  if (kind != PhysicalExprKind::Parameter)
    return false;
  func::FuncOp kernel = resource.getParentRegion()
                            ? resource.getParentRegion()->getParentOfType<func::FuncOp>()
                            : func::FuncOp();
  FailureOr<ParameterAttr> parameter =
      kernel ? queryParameterBySymbol(kernel, extent.getParameterReference().getName())
             : FailureOr<ParameterAttr>(failure());
  if (failed(parameter))
    return false;
  if (parameter->getCategory() ==
      ParameterCategory::Coverage)
    if (auto covered = parameter->getBinding().getCoverageBound();
        covered && (queryLaunchExpression(value) == covered ||
                    provenBound == covered))
      return true;
  return bound && *bound >= 0 &&
         llvm::all_of(parameter->getCandidates().asArrayRef(),
                      [&](int64_t candidate) { return *bound <= candidate; });
}

Value stripIntegerIndexCasts(Value value) {
  value = stripBroadcast(value);
  while (true) {
    if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
      auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
      auto result = dyn_cast<FragmentType>(reshape.getResult().getType());
      auto nonUnitShape = [](ArrayAttr shape) {
        SmallVector<Attribute> extents;
        for (Attribute attribute : shape) {
          auto extent = cast<PhysicalExprAttr>(attribute);
          if (constantPhysicalExpression(extent) != 1)
            extents.push_back(extent);
        }
        return extents;
      };
      // Inserting or removing size-one axes preserves every coordinate value
      // and its linear order, including the coordinate guarded by a mask.
      if (source && result &&
          nonUnitShape(source.getShape()) == nonUnitShape(result.getShape())) {
        value = stripBroadcast(reshape.getValue());
        continue;
      }
    }
    auto cast = value.getDefiningOp<CastOp>();
    if (!cast)
      break;
    auto element = [](Type type) {
      if (auto fragment = dyn_cast<FragmentType>(type))
        return fragment.getElementType();
      return type;
    };
    Type source = element(cast.getValue().getType());
    Type result = element(cast.getResult().getType());
    if (!isa<IntegerType, IndexType>(source) ||
        !isa<IntegerType, IndexType>(result))
      break;
    value = stripBroadcast(cast.getValue());
  }
  return value;
}

bool sameBroadcastCoordinateExpression(Value lhs, Value rhs) {
  auto leftType = dyn_cast<FragmentType>(lhs.getType());
  auto rightType = dyn_cast<FragmentType>(rhs.getType());
  if (!leftType || !rightType)
    return false;
  FragmentType common = leftType.getShape().size() > rightType.getShape().size()
                            ? leftType : rightType;
  auto project = [&](FragmentType source) {
    auto projection = queryBroadcastProjection(source, common);
    if (projection.isExact())
      return projection;
    auto permutation = queryAxisPermutation(source, common);
    if (!permutation)
      return projection;
    for (auto [targetAxis, sourceAxis] : llvm::enumerate(*permutation))
      if (source.getShape()[sourceAxis] != common.getShape()[targetAxis] &&
          constantPhysicalExpression(
              cast<PhysicalExprAttr>(source.getShape()[sourceAxis])) != 1)
        return projection;
    projection.targetToSource.clear();
    for (int64_t sourceAxis : *permutation)
      projection.targetToSource.push_back(sourceAxis);
    projection.state = BroadcastProjectionState::Exact;
    return projection;
  };
  auto leftProjection = project(leftType);
  auto rightProjection = project(rightType);
  if (!leftProjection.isExact() || !rightProjection.isExact())
    return false;
  using Axes = SmallVector<std::optional<unsigned>, 4>;
  auto element = [](Type type) {
    auto fragment = dyn_cast<FragmentType>(type);
    return fragment ? fragment.getElementType() : type;
  };
  auto projectOperand = [&](Value operand, Value parent,
                            ArrayRef<std::optional<unsigned>> axes)
      -> FailureOr<Axes> {
    Axes projected(axes.size());
    auto source = dyn_cast<FragmentType>(operand.getType());
    if (!source)
      return projected;
    auto target = dyn_cast<FragmentType>(parent.getType());
    if (!target)
      return failure();
    auto projection = queryBroadcastProjection(source, target);
    if (!projection.isExact())
      return failure();
    for (auto [axis, parentAxis] : llvm::enumerate(axes))
      if (parentAxis)
        projected[axis] = projection.targetToSource[*parentAxis];
    return projected;
  };
  auto sameLanes = [](Type type, ArrayRef<std::optional<unsigned>> left,
                      ArrayRef<std::optional<unsigned>> right) {
    auto fragment = dyn_cast<FragmentType>(type);
    if (!fragment)
      return true;
    auto varying = [&](std::optional<unsigned> axis) {
      if (axis) {
        auto extent = cast<PhysicalExprAttr>(fragment.getShape()[*axis]);
        if (extent.getKind() ==
                PhysicalExprKind::Constant &&
            extent.getValue() == 1)
          return std::optional<unsigned>();
      }
      return axis;
    };
    return llvm::all_of(llvm::zip(left, right), [&](auto pair) {
      return varying(std::get<0>(pair)) == varying(std::get<1>(pair));
    });
  };
  std::function<bool(Value, Axes, Value, Axes, unsigned)> equivalent;
  equivalent = [&](Value left, Axes leftAxes, Value right, Axes rightAxes,
                   unsigned depth) -> bool {
    if (depth >= 32 || element(left.getType()) != element(right.getType()))
      return false;
    if (left == right)
      return sameLanes(left.getType(), leftAxes, rightAxes);
    auto unwrap = [&](Value current, Axes axes)
        -> std::optional<std::pair<Value, Axes>> {
      Value input;
      if (auto broadcast = current.getDefiningOp<BroadcastOp>())
        input = broadcast.getValue();
      else if (auto splat = current.getDefiningOp<SplatOp>())
        input = splat.getValue();
      else if (auto transpose = current.getDefiningOp<TransposeOp>()) {
        for (std::optional<unsigned> &axis : axes)
          if (axis)
            axis = transpose.getPermutation()[*axis];
        return std::make_pair(transpose.getValue(), std::move(axes));
      } else if (auto reshape = current.getDefiningOp<ReshapeOp>()) {
        auto source = cast<FragmentType>(reshape.getValue().getType());
        auto target = cast<FragmentType>(current.getType());
        if (source.getShape() == target.getShape())
          return std::make_pair(reshape.getValue(), std::move(axes));
        unsigned sourceRank = 0, resultRank = 0;
        for (Attribute attribute : reshape.getReassociation()) {
          auto group = cast<ReshapeGroupAttr>(attribute);
          sourceRank += group.getSourceAxes().size();
          resultRank += group.getResultAxes().size();
        }
        unsigned sourcePrefix = source.getShape().size() - sourceRank;
        unsigned resultPrefix = target.getShape().size() - resultRank;
        if (sourcePrefix != resultPrefix)
          return std::nullopt;
        Axes resultToSource(target.getShape().size());
        for (unsigned axis = 0; axis < sourcePrefix; ++axis)
          resultToSource[axis] = axis;
        for (Attribute attribute : reshape.getReassociation()) {
          auto group = cast<ReshapeGroupAttr>(attribute);
          if (group.getSourceAxes().empty() || group.getResultAxes().empty())
            continue;
          if (group.getSourceAxes().size() != 1 ||
              group.getResultAxes().size() != 1)
            return std::nullopt;
          unsigned sourceAxis = sourcePrefix + group.getSourceAxes()[0];
          unsigned resultAxis = resultPrefix + group.getResultAxes()[0];
          if (source.getShape()[sourceAxis] != target.getShape()[resultAxis])
            return std::nullopt;
          resultToSource[resultAxis] = sourceAxis;
        }
        Axes projected(axes.size());
        for (auto [axis, resultAxis] : llvm::enumerate(axes))
          if (resultAxis)
            projected[axis] = resultToSource[*resultAxis];
        return std::make_pair(reshape.getValue(), std::move(projected));
      }
      if (!input)
        return std::nullopt;
      FailureOr<Axes> projected = projectOperand(input, current, axes);
      return succeeded(projected)
                 ? std::optional(std::make_pair(input, std::move(*projected)))
                 : std::nullopt;
    };
    if (auto unwrapped = unwrap(left, leftAxes))
      return equivalent(unwrapped->first, std::move(unwrapped->second), right,
                        std::move(rightAxes), depth + 1);
    if (auto unwrapped = unwrap(right, rightAxes))
      return equivalent(left, std::move(leftAxes), unwrapped->first,
                        std::move(unwrapped->second), depth + 1);
    if (!isa<FragmentType>(left.getType()) &&
        !isa<FragmentType>(right.getType()))
      return sameScalarExpression(left, right);
    auto leftRange = left.getDefiningOp<MakeRangeOp>();
    auto rightRange = right.getDefiningOp<MakeRangeOp>();
    if (leftRange || rightRange)
      return leftRange && rightRange &&
             sameLanes(left.getType(), leftAxes, rightAxes) &&
             sourceAxisIdentity(leftRange) == sourceAxisIdentity(rightRange) &&
             samePhysicalScalarExpression(leftRange.getStart(), rightRange.getStart()) &&
             samePhysicalScalarExpression(leftRange.getExtent(), rightRange.getExtent()) &&
             samePhysicalScalarExpression(leftRange.getStep(), rightRange.getStep());
    auto leftBinary = left.getDefiningOp<BinaryOp>();
    auto rightBinary = right.getDefiningOp<BinaryOp>();
    if (!leftBinary || !rightBinary ||
        leftBinary.getOperatorKind() != rightBinary.getOperatorKind() ||
        leftBinary.getApproximate() != rightBinary.getApproximate() ||
        leftBinary.getFlushToZero() != rightBinary.getFlushToZero())
      return false;
    for (auto [leftOperand, rightOperand] : llvm::zip(leftBinary->getOperands(),
                                                    rightBinary->getOperands())) {
      FailureOr<Axes> leftMapping = projectOperand(leftOperand, left, leftAxes);
      FailureOr<Axes> rightMapping = projectOperand(rightOperand, right, rightAxes);
      if (failed(leftMapping) || failed(rightMapping) ||
          !equivalent(leftOperand, *leftMapping, rightOperand, *rightMapping,
                      depth + 1))
        return false;
    }
    return true;
  };
  return equivalent(lhs, leftProjection.targetToSource, rhs,
                    rightProjection.targetToSource, 0);
}

bool derivesFromAccessCoordinate(Value value, Value coordinate) {
  // Coordinate replay can duplicate a pure expression before CSE. Its bounds
  // still apply when the complete typed expression and SSA leaves are equal.
  if (sameScalarExpression(value, coordinate))
    return true;
  auto valueType = dyn_cast<FragmentType>(value.getType());
  auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
  if (valueType && coordinateType &&
      queryBroadcastProjection(valueType, coordinateType).isExact() &&
      sameScalarExpression(stripBroadcast(value), stripBroadcast(coordinate)))
    return true;
  if (sameBroadcastCoordinateExpression(value, coordinate))
    return true;
  value = stripIntegerIndexCasts(value);
  coordinate = stripIntegerIndexCasts(coordinate);
  if (sameScalarExpression(value, coordinate))
    return true;
  auto valueRange = value.getDefiningOp<MakeRangeOp>();
  auto coordinateRange = coordinate.getDefiningOp<MakeRangeOp>();
  return valueRange && coordinateRange &&
         sourceAxisIdentity(valueRange) == sourceAxisIdentity(coordinateRange) &&
         sameScalarExpression(valueRange.getStart(),
                              coordinateRange.getStart()) &&
         sameScalarExpression(valueRange.getExtent(),
                              coordinateRange.getExtent()) &&
         sameScalarExpression(valueRange.getStep(), coordinateRange.getStep());
}

bool isInclusiveCoordinateUpperBound(Value value, Value coordinate) {
  coordinate = stripBroadcast(coordinate);
  if (!value.getType().isIndex() || !coordinate.getType().isIndex())
    return false;
  PhysicalExprAttr upper = queryNonNegativeIndexUpperBound(coordinate);
  return upper && queryLaunchExpression(value) == upper;
}

bool valueKnownPositive(Value value, unsigned depth);
bool valueKnownNonNegative(Value value, unsigned depth);

enum class IndexSign { Unknown, NonNegative, Positive };

std::optional<std::pair<int64_t, int64_t>>
positiveExtentBounds(func::FuncOp kernel, PhysicalExprAttr extent);

IndexSign physicalIndexSign(PhysicalExprAttr expression, func::FuncOp kernel) {
  auto kind = expression.getKind();
  if (kind == PhysicalExprKind::Constant)
    return expression.getValue() > 0 ? IndexSign::Positive
         : expression.getValue() == 0 ? IndexSign::NonNegative : IndexSign::Unknown;
  if (kind == PhysicalExprKind::Dimension)
    return IndexSign::NonNegative;
  if (kind == PhysicalExprKind::Parameter) {
    FailureOr<ParameterAttr> parameter = queryParameterBySymbol(kernel, expression.getParameterReference().getName());
    if (failed(parameter))
      return IndexSign::Unknown;
    auto candidates = (*parameter).getCandidates().asArrayRef();
    if (candidates.empty())
      return IndexSign::Unknown;
    if (llvm::all_of(candidates, [](int64_t value) { return value > 0; }))
      return IndexSign::Positive;
    return llvm::all_of(candidates, [](int64_t value) { return value >= 0; })
               ? IndexSign::NonNegative : IndexSign::Unknown;
  }
  if (kind == PhysicalExprKind::Select && expression.getOperands().size() == 3)
    return std::min(
        physicalIndexSign(cast<PhysicalExprAttr>(expression.getOperands()[1]), kernel),
        physicalIndexSign(cast<PhysicalExprAttr>(expression.getOperands()[2]), kernel));
  if (kind == PhysicalExprKind::NextPowerOfTwo ||
      kind == PhysicalExprKind::Multiply)
    return positiveExtentBounds(kernel, expression) ? IndexSign::Positive
                                                    : IndexSign::Unknown;
  if (expression.getOperands().size() != 2)
    return IndexSign::Unknown;
  IndexSign lhs = physicalIndexSign(
      cast<PhysicalExprAttr>(expression.getOperands()[0]), kernel);
  IndexSign rhs = physicalIndexSign(
      cast<PhysicalExprAttr>(expression.getOperands()[1]), kernel);
  if (kind == PhysicalExprKind::Maximum)
    return std::max(lhs, rhs);
  if (kind == PhysicalExprKind::Minimum)
    return std::min(lhs, rhs);
  if ((kind == PhysicalExprKind::CeilDiv || kind == PhysicalExprKind::FloorDiv) &&
      lhs != IndexSign::Unknown && rhs == IndexSign::Positive)
    return kind == PhysicalExprKind::CeilDiv ? lhs : IndexSign::NonNegative;
  // Unknown arithmetic remains unknown; in particular this does not assume
  // that a product or sum of dynamic index values cannot overflow.
  return IndexSign::Unknown;
}

bool valueBelowDelinearizeExtent(Value value, Value extent, unsigned depth) {
  if (!value || depth >= 32)
    return false;
  if (auto mapping = queryDecodedCoordinate(value))
      return mapping->extents[mapping->axis] == extent &&
             valueKnownNonNegative(mapping->linear, depth + 1) &&
             llvm::all_of(mapping->extents, [&](Value bound) {
               return valueKnownNonNegative(bound, depth + 1);
             });
  auto multiply = value.getDefiningOp<BinaryOp>();
  if (!multiply || multiply.getOperatorKind() != BinaryOperator::Multiply)
    return false;
  // For x >= 0 and s > 0, floor(x / s) * s <= x.
  for (auto [quotient, step] :
       {std::pair{multiply.getLhs(), multiply.getRhs()},
        std::pair{multiply.getRhs(), multiply.getLhs()}}) {
    auto divide = quotient.getDefiningOp<BinaryOp>();
    if (divide && divide.getOperatorKind() == BinaryOperator::FloorDivide &&
        divide.getRhs() == step && valueKnownPositive(step, depth + 1) &&
        valueKnownNonNegative(divide.getLhs(), depth + 1) &&
        valueBelowDelinearizeExtent(divide.getLhs(), extent, depth + 1))
      return true;
  }
  return false;
}

bool valueKnownPositive(Value value, unsigned depth = 0) {
  if (!value || depth >= 32)
    return false;
  value = stripIntegerIndexCasts(value);
  if (auto physical = value.getDefiningOp<PhysicalExprOp>())
    return physicalIndexSign(physical.getExpression(),
                             physical->getParentOfType<func::FuncOp>()) ==
           IndexSign::Positive;
  if (std::optional<int64_t> constant = integerConstant(value))
    return *constant > 0;
  if (auto parameter = queryParameter(value))
    return llvm::all_of(
        parameter.getCandidates().asArrayRef(),
        [](int64_t candidate) { return candidate > 0; });
  if (auto bound = value.getDefiningOp<RangeBoundOp>()) {
    auto range = bound.getRange().getDefiningOp<RangeOp>();
    return range && bound.getBound() == 2 &&
           valueKnownPositive(range.getStep(), depth + 1);
  }
  auto binary = value.getDefiningOp<BinaryOp>();
  if (!binary)
    return false;
  if (binary.getOperatorKind() == BinaryOperator::Subtract)
    return valueBelowDelinearizeExtent(binary.getRhs(), binary.getLhs(), depth + 1);
  if (binary.getOperatorKind() == BinaryOperator::Maximum &&
      value.getType().isIndex())
    return valueKnownPositive(binary.getLhs(), depth + 1) ||
           valueKnownPositive(binary.getRhs(), depth + 1);
  if (binary.getOperatorKind() == BinaryOperator::Multiply ||
      binary.getOperatorKind() == BinaryOperator::Add ||
      binary.getOperatorKind() == BinaryOperator::Minimum ||
      binary.getOperatorKind() == BinaryOperator::MinimumNum)
    return valueKnownPositive(binary.getLhs(), depth + 1) &&
           valueKnownPositive(binary.getRhs(), depth + 1);
  return false;
}

bool valueKnownNonNegative(Value value, unsigned depth = 0) {
  if (!value || depth >= 32)
    return false;
  value = stripIntegerIndexCasts(value);
  if (auto reshape = value.getDefiningOp<ReshapeOp>())
    return valueKnownNonNegative(reshape.getValue(), depth + 1);
  if (auto transpose = value.getDefiningOp<TransposeOp>())
    return valueKnownNonNegative(transpose.getValue(), depth + 1);
  if (auto physical = value.getDefiningOp<PhysicalExprOp>())
    return physicalIndexSign(physical.getExpression(),
                             physical->getParentOfType<func::FuncOp>()) !=
           IndexSign::Unknown;
  if (std::optional<int64_t> constant = integerConstant(value))
    return *constant >= 0;
  if (value.getDefiningOp<ProgramIdOp>())
    return true;
  if (auto parameter = queryParameter(value))
    return llvm::all_of(
        parameter.getCandidates().asArrayRef(),
        [](int64_t candidate) { return candidate >= 0; });
  if (auto coordinate = queryDecodedCoordinate(value))
    return valueKnownNonNegative(coordinate->linear, depth + 1) &&
           llvm::all_of(coordinate->extents, [&](Value extent) {
             return valueKnownNonNegative(extent, depth + 1);
           });
  if (value.getDefiningOp<DimOp>())
    return true;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (auto binding = getArgumentBinding(argument);
        binding && binding.getKind() == ArgumentKind::Dimension)
      return true;
    auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    if (loop && argument == loop.getInductionVar()) {
      return valueKnownPositive(loop.getStep(), depth + 1) &&
             valueKnownNonNegative(loop.getLowerBound(), depth + 1);
    }
  }
  if (auto coordinate = value.getDefiningOp<WorksetCoordinateOp>())
    return valueKnownNonNegative(coordinate.getCoordinate(), depth + 1);
  if (auto range = value.getDefiningOp<MakeRangeOp>()) {
    return valueKnownPositive(range.getStep(), depth + 1) &&
           valueKnownNonNegative(range.getStart(), depth + 1);
  }
  if (auto bound = value.getDefiningOp<RangeBoundOp>()) {
    auto range = bound.getRange().getDefiningOp<RangeOp>();
    if (!range)
      return false;
    if (bound.getBound() == 0)
      return valueKnownNonNegative(range.getStart(), depth + 1);
    if (bound.getBound() == 1)
      return valueKnownNonNegative(range.getStop(), depth + 1);
    return valueKnownPositive(range.getStep(), depth + 1);
  }
  if (auto select = value.getDefiningOp<SelectOp>())
    return valueKnownNonNegative(select.getTrueValue(), depth + 1) &&
           valueKnownNonNegative(select.getFalseValue(), depth + 1);
  auto binary = value.getDefiningOp<BinaryOp>();
  if (!binary)
    return false;
  bool lhs = valueKnownNonNegative(binary.getLhs(), depth + 1);
  bool rhs = valueKnownNonNegative(binary.getRhs(), depth + 1);
  switch (binary.getOperatorKind()) {
  case BinaryOperator::Add:
  case BinaryOperator::Multiply:
  case BinaryOperator::Minimum:
  case BinaryOperator::MinimumNum:
    return lhs && rhs;
  case BinaryOperator::Maximum:
  case BinaryOperator::MaximumNum:
    return lhs || rhs;
  case BinaryOperator::Subtract: {
    if (valueBelowDelinearizeExtent(binary.getRhs(), binary.getLhs(), depth + 1))
      return true;
    if (lhs && rhs) {
      std::function<bool(Value, unsigned)> orderedByCondition =
          [&](Value condition, unsigned remaining) {
        if (remaining == 0)
          return false;
        if (auto conjunction = condition.getDefiningOp<BinaryOp>())
          if (conjunction.getOperatorKind() == BinaryOperator::LogicalAnd)
            return orderedByCondition(conjunction.getLhs(), remaining - 1) ||
                   orderedByCondition(conjunction.getRhs(), remaining - 1);
        auto compare = condition.getDefiningOp<CompareOp>();
        return compare &&
               ((compare.getPredicate() == ComparePredicate::Ge &&
                 compare.getLhs() == binary.getLhs() &&
                 compare.getRhs() == binary.getRhs()) ||
                (compare.getPredicate() == ComparePredicate::Le &&
                 compare.getRhs() == binary.getLhs() &&
                 compare.getLhs() == binary.getRhs()));
      };
      for (Operation *parent = binary->getParentOp(); parent;
           parent = parent->getParentOp())
        if (auto conditional = dyn_cast<scf::IfOp>(parent))
          if (conditional.getThenRegion().isAncestor(binary->getParentRegion()) &&
              orderedByCondition(conditional.getCondition(), 32 - depth))
            return true;
    }
    std::optional<int64_t> subtrahend = integerConstant(binary.getRhs());
    if (!subtrahend)
      return false;
    if (*subtrahend <= 0)
      return lhs;
    Value minuend = stripIntegerIndexCasts(binary.getLhs());
    auto parameter = queryParameter(minuend);
    return parameter &&
           llvm::all_of(parameter.getCandidates().asArrayRef(),
                        [&](int64_t candidate) {
                          return candidate >= *subtrahend;
                        });
  }
  case BinaryOperator::FloorDivide: {
    return lhs && valueKnownPositive(binary.getRhs(), depth + 1);
  }
  case BinaryOperator::Remainder:
    return lhs && valueKnownPositive(binary.getRhs(), depth + 1);
  default:
    return false;
  }
}

bool coordinateKnownNonNegative(Value coordinate) {
  if (queryNonNegativeIndexUpperBound(coordinate))
    return true;
  return valueKnownNonNegative(coordinate);
}

bool coordinateRangeWithinResource(Value coordinate, Value resource,
                                   unsigned axis);
bool valueMatchesExtent(Value value, PhysicalExprAttr extent);

std::optional<std::pair<int64_t, int64_t>>
nonNegativeExtentBounds(func::FuncOp kernel, PhysicalExprAttr extent) {
  if (!extent)
    return std::nullopt;
  auto kind = extent.getKind();
  if (auto constant = constantPhysicalExpression(extent)) {
    if (*constant < 0)
      return std::nullopt;
    return std::pair{*constant, *constant};
  }
  if (kind == PhysicalExprKind::Parameter) {
    FailureOr<ParameterAttr> parameter =
        queryParameterBySymbol(kernel, extent.getParameterReference().getName());
    if (failed(parameter))
      return std::nullopt;
    // Provider configuration formation may rebind resident capacity. Its
    // positive sign is stable, but placeholder candidates and shared tuples
    // cannot prove a numeric upper bound or absence of index overflow.
    if (parameter->getRole() ==
        ParameterRole::ResidentWorkers)
      return std::nullopt;
    auto candidates = parameter->getCandidates().asArrayRef();
    if (candidates.empty() ||
        llvm::any_of(candidates, [](int64_t value) { return value <= 0; }))
      return std::nullopt;
    if (auto configurations =
            kernel->getAttrOfType<ConfigurationSetAttr>(configurationsAttr);
        configurations && !configurations.getRows().empty()) {
      int64_t minimum = std::numeric_limits<int64_t>::max();
      int64_t maximum = 0;
      bool bound = true;
      for (Attribute attribute : configurations.getRows()) {
        auto tuple = dyn_cast<DictionaryAttr>(attribute);
        auto selected = tuple ? tuple.getAs<IntegerAttr>(extent.getParameterReference().getName())
                              : IntegerAttr();
        if (!selected || selected.getInt() <= 0 ||
            !llvm::is_contained(candidates, selected.getInt())) {
          bound = false;
          break;
        }
        minimum = std::min(minimum, selected.getInt());
        maximum = std::max(maximum, selected.getInt());
      }
      if (bound)
        return std::pair{minimum, maximum};
    }
    return std::pair{*llvm::min_element(candidates),
                     *llvm::max_element(candidates)};
  }
  if (kind == PhysicalExprKind::NextPowerOfTwo &&
      extent.getOperands().size() == 1) {
    auto bounds = nonNegativeExtentBounds(
        kernel, cast<PhysicalExprAttr>(extent.getOperands()[0]));
    if (!bounds || bounds->second > (int64_t{1} << 62))
      return std::nullopt;
    return std::pair{static_cast<int64_t>(llvm::PowerOf2Ceil(
                         std::max<int64_t>(1, bounds->first))),
                     static_cast<int64_t>(llvm::PowerOf2Ceil(
                         std::max<int64_t>(1, bounds->second)))};
  }
  if (extent.getOperands().size() != 2)
    return std::nullopt;
  auto lhs = nonNegativeExtentBounds(
      kernel, cast<PhysicalExprAttr>(extent.getOperands()[0]));
  auto rhs = nonNegativeExtentBounds(
      kernel, cast<PhysicalExprAttr>(extent.getOperands()[1]));
  if (kind == PhysicalExprKind::Minimum) {
    if (lhs && rhs)
      return std::pair{std::min(lhs->first, rhs->first),
                       std::min(lhs->second, rhs->second)};
    for (unsigned known : {0u, 1u}) {
      auto bound = known == 0 ? lhs : rhs;
      auto other = cast<PhysicalExprAttr>(extent.getOperands()[1 - known]);
      if (bound && physicalIndexSign(other, kernel) == IndexSign::Positive)
        return std::pair{std::min<int64_t>(1, bound->first), bound->second};
    }
    return std::nullopt;
  }
  if (!lhs || !rhs)
    return std::nullopt;
  __int128 minimum, maximum;
  switch (kind) {
  case PhysicalExprKind::Add:
    minimum = static_cast<__int128>(lhs->first) + rhs->first;
    maximum = static_cast<__int128>(lhs->second) + rhs->second;
    break;
  case PhysicalExprKind::Subtract:
    minimum = static_cast<__int128>(lhs->first) - rhs->second;
    maximum = static_cast<__int128>(lhs->second) - rhs->first;
    break;
  case PhysicalExprKind::Multiply:
    minimum = static_cast<__int128>(lhs->first) * rhs->first;
    maximum = static_cast<__int128>(lhs->second) * rhs->second;
    break;
  case PhysicalExprKind::Maximum:
    minimum = std::max(lhs->first, rhs->first);
    maximum = std::max(lhs->second, rhs->second);
    break;
  case PhysicalExprKind::FloorDiv:
  case PhysicalExprKind::CeilDiv: {
    if (rhs->first <= 0)
      return std::nullopt;
    bool ceil = kind == PhysicalExprKind::CeilDiv;
    minimum = (static_cast<__int128>(lhs->first) +
               (ceil ? rhs->second - 1 : 0)) / rhs->second;
    maximum = (static_cast<__int128>(lhs->second) +
               (ceil ? rhs->first - 1 : 0)) / rhs->first;
    break;
  }
  default:
    return std::nullopt;
  }
  if (minimum < 0 || maximum > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return std::pair{static_cast<int64_t>(minimum), static_cast<int64_t>(maximum)};
}

std::optional<std::pair<int64_t, int64_t>>
positiveExtentBounds(func::FuncOp kernel, PhysicalExprAttr extent) {
  auto bounds = nonNegativeExtentBounds(kernel, extent);
  return bounds && bounds->first > 0 ? bounds : std::nullopt;
}

bool linearizedGatherWithinResource(Value coordinate, Value resource) {
  auto sourceReshape = resource.getDefiningOp<ReshapeOp>();
  auto source = sourceReshape
                    ? dyn_cast<FragmentType>(sourceReshape.getValue().getType())
                    : FragmentType();
  auto flatSource = dyn_cast<FragmentType>(resource.getType());
  if (!source || !flatSource || source.getShape().size() < 2 ||
      flatSource.getShape().size() != 1)
    return false;
  auto binary = [](Value value, BinaryOperator kind) {
    auto operation = stripBroadcast(value).getDefiningOp<BinaryOp>();
    Type type = value.getType();
    if (auto fragment = dyn_cast<FragmentType>(type))
      type = fragment.getElementType();
    return operation && operation.getOperatorKind() == kind && type.isIndex()
               ? operation : BinaryOp();
  };
  auto offset = binary(coordinate, BinaryOperator::Add);
  auto tail = offset ? binary(offset.getRhs(), BinaryOperator::Remainder)
                     : BinaryOp();
  auto prefix = offset ? binary(offset.getLhs(), BinaryOperator::Add)
                       : BinaryOp();
  auto outer = prefix ? binary(prefix.getLhs(), BinaryOperator::Multiply)
                      : BinaryOp();
  auto selected = prefix ? binary(prefix.getRhs(), BinaryOperator::Multiply)
                         : BinaryOp();
  auto quotient = outer ? binary(outer.getLhs(), BinaryOperator::FloorDivide)
                        : BinaryOp();
  auto sourceStride = outer ? binary(outer.getRhs(), BinaryOperator::Multiply)
                            : BinaryOp();
  auto resultStride = quotient
                          ? binary(quotient.getRhs(), BinaryOperator::Multiply)
                          : BinaryOp();
  if (!tail || !selected || !sourceStride || !resultStride ||
      !sameScalarExpression(quotient.getLhs(), tail.getLhs()) ||
      !sameScalarExpression(selected.getRhs(), tail.getRhs()) ||
      !sameScalarExpression(sourceStride.getRhs(), tail.getRhs()) ||
      !sameScalarExpression(resultStride.getRhs(), tail.getRhs()))
    return false;
  auto ordinal = quotient.getLhs().getDefiningOp<MakeRangeOp>();
  Value index = stripBroadcast(selected.getLhs());
  uint64_t castLimit = std::numeric_limits<int64_t>::max();
  auto elementType = [](Type type) {
    if (auto fragment = dyn_cast<FragmentType>(type))
      return fragment.getElementType();
    return type;
  };
  while (auto cast = index.getDefiningOp<CastOp>()) {
    Type sourceType = elementType(cast.getValue().getType());
    Type resultType = elementType(cast.getResult().getType());
    if (!isa<IndexType, IntegerType>(sourceType) ||
        !isa<IndexType, IntegerType>(resultType))
      return false;
    if (auto integer = dyn_cast<IntegerType>(resultType);
        integer && integer.getWidth() < 64) {
      unsigned valueBits = integer.getWidth() - (integer.isUnsigned() ? 0 : 1);
      castLimit = std::min(castLimit, (uint64_t{1} << valueBits) - 1);
    }
    index = stripBroadcast(cast.getValue());
  }
  auto indexReshape = index.getDefiningOp<ReshapeOp>();
  auto indices = indexReshape
                     ? dyn_cast<FragmentType>(indexReshape.getValue().getType())
                     : FragmentType();
  auto flatIndices = dyn_cast<FragmentType>(index.getType());
  if (!ordinal || !indices || !flatIndices ||
      indices.getShape().size() != source.getShape().size() ||
      flatIndices.getShape().size() != 1 ||
      integerConstant(ordinal.getStart()) != 0 ||
      !isUnitStepValue(ordinal.getStep()) ||
      !matchesResourceExtent(ordinal.getExtent(), index, 0))
    return false;

  auto kernel = sourceReshape->getParentOfType<func::FuncOp>();
  auto positiveExtentLimit = [&](PhysicalExprAttr extent) -> std::optional<int64_t> {
    auto bounds = positiveExtentBounds(kernel, extent);
    return bounds ? std::optional<int64_t>(bounds->second) : std::nullopt;
  };
  if (!positiveExtentLimit(resourceExtentExpression(resource, 0)) ||
      !positiveExtentLimit(resourceExtentExpression(index, 0)))
    return false;

  for (unsigned axis = 0; axis < source.getShape().size(); ++axis) {
    if (!matchesResourceExtent(sourceStride.getLhs(), sourceReshape.getValue(), axis) ||
        !matchesResourceExtent(resultStride.getLhs(), indexReshape.getValue(), axis))
      continue;
    bool sameOtherAxes = true;
    for (unsigned other = 0; other < source.getShape().size(); ++other)
      sameOtherAxes &= other == axis ||
                       (source.getShape()[other] == indices.getShape()[other] &&
                        source.getAxisMaps()[other] == indices.getAxisMaps()[other]);
    if (!sameOtherAxes)
      continue;
    auto selectedLimit = positiveExtentLimit(
        cast<PhysicalExprAttr>(source.getShape()[axis]));
    if (!selectedLimit || static_cast<uint64_t>(*selectedLimit - 1) > castLimit)
      continue;
    MLIRContext *context = resource.getContext();
    auto inner = PhysicalExprAttr::get(
        context, PhysicalExprKind::Constant, 1,
        StringAttr::get(context, ""), ArrayAttr::get(context, {}));
    for (unsigned dimension = axis + 1; dimension < source.getShape().size(); ++dimension) {
      auto extent = cast<PhysicalExprAttr>(source.getShape()[dimension]);
      inner = dimension == axis + 1
                  ? extent
                  : PhysicalExprAttr::get(
                        context, PhysicalExprKind::Multiply,
                        0, StringAttr::get(context, ""),
                        ArrayAttr::get(context, {inner, extent}));
    }
    if (!valueMatchesExtent(stripBroadcast(tail.getRhs()), inner) ||
        !coordinateRangeWithinResource(indexReshape.getValue(),
                                       sourceReshape.getValue(), axis))
      continue;
    // The exact reshapes preserve row-major element order.  For a bounded
    // ordinal, q < outer and remainder < inner; the selected index is < B.
    // Thus q*(B*inner) + index*inner + remainder is in [0, outer*B*inner).
    // The positive finite shape limits above also exclude index overflow.
    return true;
  }
  return false;
}

bool coordinateRangeWithinResource(Value coordinate, Value resource,
                                   unsigned axis) {
  if (axis == 0 && linearizedGatherWithinResource(coordinate, resource))
    return true;
  coordinate = stripIntegerIndexCasts(coordinate);
  // Reassociation and permutation preserve the set of coordinate values.
  if (auto reshape = coordinate.getDefiningOp<ReshapeOp>())
    return coordinateRangeWithinResource(reshape.getValue(), resource, axis);
  if (auto transpose = coordinate.getDefiningOp<TransposeOp>())
    return coordinateRangeWithinResource(transpose.getValue(), resource, axis);
  if (PhysicalExprAttr upper = queryNonNegativeIndexUpperBound(coordinate)) {
    auto maximum = constantPhysicalExpression(upper);
    auto extent = constantPhysicalExpression(
        resourceExtentExpression(resource, axis));
    // Scalar loop coordinates remain bounded after a proven mask folds away.
    // The query's bound is inclusive; the resource extent is exclusive.
    if (maximum && extent && *maximum >= 0 && *maximum < *extent)
      return true;
  }
  if (auto subtract = coordinate.getDefiningOp<BinaryOp>();
      subtract && subtract.getOperatorKind() == BinaryOperator::Subtract) {
    auto range = stripIntegerIndexCasts(subtract.getLhs()).getDefiningOp<MakeRangeOp>();
    if (range && isUnitStepValue(range.getStep()) &&
        sameScalarExpression(stripBroadcast(subtract.getRhs()), range.getStart()) &&
        matchesResourceExtent(range.getExtent(), resource, axis))
      return true;
  }
  if (std::optional<int64_t> constant = integerConstant(coordinate)) {
    PhysicalExprAttr extent = resourceExtentExpression(resource, axis);
    auto kernel = resource.getParentRegion()->getParentOfType<func::FuncOp>();
    auto bounds = positiveExtentBounds(kernel, extent);
    return bounds && *constant >= 0 && *constant < bounds->first;
  }
  if (auto select = coordinate.getDefiningOp<SelectOp>()) {
    if (!coordinateRangeWithinResource(select.getFalseValue(), resource, axis))
      return false;
    if (coordinateRangeWithinResource(select.getTrueValue(), resource, axis))
      return true;
    bool lower = false;
    bool upper = false;
    std::function<void(Value)> inspectPredicate = [&](Value predicate) {
      predicate = stripBroadcast(predicate);
      if (auto reshape = predicate.getDefiningOp<ReshapeOp>()) {
        auto source = cast<FragmentType>(reshape.getValue().getType());
        auto target = cast<FragmentType>(predicate.getType());
        auto unit = [](Attribute attribute) {
          auto extent = cast<PhysicalExprAttr>(attribute);
          return extent.getKind() ==
                     PhysicalExprKind::Constant &&
                 extent.getValue() == 1;
        };
        bool projection = llvm::all_of(
            reshape.getReassociation(), [&](Attribute attribute) {
              auto group = cast<ReshapeGroupAttr>(attribute);
              auto inputs = group.getSourceAxes().asArrayRef();
              auto outputs = group.getResultAxes().asArrayRef();
              if (inputs.empty())
                return llvm::all_of(outputs, [&](int64_t axis) {
                  return unit(target.getShape()[axis]);
                });
              if (outputs.empty())
                return llvm::all_of(inputs, [&](int64_t axis) {
                  return unit(source.getShape()[axis]);
                });
              return inputs.size() == 1 && outputs.size() == 1 &&
                     source.getShape()[inputs[0]] == target.getShape()[outputs[0]];
            });
        if (projection)
          inspectPredicate(reshape.getValue());
        return;
      }
      if (integerConstant(predicate) == 0) {
        lower = upper = true;
        return;
      }
      if (auto conjunction = predicate.getDefiningOp<BinaryOp>()) {
        if (conjunction.getOperatorKind() == BinaryOperator::LogicalAnd ||
            conjunction.getOperatorKind() == BinaryOperator::BitwiseAnd) {
          inspectPredicate(conjunction.getLhs());
          inspectPredicate(conjunction.getRhs());
        }
        return;
      }
      auto compare = predicate.getDefiningOp<CompareOp>();
      if (!compare)
        return;
      Value selected = select.getTrueValue();
      bool lhsIndex = derivesFromAccessCoordinate(compare.getLhs(), selected);
      bool rhsIndex = derivesFromAccessCoordinate(compare.getRhs(), selected);
      if ((compare.getPredicate() == ComparePredicate::Ge && lhsIndex &&
           integerConstant(compare.getRhs()) == 0) ||
          (compare.getPredicate() == ComparePredicate::Le && rhsIndex &&
           integerConstant(compare.getLhs()) == 0))
        lower = true;
      if ((compare.getPredicate() == ComparePredicate::Lt && lhsIndex &&
           upperBoundWithinResource(compare.getRhs(), resource, axis)) ||
          (compare.getPredicate() == ComparePredicate::Gt && rhsIndex &&
           upperBoundWithinResource(compare.getLhs(), resource, axis)))
        upper = true;
    };
    inspectPredicate(select.getCondition());
    return lower && upper;
  }
  auto range = coordinate.getDefiningOp<MakeRangeOp>();
  if (!range || !isUnitStepValue(range.getStep()) ||
      !coordinateKnownNonNegative(coordinate))
    return false;
  if (integerConstant(range.getStart()) == 0 &&
      matchesResourceExtent(range.getExtent(), resource, axis))
    return true;
  if (Value limit = queryCompleteTileLimit(range);
      limit && upperBoundWithinResource(limit, resource, axis))
    return true;
  PhysicalExprAttr resourceExtent = resourceExtentExpression(resource, axis);
  if (!resourceExtent ||
      resourceExtent.getKind() !=
          PhysicalExprKind::Constant)
    return false;
  std::optional<int64_t> start = integerConstant(range.getStart());
  std::optional<int64_t> extent = integerConstant(range.getExtent());
  return start && extent && *start >= 0 && *extent >= 0 &&
         static_cast<__int128>(*start) + *extent <= resourceExtent.getValue();
}

bool hasExactPhysicalRangeCoverage(Value coordinate, Value upperBound) {
  coordinate = stripBroadcast(coordinate);
  upperBound = stripScalarIdentity(upperBound);
  auto range = coordinate.getDefiningOp<MakeRangeOp>();
  auto add = upperBound.getDefiningOp<BinaryOp>();
  if (!range || !add || add.getOperatorKind() != BinaryOperator::Add ||
      !isUnitStepValue(range.getStep()))
    return false;
  Value extent;
  if (sameScalarExpression(add.getLhs(), range.getStart()))
    extent = add.getRhs();
  else if (sameScalarExpression(add.getRhs(), range.getStart()))
    extent = add.getLhs();
  return extent && sameScalarExpression(extent, range.getExtent());
}

bool reductionTypeConsumesSource(Type type, ArrayRef<int64_t> axes,
                                 PhysicalSourceAxis source,
                                 std::optional<int64_t> dimension) {
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
      return reductionTypeConsumesSource(cast<TypeAttr>(field).getValue(), axes,
                                         source, dimension);
    });
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return false;
  for (int64_t axis : axes) {
    if (axis < 0 || axis >= static_cast<int64_t>(fragment.getAxisMaps().size()))
      continue;
    auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    if (PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()} == source &&
        (!dimension || mapping.getDimensionId() == *dimension))
      return true;
  }
  return false;
}

Value accessResource(Operation *operation) {
  if (auto access = dyn_cast<AccessOpInterface>(operation);
      access && access.isMemoryAccess())
    return access.getAccessResource();
  return {};
}

bool valueMatchesExtent(Value value, PhysicalExprAttr extent) {
  if (queryLaunchExpression(value) == extent)
    return true;
  if (auto physical = value.getDefiningOp<PhysicalExprOp>())
    return physical.getExpression() == extent;
  auto kind = extent.getKind();
  if (kind == PhysicalExprKind::Constant) {
    std::optional<int64_t> constant = integerConstant(value);
    return constant && *constant == extent.getValue();
  }
  if (kind == PhysicalExprKind::Parameter) {
    auto parameter = queryParameter(value);
    return parameter &&
           parameter.getName() == extent.getParameterReference().getName();
  }
  return false;
}

bool isExclusiveProgramRange(MakeRangeOp range, func::FuncOp kernel) {
  if (!range || integerConstant(range.getLogicalStart()) != 0 ||
      integerConstant(range.getStep()) != 1)
    return false;
  Value extent = stripScalarIdentity(range.getExtent());
  if (auto expression = extent.getDefiningOp<PhysicalExprOp>();
      expression && expression.getExpression().getKind() ==
          PhysicalExprKind::Parameter) {
    auto parameter = queryParameterBySymbol(kernel, expression.getExpression().getParameterReference().getName());
    if (failed(parameter))
      return false;
  }
  bool positive = integerConstant(extent).value_or(0) > 0;
  if (auto parameter = queryParameter(extent))
    positive = llvm::all_of(parameter.getCandidates().asArrayRef(),
                           [](int64_t value) { return value > 0; });
  auto multiply = stripScalarIdentity(range.getStart()).getDefiningOp<BinaryOp>();
  if (!positive || !multiply ||
      multiply.getOperatorKind() != BinaryOperator::Multiply)
    return false;
  Value coordinate;
  if (sameScalarExpression(multiply.getLhs(), extent))
    coordinate = stripScalarIdentity(multiply.getRhs());
  else if (sameScalarExpression(multiply.getRhs(), extent))
    coordinate = stripScalarIdentity(multiply.getLhs());
  auto argument = dyn_cast_or_null<BlockArgument>(coordinate);
  auto group = argument ? dyn_cast<ExecutionGroupOp>(argument.getOwner()->getParentOp())
                        : ExecutionGroupOp();
  auto decoded = coordinate ? queryDecodedCoordinate(coordinate) : std::nullopt;
  if (!decoded || decoded->extents.size() != 1)
    return false;
  auto program = decoded->linear.getDefiningOp<ProgramIdOp>();
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  auto launchExtent = group ? cast<PhysicalExprAttr>(group.getLaunchExtents()[0])
                           : queryLaunchExpression(decoded->extents[0]);
  if (!program || program.getAxis() != 0 || !space || space.empty() ||
      space[0] != launchExtent)
    return false;
  for (Attribute attribute : space.getValue().drop_front()) {
    auto expression = cast<PhysicalExprAttr>(attribute);
    if (expression.getKind() != PhysicalExprKind::Constant ||
        expression.getValue() != 1)
      return false;
  }
  auto launch = cast<PhysicalExprAttr>(space[0]);
  return launch.getKind() == PhysicalExprKind::CeilDiv &&
         launch.getOperands().size() == 2 &&
         launch.getOperands()[0] == queryLaunchExpression(range.getLogicalStop()) &&
         launch.getOperands()[1] == queryLaunchExpression(extent);
}

Value stripRangeProjection(Value value) {
  Value original = value;
  SmallVector<Operation *> projections;
  while (value) {
    if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
      projections.push_back(broadcast);
      value = broadcast.getValue();
      continue;
    }
    auto reshape = value.getDefiningOp<ReshapeOp>();
    if (!reshape || llvm::any_of(reshape.getReassociation(), [](Attribute attribute) {
          auto group = cast<ReshapeGroupAttr>(attribute);
          return group.getSourceAxes().size() > 1 || group.getResultAxes().size() > 1;
        }))
      break;
    projections.push_back(reshape);
    value = reshape.getValue();
  }
  auto range = value.getDefiningOp<MakeRangeOp>();
  if (!range)
    return original;
  unsigned axis = 0;
  for (Operation *operation : llvm::reverse(projections)) {
    auto input = cast<FragmentType>(operation->getOperand(0).getType());
    auto output = cast<FragmentType>(operation->getResult(0).getType());
    std::optional<unsigned> projected;
    if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
      unsigned rank = 0;
      for (Attribute attribute : reshape.getReassociation())
        rank += cast<ReshapeGroupAttr>(attribute).getSourceAxes().size();
      unsigned prefix = input.getShape().size() - rank;
      if (axis < prefix)
        projected = axis;
      else
        for (Attribute attribute : reshape.getReassociation()) {
          auto group = cast<ReshapeGroupAttr>(attribute);
          if (group.getSourceAxes().size() == 1 && group.getResultAxes().size() == 1 &&
              group.getSourceAxes()[0] + prefix == axis)
            projected = group.getResultAxes()[0] + prefix;
        }
    } else {
      auto mapping = queryAxisProjection(input, output);
      if (mapping.isExact())
        for (auto [position, source] : llvm::enumerate(mapping.targetToSource))
          if (source && *source == axis) {
            if (projected)
              return original;
            projected = position;
          }
    }
    if (!projected || input.getShape()[axis] != output.getShape()[*projected])
      return original;
    axis = *projected;
  }
  return value;
}

bool isPrivateWorkspaceProgramIndex(Value coordinate, Value buffer,
                                    Operation *access, unsigned axis) {
  auto program = coordinate.getDefiningOp<ProgramIdOp>();
  auto kernel = access->getParentOfType<func::FuncOp>();
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!program || !space || axis >= space.size() || program.getAxis() != axis)
    return false;
  DominanceInfo dominance(kernel);
  return llvm::any_of(buffer.getUsers(), [&](Operation *user) {
    auto assumption = dyn_cast<AssumeInBoundsOp>(user);
    return assumption && assumption.getResource() == buffer &&
           assumption.getIndex() == coordinate &&
           assumption.getAxis() == axis &&
           dominance.properlyDominates(assumption, access);
  });
}

} // namespace

std::optional<int64_t> constantLogicalRangeCardinality(MakeRangeOp range) {
  auto constant = [](Value value) -> std::optional<int64_t> {
    auto folded = dyn_cast_or_null<IntegerAttr>(
        UniformValueAnalysis(describeUniformValue).evaluate(value));
    return folded && folded.getValue().getBitWidth() <= 64
               ? std::optional<int64_t>(folded.getInt()) : std::nullopt;
  };
  auto step = constant(range.getStep());
  if (!step || *step <= 0)
    return std::nullopt;
  std::optional<__int128> distance;
  auto start = constant(range.getLogicalStart());
  auto stop = constant(range.getLogicalStop());
  if (start && stop)
    distance = static_cast<__int128>(*stop) - *start;
  else if (auto add = stripScalarIdentity(range.getLogicalStop())
                          .getDefiningOp<BinaryOp>();
           add && add.getOperatorKind() == BinaryOperator::Add)
    for (auto [base, offset] : {std::pair{add.getLhs(), add.getRhs()},
                               std::pair{add.getRhs(), add.getLhs()}})
      if (samePhysicalScalarExpression(base, range.getLogicalStart()))
        if (auto size = constant(offset))
          distance = *size;
  if (!distance) {
    struct Offset {
      Value base;
      __int128 amount = 0;
      __int128 minimum = 0;
      __int128 maximum = 0;
      __int128 scale = 1;
    };
    std::function<Offset(Value)> splitOffset = [&](Value value) -> Offset {
      value = stripScalarIdentity(value);
      auto binary = value.getDefiningOp<BinaryOp>();
      if (!binary)
        return {value};
      Value base;
      std::optional<__int128> increment;
      if (binary.getOperatorKind() == BinaryOperator::Multiply) {
        auto factor = constant(binary.getRhs());
        base = binary.getLhs();
        if (!factor) {
          factor = constant(binary.getLhs());
          base = binary.getRhs();
        }
        if (!factor || *factor <= 0)
          return {value};
        Offset result = splitOffset(base);
        const __int128 lower = std::numeric_limits<int64_t>::min();
        const __int128 upper = std::numeric_limits<int64_t>::max();
        if (result.scale > upper / *factor || result.minimum < lower ||
            result.maximum > upper)
          return {value};
        result.scale *= *factor;
        result.amount *= *factor;
        result.minimum = std::min(result.minimum, result.minimum * *factor);
        result.maximum = std::max(result.maximum, result.maximum * *factor);
        return result;
      }
      if (binary.getOperatorKind() == BinaryOperator::Add) {
        if (auto rhs = constant(binary.getRhs())) {
          base = binary.getLhs();
          increment = *rhs;
        } else if (auto lhs = constant(binary.getLhs())) {
          base = binary.getRhs();
          increment = *lhs;
        }
      } else if (binary.getOperatorKind() == BinaryOperator::Subtract) {
        if (auto rhs = constant(binary.getRhs())) {
          base = binary.getLhs();
          increment = -static_cast<__int128>(*rhs);
        }
      }
      if (!increment)
        return {value};
      Offset result = splitOffset(base);
      result.amount += *increment;
      result.minimum = std::min(result.minimum, result.amount);
      result.maximum = std::max(result.maximum, result.amount);
      return result;
    };
    Offset begin = splitOffset(range.getLogicalStart());
    Offset end = splitOffset(range.getLogicalStop());
    if (begin.scale == end.scale &&
        samePhysicalScalarExpression(begin.base, end.base)) {
      auto upper = queryNonNegativeIndexUpperBound(begin.base);
      // Cancel a common affine base only when all intermediate index arithmetic
      // stays in range; signed wrap must not turn a short slice into a full one.
      if (upper && upper.getKind() ==
                       PhysicalExprKind::Constant &&
          std::min(begin.minimum, end.minimum) >=
              std::numeric_limits<int64_t>::min() &&
          std::max(begin.maximum, end.maximum) <=
              std::numeric_limits<int64_t>::max() &&
          static_cast<__int128>(upper.getValue()) * begin.scale +
                  std::max(begin.maximum, end.maximum) <=
              std::numeric_limits<int64_t>::max())
        distance = end.amount - begin.amount;
    }
  }
  if (!distance || *distance < 0)
    return std::nullopt;
  __int128 size = (*distance + *step - 1) / *step;
  return size <= std::numeric_limits<int64_t>::max()
             ? std::optional<int64_t>(size) : std::nullopt;
}

bool isProvablySingletonLogicalRange(MakeRangeOp range) {
  if (auto size = constantLogicalRangeCardinality(range))
    return *size == 1;

  Value logicalStop = stripScalarIdentity(range.getLogicalStop());
  auto add = logicalStop.getDefiningOp<BinaryOp>();
  if (!add || add.getOperatorKind() != BinaryOperator::Add)
    return false;
  return (sameScalarExpression(add.getLhs(), range.getLogicalStart()) &&
          sameScalarExpression(add.getRhs(), range.getStep())) ||
         (sameScalarExpression(add.getRhs(), range.getLogicalStart()) &&
          sameScalarExpression(add.getLhs(), range.getStep()));
}

bool isProgramCoordinateRange(MakeRangeOp range) {
  auto coordinate = range.getStart().getDefiningOp<WorksetCoordinateOp>();
  return coordinate && range->hasAttr(worksetCoordinateRangeAttr) &&
         coordinate.getSourceId() == range.getSourceId() &&
         coordinate.getSourceAxis() == range.getSourceAxis() &&
         !range.getDerived() &&
         sameScalarExpression(coordinate.getStep(), range.getStep());
}

std::optional<BinaryOperator> queryBinaryCombineKind(Region &region) {
  if (!llvm::hasSingleElement(region))
    return std::nullopt;
  Block &block = region.front();
  if (block.getNumArguments() != 2)
    return std::nullopt;
  auto yield = dyn_cast_or_null<YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return std::nullopt;
  Value result = stripCombineProjection(yield.getValues().front());
  auto binary = result.getDefiningOp<BinaryOp>();
  if (!binary)
    return std::nullopt;
  Value lhs = stripCombineProjection(binary.getLhs());
  Value rhs = stripCombineProjection(binary.getRhs());
  if (!((lhs == block.getArgument(0) && rhs == block.getArgument(1)) ||
        (lhs == block.getArgument(1) && rhs == block.getArgument(0))))
    return std::nullopt;
  return binary.getOperatorKind();
}

bool isPhysicalReplayNode(Operation *operation, PhysicalReplayScope scope,
                          bool allowAccesses) {
  if (!operation || operation->getNumResults() == 0)
    return false;
  if (isAccessNode(operation))
    return allowAccesses && operation->getNumRegions() == 0;
  if (operation->getNumRegions() != 0 && !isa<ReduceOp, ScanOp>(operation))
    return false;
  return scope == PhysicalReplayScope::Coordinate
             ? isCoordinateReplayNode(operation)
             : isValueReplayNode(operation);
}

bool typeCarriesTraversal(Type type, PhysicalSourceAxis source,
                          int64_t dimension) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return llvm::any_of(queryFragmentAxes(fragment, source),
                        [&](const PhysicalAxisProjection &projection) {
                          return projection.dimensionId == dimension;
                        });
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
      return typeCarriesTraversal(cast<TypeAttr>(field).getValue(), source,
                                  dimension);
    });
  return false;
}

PhysicalParameterBinding queryParameterBinding(ParameterAttr parameter) {
  PhysicalParameterBinding result;
  if (!parameter)
    return result;
  auto dimension = parameter.getBinding().getDimension();
  if (dimension && dimension.getInt() <= 0)
    return result;
  if (dimension)
    result.dimension = dimension.getInt();
  if (auto source = parameter.getBinding().getSource())
    result.source =
        PhysicalSourceAxis{source.getSourceId(), source.getSourceAxis(),
                           source.getDerived()};
  result.state = result.dimension || result.source ? PhysicalFactState::Exact
                                                   : PhysicalFactState::Unknown;
  return result;
}

ParameterAttr queryParameter(Value value) {
  if (value.getType().isIndex())
    value = stripScalarIdentity(value);
  if (auto read = value.getDefiningOp<ParameterOp>())
    return read.getDeclaration();
  auto expression = value.getDefiningOp<PhysicalExprOp>();
  if (!expression || !expression.getExpression().getParameterReference())
    return {};
  return lookupParameter(expression->getParentOfType<func::FuncOp>(),
                         expression.getExpression().getParameterReference());
}

FailureOr<ParameterAttr> queryParameterBySymbol(func::FuncOp kernel,
                                                StringAttr symbol) {
  auto parameter = lookupParameter(kernel, symbol);
  return parameter ? FailureOr<ParameterAttr>(parameter)
                   : FailureOr<ParameterAttr>(failure());
}

FailureOr<ParameterAttr> queryBlockingParameter(func::FuncOp kernel,
                                              MakeRangeOp range) {
  auto fragment = dyn_cast<FragmentType>(range.getResult().getType());
  auto extent = fragment && fragment.getShape().size() == 1
                    ? dyn_cast<PhysicalExprAttr>(fragment.getShape()[0])
                    : PhysicalExprAttr();
  if (extent && extent.getKind() ==
                    PhysicalExprKind::Parameter)
    return queryParameterBySymbol(kernel, extent.getParameterReference().getName());

  PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                            range.getDerived()};
  FailureOr<int64_t> dimension = queryRangeDimension(range);
  ParameterAttr sourceMatch;
  ParameterAttr dimensionMatch;
  bool sourceAmbiguous = false;
  bool dimensionAmbiguous = false;
  for (Attribute declaration : getParameterDeclarations(kernel)) {
    auto parameter = cast<ParameterAttr>(declaration);
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    if (!binding.isExact())
      continue;
    if (binding.source && *binding.source == source) {
      if (sourceMatch && sourceMatch != parameter)
        sourceAmbiguous = true;
      else
        sourceMatch = parameter;
    }
    if (succeeded(dimension) && binding.dimension &&
        *binding.dimension == *dimension) {
      if (dimensionMatch && dimensionMatch != parameter)
        dimensionAmbiguous = true;
      else
        dimensionMatch = parameter;
    }
  }
  if (sourceMatch && !sourceAmbiguous)
    return sourceMatch;
  return dimensionMatch && !dimensionAmbiguous
             ? FailureOr<ParameterAttr>(dimensionMatch)
             : FailureOr<ParameterAttr>(failure());
}

PhysicalSourceAxis sourceAxisIdentity(AxisMapAttr mapping) {
  return {mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived()};
}

PhysicalSourceAxis sourceAxisIdentity(MakeRangeOp range) {
  return {range.getSourceId(), range.getSourceAxis(), range.getDerived()};
}

PhysicalAxisProjection queryFragmentAxis(Type type,
                                         PhysicalSourceAxis source,
                                         std::optional<int64_t> expectedDimension) {
  PhysicalAxisProjection result;
  result.source = source;
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return result;
  std::optional<unsigned> axis;
  std::optional<int64_t> dimension;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getSourceId() != source.sourceId ||
        mapping.getSourceAxis() != source.sourceAxis ||
        mapping.getDerived() != source.derived ||
        (expectedDimension && mapping.getDimensionId() != *expectedDimension))
      continue;
    if ((axis && *axis != mapping.getFragmentAxis()) ||
        (dimension && *dimension != mapping.getDimensionId())) {
      result.state = PhysicalFactState::Ambiguous;
      return result;
    }
    axis = mapping.getFragmentAxis();
    dimension = mapping.getDimensionId();
  }
  if (!axis || !dimension)
    return result;
  result.state = PhysicalFactState::Exact;
  result.fragmentAxis = *axis;
  result.dimensionId = *dimension;
  return result;
}

FailureOr<AxisMapAttr> queryAxisMap(Type type, unsigned fragmentAxis) {
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return failure();
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getFragmentAxis() == fragmentAxis)
      return mapping;
  }
  return failure();
}

FailureOr<int64_t> queryRangeDimension(MakeRangeOp range) {
  return querySourceDimension(range.getResult().getType(),
                              sourceAxisIdentity(range));
}

bool samePhysicalScalarExpression(Value lhs, Value rhs) {
  if (sameScalarExpression(lhs, rhs))
    return true;
  if (lhs.getType() != rhs.getType() ||
      !isa<IndexType, IntegerType>(lhs.getType()))
    return false;
  UniformValueAnalysis constants(describeUniformValue);
  Attribute left = constants.evaluate(lhs);
  Attribute right = constants.evaluate(rhs);
  return left && right && equalUniformConstants(left, right);
}

bool isKnownPositiveExtent(PhysicalExprAttr extent, func::FuncOp kernel) {
  return physicalIndexSign(extent, kernel) == IndexSign::Positive;
}

std::optional<std::pair<int64_t, int64_t>>
queryPositiveExtentBounds(PhysicalExprAttr extent, func::FuncOp kernel) {
  return positiveExtentBounds(kernel, extent);
}

bool isLaunchUniformScalar(Value value, func::FuncOp kernel) {
  SmallVector<Value> pending{value};
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!current.getType().isIntOrIndexOrFloat())
      return false;
    if (auto argument = dyn_cast<BlockArgument>(current)) {
      if (argument.getOwner() != &kernel.front())
        return false;
      continue;
    }
    Operation *producer = current.getDefiningOp();
    if (!producer)
      return false;
    if (!visited.insert(producer).second)
      continue;
    if (isa<arith::ConstantOp, PhysicalExprOp, ParameterOp>(producer))
      continue;
    if (!isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(producer))
      return false;
    llvm::append_range(pending, producer->getOperands());
  }
  return true;
}

PhysicalExprAttr queryLaunchExpression(Value value) {
  std::function<PhysicalExprAttr(Value, unsigned)> query =
      [&](Value current, unsigned depth) -> PhysicalExprAttr {
    if (!current || depth >= 32 ||
        !isa<IndexType, IntegerType>(current.getType()))
      return {};
    MLIRContext *context = current.getContext();
    auto expression = [&](PhysicalExprKind kind, int64_t constant = 0,
                          StringRef symbol = {},
                          ArrayRef<Attribute> operands = {}) {
      return PhysicalExprAttr::get(
          context, kind, constant,
          kind == PhysicalExprKind::Parameter
              ? Attribute(ParameterRefAttr::get(context, StringAttr::get(context, symbol)))
              : Attribute(StringAttr::get(context, symbol)),
          ArrayAttr::get(context, operands));
    };
    if (auto constant = current.getDefiningOp<arith::ConstantOp>()) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      if (!integer || integer.getValue().getBitWidth() > 64)
        return {};
      auto type = dyn_cast<IntegerType>(current.getType());
      if (type && (type.isUnsigned() || type.getWidth() == 1)) {
        if (integer.getValue().getActiveBits() > 63)
          return {};
        return expression(PhysicalExprKind::Constant,
                          integer.getValue().getZExtValue());
      }
      return expression(PhysicalExprKind::Constant, integer.getInt());
    }
    if (auto argument = dyn_cast<BlockArgument>(current))
      return queryArgumentExpression(argument);
    if (auto dim = current.getDefiningOp<DimOp>())
      return resourceExtentExpression(dim.getView(), dim.getAxis());
    if (auto physical = current.getDefiningOp<PhysicalExprOp>())
      return physical.getExpression();
    if (auto parameter = queryParameter(current))
      return expression(PhysicalExprKind::Parameter, 0,
                        parameter.getName().getValue());
    if (auto cast = current.getDefiningOp<arith::IndexCastOp>()) {
      auto integer = dyn_cast<IntegerType>(cast.getIn().getType());
      return integer && !integer.isUnsigned() && integer.getWidth() > 1 &&
                     integer.getWidth() <= 64 && current.getType().isIndex()
                 ? query(cast.getIn(), depth + 1)
                 : PhysicalExprAttr();
    }
    if (auto cast = current.getDefiningOp<CastOp>()) {
      Type source = cast.getValue().getType();
      auto integer = dyn_cast<IntegerType>(source);
      bool preservesInteger = current.getType().isIndex() && integer &&
          (integer.getWidth() < 64 ||
           (integer.getWidth() == 64 && !integer.isUnsigned()));
      return source == current.getType() || preservesInteger
                 ? query(cast.getValue(), depth + 1)
                 : PhysicalExprAttr();
    }
    if (auto bound = current.getDefiningOp<RangeBoundOp>()) {
      auto range = bound.getRange().getDefiningOp<RangeOp>();
      if (!range)
        return {};
      return query(bound.getBound() == 0 ? range.getStart()
                   : bound.getBound() == 1 ? range.getStop() : range.getStep(),
                   depth + 1);
    }
    if (auto compare = current.getDefiningOp<CompareOp>()) {
      bool equal = compare.getPredicate() == ComparePredicate::Eq;
      if ((!equal && compare.getPredicate() != ComparePredicate::Ne) ||
          compare.getLhs().getType() != compare.getRhs().getType())
        return {};
      PhysicalExprAttr lhs = query(compare.getLhs(), depth + 1);
      PhysicalExprAttr rhs = query(compare.getRhs(), depth + 1);
      if (!lhs || !rhs)
        return {};
      // Earlier fixed-width arithmetic may wrap differently from Python's
      // launch evaluation. Direct ABI values and constants have no such step.
      if (!lhs.getOperands().empty() || !rhs.getOperands().empty())
        return {};
      // Equality only observes a zero difference, which is preserved by
      // same-width integer wraparound and host launch arithmetic alike.
      auto difference = expression(PhysicalExprKind::Subtract, 0, {}, {lhs, rhs});
      return expression(PhysicalExprKind::Select, 0, {},
          {difference, expression(PhysicalExprKind::Constant, equal ? 0 : 1),
           expression(PhysicalExprKind::Constant, equal ? 1 : 0)});
    }
    auto binary = current.getDefiningOp<BinaryOp>();
    if (!binary || !current.getType().isIndex())
      return {};
    PhysicalExprAttr lhs = query(binary.getLhs(), depth + 1);
    PhysicalExprAttr rhs = query(binary.getRhs(), depth + 1);
    if (!lhs || !rhs)
      return {};
    PhysicalExprKind kind;
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add: kind = PhysicalExprKind::Add; break;
    case BinaryOperator::Subtract: kind = PhysicalExprKind::Subtract; break;
    case BinaryOperator::Multiply: kind = PhysicalExprKind::Multiply; break;
    case BinaryOperator::FloorDivide: kind = PhysicalExprKind::FloorDiv; break;
    case BinaryOperator::Minimum: kind = PhysicalExprKind::Minimum; break;
    case BinaryOperator::Maximum: kind = PhysicalExprKind::Maximum; break;
    default: return {};
    }
    return expression(kind, 0, {}, {lhs, rhs});
  };
  return query(value, 0);
}

PhysicalExprAttr queryLaunchRangeExtent(MakeRangeOp range) {
  MLIRContext *context = range.getContext();
  auto expression = [&](PhysicalExprKind kind, int64_t value = 0,
                        ArrayRef<Attribute> operands = {}) {
    return PhysicalExprAttr::get(
        context, kind, value, StringAttr::get(context),
        ArrayAttr::get(context, operands));
  };
  if (auto count = constantLogicalRangeCardinality(range))
    return expression(PhysicalExprKind::Constant, *count);
  PhysicalExprAttr start = queryLaunchExpression(range.getLogicalStart());
  PhysicalExprAttr stop = queryLaunchExpression(range.getLogicalStop());
  PhysicalExprAttr step = queryLaunchExpression(range.getStep());
  if (!start || !stop || !step)
    return {};
  if (step.getKind() == PhysicalExprKind::Constant &&
      step.getValue() <= 0)
    return {};
  PhysicalExprAttr distance =
      expression(PhysicalExprKind::Subtract, 0, {stop, start});
  distance = expression(PhysicalExprKind::Maximum, 0,
                        {distance, expression(PhysicalExprKind::Constant, 0)});
  return expression(PhysicalExprKind::CeilDiv, 0, {distance, step});
}

namespace {
struct IndexBounds {
  bool nonNegative = false;
  PhysicalExprAttr upper;
  int64_t lower = 0;
};

Value clampedIncrementLimit(Value value, Value base) {
  auto add = stripScalarIdentity(value).getDefiningOp<BinaryOp>();
  if (!add || !value.getType().isIndex() ||
      add.getOperatorKind() != BinaryOperator::Add)
    return {};
  for (auto [start, increment] :
       {std::pair{add.getLhs(), add.getRhs()},
        std::pair{add.getRhs(), add.getLhs()}}) {
    if (!sameScalarExpression(start, base))
      continue;
    auto minimum = stripScalarIdentity(increment).getDefiningOp<BinaryOp>();
    if (!minimum || minimum.getOperatorKind() != BinaryOperator::Minimum)
      continue;
    for (auto [remaining, step] :
         {std::pair{minimum.getLhs(), minimum.getRhs()},
          std::pair{minimum.getRhs(), minimum.getLhs()}}) {
      auto difference = stripScalarIdentity(remaining).getDefiningOp<BinaryOp>();
      auto amount = integerConstant(step);
      if (difference && difference.getOperatorKind() == BinaryOperator::Subtract &&
          sameScalarExpression(difference.getRhs(), base) && amount && *amount >= 0)
        return difference.getLhs();
    }
  }
  return {};
}

IndexBounds queryIndexBounds(Value value) {
  using Bounds = IndexBounds;
  auto constantUpper = [](Bounds bounds) -> std::optional<int64_t> {
    if (bounds.nonNegative && bounds.upper &&
        bounds.upper.getKind() == PhysicalExprKind::Constant)
      return bounds.upper.getValue();
    return std::nullopt;
  };
  std::function<Bounds(Value, unsigned)> bound =
      [&](Value current, unsigned depth) -> Bounds {
    if (!current || depth >= 32 || !current.getType().isIndex())
      return {};
    current = stripScalarIdentity(current);
    auto expression = [&](PhysicalExprKind kind, int64_t constant,
                          ArrayRef<Attribute> operands = {}) {
      MLIRContext *context = current.getContext();
      return PhysicalExprAttr::get(
          context, kind, constant,
          StringAttr::get(context, ""), ArrayAttr::get(context, operands));
    };
    if (auto coordinate = current.getDefiningOp<WorksetCoordinateOp>())
      return bound(coordinate.getCoordinate(), depth + 1);
    if (current.getDefiningOp<ProgramIdOp>())
      return {true, {}};
    auto dimensionExpression = [&](BlockArgument argument) -> PhysicalExprAttr {
      auto binding = getArgumentBinding(argument);
      return binding && binding.getKind() == ArgumentKind::Dimension
          ? queryArgumentExpression(argument) : PhysicalExprAttr{};
    };
    if (auto dim = current.getDefiningOp<DimOp>())
      return {true, resourceExtentExpression(dim.getView(), dim.getAxis())};
    if (auto argument = dyn_cast<BlockArgument>(current)) {
      if (PhysicalExprAttr dimension = dimensionExpression(argument))
        return {true, dimension};
      auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
      if (loop && argument == loop.getInductionVar()) {
        Bounds lower = bound(loop.getLowerBound(), depth + 1);
        Bounds upperBound = bound(loop.getUpperBound(), depth + 1);
        std::optional<int64_t> upper =
            constantUpper(upperBound);
        std::optional<int64_t> step = integerConstant(loop.getStep());
        if (auto parameter = queryParameter(loop.getStep())) {
          auto candidates = parameter.getCandidates().asArrayRef();
          if (parameter.getRole() !=
                  ParameterRole::ResidentWorkers &&
              !candidates.empty() && llvm::all_of(candidates, [](int64_t value) {
                return value > 0;
              }))
            step = *llvm::max_element(candidates);
        }
        if (lower.nonNegative && upper && step && *step > 0 &&
            *upper <= std::numeric_limits<int64_t>::max() - *step + 1)
          return {true, expression(PhysicalExprKind::Constant,
                                   std::max<int64_t>(*upper - 1, 0)), lower.lower};
        if (lower.nonNegative && step && *step == 1) {
          // The induction value exists only in an executing loop body. Its
          // exclusive upper bound therefore exceeds the nonnegative lower
          // bound, even when that upper expression can be negative elsewhere.
          PhysicalExprAttr end = queryLaunchExpression(loop.getUpperBound());
          if (!end && upperBound.nonNegative)
            end = upperBound.upper;
          if (end)
            return {true, expression(PhysicalExprKind::Subtract, 0,
                                    {end, expression(PhysicalExprKind::Constant, 1)}),
                    lower.lower};
        }
      }
      auto whileLoop = dyn_cast<scf::WhileOp>(argument.getOwner()->getParentOp());
      if (whileLoop && argument.getOwner() == &whileLoop.getAfter().front()) {
        auto condition = cast<scf::ConditionOp>(
            whileLoop.getBefore().front().getTerminator());
        auto carried = dyn_cast<BlockArgument>(stripScalarIdentity(
            condition.getArgs()[argument.getArgNumber()]));
        auto compare = condition.getCondition().getDefiningOp<CompareOp>();
        if (carried && carried.getOwner() == &whileLoop.getBefore().front() &&
            compare && compare.getPredicate() == ComparePredicate::Lt &&
            sameScalarExpression(compare.getLhs(), carried)) {
          auto yield = cast<scf::YieldOp>(whileLoop.getAfter().front().getTerminator());
          Value limit = clampedIncrementLimit(
              yield.getResults()[carried.getArgNumber()], argument);
          PhysicalExprAttr end = limit ? queryLaunchExpression(limit) : PhysicalExprAttr();
          Bounds initial = bound(whileLoop.getInits()[carried.getArgNumber()], depth + 1);
          if (end && queryLaunchExpression(compare.getRhs()) == end &&
              initial.nonNegative && bound(limit, depth + 1).nonNegative) {
            // With 0 <= p < end, p + min(step, end-p) remains in [p,end]
            // without signed overflow. The guard bounds each executing iteration.
            return {true, expression(PhysicalExprKind::Subtract, 0,
                                    {end, expression(PhysicalExprKind::Constant, 1)}),
                    initial.lower};
          }
        }
      }
    }
    if (auto parameter = queryParameter(current)) {
      auto schema = parameter;
      if (llvm::all_of(schema.getCandidates().asArrayRef(),
                       [](int64_t candidate) { return candidate >= 0; }))
        return {true, PhysicalExprAttr::get(
            current.getContext(), PhysicalExprKind::Parameter,
            0, schema.getReference(), ArrayAttr::get(current.getContext(), {}))};
      return {};
    }
    if (auto mapping = queryDecodedCoordinate(current)) {
        if (!valueKnownNonNegative(mapping->linear) ||
            !llvm::all_of(mapping->extents, [](Value extent) {
              return valueKnownNonNegative(extent);
            }))
          return {};
        Value extent = stripScalarIdentity(
            mapping->extents[mapping->axis]);
        if (std::optional<int64_t> constant = integerConstant(extent))
          return {true, expression(PhysicalExprKind::Constant,
                                   std::max<int64_t>(*constant - 1, 0))};
        PhysicalExprAttr upper;
        if (auto physical = extent.getDefiningOp<PhysicalExprOp>())
          upper = physical.getExpression();
        else if (auto dim = extent.getDefiningOp<DimOp>())
          upper = resourceExtentExpression(dim.getView(), dim.getAxis());
        else if (auto argument = dyn_cast<BlockArgument>(extent))
          upper = dimensionExpression(argument);
        if (!upper)
          return {};
        return {true, expression(PhysicalExprKind::Maximum, 0,
            {expression(PhysicalExprKind::Subtract, 0,
                        {upper, expression(PhysicalExprKind::Constant, 1)}),
             expression(PhysicalExprKind::Constant, 0)})};
      }
    if (std::optional<int64_t> constant = integerConstant(current)) {
      if (*constant >= 0)
        return {true, expression(PhysicalExprKind::Constant, *constant), *constant};
      return {};
    }
    auto binary = current.getDefiningOp<BinaryOp>();
    if (!binary)
      return {};
    Bounds lhs = bound(binary.getLhs(), depth + 1);
    Bounds rhs = bound(binary.getRhs(), depth + 1);
    std::optional<int64_t> lhsConstant = constantUpper(lhs);
    std::optional<int64_t> rhsConstant = constantUpper(rhs);
    if (binary.getOperatorKind() == BinaryOperator::Multiply &&
        ((lhsConstant && *lhsConstant == 0) ||
         (rhsConstant && *rhsConstant == 0)))
      return {true, expression(PhysicalExprKind::Constant, 0)};
    if (binary.getOperatorKind() == BinaryOperator::Subtract) {
      PhysicalExprAttr exactLhs = queryLaunchExpression(binary.getLhs());
      if (lhs.nonNegative && rhs.nonNegative && exactLhs && rhs.upper) {
        auto preceding = expression(PhysicalExprKind::Subtract, 0,
                                    {exactLhs, expression(PhysicalExprKind::Constant, 1)});
        if (rhs.upper == exactLhs || rhs.upper == preceding)
          return {true, exactLhs, rhs.upper == preceding ? 1 : 0};
      }
      if (auto constant = integerConstant(binary.getLhs());
          constant && *constant >= 0 && rhsConstant &&
          *rhsConstant <= *constant)
        return {true, expression(PhysicalExprKind::Constant,
                                 *constant - rhs.lower),
                *constant - *rhsConstant};
      auto ordinal = dyn_cast<BlockArgument>(stripScalarIdentity(binary.getRhs()));
      auto loop = ordinal
                      ? dyn_cast<scf::ForOp>(ordinal.getOwner()->getParentOp())
                      : scf::ForOp();
      // Within 0 <= iv < end, end - iv lies in [1, end].  This
      // also proves reversed ordinals without assuming a static extent.
      if (loop && ordinal == loop.getInductionVar() &&
          integerConstant(loop.getStep()) == 1 &&
          bound(loop.getLowerBound(), depth + 1).nonNegative &&
          bound(loop.getUpperBound(), depth + 1).nonNegative) {
        if (sameScalarExpression(binary.getLhs(), loop.getUpperBound()) &&
            lhs.upper)
          return {true, lhs.upper, 1};
        auto last = binary.getLhs().getDefiningOp<BinaryOp>();
        Bounds end = bound(loop.getUpperBound(), depth + 1);
        if (last && last.getOperatorKind() == BinaryOperator::Subtract &&
            integerConstant(last.getRhs()) == 1 && end.upper &&
            sameScalarExpression(last.getLhs(), loop.getUpperBound()))
          return {true, expression(PhysicalExprKind::Subtract, 0,
                                   {end.upper,
                                    expression(PhysicalExprKind::Constant, 1)})};
      }
      std::optional<int64_t> amount = integerConstant(binary.getRhs());
      if (amount && *amount >= 0 && lhs.nonNegative && lhs.upper &&
          lhs.lower >= *amount)
        return {true, expression(PhysicalExprKind::Subtract, 0,
                                 {lhs.upper, expression(PhysicalExprKind::Constant,
                                                        *amount)}),
                lhs.lower - *amount};
    }
    if (binary.getOperatorKind() == BinaryOperator::Add) {
      for (Value start : {binary.getLhs(), binary.getRhs()}) {
        Value limit = clampedIncrementLimit(current, start);
        if (!limit)
          continue;
        Bounds lower = bound(start, depth + 1);
        Bounds upper = bound(limit, depth + 1);
        PhysicalExprAttr end = queryLaunchExpression(limit);
        if (lower.nonNegative && upper.nonNegative && end)
          return {true, end, std::min(lower.lower, upper.lower)};
      }
      for (auto [difference, increment] :
           {std::pair{binary.getLhs(), binary.getRhs()},
            std::pair{binary.getRhs(), binary.getLhs()}}) {
        auto subtract = difference.getDefiningOp<BinaryOp>();
        if (subtract && subtract.getOperatorKind() == BinaryOperator::Subtract &&
            sameScalarExpression(subtract.getRhs(), increment) &&
            valueKnownNonNegative(increment)) {
          // For nonnegative index values E and x, E-x is representable and
          // (E-x)+x is exactly E, including when the intermediate is negative.
          Bounds original = bound(subtract.getLhs(), depth + 1);
          if (original.nonNegative)
            return original;
        }
        std::optional<int64_t> amount = integerConstant(increment);
        Bounds preceding = bound(difference, depth + 1);
        if (amount && *amount >= 0 && preceding.nonNegative && preceding.upper &&
            preceding.upper.getKind() ==
                PhysicalExprKind::Subtract &&
            preceding.upper.getOperands().size() == 2) {
          auto headroom = cast<PhysicalExprAttr>(preceding.upper.getOperands()[1]);
          if (headroom.getKind() ==
                  PhysicalExprKind::Constant &&
              headroom.getValue() >= *amount &&
              preceding.lower <= std::numeric_limits<int64_t>::max() - *amount) {
            auto base = cast<PhysicalExprAttr>(preceding.upper.getOperands()[0]);
            int64_t remaining = headroom.getValue() - *amount;
            return {true, remaining == 0 ? base :
                expression(PhysicalExprKind::Subtract, 0,
                           {base, expression(PhysicalExprKind::Constant, remaining)}),
                    preceding.lower + *amount};
          }
        }
        if (subtract && subtract.getOperatorKind() == BinaryOperator::Subtract &&
            amount && *amount >= 0 &&
            integerConstant(subtract.getRhs()) == amount &&
            preceding.nonNegative) {
          Bounds original = bound(subtract.getLhs(), depth + 1);
          if (original.nonNegative)
            return original;
        }
      }
    }
    if (lhsConstant && rhsConstant) {
      constexpr int64_t maximum = std::numeric_limits<int64_t>::max();
      if (binary.getOperatorKind() == BinaryOperator::Add &&
          *lhsConstant <= maximum - *rhsConstant)
        return {true, expression(PhysicalExprKind::Constant,
                                 *lhsConstant + *rhsConstant), lhs.lower + rhs.lower};
      if (binary.getOperatorKind() == BinaryOperator::Multiply &&
          (*rhsConstant == 0 || *lhsConstant <= maximum / *rhsConstant))
        return {true, expression(PhysicalExprKind::Constant,
                *lhsConstant * *rhsConstant), lhs.lower * rhs.lower};
    }
    if (binary.getOperatorKind() == BinaryOperator::Add &&
        lhs.nonNegative && rhs.nonNegative && lhs.upper && rhs.upper) {
      PhysicalExprAttr upper = expression(
          PhysicalExprKind::Add, 0, {lhs.upper, rhs.upper});
      auto kernel = binary->getParentOfType<func::FuncOp>();
      if (nonNegativeExtentBounds(kernel, upper))
        return {true, upper, lhs.lower + rhs.lower};
    }
    if (binary.getOperatorKind() == BinaryOperator::Minimum) {
      PhysicalExprAttr upper = lhs.upper && rhs.upper
          ? expression(PhysicalExprKind::Minimum, 0, {lhs.upper, rhs.upper})
          : lhs.upper ? lhs.upper : rhs.upper;
      return {lhs.nonNegative && rhs.nonNegative, upper};
    }
    if (binary.getOperatorKind() == BinaryOperator::Maximum)
      return {lhs.nonNegative || rhs.nonNegative,
              lhs.upper && rhs.upper
                  ? expression(PhysicalExprKind::Maximum, 0, {lhs.upper, rhs.upper})
                  : PhysicalExprAttr()};
    auto positiveDivisor = [&](Value divisor) -> PhysicalExprAttr {
      if (auto constant = integerConstant(divisor))
        return *constant > 0
                   ? expression(PhysicalExprKind::Constant, *constant)
                   : PhysicalExprAttr();
      if (auto parameter = queryParameter(divisor))
        if (llvm::all_of(parameter.getCandidates().asArrayRef(),
                         [](int64_t candidate) { return candidate > 0; }))
          return bound(divisor, depth + 1).upper;
      return {};
    };
    if (binary.getOperatorKind() == BinaryOperator::FloorDivide ||
        binary.getOperatorKind() == BinaryOperator::RightShift) {
      PhysicalExprAttr divisor;
      if (binary.getOperatorKind() == BinaryOperator::FloorDivide) {
        divisor = positiveDivisor(binary.getRhs());
      } else if (auto shift = integerConstant(binary.getRhs());
                 shift && *shift >= 0 && *shift < 63) {
        // Power-of-two division canonicalization must preserve bounds used by
        // native access forms. For a nonnegative index, arithmetic right shift
        // has exactly the same inclusive upper bound as floor division.
        divisor = expression(PhysicalExprKind::Constant, int64_t{1} << *shift);
      }
      return {lhs.nonNegative && bool(divisor),
              lhs.upper && divisor
                  ? expression(PhysicalExprKind::FloorDiv, 0, {lhs.upper, divisor})
                  : PhysicalExprAttr()};
    }
    if (binary.getOperatorKind() == BinaryOperator::Multiply) {
      // Aligning a non-negative index down cannot overflow or exceed that index.
      // Other products require the explicit no-overflow proof above.
      for (auto [quotient, factor] :
           {std::pair{binary.getLhs(), binary.getRhs()},
            std::pair{binary.getRhs(), binary.getLhs()}}) {
        auto divide = quotient.getDefiningOp<BinaryOp>();
        if (divide && divide.getOperatorKind() == BinaryOperator::FloorDivide &&
            sameScalarExpression(divide.getRhs(), factor) && positiveDivisor(factor)) {
          Bounds dividend = bound(divide.getLhs(), depth + 1);
          if (dividend.nonNegative)
            return dividend;
        }
      }
    }
    return {};
  };
  return bound(value, 0);
}
} // namespace

PhysicalExprAttr queryNonNegativeIndexUpperBound(Value value) {
  IndexBounds result = queryIndexBounds(value);
  return result.nonNegative ? result.upper : PhysicalExprAttr();
}

PhysicalExprAttr queryLogicalRangeCapacity(MakeRangeOp range) {
  if (!isUnitStepRange(range))
    return {};
  IndexBounds lower = queryIndexBounds(range.getLogicalStart());
  if (lower.nonNegative)
    if (auto stop = queryNonNegativeIndexUpperBound(range.getLogicalStop())) {
      if (lower.lower == 0)
        return stop;
      auto make = [&](PhysicalExprKind kind, int64_t value,
                      ArrayRef<Attribute> operands = {}) {
        auto context = range.getContext();
        return PhysicalExprAttr::get(context, kind, value,
                                    StringAttr::get(context, ""),
                                    ArrayAttr::get(context, operands));
      };
      auto span = make(PhysicalExprKind::Subtract, 0,
                       {stop, make(PhysicalExprKind::Constant, lower.lower)});
      return make(PhysicalExprKind::Maximum, 0,
                  {span, make(PhysicalExprKind::Constant, 0)});
    }

  auto loopBound = [&](Value endpoint, bool upper) -> PhysicalExprAttr {
    Value coordinate = stripScalarIdentity(endpoint);
    if (auto add = coordinate.getDefiningOp<BinaryOp>();
        add && add.getOperatorKind() == BinaryOperator::Add) {
      if (integerConstant(add.getRhs()) == 1)
        coordinate = stripScalarIdentity(add.getLhs());
      else if (integerConstant(add.getLhs()) == 1)
        coordinate = stripScalarIdentity(add.getRhs());
    }
    auto induction = dyn_cast<BlockArgument>(coordinate);
    auto loop = induction
                    ? dyn_cast<scf::ForOp>(induction.getOwner()->getParentOp())
                    : scf::ForOp();
    if (!loop || induction != loop.getInductionVar() ||
        !loop->isAncestor(range) || integerConstant(loop.getStep()) != 1)
      return {};
    // Both iv and iv+1 lie within [lower, upper] in an executing unit-step loop.
    // The successor cannot overflow because iv < upper <= INDEX_MAX.
    return queryLaunchExpression(upper ? loop.getUpperBound()
                                       : loop.getLowerBound());
  };
  auto stop = queryLaunchExpression(range.getLogicalStop());
  if (!stop)
    stop = loopBound(range.getLogicalStop(), /*upper=*/true);
  auto start = queryLaunchExpression(range.getLogicalStart());
  if (!start)
    start = loopBound(range.getLogicalStart(), /*upper=*/false);
  if (!start || !stop)
    return {};
  auto expression = [&](PhysicalExprKind kind, ArrayRef<Attribute> operands) {
    auto context = range.getContext();
    return PhysicalExprAttr::get(context, kind, 0,
                                 StringAttr::get(context, ""),
                                 ArrayAttr::get(context, operands));
  };
  // This is host capacity arithmetic, not a proof that a device subtraction
  // cannot wrap. Coverage selection rejects spans outside its finite domain.
  auto span = expression(PhysicalExprKind::Subtract, {stop, start});
  auto zero = expression(PhysicalExprKind::Constant, {});
  return expression(PhysicalExprKind::Maximum, {span, zero});
}

bool sameLogicalRange(MakeRangeOp lhs, MakeRangeOp rhs) {
  if (!lhs || !rhs || lhs.getSourceId() != rhs.getSourceId() ||
      lhs.getSourceAxis() != rhs.getSourceAxis() ||
      lhs.getDerived() != rhs.getDerived())
    return false;
  return samePhysicalScalarExpression(lhs.getLogicalStart(), rhs.getLogicalStart()) &&
         samePhysicalScalarExpression(lhs.getLogicalStop(), rhs.getLogicalStop()) &&
         samePhysicalScalarExpression(lhs.getStep(), rhs.getStep());
}

bool canReplayReadAt(LoadOp load, Operation *insertionAnchor) {
  if (!load || !insertionAnchor)
    return false;
  auto preservesRead = [&](Operation *operation) {
    return !operation->walk([&](Operation *nested) {
      if (isa<LoadOp, GatherOp, scf::ForOp, scf::IfOp, scf::WhileOp>(nested) ||
          isMemoryEffectFree(nested))
        return WalkResult::advance();
      Value written = accessResource(nested);
      if (written && written != load.getResource()) {
        auto readBuffer = dyn_cast<BufferType>(load.getResource().getType());
        auto writtenBuffer = dyn_cast<BufferType>(written.getType());
        auto readView = dyn_cast<ViewType>(load.getResource().getType());
        auto writtenView = dyn_cast<ViewType>(written.getType());
        bool privateAllocation =
            (readBuffer && isa<ViewType>(written.getType())) ||
            (writtenBuffer && isa<ViewType>(load.getResource().getType())) ||
            (readBuffer && writtenBuffer &&
             readBuffer.getInstance() != writtenBuffer.getInstance());
        auto publicRead = getPublicView(load.getResource());
        auto publicWrite = getPublicView(written);
        bool disjointViews = readView && writtenView &&
            (isInvocationWorkspace(load.getResource()) || isInvocationWorkspace(written) ||
             (publicRead && publicRead.getConstraints().getNoalias()) ||
             (publicWrite && publicWrite.getConstraints().getNoalias()));
        if ((privateAllocation || disjointViews) && isa<StoreOp>(nested))
          return WalkResult::advance();
      }
      return WalkResult::interrupt();
    }).wasInterrupted();
  };
  // This motion stays in the same invocation of the enclosing block, including
  // the same iteration when the block belongs to an ordered loop.
  if (insertionAnchor->getBlock() == load->getBlock() &&
      insertionAnchor->isBeforeInBlock(load)) {
    for (Operation *operation = insertionAnchor; operation != load;
         operation = operation->getNextNode())
      if (!preservesRead(operation))
        return false;
    return true;
  }
  // A branch only needs its executed prefix; entering a loop also exposes
  // this read to writes from earlier iterations.
  Operation *ancestor = insertionAnchor;
  while (ancestor && ancestor->getBlock() != load->getBlock()) {
    Operation *parent = ancestor->getParentOp();
    if (!isa_and_nonnull<scf::IfOp, scf::ForOp, scf::WhileOp>(parent))
      return false;
    if (!isa<scf::IfOp>(parent) && !preservesRead(parent))
      return false;
    for (Operation &preceding : *ancestor->getBlock()) {
      if (&preceding == ancestor)
        break;
      if (!preservesRead(&preceding))
        return false;
    }
    ancestor = parent;
  }
  if (!ancestor || ancestor == load ||
      !load->isBeforeInBlock(ancestor))
    return false;
  for (Operation *next = load->getNextNode(); next != ancestor;
       next = next->getNextNode())
    if (!preservesRead(next))
      return false;
  return true;
}

bool isUnitStepRange(MakeRangeOp range) {
  return range && isUnitStepValue(range.getStep());
}

FailureOr<int64_t> querySubregionParentDimension(MakeRangeOp range) {
  if (!range)
    return failure();
  auto parent = range->getAttrOfType<IntegerAttr>(sourceSubregionAttr);
  return parent && parent.getInt() > 0
             ? FailureOr<int64_t>(parent.getInt())
             : FailureOr<int64_t>(failure());
}

FailureOr<MakeRangeOp>
queryExactLogicalRange(const PhysicalRangeFact &fact) {
  if (fact.state == PhysicalFactState::Unknown || fact.roots.empty())
    return failure();
  MakeRangeOp first = fact.roots.front();
  return llvm::all_of(fact.roots, [&](MakeRangeOp range) {
           return sameLogicalRange(first, range);
         })
             ? FailureOr<MakeRangeOp>(first)
             : FailureOr<MakeRangeOp>(failure());
}

SmallVector<PhysicalAxisProjection, 2>
queryFragmentAxes(Type type, PhysicalSourceAxis source) {
  SmallVector<PhysicalAxisProjection, 2> results;
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return results;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getSourceId() != source.sourceId ||
        mapping.getSourceAxis() != source.sourceAxis ||
        mapping.getDerived() != source.derived)
      continue;
    results.push_back(PhysicalAxisProjection{
        PhysicalFactState::Exact, source, mapping.getDimensionId(),
        mapping.getFragmentAxis()});
  }
  return results;
}

SmallVector<PhysicalAxisProjection, 2>
queryRangeProjections(Type type, MakeRangeOp range) {
  SmallVector<PhysicalAxisProjection, 2> sourceResults =
      queryFragmentAxes(type, sourceAxisIdentity(range));
  FailureOr<int64_t> dimension = queryRangeDimension(range);
  if (succeeded(dimension)) {
    SmallVector<PhysicalAxisProjection, 2> dimensionResults = sourceResults;
    llvm::erase_if(dimensionResults,
                   [&](const PhysicalAxisProjection &projection) {
      return projection.dimensionId != *dimension;
    });
    if (!dimensionResults.empty())
      return dimensionResults;
    PhysicalDimensionProjection projection =
        queryFragmentDimension(type, *dimension);
    if (projection.isExact())
      return {PhysicalAxisProjection{
          PhysicalFactState::Exact, sourceAxisIdentity(range), *dimension,
          projection.fragmentAxis}};
  }
  return sourceResults.size() == 1
             ? sourceResults
             : SmallVector<PhysicalAxisProjection, 2>{};
}

PhysicalDimensionProjection queryFragmentDimension(Type type,
                                                   int64_t dimensionId) {
  PhysicalDimensionProjection result;
  result.dimensionId = dimensionId;
  if (dimensionId <= 0)
    return result;
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return result;
  std::optional<unsigned> axis;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getDimensionId() != dimensionId)
      continue;
    if (axis && *axis != mapping.getFragmentAxis()) {
      result.state = PhysicalFactState::Ambiguous;
      return result;
    }
    axis = mapping.getFragmentAxis();
  }
  if (!axis)
    return result;
  result.state = PhysicalFactState::Exact;
  result.fragmentAxis = *axis;
  return result;
}

SmallVector<PhysicalDimensionProjection, 2>
queryFragmentDimensions(Type type, int64_t dimensionId) {
  SmallVector<PhysicalDimensionProjection, 2> results;
  if (dimensionId <= 0)
    return results;
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return results;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getDimensionId() != dimensionId)
      continue;
    results.push_back(PhysicalDimensionProjection{
        PhysicalFactState::Exact, dimensionId, mapping.getFragmentAxis()});
  }
  return results;
}

FailureOr<int64_t> querySourceDimension(Type type, PhysicalSourceAxis source) {
  if (auto range = dyn_cast<RangeType>(type))
    return range.getSourceId() == source.sourceId &&
                   range.getSourceAxis() == source.sourceAxis &&
                   range.getDerived() == source.derived &&
                   range.getDimensionId() > 0
               ? FailureOr<int64_t>(range.getDimensionId())
               : FailureOr<int64_t>(failure());
  PhysicalAxisProjection projection = queryFragmentAxis(type, source);
  return projection.isExact() && projection.dimensionId > 0
             ? FailureOr<int64_t>(projection.dimensionId)
             : FailureOr<int64_t>(failure());
}

PhysicalAxisProjection
queryCoordinateIndex(ValueRange coordinates, PhysicalSourceAxis source,
                     std::optional<int64_t> dimension) {
  PhysicalAxisProjection result;
  result.source = source;
  std::optional<unsigned> coordinateIndex;
  for (auto [index, coordinate] : llvm::enumerate(coordinates)) {
    PhysicalAxisProjection projection =
        queryFragmentAxis(coordinate.getType(), source, dimension);
    if (projection.state == PhysicalFactState::Ambiguous) {
      result.state = PhysicalFactState::Ambiguous;
      return result;
    }
    if (!projection.isExact())
      continue;
    if (coordinateIndex) {
      result.state = PhysicalFactState::Ambiguous;
      return result;
    }
    coordinateIndex = index;
  }
  if (!coordinateIndex)
    return result;
  result.state = PhysicalFactState::Exact;
  PhysicalAxisProjection projection =
      queryFragmentAxis(coordinates[*coordinateIndex].getType(), source, dimension);
  result.dimensionId = projection.dimensionId;
  result.fragmentAxis = *coordinateIndex;
  return result;
}

FailureOr<unsigned> queryCoordinatePosition(ValueRange coordinates,
                                            PhysicalSourceAxis source) {
  PhysicalAxisProjection result = queryCoordinateIndex(coordinates, source);
  return result.isExact() ? FailureOr<unsigned>(result.fragmentAxis)
                          : FailureOr<unsigned>(failure());
}

PhysicalProgramAnalysis::PhysicalProgramAnalysis(func::FuncOp kernel)
    : kernel(kernel) {}

FailureOr<unsigned>
PhysicalProgramAnalysis::fragmentAxis(Type type,
                                      PhysicalSourceAxis source) const {
  PhysicalAxisProjection result = queryFragmentAxis(type, source);
  return result.isExact() ? FailureOr<unsigned>(result.fragmentAxis)
                          : FailureOr<unsigned>(failure());
}

FailureOr<unsigned> PhysicalProgramAnalysis::accessCoordinatePosition(
    LoadOp load, AxisMapAttr mapping, Value operand) {
  if (PhysicalAxisProjection direct = queryCoordinateIndex(
          load.getCoordinates(), sourceAxisIdentity(mapping),
          mapping.getDimensionId());
      direct.isExact())
    return direct.fragmentAxis;
  FailureOr<MakeRangeOp> selected = queryExactLogicalRange(
      axisRanges(operand, mapping.getFragmentAxis()));
  std::optional<unsigned> replayed;
  for (auto [position, coordinate] : llvm::enumerate(load.getCoordinates())) {
    PhysicalAxisProjection axis = queryFragmentAxis(
        coordinate.getType(), sourceAxisIdentity(mapping),
        mapping.getDimensionId());
    FailureOr<MakeRangeOp> coordinateRange = queryExactLogicalRange(
        axis.isExact() ? axisRanges(coordinate, axis.fragmentAxis)
                       : sourceRanges(coordinate, sourceAxisIdentity(mapping)));
    if (failed(coordinateRange) ||
        (succeeded(selected) && !sameLogicalRange(*selected, *coordinateRange)))
      continue;
    if (replayed)
      return failure();
    replayed = position;
  }
  if (replayed)
    return *replayed;
  auto result = dyn_cast<FragmentType>(load.getResult().getType());
  if (result && result.getShape().size() == load.getCoordinates().size() &&
      llvm::all_of(load.getCoordinates(), [](Value coordinate) {
        auto fragment = dyn_cast<FragmentType>(coordinate.getType());
        return fragment && fragment.getShape().size() == 1;
      }) &&
      mapping.getFragmentAxis() < load.getCoordinates().size())
    return mapping.getFragmentAxis();
  auto view = dyn_cast<ViewType>(load.getResource().getType());
  if (!view || mapping.getDimensionId() <= 0)
    return failure();
  std::optional<unsigned> resourceAxis;
  for (auto [axis, dimension] :
       llvm::enumerate(view.getLayout().getDimensionIds().asArrayRef())) {
    if (dimension != mapping.getDimensionId())
      continue;
    if (resourceAxis)
      return failure();
    resourceAxis = axis;
  }
  if (!resourceAxis)
    return failure();
  std::optional<unsigned> coordinate;
  for (auto [position, axis] : llvm::enumerate(load.getSourceAxes())) {
    if (axis != *resourceAxis)
      continue;
    if (coordinate)
      return failure();
    coordinate = position;
  }
  return coordinate ? FailureOr<unsigned>(*coordinate)
                    : FailureOr<unsigned>(failure());
}

FailureOr<unsigned> PhysicalProgramAnalysis::coordinateIndex(
    ValueRange coordinates, PhysicalSourceAxis source) const {
  PhysicalAxisProjection result = queryCoordinateIndex(coordinates, source);
  return result.isExact() ? FailureOr<unsigned>(result.fragmentAxis)
                          : FailureOr<unsigned>(failure());
}

bool PhysicalProgramAnalysis::carriesSource(Type type,
                                            PhysicalSourceAxis source) const {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
      auto mapping = cast<AxisMapAttr>(attribute);
      return mapping.getSourceId() == source.sourceId &&
             mapping.getSourceAxis() == source.sourceAxis &&
             mapping.getDerived() == source.derived;
    });
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
      return carriesSource(cast<TypeAttr>(field).getValue(), source);
    });
  return false;
}

SmallVector<Value, 2> PhysicalProgramAnalysis::structuredSourcesForArgument(
    BlockArgument argument) const {
  SmallVector<Value, 2> sources;
  Block *block = argument.getOwner();
  Operation *owner = block ? block->getParentOp() : nullptr;
  if (auto structured = dyn_cast_or_null<StructuredOpInterface>(owner)) {
    bool fold = structured.getStructuredKind() == StructuredOpKind::RegionFold;
    bool scan = structured.getStructuredKind() == StructuredOpKind::RegionScan;
    if (!fold && !scan) return sources;
    for (const auto &relation : structured.getValueRelations()) {
      if (relation.to != argument) continue;
      if (block->getParent() == structured.getSummarizeRegion() &&
          (relation.kind == StructuredRelationKind::SourceSlice ||
           (fold && relation.kind == StructuredRelationKind::Capture)))
        sources.push_back(relation.from);
    }
    if (fold && block->getParent() == &structured.getCombine()) {
      for (auto [identity, lhs, rhs, summary] : llvm::zip_equal(
               structured.getIdentities(), structured.getCombineLhs(),
               structured.getCombineRhs(), structured.getSummarizeYields())) {
        if (argument != lhs && argument != rhs) continue;
        sources.push_back(identity);
        sources.push_back(summary);
      }
    }
    return sources;
  }
  if (auto loop = dyn_cast_or_null<scf::ForOp>(owner)) {
    if (argument == loop.getInductionVar())
      return sources;
    unsigned offset = argument.getArgNumber() - 1;
    if (offset >= loop.getInitArgs().size())
      return sources;
    sources.push_back(loop.getInitArgs()[offset]);
    if (auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
        yield && offset < yield.getResults().size())
      sources.push_back(yield.getResults()[offset]);
  }
  return sources;
}

void PhysicalProgramAnalysis::collectRanges(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalRangeFact &result,
    SmallPtrSetImpl<Operation *> &visited, bool followScalarDependencies) {
  if (!value)
    return;
  if (!followScalarDependencies) {
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (isa<IntegerType, FloatType, IndexType>(value.getType()) ||
        (fragment && fragment.getShape().empty()))
      return;
  }
  if (auto extract = value.getDefiningOp<ExtractOp>()) {
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
      uint64_t field = extract.getField();
      if (field >= record.getFields().size()) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, extract);
        return;
      }
      collectRanges(record.getFields()[field], source, result, visited,
                    followScalarDependencies);
      return;
    }
  }
  if (source && !carriesSource(value.getType(), *source))
    return;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    // Scalar ABI/control coordinates do not introduce a fragment range.
    if (isa<IntegerType, FloatType, IndexType>(argument.getType()))
      return;
    SmallVector<Value, 2> outer = structuredSourcesForArgument(argument);
    if (outer.empty()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, argument.getOwner()->getParentOp());
      return;
    }
    for (Value related : outer)
      collectRanges(related, source, result, visited, followScalarDependencies);
    return;
  }
  Operation *operation = value.getDefiningOp();
  if (!operation || !visited.insert(operation).second)
    return;
  if (auto range = dyn_cast<MakeRangeOp>(operation)) {
    if (!source || sourceAxisIdentity(range) == *source)
      appendUnique(result.roots, range);
    return;
  }
  if (isa<scf::ForOp>(operation)) {
    if (auto fragment = dyn_cast<FragmentType>(value.getType())) {
      // A completed loop contributes its result lanes to an indirect access,
      // not the reduction lanes used to compute each index. Follow the typed
      // yield/carry relation already used by axis-specific provenance.
      for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps())) {
        if (source &&
            !(sourceAxisIdentity(cast<AxisMapAttr>(attribute)) == *source))
          continue;
        PhysicalRangeFact fact = axisRanges(value, axis);
        if (result.state != PhysicalFactState::Unknown &&
            fact.state != PhysicalFactState::Exact)
          result.state = fact.state;
        for (MakeRangeOp range : fact.roots)
          appendUnique(result.roots, range);
        for (Operation *access : fact.accesses)
          appendUnique(result.accesses, access);
        for (Operation *blocker : fact.blockers)
          appendUnique(result.blockers, blocker);
      }
      return;
    }
  }
  // Reshape preserves the row-major coordinate relation carried by its
  // reassociation groups.  Source-specific range queries follow that typed
  // relation through the input instead of treating reshape as an opaque value
  // producer.  Any genuinely incompatible roots remain ambiguous in
  // sourceRanges(), where all collected ranges are compared.
  if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
    collectRanges(reshape.getValue(), source, result, visited,
                  followScalarDependencies);
    return;
  }
  // These operations are typed coordinate leaves.  They do not contribute a
  // fragment range root, but reaching one is an exact end of provenance rather
  // than an unknown operation in the producer graph.
  if (isa<arith::ConstantOp, PhysicalExprOp, ParameterOp, ProgramIdOp, DelinearizeOp,
          WorksetCoordinateOp, DimOp, RangeOp, RangeBoundOp>(operation))
    return;
  if (auto scan = dyn_cast<ScanOp>(operation)) {
    bool followed = false;
    for (Value scanSource : scan.getSources()) {
      if (source && !carriesSource(scanSource.getType(), *source))
        continue;
      followed = true;
      collectRanges(scanSource, source, result, visited, followScalarDependencies);
    }
    if (!followed) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
    }
    return;
  }
  if (isAccessNode(operation))
    appendUnique(result.accesses, operation);
  if (!isValueReplayNode(operation) && !isAccessNode(operation)) {
    appendUnique(result.blockers, operation);
    result.state = PhysicalFactState::Unknown;
    return;
  }
  auto load = dyn_cast<LoadOp>(operation);
  for (Value operand : operation->getOperands()) {
    // Resource identity stays in the access fact, not in fragment range provenance.
    if (load && operand == load.getResource())
      continue;
    collectRanges(operand, source, result, visited, followScalarDependencies);
  }
}

PhysicalRangeFact PhysicalProgramAnalysis::sourceRanges(
    Value value, std::optional<PhysicalSourceAxis> source) {
  if (!source) {
    if (auto cached = unrestrictedRangeCache.find(value);
        cached != unrestrictedRangeCache.end())
      return cached->second;
  }
  PhysicalRangeFact result;
  result.state = PhysicalFactState::Exact;
  SmallPtrSet<Operation *, 32> visited;
  collectRanges(value, source, result, visited);
  if (result.state == PhysicalFactState::Unknown || !result.blockers.empty() ||
      result.roots.empty())
    result.state = PhysicalFactState::Unknown;
  else if (result.roots.size() > 1) {
    MakeRangeOp authority = result.roots.front();
    if (!llvm::all_of(result.roots, [&](MakeRangeOp range) {
          return sameLogicalRange(authority, range);
        }))
      result.state = PhysicalFactState::Ambiguous;
  }
  result.unitStep = !result.roots.empty() &&
                    llvm::all_of(result.roots, isUnitStepRange);
  if (!source)
    unrestrictedRangeCache.try_emplace(value, result);
  return result;
}

PhysicalRangeFact
PhysicalProgramAnalysis::programRanges(PhysicalSourceAxis source) {
  PhysicalRangeFact result;
  result.state = PhysicalFactState::Exact;
  kernel.walk([&](MakeRangeOp range) {
    if (!(sourceAxisIdentity(range) == source))
      return;
    appendUnique(result.roots, range);
  });
  if (result.roots.empty()) {
    result.state = PhysicalFactState::Unknown;
    return result;
  }
  MakeRangeOp authority = result.roots.front();
  if (!llvm::all_of(result.roots, [&](MakeRangeOp range) {
        return sameLogicalRange(authority, range);
      }))
    result.state = PhysicalFactState::Ambiguous;
  result.unitStep = llvm::all_of(result.roots, isUnitStepRange);
  return result;
}

void PhysicalProgramAnalysis::collectAxisRanges(
    Value value, unsigned fragmentAxis, PhysicalRangeFact &result,
    llvm::DenseSet<std::pair<Value, unsigned>> &visited) {
  if (!value || !visited.insert({value, fragmentAxis}).second)
    return;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || fragmentAxis >= fragment.getShape().size()) {
    result.state = PhysicalFactState::Unknown;
    return;
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (auto reduce = dyn_cast<ReduceOp>(argument.getOwner()->getParentOp())) {
      auto structured = cast<StructuredOpInterface>(reduce.getOperation());
      for (auto [capture, formal] : llvm::zip_equal(reduce.getCaptures(), structured.getCombineCaptures()))
        if (formal == argument) {
          collectAxisRanges(capture, fragmentAxis, result, visited);
          return;
        }
      Value source;
      for (auto [input, lhs, rhs] : llvm::zip_equal(reduce.getSources(),
               structured.getCombineLhs(), structured.getCombineRhs()))
        if (argument == lhs || argument == rhs) source = input;
      if (!source) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, reduce);
        return;
      }
      auto sourceType = dyn_cast<FragmentType>(source.getType());
      if (!sourceType) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, reduce);
        return;
      }
      SmallVector<unsigned> freeAxes;
      for (unsigned axis = 0; axis < sourceType.getShape().size(); ++axis)
        if (!llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
          freeAxes.push_back(axis);
      if (fragmentAxis >= freeAxes.size()) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, reduce);
        return;
      }
      collectAxisRanges(source, freeAxes[fragmentAxis], result, visited);
      return;
    }
    SmallVector<Value, 2> outer = structuredSourcesForArgument(argument);
    bool followed = false;
    for (Value related : outer) {
      auto outerFragment = dyn_cast<FragmentType>(related.getType());
      if (!outerFragment || fragmentAxis >= outerFragment.getShape().size())
        continue;
      followed = true;
      collectAxisRanges(related, fragmentAxis, result, visited);
    }
    if (!followed) {
      result.state = PhysicalFactState::Unknown;
      return;
    }
    return;
  }
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return;
  if (auto range = dyn_cast<MakeRangeOp>(operation)) {
    if (fragmentAxis == 0)
      appendUnique(result.roots, range);
    else
      result.state = PhysicalFactState::Unknown;
    return;
  }
  if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
    auto input = dyn_cast<FragmentType>(reshape.getValue().getType());
    if (!input) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    unsigned logicalSourceRank = 0;
    unsigned logicalResultRank = 0;
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = dyn_cast<ReshapeGroupAttr>(attribute);
      if (!group) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, operation);
        return;
      }
      if (!group.getSourceAxes().empty())
        logicalSourceRank =
            std::max(logicalSourceRank,
                     static_cast<unsigned>(
                         group.getSourceAxes().asArrayRef().back() + 1));
      if (!group.getResultAxes().empty())
        logicalResultRank =
            std::max(logicalResultRank,
                     static_cast<unsigned>(
                         group.getResultAxes().asArrayRef().back() + 1));
    }
    if (logicalSourceRank > input.getShape().size() ||
        logicalResultRank > fragment.getShape().size()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    unsigned sourcePrefix = input.getShape().size() - logicalSourceRank;
    unsigned resultPrefix = fragment.getShape().size() - logicalResultRank;
    if (fragmentAxis < resultPrefix) {
      if (sourcePrefix != resultPrefix || fragmentAxis >= sourcePrefix) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, operation);
        return;
      }
      collectAxisRanges(reshape.getValue(), fragmentAxis, result, visited);
      return;
    }
    unsigned logicalResultAxis = fragmentAxis - resultPrefix;
    auto expected =
        cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
    std::optional<unsigned> sourceAxis;
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = cast<ReshapeGroupAttr>(attribute);
      if (!llvm::is_contained(group.getResultAxes().asArrayRef(),
                              logicalResultAxis))
        continue;
      if (group.getSourceAxes().size() == 1 &&
          group.getResultAxes().size() == 1) {
        sourceAxis = sourcePrefix + group.getSourceAxes()[0];
        break;
      }
      for (int64_t logicalSourceAxis : group.getSourceAxes().asArrayRef()) {
        if (logicalSourceAxis < 0 ||
            sourcePrefix + static_cast<unsigned>(logicalSourceAxis) >=
                input.getShape().size())
          continue;
        unsigned physicalSourceAxis = sourcePrefix + logicalSourceAxis;
        auto mapping =
            cast<AxisMapAttr>(input.getAxisMaps()[physicalSourceAxis]);
        if (mapping.getDimensionId() != expected.getDimensionId())
          continue;
        if (sourceAxis) {
          result.state = PhysicalFactState::Ambiguous;
          appendUnique(result.blockers, operation);
          return;
        }
        sourceAxis = physicalSourceAxis;
      }
      break;
    }
    if (sourceAxis) {
      collectAxisRanges(reshape.getValue(), *sourceAxis, result, visited);
      return;
    }
    auto extent =
        cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
    if (extent.getKind() ==
            PhysicalExprKind::Constant &&
        extent.getValue() == 1)
      return;
    result.state = PhysicalFactState::Unknown;
    appendUnique(result.blockers, operation);
    return;
  }
  if (auto scan = dyn_cast<ScanOp>(operation)) {
    auto expected =
        cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
    bool followed = false;
    for (Value scanSource : scan.getSources()) {
      auto sourceType = dyn_cast<FragmentType>(scanSource.getType());
      if (!sourceType || fragmentAxis >= sourceType.getShape().size())
        continue;
      auto mapping =
          cast<AxisMapAttr>(sourceType.getAxisMaps()[fragmentAxis]);
      if (!(sourceAxisIdentity(mapping) == sourceAxisIdentity(expected)) ||
          mapping.getDimensionId() != expected.getDimensionId())
        continue;
      followed = true;
      collectAxisRanges(scanSource, fragmentAxis, result, visited);
    }
    if (!followed) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
    }
    return;
  }
  if (auto join = dyn_cast<JoinOp>(operation)) {
    auto input = cast<FragmentType>(join.getLhs().getType());
    if (fragmentAxis < input.getShape().size()) {
      collectAxisRanges(join.getLhs(), fragmentAxis, result, visited);
      collectAxisRanges(join.getRhs(), fragmentAxis, result, visited);
    }
    return;
  }
  if (auto extract = dyn_cast<ExtractOp>(operation)) {
    bool followed = false;
    llvm::SmallPtrSet<Value, 8> visitedRecords;
    std::function<void(Value)> collectRecordField = [&](Value recordValue) {
      if (!recordValue || !visitedRecords.insert(recordValue).second)
        return;
      if (auto record = recordValue.getDefiningOp<MakeRecordOp>()) {
        if (extract.getField() >= record.getFields().size())
          return;
        Value field = record.getFields()[extract.getField()];
        auto type = dyn_cast<FragmentType>(field.getType());
        if (!type || fragmentAxis >= type.getShape().size())
          return;
        followed = true;
        collectAxisRanges(field, fragmentAxis, result, visited);
        return;
      }
      if (auto argument = dyn_cast<BlockArgument>(recordValue)) {
        for (Value related : structuredSourcesForArgument(argument))
          collectRecordField(related);
        return;
      }
      auto opResult = dyn_cast<OpResult>(recordValue);
      if (!opResult)
        return;
      if (auto fold = dyn_cast<RegionFoldOp>(opResult.getOwner())) {
        unsigned component = opResult.getResultNumber();
        if (component >= fold.getIdentities().size())
          return;
        collectRecordField(fold.getIdentities()[component]);
        if (auto yield =
                dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
            yield && component < yield.getValues().size())
          collectRecordField(yield.getValues()[component]);
        return;
      }
      if (auto loop = dyn_cast<scf::ForOp>(opResult.getOwner())) {
        unsigned component = opResult.getResultNumber();
        if (component < loop.getInitArgs().size())
          collectRecordField(loop.getInitArgs()[component]);
        if (auto yield =
                dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
            yield && component < yield.getResults().size())
          collectRecordField(yield.getResults()[component]);
      }
    };
    collectRecordField(extract.getRecord());
    if (!followed)
      result.state = PhysicalFactState::Unknown;
    return;
  }
  if (auto splat = dyn_cast<SplatOp>(operation))
    return;
  if (auto broadcast = dyn_cast<BroadcastOp>(operation)) {
    auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
    if (!input)
      return;
    BroadcastProjection projection = queryAxisProjection(input, fragment);
    if (!projection.isExact()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    std::optional<unsigned> inputAxis = projection.targetToSource[fragmentAxis];
    if (!inputAxis)
      return;
    auto inputExtent = cast<PhysicalExprAttr>(input.getShape()[*inputAxis]);
    auto outputExtent = cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
    if (inputExtent.getKind() ==
            PhysicalExprKind::Constant &&
        inputExtent.getValue() == 1 && inputExtent != outputExtent) {
      PhysicalRangeFact inputRanges;
      inputRanges.state = PhysicalFactState::Exact;
      collectAxisRanges(broadcast.getValue(), *inputAxis, inputRanges, visited);
      if (inputRanges.state != PhysicalFactState::Unknown &&
          inputRanges.blockers.empty() &&
          llvm::all_of(inputRanges.roots, isProvablySingletonLogicalRange))
        return;
      if (result.state != PhysicalFactState::Unknown &&
          inputRanges.state != PhysicalFactState::Exact)
        result.state = inputRanges.state;
      for (MakeRangeOp root : inputRanges.roots)
        appendUnique(result.roots, root);
      for (Operation *access : inputRanges.accesses)
        appendUnique(result.accesses, access);
      for (Operation *blocker : inputRanges.blockers)
        appendUnique(result.blockers, blocker);
      return;
    }
    collectAxisRanges(broadcast.getValue(), *inputAxis, result, visited);
    return;
  }
  if (auto transpose = dyn_cast<TransposeOp>(operation)) {
    ArrayRef<int64_t> permutation = transpose.getPermutation();
    if (fragmentAxis >= permutation.size() || permutation[fragmentAxis] < 0) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    collectAxisRanges(transpose.getValue(),
                      static_cast<unsigned>(permutation[fragmentAxis]), result,
                      visited);
    return;
  }
  if (auto gather = dyn_cast<GatherOp>(operation)) {
    auto input = dyn_cast<FragmentType>(gather.getSource().getType());
    auto expected = cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
    auto occurrence = [&](FragmentType type) {
      return queryFragmentAxis(type, sourceAxisIdentity(expected),
                               expected.getDimensionId());
    };
    auto retained = input ? occurrence(input) : PhysicalAxisProjection{};
    auto output = occurrence(fragment);
    if (retained.isExact() && retained.dimensionId == expected.getDimensionId() &&
        output.isExact() && output.fragmentAxis == fragmentAxis &&
        !llvm::is_contained(gather.getSourceAxes(), retained.fragmentAxis) &&
        input.getShape()[retained.fragmentAxis] == fragment.getShape()[fragmentAxis]) {
      collectAxisRanges(gather.getSource(), retained.fragmentAxis, result, visited);
      return;
    }
    bool followed = false;
    for (Value coordinate : gather.getCoordinates()) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      auto projection = type ? occurrence(type)
                             : PhysicalAxisProjection{};
      if (!projection.isExact() || projection.dimensionId != expected.getDimensionId() ||
          type.getShape()[projection.fragmentAxis] != fragment.getShape()[fragmentAxis])
        continue;
      collectAxisRanges(coordinate, projection.fragmentAxis, result, visited);
      followed = true;
    }
    if (followed)
      return;
    result.state = PhysicalFactState::Unknown;
    appendUnique(result.blockers, operation);
    return;
  }
  if (auto load = dyn_cast<LoadOp>(operation)) {
    appendUnique(result.accesses, operation);
    auto expected = cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
    SmallVector<Value> positionalCoordinates;
    bool rankOneCoordinates = true;
    for (Value coordinate : load.getCoordinates()) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      if (!type)
        continue;
      rankOneCoordinates &= type.getShape().size() == 1;
      positionalCoordinates.push_back(coordinate);
    }
    bool cartesian = rankOneCoordinates &&
        positionalCoordinates.size() == fragment.getShape().size();
    if (cartesian) {
      Value coordinate = positionalCoordinates[fragmentAxis];
      auto type = cast<FragmentType>(coordinate.getType());
      auto occurrence = cast<AxisMapAttr>(type.getAxisMaps()[0]);
      if (sourceAxisIdentity(occurrence) == sourceAxisIdentity(expected) &&
          occurrence.getDimensionId() == expected.getDimensionId()) {
        collectAxisRanges(coordinate, 0, result, visited);
        return;
      }
    }
    using CoordinateOccurrence = std::pair<Value, unsigned>;
    enum class OccurrencePriority {
      SourceAndDimension,
      Dimension,
      UniqueSource,
    };
    auto selectOccurrence = [&](OccurrencePriority priority)
        -> SmallVector<CoordinateOccurrence, 2> {
      SmallVector<CoordinateOccurrence, 2> occurrences;
      for (Value coordinate : load.getCoordinates()) {
        auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
        if (!coordinateType)
          continue;
        SmallVector<unsigned, 2> axes;
        if (priority == OccurrencePriority::Dimension) {
          for (PhysicalDimensionProjection projection : queryFragmentDimensions(
                   coordinateType, expected.getDimensionId()))
            axes.push_back(projection.fragmentAxis);
        } else {
          SmallVector<PhysicalAxisProjection, 2> projections =
              queryFragmentAxes(coordinateType, sourceAxisIdentity(expected));
          if (priority == OccurrencePriority::SourceAndDimension)
            llvm::erase_if(projections,
                           [&](const PhysicalAxisProjection &projection) {
              return projection.dimensionId != expected.getDimensionId();
            });
          else if (projections.size() != 1)
            projections.clear();
          for (PhysicalAxisProjection projection : projections)
            axes.push_back(projection.fragmentAxis);
        }
        SmallVector<unsigned, 2> sameOccurrence;
        llvm::copy_if(axes, std::back_inserter(sameOccurrence),
                      [&](unsigned axis) { return axis == fragmentAxis; });
        if (sameOccurrence.size() == 1)
          axes = std::move(sameOccurrence);
        for (unsigned axis : axes)
          occurrences.emplace_back(coordinate, axis);
      }
      return occurrences;
    };
    SmallVector<CoordinateOccurrence, 2> occurrences =
        selectOccurrence(OccurrencePriority::SourceAndDimension);
    if (occurrences.empty())
      occurrences = selectOccurrence(OccurrencePriority::Dimension);
    if (occurrences.empty())
      occurrences = selectOccurrence(OccurrencePriority::UniqueSource);
    if (occurrences.empty()) {
      auto extent = cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
      bool broadcastAxis =
          extent.getKind() == PhysicalExprKind::Constant &&
          extent.getValue() == 1 &&
          llvm::all_of(load.getCoordinates(), [&](Value coordinate) {
            auto type = dyn_cast<FragmentType>(coordinate.getType());
            if (!type)
              return true;
            BroadcastProjection projection = queryAxisProjection(type, fragment);
            return projection.isExact() &&
                   !projection.targetToSource[fragmentAxis];
          });
      if (broadcastAxis) {
        // Address coordinates do not vary along this introduced unit axis.
        // Validity or fill may still carry a traversal, so retain their roots
        // instead of mistaking an unexpanded range for a uniform value.
        for (Value dependency : {load.getValid(), load.getFill()}) {
          auto type = dependency ? dyn_cast<FragmentType>(dependency.getType())
                                 : FragmentType();
          if (!type)
            continue;
          auto projection = queryAxisProjection(type, fragment);
          if (!projection.isExact()) {
            result.state = PhysicalFactState::Unknown;
            appendUnique(result.blockers, operation);
            continue;
          }
          if (auto axis = projection.targetToSource[fragmentAxis])
            collectAxisRanges(dependency, *axis, result, visited);
        }
        return;
      }
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    SmallVector<MakeRangeOp> accessRoots;
    for (const CoordinateOccurrence &occurrence : occurrences) {
      PhysicalRangeFact nested = axisRanges(occurrence.first, occurrence.second);
      if (nested.state == PhysicalFactState::Unknown ||
          !nested.blockers.empty()) {
        result.state = PhysicalFactState::Unknown;
        for (Operation *blocker : nested.blockers)
          appendUnique(result.blockers, blocker);
        continue;
      }
      for (MakeRangeOp range : nested.roots) {
        appendUnique(result.roots, range);
        appendUnique(accessRoots, range);
      }
      for (Operation *access : nested.accesses)
        appendUnique(result.accesses, access);
    }
    // Resolve this load's coordinate occurrence, not ranges accumulated from
    // sibling operands of a pointwise expression over different resources.
    if (result.state != PhysicalFactState::Unknown && !accessRoots.empty()) {
      MakeRangeOp authority = accessRoots.front();
      if (!llvm::all_of(accessRoots, [&](MakeRangeOp range) {
            return sameLogicalRange(authority, range);
          }) && (cartesian || !lockstepRanges(accessRoots).isExact())) {
        result.state = PhysicalFactState::Ambiguous;
        appendUnique(result.blockers, operation);
      }
    }
    return;
  }
  if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
    auto opResult = dyn_cast<OpResult>(value);
    auto yield = dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
    if (!opResult || !yield ||
        opResult.getResultNumber() >= fold.getIdentities().size() ||
        opResult.getResultNumber() >= yield.getValues().size()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    unsigned index = opResult.getResultNumber();
    collectAxisRanges(fold.getIdentities()[index],
                      fragmentAxis, result, visited);
    collectAxisRanges(yield.getValues()[index], fragmentAxis, result, visited);
    return;
  }
  if (auto reduce = dyn_cast<ReduceOp>(operation)) {
    auto opResult = dyn_cast<OpResult>(value);
    auto yield = dyn_cast<YieldOp>(reduce.getCombine().front().getTerminator());
    if (!opResult || !yield ||
        opResult.getResultNumber() >= yield.getValues().size()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    // Tuple results can depend on another component through the combine body
    // (for example an arg-reduce index selected by the value comparison).
    collectAxisRanges(yield.getValues()[opResult.getResultNumber()],
                      fragmentAxis, result, visited);
    return;
  }
  if (auto contract = dyn_cast<ContractOp>(operation)) {
    auto axes = queryContractionAxes(contract);
    if (!axes || fragmentAxis >= axes->results.size()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    auto source = axes->results[fragmentAxis];
    Value operand = source.operand == ContractionOperand::Lhs
                        ? Value(contract.getLhs()) : Value(contract.getRhs());
    collectAxisRanges(operand, source.axis, result, visited);
    if (source.operand == ContractionOperand::Lhs) {
      for (const auto &pair : axes->batch) {
        if (pair.lhs != source.axis) continue;
        // A batch result depends on both paired operands. The left operand
        // may broadcast along this axis and have no coordinate range at all.
        collectAxisRanges(contract.getRhs(), pair.rhs, result, visited);
        break;
      }
    }
    return;
  }
  if (auto branch = dyn_cast<scf::IfOp>(operation)) {
    auto opResult = dyn_cast<OpResult>(value);
    if (!opResult || branch.getElseRegion().empty()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    for (Region &region : branch->getRegions()) {
      auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
      if (!yield || opResult.getResultNumber() >= yield.getNumOperands()) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, operation);
        return;
      }
      collectAxisRanges(yield.getOperand(opResult.getResultNumber()),
                        fragmentAxis, result, visited);
    }
    return;
  }
  if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    auto opResult = dyn_cast<OpResult>(value);
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!opResult || !yield ||
        opResult.getResultNumber() >= yield.getResults().size() ||
        opResult.getResultNumber() >= loop.getInitArgs().size()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    auto lower = integerConstant(loop.getLowerBound());
    auto upper = integerConstant(loop.getUpperBound());
    if (!lower || !upper || *lower >= *upper)
      collectAxisRanges(loop.getInitArgs()[opResult.getResultNumber()],
                        fragmentAxis, result, visited);
    if (!lower || !upper || *lower < *upper)
      collectAxisRanges(yield.getResults()[opResult.getResultNumber()],
                        fragmentAxis, result, visited);
    return;
  }
  if (auto select = dyn_cast<SelectOp>(operation)) {
    // Even uniform alternatives vary by lane when the condition does.  Its
    // coordinate range must be replayed with the selected result axis.
    for (Value selected : {select.getCondition(), select.getTrueValue(),
                           select.getFalseValue()}) {
      auto selectedType = dyn_cast<FragmentType>(selected.getType());
      if (!selectedType ||
          selectedType.getShape().size() != fragment.getShape().size() ||
          fragmentAxis >= selectedType.getShape().size())
        continue;
      collectAxisRanges(selected, fragmentAxis, result, visited);
    }
    return;
  }
  if (!isCoordinateReplayNode(operation)) {
    result.state = PhysicalFactState::Unknown;
    appendUnique(result.blockers, operation);
    return;
  }
  bool followed = false;
  for (Value operand : operation->getOperands()) {
    auto operandType = dyn_cast<FragmentType>(operand.getType());
    if (!operandType || operandType.getShape().size() != fragment.getShape().size() ||
        fragmentAxis >= operandType.getShape().size())
      continue;
    followed = true;
    collectAxisRanges(operand, fragmentAxis, result, visited);
  }
  if (!followed && !operation->getOperands().empty()) {
    result.state = PhysicalFactState::Unknown;
    appendUnique(result.blockers, operation);
  }
}

PhysicalRangeFact PhysicalProgramAnalysis::axisRanges(Value value,
                                                      unsigned fragmentAxis) {
  PhysicalRangeFact result;
  result.state = PhysicalFactState::Exact;
  llvm::DenseSet<std::pair<Value, unsigned>> visited;
  collectAxisRanges(value, fragmentAxis, result, visited);
  if (result.state == PhysicalFactState::Unknown || !result.blockers.empty())
    result.state = PhysicalFactState::Unknown;
  else if (result.roots.size() > 1) {
    MakeRangeOp authority = result.roots.front();
    if (!llvm::all_of(result.roots, [&](MakeRangeOp range) {
          return sameLogicalRange(authority, range);
        }))
      result.state = PhysicalFactState::Ambiguous;
  }
  result.unitStep = !result.roots.empty() &&
                    llvm::all_of(result.roots, isUnitStepRange);
  return result;
}

PhysicalAxisRealizationFact
PhysicalProgramAnalysis::axisRealization(Value value, unsigned fragmentAxis) {
  PhysicalAxisRealizationFact result;
  result.fragmentAxis = fragmentAxis;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || fragmentAxis >= fragment.getShape().size() ||
      fragmentAxis >= fragment.getAxisMaps().size())
    return result;

  auto mapping = dyn_cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
  auto extent = dyn_cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
  if (!mapping || !extent)
    return result;
  result.source = sourceAxisIdentity(mapping);
  result.dimensionId = mapping.getDimensionId();

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *owner = argument.getOwner()->getParentOp();
    if (auto loop = dyn_cast_or_null<scf::ForOp>(owner);
        loop && argument.getOwner() == loop.getBody() &&
        argument.getArgNumber() > 0) {
      Value initial = loop.getInitArgs()[argument.getArgNumber() - 1];
      if (initial.getType() == fragment) {
        PhysicalAxisRealizationFact input =
            axisRealization(initial, fragmentAxis);
        if (input.hasExtentAuthority()) {
          result.state = PhysicalFactState::Exact;
          result.physicalized = input.physicalized;
          result.constructionScalarSeed = input.constructionScalarSeed;
          result.roots = input.roots;
          result.extentAuthority =
              PhysicalAxisRealizationFact::ExtentAuthority::Structural;
          return result;
        }
      }
    }
    auto isSegmentSource = [&](uint64_t sourceCount, uint64_t axis,
                               Block &region) {
      return argument.getOwner() == &region &&
             argument.getArgNumber() < sourceCount && fragmentAxis == axis;
    };
    bool segmentSource = false;
    if (auto fold = dyn_cast_or_null<RegionFoldOp>(owner))
      segmentSource = isSegmentSource(fold.getSources().size(), fold.getAxis(),
                                      fold.getSummarize().front());
    else if (auto scan = dyn_cast_or_null<RegionScanOp>(owner))
      segmentSource =
          isSegmentSource(scan.getSources().size(), scan.getAxis(),
                          scan.getSummarize().front()) ||
          isSegmentSource(scan.getSources().size(), scan.getAxis(),
                          scan.getEmit().front());
    if (segmentSource) {
      // RegionFoldOp/RegionScanOp verification binds this exact block argument
      // axis to the operation's segment parameter.  The slice extent is a
      // first-class physical relation, not something to rediscover from the
      // unsliced outer source range.
      result.state = PhysicalFactState::Exact;
      result.extentAuthority =
          PhysicalAxisRealizationFact::ExtentAuthority::Structural;
      return result;
    }
  }

  if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
    auto source = dyn_cast<FragmentType>(broadcast.getValue().getType());
    if (source) {
      BroadcastProjection projection = queryAxisProjection(source, fragment);
      if (projection.isExact() &&
          fragmentAxis < projection.targetToSource.size())
        if (std::optional<unsigned> sourceAxis =
                projection.targetToSource[fragmentAxis];
            sourceAxis && source.getShape()[*sourceAxis] == extent) {
          PhysicalAxisRealizationFact input =
              axisRealization(broadcast.getValue(), *sourceAxis);
          if (input.hasExtentAuthority()) {
            result.state = PhysicalFactState::Exact;
            result.physicalized = input.physicalized;
            result.constructionScalarSeed = input.constructionScalarSeed;
            result.roots = input.roots;
            result.extentAuthority =
                PhysicalAxisRealizationFact::ExtentAuthority::Structural;
            return result;
          }
        }
    }
  }

  if (auto splat = value.getDefiningOp<SplatOp>()) {
    auto size = constantPhysicalExpression(extent);
    if (size) {
      DominanceInfo dominance(kernel);
      SmallVector<MakeRangeOp> visible;
      for (MakeRangeOp range : programRanges(result.source).roots) {
        auto dimension = queryRangeDimension(range);
        if (succeeded(dimension) && *dimension == result.dimensionId &&
            dominance.dominates(range.getOperation(), splat.getOperation()) &&
            valueMatchesExtent(range.getExtent(), extent) &&
            constantLogicalRangeCardinality(range) == size &&
            samePhysicalScalarExpression(range.getStart(),
                                         range.getLogicalStart()))
          visible.push_back(range);
      }
      if (lockstepRanges(visible).isExact()) {
        result.state = PhysicalFactState::Exact;
        result.physicalized = true;
        result.roots = std::move(visible);
        result.extentAuthority =
            PhysicalAxisRealizationFact::ExtentAuthority::Range;
        return result;
      }
    }
  }

  if (auto join = value.getDefiningOp<JoinOp>()) {
    if (fragmentAxis == join.getAxis()) {
      result.state = PhysicalFactState::Exact;
      result.physicalized = true;
      result.extentAuthority =
          PhysicalAxisRealizationFact::ExtentAuthority::Structural;
      return result;
    }
    auto lhs = axisRealization(join.getLhs(), fragmentAxis);
    auto rhs = axisRealization(join.getRhs(), fragmentAxis);
    if (lhs.isExact() && lhs.physicalized && !lhs.constructionScalarSeed &&
        rhs.isExact() && rhs.physicalized && !rhs.constructionScalarSeed) {
      result.state = PhysicalFactState::Exact;
      result.physicalized = true;
      auto ranges = axisRanges(value, fragmentAxis);
      if (ranges.isExact())
        result.roots = std::move(ranges.roots);
      result.extentAuthority =
          PhysicalAxisRealizationFact::ExtentAuthority::Structural;
      return result;
    }
  }

  if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
    auto inputType = cast<FragmentType>(reshape.getValue().getType());
    unsigned sourceRank = 0, resultRank = 0;
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = cast<ReshapeGroupAttr>(attribute);
      sourceRank += group.getSourceAxes().size();
      resultRank += group.getResultAxes().size();
    }
    unsigned sourcePrefix = inputType.getShape().size() - sourceRank;
    unsigned resultPrefix = fragment.getShape().size() - resultRank;
    if (fragmentAxis < resultPrefix)
      return axisRealization(reshape.getValue(), fragmentAxis);
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = cast<ReshapeGroupAttr>(attribute);
      if (!llvm::is_contained(group.getResultAxes().asArrayRef(),
                              fragmentAxis - resultPrefix))
        continue;
      bool physicalized = llvm::all_of(
          group.getSourceAxes().asArrayRef(), [&](int64_t axis) {
            auto input = axisRealization(reshape.getValue(), sourcePrefix + axis);
            return input.isExact() && input.physicalized &&
                   !input.constructionScalarSeed;
          });
      if (physicalized) {
        result.state = PhysicalFactState::Exact;
        result.physicalized = true;
        auto ranges = axisRanges(value, fragmentAxis);
        if (ranges.isExact())
          result.roots = std::move(ranges.roots);
        result.extentAuthority =
            PhysicalAxisRealizationFact::ExtentAuthority::Structural;
        return result;
      }
      break;
    }
  }

  if (Operation *producer = value.getDefiningOp();
      isa_and_nonnull<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(producer)) {
    for (Value operand : producer->getOperands()) {
      auto source = dyn_cast<FragmentType>(operand.getType());
      if (!source)
        continue;
      BroadcastProjection projection = queryAxisProjection(source, fragment);
      if (!projection.isExact() || !projection.targetToSource[fragmentAxis])
        continue;
      unsigned sourceAxis = *projection.targetToSource[fragmentAxis];
      if (source.getShape()[sourceAxis] != extent)
        continue;
      PhysicalAxisRealizationFact input = axisRealization(operand, sourceAxis);
      bool introducedUnit =
          extent.getKind() == PhysicalExprKind::Constant &&
          extent.getValue() == 1 && input.roots.empty();
      if (input.hasExtentAuthority() && input.physicalized &&
          !input.constructionScalarSeed && !introducedUnit) {
        result.state = PhysicalFactState::Exact;
        result.physicalized = true;
        result.roots = input.roots;
        result.extentAuthority =
            PhysicalAxisRealizationFact::ExtentAuthority::Structural;
        return result;
      }
    }
  }

  if (auto loop = value.getDefiningOp<scf::ForOp>()) {
    auto reductions = loop->getAttrOfType<ArrayAttr>(reductionSourcesAttr);
    auto opResult = dyn_cast<OpResult>(value);
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    bool preservesAxis = reductions && !reductions.empty() &&
                         llvm::none_of(reductions, [&](Attribute attribute) {
                           auto reduction = dyn_cast<PhysicalSourceAttr>(attribute);
                           return reduction &&
                                  PhysicalSourceAxis{reduction.getSourceId(),
                                                     reduction.getSourceAxis(),
                                                     reduction.getDerived()} ==
                                      result.source;
                         });
    if (preservesAxis && opResult && yield &&
        opResult.getResultNumber() < loop.getInitArgs().size() &&
        loop.getInitArgs()[opResult.getResultNumber()].getType() == fragment &&
        yield.getOperand(opResult.getResultNumber()).getType() == fragment) {
      PhysicalRangeFact provenance =
          sourceRanges(yield.getOperand(opResult.getResultNumber()), result.source);
      auto kind = extent.getKind();
      bool physicalExtent =
          kind != PhysicalExprKind::Dimension &&
          kind != PhysicalExprKind::ScalarABI &&
          (!(kind == PhysicalExprKind::Constant && extent.getValue() == 1) ||
           (!provenance.roots.empty() &&
            llvm::all_of(provenance.roots, [](MakeRangeOp range) {
              return isProvablySingletonLogicalRange(range) ||
                     isProgramCoordinateRange(range);
            })));
      bool exactYield = provenance.isExact() && !provenance.roots.empty() &&
                        llvm::all_of(provenance.roots, [&](MakeRangeOp range) {
                          FailureOr<int64_t> dimension =
                              queryRangeDimension(range);
                          return sourceAxisIdentity(range) == result.source &&
                                 succeeded(dimension) &&
                                 *dimension == result.dimensionId &&
                                 valueMatchesExtent(range.getExtent(), extent);
                        });
      if (physicalExtent && exactYield) {
        result.state = PhysicalFactState::Exact;
        result.physicalized = true;
        result.roots.append(provenance.roots.begin(), provenance.roots.end());
        result.extentAuthority =
            PhysicalAxisRealizationFact::ExtentAuthority::Range;
        return result;
      }
    }
  }

  if (auto reduce = value.getDefiningOp<ReduceOp>()) {
    auto opResult = dyn_cast<OpResult>(value);
    if (opResult && opResult.getResultNumber() < reduce.getSources().size()) {
      Value sourceValue = reduce.getSources()[opResult.getResultNumber()];
      auto source = dyn_cast<FragmentType>(sourceValue.getType());
      if (source) {
        llvm::SmallDenseSet<int64_t> reduced(reduce.getAxes().begin(),
                                             reduce.getAxes().end());
        SmallVector<unsigned> freeAxes;
        for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
          if (!reduced.contains(axis))
            freeAxes.push_back(axis);
        if (fragmentAxis < freeAxes.size()) {
          unsigned sourceAxis = freeAxes[fragmentAxis];
          PhysicalAxisRealizationFact input =
              axisRealization(sourceValue, sourceAxis);
          if ((!input.hasExtentAuthority() || !input.physicalized) &&
              !input.constructionScalarSeed && input.roots.empty() &&
              input.blockers.empty()) {
            for (Value peer : reduce.getSources()) {
              auto peerType = dyn_cast<FragmentType>(peer.getType());
              if (peer == sourceValue || !peerType ||
                  peerType.getShape() != source.getShape() ||
                  peerType.getAxisMaps() != source.getAxisMaps() ||
                  peerType.getValidity() != source.getValidity() ||
                  peerType.getOwner() != source.getOwner())
                continue;
              auto coverage = axisRealization(peer, sourceAxis);
              if (coverage.hasExtentAuthority() && coverage.physicalized &&
                  !coverage.constructionScalarSeed) {
                coverage.roots.clear();
                input = std::move(coverage);
                break;
              }
            }
          }
          if (input.hasExtentAuthority() &&
              source.getShape()[sourceAxis] == extent) {
            result.state = PhysicalFactState::Exact;
            result.physicalized = input.physicalized;
            result.constructionScalarSeed = input.constructionScalarSeed;
            result.roots = input.roots;
            result.extentAuthority =
                PhysicalAxisRealizationFact::ExtentAuthority::Structural;
            return result;
          }
        }
      }
    }
  }

  if (auto contract = value.getDefiningOp<ContractOp>()) {
    llvm::SmallDenseSet<int64_t> lhsReduced(
        contract.getLhsReductionAxes().begin(),
        contract.getLhsReductionAxes().end());
    llvm::SmallDenseSet<int64_t> rhsReduced(
        contract.getRhsReductionAxes().begin(),
        contract.getRhsReductionAxes().end());
    llvm::SmallDenseSet<int64_t> rhsBatched(
        contract.getRhsBatchAxes().begin(), contract.getRhsBatchAxes().end());
    SmallVector<std::pair<Value, unsigned>> resultSources;
    auto lhs = dyn_cast<FragmentType>(contract.getLhs().getType());
    auto rhs = dyn_cast<FragmentType>(contract.getRhs().getType());
    if (lhs && rhs) {
      for (unsigned axis = 0; axis < lhs.getShape().size(); ++axis)
        if (!lhsReduced.contains(axis))
          resultSources.emplace_back(contract.getLhs(), axis);
      for (unsigned axis = 0; axis < rhs.getShape().size(); ++axis)
        if (!rhsReduced.contains(axis) && !rhsBatched.contains(axis))
          resultSources.emplace_back(contract.getRhs(), axis);
      if (fragmentAxis < resultSources.size()) {
        Value sourceValue = resultSources[fragmentAxis].first;
        unsigned sourceAxis = resultSources[fragmentAxis].second;
        auto source = cast<FragmentType>(sourceValue.getType());
        PhysicalAxisRealizationFact input =
            axisRealization(sourceValue, sourceAxis);
        if (input.hasExtentAuthority() &&
            source.getShape()[sourceAxis] == extent) {
          result.state = PhysicalFactState::Exact;
          result.physicalized = input.physicalized;
          result.constructionScalarSeed = input.constructionScalarSeed;
          result.roots = input.roots;
          result.extentAuthority =
              PhysicalAxisRealizationFact::ExtentAuthority::Structural;
          return result;
        }
      }
    }
  }

  if (value.getDefiningOp<ExtractOp>()) {
    // ExtractOp::verify requires the result type to equal the selected typed
    // record field.  MakeRecord and structured fold/scan verifiers in turn
    // require their field/result schemas to match the executable values at the
    // region boundary.  The projected extent is therefore already a current-IR
    // structural fact.  Keep any exact producer ranges as independent traversal
    // provenance: dropping them makes a reduction over an extracted record field
    // look unrelated to the range that produced that field.
    PhysicalRangeFact ranges = axisRanges(value, fragmentAxis);
    if (ranges.isExact()) {
      result.roots.append(ranges.roots.begin(), ranges.roots.end());
      result.constructionScalarSeed =
          extent.getKind() ==
              PhysicalExprKind::Constant &&
          extent.getValue() == 1 && !ranges.roots.empty() &&
          llvm::any_of(ranges.roots, [](MakeRangeOp range) {
            return !isProvablySingletonLogicalRange(range) &&
                   !isProgramCoordinateRange(range) &&
                   samePhysicalScalarExpression(range.getStart(),
                                                range.getLogicalStart());
          });
      result.physicalized =
          !result.constructionScalarSeed && !ranges.roots.empty() &&
          llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
            return valueMatchesExtent(range.getExtent(), extent);
          });
    }
    result.state = PhysicalFactState::Exact;
    result.extentAuthority =
        PhysicalAxisRealizationFact::ExtentAuthority::Structural;
    return result;
  }

  PhysicalRangeFact ranges = axisRanges(value, fragmentAxis);
  result.roots.append(ranges.roots.begin(), ranges.roots.end());
  result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
  result.constructionScalarSeed =
      extent.getKind() ==
          PhysicalExprKind::Constant &&
      extent.getValue() == 1 && !ranges.roots.empty() &&
      llvm::any_of(ranges.roots, [](MakeRangeOp range) {
        return !isProvablySingletonLogicalRange(range) &&
               !isProgramCoordinateRange(range) &&
               samePhysicalScalarExpression(range.getStart(),
                                            range.getLogicalStart());
      });
  result.physicalized =
      !result.constructionScalarSeed && !ranges.roots.empty() &&
      llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
         return valueMatchesExtent(range.getExtent(), extent);
       });
  if (value.getDefiningOp<ReshapeOp>()) {
    if (ranges.isExact() && ranges.roots.empty()) {
      if (extent.getKind() ==
              PhysicalExprKind::Constant &&
          extent.getValue() == 1) {
        result.state = PhysicalFactState::Exact;
        result.physicalized = true;
        result.extentAuthority =
            PhysicalAxisRealizationFact::ExtentAuthority::Structural;
      }
      return result;
    }
    // A verified reshape carries its own row-major physical reassociation.  Its
    // result extent remains exact even when no single pre-reshape range can be
    // projected to one split/merged result axis.  Keep that extent fact
    // separate from range provenance rather than turning a legal reshape into
    // analysis unknown.
    result.state = PhysicalFactState::Exact;
    result.extentAuthority =
        PhysicalAxisRealizationFact::ExtentAuthority::Structural;
    return result;
  }
  if (ranges.state == PhysicalFactState::Unknown || !ranges.blockers.empty())
    return result;
  if (!ranges.roots.empty() && failed(queryExactLogicalRange(ranges)) &&
      !lockstepRanges(ranges.roots).isExact()) {
    result.state = PhysicalFactState::Ambiguous;
    return result;
  }

  result.state = PhysicalFactState::Exact;
  if (result.physicalized)
    result.extentAuthority =
        PhysicalAxisRealizationFact::ExtentAuthority::Range;
  return result;
}

PhysicalContractFreeAxisFact
PhysicalProgramAnalysis::contractFreeAxes(Operation *operation) {
  PhysicalContractFreeAxisFact result;
  auto axes = queryContractionAxes(operation);
  if (!axes) {
    appendUnique(result.blockers, operation);
    return result;
  }
  auto appendOperand = [&](Value value, ArrayRef<unsigned> freeAxes) {
    bool exact = true;
    for (unsigned axis : freeAxes) {
      PhysicalContractFreeAxis fact;
      fact.operand = value;
      fact.operandAxis = axis;
      fact.realization = axisRealization(value, axis);
      fact.ranges = axisRanges(value, axis);
      result.blockers.append(fact.realization.blockers.begin(),
                             fact.realization.blockers.end());
      result.blockers.append(fact.ranges.blockers.begin(),
                             fact.ranges.blockers.end());
      if (!fact.realization.isExact()) {
        appendUnique(result.blockers, operation);
        exact = false;
      }
      result.axes.push_back(std::move(fact));
    }
    return exact;
  };
  Value lhs, rhs;
  if (auto contract = dyn_cast_or_null<ContractOp>(operation)) {
    lhs = contract.getLhs();
    rhs = contract.getRhs();
  } else if (auto contract = dyn_cast_or_null<ScaledContractOp>(operation)) {
    lhs = contract.getLhs();
    rhs = contract.getRhs();
  } else if (auto contract = dyn_cast_or_null<SparseContractOp>(operation)) {
    lhs = contract.getCompressed();
    rhs = contract.getRhs();
  }
  bool lhsExact = appendOperand(lhs, axes->lhsFree);
  bool rhsExact = appendOperand(rhs, axes->rhsFree);
  if (result.axes.empty()) {
    appendUnique(result.blockers, operation);
    return result;
  }
  if (lhsExact && rhsExact)
    result.state = PhysicalFactState::Exact;
  return result;
}

PhysicalRangeAxisFact
PhysicalProgramAnalysis::rangeAxes(Value value,
                                   ArrayRef<MakeRangeOp> selectedRoots) {
  PhysicalRangeAxisFact result;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || selectedRoots.empty())
    return result;
  if (auto splat = value.getDefiningOp<SplatOp>()) {
    (void)splat;
    result.state = PhysicalFactState::Exact;
    return result;
  }
  if (auto broadcast = value.getDefiningOp<BroadcastOp>();
      broadcast && !isa<FragmentType>(broadcast.getValue().getType())) {
    result.state = PhysicalFactState::Exact;
    return result;
  }
  result.state = PhysicalFactState::Exact;
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    PhysicalRangeFact ranges = axisRanges(value, axis);
    auto axisMap = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    auto selectedRange = [&](MakeRangeOp range) {
      return llvm::any_of(selectedRoots, [&](MakeRangeOp selectedRoot) {
        if (range == selectedRoot || sameLogicalRange(range, selectedRoot))
          return true;
        // A renamed coordinate must project through the already selected axis.
        auto dimension = queryRangeDimension(range);
        auto selectedDimension = queryRangeDimension(selectedRoot);
        return succeeded(dimension) && succeeded(selectedDimension) &&
               *dimension == *selectedDimension &&
               axisMap.getDimensionId() == *selectedDimension &&
               sourceAxisIdentity(axisMap) == sourceAxisIdentity(selectedRoot) &&
               lockstepRanges({range, selectedRoot}).isExact();
      });
    };
    bool selected = llvm::any_of(ranges.roots, selectedRange);
    if (selected) {
      if (failed(queryExactLogicalRange(ranges)) &&
          (ranges.state == PhysicalFactState::Unknown ||
           !ranges.blockers.empty() ||
           !llvm::all_of(ranges.roots, selectedRange) ||
           !lockstepRanges(ranges.roots).isExact())) {
        result.state = PhysicalFactState::Ambiguous;
        result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
        return result;
      }
      result.fragmentAxes.push_back(axis);
      continue;
    }
    if (ranges.state != PhysicalFactState::Unknown)
      continue;
    auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    bool mayCarrySelectedRoot = llvm::any_of(selectedRoots, [&](MakeRangeOp root) {
      FailureOr<int64_t> dimension = queryRangeDimension(root);
      return sourceAxisIdentity(mapping) == sourceAxisIdentity(root) &&
             succeeded(dimension) && mapping.getDimensionId() == *dimension;
    });
    if (!mayCarrySelectedRoot)
      continue;
    result.state = PhysicalFactState::Unknown;
    result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
    return result;
  }
  return result;
}

PhysicalLockstepTraversalFact PhysicalProgramAnalysis::lockstepTraversal(
    ValueRange sources, ArrayRef<unsigned> fragmentAxes) {
  PhysicalLockstepTraversalFact result;
  if (sources.empty() || sources.size() != fragmentAxes.size())
    return result;
  SmallVector<MakeRangeOp> authorities;
  for (auto [source, fragmentAxis] : llvm::zip(sources, fragmentAxes)) {
    PhysicalRangeFact ranges = axisRanges(source, fragmentAxis);
    if (!ranges.isExact()) {
      result.state = ranges.state == PhysicalFactState::Ambiguous
                         ? PhysicalLockstepState::Inconsistent
                         : PhysicalLockstepState::Unknown;
      result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
      return result;
    }
    PhysicalLockstepTraversalFact sourceFact = lockstepRanges(ranges.roots);
    if (!sourceFact.isExact()) {
      result.state = sourceFact.state;
      result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
      result.blockers.append(sourceFact.blockers.begin(),
                             sourceFact.blockers.end());
      return result;
    }
    authorities.push_back(sourceFact.authority);
  }
  return lockstepRanges(authorities);
}

PhysicalLockstepTraversalFact
PhysicalProgramAnalysis::lockstepRanges(ArrayRef<MakeRangeOp> ranges) {
  PhysicalLockstepTraversalFact result;
  if (ranges.empty())
    return result;
  auto sameValue = [](Value lhs, Value rhs) {
    if (samePhysicalScalarExpression(lhs, rhs))
      return true;
    PhysicalExprAttr left = queryLaunchExpression(lhs);
    PhysicalExprAttr right = queryLaunchExpression(rhs);
    return left && right && left == right;
  };
  auto sameLogicalTraversal = [&](MakeRangeOp lhs, MakeRangeOp rhs) {
    return sameValue(lhs.getLogicalStart(), rhs.getLogicalStart()) &&
           sameValue(lhs.getLogicalStop(), rhs.getLogicalStop()) &&
           sameValue(lhs.getStep(), rhs.getStep());
  };
  auto sameTraversal = [&](MakeRangeOp lhs, MakeRangeOp rhs) {
    auto isZero = [](Value value) {
      PhysicalExprAttr bound = queryNonNegativeIndexUpperBound(value);
      return bound &&
             bound.getKind() == PhysicalExprKind::Constant &&
             bound.getValue() == 0;
    };
    bool sameStart = sameValue(lhs.getStart(), rhs.getStart()) ||
                     (isZero(lhs.getStart()) && isZero(rhs.getStart()));
    return sameStart &&
           sameValue(lhs.getExtent(), rhs.getExtent()) &&
           sameValue(lhs.getStep(), rhs.getStep());
  };
  result.authority = ranges.front();
  for (MakeRangeOp range : llvm::drop_begin(ranges))
    if (!sameTraversal(result.authority, range) ||
        !sameLogicalTraversal(result.authority, range)) {
      result.state = PhysicalLockstepState::Inconsistent;
      return result;
    }
  result.state = PhysicalLockstepState::Exact;
  return result;
}

void PhysicalProgramAnalysis::analyzeReplay(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalReplayScope scope, bool allowAccesses,
    Operation *insertionAnchor, std::optional<int64_t> sourceDimension,
    DominanceInfo *dominance,
    PhysicalReplayFact &result,
    ReplayVisits &visited) {
  if (!value)
    return;
  ReplayContext context{source, sourceDimension};
  auto ensureEnclosingReplay = [&](Operation *parent) {
    if (scope != PhysicalReplayScope::ValueGraph || !source || !sourceDimension)
      return false;
    Operation *root = nullptr;
    for (; isa_and_nonnull<scf::IfOp, scf::ForOp>(parent);
         parent = parent->getParentOp()) {
      auto found = visited.find(parent);
      if (found == visited.end() || found->second.empty())
        break;
      root = parent;
    }
    if (!root || !root->getNumResults())
      return false;
    if (!llvm::is_contained(visited.lookup(root), context))
      analyzeReplay(root->getResult(0), source, scope, allowAccesses,
                    insertionAnchor, sourceDimension, dominance, result, visited);
    return result.state != PhysicalFactState::Unknown &&
           llvm::is_contained(visited.lookup(root), context);
  };
  bool carriesRequestedTraversal = false;
  if (source) {
    if (sourceDimension) {
      carriesRequestedTraversal = llvm::any_of(
          queryFragmentAxes(value.getType(), *source),
          [&](const PhysicalAxisProjection &projection) {
            return projection.dimensionId == *sourceDimension;
          });
    } else {
      carriesRequestedTraversal = carriesSource(value.getType(), *source);
    }
  }
  Operation *definition = value.getDefiningOp();
  bool recheckRegion = isa_and_nonnull<scf::IfOp, scf::ForOp>(definition) &&
      scope == PhysicalReplayScope::ValueGraph && source && sourceDimension &&
      visited.count(definition) && !visited.lookup(definition).empty() &&
      !llvm::is_contained(visited.lookup(definition), context);
  if (insertionAnchor && dominance && !recheckRegion &&
      dominance->dominates(value, insertionAnchor) &&
      !carriesRequestedTraversal) {
    SmallPtrSet<Operation *, 16> dependencyVisited;
    collectStructuredPrograms(value, dependencyVisited,
                              result.structuredPrograms);
    result.crossesStructuredProgram |= !result.structuredPrograms.empty();
    return;
  }
  if (auto extract = value.getDefiningOp<ExtractOp>()) {
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
      uint64_t field = extract.getField();
      if (field >= record.getFields().size()) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, extract);
        return;
      }
      analyzeReplay(record.getFields()[field], source, scope, allowAccesses,
                    insertionAnchor, sourceDimension, dominance, result,
                    visited);
      return;
    }
  }
  bool physicalValue = isa<FragmentType, RecordType>(value.getType());
  if (!physicalValue && !insertionAnchor)
    return;
  if (source && physicalValue && !carriesSource(value.getType(), *source) &&
      !insertionAnchor)
    return;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *owner = argument.getOwner()->getParentOp();
    if (auto loop = dyn_cast_or_null<scf::ForOp>(owner)) {
      bool replayingLoop = ensureEnclosingReplay(loop);
      // Replaying the complete loop checks its initial operands and all
      // yields. Its block arguments are local bindings, including unchanged
      // carries whose backedge would otherwise recurse into themselves.
      if (replayingLoop)
        return;
      if (!replayingLoop && argument != loop.getInductionVar()) {
        appendUnique(result.blockers, loop);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
    SmallVector<Value, 2> outer = structuredSourcesForArgument(argument);
    if (outer.empty()) {
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value related : outer)
      analyzeReplay(related, source, scope, allowAccesses, insertionAnchor,
                    sourceDimension, dominance, result, visited);
    return;
  }
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return;
  auto &contexts = visited[operation];
  if (llvm::is_contained(contexts, context))
    return;
  contexts.push_back(context);
  if (isa<ContractOp, ScaledContractOp, SparseContractOp>(operation))
    appendUnique(result.contractions, operation);
  if (auto range = dyn_cast<MakeRangeOp>(operation)) {
    if (insertionAnchor && source && sourceDimension &&
        sourceAxisIdentity(range) == *source) {
      FailureOr<int64_t> dimension = queryRangeDimension(range);
      if (failed(dimension)) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
      }
    }
    return;
  }
  if (isa<scf::IfOp, scf::ForOp>(operation) &&
      scope == PhysicalReplayScope::ValueGraph && source && sourceDimension) {
    if (isa<scf::ForOp>(operation)) {
      auto dependency = reductionDependency(value, *source, *sourceDimension);
      if (!dependency.isExact() || dependency.depends) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
    auto readOnly = [](Operation *root) {
      return !root->walk([](Operation *nested) {
        return isa<LoadOp, GatherOp, scf::IfOp, scf::ForOp>(nested) ||
                       isMemoryEffectFree(nested)
                   ? WalkResult::advance() : WalkResult::interrupt();
      }).wasInterrupted();
    };
    SmallVector<Value> controls;
    if (auto branch = dyn_cast<scf::IfOp>(operation))
      controls.push_back(branch.getCondition());
    else {
      auto loop = cast<scf::ForOp>(operation);
      controls = {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()};
    }
    llvm::DenseSet<Value> controlValues;
    bool dependentControl = false;
    for (unsigned index = 0; index < controls.size(); ++index) {
      Value control = controls[index];
      if (!controlValues.insert(control).second)
        continue;
      dependentControl |= typeCarriesTraversal(control.getType(), *source, *sourceDimension);
      if (Operation *producer = control.getDefiningOp())
        controls.append(producer->getOperands().begin(), producer->getOperands().end());
    }
    bool preservesReads = readOnly(operation);
    if (insertionAnchor && preservesReads) {
      // Nested control is cloned as part of an already visited replay region.
      // Check motion from that enclosing region, rather than trying to walk
      // an outer insertion point into the nested operation's original block.
      Operation *replayRoot = operation;
      for (Operation *parent = operation->getParentOp();
           isa_and_nonnull<scf::IfOp, scf::ForOp>(parent);
           parent = parent->getParentOp()) {
        auto found = visited.find(parent);
        if (found == visited.end() ||
            !llvm::is_contained(found->second, context))
          break;
        replayRoot = parent;
      }
      Operation *anchor = insertionAnchor;
      while (anchor && anchor->getBlock() != replayRoot->getBlock()) {
        preservesReads &= readOnly(anchor);
        anchor = anchor->getParentOp();
      }
      preservesReads &= anchor &&
          (anchor == replayRoot || replayRoot->isBeforeInBlock(anchor));
      if (preservesReads && anchor != replayRoot)
        for (Operation *next = replayRoot->getNextNode(); next != anchor;
             next = next->getNextNode())
          preservesReads &= readOnly(next);
    }
    if (!preservesReads || dependentControl) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value operand : operation->getOperands())
      analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                    sourceDimension, dominance, result, visited);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (Value yielded : block.getTerminator()->getOperands())
          analyzeReplay(yielded, source, scope, allowAccesses, insertionAnchor,
                        sourceDimension, dominance, result, visited);
    return;
  }
  if (!physicalValue) {
    if (isa<arith::ConstantOp>(operation))
      return;
    if (!isPhysicalReplayNode(operation, scope, allowAccesses) ||
        operation->getNumRegions() != 0 || operation->getNumResults() != 1) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value operand : operation->getOperands())
      analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                    sourceDimension, dominance, result, visited);
    return;
  }
  if (isAccessNode(operation)) {
    result.crossesAccess = true;
    appendUnique(result.accesses, operation);
    if (!allowAccesses) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    if (auto load = dyn_cast<LoadOp>(operation); load && insertionAnchor) {
      bool replayedRegion = ensureEnclosingReplay(load->getParentOp());
      // Enclosing replay regions already prove read-only motion. Standalone
      // reads must also preserve their value across intervening writes.
      if (!replayedRegion && !canReplayReadAt(load, insertionAnchor)) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
  } else if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
    result.crossesStructuredProgram = true;
    appendUnique(result.structuredPrograms, operation);
    if (!source || !sourceDimension) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value segmentSource :
         fold.getSources()) {
      auto fragment = dyn_cast<FragmentType>(segmentSource.getType());
      if (!fragment || fold.getAxis() >= fragment.getAxisMaps().size()) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
      auto mapping =
          cast<AxisMapAttr>(fragment.getAxisMaps()[fold.getAxis()]);
      if (sourceAxisIdentity(mapping) == *source &&
          mapping.getDimensionId() == *sourceDimension) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
    for (Value operand : fold.getOperands())
      if (typeCarriesTraversal(operand.getType(), *source, *sourceDimension))
        analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                      sourceDimension, dominance, result, visited);
    return;
  } else if (auto scan = dyn_cast<RegionScanOp>(operation)) {
    result.crossesStructuredProgram = true;
    appendUnique(result.structuredPrograms, operation);
    if (!source || !sourceDimension) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value segmentSource :
         scan.getSources()) {
      auto fragment = dyn_cast<FragmentType>(segmentSource.getType());
      if (!fragment || scan.getAxis() >= fragment.getAxisMaps().size()) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
      auto mapping =
          cast<AxisMapAttr>(fragment.getAxisMaps()[scan.getAxis()]);
      if (sourceAxisIdentity(mapping) == *source &&
          mapping.getDimensionId() == *sourceDimension) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
    for (Value operand : scan.getOperands())
      if (typeCarriesTraversal(operand.getType(), *source, *sourceDimension))
        analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                      sourceDimension, dominance, result, visited);
    return;
  } else if (!isPhysicalReplayNode(operation, scope, allowAccesses)) {
    appendUnique(result.blockers, operation);
    result.state = PhysicalFactState::Unknown;
    return;
  }
  Region *combine = nullptr;
  if (auto reduce = dyn_cast<ReduceOp>(operation))
    combine = &reduce.getCombine();
  else if (auto scan = dyn_cast<ScanOp>(operation)) {
    // A prefix depends on earlier members of this axis. Replaying each tile
    // independently would reset that state; only the other axes are pointwise.
    SmallVector<Type> schemas{value.getType()};
    while (source && !schemas.empty()) {
      Type schema = schemas.pop_back_val();
      if (auto record = dyn_cast<RecordType>(schema)) {
        for (Attribute field : record.getFieldTypes())
          schemas.push_back(cast<TypeAttr>(field).getValue());
        continue;
      }
      auto fragment = cast<FragmentType>(schema);
      auto axis = cast<AxisMapAttr>(fragment.getAxisMaps()[scan.getAxis()]);
      if (sourceAxisIdentity(axis) == *source &&
          (!sourceDimension || axis.getDimensionId() == *sourceDimension)) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
    combine = &scan.getCombine();
  }
  if (combine) {
    WalkResult helper = combine->walk([&](Operation *nested) {
      if (isa<YieldOp, arith::ConstantOp>(nested))
        return WalkResult::advance();
      if (!isPhysicalReplayNode(nested, PhysicalReplayScope::Coordinate,
                                /*allowAccesses=*/false)) {
        appendUnique(result.blockers, nested);
        result.state = PhysicalFactState::Unknown;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (helper.wasInterrupted())
      return;
  }
  if (operation->getNumRegions() != 0 && !isa<ReduceOp, ScanOp>(operation)) {
    appendUnique(result.blockers, operation);
    result.state = PhysicalFactState::Unknown;
    return;
  }
  if (auto broadcast = dyn_cast<BroadcastOp>(operation); broadcast && source) {
    auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
    auto output = dyn_cast<FragmentType>(value.getType());
    PhysicalAxisProjection requested = queryFragmentAxis(output, *source);
    if (input && requested.isExact()) {
      BroadcastProjection projection = queryAxisProjection(input, output);
      if (projection.isExact() &&
          requested.fragmentAxis < projection.targetToSource.size())
        if (auto axis = projection.targetToSource[requested.fragmentAxis]) {
          auto mapping = cast<AxisMapAttr>(input.getAxisMaps()[*axis]);
          if (sourceDimension && mapping.getDimensionId() <= 0) {
            appendUnique(result.blockers, operation);
            result.state = PhysicalFactState::Unknown;
            return;
          }
          std::optional<int64_t> inputDimension =
              sourceDimension ? std::optional<int64_t>(mapping.getDimensionId())
                              : std::nullopt;
          analyzeReplay(broadcast.getValue(), sourceAxisIdentity(mapping), scope,
                        allowAccesses, insertionAnchor, inputDimension,
                        dominance, result, visited);
          return;
        }
    }
  }
  for (Value operand : operation->getOperands())
    analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                  sourceDimension, dominance, result, visited);
}

PhysicalReplayFact PhysicalProgramAnalysis::replayability(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalReplayScope scope, bool allowAccesses,
    Operation *insertionAnchor,
    std::optional<int64_t> sourceDimension) {
  PhysicalReplayFact result;
  result.state = PhysicalFactState::Exact;
  ReplayVisits visited;
  std::optional<DominanceInfo> dominance;
  if (insertionAnchor)
    dominance.emplace(kernel);
  analyzeReplay(value, source, scope, allowAccesses, insertionAnchor,
                sourceDimension, dominance ? &*dominance : nullptr, result,
                visited);
  return result;
}

PhysicalReductionDependencyFact PhysicalProgramAnalysis::reductionDependency(
    Value value, PhysicalSourceAxis source,
    std::optional<int64_t> sourceDimension) {
  using Traversal = std::pair<PhysicalSourceAxis, std::optional<int64_t>>;
  llvm::DenseMap<Value, SmallVector<Traversal, 2>> visited;
  std::function<PhysicalReductionDependencyFact(Value, PhysicalSourceAxis,
                                               std::optional<int64_t>)> analyze =
      [&](Value current, PhysicalSourceAxis source,
          std::optional<int64_t> sourceDimension) -> PhysicalReductionDependencyFact {
    PhysicalReductionDependencyFact exact;
    exact.state = PhysicalFactState::Exact;
    Operation *operation = current.getDefiningOp();
    auto &contexts = visited[current];
    Traversal context{source, sourceDimension};
    if (!operation || llvm::is_contained(contexts, context))
      return exact;
    contexts.push_back(context);
    if (auto range = dyn_cast<MakeRangeOp>(operation)) {
      // Reaching a coordinate proves an ordinary value dependency, not a
      // reduction dependency.  Only an operation that removes or carries this
      // axis may turn the fact into `depends=true` below.  Treating every range
      // leaf as a reduction made pointwise ownership axes both program-mapped
      // and internally traversed, so different programs replayed overlapping
      // writeback domains.
      return exact;
    }
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      for (Value input :
           reduce.getSources()) {
        if (reductionTypeConsumesSource(input.getType(), reduce.getAxes(),
                                        source, sourceDimension)) {
          exact.depends = true;
          return exact;
        }
        if (!sourceDimension)
          continue;
        auto fragment = dyn_cast<FragmentType>(input.getType());
        if (!fragment)
          continue;
        for (int64_t axis : reduce.getAxes()) {
          if (axis < 0 ||
              axis >= static_cast<int64_t>(fragment.getAxisMaps().size()))
            continue;
          auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
          PhysicalRangeFact ranges = sourceRanges(
              input, sourceAxisIdentity(mapping));
          if (llvm::any_of(ranges.roots, [&](MakeRangeOp range) {
                FailureOr<int64_t> dimension = queryRangeDimension(range);
                return succeeded(dimension) &&
                       *dimension == *sourceDimension;
              })) {
            exact.depends = true;
            return exact;
          }
        }
      }
      return exact;
    }
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      // Preserve the reduction dependency before vector/matrix realization
      // chooses the native operation for this contraction.
      exact.depends = reductionTypeConsumesSource(
                          contract.getLhs().getType(),
                          contract.getLhsReductionAxes(), source, sourceDimension) ||
                      reductionTypeConsumesSource(
                          contract.getRhs().getType(),
                          contract.getRhsReductionAxes(), source, sourceDimension);
      if (exact.depends)
        return exact;
    }
    if (auto scan = dyn_cast<ScanOp>(operation)) {
      for (Value input : scan.getSources())
        if (reductionTypeConsumesSource(
                input.getType(),
                ArrayRef<int64_t>{static_cast<int64_t>(scan.getAxis())},
                source, sourceDimension)) {
          exact.depends = true;
          return exact;
        }
      return exact;
    }
    if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
      auto result = dyn_cast<OpResult>(current);
      if (sourceDimension && result && result.getOwner() == fold &&
          typeCarriesTraversal(current.getType(), source, *sourceDimension)) {
        exact.depends = true;
        exact.throughStructuredReduction = true;
      }
      return exact;
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      auto traversals = loop->getAttrOfType<ArrayAttr>(reductionSourcesAttr);
      if (traversals && llvm::any_of(traversals, [&](Attribute attribute) {
            auto traversal = dyn_cast<PhysicalSourceAttr>(attribute);
            return traversal &&
                   PhysicalSourceAxis{traversal.getSourceId(),
                                      traversal.getSourceAxis(),
                                      traversal.getDerived()} == source;
          })) {
        exact.depends = true;
        return exact;
      }
      auto result = dyn_cast<OpResult>(current);
      auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
      if (!result || !yield ||
          result.getResultNumber() >= loop.getInitArgs().size()) {
        exact.state = PhysicalFactState::Unknown;
        appendUnique(exact.blockers, operation);
        return exact;
      }
      unsigned index = result.getResultNumber();
      auto carryFeedsReduction = [&](unsigned carryAxis) {
        SmallVector<std::pair<Value, unsigned>> pending{
            {loop.getRegionIterArgs()[index], carryAxis}};
        llvm::DenseMap<Value, SmallVector<unsigned, 2>> reached;
        for (unsigned cursor = 0; cursor < pending.size(); ++cursor) {
          auto [carried, axis] = pending[cursor];
          auto &axes = reached[carried];
          if (llvm::is_contained(axes, axis))
            continue;
          axes.push_back(axis);
          auto input = cast<FragmentType>(carried.getType());
          for (OpOperand &use : carried.getUses()) {
            Operation *user = use.getOwner();
            if (!loop->isProperAncestor(user))
              continue;
            if (auto reduce = dyn_cast<ReduceOp>(user)) {
              if (use.getOperandNumber() < reduce.getSources().size() &&
                  llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
                return true;
              continue;
            }
            if (user->getNumResults() != 1)
              continue;
            auto output = dyn_cast<FragmentType>(user->getResult(0).getType());
            if (!output)
              continue;
            if (auto reshape = dyn_cast<ReshapeOp>(user)) {
              bool positional = input.getShape() == output.getShape() &&
                  llvm::all_of(reshape.getReassociation(), [](Attribute attribute) {
                    auto group = cast<ReshapeGroupAttr>(attribute);
                    return group.getSourceAxes() == group.getResultAxes();
                  });
              if (positional)
                pending.emplace_back(user->getResult(0), axis);
            } else if (auto transpose = dyn_cast<TransposeOp>(user)) {
              for (auto [targetAxis, sourceAxis] :
                   llvm::enumerate(transpose.getPermutation()))
                if (sourceAxis == static_cast<int64_t>(axis))
                  pending.emplace_back(user->getResult(0), targetAxis);
            } else if (isa<BroadcastOp, UnaryOp, BinaryOp, CompareOp, SelectOp,
                           CastOp, BitcastOp>(user)) {
              auto projection = queryBroadcastProjection(input, output);
              if (projection.isExact())
                for (auto [targetAxis, sourceAxis] :
                     llvm::enumerate(projection.targetToSource))
                  if (sourceAxis && *sourceAxis == axis)
                    pending.emplace_back(user->getResult(0), targetAxis);
            }
          }
        }
        return false;
      };
      SmallVector<Traversal, 4> loopTraversals{context};
      for (const auto &projection : queryFragmentAxes(current.getType(), source)) {
        if (sourceDimension && projection.dimensionId != *sourceDimension)
          continue;
        // A loop carry can be rebound to another operand's positional axes
        // before it is reduced. Its output coordinate roots alone do not
        // expose that recurrence; follow the carry's exact pointwise uses.
        if (carryFeedsReduction(projection.fragmentAxis)) {
          exact.depends = true;
          return exact;
        }
        auto ranges = axisRanges(current, projection.fragmentAxis);
        if (!ranges.isExact() || !ranges.blockers.empty()) {
          exact.state = PhysicalFactState::Unknown;
          appendUnique(exact.blockers, operation);
        } else
          for (MakeRangeOp range : ranges.roots) {
            auto dimension = queryRangeDimension(range);
            if (failed(dimension))
              continue;
            Traversal related{sourceAxisIdentity(range), *dimension};
            if (!llvm::is_contained(loopTraversals, related))
              loopTraversals.push_back(related);
          }
      }
      for (Value related :
           {loop.getInitArgs()[index], yield.getResults()[index]}) {
        for (auto [identity, dimension] : loopTraversals) {
          PhysicalReductionDependencyFact nested = analyze(related, identity, dimension);
          if (nested.depends)
            return nested;
          if (!nested.isExact()) {
            exact.state = PhysicalFactState::Unknown;
            llvm::append_range(exact.blockers, nested.blockers);
          }
        }
      }
      return exact;
    }
    if (operation->getNumRegions() != 0) {
      exact.state = PhysicalFactState::Unknown;
      appendUnique(exact.blockers, operation);
      return exact;
    }
    for (Value operand : operation->getOperands()) {
      SmallVector<Traversal, 4> traversals{context};
      auto input = dyn_cast<FragmentType>(operand.getType());
      auto output = dyn_cast<FragmentType>(current.getType());
      auto requested = output ? queryFragmentAxis(output, source) : PhysicalAxisProjection{};
      if (input && requested.isExact() &&
          (!sourceDimension || requested.dimensionId == *sourceDimension)) {
        SmallVector<unsigned> inputAxes;
        if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
          unsigned sourceRank = 0, resultRank = 0;
          for (Attribute attribute : reshape.getReassociation()) {
            auto group = cast<ReshapeGroupAttr>(attribute);
            sourceRank += group.getSourceAxes().size();
            resultRank += group.getResultAxes().size();
          }
          unsigned sourcePrefix = input.getShape().size() - sourceRank;
          unsigned resultPrefix = output.getShape().size() - resultRank;
          if (requested.fragmentAxis < resultPrefix && sourcePrefix == resultPrefix)
            inputAxes.push_back(requested.fragmentAxis);
          else
            for (Attribute attribute : reshape.getReassociation()) {
              auto group = cast<ReshapeGroupAttr>(attribute);
              if (llvm::is_contained(group.getResultAxes().asArrayRef(),
                    static_cast<int64_t>(requested.fragmentAxis) - resultPrefix))
                for (int64_t axis : group.getSourceAxes().asArrayRef())
                  inputAxes.push_back(sourcePrefix + axis);
            }
        } else if (auto transpose = dyn_cast<TransposeOp>(operation)) {
          inputAxes.push_back(transpose.getPermutation()[requested.fragmentAxis]);
        } else if (isa<BroadcastOp, UnaryOp, BinaryOp, CompareOp, SelectOp,
                       CastOp, BitcastOp>(operation)) {
          auto projection = queryAxisProjection(input, output);
          if (projection.isExact() && projection.targetToSource[requested.fragmentAxis])
            inputAxes.push_back(*projection.targetToSource[requested.fragmentAxis]);
        }
        for (unsigned axis : inputAxes) {
          auto mapping = cast<AxisMapAttr>(input.getAxisMaps()[axis]);
          Traversal projected{sourceAxisIdentity(mapping), mapping.getDimensionId()};
          if (!llvm::is_contained(traversals, projected))
            traversals.push_back(projected);
        }
      }
      for (auto [selected, dimension] : traversals) {
        PhysicalReductionDependencyFact nested = analyze(operand, selected, dimension);
        if (nested.depends)
          return nested;
        if (!nested.isExact()) {
          exact.state = PhysicalFactState::Unknown;
          llvm::append_range(exact.blockers, nested.blockers);
        }
      }
    }
    return exact;
  };
  return analyze(value, source, sourceDimension);
}

bool PhysicalProgramAnalysis::isTailPredicate(
    Value value, ArrayRef<std::pair<MakeRangeOp, Value>> ranges) const {
  if (!value)
    return true;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto integer = dyn_cast<IntegerAttr>(constant.getValue());
    return integer && integer.getType().isInteger(1) && integer.getInt() != 0;
  }
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    return isTailPredicate(broadcast.getValue(), ranges);
  if (auto splat = value.getDefiningOp<SplatOp>())
    return isTailPredicate(splat.getValue(), ranges);
  if (auto reshape = value.getDefiningOp<ReshapeOp>())
    return isTailPredicate(reshape.getValue(), ranges);
  if (auto transpose = value.getDefiningOp<TransposeOp>())
    return isTailPredicate(transpose.getValue(), ranges);
  if (auto conjunction = value.getDefiningOp<BinaryOp>()) {
    Type element = conjunction.getResult().getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    bool logical =
        conjunction.getOperatorKind() == BinaryOperator::LogicalAnd;
    bool bitwiseI1 =
        conjunction.getOperatorKind() == BinaryOperator::BitwiseAnd &&
        element.isInteger(1);
    return (logical || bitwiseI1) &&
           isTailPredicate(conjunction.getLhs(), ranges) &&
           isTailPredicate(conjunction.getRhs(), ranges);
  }
  auto comparison = value.getDefiningOp<CompareOp>();
  if (!comparison ||
      (comparison.getPredicate() != ComparePredicate::Lt &&
       comparison.getPredicate() != ComparePredicate::Ge))
    return false;
  Value lhs = stripBroadcast(comparison.getLhs());
  Value rhs = stripScalarIdentity(comparison.getRhs());
  MakeRangeOp predicateRange = lhs.getDefiningOp<MakeRangeOp>();
  auto sameTailEnd = [&](Value current, Value expected) {
    if (sameScalarExpression(current, expected))
      return true;
    if (auto dim = current.getDefiningOp<DimOp>())
      return matchesResourceExtent(expected, dim.getView(), dim.getAxis());
    if (auto dim = expected.getDefiningOp<DimOp>())
      return matchesResourceExtent(current, dim.getView(), dim.getAxis());
    return false;
  };
  return llvm::any_of(ranges, [&](const auto &entry) {
    MakeRangeOp expectedRange = entry.first;
    Value expectedEnd = stripScalarIdentity(entry.second);
    bool sameRange = lhs == expectedRange.getResult();
    if (!sameRange && predicateRange &&
        sourceAxisIdentity(predicateRange) ==
            sourceAxisIdentity(expectedRange))
      sameRange = sameScalarExpression(predicateRange.getStart(),
                                       expectedRange.getStart()) &&
                  sameScalarExpression(predicateRange.getExtent(),
                                       expectedRange.getExtent()) &&
                  sameScalarExpression(predicateRange.getStep(),
                                       expectedRange.getStep());
    if (!sameRange)
      return false;
    if (comparison.getPredicate() == ComparePredicate::Ge)
      return integerConstant(rhs) == 0;
    return sameTailEnd(rhs, expectedEnd);
  });
}

PhysicalAccessFootprint
PhysicalProgramAnalysis::footprint(Operation *access) {
  PhysicalAccessFootprint result;
  auto collect = [&](Value resource, ValueRange coordinates,
                     ArrayRef<int64_t> sourceAxes, Value validity, Value fill) {
    result.resource = resource;
    result.coordinates.append(coordinates.begin(), coordinates.end());
    result.sourceAxes.append(sourceAxes.begin(), sourceAxes.end());
    result.validity = validity;
    result.fill = fill;
    result.state = resource && coordinates.size() == sourceAxes.size()
                       ? PhysicalFactState::Exact
                       : PhysicalFactState::Unknown;
    result.rangeState = PhysicalFactState::Exact;
    for (Value coordinate : coordinates) {
      PhysicalRangeFact ranges;
      ranges.state = PhysicalFactState::Exact;
      SmallPtrSet<Operation *, 32> visited;
      // A scalar coordinate may depend on a completed reduction, but that
      // reduction's input lanes are not lanes of this memory access.  Keep the
      // coordinate SSA dependency without applying its ancestors' tail masks.
      collectRanges(coordinate, std::nullopt, ranges, visited,
                    /*followScalarDependencies=*/false);
      // A footprint records the complete set of ranges, not a request for one
      // unique range.  Multiple roots are therefore exact here.  A coordinate
      // with no range roots is also exact when it is a scalar/broadcast-only
      // expression.  Only an operation that blocks provenance makes the
      // address footprint unknown.
      if (!ranges.blockers.empty())
        result.rangeState = PhysicalFactState::Unknown;
      for (Operation *blocker : ranges.blockers)
        appendUnique(result.blockers, blocker);
      for (MakeRangeOp range : ranges.roots)
        appendUnique(result.ranges, range);
    }
  };
  if (auto relation = dyn_cast<AccessOpInterface>(access))
    collect(relation.getAccessResource(), relation.getAccessCoordinates(),
            relation.getAccessSourceAxes(), relation.getAccessValidity(),
            relation.getAccessFill());
  else
    result.state = PhysicalFactState::Unknown;
  return result;
}

bool haveDisjointPrivateBufferAccesses(Operation *lhs, Operation *rhs) {
  if (!isa<LoadOp, StoreOp>(lhs) || !isa<LoadOp, StoreOp>(rhs))
    return false;
  PhysicalProgramAnalysis analysis(lhs->getParentOfType<func::FuncOp>());
  auto left = analysis.footprint(lhs);
  auto right = analysis.footprint(rhs);
  if (left.state != PhysicalFactState::Exact ||
      right.state != PhysicalFactState::Exact)
    return false;
  auto buffer = dyn_cast<BufferType>(left.resource.getType());
  if (!buffer || buffer.getScope().getValue() != BufferScope::ProgramPrivate ||
      buffer.isInvocationWorkspace() || !left.resource.getDefiningOp<BufferOp>())
    return false;
  if (left.resource != right.resource)
    return true;

  auto scalar = [](Value value) {
    value = stripBroadcast(value);
    return value.getType().isIntOrIndex() ? stripScalarIdentity(value) : Value();
  };
  auto laterIteration = [&](Value coordinate, Value point) {
    auto argument = dyn_cast_or_null<BlockArgument>(coordinate);
    auto loop = argument
                    ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp())
                    : scf::ForOp();
    if (!loop || argument != loop.getInductionVar() ||
        integerConstant(loop.getStep()) != 1)
      return false;
    auto lower = stripScalarIdentity(loop.getLowerBound()).getDefiningOp<BinaryOp>();
    if (!lower || lower.getOperatorKind() != BinaryOperator::Add)
      return false;
    for (auto [base, offset] :
         {std::pair{lower.getLhs(), lower.getRhs()},
          std::pair{lower.getRhs(), lower.getLhs()}}) {
      auto amount = integerConstant(offset);
      if (stripScalarIdentity(base) != point || !amount || *amount <= 0)
        continue;
      IndexBounds pointBounds = queryIndexBounds(point);
      IndexBounds iterationBounds = queryIndexBounds(coordinate);
      auto kernel = lhs->getParentOfType<func::FuncOp>();
      auto pointExtent = pointBounds.nonNegative && pointBounds.upper
                             ? nonNegativeExtentBounds(kernel, pointBounds.upper)
                             : std::nullopt;
      auto iterationExtent = iterationBounds.nonNegative && iterationBounds.upper
                                 ? nonNegativeExtentBounds(kernel, iterationBounds.upper)
                                 : std::nullopt;
      // The positive unit-step loop starts strictly after the point. Both
      // its start expression and final increment must remain representable.
      if (pointExtent && iterationExtent &&
          pointExtent->second <= std::numeric_limits<int64_t>::max() - *amount &&
          iterationExtent->second < std::numeric_limits<int64_t>::max())
        return true;
    }
    return false;
  };
  auto excludes = [&](const PhysicalAccessFootprint &pointAccess,
                      const PhysicalAccessFootprint &other) {
    for (auto [index, pointCoordinate] : llvm::enumerate(pointAccess.coordinates)) {
      Value point = scalar(pointCoordinate);
      if (!point)
        continue;
      auto axis = llvm::find(other.sourceAxes, pointAccess.sourceAxes[index]);
      if (axis == other.sourceAxes.end())
        continue;
      Value coordinate = other.coordinates[axis - other.sourceAxes.begin()];
      if (laterIteration(scalar(coordinate), point))
        return true;
      if (!other.validity)
        continue;
      auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
      auto predicateType = dyn_cast<FragmentType>(other.validity.getType());
      if (coordinateType) {
        if (!predicateType || coordinateType.getOwner() != predicateType.getOwner() ||
            !queryBroadcastProjection(coordinateType, predicateType).isExact())
          continue;
        // A repeated source axis could place the predicate and address on
        // different Cartesian occurrences. Require a unique target occurrence.
        bool unique = llvm::all_of(coordinateType.getAxisMaps(), [&](Attribute attr) {
          auto source = cast<AxisMapAttr>(attr);
          return llvm::count_if(predicateType.getAxisMaps(), [&](Attribute target) {
            auto mapping = cast<AxisMapAttr>(target);
            return sourceAxisIdentity(mapping) == sourceAxisIdentity(source) &&
                   mapping.getDimensionId() == source.getDimensionId();
          }) == 1;
        });
        if (!unique)
          continue;
      }
      std::function<bool(Value)> inspect = [&](Value predicate) {
        if (auto broadcast = predicate.getDefiningOp<BroadcastOp>()) {
          if (auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
              input && !queryBroadcastProjection(
                           input, cast<FragmentType>(predicate.getType())).isExact())
            return false;
          return inspect(broadcast.getValue());
        }
        if (auto splat = predicate.getDefiningOp<SplatOp>())
          return inspect(splat.getValue());
        if (auto conjunction = predicate.getDefiningOp<BinaryOp>();
            conjunction && conjunction.getOperatorKind() == BinaryOperator::LogicalAnd)
          return inspect(conjunction.getLhs()) || inspect(conjunction.getRhs());
        auto compare = predicate.getDefiningOp<CompareOp>();
        if (!compare || (compare.getPredicate() != ComparePredicate::Ne &&
                         compare.getPredicate() != ComparePredicate::Lt &&
                         compare.getPredicate() != ComparePredicate::Gt))
          return false;
        return (compare.getLhs() == coordinate && scalar(compare.getRhs()) == point) ||
               (compare.getRhs() == coordinate && scalar(compare.getLhs()) == point);
      };
      if (inspect(other.validity))
        return true;
    }
    return false;
  };
  return excludes(left, right) || excludes(right, left);
}

PhysicalAccessBoundaryFact
PhysicalProgramAnalysis::boundaryValidity(Operation *access,
                                          bool allowRangeGuards) {
  PhysicalAccessBoundaryFact result;
  PhysicalAccessFootprint accessFact = footprint(access);
  result.blockers = accessFact.blockers;
  auto view = accessFact.resource
                  ? dyn_cast<ViewType>(accessFact.resource.getType())
                  : ViewType();
  if (accessFact.state != PhysicalFactState::Exact ||
      accessFact.rangeState != PhysicalFactState::Exact || !view) {
    if (result.blockers.empty())
      appendUnique(result.blockers, access);
    return result;
  }
  if (!accessFact.validity) {
    result.state = PhysicalFactState::Exact;
    return result;
  }

  llvm::DenseSet<int64_t> boundaryAxes;
  std::optional<bool> boundedMembers;
  std::function<PhysicalFactState(Value)> analyze =
      [&](Value value) -> PhysicalFactState {
    if (auto broadcast = value.getDefiningOp<BroadcastOp>())
      return analyze(broadcast.getValue());
    if (auto splat = value.getDefiningOp<SplatOp>())
      return analyze(splat.getValue());
    if (auto reshape = value.getDefiningOp<ReshapeOp>())
      return analyze(reshape.getValue());
    if (auto transpose = value.getDefiningOp<TransposeOp>())
      return analyze(transpose.getValue());
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      return integer && integer.getType().isInteger(1) &&
                     integer.getValue().isOne()
                 ? PhysicalFactState::Exact
                 : PhysicalFactState::Unknown;
    }
    if (auto conjunction = value.getDefiningOp<BinaryOp>()) {
      Type element = conjunction.getResult().getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      bool logical =
          conjunction.getOperatorKind() == BinaryOperator::LogicalAnd;
      bool bitwiseI1 =
          conjunction.getOperatorKind() == BinaryOperator::BitwiseAnd &&
          element.isInteger(1);
      bool disjunction =
          conjunction.getOperatorKind() == BinaryOperator::LogicalOr ||
          (conjunction.getOperatorKind() == BinaryOperator::BitwiseOr &&
           element.isInteger(1));
      if (disjunction) {
        // Only this implication preserves the exact native boundary mask:
        // (upper(coordinate) < bound) OR (coordinate < bound). Merely proving
        // each arm safe would not make an arbitrary OR a conjunction of axes.
        for (auto [directValue, proofValue] :
             {std::pair{conjunction.getLhs(), conjunction.getRhs()},
              std::pair{conjunction.getRhs(), conjunction.getLhs()}}) {
          auto direct = directValue.getDefiningOp<CompareOp>();
          auto proof = proofValue.getDefiningOp<CompareOp>();
          if (!direct || !proof ||
              direct.getPredicate() != ComparePredicate::Lt ||
              proof.getPredicate() != ComparePredicate::Lt)
            continue;
          auto directLimit = queryLaunchExpression(direct.getRhs());
          auto proofLimit = queryLaunchExpression(proof.getRhs());
          if (!samePhysicalScalarExpression(direct.getRhs(), proof.getRhs()) &&
              !(directLimit && proofLimit && directLimit == proofLimit))
            continue;
          for (Value coordinate : accessFact.coordinates)
            if (derivesFromAccessCoordinate(direct.getLhs(), coordinate) &&
                isInclusiveCoordinateUpperBound(proof.getLhs(), coordinate))
              return analyze(directValue);
        }
        appendUnique(result.blockers, conjunction);
        return PhysicalFactState::Unknown;
      }
      if (!logical && !bitwiseI1) {
        appendUnique(result.blockers, conjunction);
        return PhysicalFactState::Unknown;
      }
      PhysicalFactState lhs = analyze(conjunction.getLhs());
      PhysicalFactState rhs = analyze(conjunction.getRhs());
      if (lhs == PhysicalFactState::Ambiguous ||
          rhs == PhysicalFactState::Ambiguous)
        return PhysicalFactState::Ambiguous;
      return lhs == PhysicalFactState::Exact &&
                     rhs == PhysicalFactState::Exact
                 ? PhysicalFactState::Exact
                 : PhysicalFactState::Unknown;
    }
    auto comparison = value.getDefiningOp<CompareOp>();
    bool upperComparison =
        comparison && comparison.getPredicate() == ComparePredicate::Lt;
    bool lowerComparison =
        comparison && comparison.getPredicate() == ComparePredicate::Ge &&
        integerConstant(comparison.getRhs()) == 0;
    if (!upperComparison && !lowerComparison) {
      appendUnique(result.blockers, value.getDefiningOp());
      return PhysicalFactState::Unknown;
    }

    std::optional<int64_t> matchedAxis;
    bool requiresBoundary = false;
    for (auto [coordinate, sourceAxis] :
         llvm::zip(accessFact.coordinates, accessFact.sourceAxes)) {
      if (!derivesFromAccessCoordinate(comparison.getLhs(), coordinate))
        continue;
      bool worksetViewBoundary = false;
      if (upperComparison && comparison->hasAttr(physicalTailAttr)) {
        Value accessCoordinate = stripIntegerIndexCasts(coordinate);
        auto range = accessCoordinate.getDefiningOp<MakeRangeOp>();
        auto workset =
            range ? range.getStart().getDefiningOp<WorksetCoordinateOp>()
                  : WorksetCoordinateOp();
        auto accessView = dyn_cast<ViewType>(accessFact.resource.getType());
        if (range && range->hasAttr(worksetCoordinateRangeAttr) &&
            !range->hasAttr(sourceSubregionAttr) && workset && accessView &&
            sourceAxisIdentity(range) ==
                PhysicalSourceAxis{workset.getSourceId(),
                                   workset.getSourceAxis(), false} &&
            sourceAxis >= 0 &&
            sourceAxis < static_cast<int64_t>(accessView.getRank())) {
          ArrayRef<int64_t> dimensions =
              accessView.getLayout().getDimensionIds().asArrayRef();
          worksetViewBoundary =
              dimensions[sourceAxis] ==
              static_cast<int64_t>(workset.getDimensionId());
        }
      }
      bool viewBoundary =
          lowerComparison || worksetViewBoundary ||
          matchesResourceExtent(comparison.getRhs(), accessFact.resource,
                                sourceAxis);
      bool exactRange =
          upperComparison && hasExactPhysicalRangeCoverage(
                                 coordinate, comparison.getRhs());
      if (upperComparison && !viewBoundary && !exactRange &&
          capacityCoversResourceExtent(comparison.getRhs(), accessFact.resource,
                                      sourceAxis)) {
        // A padded capacity is a weaker bound than the view extent. It is
        // redundant only when the original validity (or an unconditional
        // range fact) already confines active members to that view.
        if (!boundedMembers)
          boundedMembers = accessBounds(access).isExact();
        exactRange = *boundedMembers;
      }
      MakeRangeOp guardedRange;
      Value rangeBound;
      if (allowRangeGuards && upperComparison && !viewBoundary && !exactRange) {
        auto range = stripIntegerIndexCasts(coordinate)
                         .getDefiningOp<MakeRangeOp>();
        Value bound = stripScalarIdentity(comparison.getRhs());
        if (range && isUnitStepRange(range) && bound.getType().isIndex() &&
            sameScalarExpression(bound, range.getLogicalStop())) {
          guardedRange = range;
          rangeBound = bound;
        }
      }
      if (!viewBoundary && !exactRange && !guardedRange)
        continue;
      if (matchedAxis) {
        appendUnique(result.blockers, comparison);
        return PhysicalFactState::Ambiguous;
      }
      matchedAxis = sourceAxis;
      if (guardedRange &&
          !llvm::is_contained(result.rangeBounds,
                             std::make_pair(guardedRange, rangeBound)))
        result.rangeBounds.emplace_back(guardedRange, rangeBound);
      requiresBoundary =
          viewBoundary && !coordinateRangeWithinResource(
                              coordinate, accessFact.resource, sourceAxis);
    }
    if (!matchedAxis) {
      // A component range of a composed address may have its own logical tail
      // (for example group*width+channel). A whole-range guard discharges that
      // predicate without equating it to a boundary of the resource axis.
      if (allowRangeGuards)
        for (MakeRangeOp range : accessFact.ranges) {
          Value bound = stripScalarIdentity(range.getLogicalStop());
          if (!isUnitStepRange(range) || !bound.getType().isIndex() ||
              !isTailPredicate(value, {{range, bound}}))
            continue;
          if (!llvm::is_contained(result.rangeBounds,
                                  std::make_pair(range, bound)))
            result.rangeBounds.emplace_back(range, bound);
          return PhysicalFactState::Exact;
        }
      appendUnique(result.blockers, comparison);
      return PhysicalFactState::Unknown;
    }
    if (requiresBoundary)
      boundaryAxes.insert(*matchedAxis);
    return PhysicalFactState::Exact;
  };

  result.state = analyze(accessFact.validity);
  if (result.state == PhysicalFactState::Exact) {
    result.boundaryAxes.append(boundaryAxes.begin(), boundaryAxes.end());
    llvm::sort(result.boundaryAxes);
  }
  return result;
}

PhysicalAccessBoundsFact
PhysicalProgramAnalysis::accessBounds(Operation *access) {
  PhysicalAccessBoundsFact result;
  PhysicalAccessFootprint accessFact = footprint(access);
  result.blockers = accessFact.blockers;
  if (accessFact.state != PhysicalFactState::Exact || !accessFact.resource ||
      accessFact.coordinates.size() != accessFact.sourceAxes.size()) {
    appendUnique(result.blockers, access);
    return result;
  }

  unsigned rank = 0;
  if (auto view = dyn_cast<ViewType>(accessFact.resource.getType()))
    rank = view.getRank();
  else if (auto buffer = dyn_cast<BufferType>(accessFact.resource.getType()))
    rank = buffer.getShape().size();
  else if (auto fragment =
               dyn_cast<FragmentType>(accessFact.resource.getType()))
    rank = fragment.getShape().size();
  if (rank == 0 && !accessFact.coordinates.empty()) {
    appendUnique(result.blockers, access);
    return result;
  }

  struct AxisBounds {
    bool lower = false;
    bool upper = false;
  };
  SmallVector<AxisBounds> bounds(rank);
  SmallVector<bool> required(rank, false);
  for (auto [coordinate, sourceAxis] :
       llvm::zip(accessFact.coordinates, accessFact.sourceAxes)) {
    if (sourceAxis < 0 || sourceAxis >= static_cast<int64_t>(rank)) {
      appendUnique(result.blockers, access);
      return result;
    }
    required[sourceAxis] = true;
    if (coordinateRangeWithinResource(coordinate, accessFact.resource,
                                      sourceAxis)) {
      bounds[sourceAxis].lower = true;
      bounds[sourceAxis].upper = true;
    } else if (coordinateKnownNonNegative(coordinate)) {
      bounds[sourceAxis].lower = true;
    }
  }

  DominanceInfo dominance(kernel);
  kernel.walk([&](AssumeInBoundsOp assumption) {
    if (assumption.getResource() != accessFact.resource ||
        assumption.getAxis() >= rank ||
        !dominance.properlyDominates(assumption.getOperation(), access))
      return;
    for (auto [coordinate, sourceAxis] :
         llvm::zip(accessFact.coordinates, accessFact.sourceAxes)) {
      if (sourceAxis != static_cast<int64_t>(assumption.getAxis()) ||
          !derivesFromAccessCoordinate(assumption.getIndex(), coordinate))
        continue;
      bounds[sourceAxis].lower = true;
      bounds[sourceAxis].upper = true;
      if (!llvm::is_contained(result.assumedAxes, sourceAxis))
        result.assumedAxes.push_back(sourceAxis);
    }
  });

  std::function<bool(Value)> analyze = [&](Value value) -> bool {
    if (!value)
      return false;
    if (auto broadcast = value.getDefiningOp<BroadcastOp>())
      return analyze(broadcast.getValue());
    if (auto splat = value.getDefiningOp<SplatOp>())
      return analyze(splat.getValue());
    if (auto reshape = value.getDefiningOp<ReshapeOp>())
      return analyze(reshape.getValue());
    if (auto transpose = value.getDefiningOp<TransposeOp>())
      return analyze(transpose.getValue());
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      return integer && integer.getType().isInteger(1) &&
             integer.getValue().isZero();
    }
    if (auto conjunction = value.getDefiningOp<BinaryOp>()) {
      Type element = conjunction.getResult().getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      bool logical =
          conjunction.getOperatorKind() == BinaryOperator::LogicalAnd;
      bool bitwiseI1 =
          conjunction.getOperatorKind() == BinaryOperator::BitwiseAnd &&
          element.isInteger(1);
      if (logical || bitwiseI1) {
        bool lhsInactive = analyze(conjunction.getLhs());
        bool rhsInactive = analyze(conjunction.getRhs());
        return lhsInactive || rhsInactive;
      }
      bool disjunction =
          conjunction.getOperatorKind() == BinaryOperator::LogicalOr ||
          (conjunction.getOperatorKind() == BinaryOperator::BitwiseOr &&
           element.isInteger(1));
      if (disjunction) {
        // Each active arm must establish a bound. Facts from an enclosing
        // conjunction apply to both arms; facts discovered in one arm do not.
        auto before = bounds;
        bool lhsInactive = analyze(conjunction.getLhs());
        auto left = bounds;
        bounds = before;
        bool rhsInactive = analyze(conjunction.getRhs());
        for (unsigned axis = 0; axis < rank; ++axis) {
          bounds[axis].lower = before[axis].lower ||
              ((lhsInactive || left[axis].lower) &&
               (rhsInactive || bounds[axis].lower));
          bounds[axis].upper = before[axis].upper ||
              ((lhsInactive || left[axis].upper) &&
               (rhsInactive || bounds[axis].upper));
        }
        return lhsInactive && rhsInactive;
      }
      return false;
    }
    auto comparison = value.getDefiningOp<CompareOp>();
    if (!comparison)
      return false;
    for (auto [coordinate, sourceAxis] :
         llvm::zip(accessFact.coordinates, accessFact.sourceAxes)) {
      if (sourceAxis < 0 || sourceAxis >= static_cast<int64_t>(rank))
        continue;
      bool lhsCoordinate =
          derivesFromAccessCoordinate(comparison.getLhs(), coordinate);
      bool rhsCoordinate =
          derivesFromAccessCoordinate(comparison.getRhs(), coordinate);
      if ((comparison.getPredicate() == ComparePredicate::Ge &&
           lhsCoordinate && integerConstant(comparison.getRhs()) == 0) ||
          (comparison.getPredicate() == ComparePredicate::Le &&
           rhsCoordinate && integerConstant(comparison.getLhs()) == 0))
        bounds[sourceAxis].lower = true;
      if ((comparison.getPredicate() == ComparePredicate::Lt &&
           (lhsCoordinate || isInclusiveCoordinateUpperBound(
                                 comparison.getLhs(), coordinate)) &&
           upperBoundWithinResource(
                                comparison.getRhs(), accessFact.resource,
                                sourceAxis)) ||
          (comparison.getPredicate() == ComparePredicate::Gt &&
           (rhsCoordinate || isInclusiveCoordinateUpperBound(
                                 comparison.getRhs(), coordinate)) &&
           upperBoundWithinResource(
                                comparison.getLhs(), accessFact.resource,
                                sourceAxis)))
        bounds[sourceAxis].upper = true;
    }
    return false;
  };

  bool alwaysInactive = analyze(accessFact.validity);
  for (unsigned axis = 0; axis < rank; ++axis) {
    if (!required[axis] || alwaysInactive)
      continue;
    if (!bounds[axis].lower)
      result.missingLowerAxes.push_back(axis);
    if (!bounds[axis].upper)
      result.missingUpperAxes.push_back(axis);
    if (!bounds[axis].lower || !bounds[axis].upper)
      result.unprovenAxes.push_back(axis);
  }
  if (result.unprovenAxes.empty()) {
    result.state = PhysicalFactState::Exact;
    return result;
  }
  appendUnique(result.blockers,
               accessFact.validity ? accessFact.validity.getDefiningOp()
                                   : access);
  return result;
}

bool PhysicalProgramAnalysis::isProgramOwnedRange(MakeRangeOp range) const {
  return isExclusiveProgramRange(range, kernel);
}

bool PhysicalProgramAnalysis::hasDisjointWorkspaceSlices(Value buffer) const {
  auto type = dyn_cast<BufferType>(buffer.getType());
  if (!type)
    return false;
  SmallVector<AccessOpInterface> accesses;
  for (Operation *user : buffer.getUsers()) {
    if (isa<DimOp, AssumeInBoundsOp>(user))
      continue;
    auto access = dyn_cast<AccessOpInterface>(user);
    if (!access || (access.getAccessKind() != AccessKind::Load &&
                    access.getAccessKind() != AccessKind::Store))
      return false;
    accesses.push_back(access);
  }
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  // Every access must identify the whole program, including all grid axes.
  bool programPrefix = space && !space.empty() &&
                       type.getShape().size() >= space.size();
  for (AccessOpInterface access : accesses) {
    if (!programPrefix)
      break;
    ValueRange coordinates = access.getAccessCoordinates();
    ArrayRef<int64_t> sourceAxes = access.getAccessSourceAxes();
    for (unsigned axis = 0; axis < space.size(); ++axis) {
      auto position = llvm::find(sourceAxes, axis);
      if (position == sourceAxes.end() ||
          !isPrivateWorkspaceProgramIndex(
              coordinates[position - sourceAxes.begin()], buffer, access, axis)) {
        programPrefix = false;
        break;
      }
    }
  }
  if (programPrefix && !accesses.empty())
    return true;
  for (unsigned axis = 0; axis < type.getShape().size(); ++axis) {
    MakeRangeOp owner;
    bool consistent = true;
    for (AccessOpInterface access : accesses) {
      ValueRange coordinates = access.getAccessCoordinates();
      ArrayRef<int64_t> sourceAxes = access.getAccessSourceAxes();
      auto position = llvm::find(sourceAxes, axis);
      auto range = position == sourceAxes.end() ? MakeRangeOp() :
          stripRangeProjection(coordinates[position - sourceAxes.begin()])
              .getDefiningOp<MakeRangeOp>();
      if (!range || !isExclusiveProgramRange(range, kernel) ||
          queryLaunchExpression(range.getLogicalStop()) != type.getShape()[axis] ||
          (owner && (!sameLogicalRange(owner, range) ||
                     !sameScalarExpression(owner.getStart(), range.getStart()) ||
                     !sameScalarExpression(owner.getExtent(), range.getExtent())))) {
        consistent = false;
        break;
      }
      owner = range;
    }
    if (consistent && owner)
      return true;
  }
  return false;
}

} // namespace intent::gpu
