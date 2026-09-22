#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <algorithm>
#include <functional>
#include <optional>

using namespace mlir;

namespace intent::gpu {

Type scalarCallbackType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragment.getElementType();
  if (auto record = dyn_cast<RecordType>(type)) {
    SmallVector<Attribute> fields;
    for (Attribute field : record.getFieldTypes())
      fields.push_back(TypeAttr::get(
          scalarCallbackType(cast<TypeAttr>(field).getValue())));
    return RecordType::get(type.getContext(), record.getFieldNames(),
                           ArrayAttr::get(type.getContext(), fields),
                           record.getOwner());
  }
  return type;
}

LogicalResult scalarizeElementwiseCallback(Region &source, Region &target) {
  if (!target.empty())
    return target.getParentOp()->emitOpError(
        "scalar callback target region must be empty");
  Block &body = source.front();
  for (Operation &nested : body) {
    for (Value operand : nested.getOperands())
      if (operand.getParentBlock() != &body)
        return nested.emitOpError(
            "native collective callback cannot capture enclosing values");
    if (!isa<arith::ConstantOp, SplatOp, BroadcastOp, UnaryOp, BinaryOp,
             CompareOp, SelectOp, CastOp, BitcastOp, MakeRecordOp, ExtractOp,
             YieldOp>(nested))
      return nested.emitOpError(
          "native collective requires an elementwise scalarizable combine");
    if (auto broadcast = dyn_cast<BroadcastOp>(nested)) {
      auto sourceType = dyn_cast<FragmentType>(broadcast.getValue().getType());
      if (sourceType) {
        auto targetType = broadcast.getResult().getType();
        auto projection = queryAxisProjection(sourceType, targetType);
        if (sourceType.getShape() != targetType.getShape() ||
            !projection.isExact() ||
            llvm::any_of(llvm::enumerate(projection.targetToSource),
                         [](auto pair) {
                           return !pair.value() ||
                                  *pair.value() != pair.index();
                         }))
          return broadcast.emitOpError(
              "non-identity fragment broadcast in a collective requires prior lane-wise legalization");
      }
    }
  }

  Block *scalarBody = new Block();
  target.push_back(scalarBody);
  IRMapping mapping;
  for (BlockArgument argument : body.getArguments())
    mapping.map(argument, scalarBody->addArgument(
                             scalarCallbackType(argument.getType()),
                             argument.getLoc()));
  OpBuilder builder(target.getContext());
  builder.setInsertionPointToEnd(scalarBody);
  for (Operation &nested : body) {
    if (isa<SplatOp, BroadcastOp>(nested)) {
      mapping.map(nested.getResult(0), mapping.lookup(nested.getOperand(0)));
      continue;
    }
    Operation *cloned = builder.clone(nested, mapping);
    for (Value result : cloned->getResults())
      result.setType(scalarCallbackType(result.getType()));
  }
  return success();
}

bool variesWithIteration(Value value, scf::ForOp loop,
                        llvm::DenseMap<Value, bool> &known) {
  if (loop.isDefinedOutsideOfLoop(value))
    return false;
  if (auto found = known.find(value); found != known.end())
    return found->second;
  bool varies = true;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto nested = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    if (nested && nested != loop && argument == nested.getInductionVar())
      varies = variesWithIteration(nested.getLowerBound(), loop, known) ||
               variesWithIteration(nested.getUpperBound(), loop, known) ||
               variesWithIteration(nested.getStep(), loop, known);
  } else if (Operation *producer = value.getDefiningOp();
             producer && !producer->getNumRegions() &&
             canPredicateValueOperation(producer) &&
             !isa<LoadOp, GatherOp>(producer)) {
    varies = llvm::any_of(producer->getOperands(), [&](Value operand) {
      return variesWithIteration(operand, loop, known);
    });
  }
  known[value] = varies;
  return varies;
}

FailureOr<Value> materializeNonOverlappingView(func::FuncOp kernel,
                                             Value resource) {
  auto view = cast<ViewType>(resource.getType());
  SmallVector<OpFoldResult> strides;
  for (Attribute attribute : view.getLayout().getStrides()) {
    if (auto constant = dyn_cast<IntegerAttr>(attribute)) {
      strides.push_back(constant);
      continue;
    }
    auto name = dyn_cast<StringAttr>(attribute);
    Value stride;
    if (name)
      for (BlockArgument argument : kernel.getArguments())
        if (kernel.getArgAttrOfType<StringAttr>(argument.getArgNumber(),
                                               abiNameAttr) == name)
          stride = argument;
    if (!stride || !stride.getType().isIndex())
      return failure();
    strides.push_back(stride);
  }
  if (strides.empty())
    return failure();
  OpBuilder builder(&kernel.front(), kernel.front().begin());
  Location location = resource.getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  SmallVector<Value> values;
  for (OpFoldResult stride : strides)
    values.push_back(isa<Value>(stride) ? cast<Value>(stride) : Value(
        builder.create<arith::ConstantIndexOp>(
            location, cast<IntegerAttr>(cast<Attribute>(stride)).getInt())));
  Value valid = builder.create<CompareOp>(location, builder.getI1Type(),
                                         values.back(), zero, ComparePredicate::Gt);
  for (unsigned axis = values.size() - 1; axis > 0; --axis) {
    Value divisor = builder.create<BinaryOp>(location, builder.getIndexType(),
        values[axis], one, BinaryOperator::Maximum);
    Value slots = builder.create<BinaryOp>(location, builder.getIndexType(),
        values[axis - 1], divisor, BinaryOperator::FloorDivide);
    Value extent = builder.create<DimOp>(location, builder.getIndexType(),
                                         resource, axis);
    extent = builder.create<BinaryOp>(location, builder.getIndexType(), extent,
                                       one, BinaryOperator::Maximum);
    // Division avoids overflow in stride >= extent * next_stride. Every
    // accepted suffix has positive strides and fits inside its outer stride.
    Value separate = builder.create<CompareOp>(location, builder.getI1Type(),
                                               slots, extent, ComparePredicate::Ge);
    valid = builder.create<BinaryOp>(location, builder.getI1Type(), valid,
                                      separate, BinaryOperator::LogicalAnd);
  }
  return valid;
}

std::pair<uint64_t, int64_t> nextPhysicalAxisIdentities(func::FuncOp kernel) {
  uint64_t nextSource = 1;
  int64_t nextDimension = 1;
  AttrTypeWalker identities;
  identities.addWalk([&](AxisMapAttr axis) {
    nextSource = std::max(nextSource, axis.getSourceId() + 1);
    nextDimension = std::max(nextDimension, axis.getDimensionId() + 1);
  });
  identities.addWalk([&](PhysicalSourceAttr source) {
    nextSource = std::max(nextSource, source.getSourceId() + 1);
  });
  identities.addWalk([&](ViewType view) {
    nextSource = std::max(nextSource, view.getSourceId() + 1);
  });
  identities.addWalk([&](ViewLayoutAttr layout) {
    for (int64_t dimension : layout.getDimensionIds().asArrayRef())
      nextDimension = std::max(nextDimension, dimension + 1);
  });
  identities.addWalk([&](PhysicalExprAttr expression) {
    if (expression.getKind() == static_cast<uint32_t>(PhysicalExprKind::Dimension))
      nextDimension = std::max(nextDimension, expression.getValue() + 1);
  });
  kernel.walk([&](Operation *operation) {
    identities.walk(operation->getAttrDictionary());
    for (Type type : operation->getOperandTypes()) identities.walk(type);
    for (Type type : operation->getResultTypes()) identities.walk(type);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          identities.walk(argument.getType());
    for (StringRef name : {dimensionAttr, coverageDimensionAttr})
      if (auto dimension = operation->getAttrOfType<IntegerAttr>(name))
        nextDimension = std::max(nextDimension, dimension.getInt() + 1);
  });
  return {nextSource, nextDimension};
}

ParameterOp getOrCreatePhysicalParameter(
    func::FuncOp kernel, StringRef name, ParameterRole role,
    ParameterCategory category, uint32_t elementBitWidth,
    ArrayRef<int64_t> candidates) {
  ParameterOp existing;
  bool ambiguous = false;
  kernel.walk([&](ParameterOp parameter) {
    if (parameter.getParameter().getName().getValue() != name)
      return;
    if (existing && existing != parameter)
      ambiguous = true;
    else
      existing = parameter;
  });
  if (ambiguous) {
    kernel.emitError("physical parameter symbol has multiple declarations")
        << "; name=" << name;
    return ParameterOp();
  }
  auto expectedCandidates =
      DenseI64ArrayAttr::get(kernel.getContext(), candidates);
  if (existing) {
    ParameterAttr schema = existing.getParameter();
    if (schema.getRole() != static_cast<uint32_t>(role) ||
        schema.getCategory() != static_cast<uint32_t>(category) ||
        schema.getCandidates() != expectedCandidates) {
      existing.emitOpError(
          "physical parameter symbol is reused with an incompatible decision domain")
          << "; name=" << name << "; existing_role=" << schema.getRole()
          << "; requested_role=" << static_cast<uint32_t>(role)
          << "; existing_category=" << schema.getCategory()
          << "; requested_category=" << static_cast<uint32_t>(category)
          << "; existing_candidates=" << schema.getCandidates()
          << "; requested_candidates=" << expectedCandidates;
      return ParameterOp();
    }
    uint32_t aggregateWidth =
        std::max(schema.getElementBitWidth(), elementBitWidth);
    if (aggregateWidth != schema.getElementBitWidth())
      existing->setAttr(
          "parameter",
          ParameterAttr::get(kernel.getContext(), schema.getName(),
                             schema.getRole(), schema.getCategory(),
                             aggregateWidth, schema.getCandidates()));
    return existing;
  }
  OpBuilder builder(&kernel.getBody().front(), kernel.getBody().front().begin());
  auto schema = ParameterAttr::get(
      kernel.getContext(), builder.getStringAttr(name),
      static_cast<uint32_t>(role), static_cast<uint32_t>(category),
      elementBitWidth, expectedCandidates);
  return builder.create<ParameterOp>(kernel.getLoc(), builder.getIndexType(),
                                     schema);
}

namespace {

void collectParameterSymbols(Attribute attribute, llvm::StringSet<> &symbols);

PhysicalExprAttr multiplyExtent(PhysicalExprAttr lhs, PhysicalExprAttr rhs) {
  auto leftKind = static_cast<PhysicalExprKind>(lhs.getKind());
  auto rightKind = static_cast<PhysicalExprKind>(rhs.getKind());
  if (leftKind == PhysicalExprKind::Constant && lhs.getValue() == 1)
    return rhs;
  if (rightKind == PhysicalExprKind::Constant && rhs.getValue() == 1)
    return lhs;
  if (leftKind == PhysicalExprKind::Constant &&
      rightKind == PhysicalExprKind::Constant)
    return PhysicalExprAttr::get(
        lhs.getContext(), static_cast<uint32_t>(PhysicalExprKind::Constant),
        lhs.getValue() * rhs.getValue(), StringAttr::get(lhs.getContext()),
        ArrayAttr::get(lhs.getContext(), {}));
  return PhysicalExprAttr::get(
      lhs.getContext(), static_cast<uint32_t>(PhysicalExprKind::Multiply), 0,
      StringAttr::get(lhs.getContext()),
      ArrayAttr::get(lhs.getContext(), {lhs, rhs}));
}

PhysicalExprAttr productExtent(MLIRContext *context,
                               ArrayRef<Attribute> shape,
                               ArrayRef<int64_t> axes, unsigned prefix) {
  PhysicalExprAttr product = PhysicalExprAttr::get(
      context, static_cast<uint32_t>(PhysicalExprKind::Constant), 1,
      StringAttr::get(context), ArrayAttr::get(context, {}));
  for (int64_t axis : axes)
    product = multiplyExtent(
        product, cast<PhysicalExprAttr>(shape[prefix + axis]));
  return product;
}

PhysicalExprAttr replaceParameterSymbol(PhysicalExprAttr expression,
                                        StringAttr previous,
                                        StringAttr replacement) {
  if (expression.getKind() ==
          static_cast<uint32_t>(PhysicalExprKind::Parameter) &&
      expression.getSymbol() == previous)
    return PhysicalExprAttr::get(
        expression.getContext(), expression.getKind(), expression.getValue(),
        replacement, expression.getOperands());
  SmallVector<Attribute> operands;
  bool changed = false;
  for (Attribute operand : expression.getOperands()) {
    auto rewritten = replaceParameterSymbol(
        cast<PhysicalExprAttr>(operand), previous, replacement);
    operands.push_back(rewritten);
    changed |= rewritten != operand;
  }
  return changed
             ? PhysicalExprAttr::get(
                   expression.getContext(), expression.getKind(),
                   expression.getValue(), expression.getSymbol(),
                   ArrayAttr::get(expression.getContext(), operands))
             : expression;
}

Attribute replaceParameterSymbol(Attribute attribute, StringAttr previous,
                                 StringAttr replacement);

Type replaceParameterSymbol(Type type, StringAttr previous,
                            StringAttr replacement) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    auto shape = cast<ArrayAttr>(replaceParameterSymbol(
        fragment.getShape(), previous, replacement));
    if (shape == fragment.getShape())
      return type;
    return FragmentType::get(type.getContext(), fragment.getElementType(), shape,
                             fragment.getAxisMaps(), fragment.getValidity(),
                             fragment.getOwner());
  }
  if (auto buffer = dyn_cast<BufferType>(type)) {
    auto shape = cast<ArrayAttr>(replaceParameterSymbol(
        buffer.getShape(), previous, replacement));
    Type element = replaceParameterSymbol(buffer.getElementType(), previous,
                                          replacement);
    if (shape == buffer.getShape() && element == buffer.getElementType())
      return type;
    return BufferType::get(type.getContext(), element, shape, buffer.getScope(),
                           buffer.getInstance(), buffer.getOwner(),
                           buffer.getInitialization(), buffer.getLifetime(),
                           buffer.getVisibility(), buffer.getWorkspace());
  }
  if (auto view = dyn_cast<ViewType>(type)) {
    ViewLayoutAttr layout = view.getLayout();
    auto extents = cast<ArrayAttr>(replaceParameterSymbol(
        layout.getExtents(), previous, replacement));
    auto strides = cast<ArrayAttr>(replaceParameterSymbol(
        layout.getStrides(), previous, replacement));
    if (extents == layout.getExtents() && strides == layout.getStrides())
      return type;
    auto rewrittenLayout = ViewLayoutAttr::get(
        type.getContext(), extents, layout.getDimensionIds(),
        layout.getHasStrides(), strides, layout.getAlias(), layout.getNoalias());
    return ViewType::get(type.getContext(), view.getElementType(), view.getRank(),
                         view.getAccess(), view.getAbiIndex(), view.getSourceId(),
                         rewrittenLayout);
  }
  if (auto record = dyn_cast<RecordType>(type)) {
    auto fields = cast<ArrayAttr>(replaceParameterSymbol(
        record.getFieldTypes(), previous, replacement));
    if (fields == record.getFieldTypes())
      return type;
    return RecordType::get(type.getContext(), record.getFieldNames(), fields,
                           record.getOwner());
  }
  return type;
}

Attribute replaceParameterSymbol(Attribute attribute, StringAttr previous,
                                 StringAttr replacement) {
  if (!attribute)
    return attribute;
  if (auto expression = dyn_cast<PhysicalExprAttr>(attribute))
    return replaceParameterSymbol(expression, previous, replacement);
  if (auto array = dyn_cast<ArrayAttr>(attribute)) {
    SmallVector<Attribute> elements;
    bool changed = false;
    for (Attribute element : array) {
      Attribute rewritten =
          replaceParameterSymbol(element, previous, replacement);
      elements.push_back(rewritten);
      changed |= rewritten != element;
    }
    return changed ? Attribute(ArrayAttr::get(attribute.getContext(), elements))
                   : attribute;
  }
  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute)) {
    SmallVector<NamedAttribute> elements;
    bool changed = false;
    for (NamedAttribute element : dictionary) {
      Attribute rewritten = replaceParameterSymbol(
          element.getValue(), previous, replacement);
      elements.emplace_back(element.getName(), rewritten);
      changed |= rewritten != element.getValue();
    }
    return changed
               ? Attribute(DictionaryAttr::get(attribute.getContext(), elements))
               : attribute;
  }
  if (auto typed = dyn_cast<TypeAttr>(attribute)) {
    Type rewritten =
        replaceParameterSymbol(typed.getValue(), previous, replacement);
    return rewritten != typed.getValue() ? Attribute(TypeAttr::get(rewritten))
                                         : attribute;
  }
  return attribute;
}

using AxisSelector = llvm::function_ref<bool(AxisMapAttr)>;

FragmentType replaceExtent(FragmentType source, AxisSelector selects,
                           ArrayRef<Attribute> previousExtents,
                           PhysicalExprAttr extent) {
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  bool changed = false;
  for (auto [axis, attribute] : llvm::enumerate(source.getAxisMaps())) {
    if (!selects(cast<AxisMapAttr>(attribute)) ||
        !llvm::is_contained(previousExtents, source.getShape()[axis]))
      continue;
    shape[axis] = extent;
    changed = true;
  }
  if (!changed)
    return source;
  return FragmentType::get(
      source.getContext(), source.getElementType(),
      ArrayAttr::get(source.getContext(), shape), source.getAxisMaps(),
      source.getValidity(), source.getOwner());
}

Type replaceExtent(Type source, AxisSelector selects,
                   ArrayRef<Attribute> previousExtents,
                   PhysicalExprAttr extent) {
  if (auto fragment = dyn_cast<FragmentType>(source))
    return replaceExtent(fragment, selects, previousExtents, extent);
  auto record = dyn_cast<RecordType>(source);
  if (!record)
    return source;
  SmallVector<Attribute> fields;
  bool changed = false;
  for (Attribute attribute : record.getFieldTypes()) {
    Type field = cast<TypeAttr>(attribute).getValue();
    Type replacement = replaceExtent(field, selects, previousExtents, extent);
    fields.push_back(TypeAttr::get(replacement));
    changed |= replacement != field;
  }
  if (!changed)
    return source;
  return RecordType::get(source.getContext(), record.getFieldNames(),
                         ArrayAttr::get(source.getContext(), fields),
                         record.getOwner());
}

bool carriesExtent(Type type, AxisSelector selects,
                   ArrayRef<Attribute> extents) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps()))
      if (selects(cast<AxisMapAttr>(attribute)) &&
          llvm::is_contained(extents, fragment.getShape()[axis]))
        return true;
    return false;
  }
  auto record = dyn_cast<RecordType>(type);
  if (!record)
    return false;
  return llvm::any_of(record.getFieldTypes(), [&](Attribute attribute) {
    return carriesExtent(cast<TypeAttr>(attribute).getValue(), selects, extents);
  });
}

bool carriesSelectedAxis(Type type, AxisSelector selects) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
      return selects(cast<AxisMapAttr>(attribute));
    });
  auto record = dyn_cast<RecordType>(type);
  return record && llvm::any_of(record.getFieldTypes(), [&](Attribute attribute) {
           return carriesSelectedAxis(cast<TypeAttr>(attribute).getValue(),
                                      selects);
         });
}

void collectSelectedExtents(Type type, AxisSelector selects,
                            SmallVectorImpl<Attribute> &extents) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps()))
      if (selects(cast<AxisMapAttr>(attribute)) &&
          !llvm::is_contained(extents, fragment.getShape()[axis]))
        extents.push_back(fragment.getShape()[axis]);
    return;
  }
  if (auto record = dyn_cast<RecordType>(type))
    for (Attribute attribute : record.getFieldTypes())
      collectSelectedExtents(cast<TypeAttr>(attribute).getValue(), selects,
                             extents);
}

bool preservesIntroducedUnitAxis(Value value, AxisSelector selects) {
  auto reshape = value.getDefiningOp<ReshapeOp>();
  auto result = dyn_cast<FragmentType>(value.getType());
  if (!result)
    return false;
  std::optional<unsigned> axis;
  for (Attribute attribute : result.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (!selects(mapping))
      continue;
    if (axis)
      return false;
    axis = mapping.getFragmentAxis();
  }
  if (!axis)
    return false;
  auto extent = cast<PhysicalExprAttr>(result.getShape()[*axis]);
  bool unit = extent.getKind() ==
                  static_cast<uint32_t>(PhysicalExprKind::Constant) &&
              extent.getValue() == 1;
  if (!unit)
    return false;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (kernel && PhysicalProgramAnalysis(kernel)
                    .axisRealization(value, *axis)
                    .constructionScalarSeed)
    return false;
  if (!reshape) {
    bool expanded = false;
    for (Operation *user : value.getUsers()) {
      auto broadcast = dyn_cast<BroadcastOp>(user);
      if (!broadcast || broadcast.getValue() != value) {
        if (llvm::any_of(user->getResultTypes(), [&](Type type) {
              return carriesSelectedAxis(type, selects);
            }))
          return false;
        continue;
      }
      auto target = dyn_cast<FragmentType>(broadcast.getResult().getType());
      BroadcastProjection projection =
          target ? queryAxisProjection(result, target) : BroadcastProjection();
      if (!target || !projection.isExact())
        return false;
      std::optional<unsigned> targetAxis;
      for (auto [candidate, sourceAxis] :
           llvm::enumerate(projection.targetToSource))
        if (sourceAxis && *sourceAxis == *axis)
          targetAxis = candidate;
      if (!targetAxis)
        return false;
      expanded |= target.getShape()[*targetAxis] != result.getShape()[*axis];
    }
    return expanded;
  }
  return false;
}

bool isIntroducedReshapeUnitAxis(Value value, unsigned fragmentAxis) {
  auto reshape = value.getDefiningOp<ReshapeOp>();
  auto result = dyn_cast<FragmentType>(value.getType());
  if (!reshape || !result || fragmentAxis >= result.getShape().size())
    return false;
  auto extent = cast<PhysicalExprAttr>(result.getShape()[fragmentAxis]);
  if (extent.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Constant) ||
      extent.getValue() != 1)
    return false;
  unsigned logicalResultRank = 0;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = dyn_cast<ReshapeGroupAttr>(attribute);
    if (!group || group.getResultAxes().empty())
      continue;
    logicalResultRank =
        std::max(logicalResultRank,
                 static_cast<unsigned>(
                     group.getResultAxes().asArrayRef().back() + 1));
  }
  if (logicalResultRank > result.getShape().size())
    return false;
  unsigned resultPrefix = result.getShape().size() - logicalResultRank;
  if (fragmentAxis < resultPrefix)
    return false;
  int64_t logicalAxis = fragmentAxis - resultPrefix;
  return llvm::any_of(reshape.getReassociation(), [&](Attribute attribute) {
    auto group = dyn_cast<ReshapeGroupAttr>(attribute);
    return group && group.getSourceAxes().empty() &&
           llvm::is_contained(group.getResultAxes().asArrayRef(), logicalAxis);
  });
}

bool selectsSegmentAxis(Type type, uint64_t axis, AxisSelector selects) {
  auto fragment = dyn_cast<FragmentType>(type);
  return fragment && axis < fragment.getAxisMaps().size() &&
         selects(cast<AxisMapAttr>(fragment.getAxisMaps()[axis]));
}

void appendStructuredParentRelations(BlockArgument argument,
                                     SmallVectorImpl<Value> &worklist,
                                     AxisSelector selects) {
  Block *block = argument.getOwner();
  Region *region = block->getParent();
  Operation *parent = region ? region->getParentOp() : nullptr;
  unsigned index = argument.getArgNumber();
  if (auto fold = dyn_cast_or_null<RegionFoldOp>(parent)) {
    unsigned sources = fold.getSourceCount();
    unsigned identities = fold.getIdentityCount();
    if (region == &fold.getSummarize()) {
      if (index < sources) {
        if (!selectsSegmentAxis(argument.getType(), fold.getAxis(), selects))
          worklist.push_back(fold.getInputs()[index]);
        return;
      }
      unsigned capture = index - sources;
      worklist.push_back(fold.getInputs()[sources + identities + capture]);
      return;
    }
    if (region == &fold.getCombine()) {
      unsigned component = index % identities;
      worklist.push_back(fold.getInputs()[sources + component]);
      worklist.push_back(fold.getResults()[component]);
    }
    return;
  }
  auto scan = dyn_cast_or_null<RegionScanOp>(parent);
  if (!scan)
    return;
  unsigned sources = scan.getSourceCount();
  unsigned identities = scan.getIdentityCount();
  unsigned states = scan.getStateCount();
  unsigned outputs = scan.getOutputCount();
  unsigned stateOffset = sources + identities;
  unsigned captureOffset = stateOffset + states;
  if (region == &scan.getSummarize()) {
    if (index < sources) {
      if (!selectsSegmentAxis(argument.getType(), scan.getAxis(), selects))
        worklist.push_back(scan.getInputs()[index]);
      return;
    }
    worklist.push_back(scan.getInputs()[captureOffset + index - sources]);
    return;
  }
  if (region == &scan.getCombine()) {
    worklist.push_back(scan.getInputs()[sources + index % identities]);
    return;
  }
  if (region == &scan.getApply()) {
    if (index < identities) {
      worklist.push_back(scan.getInputs()[sources + index]);
      return;
    }
    unsigned state = index - identities;
    worklist.push_back(scan.getInputs()[stateOffset + state]);
    worklist.push_back(scan.getResults()[outputs + state]);
    return;
  }
  if (region != &scan.getEmit())
    return;
  if (index < sources) {
    if (!selectsSegmentAxis(argument.getType(), scan.getAxis(), selects))
      worklist.push_back(scan.getInputs()[index]);
    return;
  }
  if (index < sources + states) {
    unsigned state = index - sources;
    worklist.push_back(scan.getInputs()[stateOffset + state]);
    worklist.push_back(scan.getResults()[outputs + state]);
    return;
  }
  worklist.push_back(
      scan.getInputs()[captureOffset + index - sources - states]);
}

void appendStructuredChildRelations(Operation *user, Value value,
                                    SmallVectorImpl<Value> &worklist,
                                    AxisSelector selects) {
  if (auto fold = dyn_cast<RegionFoldOp>(user)) {
    unsigned sources = fold.getSourceCount();
    unsigned identities = fold.getIdentityCount();
    for (auto [index, operand] : llvm::enumerate(fold.getInputs())) {
      if (operand != value)
        continue;
      if (index < sources) {
        BlockArgument slice =
            fold.getSummarize().front().getArgument(index);
        if (!selectsSegmentAxis(slice.getType(), fold.getAxis(), selects))
          worklist.push_back(slice);
        continue;
      }
      if (index < sources + identities) {
        unsigned component = index - sources;
        worklist.push_back(fold.getResults()[component]);
        Block &combine = fold.getCombine().front();
        worklist.push_back(combine.getArgument(component));
        worklist.push_back(combine.getArgument(identities + component));
        continue;
      }
      Block &summarize = fold.getSummarize().front();
      unsigned capture = index - sources - identities;
      worklist.push_back(summarize.getArgument(sources + capture));
    }
    return;
  }
  auto scan = dyn_cast<RegionScanOp>(user);
  if (!scan)
    return;
  unsigned sources = scan.getSourceCount();
  unsigned identities = scan.getIdentityCount();
  unsigned states = scan.getStateCount();
  unsigned outputs = scan.getOutputCount();
  unsigned stateOffset = sources + identities;
  unsigned captureOffset = stateOffset + states;
  for (auto [index, operand] : llvm::enumerate(scan.getInputs())) {
    if (operand != value)
      continue;
    if (index < sources) {
      BlockArgument summarize =
          scan.getSummarize().front().getArgument(index);
      BlockArgument emit = scan.getEmit().front().getArgument(index);
      if (!selectsSegmentAxis(summarize.getType(), scan.getAxis(), selects)) {
        worklist.push_back(summarize);
        worklist.push_back(emit);
      }
      continue;
    }
    if (index < stateOffset) {
      unsigned component = index - sources;
      Block &combine = scan.getCombine().front();
      Block &apply = scan.getApply().front();
      worklist.push_back(combine.getArgument(component));
      worklist.push_back(combine.getArgument(identities + component));
      worklist.push_back(apply.getArgument(component));
      continue;
    }
    if (index < captureOffset) {
      unsigned state = index - stateOffset;
      worklist.push_back(scan.getResults()[outputs + state]);
      worklist.push_back(
          scan.getApply().front().getArgument(identities + state));
      worklist.push_back(scan.getEmit().front().getArgument(sources + state));
      continue;
    }
    unsigned capture = index - captureOffset;
    worklist.push_back(
        scan.getSummarize().front().getArgument(sources + capture));
    worklist.push_back(
        scan.getEmit().front().getArgument(sources + states + capture));
  }
}

void appendStructuredResultRelations(Value value,
                                     SmallVectorImpl<Value> &worklist,
                                     AxisSelector selects) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return;
  unsigned index = result.getResultNumber();
  if (auto reduce = dyn_cast<ReduceOp>(result.getOwner())) {
    unsigned sources = reduce.getSourceCount();
    unsigned identities = reduce.getIdentityCount();
    Value source = reduce.getInputs()[index];
    auto fragment = dyn_cast<FragmentType>(source.getType());
    // Only retained axes connect a reduction result back to its source.
    if (fragment && llvm::none_of(reduce.getAxes(), [&](int64_t axis) {
          return selects(cast<AxisMapAttr>(fragment.getAxisMaps()[axis]));
        }))
      worklist.push_back(source);
    worklist.push_back(reduce.getInputs()[sources + index]);
    worklist.push_back(reduce.getCombine().front().getArgument(index));
    worklist.push_back(
        reduce.getCombine().front().getArgument(identities + index));
    worklist.push_back(cast<YieldOp>(reduce.getCombine().front().getTerminator())
                           .getValues()[index]);
    return;
  }
  if (auto fold = dyn_cast<RegionFoldOp>(result.getOwner())) {
    unsigned sources = fold.getSourceCount();
    unsigned identities = fold.getIdentityCount();
    worklist.push_back(fold.getInputs()[sources + index]);
    worklist.push_back(fold.getCombine().front().getArgument(index));
    worklist.push_back(
        fold.getCombine().front().getArgument(identities + index));
    worklist.push_back(
        cast<YieldOp>(fold.getSummarize().front().getTerminator())
            .getValues()[index]);
    worklist.push_back(cast<YieldOp>(fold.getCombine().front().getTerminator())
                           .getValues()[index]);
    return;
  }
  auto scan = dyn_cast<RegionScanOp>(result.getOwner());
  if (!scan)
    return;
  unsigned outputs = scan.getOutputCount();
  if (index < outputs) {
    worklist.push_back(
        cast<YieldOp>(scan.getEmit().front().getTerminator()).getValues()[index]);
    return;
  }
  unsigned state = index - outputs;
  unsigned sources = scan.getSourceCount();
  unsigned identities = scan.getIdentityCount();
  unsigned stateOffset = sources + identities;
  worklist.push_back(scan.getInputs()[stateOffset + state]);
  worklist.push_back(
      scan.getApply().front().getArgument(identities + state));
  worklist.push_back(scan.getEmit().front().getArgument(sources + state));
  worklist.push_back(
      cast<YieldOp>(scan.getApply().front().getTerminator()).getValues()[state]);
}

FailureOr<Value> projectPredicate(OpBuilder &builder, Location location,
                                  Value predicate, FragmentType target,
                                  unsigned axis) {
  auto base = dyn_cast<FragmentType>(predicate.getType());
  if (!base || base.getShape().size() != 1 || axis >= target.getShape().size())
    return failure();
  SmallVector<Attribute> shape(
      target.getShape().size(),
      PhysicalExprAttr::get(
          target.getContext(),
          static_cast<uint32_t>(PhysicalExprKind::Constant), 1,
          StringAttr::get(target.getContext()),
          ArrayAttr::get(target.getContext(), {})));
  shape[axis] = base.getShape()[0];
  auto reshaped = FragmentType::get(
      target.getContext(), builder.getI1Type(),
      ArrayAttr::get(target.getContext(), shape), target.getAxisMaps(),
      target.getValidity(), target.getOwner());
  Value result = predicate;
  if (base != reshaped) {
    SmallVector<Attribute> groups;
    for (unsigned resultAxis = 0; resultAxis < shape.size(); ++resultAxis) {
      SmallVector<int64_t> sourceAxes;
      if (resultAxis == axis)
        sourceAxes.push_back(0);
      groups.push_back(ReshapeGroupAttr::get(
          target.getContext(), builder.getDenseI64ArrayAttr(sourceAxes),
          builder.getDenseI64ArrayAttr({static_cast<int64_t>(resultAxis)})));
    }
    result = builder.create<ReshapeOp>(location, reshaped, result,
                                      builder.getArrayAttr(groups));
  }
  auto projected = FragmentType::get(
      target.getContext(), builder.getI1Type(), target.getShape(),
      target.getAxisMaps(), target.getValidity(), target.getOwner());
  if (reshaped != projected)
    result = builder.create<BroadcastOp>(location, projected, result);
  return result;
}

FailureOr<Value> zeroFill(OpBuilder &builder, Location location,
                          FragmentType type) {
  TypedAttr zero;
  if (auto integer = dyn_cast<IntegerType>(type.getElementType()))
    zero = builder.getIntegerAttr(integer, 0);
  else if (auto floating = dyn_cast<FloatType>(type.getElementType()))
    zero = builder.getFloatAttr(floating, 0.0);
  else if (isa<IndexType>(type.getElementType()))
    zero = builder.getIndexAttr(0);
  if (!zero)
    return failure();
  FailureOr<Value> scalar =
      materializeScalarConstant(builder, location, zero, type.getElementType());
  if (failed(scalar))
    return failure();
  return Value(builder.create<BroadcastOp>(location, type, *scalar));
}

void collectParameterSymbols(Type type, llvm::StringSet<> &symbols) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    collectParameterSymbols(fragment.getShape(), symbols);
    return;
  }
  if (auto buffer = dyn_cast<BufferType>(type)) {
    collectParameterSymbols(buffer.getShape(), symbols);
    return;
  }
  if (auto view = dyn_cast<ViewType>(type)) {
    collectParameterSymbols(view.getLayout().getExtents(), symbols);
    return;
  }
  if (auto record = dyn_cast<RecordType>(type))
    collectParameterSymbols(record.getFieldTypes(), symbols);
}

void collectParameterSymbols(Attribute attribute, llvm::StringSet<> &symbols) {
  if (!attribute)
    return;
  if (auto expression = dyn_cast<PhysicalExprAttr>(attribute)) {
    if (expression.getKind() ==
        static_cast<uint32_t>(PhysicalExprKind::Parameter))
      symbols.insert(expression.getSymbol().getValue());
    for (Attribute operand : expression.getOperands())
      collectParameterSymbols(operand, symbols);
    return;
  }
  if (auto array = dyn_cast<ArrayAttr>(attribute)) {
    for (Attribute element : array)
      collectParameterSymbols(element, symbols);
    return;
  }
  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute)) {
    for (NamedAttribute element : dictionary)
      collectParameterSymbols(element.getValue(), symbols);
    return;
  }
  if (auto typed = dyn_cast<TypeAttr>(attribute))
    collectParameterSymbols(typed.getValue(), symbols);
}

} // namespace

FailureOr<Value> materializeScalarConstant(OpBuilder &builder,
                                           Location location, Attribute value,
                                           Type resultType) {
  if (isa<IndexType>(resultType)) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer)
      return failure();
    return Value(
        builder.create<arith::ConstantIndexOp>(location, integer.getInt()));
  }
  if (auto integerType = dyn_cast<IntegerType>(resultType)) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer)
      return failure();
    auto signless = IntegerType::get(builder.getContext(), integerType.getWidth());
    llvm::APInt bits = integer.getValue().sextOrTrunc(integerType.getWidth());
    Value raw = builder.create<arith::ConstantOp>(
        location, signless, IntegerAttr::get(signless, bits));
    if (integerType.isSignless())
      return raw;
    return Value(builder.create<CastOp>(location, integerType, raw));
  }
  if (auto floatType = dyn_cast<FloatType>(resultType)) {
    auto floating = dyn_cast<FloatAttr>(value);
    if (!floating)
      return failure();
    return Value(builder.create<arith::ConstantOp>(
        location, floatType,
        FloatAttr::get(floatType, floating.getValueAsDouble())));
  }
  return failure();
}

static FailureOr<Value> projectFragmentValue(OpBuilder &builder,
                                             Location location, Value value,
                                             FragmentType target) {
  if (value.getType() == target)
    return value;
  Type element = value.getType();
  auto source = dyn_cast<FragmentType>(element);
  if (source)
    element = source.getElementType();
  if (!isa<IntegerType, FloatType, IndexType>(element) ||
      element != target.getElementType())
    return failure();
  if (auto extract = value.getDefiningOp<ExtractOp>())
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>())
      return projectFragmentValue(builder, location,
                                  record.getFields()[extract.getField()], target);
  Value scalar = value;
  while (isa<FragmentType>(scalar.getType())) {
    UniformExpression expression = describeUniformValue(scalar);
    if (expression.kind != UniformKind::Forward || expression.operands.size() != 1)
      break;
    scalar = expression.operands.front();
  }
  if (scalar.getType() == element)
    return Value(builder.create<SplatOp>(location, target, scalar));
  if (source)
    if (auto constant = dyn_cast_or_null<TypedAttr>(
            UniformValueAnalysis(describeUniformValue).evaluate(value));
        constant && constant.getType() == element) {
      Value scalar = builder.create<arith::ConstantOp>(location, element, constant);
      return Value(builder.create<SplatOp>(location, target, scalar));
    }
  Operation *projection = nullptr;
  if (!source) {
    projection = builder.create<SplatOp>(location, target, value);
  } else if (auto splat = value.getDefiningOp<SplatOp>()) {
    projection = builder.create<SplatOp>(location, target, splat.getValue());
  } else if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
    FailureOr<Value> projected =
        projectFragmentValue(builder, location, broadcast.getValue(), target);
    if (succeeded(projected))
      return *projected;
  } else if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
    auto input = cast<FragmentType>(reshape.getValue().getType());
    unsigned inputRank = 0, resultRank = 0;
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = cast<ReshapeGroupAttr>(attribute);
      inputRank += group.getSourceAxes().size();
      resultRank += group.getResultAxes().size();
    }
    bool projects = inputRank <= input.getShape().size() &&
                    resultRank <= source.getShape().size() &&
                    source.getShape().size() == target.getShape().size();
    BroadcastProjection relation = queryAxisProjection(source, target);
    projects &= relation.isExact() && llvm::all_of(
        llvm::enumerate(relation.targetToSource), [](auto item) {
          return item.value() && *item.value() == item.index();
        });
    SmallVector<Attribute> inputShape(input.getShape().begin(),
                                       input.getShape().end());
    if (projects) {
      unsigned inputPrefix = input.getShape().size() - inputRank;
      unsigned resultPrefix = source.getShape().size() - resultRank;
      projects = inputPrefix == resultPrefix;
      if (projects)
        for (unsigned axis = 0; axis < inputPrefix; ++axis)
          inputShape[axis] = target.getShape()[axis];
      for (Attribute attribute : reshape.getReassociation()) {
        auto group = cast<ReshapeGroupAttr>(attribute);
        bool changed = llvm::any_of(group.getResultAxes().asArrayRef(),
                                    [&](int64_t axis) {
          return source.getShape()[resultPrefix + axis] !=
                 target.getShape()[resultPrefix + axis];
        });
        if (!changed)
          continue;
        if (group.getSourceAxes().size() != 1 ||
            group.getResultAxes().size() != 1) {
          projects = false;
          break;
        }
        inputShape[inputPrefix + group.getSourceAxes()[0]] =
            target.getShape()[resultPrefix + group.getResultAxes()[0]];
      }
    }
    if (projects) {
      auto inputTarget = FragmentType::get(
          target.getContext(), input.getElementType(), builder.getArrayAttr(inputShape),
          input.getAxisMaps(), target.getValidity(), target.getOwner());
      FailureOr<Value> projected =
          projectFragmentValue(builder, location, reshape.getValue(), inputTarget);
      if (succeeded(projected))
        projection = builder.create<ReshapeOp>(
            location, target, *projected, reshape.getReassociation());
    }
  } else if (auto reduce = value.getDefiningOp<ReduceOp>();
             reduce && reduce.getSourceCount() == 1 &&
             reduce.getIdentityCount() == 1 && reduce.getCaptureCount() == 0 &&
             reduce.getNumResults() == 1 &&
             source.getOwner() == target.getOwner() &&
             queryBinaryCombineKind(reduce.getCombine())) {
    // A lane-wise combine preserves every non-reduced axis. Project the
    // source's free axes with the result, rather than resizing a finished
    // reduction or treating its construction extent as a broadcast scalar.
    auto relation = queryAxisProjection(source, target);
    bool projects =
        source.getShape().size() == target.getShape().size() &&
        relation.isExact() && llvm::all_of(
            llvm::enumerate(relation.targetToSource), [](auto item) {
              return item.value() && *item.value() == item.index();
            });
    if (projects) {
      auto input = cast<FragmentType>(reduce.getInputs().front().getType());
      llvm::SmallDenseSet<int64_t> axes(reduce.getAxes().begin(),
                                       reduce.getAxes().end());
      SmallVector<Attribute> shape(input.getShape().getValue());
      unsigned resultAxis = 0;
      for (unsigned axis = 0; axis < shape.size(); ++axis)
        if (!axes.contains(axis))
          shape[axis] = target.getShape()[resultAxis++];
      auto inputTarget = FragmentType::get(
          target.getContext(), input.getElementType(), builder.getArrayAttr(shape),
          input.getAxisMaps(), input.getValidity(), input.getOwner());
      auto resultTarget = FragmentType::get(
          target.getContext(), target.getElementType(), target.getShape(),
          source.getAxisMaps(), target.getValidity(), target.getOwner());
      auto projectedSource = projectFragmentValue(
          builder, location, reduce.getInputs().front(), inputTarget);
      auto projectedIdentity = projectFragmentValue(
          builder, location, reduce.getInputs()[1], resultTarget);
      if (succeeded(projectedSource) && succeeded(projectedIdentity)) {
        IRMapping mapping;
        mapping.map(reduce.getInputs().front(), *projectedSource);
        mapping.map(reduce.getInputs()[1], *projectedIdentity);
        auto clone = cast<ReduceOp>(builder.clone(*reduce, mapping));
        for (BlockArgument argument : clone.getCombine().front().getArguments())
          for (unsigned axis = 0; axis < target.getShape().size(); ++axis)
            if (source.getShape()[axis] != target.getShape()[axis])
              retargetSourceExtent(
                  argument,
                  sourceAxisIdentity(
                      cast<AxisMapAttr>(source.getAxisMaps()[axis])),
                  cast<PhysicalExprAttr>(target.getShape()[axis]));
        clone.getResult(0).setType(resultTarget);
        projection = clone.getOperation();
        if (resultTarget != target)
          projection = builder.create<BroadcastOp>(location, target,
                                                   clone.getResult(0));
      }
    }
  }
  if (!projection && queryBroadcastProjection(source, target).isExact())
    projection = builder.create<BroadcastOp>(location, target, value);
  Operation *definition = value.getDefiningOp();
  if (!projection &&
      isa_and_nonnull<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(
          definition)) {
    IRMapping mapping;
    for (Value operand : definition->getOperands()) {
      Type element = operand.getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      auto operandTarget = FragmentType::get(
          target.getContext(), element, target.getShape(), target.getAxisMaps(),
          target.getValidity(), target.getOwner());
      FailureOr<Value> projected =
          projectFragmentValue(builder, location, operand, operandTarget);
      if (failed(projected))
        return failure();
      mapping.map(operand, *projected);
    }
    projection = builder.clone(*definition, mapping);
    projection->getResult(0).setType(target);
  }
  if (!projection)
    return failure();
  if (Operation *definition = value.getDefiningOp())
    if (Attribute origin = definition->getAttr(originAttr))
      projection->setAttr(originAttr, origin);
  return projection->getResult(0);
}

FailureOr<Value> materializeBroadcastToFragment(OpBuilder &builder,
                                                Location location, Value value,
                                                FragmentType target) {
  return projectFragmentValue(builder, location, value, target);
}

FailureOr<Value> projectPhysicalValueToSchema(OpBuilder &builder,
                                              Location location, Value value,
                                              Type target) {
  if (value.getType() == target)
    return value;
  if (auto fragment = dyn_cast<FragmentType>(target))
    return projectFragmentValue(builder, location, value, fragment);
  auto targetRecord = dyn_cast<RecordType>(target);
  auto sourceRecord = dyn_cast<RecordType>(value.getType());
  if (!targetRecord || !sourceRecord ||
      sourceRecord.getFieldNames() != targetRecord.getFieldNames() ||
      sourceRecord.getFieldTypes().size() !=
          targetRecord.getFieldTypes().size())
    return failure();
  auto record = value.getDefiningOp<MakeRecordOp>();
  SmallVector<Value> projectedFields;
  for (auto [index, targetField] :
       llvm::enumerate(targetRecord.getFieldTypes())) {
    Type sourceType =
        cast<TypeAttr>(sourceRecord.getFieldTypes()[index]).getValue();
    Value field = record
                      ? record.getFields()[index]
                      : Value(builder.create<ExtractOp>(location, sourceType,
                                                        value, index));
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, location, field, cast<TypeAttr>(targetField).getValue());
    if (failed(projected))
      return failure();
    projectedFields.push_back(*projected);
  }
  auto projected =
      builder.create<MakeRecordOp>(location, targetRecord, projectedFields);
  if (Operation *definition = value.getDefiningOp())
    if (Attribute origin = definition->getAttr(originAttr))
      projected->setAttr(originAttr, origin);
  return projected.getResult();
}

FailureOr<FragmentType> refinePhysicalSchema(func::FuncOp kernel,
                                             FragmentType target,
                                             ValueRange contributors);

static WalkResult alignReductionResultRelation(Operation *operation) {
    auto kernel = operation->getParentOfType<func::FuncOp>();
    SmallVector<Value> sources;
    SmallVector<Value> results;
    llvm::SmallDenseSet<int64_t> reducedAxes;
    bool scan = false;
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      sources.append(reduce.getInputs().begin(),
                     reduce.getInputs().begin() + reduce.getSourceCount());
      results.append(reduce.getResults().begin(), reduce.getResults().end());
      reducedAxes.insert(reduce.getAxes().begin(), reduce.getAxes().end());
    } else if (auto currentScan = dyn_cast<ScanOp>(operation)) {
      sources.append(currentScan.getInputs().begin(),
                     currentScan.getInputs().begin() +
                         currentScan.getSourceCount());
      results.append(currentScan.getResults().begin(),
                     currentScan.getResults().end());
      scan = true;
    } else {
      return WalkResult::advance();
    }
    if (sources.size() != results.size())
      return WalkResult::interrupt();
    for (auto [index, source] : llvm::enumerate(sources)) {
      auto sourceType = dyn_cast<FragmentType>(source.getType());
      if (!sourceType || llvm::all_of(sources, [&](Value other) {
            auto type = dyn_cast<FragmentType>(other.getType());
            return type && type.getShape() == sourceType.getShape();
          }))
        continue;
      SmallVector<Value> related;
      llvm::copy_if(sources, std::back_inserter(related), [&](Value other) {
        auto type = dyn_cast<FragmentType>(other.getType());
        return type && type.getAxisMaps() == sourceType.getAxisMaps();
      });
      FailureOr<FragmentType> refined =
          refinePhysicalSchema(kernel, sourceType, related);
      if (failed(refined)) {
        operation->emitOpError(
            "tuple reduction sources have no common physical extent relation");
        return WalkResult::interrupt();
      }
      auto target = FragmentType::get(
          kernel.getContext(), sourceType.getElementType(), (*refined).getShape(),
          sourceType.getAxisMaps(), sourceType.getValidity(), sourceType.getOwner());
      OpBuilder builder(operation);
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, operation->getLoc(), source, target);
      if (failed(projected)) {
        operation->emitOpError(
            "tuple reduction source cannot adopt its physical extent relation");
        return WalkResult::interrupt();
      }
      operation->setOperand(index, *projected);
      sources[index] = *projected;
    }
    for (auto [source, currentResult] : llvm::zip_equal(sources, results)) {
      if (scan) {
        currentResult.setType(source.getType());
        continue;
      }
      auto sourceType = dyn_cast<FragmentType>(source.getType());
      if (!sourceType) {
        operation->emitOpError(
            "physical reduction source has no fragment result relation")
            << "; source=" << source.getType();
        return WalkResult::interrupt();
      }
      SmallVector<Attribute> shape;
      SmallVector<Attribute> mappings;
      for (auto [axis, extent] : llvm::enumerate(sourceType.getShape())) {
        if (reducedAxes.contains(static_cast<int64_t>(axis)))
          continue;
        shape.push_back(extent);
        auto mapping = cast<AxisMapAttr>(sourceType.getAxisMaps()[axis]);
        mappings.push_back(AxisMapAttr::get(
            kernel.getContext(), mapping.getSourceId(),
            mapping.getSourceAxis(), mapping.getDimensionId(),
            static_cast<uint32_t>(mappings.size()), mapping.getDerived()));
      }
      Type element = currentResult.getType();
      if (auto current = dyn_cast<FragmentType>(element))
        element = current.getElementType();
      if (shape.empty()) {
        currentResult.setType(element);
        continue;
      }
      currentResult.setType(FragmentType::get(
          kernel.getContext(), element,
          ArrayAttr::get(kernel.getContext(), shape),
          ArrayAttr::get(kernel.getContext(), mappings),
          sourceType.getValidity(), sourceType.getOwner()));
    }
    return WalkResult::advance();
}

LogicalResult alignReductionResultRelations(func::FuncOp kernel) {
  WalkResult result = kernel.walk(alignReductionResultRelation);
  return result.wasInterrupted() ? failure() : success();
}

static WalkResult alignReductionIdentityRelation(Operation *operation) {
    auto align = [&](ValueRange inputs, ValueRange results, Region &combine,
                     uint64_t sourceCount,
                     uint64_t identityCount) -> LogicalResult {
      if (identityCount != results.size())
        return failure();
      OpBuilder builder(operation);
      for (unsigned index = 0; index < identityCount; ++index) {
        unsigned operand = sourceCount + index;
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, operation->getLoc(), inputs[operand],
            results[index].getType());
        if (failed(projected))
          return operation->emitOpError(
              "physical reduction identity cannot adopt its result relation");
        operation->setOperand(operand, *projected);
      }
      if (!llvm::hasSingleElement(combine) ||
          combine.front().getNumArguments() < 2 * identityCount)
        return operation->emitOpError(
            "physical reduction helper has no complete accumulator schema");
      for (unsigned index = 0; index < identityCount; ++index) {
        Type target = results[index].getType();
        combine.front().getArgument(index).setType(target);
        combine.front().getArgument(identityCount + index).setType(target);
      }
      return success();
    };
    if (auto reduce = dyn_cast<ReduceOp>(operation))
      return failed(align(reduce.getInputs(), reduce.getResults(),
                          reduce.getCombine(),
                          reduce.getSourceCount(), reduce.getIdentityCount()))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    if (auto scan = dyn_cast<ScanOp>(operation))
      return failed(align(scan.getInputs(), scan.getResults(), scan.getCombine(),
                          scan.getSourceCount(), scan.getIdentityCount()))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    return WalkResult::advance();
}

LogicalResult alignReductionIdentityRelations(func::FuncOp kernel) {
  WalkResult result = kernel.walk(alignReductionIdentityRelation);
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult alignReductionYieldRelations(func::FuncOp kernel) {
  WalkResult result = kernel.walk([&](Operation *operation) {
    Region *combine = nullptr;
    ValueRange results;
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      combine = &reduce.getCombine();
      results = reduce.getResults();
    } else if (auto scan = dyn_cast<ScanOp>(operation)) {
      combine = &scan.getCombine();
      results = scan.getResults();
    } else {
      return WalkResult::advance();
    }
    if (!combine || !llvm::hasSingleElement(*combine))
      return WalkResult::interrupt();
    auto yield = dyn_cast<YieldOp>(combine->front().getTerminator());
    if (!yield || yield.getValues().size() != results.size())
      return WalkResult::interrupt();
    OpBuilder builder(yield);
    for (auto [index, target] : llvm::enumerate(results)) {
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, operation->getLoc(), yield.getValues()[index],
          target.getType());
      if (failed(projected)) {
        operation->emitOpError(
            "physical reduction yield cannot adopt its result relation")
            << "; result_index=" << index
            << "; yield_type=" << yield.getValues()[index].getType()
            << "; result_type=" << target.getType();
        return WalkResult::interrupt();
      }
      yield->setOperand(index, *projected);
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

FailureOr<Value> materializeZeroFragment(OpBuilder &builder,
                                         Location location,
                                         FragmentType target) {
  return zeroFill(builder, location, target);
}

Value stripAdditiveProjection(Value value, bool singleUse) {
  auto elementType = [](Type type) {
    auto fragment = dyn_cast<FragmentType>(type);
    return fragment ? fragment.getElementType() : type;
  };
  while (value) {
    if (singleUse && !value.hasOneUse())
      return {};
    Operation *operation = value.getDefiningOp();
    if (!operation)
      break;
    if (auto cast = dyn_cast<CastOp>(operation)) {
      if (elementType(cast.getValue().getType()) !=
          elementType(cast.getResult().getType()))
        break;
    } else if (!isa<BroadcastOp, ReshapeOp, TransposeOp>(operation)) {
      break;
    }
    value = operation->getOperand(0);
  }
  return value;
}

bool isLiteralZeroProjection(Value value) {
  while (Operation *operation = value.getDefiningOp()) {
    if (auto extract = dyn_cast<ExtractOp>(operation)) {
      auto record = extract.getRecord().getDefiningOp<MakeRecordOp>();
      if (!record)
        return false;
      value = record.getFields()[extract.getField()];
      continue;
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
        return integer.getValue().isZero();
      if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
        return floating.getValue().isZero();
      return false;
    }
    if (!isa<SplatOp, BroadcastOp, ReshapeOp, TransposeOp, CastOp>(operation))
      return false;
    value = operation->getOperand(0);
  }
  return false;
}

FailureOr<Value> materializeReplayedValue(
    OpBuilder &builder, Location location, Value value,
    PhysicalSourceAxis source, PhysicalExprAttr blockedExtent,
    IRMapping &mapping, ReplayMaterializationOptions options) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalReplayFact replay = analysis.replayability(
      value, source, options.scope, options.allowAccesses);
  if (!replay.isReplayable())
    return failure();

  SmallVector<PhysicalSourceAxis> replaySources{source};
  for (MakeRangeOp range : options.traversalRanges) {
    PhysicalSourceAxis occurrence = sourceAxisIdentity(range);
    if (!llvm::is_contained(replaySources, occurrence))
      replaySources.push_back(occurrence);
    if (!analysis.replayability(value, occurrence, options.scope,
                               options.allowAccesses).isReplayable())
      return failure();
  }
  bool projectionFailed = false;
  auto replayProjection = [&](Type type,
                              std::optional<unsigned> axis = std::nullopt) {
    PhysicalAxisProjection result;
    for (PhysicalSourceAxis occurrence : replaySources) {
      auto projections = queryFragmentAxes(type, occurrence);
      if (axis)
        llvm::erase_if(projections, [&](PhysicalAxisProjection projection) {
          return projection.fragmentAxis != *axis;
        });
      PhysicalAxisProjection current;
      if (projections.size() == 1)
        current = projections.front();
      if (projections.size() > 1 ||
          (result.isExact() && current.isExact() &&
           (result.fragmentAxis != current.fragmentAxis ||
            result.dimensionId != current.dimensionId))) {
        result.state = PhysicalFactState::Ambiguous;
        projectionFailed = true;
        return result;
      }
      if (!current.isExact())
        continue;
      result = current;
    }
    return result;
  };
  auto replaceReplayAxis = [&](FragmentType type, unsigned axis) {
    SmallVector<Attribute> shape(type.getShape().begin(), type.getShape().end());
    SmallVector<Attribute> axes(type.getAxisMaps().begin(),
                                type.getAxisMaps().end());
    shape[axis] = blockedExtent;
    if (options.segmentMapping)
      axes[axis] = AxisMapAttr::get(
          type.getContext(), options.segmentMapping.getSourceId(),
          options.segmentMapping.getSourceAxis(),
          options.segmentMapping.getDimensionId(), axis,
          options.segmentMapping.getDerived());
    return FragmentType::get(
        type.getContext(), type.getElementType(),
        ArrayAttr::get(type.getContext(), shape),
        ArrayAttr::get(type.getContext(), axes), type.getValidity(),
        type.getOwner());
  };
  std::function<Type(Type)> replaceReplayType = [&](Type type) -> Type {
    if (auto fragment = dyn_cast<FragmentType>(type)) {
      PhysicalAxisProjection projection = replayProjection(fragment);
      return projection.isExact()
                 ? Type(replaceReplayAxis(fragment, projection.fragmentAxis))
                 : type;
    }
    auto record = dyn_cast<RecordType>(type);
    if (!record)
      return type;
    SmallVector<Attribute> fields;
    bool changed = false;
    for (Attribute field : record.getFieldTypes()) {
      Type current = cast<TypeAttr>(field).getValue();
      Type replacement = replaceReplayType(current);
      fields.push_back(TypeAttr::get(replacement));
      changed |= replacement != current;
    }
    return changed ? Type(RecordType::get(
                         type.getContext(), record.getFieldNames(),
                         ArrayAttr::get(type.getContext(), fields),
                         record.getOwner()))
                   : type;
  };
  auto carriesReplaySource = [&](Type type) {
    return replaceReplayType(type) != type;
  };
  auto selectedReplayAxis = [&](Value current,
                                std::optional<unsigned> axis = std::nullopt)
      -> std::optional<unsigned> {
    auto fragment = dyn_cast<FragmentType>(current.getType());
    PhysicalAxisProjection projection =
        fragment ? replayProjection(fragment, axis)
                 : PhysicalAxisProjection{};
    if (fragment && !projection.isExact() &&
        projection.state != PhysicalFactState::Ambiguous &&
        !options.traversalRanges.empty()) {
      std::optional<unsigned> selected;
      for (unsigned candidate = 0; candidate < fragment.getShape().size(); ++candidate) {
        if (axis && candidate != *axis)
          continue;
        // Equal traversal bounds do not identify independent logical axes.
        auto candidateMap = cast<AxisMapAttr>(fragment.getAxisMaps()[candidate]);
        if (!llvm::any_of(options.traversalRanges, [&](MakeRangeOp range) {
              auto dimension = queryRangeDimension(range);
              return succeeded(dimension) &&
                     *dimension == candidateMap.getDimensionId();
            }))
          continue;
        PhysicalRangeFact fact = analysis.axisRanges(current, candidate);
        if (fact.state == PhysicalFactState::Unknown || !fact.blockers.empty() ||
            fact.roots.empty())
          continue;
        if (!llvm::any_of(fact.roots, [&](MakeRangeOp root) {
              return llvm::any_of(options.traversalRanges, [&](MakeRangeOp range) {
                return sameLogicalRange(root, range);
              });
            }))
          continue;
        SmallVector<MakeRangeOp> combined(fact.roots.begin(), fact.roots.end());
        combined.append(options.traversalRanges.begin(), options.traversalRanges.end());
        if (!analysis.lockstepRanges(combined).isExact())
          continue;
        if (selected)
          return std::nullopt;
        selected = candidate;
      }
      if (selected)
        return selected;
    }
    if (!fragment || !projection.isExact())
      return std::nullopt;
    if (options.traversalRanges.empty())
      return projection.fragmentAxis;
    PhysicalRangeFact fact =
        analysis.axisRanges(current, projection.fragmentAxis);
    if (fact.roots.empty()) {
      Operation *producer = current.getDefiningOp();
      bool neutralSchemaCarrier = isa_and_nonnull<SplatOp>(producer);
      if (auto broadcast = dyn_cast_or_null<BroadcastOp>(producer)) {
        auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
        neutralSchemaCarrier |=
            !isa<FragmentType, RecordType>(broadcast.getValue().getType());
        if (input) {
          BroadcastProjection relation = queryAxisProjection(input, fragment);
          if (relation.isExact()) {
            std::optional<unsigned> inputAxis =
                relation.targetToSource[projection.fragmentAxis];
            if (!inputAxis) {
              neutralSchemaCarrier = true;
            } else {
              auto extent = cast<PhysicalExprAttr>(input.getShape()[*inputAxis]);
              neutralSchemaCarrier |=
                  extent.getKind() ==
                      static_cast<uint32_t>(PhysicalExprKind::Constant) &&
                  extent.getValue() == 1;
            }
          }
        }
      }
      return neutralSchemaCarrier
                 ? std::optional<unsigned>(projection.fragmentAxis)
                 : std::nullopt;
    }
    SmallVector<MakeRangeOp> combined(fact.roots.begin(), fact.roots.end());
    combined.append(options.traversalRanges.begin(),
                    options.traversalRanges.end());
    return analysis.lockstepRanges(combined).isExact()
               ? std::optional<unsigned>(projection.fragmentAxis)
               : std::nullopt;
  };
  auto isUnitExtent = [](Attribute attribute) {
    auto extent = dyn_cast<PhysicalExprAttr>(attribute);
    return extent &&
           extent.getKind() ==
               static_cast<uint32_t>(PhysicalExprKind::Constant) &&
           extent.getValue() == 1;
  };
  auto retargetHelperSourceExtent = [&](Region &region,
                                        PhysicalExprAttr logicalExtent) {
    auto retarget = [&](Value current) {
      auto fragment = dyn_cast<FragmentType>(current.getType());
      PhysicalAxisProjection projection =
          fragment ? queryFragmentAxis(fragment, source)
                   : PhysicalAxisProjection{};
      if (!fragment || !projection.isExact() ||
          fragment.getShape()[projection.fragmentAxis] != logicalExtent)
        return;
      current.setType(replaceReplayAxis(fragment, projection.fragmentAxis));
    };
    for (Block &block : region) {
      for (BlockArgument argument : block.getArguments())
        retarget(argument);
      block.walk([&](Operation *operation) {
        for (Value result : operation->getResults())
          retarget(result);
      });
    }
  };

  llvm::DenseMap<std::pair<Value, unsigned>, Value> axisValues;
  auto hasMultipleReplayAxes = [&](Type type) {
    std::optional<unsigned> selected;
    for (PhysicalSourceAxis occurrence : replaySources)
      for (PhysicalAxisProjection projection :
           queryFragmentAxes(type, occurrence)) {
        if (selected && *selected != projection.fragmentAxis)
          return true;
        selected = projection.fragmentAxis;
      }
    return false;
  };
  auto remember = [&](Value original, Value replacement,
                       std::optional<unsigned> axis) {
    if (axis)
      axisValues[{original, *axis}] = replacement;
    if (!hasMultipleReplayAxes(original.getType()))
      mapping.map(original, replacement);
  };
  std::function<FailureOr<Value>(Value, std::optional<unsigned>)> materialize =
      [&](Value current,
          std::optional<unsigned> requestedAxis) -> FailureOr<Value> {
    auto fragment = dyn_cast<FragmentType>(current.getType());
    std::optional<unsigned> projection =
        selectedReplayAxis(current, requestedAxis);
    if (projection) {
      auto cached = axisValues.find({current, *projection});
      if (cached != axisValues.end())
        return cached->second;
    }
    if (Value mapped = mapping.lookupOrNull(current)) {
      // A prebound load must identify its occurrence by the changed axis alone.
      if (hasMultipleReplayAxes(current.getType()) &&
          (!projection || fragment.getShape()[*projection] == blockedExtent ||
           mapped.getType() != replaceReplayAxis(fragment, *projection)))
        return failure();
      return mapped;
    }
    if (auto extract = current.getDefiningOp<ExtractOp>()) {
      if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
        uint64_t field = extract.getField();
        if (field >= record.getFields().size())
          return failure();
        FailureOr<Value> replayed =
            materialize(record.getFields()[field], projection);
        if (succeeded(replayed))
          remember(current, *replayed, projection);
        return replayed;
      }
      if (carriesReplaySource(extract.getRecord().getType())) {
        FailureOr<Value> replayedRecord =
            materialize(extract.getRecord(), std::nullopt);
        if (failed(replayedRecord))
          return failure();
        Type target = replaceReplayType(current.getType());
        Value replayed = builder.create<ExtractOp>(
            location, target, *replayedRecord, extract.getField());
        remember(current, replayed, projection);
        return replayed;
      }
    }

    if (!fragment && carriesReplaySource(current.getType())) {
      Operation *producer = current.getDefiningOp();
      if (!producer ||
          !isPhysicalReplayNode(producer, options.scope,
                                /*allowAccesses=*/false))
        return failure();
      for (Value operand : producer->getOperands()) {
        if (!carriesReplaySource(operand.getType()))
          continue;
        FailureOr<Value> replayed = materialize(operand, std::nullopt);
        if (failed(replayed))
          return failure();
        if (!mapping.lookupOrNull(operand) && *replayed != operand)
          mapping.map(operand, *replayed);
      }
      Operation *clone = builder.clone(*producer, mapping);
      for (auto [original, result] :
           llvm::zip(producer->getResults(), clone->getResults())) {
        result.setType(replaceReplayType(result.getType()));
        if (!mapping.lookupOrNull(original))
          mapping.map(original, result);
      }
      Region *combine = nullptr;
      if (auto reduce = dyn_cast<ReduceOp>(clone))
        combine = &reduce.getCombine();
      else if (auto scan = dyn_cast<ScanOp>(clone))
        combine = &scan.getCombine();
      if (combine)
        for (Block &block : *combine) {
          for (BlockArgument argument : block.getArguments())
            argument.setType(replaceReplayType(argument.getType()));
          block.walk([&](Operation *operation) {
            for (Value result : operation->getResults())
              result.setType(replaceReplayType(result.getType()));
          });
        }
      auto result = dyn_cast<OpResult>(current);
      if (!result || result.getResultNumber() >= clone->getNumResults())
        return failure();
      Value replayed = clone->getResult(result.getResultNumber());
      return replayed;
    }
    if (!fragment)
      return current;
    if (!projection) {
      Operation *producer = current.getDefiningOp();
      if (!producer || !isPhysicalReplayNode(
                           producer, PhysicalReplayScope::Coordinate,
                           /*allowAccesses=*/false))
        return current;
      IRMapping cloneMapping(mapping);
      bool changed = false;
      for (Value operand : producer->getOperands()) {
        FailureOr<Value> replayed = materialize(operand, std::nullopt);
        if (failed(replayed))
          return failure();
        cloneMapping.map(operand, *replayed);
        changed |= *replayed != operand;
      }
      if (!changed)
        return current;
      Operation *clone = builder.clone(*producer, cloneMapping);
      remember(current, clone->getResult(0), std::nullopt);
      return clone->getResult(0);
    }
    unsigned axis = *projection;
    Operation *producer = current.getDefiningOp();
    if (!producer)
      return failure();

    auto replayOperand = [&](Value operand) -> FailureOr<Value> {
      std::optional<unsigned> operandAxis;
      auto input = dyn_cast<FragmentType>(operand.getType());
      if (input && (hasMultipleReplayAxes(input) ||
                    hasMultipleReplayAxes(fragment))) {
        auto reduction = dyn_cast<ReduceOp>(producer);
        bool reductionSource = reduction && llvm::is_contained(
            reduction.getInputs().take_front(reduction.getSourceCount()), operand);
        if (isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
                BroadcastOp, SplatOp>(producer) ||
            (reduction && !reductionSource)) {
          BroadcastProjection relation = queryAxisProjection(input, fragment);
          if (!relation.isExact())
            return failure();
          operandAxis = relation.targetToSource[axis];
          if (!operandAxis)
            return operand;
        } else if (auto transpose = dyn_cast<TransposeOp>(producer)) {
          operandAxis = transpose.getPermutation()[axis];
        } else if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
          for (Attribute attribute : reshape.getReassociation()) {
            auto group = cast<ReshapeGroupAttr>(attribute);
            if (group.getSourceAxes().empty() &&
                llvm::is_contained(group.getResultAxes().asArrayRef(),
                                   static_cast<int64_t>(axis)))
              return operand;
            if (group.getResultAxes().size() == 1 &&
                group.getResultAxes()[0] == static_cast<int64_t>(axis) &&
                group.getSourceAxes().size() == 1)
              operandAxis = group.getSourceAxes()[0];
          }
        } else if (reduction) {
          SmallVector<unsigned> freeAxes;
          for (unsigned inputAxis = 0; inputAxis < input.getShape().size();
               ++inputAxis)
            if (!llvm::is_contained(reduction.getAxes(),
                                   static_cast<int64_t>(inputAxis)))
              freeAxes.push_back(inputAxis);
          if (axis < freeAxes.size())
            operandAxis = freeAxes[axis];
        }
        if (!operandAxis)
          return failure();
      }
      if (Value mapped = mapping.lookupOrNull(operand);
          mapped && !hasMultipleReplayAxes(operand.getType()))
        return mapped;
      return materialize(operand, operandAxis);
    };
    auto combineTail = [&](FragmentType target,
                           Value valid) -> FailureOr<Value> {
      if (!options.segmentTail)
        return valid ? FailureOr<Value>(valid)
                     : FailureOr<Value>(Value());
      FailureOr<Value> tail = projectPredicateToFragmentAxis(
          builder, location, options.segmentTail, target, axis);
      if (failed(tail))
        return failure();
      if (!valid)
        return *tail;
      return materializeValidityConjunction(builder, location, valid, *tail,
                                            target);
    };
    auto replayFill = [&](Value fill,
                          FragmentType target) -> FailureOr<Value> {
      if (fill) {
        FailureOr<Value> replayed = replayOperand(fill);
        if (failed(replayed))
          return failure();
        fill = *replayed;
        if (fill.getType() != target) {
          FailureOr<Value> projected =
              projectPhysicalValueToSchema(builder, location, fill, target);
          if (failed(projected))
            return failure();
          fill = *projected;
        }
        return fill;
      }
      if (!options.materializeZeroFill)
        return Value();
      return materializeZeroFragment(builder, location, target);
    };

    if (auto broadcast = dyn_cast<BroadcastOp>(producer)) {
      auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
      BroadcastProjection relation =
          input ? queryAxisProjection(input, fragment) : BroadcastProjection{};
      if (relation.isExact() && axis < relation.targetToSource.size()) {
        if (auto inputAxis = relation.targetToSource[axis]) {
          auto inputMap = cast<AxisMapAttr>(input.getAxisMaps()[*inputAxis]);
          auto resultMap = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
          auto inputExtent =
              cast<PhysicalExprAttr>(input.getShape()[*inputAxis]);
          bool sameLogicalAxis = inputMap.getDimensionId() > 0 &&
                                 inputMap.getDimensionId() ==
                                     resultMap.getDimensionId();
          bool nonUnitStaticExtent =
              inputExtent.getKind() ==
                  static_cast<uint32_t>(PhysicalExprKind::Constant) &&
              inputExtent.getValue() > 1 &&
              queryBroadcastProjection(input, fragment).isExact();
          if (!(sourceAxisIdentity(inputMap) == sourceAxisIdentity(resultMap)) &&
              (sameLogicalAxis || nonUnitStaticExtent) &&
              input.getShape()[*inputAxis] == fragment.getShape()[axis]) {
            // An extent-preserving projection can rename an occurrence. Replay
            // its input with that input's identity, retaining the output map.
            ReplayMaterializationOptions inputOptions = options;
            inputOptions.fragmentAxis = *inputAxis;
            FailureOr<Value> replayed = materializeReplayedValue(
                builder, location, broadcast.getValue(),
                sourceAxisIdentity(inputMap), blockedExtent, mapping,
                inputOptions);
            if (failed(replayed))
              return failure();
            FailureOr<Value> projected = projectPhysicalValueToSchema(
                builder, location, *replayed, replaceReplayAxis(fragment, axis));
            if (failed(projected))
              return failure();
            remember(current, *projected, projection);
            return *projected;
          }
        }
      }
    }

    if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
      auto input = cast<FragmentType>(reshape.getValue().getType());
      for (Attribute attribute : reshape.getReassociation()) {
        auto group = cast<ReshapeGroupAttr>(attribute);
        if (group.getSourceAxes().size() != 1 ||
            group.getResultAxes().size() != 1 ||
            group.getResultAxes()[0] != static_cast<int64_t>(axis))
          continue;
        unsigned inputAxis = group.getSourceAxes()[0];
        auto inputMap = cast<AxisMapAttr>(input.getAxisMaps()[inputAxis]);
        auto resultMap = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
        if (sourceAxisIdentity(inputMap) == sourceAxisIdentity(resultMap) ||
            input.getShape()[inputAxis] != fragment.getShape()[axis])
          continue;
        ReplayMaterializationOptions inputOptions = options;
        inputOptions.fragmentAxis = inputAxis;
        FailureOr<Value> replayed = materializeReplayedValue(
            builder, location, reshape.getValue(), sourceAxisIdentity(inputMap),
            blockedExtent, mapping, inputOptions);
        if (failed(replayed))
          return failure();
        Value projected = builder.create<ReshapeOp>(
            location, replaceReplayAxis(fragment, axis), *replayed,
            reshape.getReassociation());
        remember(current, projected, projection);
        return projected;
      }
    }

    if (auto contract = dyn_cast<ContractOp>(producer)) {
      SmallVector<std::pair<unsigned, unsigned>> freeAxes;
      auto lhsType = cast<FragmentType>(contract.getLhs().getType());
      auto rhsType = cast<FragmentType>(contract.getRhs().getType());
      for (int64_t inputAxis : contract.getLhsBatchAxes())
        freeAxes.emplace_back(0, inputAxis);
      for (unsigned inputAxis = 0; inputAxis < lhsType.getShape().size();
           ++inputAxis)
        if (!llvm::is_contained(contract.getLhsReductionAxes(),
                               static_cast<int64_t>(inputAxis)) &&
            !llvm::is_contained(contract.getLhsBatchAxes(),
                               static_cast<int64_t>(inputAxis)))
          freeAxes.emplace_back(0, inputAxis);
      for (unsigned inputAxis = 0; inputAxis < rhsType.getShape().size();
           ++inputAxis)
        if (!llvm::is_contained(contract.getRhsReductionAxes(),
                               static_cast<int64_t>(inputAxis)) &&
            !llvm::is_contained(contract.getRhsBatchAxes(),
                               static_cast<int64_t>(inputAxis)))
          freeAxes.emplace_back(1, inputAxis);
      if (axis >= freeAxes.size())
        return failure();
      // One SSA value can have different matrix roles at the two input uses.
      SmallVector<Value> operands{contract.getLhs(), contract.getRhs(),
                                   contract.getAccumulator()};
      auto replayAxis = [&](unsigned operand,
                            unsigned inputAxis) -> LogicalResult {
        auto type = cast<FragmentType>(operands[operand].getType());
        auto axisMap = cast<AxisMapAttr>(type.getAxisMaps()[inputAxis]);
        FailureOr<Value> replayed = failure();
        if (llvm::is_contained(replaySources, sourceAxisIdentity(axisMap))) {
          replayed = materialize(operands[operand], inputAxis);
        } else {
          ReplayMaterializationOptions inputOptions = options;
          inputOptions.fragmentAxis = inputAxis;
          replayed = materializeReplayedValue(
              builder, location, operands[operand], sourceAxisIdentity(axisMap),
              blockedExtent, mapping, inputOptions);
        }
        if (failed(replayed))
          return failure();
        operands[operand] = *replayed;
        return success();
      };
      auto [operand, inputAxis] = freeAxes[axis];
      if (failed(replayAxis(operand, inputAxis)) || failed(replayAxis(2, axis)))
        return failure();
      if (operand == 0)
        for (auto [batch, lhsAxis] : llvm::enumerate(contract.getLhsBatchAxes()))
          if (lhsAxis == static_cast<int64_t>(inputAxis) &&
              failed(replayAxis(1, contract.getRhsBatchAxes()[batch])))
            return failure();
      IRMapping cloneMapping(mapping);
      Operation *clone = builder.clone(*producer, cloneMapping);
      clone->setOperands(operands);
      clone->getResult(0).setType(replaceReplayAxis(fragment, axis));
      remember(current, clone->getResult(0), projection);
      return clone->getResult(0);
    }

    if (isa<BroadcastOp, SplatOp>(producer)) {
      FailureOr<Value> replayed = replayOperand(producer->getOperand(0));
      if (failed(replayed))
        return failure();
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, location, *replayed, replaceReplayAxis(fragment, axis));
      if (failed(projected))
        return failure();
      remember(current, *projected, projection);
      return *projected;
    }
    if (auto load = dyn_cast<LoadOp>(producer)) {
      SmallVector<Value> coordinates;
      for (Value coordinate : load.getCoordinates()) {
        FailureOr<Value> replayed = replayOperand(coordinate);
        if (failed(replayed))
          return failure();
        coordinates.push_back(*replayed);
      }
      Value valid;
      if (load.getValid()) {
        FailureOr<Value> replayed = replayOperand(load.getValid());
        if (failed(replayed))
          return failure();
        valid = *replayed;
      }
      FragmentType resultType = replaceReplayAxis(fragment, axis);
      FailureOr<Value> combined = combineTail(resultType, valid);
      if (failed(combined))
        return failure();
      FailureOr<Value> fill = replayFill(load.getFill(), resultType);
      if (failed(fill))
        return failure();
      auto clone = builder.create<LoadOp>(
          location, resultType, load.getResource(), coordinates, *combined,
          *fill, load.getSourceAxes());
      if (Attribute origin = load->getAttr(originAttr))
        clone->setAttr(originAttr, origin);
      remember(current, clone.getResult(), projection);
      return clone.getResult();
    }
    if (auto gather = dyn_cast<GatherOp>(producer)) {
      FailureOr<Value> replayedSource = replayOperand(gather.getSource());
      if (failed(replayedSource))
        return failure();
      SmallVector<Value> coordinates;
      for (Value coordinate : gather.getCoordinates()) {
        FailureOr<Value> replayed = replayOperand(coordinate);
        if (failed(replayed))
          return failure();
        coordinates.push_back(*replayed);
      }
      Value valid;
      if (gather.getValid()) {
        FailureOr<Value> replayed = replayOperand(gather.getValid());
        if (failed(replayed))
          return failure();
        valid = *replayed;
      }
      FragmentType resultType = replaceReplayAxis(fragment, axis);
      FailureOr<Value> combined = combineTail(resultType, valid);
      if (failed(combined))
        return failure();
      FailureOr<Value> fill = replayFill(gather.getFill(), resultType);
      if (failed(fill))
        return failure();
      auto clone = builder.create<GatherOp>(
          location, resultType, *replayedSource, coordinates, *combined, *fill,
          gather.getSourceAxes());
      if (Attribute origin = gather->getAttr(originAttr))
        clone->setAttr(originAttr, origin);
      remember(current, clone.getResult(), projection);
      return clone.getResult();
    }
    if (!isPhysicalReplayNode(producer, options.scope,
                              /*allowAccesses=*/false))
      return failure();
    IRMapping cloneMapping(mapping);
    for (Value operand : producer->getOperands()) {
      FailureOr<Value> replayed = replayOperand(operand);
      if (failed(replayed))
        return failure();
      cloneMapping.map(operand, *replayed);
    }
    FragmentType pointwiseType;
    if (isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(producer)) {
      SmallVector<Value> operands;
      for (Value operand : producer->getOperands())
        operands.push_back(cloneMapping.lookupOrDefault(operand));
      auto refined = refinePhysicalSchema(
          kernel, replaceReplayAxis(fragment, axis), operands);
      if (failed(refined))
        return failure();
      pointwiseType = *refined;
      for (Value operand : producer->getOperands()) {
        Value replayed = cloneMapping.lookupOrDefault(operand);
        Type element = replayed.getType();
        if (auto type = dyn_cast<FragmentType>(element))
          element = type.getElementType();
        auto target = FragmentType::get(
            kernel.getContext(), element, pointwiseType.getShape(),
            pointwiseType.getAxisMaps(), pointwiseType.getValidity(),
            pointwiseType.getOwner());
        auto projected = projectPhysicalValueToSchema(builder, location, replayed, target);
        if (failed(projected))
          return failure();
        cloneMapping.map(operand, *projected);
      }
    }
    Operation *clone = builder.clone(*producer, cloneMapping);
    bool structuredResults = isa<ReduceOp, ScanOp>(clone);
    for (auto [original, cloned] :
         llvm::zip(producer->getResults(), clone->getResults())) {
      if (structuredResults)
        cloned.setType(replaceReplayType(cloned.getType()));
      if (!hasMultipleReplayAxes(original.getType()))
        mapping.map(original, cloned);
    }
    auto result = dyn_cast<OpResult>(current);
    if (!result || result.getResultNumber() >= clone->getNumResults())
      return failure();
    Value clonedValue = clone->getResult(result.getResultNumber());
    if (auto clonedReduce = dyn_cast<ReduceOp>(clone))
      retargetHelperSourceExtent(
          clonedReduce.getCombine(),
          cast<PhysicalExprAttr>(fragment.getShape()[axis]));
    if (auto clonedScan = dyn_cast<ScanOp>(clone))
      retargetHelperSourceExtent(
          clonedScan.getCombine(),
          cast<PhysicalExprAttr>(fragment.getShape()[axis]));
    auto clonedType = dyn_cast<FragmentType>(clonedValue.getType());
    if (!clonedType || axis >= clonedType.getShape().size())
      return failure();
    bool introducedUnitAxis = false;
    if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
      auto input = dyn_cast<FragmentType>(reshape.getValue().getType());
      introducedUnitAxis =
          isUnitExtent(clonedType.getShape()[axis]) &&
          (!input || !queryFragmentAxis(input, source).isExact() ||
           llvm::any_of(reshape.getReassociation(), [&](Attribute attribute) {
             auto group = cast<ReshapeGroupAttr>(attribute);
             return group.getSourceAxes().empty() &&
                    llvm::is_contained(group.getResultAxes().asArrayRef(),
                                       static_cast<int64_t>(axis));
           }));
    }
    if (pointwiseType)
      clonedValue.setType(pointwiseType);
    else if (!introducedUnitAxis)
      clonedValue.setType(replaceReplayAxis(clonedType, axis));
    remember(current, clonedValue, projection);
    return clonedValue;
  };

  FailureOr<Value> result = materialize(value, options.fragmentAxis);
  return projectionFailed ? FailureOr<Value>(failure()) : result;
}

FailureOr<Value> projectPredicateToFragmentAxis(OpBuilder &builder,
                                                Location location,
                                                Value predicate,
                                                FragmentType target,
                                                unsigned fragmentAxis) {
  return projectPredicate(builder, location, predicate, target, fragmentAxis);
}

FailureOr<Value> projectPredicateToFragment(OpBuilder &builder,
                                            Location location, Value predicate,
                                            FragmentType target,
                                            PhysicalSourceAxis source) {
  PhysicalAxisProjection projection = queryFragmentAxis(target, source);
  if (!projection.isExact())
    return failure();
  return projectPredicate(builder, location, predicate, target,
                          projection.fragmentAxis);
}

FailureOr<Value> projectPredicateToFragment(OpBuilder &builder,
                                            Location location, Value predicate,
                                            FragmentType target,
                                            int64_t dimensionId) {
  PhysicalDimensionProjection projection =
      queryFragmentDimension(target, dimensionId);
  if (!projection.isExact())
    return failure();
  return projectPredicate(builder, location, predicate, target,
                          projection.fragmentAxis);
}

FailureOr<Value> materializeValidityConjunction(
    OpBuilder &builder, Location location, Value lhs, Value rhs,
    FragmentType valueType) {
  auto predicateType = FragmentType::get(
      valueType.getContext(), builder.getI1Type(), valueType.getShape(),
      valueType.getAxisMaps(), valueType.getValidity(), valueType.getOwner());
  Value result;
  if (lhs) {
    FailureOr<Value> projected =
        materializeBroadcastToFragment(builder, location, lhs, predicateType);
    if (failed(projected))
      return failure();
    result = *projected;
  }
  if (rhs) {
    FailureOr<Value> projected =
        materializeBroadcastToFragment(builder, location, rhs, predicateType);
    if (failed(projected))
      return failure();
    result = result ? Value(builder.create<BinaryOp>(
                          location, predicateType, result, *projected,
                          BinaryOperator::LogicalAnd))
                    : *projected;
  }
  return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
}

FailureOr<Value> materializeRetargetedValidity(
    OpBuilder &builder, Location location, Value original,
    ArrayRef<std::pair<MakeRangeOp, Value>> originalTailRanges,
    Value physicalTail, FragmentType target) {
  func::FuncOp kernel;
  if (!originalTailRanges.empty())
    kernel = originalTailRanges.front().first->getParentOfType<func::FuncOp>();
  else if (original)
    if (Operation *definition = original.getDefiningOp())
      kernel = definition->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<std::pair<MakeRangeOp, Value>> equivalentTailRanges(
      originalTailRanges.begin(), originalTailRanges.end());
  if (original && isa<FragmentType>(original.getType())) {
    for (auto [expected, end] : originalTailRanges) {
      auto axis =
          queryFragmentAxis(original.getType(), sourceAxisIdentity(expected));
      auto dimension = queryRangeDimension(expected);
      if (!axis.isExact() || failed(dimension) || axis.dimensionId != *dimension)
        continue;
      auto ranges = analysis.axisRanges(original, axis.fragmentAxis);
      if (ranges.state == PhysicalFactState::Unknown || !ranges.blockers.empty())
        continue;
      SmallVector<MakeRangeOp> coincident{expected};
      llvm::append_range(coincident, ranges.roots);
      if (!analysis.lockstepRanges(coincident).isExact())
        continue;
      // A predicate may carry the output's coordinate identity on this input
      // axis. Keep the positional proof as well as equal physical traversal;
      // equal range bounds alone do not equate independent Cartesian axes.
      for (MakeRangeOp range : ranges.roots)
        if (!llvm::is_contained(equivalentTailRanges,
                                std::pair<MakeRangeOp, Value>{range, end}))
          equivalentTailRanges.emplace_back(range, end);
    }
  }

  std::function<FailureOr<Value>(Value)> residual =
      [&](Value value) -> FailureOr<Value> {
    if (!value || analysis.isTailPredicate(value, equivalentTailRanges))
      return Value();
    if (value.getType().isInteger(1))
      return value;
    if (auto broadcast = value.getDefiningOp<BroadcastOp>())
      return residual(broadcast.getValue());
    if (auto splat = value.getDefiningOp<SplatOp>())
      return residual(splat.getValue());
    if (auto reshape = value.getDefiningOp<ReshapeOp>())
      return residual(reshape.getValue());
    if (auto transpose = value.getDefiningOp<TransposeOp>())
      return residual(transpose.getValue());
    auto conjunction = value.getDefiningOp<BinaryOp>();
    if (!conjunction)
      return failure();
    Type element = conjunction.getResult().getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    bool logical =
        conjunction.getOperatorKind() == BinaryOperator::LogicalAnd;
    bool bitwiseI1 =
        conjunction.getOperatorKind() == BinaryOperator::BitwiseAnd &&
        element.isInteger(1);
    if (!logical && !bitwiseI1)
      return failure();
    FailureOr<Value> lhs = residual(conjunction.getLhs());
    FailureOr<Value> rhs = residual(conjunction.getRhs());
    if (failed(lhs) || failed(rhs))
      return failure();
    if (!*lhs)
      return *rhs;
    if (!*rhs)
      return *lhs;
    if (!(*lhs).getType().isInteger(1) || !(*rhs).getType().isInteger(1))
      return failure();
    return Value(builder.create<BinaryOp>(location, builder.getI1Type(), *lhs,
                                          *rhs,
                                          BinaryOperator::LogicalAnd));
  };

  FailureOr<Value> authorResidual = residual(original);
  if (failed(authorResidual))
    return failure();
  if (!*authorResidual)
    return physicalTail;
  if (!physicalTail)
    return materializeBroadcastToFragment(builder, location, *authorResidual,
                                          target);
  return materializeValidityConjunction(builder, location, physicalTail,
                                        *authorResidual, target);
}

FailureOr<FragmentType> refinePhysicalSchema(func::FuncOp kernel,
                                             FragmentType target,
                                             ValueRange contributors) {
  SmallVector<Attribute> shape(target.getShape().begin(), target.getShape().end());
  SmallVector<Attribute> mappings(target.getAxisMaps().begin(),
                                  target.getAxisMaps().end());
  SmallVector<bool> refined(shape.size(), false);
  SmallVector<Value> authorities(shape.size());
  SmallVector<unsigned> authorityAxes(shape.size(), 0);
  bool changed = false;
  PhysicalProgramAnalysis analysis(kernel);
  for (Value contributor : contributors) {
    auto source = dyn_cast<FragmentType>(contributor.getType());
    if (!source)
      continue;
    SmallVector<bool> sourceAuthority(source.getShape().size(), false);
    bool hasAuthority = false;
    for (unsigned sourceAxis = 0; sourceAxis < source.getShape().size();
         ++sourceAxis) {
      PhysicalAxisRealizationFact realization =
          analysis.axisRealization(contributor, sourceAxis);
      auto sourceExtent =
          cast<PhysicalExprAttr>(source.getShape()[sourceAxis]);
      bool singleton =
          sourceExtent.getKind() ==
              static_cast<uint32_t>(PhysicalExprKind::Constant) &&
          sourceExtent.getValue() == 1;
      // A logical singleton may broadcast. A selected one-lane slice of a
      // larger logical range instead owns that consumer's physical extent.
      sourceAuthority[sourceAxis] = realization.hasExtentAuthority() &&
                                    !realization.constructionScalarSeed &&
                                    (!singleton ||
                                     (realization.physicalized &&
                                      llvm::any_of(realization.roots, [](MakeRangeOp range) {
                                        return !isProvablySingletonLogicalRange(range);
                                      })));
      hasAuthority |= sourceAuthority[sourceAxis];
    }
    if (!hasAuthority)
      continue;
    BroadcastProjection projection = queryAxisProjection(source, target);
    if (!projection.isExact())
      return failure();
    for (auto [targetAxis, sourceAxis] :
         llvm::enumerate(projection.targetToSource)) {
      if (!sourceAxis)
        continue;
      if (!sourceAuthority[*sourceAxis])
        continue;
      Attribute extent = source.getShape()[*sourceAxis];
      auto sourceMapping = cast<AxisMapAttr>(source.getAxisMaps()[*sourceAxis]);
      auto candidate = AxisMapAttr::get(
          target.getContext(), sourceMapping.getSourceId(),
          sourceMapping.getSourceAxis(), sourceMapping.getDimensionId(),
          targetAxis, sourceMapping.getDerived());
      if (refined[targetAxis]) {
        if (shape[targetAxis] != extent)
          return failure();
        auto selected = cast<AxisMapAttr>(mappings[targetAxis]);
        if (selected == candidate)
          continue;
        // A derived axis only records a broadcast occurrence.  Once an exact
        // non-derived source relation reaches the same physical axis, it is the
        // unique coordinate authority.  Two unrelated direct sources (or two
        // unrelated derived occurrences) remain ambiguous unless the canonical
        // result already selected one of them.
        if (selected.getDerived() != candidate.getDerived()) {
          if (!candidate.getDerived()) {
            mappings[targetAxis] = candidate;
            authorities[targetAxis] = contributor;
            authorityAxes[targetAxis] = *sourceAxis;
            changed = true;
          }
          continue;
        }
        auto canonical = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
        if (selected == canonical)
          continue;
        if (candidate == canonical) {
          mappings[targetAxis] = candidate;
          changed = true;
          continue;
        }
        // Distinct coordinate SSA graphs may be lockstep occurrences of one
        // result axis (for example row and column advanced indices driven by
        // the same filter traversal).  Producer identity and the intermediate
        // value's dimension are not enough to decide that relation.  Consume
        // the shared range analysis and retain the result's canonical axis only
        // when both current traversals are proven identical.
        if (authorities[targetAxis]) {
          SmallVector<Value> sources = {authorities[targetAxis], contributor};
          SmallVector<unsigned> axes = {authorityAxes[targetAxis], *sourceAxis};
          PhysicalLockstepTraversalFact lockstep =
              analysis.lockstepTraversal(sources, axes);
          if (lockstep.isExact()) {
            mappings[targetAxis] = canonical;
            changed = true;
            continue;
          }
        }
        return failure();
      }
      shape[targetAxis] = extent;
      mappings[targetAxis] = candidate;
      authorities[targetAxis] = contributor;
      authorityAxes[targetAxis] = *sourceAxis;
      refined[targetAxis] = true;
      changed = true;
    }
  }
  if (!changed)
    return target;
  return FragmentType::get(target.getContext(), target.getElementType(),
                           ArrayAttr::get(target.getContext(), shape),
                           ArrayAttr::get(target.getContext(), mappings),
                           target.getValidity(), target.getOwner());
}

FailureOr<FragmentType> refineAccessResultSchema(
    func::FuncOp kernel, FragmentType target, ValueRange coordinates) {
  SmallVector<Attribute> shape(target.getShape().begin(),
                               target.getShape().end());
  SmallVector<bool> refined(shape.size(), false);
  bool changed = false;
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<Value> fragmentCoordinates;
  for (Value coordinate : coordinates)
    if (isa<FragmentType>(coordinate.getType()))
      fragmentCoordinates.push_back(coordinate);
  bool cartesian = fragmentCoordinates.size() == target.getShape().size() &&
      llvm::all_of(fragmentCoordinates, [](Value coordinate) {
        return cast<FragmentType>(coordinate.getType()).getShape().size() == 1;
      });
  unsigned fragmentSlot = 0;
  for (Value coordinate : coordinates) {
    auto source = dyn_cast<FragmentType>(coordinate.getType());
    if (!source)
      continue;
    unsigned coordinateSlot = fragmentSlot++;
    bool positional = source.getOwner() == target.getOwner() &&
                      source.getShape() == target.getShape() &&
                      source.getAxisMaps() == target.getAxisMaps();
    for (unsigned sourceAxis = 0; sourceAxis < source.getShape().size();
         ++sourceAxis) {
      PhysicalAxisRealizationFact realization =
          analysis.axisRealization(coordinate, sourceAxis);
      auto extent = cast<PhysicalExprAttr>(source.getShape()[sourceAxis]);
      bool singleton =
          extent.getKind() ==
              static_cast<uint32_t>(PhysicalExprKind::Constant) &&
          extent.getValue() == 1;
      PhysicalRangeFact ranges = analysis.axisRanges(coordinate, sourceAxis);
      FailureOr<MakeRangeOp> range = queryExactLogicalRange(ranges);
      bool scalarCoordinate = singleton && succeeded(range) &&
                              (*range).getResult() == coordinate &&
                              isUnitStepRange(*range) &&
                              queryLaunchExpression((*range).getExtent()) == extent;
      if (!scalarCoordinate &&
          (!realization.hasExtentAuthority() ||
           realization.constructionScalarSeed || singleton))
        continue;

      auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
      std::optional<unsigned> targetAxis;
      if (positional) {
        // Matching access schemas already identify each occurrence. Refining
        // repeated-source extents still requires a unique propagation relation.
        targetAxis = sourceAxis;
      } else if (cartesian) {
        auto candidate = cast<AxisMapAttr>(target.getAxisMaps()[coordinateSlot]);
        if (sourceAxisIdentity(candidate) == sourceAxisIdentity(mapping) &&
            candidate.getDimensionId() == mapping.getDimensionId())
          targetAxis = coordinateSlot;
      }
      for (auto [axis, attribute] : llvm::enumerate(target.getAxisMaps())) {
        if ((positional || cartesian) && targetAxis)
          break;
        auto candidate = cast<AxisMapAttr>(attribute);
        if (!(sourceAxisIdentity(candidate) == sourceAxisIdentity(mapping)) ||
            candidate.getDimensionId() != mapping.getDimensionId())
          continue;
        if (targetAxis)
          return failure();
        targetAxis = axis;
      }
      if (!targetAxis) {
        PhysicalAxisProjection sourceProjection =
            queryFragmentAxis(target, sourceAxisIdentity(mapping));
        if (sourceProjection.isExact())
          targetAxis = sourceProjection.fragmentAxis;
        else if (sourceProjection.state == PhysicalFactState::Ambiguous)
          return failure();
      }
      // A coordinate may carry an ownership axis that the indexed result does
      // not expose.  Such an axis is not an access-result extent authority.
      if (!targetAxis)
        continue;
      if (refined[*targetAxis] && shape[*targetAxis] != extent) {
        emitError(coordinate.getLoc(), "access coordinate relations disagree")
            << "; payload_axis=" << *targetAxis
            << "; selected_extent=" << shape[*targetAxis]
            << "; coordinate_extent=" << extent
            << "; coordinate=" << coordinate;
        return failure();
      }
      shape[*targetAxis] = extent;
      refined[*targetAxis] = true;
      changed |= target.getShape()[*targetAxis] != extent;
    }
  }
  if (!changed)
    return target;
  // The index relation selected the access-result coordinate identity during
  // KIR-to-GPU construction.  Coordinates refine only its physical extents;
  // replacing those axis maps from a rank-aligned broadcast would create a
  // second, and potentially different, index relation.
  return FragmentType::get(target.getContext(), target.getElementType(),
                           ArrayAttr::get(target.getContext(), shape),
                           target.getAxisMaps(), target.getValidity(),
                           target.getOwner());
}

LogicalResult alignAccessResultRelations(func::FuncOp kernel) {
  WalkResult result = kernel.walk([&](Operation *operation) {
    ValueRange coordinates;
    Value result;
    if (auto load = dyn_cast<LoadOp>(operation)) {
      coordinates = load.getCoordinates();
      result = load.getResult();
    } else if (auto gather = dyn_cast<GatherOp>(operation)) {
      coordinates = gather.getCoordinates();
      result = gather.getResult();
    } else {
      return WalkResult::advance();
    }
    auto current = dyn_cast<FragmentType>(result.getType());
    if (!current)
      return WalkResult::advance();
    FailureOr<FragmentType> refined =
        refineAccessResultSchema(kernel, current, coordinates);
    if (failed(refined)) {
      InFlightDiagnostic diagnostic = operation->emitOpError(
          "access result has no unique physical coordinate projection");
      diagnostic << "; result=" << current;
      for (Value coordinate : coordinates)
        diagnostic << "; coordinate=" << coordinate.getType();
      return WalkResult::interrupt();
    }
    for (auto [axis, mapping] : llvm::enumerate((*refined).getAxisMaps())) {
      if (axis >= current.getShape().size() ||
          current.getShape()[axis] == (*refined).getShape()[axis])
        continue;
      retargetSourceExtent(
          result, sourceAxisIdentity(cast<AxisMapAttr>(mapping)),
          cast<PhysicalExprAttr>((*refined).getShape()[axis]));
    }
    result.setType(*refined);
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

static LogicalResult refreshReshapeRelation(ReshapeOp reshape);

LogicalResult alignPointwiseValueRelations(func::FuncOp kernel) {
  auto isParameterExtent = [](Attribute attribute) {
    auto extent = dyn_cast<PhysicalExprAttr>(attribute);
    return extent &&
           extent.getKind() ==
               static_cast<uint32_t>(PhysicalExprKind::Parameter);
  };
  auto sameSchema = [](FragmentType lhs, FragmentType rhs) {
    return lhs.getShape() == rhs.getShape() &&
           lhs.getAxisMaps() == rhs.getAxisMaps() &&
           lhs.getValidity() == rhs.getValidity() &&
           lhs.getOwner() == rhs.getOwner();
  };
  auto align = [&](Operation *operation, unsigned operandIndex,
                   FragmentType targetShape) -> LogicalResult {
    Value value = operation->getOperand(operandIndex);
    auto source = dyn_cast<FragmentType>(value.getType());
    if (source && sameSchema(source, targetShape))
      return success();
    auto target = FragmentType::get(
        kernel.getContext(), source ? source.getElementType() : value.getType(),
        targetShape.getShape(), targetShape.getAxisMaps(), targetShape.getValidity(),
        targetShape.getOwner());
    OpBuilder builder(operation);
    FailureOr<Value> replacement =
        projectFragmentValue(builder, operation->getLoc(), value, target);
    if (failed(replacement))
      return failure();
    operation->setOperand(operandIndex, *replacement);
    return success();
  };

  WalkResult result = kernel.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (isa<ReduceOp, ScanOp>(operation)) {
      if (alignReductionResultRelation(operation).wasInterrupted() ||
          alignReductionIdentityRelation(operation).wasInterrupted())
        return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (auto reshape = dyn_cast<ReshapeOp>(operation))
      return failed(refreshReshapeRelation(reshape)) ? WalkResult::interrupt()
                                                     : WalkResult::advance();
    if (auto transpose = dyn_cast<TransposeOp>(operation)) {
      auto source = cast<FragmentType>(transpose.getValue().getType());
      auto target = cast<FragmentType>(transpose.getResult().getType());
      SmallVector<Attribute> shape;
      for (int64_t input : transpose.getPermutation())
        shape.push_back(source.getShape()[input]);
      transpose.getResult().setType(FragmentType::get(
          kernel.getContext(), target.getElementType(),
          ArrayAttr::get(kernel.getContext(), shape), target.getAxisMaps(),
          target.getValidity(), target.getOwner()));
      return WalkResult::advance();
    }
    if (auto broadcast = dyn_cast<BroadcastOp>(operation)) {
      auto target = dyn_cast<FragmentType>(broadcast.getResult().getType());
      auto source = dyn_cast<FragmentType>(broadcast.getValue().getType());
      if (!target || !source)
        return WalkResult::advance();
      if (!queryBroadcastProjection(source, target).isExact()) {
        OpBuilder builder(broadcast);
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, broadcast.getLoc(), broadcast.getValue(), target);
        if (succeeded(projected)) {
          broadcast->setOperand(0, *projected);
          source = cast<FragmentType>(projected->getType());
        }
      }
      BroadcastProjection projection = queryAxisProjection(source, target);
      if (!projection.isExact()) {
        broadcast.emitOpError(
            projection.state == BroadcastProjectionState::Ambiguous
                ? "broadcast has an ambiguous physical source projection"
                : "broadcast has no physical source projection")
            << "; input=" << source << "; result=" << target;
        return WalkResult::interrupt();
      }
      // A non-singleton BroadcastOp is an explicit extent-preserving value
      // relation even when its input and result use distinct canonical
      // occurrence identities.  Pointwise ownership may first reach only one
      // side of that relation (for example a RegionFold summary schema).  Close
      // the already-selected parameter extent across the typed projection
      // before asking the verifier to observe the intermediate program.  A true
      // singleton broadcast remains an expansion and never acquires the
      // consumer's extent.
      for (auto [targetAxis, sourceAxis] :
           llvm::enumerate(projection.targetToSource)) {
        if (!sourceAxis ||
            source.getShape()[*sourceAxis] == target.getShape()[targetAxis])
          continue;
        auto sourceExtent =
            cast<PhysicalExprAttr>(source.getShape()[*sourceAxis]);
        bool singleton =
            sourceExtent.getKind() ==
                static_cast<uint32_t>(PhysicalExprKind::Constant) &&
            sourceExtent.getValue() == 1;
        if (singleton)
          continue;
        auto targetExtent =
            cast<PhysicalExprAttr>(target.getShape()[targetAxis]);
        bool targetSingleton =
            targetExtent.getKind() ==
                static_cast<uint32_t>(PhysicalExprKind::Constant) &&
            targetExtent.getValue() == 1;
        if (targetSingleton) {
          PhysicalProgramAnalysis analysis(kernel);
          auto sourceMap = cast<AxisMapAttr>(source.getAxisMaps()[*sourceAxis]);
          auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
          bool derivedOccurrence =
              targetMap.getDerived() && sourceMap.getDimensionId() > 0 &&
              sourceMap.getDimensionId() == targetMap.getDimensionId();
          if (analysis.axisRealization(broadcast.getResult(), targetAxis)
                  .constructionScalarSeed || derivedOccurrence) {
            retargetSourceExtent(broadcast.getResult(),
                                 sourceAxisIdentity(targetMap), sourceExtent);
            target = cast<FragmentType>(broadcast.getResult().getType());
            continue;
          }
          broadcast.emitOpError(
              "broadcast cannot contract a non-singleton physical axis")
              << "; input=" << source << "; result=" << target;
          return WalkResult::interrupt();
        }
        bool sourceParameter =
            isParameterExtent(source.getShape()[*sourceAxis]);
        bool targetParameter = isParameterExtent(targetExtent);
        if (!sourceParameter && !targetParameter) {
          PhysicalProgramAnalysis analysis(kernel);
          auto input = analysis.axisRealization(broadcast.getValue(), *sourceAxis);
          auto output = analysis.axisRealization(broadcast.getResult(), targetAxis);
          if (input.hasExtentAuthority() && !input.constructionScalarSeed &&
              !output.hasExtentAuthority()) {
            auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
            retargetSourceExtent(broadcast.getResult(),
                                 sourceAxisIdentity(targetMap), sourceExtent);
            target = cast<FragmentType>(broadcast.getResult().getType());
            continue;
          }
        }
        if (sourceParameter == targetParameter) {
          broadcast.emitOpError(
              "broadcast has conflicting non-singleton physical extents")
              << "; input=" << source << "; result=" << target;
          return WalkResult::interrupt();
        }
        auto sourceMap =
            cast<AxisMapAttr>(source.getAxisMaps()[*sourceAxis]);
        auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
        if (sourceMap.getDimensionId() <= 0 || targetMap.getDimensionId() <= 0) {
          broadcast.emitOpError(
              "broadcast extent refinement has no logical occurrence authority");
          return WalkResult::interrupt();
        }
        if (targetParameter)
          retargetSourceExtent(
              broadcast.getValue(), sourceAxisIdentity(sourceMap),
              cast<PhysicalExprAttr>(target.getShape()[targetAxis]));
        else
          retargetSourceExtent(broadcast.getResult(),
                               sourceAxisIdentity(targetMap), sourceExtent);
        source = cast<FragmentType>(broadcast.getValue().getType());
        target = cast<FragmentType>(broadcast.getResult().getType());
      }
      FailureOr<FragmentType> refined =
          refinePhysicalSchema(kernel, target, ValueRange{broadcast.getValue()});
      if (failed(refined)) {
        broadcast.emitOpError(
            "broadcast result has no unique physical source projection");
        return WalkResult::interrupt();
      }
      // Broadcast has two independent typed relations: its input supplies the
      // physical extents, while the result type supplies the logical occurrence
      // seen by consumers.  Adopting the input's AxisMap would erase an explicit
      // output/index relation (for example a reshaped value stored into a view)
      // and force a later access pass to reconstruct it.
      broadcast.getResult().setType(FragmentType::get(
          kernel.getContext(), target.getElementType(), (*refined).getShape(),
          target.getAxisMaps(), target.getValidity(), target.getOwner()));
      return WalkResult::advance();
    }
    FragmentType target;
    if (operation->getNumResults() == 1)
      target = dyn_cast<FragmentType>(operation->getResult(0).getType());
    if (!isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(
            operation))
      return WalkResult::advance();
    if (!target && operation->getNumResults() == 1 &&
        isa<IntegerType, FloatType, IndexType>(
            operation->getResult(0).getType())) {
      FragmentType prototype;
      for (Value operand : operation->getOperands())
        if ((prototype = dyn_cast<FragmentType>(operand.getType())))
          break;
      if (prototype) {
        target = FragmentType::get(
            kernel.getContext(), operation->getResult(0).getType(),
            prototype.getShape(), prototype.getAxisMaps(),
            prototype.getValidity(), prototype.getOwner());
        operation->getResult(0).setType(target);
      }
    }
    if (!target)
      return WalkResult::advance();
    if (isa<UnaryOp, CastOp, BitcastOp>(operation) &&
        operation->getNumOperands() == 1) {
      auto source = dyn_cast<FragmentType>(operation->getOperand(0).getType());
      if (source) {
        operation->getResult(0).setType(FragmentType::get(
            kernel.getContext(), target.getElementType(), source.getShape(),
            source.getAxisMaps(), source.getValidity(), source.getOwner()));
        return WalkResult::advance();
      }
    }
    FailureOr<FragmentType> refined =
        refinePhysicalSchema(kernel, target, operation->getOperands());
    if (failed(refined)) {
      InFlightDiagnostic diagnostic = operation->emitOpError(
          "pointwise result has no unique physical operand projection");
      diagnostic << "; result=" << target;
      for (Value operand : operation->getOperands())
        diagnostic << "; operand=" << operand.getType();
      return WalkResult::interrupt();
    }
    for (auto [axis, mapping] : llvm::enumerate((*refined).getAxisMaps())) {
      if (axis >= target.getShape().size() ||
          target.getShape()[axis] == (*refined).getShape()[axis])
        continue;
      int64_t dimension = cast<AxisMapAttr>(mapping).getDimensionId();
      if (dimension <= 0) {
        operation->emitOpError(
            "pointwise result refinement has no logical dimension authority");
        return WalkResult::interrupt();
      }
      retargetSourceExtent(
          operation->getResult(0),
          sourceAxisIdentity(cast<AxisMapAttr>(target.getAxisMaps()[axis])),
          cast<PhysicalExprAttr>((*refined).getShape()[axis]));
    }
    target = *refined;
    operation->getResult(0).setType(target);
    for (unsigned index = 0; index < operation->getNumOperands(); ++index)
      if (failed(align(operation, index, target))) {
        operation->emitOpError(
            "pointwise operand cannot adopt the result relation")
            << "; operand_index=" << index
            << "; operand=" << operation->getOperand(index).getType()
            << "; result=" << target;
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult alignAccessValueRelations(func::FuncOp kernel) {
  auto predicateType = [&](FragmentType value) {
    return FragmentType::get(
        kernel.getContext(), IntegerType::get(kernel.getContext(), 1),
        value.getShape(), value.getAxisMaps(), value.getValidity(),
        value.getOwner());
  };
  auto project = [&](OpBuilder &builder, Location location, Value value,
                     Type target) -> FailureOr<Value> {
    if (!value)
      return failure();
    return projectPhysicalValueToSchema(builder, location, value, target);
  };
  auto alignCoordinates = [&](auto access, Type valueType) -> LogicalResult {
    auto payload = dyn_cast<FragmentType>(valueType);
    if (!payload)
      return success();
    PhysicalProgramAnalysis analysis(kernel);
    for (auto [slot, coordinate] : llvm::enumerate(access.getCoordinates())) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      if (!type)
        continue;
      SmallVector<Attribute> shape(type.getShape().begin(), type.getShape().end());
      bool changed = false;
      for (auto [axis, attribute] : llvm::enumerate(type.getAxisMaps())) {
        auto mapping = cast<AxisMapAttr>(attribute);
        auto projection = queryFragmentAxis(payload, sourceAxisIdentity(mapping));
        if (!projection.isExact() || projection.dimensionId != mapping.getDimensionId() ||
            shape[axis] == payload.getShape()[projection.fragmentAxis])
          continue;
        PhysicalRangeFact ranges = analysis.axisRanges(coordinate, axis);
        if (!ranges.isExact() || !ranges.roots.empty())
          continue;
        shape[axis] = payload.getShape()[projection.fragmentAxis];
        changed = true;
      }
      if (!changed)
        continue;
      auto target = FragmentType::get(
          kernel.getContext(), type.getElementType(), ArrayAttr::get(kernel.getContext(), shape),
          type.getAxisMaps(), type.getValidity(), type.getOwner());
      OpBuilder builder(access);
      FailureOr<Value> aligned = project(builder, access.getLoc(), coordinate, target);
      if (failed(aligned))
        return access.emitOpError("cannot align broadcast coordinate with its access schema")
               << "; coordinate=" << coordinate;
      access.getCoordinatesMutable().slice(slot, 1).assign(*aligned);
    }
    return success();
  };

  SmallVector<LoadOp> loads;
  SmallVector<GatherOp> gathers;
  SmallVector<StoreOp> stores;
  kernel.walk([&](LoadOp load) { loads.push_back(load); });
  kernel.walk([&](GatherOp gather) { gathers.push_back(gather); });
  kernel.walk([&](StoreOp store) { stores.push_back(store); });

  for (LoadOp load : loads) {
    if (failed(alignCoordinates(load, load.getResult().getType())))
      return failure();
    if (!load.getValid() && !load.getFill())
      continue;
    if (!load.getValid())
      return load.emitOpError("load fill has no validity authority");
    OpBuilder builder(load);
    Type valueType = load.getResult().getType();
    Type validType = builder.getI1Type();
    if (auto fragment = dyn_cast<FragmentType>(valueType))
      validType = predicateType(fragment);
    FailureOr<Value> valid = project(builder, load.getLoc(), load.getValid(),
                                     validType);
    if (failed(valid))
      return load.emitOpError("cannot align load validity with its value schema");
    FailureOr<Value> fill = failure();
    if (load.getFill())
      fill = project(builder, load.getLoc(), load.getFill(), valueType);
    else if (auto fragment = dyn_cast<FragmentType>(valueType))
      fill = materializeZeroFragment(builder, load.getLoc(), fragment);
    else {
      TypedAttr zero;
      if (auto integer = dyn_cast<IntegerType>(valueType))
        zero = builder.getIntegerAttr(integer, 0);
      else if (auto floating = dyn_cast<FloatType>(valueType))
        zero = builder.getFloatAttr(floating, 0.0);
      else if (isa<IndexType>(valueType))
        zero = builder.getIndexAttr(0);
      if (zero)
        fill = materializeScalarConstant(builder, load.getLoc(), zero, valueType);
    }
    if (failed(fill))
      return load.emitOpError("cannot align load fill with its value schema");
    if (*valid == load.getValid() && *fill == load.getFill())
      continue;
    auto replacement = builder.create<LoadOp>(
        load.getLoc(), valueType, load.getResource(), load.getCoordinates(),
        *valid, *fill, load.getSourceAxes());
    if (Attribute origin = load->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    load.getResult().replaceAllUsesWith(replacement.getResult());
    load.erase();
  }

  for (GatherOp gather : gathers) {
    if (failed(alignCoordinates(gather, gather.getResult().getType())))
      return failure();
    if (!gather.getValid() && !gather.getFill())
      continue;
    if (!gather.getValid())
      return gather.emitOpError("gather fill has no validity authority");
    OpBuilder builder(gather);
    Type valueType = gather.getResult().getType();
    auto fragment = dyn_cast<FragmentType>(valueType);
    if (!fragment) {
      if (gather.getValid() && gather.getFill())
        continue;
      return gather.emitOpError(
          "scalar gather validity and fill must remain paired");
    }
    FailureOr<Value> valid = project(
        builder, gather.getLoc(), gather.getValid(), predicateType(fragment));
    if (failed(valid))
      return gather.emitOpError(
          "cannot align gather validity with its value schema");
    FailureOr<Value> fill =
        gather.getFill()
            ? project(builder, gather.getLoc(), gather.getFill(), valueType)
            : materializeZeroFragment(builder, gather.getLoc(), fragment);
    if (failed(fill))
      return gather.emitOpError("cannot align gather fill with its value schema");
    if (*valid == gather.getValid() && *fill == gather.getFill())
      continue;
    auto replacement = builder.create<GatherOp>(
        gather.getLoc(), valueType, gather.getSource(), gather.getCoordinates(),
        *valid, *fill, gather.getSourceAxes());
    if (Attribute origin = gather->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement.getResult());
    gather.erase();
  }

  for (StoreOp store : stores) {
    if (failed(alignCoordinates(store, store.getValue().getType())))
      return failure();
    if (!store.getValid())
      continue;
    auto currentType = dyn_cast<FragmentType>(store.getValue().getType());
    if (!currentType) {
      auto validity = dyn_cast<FragmentType>(store.getValid().getType());
      if (!validity || !llvm::all_of(validity.getShape(), [](Attribute extent) {
            return constantPhysicalExpression(cast<PhysicalExprAttr>(extent)) == 1;
          }))
        continue;
      // Ownership can retain a one-lane predicate for a scalar write.  Adopt
      // that schema without widening the write into additional lanes.
      currentType = FragmentType::get(
          kernel.getContext(), store.getValue().getType(), validity.getShape(),
          validity.getAxisMaps(), validity.getValidity(), validity.getOwner());
      OpBuilder builder(store);
      FailureOr<Value> value = project(builder, store.getLoc(), store.getValue(),
                                       currentType);
      if (failed(value))
        return store.emitOpError("cannot align scalar store with its one-lane validity");
      store.getValueMutable().assign(*value);
    }
    // A value whose extent is already selected by an exact range or verified
    // reshape is the physical data authority for the write.  Retarget the
    // address relation to that extent before reconciling schemas.  This keeps
    // extents and coordinate provenance separate: the value does not acquire
    // the view's source identity, and the address does not overwrite a verified
    // row-major reshape decision.
    PhysicalProgramAnalysis analysis(kernel);
    SmallVector<Value> fragmentCoordinates;
    for (Value coordinate : store.getCoordinates())
      if (isa<FragmentType>(coordinate.getType()))
        fragmentCoordinates.push_back(coordinate);
    bool cartesian = fragmentCoordinates.size() == currentType.getShape().size() &&
        llvm::all_of(fragmentCoordinates, [](Value coordinate) {
          return cast<FragmentType>(coordinate.getType()).getShape().size() == 1;
        });
    for (unsigned axis = 0; axis < currentType.getShape().size(); ++axis) {
      PhysicalAxisRealizationFact realization =
          analysis.axisRealization(store.getValue(), axis);
      if (!realization.hasExtentAuthority())
        continue;
      auto mapping = cast<AxisMapAttr>(currentType.getAxisMaps()[axis]);
      auto extent = cast<PhysicalExprAttr>(currentType.getShape()[axis]);
      auto positional = cartesian
          ? queryFragmentAxis(fragmentCoordinates[axis].getType(), sourceAxisIdentity(mapping))
          : PhysicalAxisProjection{};
      for (Value coordinate : store.getCoordinates()) {
        if (positional.isExact() &&
            positional.dimensionId == mapping.getDimensionId() &&
            coordinate != fragmentCoordinates[axis])
          continue;
        auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
        if (!coordinateType)
          continue;
        PhysicalAxisProjection projection =
            queryFragmentAxis(coordinateType, sourceAxisIdentity(mapping));
        if (!projection.isExact() ||
            projection.dimensionId != mapping.getDimensionId() ||
            coordinateType.getShape()[projection.fragmentAxis] == extent)
          continue;
        retargetSourceExtent(coordinate, projection.source, extent);
      }
    }
    currentType = cast<FragmentType>(store.getValue().getType());
    OpBuilder builder(store);
    FailureOr<FragmentType> valueType =
        refineAccessResultSchema(kernel, currentType, store.getCoordinates());
    if (failed(valueType))
      return store.emitOpError(
          "store value has no unique physical coordinate projection");
    for (auto [axis, mapping] : llvm::enumerate((*valueType).getAxisMaps())) {
      if (axis >= currentType.getShape().size() ||
          currentType.getShape()[axis] == (*valueType).getShape()[axis])
        continue;
      int64_t dimension = cast<AxisMapAttr>(mapping).getDimensionId();
      if (dimension <= 0)
        return store.emitOpError(
            "store coordinate refinement has no logical dimension authority");
      retargetSourceExtent(
          store.getValue(), sourceAxisIdentity(cast<AxisMapAttr>(mapping)),
          cast<PhysicalExprAttr>((*valueType).getShape()[axis]));
    }
    FailureOr<Value> value = project(builder, store.getLoc(), store.getValue(),
                                     *valueType);
    if (failed(value)) {
      InFlightDiagnostic diagnostic = store.emitOpError(
          "cannot align store value with its coordinate schema");
      diagnostic << "; value=" << store.getValue().getType()
                 << "; coordinate_schema=" << *valueType;
      for (Value coordinate : store.getCoordinates())
        diagnostic << "; coordinate=" << coordinate.getType();
      return failure();
    }
    FailureOr<Value> valid = project(
        builder, store.getLoc(), store.getValid(), predicateType(*valueType));
    if (failed(valid)) {
      InFlightDiagnostic diagnostic = store.emitOpError(
          "cannot align store validity with its value schema");
      diagnostic << "; value=" << *valueType
                 << "; validity=" << store.getValid().getType();
      if (Operation *producer = store.getValue().getDefiningOp())
        diagnostic << "; value_producer=" << producer->getName();
      return failure();
    }
    if (*value == store.getValue() && *valid == store.getValid())
      continue;
    auto replacement = builder.create<StoreOp>(
        store.getLoc(), store.getResource(), store.getCoordinates(),
        *value, *valid, store.getSourceAxes());
    if (Attribute origin = store->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    store.erase();
  }
  return success();
}

static LogicalResult refreshReshapeRelation(ReshapeOp reshape) {
    auto kernel = reshape->getParentOfType<func::FuncOp>();
    // Reassociation is the typed row-major relation selected from canonical
    // KIR.  Refinement passes may change physical extents, but they may not
    // rediscover and replace that relation from the new shapes: doing so turns
    // physical shape coincidence into a second algorithm authority.  Validate
    // the preserved carrier here; the mutating pass that changed an extent is
    // responsible for updating both sides of each existing group.
    auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
    auto target = dyn_cast<FragmentType>(reshape.getResult().getType());
    if (!source || !target) {
      reshape.emitOpError(
          "reshape relation requires physical fragment operands and result");
      return failure();
    }
    unsigned logicalSourceRank = 0;
    unsigned logicalResultRank = 0;
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = dyn_cast<ReshapeGroupAttr>(attribute);
      if (!group) {
        reshape.emitOpError("reshape relation contains an untyped group");
        return failure();
      }
      if (!group.getSourceAxes().empty())
        logicalSourceRank = std::max(
            logicalSourceRank,
            static_cast<unsigned>(
                group.getSourceAxes().asArrayRef().back() + 1));
      if (!group.getResultAxes().empty())
        logicalResultRank = std::max(
            logicalResultRank,
            static_cast<unsigned>(
                group.getResultAxes().asArrayRef().back() + 1));
    }
    if (logicalSourceRank > source.getShape().size() ||
        logicalResultRank > target.getShape().size()) {
      reshape.emitOpError(
          "reshape relation rank exceeds the current physical fragments");
      return failure();
    }
    unsigned sourcePrefix = source.getShape().size() - logicalSourceRank;
    unsigned resultPrefix = target.getShape().size() - logicalResultRank;
    SmallVector<Attribute> resultShape(target.getShape().begin(),
                                       target.getShape().end());
    if (sourcePrefix == resultPrefix &&
        llvm::equal(source.getAxisMaps().getValue().take_front(sourcePrefix),
                    target.getAxisMaps().getValue().take_front(resultPrefix)))
      llvm::copy(source.getShape().getValue().take_front(sourcePrefix),
                 resultShape.begin());
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = cast<ReshapeGroupAttr>(attribute);
      ArrayRef<int64_t> sourceAxes = group.getSourceAxes().asArrayRef();
      ArrayRef<int64_t> resultAxes = group.getResultAxes().asArrayRef();
      if (resultAxes.size() == 1) {
        resultShape[resultPrefix + resultAxes.front()] = productExtent(
            kernel.getContext(), source.getShape().getValue(), sourceAxes,
            sourcePrefix);
        continue;
      }
      if (sourceAxes.size() != 1 || resultAxes.empty())
        continue;
      SmallVector<int64_t> nonUnitAxes;
      for (int64_t axis : resultAxes) {
        auto extent = cast<PhysicalExprAttr>(
            resultShape[resultPrefix + axis]);
        if (extent.getKind() !=
                static_cast<uint32_t>(PhysicalExprKind::Constant) ||
            extent.getValue() != 1)
          nonUnitAxes.push_back(axis);
      }
      if (nonUnitAxes.size() == 1)
        resultShape[resultPrefix + nonUnitAxes.front()] =
            cast<PhysicalExprAttr>(
                source.getShape()[sourcePrefix + sourceAxes.front()]);
    }
    reshape.getResult().setType(FragmentType::get(
        kernel.getContext(), target.getElementType(),
        ArrayAttr::get(kernel.getContext(), resultShape), target.getAxisMaps(),
        target.getValidity(), target.getOwner()));
    return reshape.verify();
}

LogicalResult refreshReshapeRelations(func::FuncOp kernel) {
  WalkResult result = kernel.walk([&](ReshapeOp reshape) {
    return failed(refreshReshapeRelation(reshape)) ? WalkResult::interrupt()
                                                  : WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult alignAggregateValueRelations(func::FuncOp kernel) {
  kernel.walk([&](MakeRecordOp record) {
    RecordType current = record.getResult().getType();
    SmallVector<Attribute> fields;
    fields.reserve(record.getFields().size());
    for (Value field : record.getFields())
      fields.push_back(TypeAttr::get(field.getType()));
    record.getResult().setType(RecordType::get(
        kernel.getContext(), current.getFieldNames(),
        ArrayAttr::get(kernel.getContext(), fields), current.getOwner()));
  });
  std::function<FailureOr<Type>(Type, Type, Type)> joinTypes =
      [&](Type current, Type lhs, Type rhs) -> FailureOr<Type> {
    if (auto lhsFragment = dyn_cast<FragmentType>(lhs)) {
      auto rhsFragment = dyn_cast<FragmentType>(rhs);
      auto currentFragment = dyn_cast<FragmentType>(current);
      if (!rhsFragment || !currentFragment ||
          lhsFragment.getElementType() != rhsFragment.getElementType() ||
          lhsFragment.getElementType() != currentFragment.getElementType() ||
          lhsFragment.getShape().size() != rhsFragment.getShape().size() ||
          lhsFragment.getShape().size() != currentFragment.getShape().size() ||
          lhsFragment.getAxisMaps() != rhsFragment.getAxisMaps() ||
          lhsFragment.getValidity() != rhsFragment.getValidity() ||
          lhsFragment.getOwner() != rhsFragment.getOwner())
        return failure();
      SmallVector<Attribute> shape;
      for (auto [left, right] :
           llvm::zip(lhsFragment.getShape(), rhsFragment.getShape())) {
        if (left == right) {
          shape.push_back(left);
          continue;
        }
        auto leftExtent = cast<PhysicalExprAttr>(left);
        auto rightExtent = cast<PhysicalExprAttr>(right);
        bool leftUnit =
            leftExtent.getKind() ==
                static_cast<uint32_t>(PhysicalExprKind::Constant) &&
            leftExtent.getValue() == 1;
        bool rightUnit =
            rightExtent.getKind() ==
                static_cast<uint32_t>(PhysicalExprKind::Constant) &&
            rightExtent.getValue() == 1;
        if (leftUnit == rightUnit)
          return failure();
        shape.push_back(leftUnit ? right : left);
      }
      return Type(FragmentType::get(
          kernel.getContext(), lhsFragment.getElementType(),
          ArrayAttr::get(kernel.getContext(), shape), lhsFragment.getAxisMaps(),
          lhsFragment.getValidity(), lhsFragment.getOwner()));
    }
    auto lhsRecord = dyn_cast<RecordType>(lhs);
    auto rhsRecord = dyn_cast<RecordType>(rhs);
    auto currentRecord = dyn_cast<RecordType>(current);
    if (lhsRecord || rhsRecord || currentRecord) {
      if (!lhsRecord || !rhsRecord || !currentRecord ||
          lhsRecord.getFieldNames() != rhsRecord.getFieldNames() ||
          lhsRecord.getFieldNames() != currentRecord.getFieldNames() ||
          lhsRecord.getFieldTypes().size() != rhsRecord.getFieldTypes().size() ||
          lhsRecord.getFieldTypes().size() !=
              currentRecord.getFieldTypes().size() ||
          lhsRecord.getOwner() != rhsRecord.getOwner())
        return failure();
      SmallVector<Attribute> fields;
      for (auto [base, left, right] :
           llvm::zip(currentRecord.getFieldTypes(), lhsRecord.getFieldTypes(),
                     rhsRecord.getFieldTypes())) {
        FailureOr<Type> joined = joinTypes(
            cast<TypeAttr>(base).getValue(), cast<TypeAttr>(left).getValue(),
            cast<TypeAttr>(right).getValue());
        if (failed(joined))
          return failure();
        fields.push_back(TypeAttr::get(*joined));
      }
      return Type(RecordType::get(
          kernel.getContext(), lhsRecord.getFieldNames(),
          ArrayAttr::get(kernel.getContext(), fields), lhsRecord.getOwner()));
    }
    return current == lhs && lhs == rhs ? FailureOr<Type>(current)
                                        : FailureOr<Type>(failure());
  };
  WalkResult branches = kernel.walk([&](scf::IfOp branch) {
    if (branch.getNumResults() == 0)
      return WalkResult::advance();
    auto thenYield = dyn_cast<scf::YieldOp>(branch.thenBlock()->getTerminator());
    auto elseYield = dyn_cast<scf::YieldOp>(branch.elseBlock()->getTerminator());
    if (!thenYield || !elseYield ||
        thenYield.getResults().size() != branch.getNumResults() ||
        elseYield.getResults().size() != branch.getNumResults())
      return WalkResult::interrupt();
    for (unsigned index = 0; index < branch.getNumResults(); ++index) {
      Value leftValue = thenYield.getResults()[index];
      Value rightValue = elseYield.getResults()[index];
      Type leftType = leftValue.getType(), rightType = rightValue.getType();
      UniformValueAnalysis uniform(describeUniformValue);
      bool leftUniform = static_cast<bool>(uniform.evaluate(leftValue));
      bool rightUniform = static_cast<bool>(uniform.evaluate(rightValue));
      if (leftUniform != rightUniform) {
        Type &uniformType = leftUniform ? leftType : rightType;
        if (auto fragment = dyn_cast<FragmentType>(uniformType)) {
          // A uniform branch adopts the other branch's selected physical
          // extents, while retaining its axis, dtype and ownership obligations.
          auto unit = PhysicalExprAttr::get(
              kernel.getContext(), static_cast<uint32_t>(PhysicalExprKind::Constant),
              1, StringAttr::get(kernel.getContext(), ""),
              ArrayAttr::get(kernel.getContext(), {}));
          SmallVector<Attribute> units(fragment.getShape().size(), unit);
          uniformType = FragmentType::get(
              kernel.getContext(), fragment.getElementType(),
              ArrayAttr::get(kernel.getContext(), units), fragment.getAxisMaps(),
              fragment.getValidity(), fragment.getOwner());
        }
      }
      FailureOr<Type> target = joinTypes(
          branch.getResult(index).getType(), leftType, rightType);
      if (failed(target)) {
        branch.emitOpError(
            "control-flow branches have no unique physical result relation")
            << "; result_index=" << index
            << "; then=" << thenYield.getResults()[index].getType()
            << "; else=" << elseYield.getResults()[index].getType();
        return WalkResult::interrupt();
      }
      OpBuilder thenBuilder(thenYield);
      FailureOr<Value> projectedThen = projectPhysicalValueToSchema(
          thenBuilder, branch.getLoc(), thenYield.getResults()[index], *target);
      OpBuilder elseBuilder(elseYield);
      FailureOr<Value> projectedElse = projectPhysicalValueToSchema(
          elseBuilder, branch.getLoc(), elseYield.getResults()[index], *target);
      if (failed(projectedThen) || failed(projectedElse)) {
        branch.emitOpError(
            "control-flow branch cannot adopt its joined physical relation")
            << "; result_index=" << index << "; target=" << *target;
        return WalkResult::interrupt();
      }
      thenYield->setOperand(index, *projectedThen);
      elseYield->setOperand(index, *projectedElse);
      branch.getResult(index).setType(*target);
    }
    return WalkResult::advance();
  });
  if (branches.wasInterrupted())
    return failure();
  WalkResult result = kernel.walk([&](RegionFoldOp fold) {
    if (!llvm::hasSingleElement(fold.getSummarize()) ||
        !llvm::hasSingleElement(fold.getCombine()))
      return WalkResult::interrupt();
    auto summarizeYield =
        dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
    if (!summarizeYield ||
        summarizeYield.getValues().size() != fold.getIdentityCount())
      return WalkResult::interrupt();
    Block &combine = fold.getCombine().front();
    if (combine.getNumArguments() != 2 * fold.getIdentityCount())
      return WalkResult::interrupt();
    OpBuilder builder(fold);
    for (unsigned index = 0; index < fold.getIdentityCount(); ++index) {
      Type target = summarizeYield.getValues()[index].getType();
      SmallVector<std::pair<int64_t, PhysicalExprAttr>> dimensions;
      std::function<LogicalResult(Type)> collectDimensions =
          [&](Type type) -> LogicalResult {
        if (auto fragment = dyn_cast<FragmentType>(type)) {
          for (auto [axis, mapping] :
               llvm::enumerate(fragment.getAxisMaps())) {
            int64_t dimension = cast<AxisMapAttr>(mapping).getDimensionId();
            if (dimension <= 0)
              continue;
            auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
            auto found = llvm::find_if(dimensions, [&](const auto &entry) {
              return entry.first == dimension;
            });
            if (found != dimensions.end()) {
              if (found->second != extent)
                return failure();
              continue;
            }
            dimensions.emplace_back(dimension, extent);
          }
          return success();
        }
        if (auto record = dyn_cast<RecordType>(type))
          for (Attribute field : record.getFieldTypes())
            if (failed(collectDimensions(cast<TypeAttr>(field).getValue())))
              return failure();
        return success();
      };
      if (failed(collectDimensions(target))) {
        fold.emitOpError(
            "region-fold summary has conflicting physical dimension extents")
            << "; summary_index=" << index << "; summary=" << target;
        return WalkResult::interrupt();
      }
      for (auto [dimension, extent] : dimensions)
        retargetDimensionExtent(fold.getResult(index), dimension, extent);
      unsigned identity = fold.getSourceCount() + index;
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, fold.getLoc(), fold.getInputs()[identity], target);
      if (failed(projected)) {
        fold.emitOpError(
            "region-fold identity cannot adopt its summary relation");
        return WalkResult::interrupt();
      }
      fold->setOperand(identity, *projected);
      fold.getResult(index).setType(target);
      combine.getArgument(index).setType(target);
      combine.getArgument(fold.getIdentityCount() + index).setType(target);
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  WalkResult loops = kernel.walk([&](scf::ForOp loop) {
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield || yield.getResults().size() != loop.getInitArgs().size())
      return WalkResult::interrupt();
    for (unsigned index = 0; index < loop.getInitArgs().size(); ++index) {
      Value init = loop.getInitArgs()[index];
      Value yielded = yield.getResults()[index];
      // The loop-body update is the executable relation selected by the
      // transformation that produced it.  The init value is an identity/seed
      // at the structural boundary and must be projected to that relation; the
      // boundary must not rank shapes or independently choose a competing one.
      Type target = yielded.getType();
      OpBuilder initBuilder(loop);
      FailureOr<Value> projectedInit = projectPhysicalValueToSchema(
          initBuilder, loop.getLoc(), init, target);
      OpBuilder yieldBuilder(yield);
      FailureOr<Value> projectedYield = projectPhysicalValueToSchema(
          yieldBuilder, loop.getLoc(), yielded, target);
      if (failed(projectedInit) || failed(projectedYield)) {
        loop.emitOpError(
            "loop-carried value cannot adopt its unique physical relation")
            << "; init=" << init.getType() << "; yield=" << yielded.getType()
            << "; target=" << target;
        return WalkResult::interrupt();
      }
      loop.getInitArgsMutable()[index].assign(*projectedInit);
      yield->setOperand(index, *projectedYield);
      loop.getRegionIterArgs()[index].setType(target);
      loop.getResult(index).setType(target);
    }
    return WalkResult::advance();
  });
  if (loops.wasInterrupted())
    return failure();
  // Structured arguments/results are the record-schema authority.  Refresh
  // projections only after those schemas have been aligned; doing this before
  // the region owner leaves combine-body fields one refinement behind.
  kernel.walk([&](ExtractOp extract) {
    RecordType record = extract.getRecord().getType();
    if (extract.getField() < record.getFieldTypes().size())
      extract.getResult().setType(
          cast<TypeAttr>(record.getFieldTypes()[extract.getField()]).getValue());
  });
  return success();
}

FailureOr<Value> resolveLogicalRangeEnd(func::FuncOp kernel,
                                        MakeRangeOp range) {
  (void)kernel;
  return range.getLogicalStop();
}

static void retargetExtent(Value root, AxisSelector selects,
                           PhysicalExprAttr extent,
                           bool followLogicalDimension) {
  SmallVector<Attribute> previousExtents;
  collectSelectedExtents(root.getType(), selects, previousExtents);
  if (previousExtents.empty())
    return;
  SmallVector<Attribute> connectedExtents(previousExtents.begin(),
                                          previousExtents.end());
  if (!llvm::is_contained(connectedExtents, Attribute(extent)))
    connectedExtents.push_back(extent);
  SmallVector<Value> worklist{root};
  llvm::DenseMap<Value, Type> visitedTypes;
  SmallVector<std::pair<Value, AxisMapAttr>> valueAliases;
  auto isSegmentSourceSlice = [&](Value value) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument)
      return false;
    Operation *parent = argument.getOwner()->getParentOp();
    if (auto fold = dyn_cast_or_null<RegionFoldOp>(parent))
      return argument.getOwner()->getParent() == &fold.getSummarize() &&
             argument.getArgNumber() < fold.getSourceCount() &&
             selectsSegmentAxis(argument.getType(), fold.getAxis(), selects);
    if (auto scan = dyn_cast_or_null<RegionScanOp>(parent))
      return (argument.getOwner()->getParent() == &scan.getSummarize() ||
              argument.getOwner()->getParent() == &scan.getEmit()) &&
             argument.getArgNumber() < scan.getSourceCount() &&
             selectsSegmentAxis(argument.getType(), scan.getAxis(), selects);
    return false;
  };
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    auto previousFragment = dyn_cast<FragmentType>(value.getType());
    auto [visited, inserted] = visitedTypes.try_emplace(value, value.getType());
    if (!inserted && visited->second == value.getType())
      continue;
    visited->second = value.getType();
    // A structured segment slice has one exact extent authority: the segment
    // parameter owned by its region_fold/region_scan operation.  Retargeting
    // stops only for that segmented axis; every non-segment axis must preserve
    // the source schema across the helper boundary.
    if (isSegmentSourceSlice(value))
      continue;
    // Each make_range is an independent physical traversal authority.  A
    // source-axis extent selected for one range may flow through its users,
    // but must not cross a shared consumer and rewrite a different range that
    // happens to carry the same logical provenance.
    if (value != root && value.getDefiningOp<MakeRangeOp>())
      continue;
    // A reduction or other structured operation can consume an axis and
    // produce a scalar/record that no longer carries it.  The extent decision
    // ends there; following the scalar into a later broadcast would conflate a
    // new traversal with the one being retargeted.
    if (!(followLogicalDimension
              ? carriesSelectedAxis(value.getType(), selects)
              : carriesExtent(value.getType(), selects, connectedExtents)))
      continue;
    // The root is the value whose physical extent this decision owns.  A
    // construction-time singleton on that root is only a conservative seed;
    // it cannot veto the decision and leave a make_range type out of sync with
    // its extent operand.  The guard applies only while propagating through
    // downstream value flow, where a genuinely introduced unit axis must stay
    // scalar across an expanding broadcast.
    if (value == root || !preservesIntroducedUnitAxis(value, selects)) {
      SmallVector<Attribute> replaceableExtents(previousExtents.begin(),
                                                previousExtents.end());
      if (followLogicalDimension) {
        replaceableExtents.clear();
        collectSelectedExtents(value.getType(), selects, replaceableExtents);
      }
      auto replaceableAxis = [&](AxisMapAttr mapping) {
        return selects(mapping) &&
               !isIntroducedReshapeUnitAxis(value,
                                            mapping.getFragmentAxis());
      };
      Type replacement = replaceExtent(value.getType(), replaceableAxis,
                                       replaceableExtents, extent);
      if (replacement != value.getType()) {
        value.setType(replacement);
        // A make_range owns both the fragment schema and the SSA extent used
        // to materialize that schema.  Retarget them from the same physical
        // decision; leaving the operand behind creates two executable
        // authorities for one traversal.
        if (auto range = value.getDefiningOp<MakeRangeOp>()) {
          OpBuilder builder(range);
          Value physicalExtent;
          if (extent.getKind() ==
              static_cast<uint32_t>(PhysicalExprKind::Constant))
            physicalExtent = builder.create<arith::ConstantIndexOp>(
                range.getLoc(), extent.getValue());
          else
            physicalExtent = builder.create<PhysicalExprOp>(
                range.getLoc(), builder.getIndexType(), extent);
          range->setOperand(1, physicalExtent);
        }
      }
    }
    // Product fields and structured helper arguments are part of the same
    // physical value flow even though MLIR does not connect them with ordinary
    // result uses.  A blocking decision for one provenance axis must cross
    // those boundaries; otherwise an operation can retain two physical
    // schemas for one summary/carry value.
    if (auto record = value.getDefiningOp<MakeRecordOp>())
      worklist.append(record.getFields().begin(), record.getFields().end());
    if (auto extract = value.getDefiningOp<ExtractOp>())
      worklist.push_back(extract.getRecord());
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      appendStructuredParentRelations(argument, worklist, selects);
      auto loop = dyn_cast_or_null<scf::ForOp>(
          argument.getOwner()->getParentOp());
      if (loop)
        for (auto [index, iterArgument] :
             llvm::enumerate(loop.getRegionIterArgs()))
          if (argument == iterArgument) {
            worklist.push_back(loop.getInitArgs()[index]);
            worklist.push_back(loop.getResult(index));
          }
    }
    if (auto loop = value.getDefiningOp<scf::ForOp>())
      for (auto [index, result] : llvm::enumerate(loop.getResults()))
        if (value == result) {
          worklist.push_back(loop.getInitArgs()[index]);
          worklist.push_back(loop.getRegionIterArgs()[index]);
          worklist.push_back(loop.getBody()->getTerminator()->getOperand(index));
        }
    if (auto branch = value.getDefiningOp<scf::IfOp>())
      for (auto [index, result] : llvm::enumerate(branch.getResults())) {
        if (value != result)
          continue;
        auto thenYield = cast<scf::YieldOp>(branch.thenBlock()->getTerminator());
        auto elseYield = cast<scf::YieldOp>(branch.elseBlock()->getTerminator());
        worklist.push_back(thenYield.getResults()[index]);
        worklist.push_back(elseYield.getResults()[index]);
      }
    appendStructuredResultRelations(value, worklist, selects);
    for (Operation *user : value.getUsers()) {
      if (auto buffer = dyn_cast<BufferOp>(user);
          buffer && buffer.getInitialValue() == value && previousFragment) {
        auto storage = buffer.getResult().getType();
        auto initial = dyn_cast<FragmentType>(value.getType());
        if (initial && storage.getShape() == previousFragment.getShape() &&
            storage.getOwner() == previousFragment.getOwner() &&
            initial.getOwner() == storage.getOwner()) {
          bool padding = llvm::all_of(
              llvm::zip(previousFragment.getShape(), initial.getShape()),
              [](auto dimensions) {
                auto [before, after] = dimensions;
                if (before == after)
                  return true;
                auto oldExtent = cast<PhysicalExprAttr>(before);
                auto newExtent = cast<PhysicalExprAttr>(after);
                return oldExtent.getKind() ==
                           static_cast<uint32_t>(PhysicalExprKind::Constant) &&
                       newExtent.getKind() ==
                           static_cast<uint32_t>(PhysicalExprKind::Constant) &&
                       oldExtent.getValue() > 0 &&
                       static_cast<uint64_t>(newExtent.getValue()) ==
                           llvm::PowerOf2Ceil(static_cast<uint64_t>(oldExtent.getValue()));
              });
          if (padding)
            buffer.getResult().setType(BufferType::get(
                storage.getContext(), storage.getElementType(), initial.getShape(),
                storage.getScope(), storage.getInstance(), storage.getOwner(),
                storage.getInitialization(), storage.getLifetime(),
                storage.getVisibility(), storage.getWorkspace()));
        }
      }
      if (auto broadcast = dyn_cast<BroadcastOp>(user)) {
        auto target = dyn_cast<FragmentType>(broadcast.getResult().getType());
        if (previousFragment && target) {
          BroadcastProjection projection =
              queryBroadcastProjection(previousFragment, target);
          if (projection.isExact())
            for (auto [targetAxis, sourceAxis] :
                 llvm::enumerate(projection.targetToSource)) {
              if (!sourceAxis)
                continue;
              auto inputMap =
                  cast<AxisMapAttr>(previousFragment.getAxisMaps()[*sourceAxis]);
              auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
              auto oldExtent =
                  cast<PhysicalExprAttr>(previousFragment.getShape()[*sourceAxis]);
              bool unit = oldExtent.getKind() ==
                              static_cast<uint32_t>(PhysicalExprKind::Constant) &&
                          oldExtent.getValue() <= 1;
              bool sameLogicalExtent = inputMap.getDimensionId() > 0 &&
                  inputMap.getDimensionId() == targetMap.getDimensionId();
              if (selects(inputMap) && !selects(targetMap) &&
                  (!unit || sameLogicalExtent) &&
                  !isIntroducedReshapeUnitAxis(value, *sourceAxis) &&
                  oldExtent == target.getShape()[targetAxis] &&
                  target.getShape()[targetAxis] != extent) {
                std::pair<Value, AxisMapAttr> alias{broadcast.getResult(),
                                                    targetMap};
                if (!llvm::is_contained(valueAliases, alias))
                  valueAliases.push_back(alias);
              }
            }
        }
      }
      if (auto reshape = dyn_cast<ReshapeOp>(user)) {
        auto target = dyn_cast<FragmentType>(reshape.getResult().getType());
        if (previousFragment && target) {
          unsigned sourceRank = 0, resultRank = 0;
          for (Attribute attribute : reshape.getReassociation()) {
            auto group = cast<ReshapeGroupAttr>(attribute);
            sourceRank += group.getSourceAxes().size();
            resultRank += group.getResultAxes().size();
          }
          unsigned sourcePrefix = previousFragment.getShape().size() - sourceRank;
          unsigned resultPrefix = target.getShape().size() - resultRank;
          for (Attribute attribute : reshape.getReassociation()) {
            auto group = cast<ReshapeGroupAttr>(attribute);
            if (group.getSourceAxes().size() != 1 ||
                group.getResultAxes().size() != 1)
              continue;
            unsigned sourceAxis = sourcePrefix + group.getSourceAxes()[0];
            unsigned targetAxis = resultPrefix + group.getResultAxes()[0];
            auto inputMap =
                cast<AxisMapAttr>(previousFragment.getAxisMaps()[sourceAxis]);
            auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
            if (!selects(inputMap) || selects(targetMap) ||
                previousFragment.getShape()[sourceAxis] != target.getShape()[targetAxis] ||
                target.getShape()[targetAxis] == extent)
              continue;
            std::pair<Value, AxisMapAttr> alias{reshape.getResult(), targetMap};
            if (!llvm::is_contained(valueAliases, alias))
              valueAliases.push_back(alias);
          }
        }
      }
      if (isa<RegionFoldOp, RegionScanOp>(user)) {
        appendStructuredChildRelations(user, value, worklist, selects);
        for (Value result : user->getResults())
          worklist.push_back(result);
        continue;
      }
      if (auto loop = dyn_cast<scf::ForOp>(user))
        for (auto [index, init] : llvm::enumerate(loop.getInitArgs()))
          if (value == init) {
            worklist.push_back(loop.getRegionIterArgs()[index]);
            worklist.push_back(loop.getResult(index));
          }
      if (auto yield = dyn_cast<scf::YieldOp>(user)) {
        if (auto loop = dyn_cast_or_null<scf::ForOp>(yield->getParentOp()))
          for (auto [index, yielded] : llvm::enumerate(yield.getOperands()))
            if (value == yielded) {
              worklist.push_back(loop.getInitArgs()[index]);
              worklist.push_back(loop.getRegionIterArgs()[index]);
              worklist.push_back(loop.getResult(index));
            }
        if (auto branch = dyn_cast_or_null<scf::IfOp>(yield->getParentOp()))
          for (auto [index, yielded] : llvm::enumerate(yield.getOperands()))
            if (value == yielded)
              worklist.push_back(branch.getResult(index));
      }
      if (isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
              ContractOp, ScaledContractOp, SparseContractOp, ReduceOp,
              ScanOp, RandomBitsOp,
              ScatterReduceOp, AtomicStoreOp, AtomicRMWOp,
              AtomicCompareExchangeOp>(user)) {
        for (Region &region : user->getRegions())
          for (Block &block : region)
            for (BlockArgument argument : block.getArguments())
              if (!isSegmentSourceSlice(argument))
                worklist.push_back(argument);
      }
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }
  for (auto [value, axis] : valueAliases) {
    auto type = cast<FragmentType>(value.getType());
    if (type.getShape()[axis.getFragmentAxis()] == extent)
      continue;
    retargetExtent(
        value,
        [=](AxisMapAttr mapping) {
          return sourceAxisIdentity(mapping) == sourceAxisIdentity(axis) &&
                 mapping.getDimensionId() == axis.getDimensionId();
        },
        extent, /*followLogicalDimension=*/false);
  }
}

void retargetSourceExtent(Value root, PhysicalSourceAxis source,
                          PhysicalExprAttr extent) {
  retargetExtent(
      root,
      [=](AxisMapAttr mapping) {
        return mapping.getSourceId() == source.sourceId &&
               mapping.getSourceAxis() == source.sourceAxis &&
               mapping.getDerived() == source.derived;
      },
      extent, /*followLogicalDimension=*/false);
}

void retargetDimensionExtent(Value root, int64_t dimensionId,
                             PhysicalExprAttr extent) {
  if (dimensionId <= 0)
    return;
  retargetExtent(root,
                 [=](AxisMapAttr mapping) {
                   return mapping.getDimensionId() == dimensionId;
                 },
                 extent, /*followLogicalDimension=*/true);
}

LogicalResult alignStructuredCaptureRelations(func::FuncOp kernel) {
  auto align = [&](Operation *owner, Value capture,
                   BlockArgument argument) -> LogicalResult {
    auto authority = dyn_cast<FragmentType>(capture.getType());
    auto target = dyn_cast<FragmentType>(argument.getType());
    if (!authority || !target)
      return capture.getType() == argument.getType()
                 ? success()
                 : owner->emitOpError(
                       "physical structured capture lost its parent schema");
    if (authority.getElementType() != target.getElementType() ||
        authority.getShape().size() != target.getShape().size() ||
        authority.getAxisMaps() != target.getAxisMaps() ||
        authority.getOwner() != target.getOwner())
      return owner->emitOpError(
          "physical structured capture changed its coordinate relation");
    for (auto [axis, attribute] : llvm::enumerate(authority.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      if (mapping.getDimensionId() <= 0)
        continue;
      retargetDimensionExtent(argument, mapping.getDimensionId(),
                              cast<PhysicalExprAttr>(authority.getShape()[axis]));
    }
    return capture.getType() == argument.getType()
               ? success()
               : owner->emitOpError(
                     "physical structured capture extent is inconsistent");
  };

  WalkResult result = kernel.walk([&](Operation *operation) {
    if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
      unsigned sourceCount = fold.getSourceCount();
      unsigned captureOffset = sourceCount + fold.getIdentityCount();
      for (unsigned index = 0; index < fold.getCaptureCount(); ++index)
        if (failed(align(operation, fold.getInputs()[captureOffset + index],
                         fold.getSummarize().front().getArgument(sourceCount +
                                                                 index))))
          return WalkResult::interrupt();
      return WalkResult::advance();
    }
    auto scan = dyn_cast<RegionScanOp>(operation);
    if (!scan)
      return WalkResult::advance();
    unsigned sourceCount = scan.getSourceCount();
    unsigned stateCount = scan.getStateCount();
    unsigned captureOffset =
        sourceCount + scan.getIdentityCount() + stateCount;
    for (unsigned index = 0; index < scan.getCaptureCount(); ++index) {
      Value capture = scan.getInputs()[captureOffset + index];
      if (failed(align(operation, capture,
                       scan.getSummarize().front().getArgument(sourceCount +
                                                               index))) ||
          failed(align(operation, capture,
                       scan.getEmit().front().getArgument(sourceCount +
                                                          stateCount + index))))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

FailureOr<uint64_t> blockedDimension(Attribute attribute) {
  auto extent = dyn_cast<PhysicalExprAttr>(attribute);
  if (!extent ||
      extent.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::CeilDiv) ||
      extent.getOperands().size() != 2)
    return failure();
  auto logical = dyn_cast<PhysicalExprAttr>(extent.getOperands()[0]);
  auto block = dyn_cast<PhysicalExprAttr>(extent.getOperands()[1]);
  if (!logical || !block ||
      logical.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Dimension) ||
      block.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Parameter))
    return failure();
  return logical.getValue() > 0
             ? FailureOr<uint64_t>(logical.getValue())
             : FailureOr<uint64_t>(failure());
}

bool hasBlockedDimension(func::FuncOp kernel, uint64_t dimension) {
  bool found = false;
  kernel.walk([&](DelinearizeOp mapping) {
    for (Attribute extent : mapping.getLaunchExtents()) {
      FailureOr<uint64_t> blocked = blockedDimension(extent);
      found |= succeeded(blocked) && *blocked == dimension;
    }
  });
  return found;
}

LogicalResult realizeFullCoverageDimension(func::FuncOp kernel, Value source,
                                           unsigned fragmentAxis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || fragmentAxis >= fragment.getShape().size())
    return failure();
  auto mapping =
      dyn_cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
  if (!mapping || mapping.getDimensionId() <= 0)
    return failure();
  const int64_t dimension = mapping.getDimensionId();
  auto currentExtent =
      cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact ranges = analysis.axisRanges(source, fragmentAxis);
  if (ranges.roots.empty())
    for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps()))
      if (cast<AxisMapAttr>(attribute).getDimensionId() == dimension &&
          fragment.getShape()[axis] != currentExtent)
        return emitError(source.getLoc(),
                         "full coverage has no range authority for distinct physical occurrences");
  int64_t coverageDimension = dimension;
  bool subregion = llvm::any_of(ranges.roots, [](MakeRangeOp range) {
    return range->hasAttr(sourceSubregionAttr);
  });
  if (subregion) {
    if (failed(queryExactLogicalRange(ranges)))
      return kernel.emitError(
          "subregion full coverage has no exact logical range authority");
    std::optional<int64_t> parent;
    for (MakeRangeOp range : ranges.roots) {
      auto bound = range->getAttrOfType<IntegerAttr>(sourceSubregionAttr);
      if (!bound || bound.getInt() <= 0 ||
          (parent && *parent != bound.getInt()) ||
          !queryNonNegativeIndexUpperBound(range.getLogicalStart()) ||
          !isUnitStepRange(range))
        return range.emitOpError(
            "subregion full coverage has no proven nonnegative parent-bounded traversal");
      parent = bound.getInt();
    }
    coverageDimension = *parent;
  }

  Value runtimeDimension;
  for (BlockArgument argument : kernel.getArguments()) {
    DictionaryAttr attributes = kernel.getArgAttrDict(argument.getArgNumber());
    auto kind = attributes.getAs<StringAttr>(abiKindAttr);
    auto identity = attributes.getAs<IntegerAttr>(dimensionAttr);
    if (kind && kind.getValue() == "dimension" && identity &&
        identity.getInt() == coverageDimension) {
      if (runtimeDimension && runtimeDimension != argument)
        return kernel.emitError(
            "logical dimension has multiple runtime ABI authorities");
      runtimeDimension = argument;
    }
  }

  std::optional<int64_t> staticDimension;
  for (BlockArgument argument : kernel.getArguments()) {
    auto view = dyn_cast<ViewType>(argument.getType());
    if (!view)
      continue;
    for (auto [axis, identity] :
         llvm::enumerate(view.getLayout().getDimensionIds().asArrayRef())) {
      if (identity != coverageDimension)
        continue;
      auto extent = cast<PhysicalExprAttr>(
          view.getLayout().getExtents()[axis]);
      if (extent.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Constant))
        continue;
      if (staticDimension && *staticDimension != extent.getValue())
        return kernel.emitError(
            "logical dimension has conflicting static ABI extents");
      staticDimension = extent.getValue();
    }
  }
  if (!runtimeDimension && staticDimension &&
      currentExtent.getKind() ==
          static_cast<uint32_t>(PhysicalExprKind::Constant) &&
      currentExtent.getValue() == *staticDimension &&
      llvm::isPowerOf2_64(*staticDimension))
    return success();
  if (!runtimeDimension) {
    FailureOr<MakeRangeOp> authority = queryExactLogicalRange(ranges);
    PhysicalExprAttr capacity = succeeded(authority)
                                   ? queryLogicalRangeCapacity(*authority)
                                   : PhysicalExprAttr();
    if (capacity) {
      OpBuilder builder(&kernel.front(), kernel.front().begin());
      runtimeDimension = builder.create<PhysicalExprOp>(
          source.getLoc(), builder.getIndexType(), capacity);
    }
  }
  if (!runtimeDimension)
    return emitError(source.getLoc(),
                     "full-coverage physicalization has no runtime or static dimension authority")
           << "; dimension=" << dimension << "; source=" << source;
  PhysicalExprAttr coverageBound = queryLaunchExpression(runtimeDimension);
  if (!coverageBound)
    return kernel.emitError(
        "full-coverage physicalization has no launch-visible extent expression");
  if (subregion)
    for (MakeRangeOp range : ranges.roots) {
      auto end = queryNonNegativeIndexUpperBound(range.getLogicalStop());
      auto endConstant = end ? constantPhysicalExpression(end) : std::nullopt;
      auto coverageConstant = constantPhysicalExpression(coverageBound);
      bool bounded = end &&
          (end == coverageBound ||
           (endConstant && coverageConstant && *endConstant <= *coverageConstant));
      if (!samePhysicalScalarExpression(range.getLogicalStop(), runtimeDimension) &&
          !bounded)
        return range.emitOpError(
            "subregion full coverage has no proven stop bound");
    }

  if (!staticDimension)
    if (auto value = dyn_cast_or_null<IntegerAttr>(
            UniformValueAnalysis(describeUniformValue).evaluate(runtimeDimension)))
      staticDimension = value.getInt();
  if (!staticDimension && coverageBound.getKind() ==
                              static_cast<uint32_t>(PhysicalExprKind::Constant))
    staticDimension = coverageBound.getValue();
  if (staticDimension && *staticDimension >= 0) {
    uint64_t size = llvm::PowerOf2Ceil(
        static_cast<uint64_t>(std::max<int64_t>(*staticDimension, 1)));
    if (size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return kernel.emitError("full-coverage extent exceeds the index range");
    auto covered = PhysicalExprAttr::get(kernel.getContext(),
        static_cast<uint32_t>(PhysicalExprKind::Constant), size,
        StringAttr::get(kernel.getContext(), ""), ArrayAttr::get(kernel.getContext(), {}));
    if (currentExtent == covered &&
        analysis.axisRealization(source, fragmentAxis).physicalized)
      return success();
    OpBuilder builder(&kernel.front(), kernel.front().begin());
    Value physicalExtent = builder.create<arith::ConstantIndexOp>(source.getLoc(), size);
    llvm::SmallDenseSet<int64_t> rangeDimensions;
    for (MakeRangeOp range : ranges.roots) {
      FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
      if (failed(rangeDimension))
        return range.emitOpError("full coverage has no range dimension authority");
      rangeDimensions.insert(*rangeDimension);
      retargetSourceExtent(range.getResult(), sourceAxisIdentity(range), covered);
    }
    if (ranges.roots.empty())
      retargetDimensionExtent(source, dimension, covered);
    if (rangeDimensions.empty())
      rangeDimensions.insert(dimension);
    for (int64_t rangeDimension : rangeDimensions)
      if (failed(bindFullCoverageDimension(kernel, rangeDimension, physicalExtent)))
        return failure();
    return success();
  }

  ParameterOp parameter;
  if (currentExtent.getKind() ==
      static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
    FailureOr<ParameterOp> declaration =
        queryParameterBySymbol(kernel, currentExtent.getSymbol());
    if (succeeded(declaration)) {
      auto covered = (*declaration)->getAttrOfType<IntegerAttr>(
          coverageDimensionAttr);
      if (covered && covered.getInt() == coverageDimension &&
          (*declaration).getParameter().getRole() ==
              static_cast<uint32_t>(ParameterRole::FullCoverage))
        parameter = *declaration;
    }
  }
  if (!parameter) {
    kernel.walk([&](ParameterOp candidate) {
      auto covered = candidate->getAttrOfType<IntegerAttr>(
          coverageDimensionAttr);
      if (!covered || covered.getInt() != coverageDimension ||
          candidate.getParameter().getRole() !=
              static_cast<uint32_t>(ParameterRole::FullCoverage))
        return;
      if (parameter && parameter != candidate) {
        parameter = ParameterOp();
        return;
      }
      parameter = candidate;
    });
  }
  if (parameter &&
      currentExtent.getKind() ==
          static_cast<uint32_t>(PhysicalExprKind::Parameter) &&
      currentExtent.getSymbol() == parameter.getParameter().getName() &&
      parameter.getParameter().getRole() ==
          static_cast<uint32_t>(ParameterRole::FullCoverage)) {
    parameter->setAttr(coverageBoundAttr, coverageBound);
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    if (!binding.isExact() || !binding.dimension ||
        *binding.dimension != coverageDimension)
      return parameter.emitOpError(
          "full-coverage parameter lost its typed dimension authority");
    if (failed(bindFullCoverageDimension(kernel, dimension,
                                         parameter.getResult())))
      return kernel.emitError(
          "existing full-coverage decision could not preserve access validity");
    return success();
  }
  static constexpr int64_t candidates[] = {
      1,    2,    4,     8,     16,    32,    64,    128,   256,
      512,  1024, 2048,  4096,  8192,  16384, 32768, 65536};
  if (!parameter) {
    std::string name = ("FULL_D" + Twine(coverageDimension)).str();
    bool nameCollision = false;
    kernel.walk([&](ParameterOp candidate) {
      nameCollision |=
          candidate.getParameter().getName().getValue() == name;
    });
    if (nameCollision) {
      InFlightDiagnostic diagnostic = kernel.emitError(
          "full-coverage parameter name is already owned by another decision");
      kernel.walk([&](ParameterOp candidate) {
        if (candidate.getParameter().getName().getValue() != name)
          return;
        diagnostic << "; role=" << candidate.getParameter().getRole();
        if (auto covered = candidate->getAttrOfType<IntegerAttr>(
                coverageDimensionAttr))
          diagnostic << ", coverage_dimension=" << covered.getInt();
        if (auto bound = candidate->getAttrOfType<IntegerAttr>(dimensionAttr))
          diagnostic << ", dimension=" << bound.getInt();
      });
      return failure();
    }
    OpBuilder builder(&kernel.getBody().front(),
                      kernel.getBody().front().begin());
    auto schema = ParameterAttr::get(
        kernel.getContext(), builder.getStringAttr(name),
        static_cast<uint32_t>(ParameterRole::FullCoverage),
        static_cast<uint32_t>(ParameterCategory::Coverage),
        /*elementBitWidth=*/0,
        DenseI64ArrayAttr::get(kernel.getContext(), candidates));
    parameter = builder.create<ParameterOp>(source.getLoc(),
                                            builder.getIndexType(), schema);
  } else {
    ParameterAttr schema = parameter.getParameter();
    parameter->setAttr(
        "parameter",
        ParameterAttr::get(
            kernel.getContext(), schema.getName(), schema.getRole(),
            schema.getCategory(), schema.getElementBitWidth(),
            DenseI64ArrayAttr::get(kernel.getContext(), candidates)));
  }
  parameter->setAttr(
      dimensionAttr,
      IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), coverageDimension));
  parameter->setAttr(
      coverageDimensionAttr,
      IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), coverageDimension));
  parameter->setAttr(coverageBoundAttr, coverageBound);

  auto covered = PhysicalExprAttr::get(
      kernel.getContext(),
      static_cast<uint32_t>(PhysicalExprKind::Parameter), 0,
      parameter.getParameter().getName(),
      ArrayAttr::get(kernel.getContext(), {}));
  if (!ranges.roots.empty() && failed(queryExactLogicalRange(ranges)))
    return kernel.emitError(
        "full-coverage source has ambiguous physical range authority");
  if (ranges.state == PhysicalFactState::Unknown || ranges.roots.empty()) {
    PhysicalReplayFact replay = analysis.replayability(
        source, sourceAxisIdentity(mapping), PhysicalReplayScope::ValueGraph,
        /*allowAccesses=*/false);
    if (!ranges.roots.empty() || !replay.isReplayable()) {
      InFlightDiagnostic diagnostic = kernel.emitError(
          "full-coverage source has no exact physical range authority");
      for (Operation *blocker : ranges.blockers)
        diagnostic << "; blocker=" << blocker->getName();
      for (Operation *blocker : replay.blockers)
        diagnostic << "; replay_blocker=" << blocker->getName();
      return failure();
    }
  }
  for (MakeRangeOp range : ranges.roots) {
    FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
    if (failed(rangeDimension) || *rangeDimension != dimension)
      return range.emitOpError(
          "full-coverage range does not cover the selected logical dimension");
    retargetSourceExtent(range.getResult(), sourceAxisIdentity(range), covered);
  }
  if (ranges.roots.empty())
    retargetDimensionExtent(source, dimension, covered);
  if (failed(bindFullCoverageDimension(kernel, dimension,
                                       parameter.getResult())))
    return kernel.emitError(
        "full-coverage decision could not preserve access validity");
  return success();
}

LogicalResult bindFullCoverageDimension(func::FuncOp kernel, uint64_t dimension,
                                        Value physicalExtent) {
  auto parameter = physicalExtent.getDefiningOp<ParameterOp>();
  PhysicalExprAttr parameterExtent = queryLaunchExpression(physicalExtent);
  if (!parameterExtent ||
      (!parameter && parameterExtent.getKind() !=
                         static_cast<uint32_t>(PhysicalExprKind::Constant)))
    return failure();
  SmallVector<MakeRangeOp> ranges;
  kernel.walk([&](MakeRangeOp range) {
    FailureOr<int64_t> sourceDimension = querySourceDimension(
        range.getResult().getType(), sourceAxisIdentity(range));
    auto fragment = dyn_cast<FragmentType>(range.getResult().getType());
    if (failed(sourceDimension) ||
        *sourceDimension != static_cast<int64_t>(dimension) ||
        !fragment ||
        fragment.getShape().size() != 1)
      return;
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
    if (extent.getKind() ==
        static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
      FailureOr<ParameterOp> declaration =
          queryParameterBySymbol(kernel, extent.getSymbol());
      if (succeeded(declaration) && *declaration != parameter &&
          (*declaration).getParameter().getRole() !=
              static_cast<uint32_t>(ParameterRole::FullCoverage))
        return;
    }
    ranges.push_back(range);
  });
  // Validity can carry an independent occurrence of the same coordinates.
  SmallVector<MakeRangeOp> authorities(ranges.begin(), ranges.end());
  kernel.walk([&](MakeRangeOp range) {
    if (llvm::is_contained(ranges, range) ||
        range->getParentOfType<RegionFoldOp>() ||
        range->getParentOfType<RegionScanOp>())
      return;
    auto sourceDimension = queryRangeDimension(range);
    if (failed(sourceDimension) ||
        *sourceDimension != static_cast<int64_t>(dimension))
      return;
    if (llvm::any_of(authorities, [&](MakeRangeOp authority) {
          return sameLogicalRange(range, authority) &&
                 samePhysicalScalarExpression(range.getStart(),
                                              authority.getStart());
        }))
      ranges.push_back(range);
  });
  bool alreadyBound = !ranges.empty() &&
                      llvm::all_of(ranges, [&](MakeRangeOp range) {
                        return range.getExtent() == physicalExtent;
                      });
  for (MakeRangeOp range : ranges)
    retargetSourceExtent(range.getResult(), sourceAxisIdentity(range),
                         parameterExtent);
  if (ranges.empty() || alreadyBound)
    return success();

  llvm::SmallPtrSet<Operation *, 32> rangeSet;
  for (MakeRangeOp range : ranges)
    rangeSet.insert(range.getOperation());

  SmallVector<LoadOp> loads;
  SmallVector<GatherOp> gathers;
  SmallVector<StoreOp> stores;
  SmallVector<ScatterReduceOp> scatters;
  SmallVector<AtomicLoadOp> atomicLoads;
  SmallVector<AtomicStoreOp> atomicStores;
  SmallVector<AtomicRMWOp> atomicRMWs;
  SmallVector<AtomicCompareExchangeOp> atomicCAS;
  kernel.walk([&](LoadOp load) { loads.push_back(load); });
  kernel.walk([&](GatherOp gather) { gathers.push_back(gather); });
  kernel.walk([&](StoreOp store) { stores.push_back(store); });
  kernel.walk([&](ScatterReduceOp scatter) { scatters.push_back(scatter); });
  kernel.walk([&](AtomicLoadOp atomic) { atomicLoads.push_back(atomic); });
  kernel.walk([&](AtomicStoreOp atomic) { atomicStores.push_back(atomic); });
  kernel.walk([&](AtomicRMWOp atomic) { atomicRMWs.push_back(atomic); });
  kernel.walk(
      [&](AtomicCompareExchangeOp atomic) { atomicCAS.push_back(atomic); });
  llvm::DenseMap<Operation *, SmallVector<MakeRangeOp>> accessRanges;
  PhysicalProgramAnalysis accessAnalysis(kernel);
  auto recordAccessRanges = [&](Operation *access) -> LogicalResult {
    PhysicalAccessFootprint footprint = accessAnalysis.footprint(access);
    if (footprint.state != PhysicalFactState::Exact ||
        footprint.rangeState != PhysicalFactState::Exact)
      return access->emitOpError(
          "full-coverage rewrite requires one exact physical access footprint");
    auto &relevant = accessRanges[access];
    for (MakeRangeOp range : footprint.ranges)
      if (rangeSet.contains(range.getOperation()) &&
          !llvm::is_contained(relevant, range))
        relevant.push_back(range);
    return success();
  };
  for (LoadOp load : loads)
    if (failed(recordAccessRanges(load)))
      return failure();
  for (GatherOp gather : gathers)
    if (failed(recordAccessRanges(gather)))
      return failure();
  for (StoreOp store : stores)
    if (failed(recordAccessRanges(store)))
      return failure();
  for (ScatterReduceOp scatter : scatters)
    if (failed(recordAccessRanges(scatter)))
      return failure();
  for (AtomicLoadOp atomic : atomicLoads)
    if (failed(recordAccessRanges(atomic)))
      return failure();
  for (AtomicStoreOp atomic : atomicStores)
    if (failed(recordAccessRanges(atomic)))
      return failure();
  for (AtomicRMWOp atomic : atomicRMWs)
    if (failed(recordAccessRanges(atomic)))
      return failure();
  for (AtomicCompareExchangeOp atomic : atomicCAS)
    if (failed(recordAccessRanges(atomic)))
      return failure();

  llvm::DenseMap<Operation *, Value> predicates;
  for (MakeRangeOp range : ranges) {
    OpBuilder builder(range);
    range->setOperand(1, physicalExtent);
    builder.setInsertionPointAfter(range);
    Value distance = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), range.getLogicalStop(),
        range.getLogicalStart(), BinaryOperator::Subtract);
    Value one = builder.create<arith::ConstantIndexOp>(range.getLoc(), 1);
    Value adjusted = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), distance,
        builder.create<BinaryOp>(range.getLoc(), builder.getIndexType(),
                                 range.getStep(), one,
                                 BinaryOperator::Subtract),
        BinaryOperator::Add);
    Value logicalExtent = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), adjusted, range.getStep(),
        BinaryOperator::FloorDivide);
    Value logicalDistance = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), logicalExtent, range.getStep(),
        BinaryOperator::Multiply);
    auto coordinate = cast<FragmentType>(range.getResult().getType());
    Value member = range.getResult();
    Value stop;
    if (isUnitStepRange(range) &&
        (!queryNonNegativeIndexUpperBound(range.getLogicalStart()) ||
         !queryNonNegativeIndexUpperBound(range.getLogicalStop()))) {
      Value nonempty = builder.create<CompareOp>(
          range.getLoc(), builder.getI1Type(), range.getLogicalStart(),
          range.getLogicalStop(), ComparePredicate::Lt);
      Value zero = builder.create<arith::ConstantIndexOp>(range.getLoc(), 0);
      stop = builder.create<SelectOp>(range.getLoc(), builder.getIndexType(),
                                       nonempty, logicalDistance, zero);
      Value base = builder.create<BroadcastOp>(range.getLoc(), coordinate,
                                               range.getStart());
      // Unit-step offsets remain [0, extent) even when an inactive padded
      // absolute coordinate wraps. Preserve the range's physical origin.
      member = builder.create<BinaryOp>(range.getLoc(), coordinate, member,
                                         base, BinaryOperator::Subtract);
    } else {
      stop = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), range.getStart(), logicalDistance,
          BinaryOperator::Add);
    }
    Value stopFragment =
        builder.create<BroadcastOp>(range.getLoc(), coordinate, stop);
    auto predicate = FragmentType::get(
        kernel.getContext(), builder.getI1Type(), coordinate.getShape(),
        coordinate.getAxisMaps(), coordinate.getValidity(),
        coordinate.getOwner());
    auto tail = builder.create<CompareOp>(range.getLoc(), predicate,
                                          member, stopFragment,
                                          ComparePredicate::Lt);
    tail->setAttr(physicalTailAttr, builder.getUnitAttr());
    predicates[range.getOperation()] = tail.getResult();
  }

  auto materializeTail = [&](OpBuilder &builder, Location location,
                             FragmentType target,
                             ArrayRef<MakeRangeOp> sources,
                             Value data = Value()) -> FailureOr<Value> {
    Value result;
    for (MakeRangeOp range : sources) {
      SmallVector<unsigned, 2> axes;
      for (PhysicalAxisProjection projection : queryRangeProjections(target, range))
        axes.push_back(projection.fragmentAxis);
      if (axes.empty() && data) {
        PhysicalRangeAxisFact relation =
            PhysicalProgramAnalysis(kernel).rangeAxes(data, {range});
        if (relation.isExact())
          axes.append(relation.fragmentAxes.begin(), relation.fragmentAxes.end());
      }
      if (axes.empty())
        return failure();
      for (unsigned axis : axes) {
        FailureOr<Value> current = projectPredicate(
            builder, location, predicates.lookup(range.getOperation()), target,
            axis);
        if (failed(current))
          return failure();
        result = result ? Value(builder.create<BinaryOp>(
                              location, current->getType(), result, *current,
                              BinaryOperator::LogicalAnd))
                        : *current;
      }
    }
    return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
  };

  auto combineValidity = [&](OpBuilder &builder, Location location,
                             FragmentType type, ArrayRef<MakeRangeOp> sources,
                             Value existing) -> FailureOr<Value> {
    FailureOr<Value> tail = materializeTail(builder, location, type, sources);
    if (failed(tail))
      return failure();
    Value valid = *tail;
    auto predicate = cast<FragmentType>(valid.getType());
    if (!existing)
      return valid;
    Type element = existing.getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    if (!element.isInteger(1))
      return failure();
    if (existing.getType() != predicate) {
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, location, existing, predicate);
      if (failed(projected))
        return failure();
      existing = *projected;
    }
    return Value(builder.create<BinaryOp>(location, predicate, existing, valid,
                                          BinaryOperator::LogicalAnd));
  };

  for (LoadOp load : loads) {
    auto sources = accessRanges.lookup(load.getOperation());
    auto type = dyn_cast<FragmentType>(load.getResult().getType());
    if (!load->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(load);
    FailureOr<Value> tail =
        materializeTail(builder, load.getLoc(), type, sources, load.getResult());
    if (failed(tail))
      return load.emitOpError(
          "cannot project full-coverage dimension to load validity")
             << "; dimension=" << dimension << "; result=" << type
             << "; range_count=" << sources.size();
    Value valid = *tail;
    auto predicate = cast<FragmentType>(valid.getType());
    if (load.getValid()) {
      Value existing = load.getValid();
      Type element = existing.getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      if (!element.isInteger(1))
        return load.emitOpError(
            "full-coverage load carried non-predicate validity");
      if (existing.getType() != predicate) {
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, load.getLoc(), existing, predicate);
        if (failed(projected))
          return load.emitOpError(
              "full-coverage validity has no exact physical projection")
                 << "; existing=" << existing.getType()
                 << "; required=" << predicate;
        existing = *projected;
      }
      valid = builder.create<BinaryOp>(load.getLoc(), predicate, existing, valid,
                                       BinaryOperator::LogicalAnd);
    }
    Value fill = load.getFill();
    if (!fill) {
      FailureOr<Value> zero = zeroFill(builder, load.getLoc(), type);
      if (failed(zero))
        return load.emitOpError("full-coverage load has no neutral fill");
      fill = *zero;
    } else if (fill.getType() != type) {
      FailureOr<Value> projected =
          projectPhysicalValueToSchema(builder, load.getLoc(), fill, type);
      if (failed(projected))
        return load.emitOpError(
            "full-coverage fill has no exact physical projection");
      fill = *projected;
    }
    // Callers may be realizing coverage for this exact SSA result.  Updating
    // its access operands keeps that value live across physicalization.
    load.getValidMutable().assign(ValueRange{valid});
    load.getFillMutable().assign(ValueRange{fill});
  }

  for (GatherOp gather : gathers) {
    auto sources = accessRanges.lookup(gather.getOperation());
    auto type = dyn_cast<FragmentType>(gather.getResult().getType());
    if (!gather->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(gather);
    FailureOr<Value> valid = combineValidity(
        builder, gather.getLoc(), type, sources, gather.getValid());
    if (failed(valid))
      return gather.emitOpError(
          "cannot project full-coverage dimension to gather validity");
    Value fill = gather.getFill();
    if (!fill) {
      FailureOr<Value> zero = zeroFill(builder, gather.getLoc(), type);
      if (failed(zero))
        return gather.emitOpError("full-coverage gather has no neutral fill");
      fill = *zero;
    } else if (fill.getType() != type) {
      FailureOr<Value> projected =
          projectPhysicalValueToSchema(builder, gather.getLoc(), fill, type);
      if (failed(projected))
        return gather.emitOpError(
            "full-coverage fill has no exact physical projection");
      fill = *projected;
    }
    gather.getValidMutable().assign(ValueRange{*valid});
    gather.getFillMutable().assign(ValueRange{fill});
  }

  for (StoreOp store : stores) {
    auto sources = accessRanges.lookup(store.getOperation());
    auto type = dyn_cast<FragmentType>(store.getValue().getType());
    if (!store->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(store);
    SmallVector<Attribute> shape(type.getShape().begin(), type.getShape().end());
    for (MakeRangeOp range : sources) {
      auto coordinate = cast<FragmentType>(range.getResult().getType());
      for (PhysicalAxisProjection projection : queryRangeProjections(type, range))
        shape[projection.fragmentAxis] = coordinate.getShape()[0];
    }
    type = FragmentType::get(kernel.getContext(), type.getElementType(),
                             builder.getArrayAttr(shape), type.getAxisMaps(),
                             type.getValidity(), type.getOwner());
    // The access coordinates own the physical extent.  Project a uniform
    // payload to that schema without retargeting its independent value graph.
    FailureOr<Value> value = projectPhysicalValueToSchema(
        builder, store.getLoc(), store.getValue(), type);
    if (failed(value))
      return store.emitOpError(
          "full-coverage stored value has no exact physical projection");
    FailureOr<Value> tail =
        materializeTail(builder, store.getLoc(), type, sources);
    if (failed(tail))
      return store.emitOpError(
          "cannot project full-coverage dimension to store validity");
    Value valid = *tail;
    auto predicate = cast<FragmentType>(valid.getType());
    if (store.getValid()) {
      Value existing = store.getValid();
      if (existing.getType() != predicate) {
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, store.getLoc(), existing, predicate);
        if (failed(projected))
          return store.emitOpError(
              "full-coverage validity has no exact physical projection")
                 << "; existing=" << existing.getType()
                 << "; required=" << predicate;
        existing = *projected;
      }
      valid = builder.create<BinaryOp>(store.getLoc(), predicate, existing,
                                       valid, BinaryOperator::LogicalAnd);
    }
    auto replacement = builder.create<StoreOp>(
        store.getLoc(), store.getResource(), store.getCoordinates(),
        *value, valid, store.getSourceAxes());
    if (Attribute origin = store->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    store.erase();
  }

  for (ScatterReduceOp scatter : scatters) {
    auto sources = accessRanges.lookup(scatter.getOperation());
    auto type = dyn_cast<FragmentType>(scatter.getValue().getType());
    if (!scatter->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(scatter);
    FailureOr<Value> valid = combineValidity(
        builder, scatter.getLoc(), type, sources, scatter.getValid());
    if (failed(valid))
      return scatter.emitOpError(
          "cannot project full-coverage dimension to scatter validity");
    OperationState state(scatter.getLoc(), ScatterReduceOp::getOperationName());
    state.addOperands(scatter.getResource());
    state.addOperands(scatter.getCoordinates());
    state.addOperands(scatter.getValue());
    state.addOperands(*valid);
    for (NamedAttribute attribute : scatter->getAttrs())
      if (attribute.getName() != "operandSegmentSizes")
        state.addAttribute(attribute.getName(), attribute.getValue());
    state.addAttribute(
        "operandSegmentSizes",
        builder.getDenseI32ArrayAttr(
            {1, static_cast<int32_t>(scatter.getCoordinates().size()), 1, 1}));
    state.addRegion();
    auto replacement = cast<ScatterReduceOp>(builder.create(state));
    replacement.getCombine().takeBody(scatter.getCombine());
    scatter.erase();
  }

  for (AtomicLoadOp atomic : atomicLoads) {
    auto sources = accessRanges.lookup(atomic.getOperation());
    auto type = dyn_cast<FragmentType>(atomic.getResult().getType());
    if (!atomic->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(atomic);
    FailureOr<Value> valid = combineValidity(
        builder, atomic.getLoc(), type, sources, atomic.getValid());
    if (failed(valid))
      return atomic.emitOpError(
          "cannot project full-coverage dimension to atomic-load validity");
    auto replacement = builder.create<AtomicLoadOp>(
        atomic.getLoc(), type, atomic.getResource(), atomic.getCoordinates(),
        *valid, atomic.getOrdering(), atomic.getSharing(), atomic.getSourceAxes());
    if (Attribute origin = atomic->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    atomic.getResult().replaceAllUsesWith(replacement.getResult());
    atomic.erase();
  }

  for (AtomicStoreOp atomic : atomicStores) {
    auto sources = accessRanges.lookup(atomic.getOperation());
    auto type = dyn_cast<FragmentType>(atomic.getValue().getType());
    if (!atomic->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(atomic);
    FailureOr<Value> valid = combineValidity(
        builder, atomic.getLoc(), type, sources, atomic.getValid());
    if (failed(valid))
      return atomic.emitOpError(
          "cannot project full-coverage dimension to atomic-store validity");
    auto replacement = builder.create<AtomicStoreOp>(
        atomic.getLoc(), atomic.getResource(), atomic.getCoordinates(),
        atomic.getValue(), *valid, atomic.getOrdering(), atomic.getSharing(),
        atomic.getSourceAxes());
    if (Attribute origin = atomic->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    atomic.erase();
  }

  for (AtomicRMWOp atomic : atomicRMWs) {
    auto sources = accessRanges.lookup(atomic.getOperation());
    auto type = dyn_cast<FragmentType>(atomic.getValue().getType());
    if (!atomic->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(atomic);
    FailureOr<Value> valid = combineValidity(
        builder, atomic.getLoc(), type, sources, atomic.getValid());
    if (failed(valid))
      return atomic.emitOpError(
          "cannot project full-coverage dimension to atomic-RMW validity");
    auto replacement = builder.create<AtomicRMWOp>(
        atomic.getLoc(), atomic.getResult().getType(), atomic.getResource(),
        atomic.getCoordinates(), atomic.getValue(), *valid, atomic.getKind(),
        atomic.getOrdering(), atomic.getSharing(), atomic.getSourceAxes());
    if (Attribute origin = atomic->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    atomic.getResult().replaceAllUsesWith(replacement.getResult());
    atomic.erase();
  }

  for (AtomicCompareExchangeOp atomic : atomicCAS) {
    auto sources = accessRanges.lookup(atomic.getOperation());
    auto type = dyn_cast<FragmentType>(atomic.getExpected().getType());
    if (!atomic->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(atomic);
    FailureOr<Value> valid = combineValidity(
        builder, atomic.getLoc(), type, sources, atomic.getValid());
    if (failed(valid))
      return atomic.emitOpError(
          "cannot project full-coverage dimension to compare-exchange validity");
    auto replacement = builder.create<AtomicCompareExchangeOp>(
        atomic.getLoc(), atomic.getResult().getType(), atomic.getResource(),
        atomic.getCoordinates(), atomic.getExpected(), atomic.getDesired(),
        *valid, atomic.getOrdering(), atomic.getSharing(),
        atomic.getSourceAxes());
    if (Attribute origin = atomic->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    atomic.getResult().replaceAllUsesWith(replacement.getResult());
    atomic.erase();
  }
  return success();
}

void eraseDeadPhysicalValues(func::FuncOp kernel) {
  bool changed = false;
  do {
    changed = false;
    kernel.walk<WalkOrder::PostOrder>([&](Operation *operation) {
      if (!operation->getBlock() || isa<DelinearizeOp, ParameterOp>(operation) ||
          operation == kernel.getOperation() || !operation->getNumResults() ||
          !llvm::all_of(operation->getResults(),
                        [](Value value) { return value.use_empty(); }))
        return;
      bool unusedReadOnlyLoop = isa<scf::ForOp>(operation) &&
          !operation->walk([](Operation *nested) {
            if (isa<scf::WhileOp>(nested))
              return WalkResult::interrupt();
            return isa<scf::ForOp, scf::IfOp, LoadOp, GatherOp>(nested) ||
                           isMemoryEffectFree(nested)
                       ? WalkResult::advance() : WalkResult::interrupt();
          }).wasInterrupted();
      if (isMemoryEffectFree(operation) || isa<LoadOp, GatherOp>(operation) ||
          unusedReadOnlyLoop) {
        operation->erase();
        changed = true;
      }
    });
  } while (changed);

}

void eraseUnusedPhysicalParameters(func::FuncOp kernel) {
  llvm::StringSet<> referencedParameters;
  kernel.walk([&](Operation *operation) {
    for (Type type : operation->getResultTypes())
      collectParameterSymbols(type, referencedParameters);
    for (NamedAttribute attribute : operation->getAttrs())
      collectParameterSymbols(attribute.getValue(), referencedParameters);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          collectParameterSymbols(argument.getType(), referencedParameters);
  });
  SmallVector<Operation *> operations;
  kernel.walk([&](Operation *operation) { operations.push_back(operation); });
  for (Operation *operation : llvm::reverse(operations)) {
    // Delinearize is the explicit execution-workset authority.  A singleton
    // workset may not consume its coordinate until a later blocking pass, so
    // ordinary SSA liveness cannot remove it between shared realizations.
    if (isa<DelinearizeOp>(operation))
      continue;
    auto parameter = dyn_cast<ParameterOp>(operation);
    if (!parameter ||
        referencedParameters.contains(
            parameter.getParameter().getName().getValue()))
      continue;
    if (operation == kernel.getOperation() || !operation->getNumResults() ||
        !llvm::all_of(operation->getResults(),
                      [](Value value) { return value.use_empty(); }))
      continue;
    if (isMemoryEffectFree(operation) || isa<LoadOp, GatherOp>(operation))
      operation->erase();
  }
}

LogicalResult replacePhysicalParameter(func::FuncOp kernel,
                                       ParameterOp previous,
                                       ParameterOp replacement) {
  if (!previous || !replacement || previous == replacement)
    return success();
  StringAttr previousName = previous.getParameter().getName();
  StringAttr replacementName = replacement.getParameter().getName();
  previous.getResult().replaceAllUsesWith(replacement.getResult());
  kernel.walk([&](Operation *operation) {
    for (Value result : operation->getResults())
      result.setType(replaceParameterSymbol(result.getType(), previousName,
                                            replacementName));
    SmallVector<NamedAttribute> attributes;
    bool changed = false;
    for (NamedAttribute attribute : operation->getAttrs()) {
      Attribute rewritten = replaceParameterSymbol(
          attribute.getValue(), previousName, replacementName);
      attributes.emplace_back(attribute.getName(), rewritten);
      changed |= rewritten != attribute.getValue();
    }
    if (changed)
      operation->setAttrs(
          DictionaryAttr::get(operation->getContext(), attributes));
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          argument.setType(replaceParameterSymbol(
              argument.getType(), previousName, replacementName));
  });

  llvm::StringSet<> remaining;
  kernel.walk([&](Operation *operation) {
    for (Type type : operation->getResultTypes())
      collectParameterSymbols(type, remaining);
    for (NamedAttribute attribute : operation->getAttrs())
      collectParameterSymbols(attribute.getValue(), remaining);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          collectParameterSymbols(argument.getType(), remaining);
  });
  if (!previous.getResult().use_empty() ||
      remaining.contains(previousName.getValue()))
    return previous.emitOpError(
        "physical parameter refinement left a second executable authority");
  previous.erase();
  return success();
}

} // namespace intent::gpu
