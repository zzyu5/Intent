#include "Intent/Target/Common/Analysis/IndexRelation.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

using namespace mlir;

namespace intent::target {
namespace {

LogicalResult collectDomainSource(Value source,
                                  SmallVectorImpl<Operation *> &domains,
                                  Operation *consumer) {
  Operation *definition = source.getDefiningOp();
  auto fail = [&]() {
    if (consumer)
      consumer->emitOpError("has an unsupported logical iteration source");
    return failure();
  };
  if (!definition)
    return fail();
  StringRef name = definition->getName().getStringRef();
  if (name == "intent.domain" || name == "intent.ragged_outer" ||
      name == "intent.ragged_member") {
    domains.push_back(definition);
    return success();
  }
  if (name == "intent.partition") {
    if (definition->getNumOperands() < 1 || definition->getNumOperands() > 2)
      return fail();
    return collectDomainSource(definition->getOperand(0), domains, consumer);
  }
  if (name != "intent.domain_product" || definition->getNumOperands() == 0)
    return fail();
  for (Value operand : definition->getOperands())
    if (failed(collectDomainSource(operand, domains, consumer)))
      return failure();
  return success();
}

Operation *structuralDomain(Value value) {
  if (Operation *definition = value.getDefiningOp()) {
    StringRef name = definition->getName().getStringRef();
    if (name == "intent.domain" || name == "intent.ragged_outer" ||
        name == "intent.ragged_member")
      return definition;
  }
  auto argument = dyn_cast<BlockArgument>(value);
  Operation *owner = argument ? argument.getOwner()->getParentOp() : nullptr;
  if (!owner)
    return nullptr;
  StringRef name = owner->getName().getStringRef();
  if (name == "intent.state_stream" && argument.getArgNumber() == 0 &&
      owner->getNumOperands() > 0)
    return owner->getOperand(0).getDefiningOp();
  if (name == "intent.for" && argument.getArgNumber() == 0 &&
      owner->getNumOperands() > 0)
    return owner->getOperand(0).getDefiningOp();
  if (name == "intent.ordered" && owner->getNumOperands() > 0) {
    SmallVector<Operation *> domains;
    if (failed(collectDomainSource(owner->getOperand(0), domains, nullptr)) ||
        argument.getArgNumber() >= domains.size())
      return nullptr;
    return domains[argument.getArgNumber()];
  }
  if (name != "intent.parallel" || owner->getNumOperands() != 1)
    return nullptr;
  SmallVector<Operation *> domains;
  if (failed(collectDomainSource(owner->getOperand(0), domains, nullptr)) ||
      argument.getArgNumber() >= domains.size())
    return nullptr;
  return domains[argument.getArgNumber()];
}

FailureOr<ScalarIndexSource>
traceScalarIndexSourceImpl(Value value, Operation &consumer,
                           llvm::DenseSet<Value> &active) {
  if (!active.insert(value).second)
    return consumer.emitOpError("contains a cyclic scalar index expression");
  auto finish = [&](ScalarIndexSource source) -> FailureOr<ScalarIndexSource> {
    active.erase(value);
    return source;
  };
  if (Operation *domain = structuralDomain(value))
    return finish(ScalarIndexSource{domain, {domain}, false, false});
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return finish(ScalarIndexSource{nullptr, {}, true, false});
  StringRef name = definition->getName().getStringRef();
  if (name == "intent.constant")
    return finish(ScalarIndexSource{});
  bool transparent = name == "intent.binary" || name == "intent.unary" ||
                     name == "intent.cast";
  if (!transparent || isa<RankedTensorType>(value.getType()))
    return finish(ScalarIndexSource{nullptr, {}, true, false});
  SmallVector<Operation *> domains;
  bool opaque = false;
  for (Value operand : definition->getOperands()) {
    FailureOr<ScalarIndexSource> source =
        traceScalarIndexSourceImpl(operand, consumer, active);
    if (failed(source)) {
      active.erase(value);
      return failure();
    }
    opaque |= source->opaque;
    for (Operation *domain : source->domains)
      if (!llvm::is_contained(domains, domain))
        domains.push_back(domain);
  }
  Operation *uniqueDomain = domains.size() == 1 ? domains.front() : nullptr;
  return finish(ScalarIndexSource{uniqueDomain, std::move(domains), opaque, true});
}

} // namespace

FailureOr<SmallVector<Operation *>>
expandDomainSource(Value source, Operation &consumer) {
  SmallVector<Operation *> domains;
  if (failed(collectDomainSource(source, domains, &consumer)))
    return failure();
  return domains;
}

TensorIndexGroup tensorIndexGroup(Operation &operation,
                                  ArrayRef<IndexTerm> relation) {
  TensorIndexGroup group;
  for (const IndexTerm &term : relation) {
    if (term.kind != "value_index" || term.operands.size() != 1 ||
        !term.operands.front())
      continue;
    auto tensor = dyn_cast<RankedTensorType>(
        operation.getOperand(*term.operands.front()).getType());
    if (!tensor)
      continue;
    ++group.count;
    group.rank = std::max(group.rank, static_cast<unsigned>(tensor.getRank()));
  }
  return group;
}

FailureOr<llvm::SmallVector<IndexTerm>>
parseIndexRelation(Operation &operation) {
  auto relation = operation.getAttrOfType<ArrayAttr>("intent.index");
  if (!relation) {
    operation.emitOpError("requires an intent.index relation for target lowering");
    return failure();
  }
  llvm::SmallVector<IndexTerm> terms;
  for (Attribute attribute : relation) {
    auto dictionary = dyn_cast<DictionaryAttr>(attribute);
    auto kind = dictionary ? dictionary.getAs<StringAttr>("kind") : StringAttr();
    auto operands =
        dictionary ? dictionary.getAs<ArrayAttr>("operands") : ArrayAttr();
    auto staticValues =
        dictionary ? dictionary.getAs<ArrayAttr>("static") : ArrayAttr();
    if (!dictionary || !kind || !operands || !staticValues) {
      operation.emitOpError("contains a malformed intent.index term");
      return failure();
    }
    IndexTerm term{kind.getValue().str(), {}, {}};
    for (Attribute operand : operands) {
      if (isa<UnitAttr>(operand)) {
        term.operands.push_back(std::nullopt);
        continue;
      }
      auto position = dyn_cast<IntegerAttr>(operand);
      if (!position || position.getInt() < 0 ||
          static_cast<unsigned>(position.getInt()) >= operation.getNumOperands()) {
        operation.emitOpError("contains an invalid intent.index operand position");
        return failure();
      }
      term.operands.push_back(static_cast<unsigned>(position.getInt()));
    }
    for (Attribute value : staticValues) {
      if (isa<UnitAttr>(value)) {
        term.staticValues.push_back(std::nullopt);
        continue;
      }
      auto integer = dyn_cast<IntegerAttr>(value);
      if (!integer) {
        operation.emitOpError("contains a non-integer static index value");
        return failure();
      }
      term.staticValues.push_back(integer.getInt());
    }
    terms.push_back(std::move(term));
  }
  return terms;
}

FailureOr<ScalarIndexSource>
traceScalarIndexSource(Value value, Operation &consumer) {
  llvm::DenseSet<Value> active;
  return traceScalarIndexSourceImpl(value, consumer, active);
}

bool hasInBoundsPrecondition(Value index, Value view, unsigned axis,
                             Operation &access) {
  Operation *position = &access;
  while (Block *block = position->getBlock()) {
    for (Operation &candidate : *block) {
      if (&candidate == position)
        break;
      if (candidate.getName().getStringRef() != "intent.assume_in_bounds" ||
          candidate.getNumOperands() != 2 || candidate.getOperand(0) != index ||
          candidate.getOperand(1) != view)
        continue;
      auto candidateAxis =
          candidate.getAttrOfType<IntegerAttr>("intent.axis");
      if (candidateAxis && candidateAxis.getInt() == axis)
        return true;
    }
    Operation *parent = block->getParentOp();
    if (!parent)
      break;
    position = parent;
  }
  return false;
}

FailureOr<bool> hasDerivedScalarIndex(Operation &operation) {
  FailureOr<llvm::SmallVector<IndexTerm>> relation =
      parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  unsigned sourceAxis = 0;
  for (const IndexTerm &term : *relation) {
    if (term.kind == "new_axis")
      continue;
    unsigned currentAxis = sourceAxis++;
    if (term.kind != "value_index" || term.operands.size() != 1 ||
        !term.operands.front())
      continue;
    Value indexed = operation.getOperand(*term.operands.front());
    if (hasInBoundsPrecondition(indexed, operation.getOperand(0), currentAxis,
                                operation))
      continue;
    if (isa<RankedTensorType>(indexed.getType()))
      continue;
    FailureOr<ScalarIndexSource> source =
        traceScalarIndexSource(indexed, operation);
    if (failed(source))
      return failure();
    if (source->hasDomain() && source->transformed)
      return true;
  }
  return false;
}

FailureOr<bool> isWholeViewAccess(Operation &operation) {
  FailureOr<llvm::SmallVector<IndexTerm>> relation =
      parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  return !relation->empty() &&
         llvm::all_of(*relation, [](const IndexTerm &term) {
           return term.kind == "full_slice";
         });
}

} // namespace intent::target
