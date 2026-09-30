#include "Intent/Dialect/GPU/Transforms/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include <algorithm>
#include <functional>
#include <optional>

using namespace mlir;

namespace intent::gpu {

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

llvm::StringSet<> referencedParameterSymbols(func::FuncOp kernel) {
  llvm::StringSet<> symbols;
  AttrTypeWalker walker;
  walker.addWalk([&](PhysicalExprAttr expression) {
    if (expression.getKind() == static_cast<uint32_t>(PhysicalExprKind::Parameter))
      symbols.insert(expression.getSymbol().getValue());
  });
  kernel.walk([&](Operation *operation) {
    walker.walk(operation->getAttrDictionary());
    for (Type type : operation->getResultTypes())
      walker.walk(type);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          walker.walk(argument.getType());
  });
  return symbols;
}

} // namespace

void eraseUnusedPhysicalParameters(func::FuncOp kernel) {
  auto referenced = referencedParameterSymbols(kernel);
  SmallVector<ParameterOp> unused;
  kernel.walk([&](ParameterOp parameter) {
    if (parameter.getResult().use_empty() &&
        !referenced.contains(parameter.getParameter().getName().getValue()))
      unused.push_back(parameter);
  });
  for (ParameterOp parameter : llvm::reverse(unused))
    parameter.erase();
}

LogicalResult replacePhysicalParameter(func::FuncOp kernel,
                                       ParameterOp previous,
                                       ParameterOp replacement) {
  if (!previous || !replacement || previous == replacement)
    return success();
  StringAttr previousName = previous.getParameter().getName();
  StringAttr replacementName = replacement.getParameter().getName();
  previous.getResult().replaceAllUsesWith(replacement.getResult());
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](PhysicalExprAttr expression)
      -> std::optional<Attribute> {
    if (expression.getKind() != static_cast<uint32_t>(PhysicalExprKind::Parameter) ||
        expression.getSymbol() != previousName)
      return std::nullopt;
    return PhysicalExprAttr::get(
        expression.getContext(), expression.getKind(), expression.getValue(),
        replacementName, expression.getOperands());
  });
  kernel.walk([&](Operation *operation) {
    for (Value result : operation->getResults())
      result.setType(replacer.replace(result.getType()));
    operation->setAttrs(cast<DictionaryAttr>(
        replacer.replace(operation->getAttrDictionary())));
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          argument.setType(replacer.replace(argument.getType()));
  });
  if (!previous.getResult().use_empty() ||
      referencedParameterSymbols(kernel).contains(previousName.getValue()))
    return previous.emitOpError(
        "physical parameter refinement left a second executable authority");
  previous.erase();
  return success();
}

} // namespace intent::gpu
