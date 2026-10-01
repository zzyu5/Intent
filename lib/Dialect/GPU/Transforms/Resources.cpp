#include "Intent/Dialect/GPU/Transforms/Resources.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;

namespace intent::gpu {

void materializeDeferredReductionBounds(
    func::FuncOp kernel, ArrayRef<ValueRange> sourceGroups) {
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getRegistersPerUnit() <= 0)
    return;
  llvm::DenseMap<StringAttr, int64_t> knownParameters;
  for (Attribute declaration : getParameterDeclarations(kernel)) {
    auto schema = cast<ParameterAttr>(declaration);
    if (!schema.isDeferred() &&
        schema.getCategory() != ParameterCategory::Coverage)
      knownParameters[schema.getName()] = schema.getCandidates().asArrayRef().front();
  }
  auto resolve = [&](PhysicalExprAttr expression) -> std::optional<int64_t> {
    if (expression.getKind() != PhysicalExprKind::Parameter)
      return std::nullopt;
    auto found = knownParameters.find(expression.getSymbolName());
    return found == knownParameters.end() ? std::nullopt
                                          : std::optional<int64_t>(found->second);
  };
  SmallVector<PhysicalExprAttr> bounds;
  for (ValueRange sources : sourceGroups) {
    auto footprint = reductionRegisterFootprint(sources, kernel);
    if (footprint && !evaluatePhysicalExpression(footprint, resolve) &&
        !llvm::is_contained(bounds, footprint))
      bounds.push_back(footprint);
  }
  kernel.getContext()->getOrLoadDialect<cf::ControlFlowDialect>();
  OpBuilder builder = OpBuilder::atBlockBegin(&kernel.front());
  for (PhysicalExprAttr footprint : bounds) {
    Value count = builder.create<PhysicalExprOp>(
        kernel.getLoc(), builder.getIndexType(), footprint);
    Value maximum = builder.create<arith::ConstantIndexOp>(
        kernel.getLoc(), capabilities.getRegistersPerUnit());
    Value valid = builder.create<CompareOp>(kernel.getLoc(), builder.getI1Type(),
                                           count, maximum, ComparePredicate::Le);
    builder.create<cf::AssertOp>(
        kernel.getLoc(), valid, "reduction source exceeds the candidate register budget");
  }
}

} // namespace intent::gpu
