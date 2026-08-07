#include "Intent/Conversion/Passes.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "Intent/Dialect/Intent/IR/IntentTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

FailureOr<SmallVector<Value>> materializePointIndices(
    Operation *operation, ValueRange operands, PatternRewriter &rewriter) {
  auto relation = operation->getAttrOfType<ArrayAttr>("intent.index");
  if (!relation)
    return failure();

  SmallVector<Value> indices;
  for (Attribute attribute : relation) {
    auto term = dyn_cast<DictionaryAttr>(attribute);
    if (!term)
      return failure();
    auto kind = term.getAs<StringAttr>("kind");
    auto positions = term.getAs<ArrayAttr>("operands");
    auto staticValues = term.getAs<ArrayAttr>("static");
    if (!kind || !positions || !staticValues)
      return failure();

    if (kind.getValue() == "value_index") {
      if (positions.size() != 1)
        return failure();
      auto position = dyn_cast<IntegerAttr>(positions[0]);
      if (!position || position.getInt() < 0 ||
          static_cast<uint64_t>(position.getInt()) >= operands.size())
        return failure();
      Value index = operands[position.getInt()];
      if (!index.getType().isIndex())
        return failure();
      indices.push_back(index);
      continue;
    }

    if (kind.getValue() == "static_index") {
      if (staticValues.size() != 1)
        return failure();
      auto value = dyn_cast<IntegerAttr>(staticValues[0]);
      if (!value)
        return failure();
      indices.push_back(rewriter.create<arith::ConstantIndexOp>(
          operation->getLoc(), value.getInt()));
      continue;
    }

    return failure();
  }
  return indices;
}

struct LowerConstant final : OpRewritePattern<intent::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(intent::ConstantOp operation,
                                PatternRewriter &rewriter) const override {
    Type resultType = operation->getResult(0).getType();
    Attribute value = operation->getAttr("intent.value");
    if (!value)
      return rewriter.notifyMatchFailure(operation, "missing intent.value");

    if (resultType.isIndex()) {
      auto integer = dyn_cast<IntegerAttr>(value);
      if (!integer)
        return rewriter.notifyMatchFailure(operation, "index constant is not integer");
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(operation,
                                                          integer.getInt());
      return success();
    }

    if (auto integerType = dyn_cast<IntegerType>(resultType)) {
      if (auto boolean = dyn_cast<BoolAttr>(value)) {
        if (integerType.getWidth() != 1)
          return rewriter.notifyMatchFailure(operation,
                                              "bool constant requires i1");
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(operation, resultType,
                                                       boolean);
        return success();
      }
      auto integer = dyn_cast<IntegerAttr>(value);
      if (!integer)
        return rewriter.notifyMatchFailure(operation,
                                            "integer constant is not integer");
      auto converted = IntegerAttr::get(integerType, integer.getValue());
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(operation, resultType,
                                                     converted);
      return success();
    }

    if (auto floatType = dyn_cast<FloatType>(resultType)) {
      auto floating = dyn_cast<FloatAttr>(value);
      if (!floating)
        return rewriter.notifyMatchFailure(operation,
                                            "floating constant is not float");
      auto converted = FloatAttr::get(floatType, floating.getValueAsDouble());
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(operation, resultType,
                                                     converted);
      return success();
    }

    return rewriter.notifyMatchFailure(operation,
                                        "constant result is not scalar");
  }
};

struct LowerBinary final : OpRewritePattern<intent::BinaryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(intent::BinaryOp operation,
                                PatternRewriter &rewriter) const override {
    auto kind = operation->getAttrOfType<StringAttr>("intent.operator");
    if (!kind)
      return rewriter.notifyMatchFailure(operation, "missing intent.operator");
    Type type = operation->getResult(0).getType();
    Value lhs = operation->getOperand(0);
    Value rhs = operation->getOperand(1);

    if (isa<FloatType>(type)) {
      if (kind.getValue() == "add")
        rewriter.replaceOpWithNewOp<arith::AddFOp>(operation, lhs, rhs);
      else if (kind.getValue() == "subtract")
        rewriter.replaceOpWithNewOp<arith::SubFOp>(operation, lhs, rhs);
      else if (kind.getValue() == "multiply")
        rewriter.replaceOpWithNewOp<arith::MulFOp>(operation, lhs, rhs);
      else if (kind.getValue() == "true_divide")
        rewriter.replaceOpWithNewOp<arith::DivFOp>(operation, lhs, rhs);
      else
        return rewriter.notifyMatchFailure(operation,
                                            "unsupported floating binary operator");
      return success();
    }

    if (isa<IntegerType, IndexType>(type)) {
      if (kind.getValue() == "add")
        rewriter.replaceOpWithNewOp<arith::AddIOp>(operation, lhs, rhs);
      else if (kind.getValue() == "subtract")
        rewriter.replaceOpWithNewOp<arith::SubIOp>(operation, lhs, rhs);
      else if (kind.getValue() == "multiply")
        rewriter.replaceOpWithNewOp<arith::MulIOp>(operation, lhs, rhs);
      else
        return rewriter.notifyMatchFailure(operation,
                                            "unsupported integer binary operator");
      return success();
    }

    return rewriter.notifyMatchFailure(operation,
                                        "binary result is not scalar");
  }
};

struct LowerViewLoad final : OpRewritePattern<intent::ViewLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(intent::ViewLoadOp operation,
                                PatternRewriter &rewriter) const override {
    ValueRange operands = operation->getOperands();
    if (operands.empty() || !isa<MemRefType>(operands.front().getType()))
      return rewriter.notifyMatchFailure(operation,
                                          "view source is not a ranked memref");
    FailureOr<SmallVector<Value>> indices =
        materializePointIndices(operation, operands, rewriter);
    if (failed(indices))
      return rewriter.notifyMatchFailure(operation,
                                          "only point view indices are implemented");
    auto sourceType = cast<MemRefType>(operands.front().getType());
    if (indices->size() != static_cast<size_t>(sourceType.getRank()) ||
        operation->getResult(0).getType() != sourceType.getElementType())
      return rewriter.notifyMatchFailure(operation,
                                          "view load rank or element type mismatch");
    rewriter.replaceOpWithNewOp<memref::LoadOp>(operation, operands.front(),
                                                *indices);
    return success();
  }
};

struct LowerViewStore final : OpRewritePattern<intent::ViewStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(intent::ViewStoreOp operation,
                                PatternRewriter &rewriter) const override {
    ValueRange operands = operation->getOperands();
    if (operands.size() < 2 || !isa<MemRefType>(operands.front().getType()))
      return rewriter.notifyMatchFailure(operation,
                                          "view destination is not a ranked memref");
    auto valuePosition =
        operation->getAttrOfType<IntegerAttr>("intent.value_operand_index");
    if (!valuePosition || valuePosition.getInt() <= 0 ||
        static_cast<uint64_t>(valuePosition.getInt()) >= operands.size())
      return rewriter.notifyMatchFailure(operation,
                                          "invalid view store value position");
    FailureOr<SmallVector<Value>> indices =
        materializePointIndices(operation, operands, rewriter);
    if (failed(indices))
      return rewriter.notifyMatchFailure(operation,
                                          "only point view indices are implemented");
    auto destinationType = cast<MemRefType>(operands.front().getType());
    Value value = operands[valuePosition.getInt()];
    if (indices->size() != static_cast<size_t>(destinationType.getRank()) ||
        value.getType() != destinationType.getElementType())
      return rewriter.notifyMatchFailure(operation,
                                          "view store rank or element type mismatch");
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        operation, value, operands.front(), *indices);
    return success();
  }
};

struct LowerFor final : OpRewritePattern<intent::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(intent::ForOp operation,
                                PatternRewriter &rewriter) const override {
    ValueRange operands = operation->getOperands();
    if (operands.empty())
      return rewriter.notifyMatchFailure(operation, "for has no domain source");
    auto domain = operands.front().getDefiningOp<intent::DomainOp>();
    if (!domain)
      return rewriter.notifyMatchFailure(operation,
                                          "for source is not an intent.domain");
    ValueRange bounds = domain->getOperands();
    if (bounds.size() != 2 && bounds.size() != 3)
      return rewriter.notifyMatchFailure(operation,
                                          "domain must have start/stop[/step]");
    if (llvm::any_of(bounds,
                     [](Value value) { return !value.getType().isIndex(); }))
      return rewriter.notifyMatchFailure(operation,
                                          "domain bounds must have index type");
    if (operation->getNumRegions() != 1 ||
        !llvm::hasSingleElement(operation->getRegion(0)))
      return rewriter.notifyMatchFailure(operation,
                                          "for body must contain one block");

    SmallVector<Value> initial(operands.drop_front());
    Block &oldBody = operation->getRegion(0).front();
    if (oldBody.getNumArguments() != initial.size() + 1)
      return rewriter.notifyMatchFailure(operation,
                                          "for block argument schema mismatch");
    auto yield = dyn_cast<intent::YieldOp>(oldBody.getTerminator());
    if (!yield || yield->getNumOperands() != initial.size())
      return rewriter.notifyMatchFailure(operation,
                                          "for body must end in intent.yield");

    Value step = bounds.size() == 3
                     ? bounds[2]
                     : rewriter.create<arith::ConstantIndexOp>(
                           operation.getLoc(), 1);
    auto lowered = rewriter.create<scf::ForOp>(
        operation.getLoc(), bounds[0], bounds[1], step, initial);
    Block *newBody = lowered.getBody();
    rewriter.eraseOp(newBody->getTerminator());

    IRMapping mapping;
    for (auto [oldArgument, newArgument] :
         llvm::zip(oldBody.getArguments(), newBody->getArguments()))
      mapping.map(oldArgument, newArgument);

    rewriter.setInsertionPointToEnd(newBody);
    for (Operation &nested : oldBody.without_terminator())
      rewriter.clone(nested, mapping);
    SmallVector<Value> yielded;
    for (Value value : yield->getOperands())
      yielded.push_back(mapping.lookupOrDefault(value));
    rewriter.create<scf::YieldOp>(operation.getLoc(), yielded);
    rewriter.replaceOp(operation, lowered.getResults());
    return success();
  }
};

struct EraseDeadDomain final : OpRewritePattern<intent::DomainOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(intent::DomainOp operation,
                                PatternRewriter &rewriter) const override {
    if (!operation->getResult(0).use_empty())
      return failure();
    rewriter.eraseOp(operation);
    return success();
  }
};

struct LowerReturn final : OpRewritePattern<intent::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(intent::ReturnOp operation,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(operation,
                                                operation->getOperands());
    return success();
  }
};

struct ConvertIntentToSCFPass final
    : PassWrapper<ConvertIntentToSCFPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertIntentToSCFPass)

  StringRef getArgument() const final { return "convert-intent-to-scf"; }
  StringRef getDescription() const final {
    return "Lower implemented Intent logical loops and scalar view operations to SCF";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    bool signatureFailure = false;
    module.walk([&](func::FuncOp function) {
      SmallVector<Type> inputs(function.getFunctionType().getInputs());
      for (auto [index, argument] :
           llvm::enumerate(function.getArguments())) {
        auto view = dyn_cast<intent::ViewType>(argument.getType());
        if (!view)
          continue;
        auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
        if (!tensor) {
          function.emitError("Intent view must contain a ranked tensor schema");
          signatureFailure = true;
          return;
        }
        auto memref = MemRefType::get(tensor.getShape(), tensor.getElementType());
        inputs[index] = memref;
        argument.setType(memref);
      }
      function.setType(FunctionType::get(function.getContext(), inputs,
                                         function.getResultTypes()));
    });
    if (signatureFailure) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerConstant, LowerBinary, LowerViewLoad, LowerViewStore,
                 LowerFor, EraseDeadDomain, LowerReturn>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    bool unsupported = false;
    module.walk([&](Operation *operation) {
      if (operation->getName().getDialectNamespace() == "intent") {
        operation->emitError("Intent-to-SCF lowering is not implemented for this operation");
        unsupported = true;
      }
    });
    if (unsupported)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> intent::createConvertIntentToSCFPass() {
  return std::make_unique<ConvertIntentToSCFPass>();
}

void intent::registerIntentConversionPasses() {
  PassRegistration<ConvertIntentToSCFPass>();
}
