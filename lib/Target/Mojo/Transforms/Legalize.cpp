#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::mojo {
namespace {

bool directAtomicAdd(Type element) {
  return element.isF32() || element.isF64() || element.isSignlessInteger(32) || element.isSignlessInteger(64);
}

bool supportedType(Type type) {
  if (auto vector = dyn_cast<VectorType>(type)) {
    int64_t width = vector.getNumElements();
    return vector.getRank() == 1 && supportedType(vector.getElementType()) &&
        width > 0 && (width & (width - 1)) == 0;
  }
  if (auto memory = dyn_cast<MemRefType>(type)) return supportedType(memory.getElementType());
  return type.isIndex() || type.isF16() || type.isBF16() || type.isF32() || type.isF64() ||
      isa<Float8E4M3FNType, Float8E5M2Type>(type) ||
      type.isSignlessInteger(8) || type.isSignlessInteger(16) || type.isSignlessInteger(32) ||
      type.isSignlessInteger(64) || type.isInteger(1);
}

bool needsFloatingPointEnvironment(Operation *scope) {
  auto floating = [](Type type) {
    if (auto vector = dyn_cast<VectorType>(type)) type = vector.getElementType();
    return isa<FloatType>(type);
  };
  return scope->walk([&](Operation *operation) {
    // Calls may depend on the caller's FP state even with an integer-only ABI.
    if (isa<func::CallOp>(operation) ||
        llvm::any_of(operation->getOperandTypes(), floating) ||
        llvm::any_of(operation->getResultTypes(), floating))
      return WalkResult::interrupt();
    return WalkResult::advance();
  }).wasInterrupted();
}

void promotePrivateScratch(func::FuncOp function, int64_t budget) {
  cpu::PhysicalProgramAnalysis physical(function);
  auto allocations = physical.allocations();
  auto alignmentOf = [](auto allocation) -> int64_t {
    Type element = allocation.getType().getElementType();
    int64_t bytes = element.isIndex() ? 8 : (element.getIntOrFloatBitWidth() + 7) / 8;
    return allocation.getAlignment().value_or(bytes);
  };
  auto fits = [&](int64_t bytes, int64_t alignment) {
    return bytes >= 0 && bytes <= budget && alignment > 0 && alignment - 1 <= budget - bytes;
  };
  // Count every static slot, including mutually exclusive lifetimes, to bound
  // the worker frame without relying on the lower compiler's stack coloring.
  for (auto facts : allocations) {
    if (!facts.stack) continue;
    auto allocation = facts.value.getDefiningOp<memref::AllocaOp>();
    int64_t alignment = alignmentOf(allocation);
    if (!facts.bytes || !fits(*facts.bytes, alignment)) return;
    budget -= *facts.bytes + alignment - 1;
  }
  for (auto facts : allocations) {
    auto allocation = facts.value.getDefiningOp<memref::AllocOp>();
    if (!allocation || !facts.bytes || *facts.bytes <= 0 || *facts.bytes > 1024 ||
        !allocation.getType().getLayout().isIdentity()) continue;
    int64_t alignment = alignmentOf(allocation);
    if (!fits(*facts.bytes, alignment)) continue;
    SmallVector<Value> aliases{facts.value};
    llvm::SmallDenseSet<Value> seen;
    SmallVector<Operation *> users;
    memref::DeallocOp end;
    bool closed = true;
    for (unsigned i = 0; i < aliases.size() && closed; ++i) {
      Value value = aliases[i];
      if (!seen.insert(value).second) continue;
      for (Operation *user : value.getUsers()) {
        if (auto deallocation = dyn_cast<memref::DeallocOp>(user)) {
          if (value != facts.value || end || user->getBlock() != allocation->getBlock()) {
            closed = false; break;
          }
          end = deallocation;
          continue;
        }
        if (isa<memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp,
                memref::ExtractStridedMetadataOp>(user)) {
          for (Value result : user->getResults())
            if (isa<MemRefType>(result.getType())) aliases.push_back(result);
        } else if (!isa<memref::LoadOp, memref::StoreOp, memref::DimOp,
                       vector::LoadOp, vector::StoreOp, memref::PrefetchOp>(user)) {
          closed = false; break;
        }
        users.push_back(user);
      }
    }
    if (!closed || !end || !allocation->isBeforeInBlock(end)) continue;
    if (!llvm::all_of(users, [&](Operation *user) {
          Operation *ancestor = allocation->getBlock()->findAncestorOpInBlock(*user);
          return ancestor && allocation->isBeforeInBlock(ancestor) && ancestor->isBeforeInBlock(end);
        })) continue;
    OpBuilder builder(allocation);
    auto stack = builder.create<memref::AllocaOp>(allocation.getLoc(), allocation.getType(),
        ValueRange{}, allocation.getAlignmentAttr());
    end.erase();
    allocation.replaceAllUsesWith(stack.getResult());
    allocation.erase();
    budget -= *facts.bytes + alignment - 1;
  }
}

LogicalResult checkSurface(ModuleOp module) {
  bool invalid = false;
  module.walk([&](Operation *operation) {
    if (isa<ModuleOp, func::FuncOp, func::ReturnOp, scf::YieldOp, scf::ConditionOp, cpu::TaskYieldOp>(operation)) return;
    bool supported = isa<arith::ConstantOp, arith::AddFOp, arith::AddIOp,
        arith::SubFOp, arith::SubIOp, arith::MulFOp, arith::MulIOp,
        arith::DivFOp, arith::DivSIOp, arith::RemSIOp, arith::DivUIOp, arith::RemUIOp,
        arith::MinSIOp, arith::MaxSIOp, arith::MinUIOp, arith::MaxUIOp, arith::NegFOp,
        arith::IndexCastOp, arith::IndexCastUIOp, arith::BitcastOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp,
        arith::ExtFOp, arith::TruncFOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
        arith::MaxNumFOp, arith::MinNumFOp, arith::MaximumFOp, arith::MinimumFOp,
        arith::CmpFOp, arith::SelectOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
        arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
        math::FmaOp, math::SqrtOp, math::ExpOp, math::Exp2Op, math::LogOp, math::TanhOp,
        math::SinOp, math::CosOp, math::FloorOp, math::ErfOp, math::AbsFOp, math::AbsIOp, math::PowFOp, memref::DimOp,
        memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp, memref::LoadOp, memref::StoreOp,
        memref::ExtractStridedMetadataOp,
        memref::AllocaOp, memref::AllocOp, memref::DeallocOp, memref::PrefetchOp,
        vector::LoadOp, vector::StoreOp, vector::BroadcastOp, vector::FromElementsOp, vector::ShuffleOp, vector::StepOp,
        vector::ExtractElementOp, arith::CmpIOp, scf::IfOp, scf::ForOp, scf::WhileOp, cpu::TaskDispatchOp,
        cpu::AtomicLoadOp, cpu::AtomicStoreOp, cpu::AtomicRMWOp, cpu::AtomicCompareExchangeOp>(operation);
    supported &= llvm::all_of(operation->getOperandTypes(), supportedType);
    supported &= llvm::all_of(operation->getResultTypes(), supportedType);
    if (auto constant = dyn_cast<arith::ConstantOp>(operation))
      supported &= isa<IntegerAttr, FloatAttr, DenseElementsAttr>(constant.getValue());
    if (auto dimension = dyn_cast<memref::DimOp>(operation))
      supported &= dimension.getConstantIndex().has_value();
    if (auto stack = dyn_cast<memref::AllocaOp>(operation))
      supported &= stack.getType().hasStaticShape();
    if (auto prefetch = dyn_cast<memref::PrefetchOp>(operation))
      supported &= !prefetch.getIsWrite() && prefetch.getLocalityHint() == 3 && prefetch.getIsDataCache();
    if (auto conditional = dyn_cast<scf::IfOp>(operation))
      supported &= llvm::none_of(conditional.getResultTypes(), [](Type type) { return isa<MemRefType>(type); });
    if (auto loop = dyn_cast<scf::WhileOp>(operation))
      supported &= llvm::none_of(loop.getResultTypes(), [](Type type) { return isa<MemRefType>(type); });
    if (isa<cpu::AtomicLoadOp, cpu::AtomicStoreOp, cpu::AtomicRMWOp, cpu::AtomicCompareExchangeOp>(operation)) {
      Type element = cast<MemRefType>(operation->getOperand(0).getType()).getElementType();
      supported &= directAtomicAdd(element) || element.isF16() || element.isBF16();
      if (auto rmw = dyn_cast<cpu::AtomicRMWOp>(operation))
        supported &= directAtomicAdd(element) && rmw.getKind() == AtomicRMWKind::Add;
    }
    if (!supported) {
      operation->emitError("current operation/type has no supported Mojo CPU surface form: ")
          << operation->getName() << "; operands=" << operation->getOperandTypes()
          << "; results=" << operation->getResultTypes();
      invalid = true;
    }
  });
  return failure(invalid);
}

LogicalResult expandAtomicUpdates(ModuleOp module) {
  SmallVector<memref::GenericAtomicRMWOp> genericUpdates;
  module.walk([&](memref::GenericAtomicRMWOp operation) { genericUpdates.push_back(operation); });
  for (auto operation : genericUpdates) {
    if (!operation.getResult().use_empty())
      return operation.emitError("Mojo generic atomic update currently requires a discarded scatter result");
    OpBuilder builder(operation);
    Location loc = operation.getLoc();
    Block &body = operation.getRegion().front();
    Value yielded = cast<memref::AtomicYieldOp>(body.getTerminator()).getResult();
    auto *combine = yielded.getDefiningOp();
    if (combine && isa<arith::AddFOp, arith::AddIOp>(combine) && directAtomicAdd(yielded.getType()) &&
        std::distance(body.begin(), body.end()) == 2 &&
        llvm::is_contained(combine->getOperands(), body.getArgument(0))) {
      Value value = combine->getOperand(combine->getOperand(0) == body.getArgument(0) ? 1 : 0);
      if (value != body.getArgument(0)) {
        builder.create<cpu::AtomicRMWOp>(loc, value.getType(), operation.getMemref(), value,
            operation.getIndices(), AtomicOrdering::Relaxed, AtomicRMWKind::Add, false);
        operation.erase();
        continue;
      }
    }
    Type type = operation.getResult().getType();
    Value initial = builder.create<cpu::AtomicLoadOp>(loc, type, operation.getMemref(), operation.getIndices(), AtomicOrdering::Relaxed);
    Value pending = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    auto retry = builder.create<scf::WhileOp>(loc, TypeRange{type, builder.getI1Type()}, ValueRange{initial, pending});
    auto *before = builder.createBlock(&retry.getBefore(), {}, {type, builder.getI1Type()}, {loc, loc});
    Value no = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    Value again = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, before->getArgument(1), no);
    builder.create<scf::ConditionOp>(loc, again, before->getArguments());
    auto *after = builder.createBlock(&retry.getAfter(), {}, {type, builder.getI1Type()}, {loc, loc});
    IRMapping mapping;
    mapping.map(body.getArgument(0), after->getArgument(0));
    for (Operation &instruction : body.without_terminator()) builder.clone(instruction, mapping);
    auto exchange = builder.create<cpu::AtomicCompareExchangeOp>(loc, type, builder.getI1Type(), operation.getMemref(),
        after->getArgument(0), mapping.lookupOrDefault(yielded), operation.getIndices(), AtomicOrdering::Relaxed);
    builder.create<scf::YieldOp>(loc, exchange.getResults());
    operation.erase();
  }
  SmallVector<cpu::AtomicRMWOp> updates;
  module.walk([&](cpu::AtomicRMWOp operation) {
    if (operation.getKind() != AtomicRMWKind::Add || !directAtomicAdd(operation.getValue().getType()))
      updates.push_back(operation);
  });
  for (auto operation : updates) {
    OpBuilder builder(operation);
    Location loc = operation.getLoc();
    Type type = operation.getValue().getType();
    Value initial = builder.create<cpu::AtomicLoadOp>(loc, type, operation.getTarget(), operation.getIndices(), AtomicOrdering::Relaxed);
    Value pending = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    auto retry = builder.create<scf::WhileOp>(loc, TypeRange{type, builder.getI1Type()}, ValueRange{initial, pending});
    auto *before = builder.createBlock(&retry.getBefore(), {}, {type, builder.getI1Type()}, {loc, loc});
    Value no = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    Value again = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, before->getArgument(1), no);
    builder.create<scf::ConditionOp>(loc, again, before->getArguments());
    auto *after = builder.createBlock(&retry.getAfter(), {}, {type, builder.getI1Type()}, {loc, loc});
    Value old = after->getArgument(0), value = operation.getValue(), desired;
    switch (operation.getKind()) {
    case AtomicRMWKind::Exchange: desired = value; break;
    case AtomicRMWKind::Maximum:
      desired = isa<FloatType>(type) ? Value(builder.create<arith::MaximumFOp>(loc, old, value))
          : operation.getUnsignedInteger() ? Value(builder.create<arith::MaxUIOp>(loc, old, value))
          : Value(builder.create<arith::MaxSIOp>(loc, old, value)); break;
    case AtomicRMWKind::Minimum:
      desired = isa<FloatType>(type) ? Value(builder.create<arith::MinimumFOp>(loc, old, value))
          : operation.getUnsignedInteger() ? Value(builder.create<arith::MinUIOp>(loc, old, value))
          : Value(builder.create<arith::MinSIOp>(loc, old, value)); break;
    case AtomicRMWKind::BitwiseAnd: desired = builder.create<arith::AndIOp>(loc, old, value); break;
    case AtomicRMWKind::BitwiseOr: desired = builder.create<arith::OrIOp>(loc, old, value); break;
    case AtomicRMWKind::BitwiseXor: desired = builder.create<arith::XOrIOp>(loc, old, value); break;
    case AtomicRMWKind::Add:
      desired = isa<FloatType>(type) ? Value(builder.create<arith::AddFOp>(loc, old, value))
                                    : Value(builder.create<arith::AddIOp>(loc, old, value)); break;
    }
    auto exchange = builder.create<cpu::AtomicCompareExchangeOp>(loc, type, builder.getI1Type(),
        operation.getTarget(), old, desired, operation.getIndices(), operation.getOrdering());
    builder.create<scf::YieldOp>(loc, exchange.getResults());
    operation.getOldValue().replaceAllUsesWith(retry.getResult(0));
    operation.erase();
  }
  return success();
}

}

LogicalResult legalizeProgram(ModuleOp module) {
  auto capabilities = module->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities");
  if (!capabilities || (capabilities.getVectorBits() != 256 && capabilities.getVectorBits() != 512))
    return module.emitError("Mojo native currently requires an AVX2 or AVX512 CPU capability");
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (failed(cpu::materializeTaskDispatches(function))) return failure();
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (failed(materializeRegisterContractions(function)) ||
        failed(cpu::materializeStructuredComputations(function))) return failure();
    // Implementation supplies may introduce additional independent worksets.
    if (failed(cpu::isolateTasks(function)) ||
        failed(cpu::materializeTaskDispatches(function))) return failure();
  }
  auto normalize = [&]() {
    PassManager manager(module.getContext());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(createCSEPass());
    return manager.run(module);
  };
  if (failed(normalize())) return failure();
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (failed(cpu::fuseIntermediateBuffers(function))) return failure();
  if (failed(normalize())) return failure();
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto bindings = function->getAttrOfType<ArrayAttr>("intent_cpu.implementations");
    if (!bindings || bindings.empty()) return function.emitError("Mojo lowering requires selected implementations");
    auto binding = cast<cpu::ImplementationAttr>(bindings[0]);
    for (Attribute value : bindings)
      for (StringRef name : {"vector_width", "register_replicas", "reduction_replicas"})
        if (cpu::implementationParameter(cast<cpu::ImplementationAttr>(value), name) != cpu::implementationParameter(binding, name))
          return function.emitError("Mojo loop materialization requires coordinated vector bindings");
    if (failed(cpu::fuseReductionTraversals(function)) ||
        failed(cpu::vectorizeLoops(function,
            cpu::implementationParameter(binding, "vector_width"),
            cpu::implementationParameter(binding, "register_replicas"),
            cpu::implementationParameter(binding, "reduction_replicas")))) return failure();
  }
  if (failed(normalize())) return failure();
  SmallVector<math::RsqrtOp> roots;
  module.walk([&](math::RsqrtOp operation) { roots.push_back(operation); });
  for (auto operation : roots) {
    OpBuilder b(operation);
    Type type = operation.getType();
    TypedAttr one;
    if (auto vector = dyn_cast<VectorType>(type))
      one = DenseElementsAttr::get(vector, b.getFloatAttr(vector.getElementType(), 1.0));
    else one = b.getFloatAttr(type, 1.0);
    Value root = b.create<math::SqrtOp>(operation.getLoc(), operation.getOperand());
    Value unit = b.create<arith::ConstantOp>(operation.getLoc(), type, one);
    Value result = b.create<arith::DivFOp>(operation.getLoc(), unit, root);
    operation.getResult().replaceAllUsesWith(result);
    operation.erase();
  }
  RewritePatternSet integerDivision(module.getContext());
  arith::populateCeilFloorDivExpandOpsPatterns(integerDivision);
  if (failed(applyPatternsGreedily(module, std::move(integerDivision)))) return failure();
  if (failed(expandAtomicUpdates(module))) return failure();
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    promotePrivateScratch(function, capabilities.getPrivateBytes());
  if (failed(cpu::verifyCPUProgram(module, true)) || failed(checkSurface(module))) return failure();
  SmallVector<Block *> scopes;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal()) continue;
    if (needsFloatingPointEnvironment(function)) scopes.push_back(&function.front());
    function.walk([&](cpu::TaskDispatchOp dispatch) {
      if (needsFloatingPointEnvironment(dispatch)) scopes.push_back(&dispatch.getBody().front());
    });
  }
  if (scopes.empty()) return success();
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto enter = builder.create<func::FuncOp>(module.getLoc(), "intent_cpu_enter_ieee",
      builder.getFunctionType({}, {builder.getI32Type()}));
  auto leave = builder.create<func::FuncOp>(module.getLoc(), "intent_cpu_leave_ieee",
      builder.getFunctionType({builder.getI32Type()}, {}));
  enter.setPrivate(); leave.setPrivate();
  enter->setAttr("cpu.external_runtime", builder.getUnitAttr());
  leave->setAttr("cpu.external_runtime", builder.getUnitAttr());
  for (Block *scope : scopes) {
    builder.setInsertionPointToStart(scope);
    Value previous = builder.create<func::CallOp>(module.getLoc(), enter, ValueRange{}).getResult(0);
    builder.setInsertionPoint(scope->getTerminator());
    builder.create<func::CallOp>(module.getLoc(), leave, ValueRange{previous});
  }
  return cpu::verifyCPUProgram(module, true);
}

}
