#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "Legalize.h"
#include "Intent/Dialect/CPU/IR/CPUDialect.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Transforms/PassManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace intent::mojo {
namespace {

void programDialects(DialectRegistry &registry) {
  registry.insert<cpu::IntentCPUDialect, arith::ArithDialect, func::FuncDialect,
                  linalg::LinalgDialect, math::MathDialect, memref::MemRefDialect,
                  scf::SCFDialect, vector::VectorDialect>();
}

LogicalResult finishGroup(ModuleOp module, StringRef name, LogicalResult result,
                          bool realized = false) {
  if (failed(result))
    return module.emitError() << "Mojo transformation failed: " << name;
  if (failed(cpu::verifyCPUProgram(module, realized)))
    return module.emitError() << "Mojo postcondition failed: " << name;
  return success();
}

class MaterializeProgramPass
    : public PassWrapper<MaterializeProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeProgramPass)
  void getDependentDialects(DialectRegistry &registry) const final {
    programDialects(registry);
  }
  StringRef getArgument() const final { return "intent-mojo-materialize-program"; }
  StringRef getDescription() const final {
    return "Expand selected CPU implementations and materialize native task dispatch";
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), prepareNativeProgram(module))))
      signalPassFailure();
  }
};

class FusePrivateComputationsPass
    : public PassWrapper<FusePrivateComputationsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusePrivateComputationsPass)
  void getDependentDialects(DialectRegistry &registry) const final {
    programDialects(registry);
  }
  StringRef getArgument() const final { return "intent-mojo-fuse-private-computations"; }
  StringRef getDescription() const final {
    return "Replay private computations under coordinate and storage lifetime proofs";
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), fusePrivateComputations(module))))
      signalPassFailure();
  }
};

class VectorizeProgramPass
    : public PassWrapper<VectorizeProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeProgramPass)
  void getDependentDialects(DialectRegistry &registry) const final {
    programDialects(registry);
  }
  StringRef getArgument() const final { return "intent-mojo-vectorize-program"; }
  StringRef getDescription() const final {
    return "Materialize vector traversals using coordinated implementation bindings";
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), vectorizeNativeProgram(module))))
      signalPassFailure();
  }
};

class FinalizeProgramPass
    : public PassWrapper<FinalizeProgramPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeProgramPass)
  void getDependentDialects(DialectRegistry &registry) const final {
    programDialects(registry);
  }
  StringRef getArgument() const final { return "intent-mojo-finalize-program"; }
  StringRef getDescription() const final {
    return "Legalize arithmetic, scratch and floating-point environment and verify the Mojo surface";
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), finalizeNativeProgram(module), true)))
      signalPassFailure();
  }
};

} // namespace

void registerMojoPasses() {
  PassRegistration<MaterializeProgramPass>();
  PassRegistration<FusePrivateComputationsPass>();
  PassRegistration<VectorizeProgramPass>();
  PassRegistration<FinalizeProgramPass>();
}

LogicalResult legalizeProgram(ModuleOp module) {
  if (failed(cpu::verifyCPUProgram(module, false))) return failure();
  PassManager manager(module.getContext(), ModuleOp::getOperationName());
  auto normalize = [&]() {
    manager.addPass(createCanonicalizerPass());
    manager.addPass(createCSEPass());
  };
  manager.addPass(std::make_unique<MaterializeProgramPass>());
  normalize();
  manager.addPass(std::make_unique<FusePrivateComputationsPass>());
  normalize();
  manager.addPass(std::make_unique<VectorizeProgramPass>());
  normalize();
  manager.addPass(std::make_unique<FinalizeProgramPass>());
  if (failed(intent::configurePassManager(manager))) return failure();
  return manager.run(module);
}

} // namespace intent::mojo
