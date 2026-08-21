#ifndef INTENT_TARGET_COMMON_LOWERING_DRIVER_H
#define INTENT_TARGET_COMMON_LOWERING_DRIVER_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace intent::target {

using ProgramSourceMaterialization = mlir::LogicalResult (*)(
    KernelModel, intent::plan::ProgramOp, intent::plan::SearchSpaceOp,
    llvm::raw_ostream &);
using ProviderPassPipeline = void (*)(mlir::PassManager &);
using ProviderProgramVerification = mlir::LogicalResult (*)(
    const KernelModel &, intent::plan::ProgramOp,
    intent::plan::SearchSpaceOp);

struct MaterializationTarget {
  llvm::StringRef provider;
  llvm::StringRef displayName;
  ProviderPassPipeline addProviderPasses;
  ProviderProgramVerification verifyProviderProgram;
  ProgramSourceMaterialization materializeProgramSource;
};

std::unique_ptr<mlir::Pass>
createMaterializeTargetProgramPass(const MaterializationTarget &target);
mlir::LogicalResult runTargetMaterializationPipeline(
    mlir::ModuleOp module, const MaterializationTarget &target);
mlir::LogicalResult translateTargetProgram(mlir::ModuleOp module,
                                           llvm::StringRef provider,
                                           llvm::raw_ostream &output);

} // namespace intent::target

#endif
