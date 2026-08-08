#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/Plan/IR/PlanDialect.h"
#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "Intent/Target/CuTile/Realization/Realize.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

int main(int argc, char **argv) {
  llvm::InitLLVM initialization(argc, argv);
  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input Intent Kernel MLIR>"),
      llvm::cl::init("-"));
  llvm::cl::opt<std::string> architecture(
      "architecture", llvm::cl::desc("cuTile target architecture"),
      llvm::cl::Required);
  llvm::cl::opt<int64_t> device("device", llvm::cl::desc("CUDA device"),
                                llvm::cl::init(0));
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Intent Kernel IR cuTile realizer\n");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<intent::IntentDialect, intent::plan::IntentPlanDialect,
                  intent::cutile::plan::IntentCuTileDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<intent::IntentDialect, intent::plan::IntentPlanDialect,
                      intent::cutile::plan::IntentCuTileDialect>();
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  intent::cutile::TargetOptions target{architecture, device};
  if (!module || mlir::failed(intent::cutile::realizeKernel(*module, target)) ||
      mlir::failed(mlir::verify(*module)))
    return 1;
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();
  module->print(llvm::outs(), flags);
  llvm::outs() << '\n';
  return 0;
}
