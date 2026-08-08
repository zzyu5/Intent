#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/Plan/IR/PlanDialect.h"
#include "Intent/Target/TileLang/IR/TileLangDialect.h"
#include "Intent/Target/TileLang/Realization/Realize.h"
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
      "architecture", llvm::cl::desc("TileLang target architecture"),
      llvm::cl::Required);
  llvm::cl::opt<int64_t> device("device", llvm::cl::desc("CUDA device"),
                                llvm::cl::init(0));
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Intent Kernel IR TileLang realizer\n");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<intent::IntentDialect, intent::plan::IntentPlanDialect,
                  intent::tilelang::plan::IntentTileLangDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<intent::IntentDialect, intent::plan::IntentPlanDialect,
                      intent::tilelang::plan::IntentTileLangDialect>();
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  intent::tilelang::TargetOptions target{architecture, device};
  if (!module || mlir::failed(intent::tilelang::realizeKernel(*module, target)) ||
      mlir::failed(mlir::verify(*module)))
    return 1;
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();
  module->print(llvm::outs(), flags);
  llvm::outs() << '\n';
  return 0;
}
