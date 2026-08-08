#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/Plan/IR/PlanDialect.h"
#include "Intent/Target/TileLang/Emission/Translate.h"
#include "Intent/Target/TileLang/IR/TileLangDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

int main(int argc, char **argv) {
  llvm::InitLLVM initialization(argc, argv);
  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input Intent MLIR>"),
      llvm::cl::init("-"));
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Intent MLIR to TileLang source\n");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<intent::IntentDialect, intent::plan::IntentPlanDialect,
                  intent::tilelang::plan::IntentTileLangDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  if (!module)
    return 1;
  return mlir::failed(intent::tilelang::emitTileLangSource(*module, llvm::outs()));
}
