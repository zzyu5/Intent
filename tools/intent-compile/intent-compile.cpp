#include "Intent/Conversion/KIRToGPU/KIRToGPU.h"
#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace {

enum class TargetKind { Triton, CuTile, TileLang };

enum class ExitCode : int {
  Success = 0,
  Invocation = 1,
  KernelIR = 2,
  PhysicalProgram = 3,
};

int exitCode(ExitCode code) { return static_cast<int>(code); }

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initialization(argc, argv);
  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input canonical Intent KIR>"),
      llvm::cl::init("-"));
  llvm::cl::opt<TargetKind> target(
      "target", llvm::cl::desc("eventual target backend"), llvm::cl::Required,
      llvm::cl::values(
          clEnumValN(TargetKind::Triton, "triton", "Triton DSL"),
          clEnumValN(TargetKind::CuTile, "cutile", "cuTile DSL"),
          clEnumValN(TargetKind::TileLang, "tilelang", "TileLang DSL")));

  // Compile-call inputs are parsed but not interpreted before the shared
  // executable GPU Program exists.
  llvm::cl::opt<int64_t> device("device", llvm::cl::init(0));
  llvm::cl::opt<int64_t> computeUnits("compute-units", llvm::cl::init(0));
  llvm::cl::opt<int64_t> sharedMemoryPerUnit("shared-memory-per-unit",
                                             llvm::cl::init(0));
  llvm::cl::opt<int64_t> registersPerUnit("registers-per-unit",
                                          llvm::cl::init(0));
  llvm::cl::opt<bool> matrixUnits("matrix-units", llvm::cl::init(false));
  llvm::cl::opt<bool> dynamicVectorWidth("dynamic-vector-width",
                                         llvm::cl::init(false));
  llvm::cl::opt<std::string> irOutputFilename("ir-output",
                                              llvm::cl::init(""));
  llvm::cl::opt<std::string> sourceOutputFilename("source-output",
                                                  llvm::cl::init(""));
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Intent canonical KIR compiler boundary\n");

  (void)target;
  (void)device;
  (void)computeUnits;
  (void)sharedMemoryPerUnit;
  (void)registersPerUnit;
  (void)matrixUnits;
  (void)dynamicVectorWidth;
  (void)irOutputFilename;
  (void)sourceOutputFilename;

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<intent::IntentDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<intent::IntentDialect>();

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  if (!module || mlir::failed(intent::verifyKernelModule(*module)))
    return exitCode(ExitCode::KernelIR);
  if (mlir::failed(intent::lowerCanonicalKIRToGPU(*module))) {
    llvm::errs() << "Intent KIR-to-GPU boundary is not implemented\n";
    return exitCode(ExitCode::PhysicalProgram);
  }
  return exitCode(ExitCode::Success);
}
