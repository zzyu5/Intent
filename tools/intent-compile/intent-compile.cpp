#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/Plan/IR/PlanDialect.h"
#include "Intent/Target/CuTile/Emission/Translate.h"
#include "Intent/Target/GPU/Realization/Realize.h"
#include "Intent/Target/TileLang/Emission/Translate.h"
#include "Intent/Target/Triton/Emission/Translate.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

namespace {

enum class TargetKind { Triton, CuTile, TileLang };

mlir::LogicalResult realize(
    mlir::ModuleOp module,
    const intent::gpu::DeviceCapabilities &device) {
  return intent::gpu::realizeKernel(module, device);
}

mlir::LogicalResult emit(mlir::ModuleOp module, TargetKind target,
                         llvm::raw_ostream &output) {
  switch (target) {
  case TargetKind::Triton:
    return intent::triton::emitTritonSource(module, output);
  case TargetKind::CuTile:
    return intent::cutile::emitCuTileSource(module, output);
  case TargetKind::TileLang:
    return intent::tilelang::emitTileLangSource(module, output);
  }
  llvm_unreachable("unknown Intent target");
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initialization(argc, argv);
  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input Intent Kernel MLIR>"),
      llvm::cl::init("-"));
  llvm::cl::opt<TargetKind> target(
      "target", llvm::cl::desc("target backend"), llvm::cl::Required,
      llvm::cl::values(
          clEnumValN(TargetKind::Triton, "triton", "emit Triton source"),
          clEnumValN(TargetKind::CuTile, "cutile", "emit cuTile source"),
          clEnumValN(TargetKind::TileLang, "tilelang",
                     "emit TileLang source")));
  llvm::cl::opt<int64_t> device("device", llvm::cl::desc("target device"),
                                llvm::cl::init(0));
  llvm::cl::opt<int64_t> computeUnits(
      "compute-units", llvm::cl::desc("GPU compute-unit count"),
      llvm::cl::Required);
  llvm::cl::opt<int64_t> sharedMemoryPerUnit(
      "shared-memory-per-unit",
      llvm::cl::desc("GPU shared-memory bytes per compute unit"),
      llvm::cl::Required);
  llvm::cl::opt<int64_t> registersPerUnit(
      "registers-per-unit",
      llvm::cl::desc("GPU registers per compute unit"), llvm::cl::Required);
  llvm::cl::opt<bool> matrixUnits(
      "matrix-units", llvm::cl::desc("GPU exposes matrix units"),
      llvm::cl::Required);
  llvm::cl::opt<bool> dynamicVectorWidth(
      "dynamic-vector-width",
      llvm::cl::desc("GPU worker vector width is dynamic"),
      llvm::cl::Required);
  llvm::cl::opt<std::string> irOutputFilename(
      "ir-output", llvm::cl::desc("realized Intent MLIR output"),
      llvm::cl::Required);
  llvm::cl::opt<std::string> sourceOutputFilename(
      "source-output", llvm::cl::desc("generated target source output"),
      llvm::cl::Required);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Intent kernel compiler\n");

  if (irOutputFilename == sourceOutputFilename) {
    llvm::errs() << "Intent compiler outputs must use distinct paths\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<intent::IntentDialect, intent::plan::IntentPlanDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<intent::IntentDialect, intent::plan::IntentPlanDialect>();

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  intent::gpu::DeviceCapabilities capabilities{
      device, computeUnits, sharedMemoryPerUnit, registersPerUnit, matrixUnits,
      dynamicVectorWidth};
  if (!module)
    return 1;
  if (mlir::failed(realize(*module, capabilities))) {
    llvm::errs() << "Intent realization failed\n";
    return 1;
  }
  if (mlir::failed(mlir::verify(*module))) {
    llvm::errs() << "realized Intent module verification failed\n";
    return 1;
  }

  std::string source;
  llvm::raw_string_ostream sourceStream(source);
  if (mlir::failed(emit(*module, target, sourceStream)))
    return 1;
  sourceStream.flush();

  std::error_code irError;
  llvm::ToolOutputFile irOutput(irOutputFilename, irError,
                                llvm::sys::fs::OF_Text);
  if (irError) {
    llvm::errs() << "cannot open realized MLIR output: " << irError.message()
                 << '\n';
    return 1;
  }
  std::error_code sourceError;
  llvm::ToolOutputFile sourceOutput(sourceOutputFilename, sourceError,
                                    llvm::sys::fs::OF_Text);
  if (sourceError) {
    llvm::errs() << "cannot open target source output: "
                 << sourceError.message() << '\n';
    return 1;
  }

  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();
  module->print(irOutput.os(), flags);
  irOutput.os() << '\n';
  sourceOutput.os() << source;
  irOutput.keep();
  sourceOutput.keep();
  return 0;
}
