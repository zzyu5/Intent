#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/Plan/IR/PlanDialect.h"
#include "Intent/Target/CuTile/Lowering/TargetProgram.h"
#include "Intent/Target/GPU/Transforms/Passes.h"
#include "Intent/Target/TileLang/Lowering/TargetProgram.h"
#include "Intent/Target/Triton/Lowering/TargetProgram.h"
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

enum class ExitCode : int {
  Success = 0,
  Invocation = 1,
  KernelIR = 2,
  PhysicalProgram = 3,
  PhysicalVerification = 4,
  ProviderProgram = 5,
  ProviderVerification = 6,
  TerminalTranslation = 7,
  Output = 8,
};

int exitCode(ExitCode code) { return static_cast<int>(code); }

mlir::LogicalResult realize(
    mlir::ModuleOp module,
    const intent::gpu::DeviceCapabilities &device) {
  return intent::gpu::runPhysicalProgramPipeline(module, device);
}

mlir::LogicalResult materialize(mlir::ModuleOp module, TargetKind target) {
  switch (target) {
  case TargetKind::Triton:
    return intent::triton::materializeTritonProgram(module);
  case TargetKind::CuTile:
    return intent::cutile::materializeCuTileProgram(module);
  case TargetKind::TileLang:
    return intent::tilelang::materializeTileLangProgram(module);
  }
  llvm_unreachable("unknown Intent target");
}

mlir::LogicalResult translate(mlir::ModuleOp module, TargetKind target,
                              llvm::raw_ostream &output) {
  switch (target) {
  case TargetKind::Triton:
    return intent::triton::translateTritonProgram(module, output);
  case TargetKind::CuTile:
    return intent::cutile::translateCuTileProgram(module, output);
  case TargetKind::TileLang:
    return intent::tilelang::translateTileLangProgram(module, output);
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
      "ir-output", llvm::cl::desc("physical and target-program MLIR output"),
      llvm::cl::Required);
  llvm::cl::opt<std::string> sourceOutputFilename(
      "source-output", llvm::cl::desc("generated target source output"),
      llvm::cl::Required);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Intent kernel compiler\n");

  if (irOutputFilename == sourceOutputFilename) {
    llvm::errs() << "Intent compiler outputs must use distinct paths\n";
    return exitCode(ExitCode::Invocation);
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
    return exitCode(ExitCode::KernelIR);
  if (mlir::failed(realize(*module, capabilities))) {
    llvm::errs() << "Intent physical-program pipeline failed\n";
    return exitCode(ExitCode::PhysicalProgram);
  }
  if (mlir::failed(mlir::verify(*module))) {
    llvm::errs() << "Intent physical program verification failed\n";
    return exitCode(ExitCode::PhysicalVerification);
  }

  if (mlir::failed(materialize(*module, target))) {
    llvm::errs() << "Intent target-program materialization failed\n";
    return exitCode(ExitCode::ProviderProgram);
  }
  if (mlir::failed(mlir::verify(*module))) {
    llvm::errs() << "materialized target program verification failed\n";
    return exitCode(ExitCode::ProviderVerification);
  }

  std::string source;
  llvm::raw_string_ostream sourceStream(source);
  if (mlir::failed(translate(*module, target, sourceStream))) {
    llvm::errs() << "Intent terminal target translation failed\n";
    return exitCode(ExitCode::TerminalTranslation);
  }
  sourceStream.flush();

  std::error_code irError;
  llvm::ToolOutputFile irOutput(irOutputFilename, irError,
                                llvm::sys::fs::OF_Text);
  if (irError) {
    llvm::errs() << "cannot open physical-program MLIR output: " << irError.message()
                 << '\n';
    return exitCode(ExitCode::Output);
  }
  std::error_code sourceError;
  llvm::ToolOutputFile sourceOutput(sourceOutputFilename, sourceError,
                                    llvm::sys::fs::OF_Text);
  if (sourceError) {
    llvm::errs() << "cannot open target source output: "
                 << sourceError.message() << '\n';
    return exitCode(ExitCode::Output);
  }

  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();
  module->print(irOutput.os(), flags);
  irOutput.os() << '\n';
  sourceOutput.os() << source;
  irOutput.keep();
  sourceOutput.keep();
  return exitCode(ExitCode::Success);
}
