#include "Intent/Conversion/KIRToGPU/KIRToGPU.h"
#include "Intent/Conversion/KIRToCPU/KIRToCPU.h"
#include "Intent/Transforms/CPU/Passes.h"
#include "Intent/Target/Mojo/Serialization/Serializer.h"
#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "Intent/Target/CuTile/Serialization/Serializer.h"
#include "Intent/Target/CuTile/Transforms/Passes.h"
#include "Intent/Target/TileLang/IR/TileLangDialect.h"
#include "Intent/Target/TileLang/Serialization/Serializer.h"
#include "Intent/Target/TileLang/Transforms/Passes.h"
#include "Intent/Target/Triton/IR/TritonDialect.h"
#include "Intent/Target/Triton/Serialization/Serializer.h"
#include "Intent/Target/Triton/Transforms/Passes.h"
#include "Intent/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace {

enum class TargetKind { Triton, CuTile, TileLang, Mojo };

enum class ExitCode : int {
  Success = 0,
  Invocation = 1,
  KernelIR = 2,
  PhysicalProgram = 3,
  PhysicalProgramVerification = 4,
  ProviderProgram = 5,
  ProviderProgramVerification = 6,
  TerminalTranslation = 7,
  CompilerOutput = 8,
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
          clEnumValN(TargetKind::TileLang, "tilelang", "TileLang DSL"),
          clEnumValN(TargetKind::Mojo, "mojo", "Mojo CPU SIMD")));

  // Compile-call inputs are parsed but not interpreted before the shared
  // executable GPU Program exists.
  llvm::cl::opt<int64_t> computeUnits("compute-units", llvm::cl::init(0));
  llvm::cl::opt<int64_t> sharedMemoryPerUnit("shared-memory-per-unit",
                                             llvm::cl::init(0));
  llvm::cl::opt<int64_t> maxDynamicSharedMemoryPerBlock(
      "max-dynamic-shared-memory-per-block", llvm::cl::init(0));
  llvm::cl::opt<int64_t> registersPerUnit("registers-per-unit",
                                          llvm::cl::init(0));
  llvm::cl::opt<int64_t> maxThreadsPerBlock("max-threads-per-block",
                                            llvm::cl::init(0));
  llvm::cl::opt<int64_t> computeCapabilityMajor("compute-capability-major",
                                                llvm::cl::init(0));
  llvm::cl::opt<int64_t> computeCapabilityMinor("compute-capability-minor",
                                                llvm::cl::init(-1));
  llvm::cl::opt<bool> matrixUnits("matrix-units", llvm::cl::init(false));
  llvm::cl::opt<bool> dynamicVectorWidth("dynamic-vector-width",
                                         llvm::cl::init(false));
  llvm::cl::opt<std::string> irOutputFilename("ir-output",
                                              llvm::cl::init(""));
  llvm::cl::opt<std::string> tuningConfigFilename(
      "tuning-config", llvm::cl::desc("JSON overrides of declared shared/provider profile families"),
      llvm::cl::init(""));
  llvm::cl::opt<std::string> sourceOutputFilename("source-output",
                                                  llvm::cl::init(""));
  llvm::cl::opt<std::string> metadataOutputFilename("metadata-output", llvm::cl::init(""));
  llvm::cl::opt<int64_t> cpuVectorBits("cpu-vector-bits", llvm::cl::init(0));
  llvm::cl::opt<int64_t> cpuWorkers("cpu-workers", llvm::cl::init(0));
  llvm::cl::opt<bool> stopAfterShared(
      "stop-after-shared",
      llvm::cl::desc("stop after the full shared GPU verifier"),
      llvm::cl::init(false));
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Intent canonical KIR compiler boundary\n");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<intent::IntentDialect, intent::gpu::IntentGPUDialect,
                  intent::cutile::IntentCuTileDialect,
                  intent::tilelang::IntentTileLangDialect,
                  intent::triton::IntentTritonDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<intent::IntentDialect, intent::gpu::IntentGPUDialect,
                      intent::cutile::IntentCuTileDialect,
                      intent::tilelang::IntentTileLangDialect,
                      intent::triton::IntentTritonDialect,
                      mlir::arith::ArithDialect, mlir::func::FuncDialect,
                      mlir::scf::SCFDialect>();

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  if (!module || mlir::failed(intent::verifyKernelModule(*module)))
    return exitCode(ExitCode::KernelIR);
  llvm::SmallString<256> profilesDirectory(
      llvm::sys::fs::getMainExecutable(argv[0], reinterpret_cast<void *>(&main)));
  llvm::sys::path::remove_filename(profilesDirectory);
  llvm::sys::path::append(profilesDirectory, "profiles");
  auto profilePath = [&](llvm::StringRef name) {
    llvm::SmallString<256> path(profilesDirectory);
    llvm::sys::path::append(path, name);
    return path.str().str();
  };
  std::string source;
  std::string metadata = "{}";
  if (target == TargetKind::Mojo) {
    if (stopAfterShared) {
      llvm::errs() << "--stop-after-shared selects the GPU family, not CPU\n";
      return exitCode(ExitCode::Invocation);
    }
    context.loadDialect<mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                        mlir::memref::MemRefDialect, mlir::vector::VectorDialect>();
    if (mlir::failed(intent::lowerCanonicalKIRToCPU(*module)))
      return exitCode(ExitCode::PhysicalProgram);
    if (mlir::failed(intent::cpu::runCPUPasses(*module, cpuVectorBits, cpuWorkers,
                                              profilePath("cpu.json"), tuningConfigFilename)))
      return exitCode(ExitCode::PhysicalProgramVerification);
    metadata.clear();
    if (mlir::failed(intent::mojo::legalizeProgram(*module)))
      return exitCode(ExitCode::ProviderProgramVerification);
    if (mlir::failed(intent::mojo::serializeProgram(*module, source, metadata)))
      return exitCode(ExitCode::TerminalTranslation);
  } else {
  const llvm::StringRef sharedColumns[] = {
      "ownership_m", "ownership_n", "reduction", "reduction_outer", "scan",
      "traversal_workers", "traversal_group"};
  const llvm::StringRef tritonColumns[] = {"warps", "stages", "ctas"};
  const llvm::StringRef valueColumn[] = {"value"};
  const intent::gpu::TuningProfileSource profileSources[] = {
      {"shared", profilePath("shared.json"), sharedColumns},
      {"triton", profilePath("triton.json"), tritonColumns},
      {"cutile", profilePath("cutile.json"), valueColumn},
      {"tilelang", profilePath("tilelang.json"), valueColumn}};
  auto profiles = intent::gpu::TuningProfiles::read(
      module->getLoc(), profileSources, tuningConfigFilename);
  if (mlir::failed(profiles))
    return exitCode(ExitCode::Invocation);
  intent::GPUCapabilities capabilities{
      computeUnits,
      sharedMemoryPerUnit,
      maxDynamicSharedMemoryPerBlock,
      registersPerUnit,
      maxThreadsPerBlock,
      computeCapabilityMajor,
      computeCapabilityMinor,
      matrixUnits,
      dynamicVectorWidth};
  if (mlir::failed(intent::lowerCanonicalKIRToGPU(*module, capabilities))) {
    llvm::errs() << "Intent KIR-to-GPU construction failed\n";
    return exitCode(ExitCode::PhysicalProgram);
  }
  if (mlir::failed(intent::gpu::runSharedGPUPasses(*module, *profiles)))
    return exitCode(ExitCode::PhysicalProgramVerification);
  if (stopAfterShared) {
    if (irOutputFilename.empty()) {
      llvm::errs() << "--ir-output is required with --stop-after-shared\n";
      return exitCode(ExitCode::CompilerOutput);
    }
    std::error_code error;
    llvm::raw_fd_ostream irOutput(irOutputFilename, error,
                                 llvm::sys::fs::OF_Text);
    if (error) {
      llvm::errs() << "cannot open physical IR output: " << error.message()
                   << "\n";
      return exitCode(ExitCode::CompilerOutput);
    }
    module->print(irOutput);
    irOutput << "\n";
    return exitCode(ExitCode::Success);
  }
  mlir::LogicalResult provider = mlir::failure();
  mlir::LogicalResult serialized = mlir::failure();
  switch (target) {
  case TargetKind::Triton:
    provider = intent::triton::legalizeGPUProgram(*module, *profiles);
    if (mlir::succeeded(provider))
      serialized = intent::triton::serializeProgram(*module, source);
    break;
  case TargetKind::CuTile:
    provider = intent::cutile::legalizeGPUProgram(*module, *profiles);
    if (mlir::succeeded(provider))
      serialized = intent::cutile::serializeProgram(*module, source);
    break;
  case TargetKind::TileLang:
    provider = intent::tilelang::legalizeGPUProgram(*module, *profiles);
    if (mlir::succeeded(provider))
      serialized = intent::tilelang::serializeProgram(*module, source);
    break;
  case TargetKind::Mojo:
    llvm_unreachable("CPU construction is selected before GPU construction");
  }
  if (mlir::failed(provider))
    return exitCode(ExitCode::ProviderProgramVerification);
  if (mlir::failed(serialized))
    return exitCode(ExitCode::TerminalTranslation);
  }
  if (irOutputFilename.empty() || sourceOutputFilename.empty()) {
    llvm::errs() << "both --ir-output and --source-output are required\n";
    return exitCode(ExitCode::CompilerOutput);
  }
  std::error_code error;
  llvm::raw_fd_ostream irOutput(irOutputFilename, error,
                               llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "cannot open physical IR output: " << error.message() << "\n";
    return exitCode(ExitCode::CompilerOutput);
  }
  module->print(irOutput);
  irOutput << "\n";
  irOutput.close();
  llvm::raw_fd_ostream sourceOutput(sourceOutputFilename, error,
                                   llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "cannot open provider source output: " << error.message()
                 << "\n";
    return exitCode(ExitCode::CompilerOutput);
  }
  sourceOutput << source;
  sourceOutput.close();
  if (!metadataOutputFilename.empty()) {
    llvm::raw_fd_ostream metadataOutput(metadataOutputFilename, error, llvm::sys::fs::OF_Text);
    if (error) {
      llvm::errs() << "cannot open artifact metadata output: " << error.message() << "\n";
      return exitCode(ExitCode::CompilerOutput);
    }
    metadataOutput << metadata << "\n";
  }
  return exitCode(ExitCode::Success);
}
