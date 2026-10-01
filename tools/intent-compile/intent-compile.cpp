#include "Intent/Compiler/Compiler.h"
#include "Intent/Compiler/Registration.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace {
using intent::compiler::Failure;
int exitCode(Failure failure) { return static_cast<int>(failure); }

bool writeFile(llvm::StringRef path,
               llvm::function_ref<void(llvm::raw_ostream &)> write) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error) { llvm::errs() << "cannot open " << path << ": " << error.message() << "\n"; return false; }
  write(output);
  output.close();
  if (output.has_error()) {
    llvm::errs() << "cannot write " << path << ": " << output.error().message() << "\n";
    output.clear_error();
    return false;
  }
  return true;
}
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initialization(argc, argv);
  intent::compiler::registerPasses();
  mlir::registerMLIRContextCLOptions();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input Intent KIR>"), llvm::cl::init("-"));
  llvm::cl::opt<std::string> target("target", llvm::cl::desc("source provider"));
  llvm::cl::opt<std::string> inputStage("input-stage", llvm::cl::init("kir"),
      llvm::cl::desc("Input IR stage: kir or shared; shared input requires the original target resources"));
  llvm::cl::opt<int64_t> computeUnits("compute-units", llvm::cl::init(0));
  llvm::cl::opt<int64_t> sharedMemoryPerUnit("shared-memory-per-unit", llvm::cl::init(0));
  llvm::cl::opt<int64_t> maxDynamicSharedMemoryPerBlock("max-dynamic-shared-memory-per-block", llvm::cl::init(0));
  llvm::cl::opt<int64_t> registersPerUnit("registers-per-unit", llvm::cl::init(0));
  llvm::cl::opt<int64_t> maxThreadsPerBlock("max-threads-per-block", llvm::cl::init(0));
  llvm::cl::opt<int64_t> computeCapabilityMajor("compute-capability-major", llvm::cl::init(0));
  llvm::cl::opt<int64_t> computeCapabilityMinor("compute-capability-minor", llvm::cl::init(-1));
  llvm::cl::opt<int64_t> singleToDoublePrecisionPerfRatio("single-to-double-precision-perf-ratio", llvm::cl::init(0));
  llvm::cl::opt<bool> matrixUnits("matrix-units", llvm::cl::init(false));
  llvm::cl::opt<bool> dynamicVectorWidth("dynamic-vector-width", llvm::cl::init(false));
  llvm::cl::opt<std::string> irOutputFilename("ir-output", llvm::cl::init(""));
  llvm::cl::opt<std::string> sourceOutputFilename("source-output", llvm::cl::init(""));
  llvm::cl::opt<std::string> metadataOutputFilename("metadata-output", llvm::cl::init(""));
  llvm::cl::opt<std::string> tuningConfigFilename("tuning-config", llvm::cl::init(""),
      llvm::cl::desc("JSON overrides of declared shared/provider profile families"));
  llvm::cl::opt<int64_t> cpuVectorBits("cpu-vector-bits", llvm::cl::init(0));
  llvm::cl::opt<int64_t> cpuWorkers("cpu-workers", llvm::cl::init(0));
  llvm::cl::opt<bool> cpuMatrixI8I32("cpu-matrix-i8-i32", llvm::cl::init(false));
  llvm::cl::opt<std::string> dsaArchitecture("dsa-architecture", llvm::cl::init("mtp_372"));
  llvm::cl::opt<int64_t> dsaTile("dsa-tile", llvm::cl::init(1024));
  llvm::cl::opt<int64_t> dsaTileM("dsa-tile-m", llvm::cl::init(16));
  llvm::cl::opt<int64_t> dsaTileN("dsa-tile-n", llvm::cl::init(64));
  llvm::cl::opt<int64_t> dsaTileK("dsa-tile-k", llvm::cl::init(64));
  llvm::cl::opt<int64_t> dsaRegionTile("dsa-region-tile", llvm::cl::init(64));
  llvm::cl::opt<std::string> dsaShapes("dsa-shapes", llvm::cl::init("{}"),
      llvm::cl::desc("JSON parameter shapes; -1 keeps an axis dynamic"));
  llvm::cl::opt<std::string> dsaStrides("dsa-strides", llvm::cl::init("{}"),
      llvm::cl::desc("JSON parameter element strides"));
  llvm::cl::opt<int64_t> dsaTasks("dsa-tasks", llvm::cl::init(16));
  llvm::cl::opt<int64_t> dsaLocalBytes("dsa-local-bytes", llvm::cl::init(512 * 1024));
  llvm::cl::opt<bool> stopAfterShared("stop-after-shared", llvm::cl::init(false),
      llvm::cl::desc("Stop after the selected family's complete shared pipeline"));
  llvm::cl::opt<bool> stopAfterKIR("stop-after-kir", llvm::cl::init(false),
      llvm::cl::desc("Normalize and verify KIR without selecting a target"));
  llvm::cl::opt<bool> compilerInfo("compiler-info", llvm::cl::init(false),
      llvm::cl::desc("Print compiled providers, stages, outputs and failure categories"));
  llvm::cl::ParseCommandLineOptions(argc, argv, "Intent compiler\n");

  if (compilerInfo) {
    llvm::outs() << llvm::json::Value(intent::compiler::information()) << "\n";
    return 0;
  }
  if (stopAfterKIR && stopAfterShared) {
    llvm::errs() << "--stop-after-kir and --stop-after-shared are mutually exclusive\n";
    return exitCode(Failure::Invocation);
  }
  intent::compiler::Request request;
  if (inputStage == "shared") request.inputStage = intent::compiler::InputStage::Shared;
  else if (inputStage != "kir") {
    llvm::errs() << "--input-stage must be kir or shared\n";
    return exitCode(Failure::Invocation);
  }
  if (!target.empty()) {
    request.provider = intent::compiler::parseProvider(target);
    if (!request.provider) {
      llvm::errs() << "unknown source provider: " << target << "\n";
      return exitCode(Failure::Invocation);
    }
  }
  request.stopAfter = stopAfterKIR ? intent::compiler::Stage::Kernel
      : stopAfterShared ? intent::compiler::Stage::Shared : intent::compiler::Stage::Provider;
  if (!stopAfterKIR && !request.provider) {
    llvm::errs() << "--target is required unless --stop-after-kir is selected\n";
    return exitCode(Failure::Invocation);
  }
  if (irOutputFilename.empty() ||
      (request.stopAfter == intent::compiler::Stage::Provider && sourceOutputFilename.empty())) {
    llvm::errs() << "--ir-output is required; provider compilation also requires --source-output\n";
    return exitCode(Failure::Output);
  }
  request.gpu = {computeUnits, sharedMemoryPerUnit, maxDynamicSharedMemoryPerBlock,
      registersPerUnit, maxThreadsPerBlock, computeCapabilityMajor, computeCapabilityMinor,
      singleToDoublePrecisionPerfRatio, matrixUnits, dynamicVectorWidth, false, false};
  request.cpu = {cpuVectorBits, cpuWorkers, cpuMatrixI8I32};
  request.dsa = {dsaArchitecture, dsaTile, dsaTileM, dsaTileN, dsaTileK,
      dsaRegionTile, dsaTasks, dsaLocalBytes, dsaShapes, dsaStrides};
  request.tuningConfig = tuningConfigFilename;
  {
    llvm::SmallString<256> directory(llvm::sys::fs::getMainExecutable(argv[0], reinterpret_cast<void *>(&main)));
    llvm::sys::path::remove_filename(directory);
    llvm::sys::path::append(directory, "profiles");
    request.profileDirectory = std::string(directory);
  }

  mlir::DialectRegistry registry;
  intent::compiler::registerDialects(registry);
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
  auto result = intent::compiler::compile(std::move(module), request);
  if (result.failure != Failure::None) return exitCode(result.failure);
  if (!writeFile(irOutputFilename, [&](llvm::raw_ostream &output) {
        result.module->print(output, mlir::OpPrintingFlags().enableDebugInfo());
        output << "\n";
      })) return exitCode(Failure::Output);
  if (request.stopAfter != intent::compiler::Stage::Provider) return 0;
  if (!writeFile(sourceOutputFilename, [&](llvm::raw_ostream &output) { output << result.source; }))
    return exitCode(Failure::Output);
  if (!metadataOutputFilename.empty() &&
      !writeFile(metadataOutputFilename, [&](llvm::raw_ostream &output) { output << result.metadata << "\n"; }))
    return exitCode(Failure::Output);
  return 0;
}
