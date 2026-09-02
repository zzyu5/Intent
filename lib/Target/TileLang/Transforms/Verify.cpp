#include "PassDetail.h"

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace intent::tilelang {
namespace {

bool isTileLangScalarType(Type type) {
  if (type.isIndex() ||
      isa<Float16Type, BFloat16Type, Float32Type, Float8E4M3FNType,
          Float8E5M2Type>(type))
    return true;
  auto integer = dyn_cast<IntegerType>(type);
  return integer && (integer.getWidth() == 1 || integer.getWidth() == 8 ||
                     integer.getWidth() == 16 || integer.getWidth() == 32 ||
                     integer.getWidth() == 64);
}

bool isAllowed(Operation *operation) {
  return isa<LaunchConfigOp, PipelineOp, ParallelOp, YieldOp, AllocOp, ClearOp,
             FillOp, SyncOp, CopyInOp, CopyOutOp, CastCopyOutOp, BufferLoadOp,
             BufferStoreOp,
             ViewLoadOp, ViewStoreOp, ReduceOp, ScanOp, GemmOp, SparseGemmOp,
             gpu::ParameterOp, gpu::PhysicalExprOp, gpu::ProgramIdOp,
             gpu::WorksetCoordinateOp, gpu::DelinearizeOp, gpu::DimOp,
             gpu::RangeOp,
             gpu::RangeBoundOp, gpu::UnaryOp, gpu::BinaryOp, gpu::CompareOp,
             gpu::SelectOp, gpu::CastOp, gpu::BitcastOp, gpu::MakeRecordOp,
             gpu::ExtractOp, arith::ConstantOp,
             scf::ForOp, scf::YieldOp, func::FuncOp, func::ReturnOp>(operation);
}

} // namespace

LogicalResult verifyTileLangKernel(func::FuncOp kernel) {
  auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
  if (!space || space.size() != 1)
    return kernel.emitError(
        "TileLang provider currently requires one explicit linear program space");
  auto topLevelLaunches = kernel.getBody().front().getOps<LaunchConfigOp>();
  if (!llvm::hasSingleElement(topLevelLaunches))
    return kernel.emitError(
        "TileLang provider program requires one top-level launch configuration");
  unsigned launchConfigs = 0;
  kernel.walk([&](LaunchConfigOp) { ++launchConfigs; });
  if (launchConfigs != 1)
    return kernel.emitError(
        "TileLang provider program contains a nested launch configuration");
  WalkResult result = kernel.walk([&](Operation *operation) {
    for (Type type : operation->getOperandTypes())
      if (isa<gpu::FragmentType>(type)) {
        operation->emitOpError(
            "still consumes an unbufferized shared GPU fragment");
        return WalkResult::interrupt();
      }
    for (Type type : operation->getResultTypes()) {
      if (isa<gpu::FragmentType>(type)) {
        operation->emitOpError(
            "still produces an unbufferized shared GPU fragment");
        return WalkResult::interrupt();
      }
      if (!isa<gpu::ViewType, gpu::RangeType, gpu::RecordType, BufferType>(type) &&
          !isTileLangScalarType(type)) {
        operation->emitOpError("has a result type outside the TileLang surface");
        return WalkResult::interrupt();
      }
      if (auto buffer = dyn_cast<BufferType>(type))
        if (!isTileLangScalarType(buffer.getElementType())) {
          operation->emitOpError(
              "produces a buffer element type outside the TileLang surface");
          return WalkResult::interrupt();
        }
    }
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (Type type : block.getArgumentTypes())
          if (isa<gpu::FragmentType>(type)) {
            operation->emitOpError(
                "contains an unbufferized fragment block argument");
            return WalkResult::interrupt();
          }
    if (auto loop = dyn_cast<scf::ForOp>(operation))
      if (loop.getNumResults() != 0) {
        loop.emitOpError("still carries an unbufferized SSA value");
        return WalkResult::interrupt();
      }
    if (isAllowed(operation))
      return WalkResult::advance();
    operation->emitOpError("is outside the closed TileLang provider surface");
    return WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

} // namespace intent::tilelang
