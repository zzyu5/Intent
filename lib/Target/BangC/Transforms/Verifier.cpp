#include "PassDetail.h"
#include "../Serialization/Scalar.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
namespace intent::bangc {

LogicalResult verifyProgram(ModuleOp module) {
  if (failed(dsa::verifyProgram(module))) return failure();
  auto architecture = module->getAttrOfType<StringAttr>("bangc.architecture");
  if (!architecture || architecture.getValue() != "mtp_372")
    return module.emitError("BANG C program requires the selected mtp_372 implementation profile");
  auto function = *module.getOps<func::FuncOp>().begin();
  if (failed(verifySurfaceOperations(function))) return failure();
  auto nram = function->getAttrOfType<IntegerAttr>("bangc.nram_bytes");
  auto wram = function->getAttrOfType<IntegerAttr>("bangc.wram_bytes");
  auto sram = function->getAttrOfType<IntegerAttr>("bangc.sram_bytes");
  if (!nram || !wram || !sram || nram.getInt() < 0 || wram.getInt() < 0 || sram.getInt() < 0)
    return function.emitError("BANG C program requires completed storage binding");
  auto walk = function.walk([&](Operation *operation) {
    if (auto allocation = dyn_cast<memref::AllocaOp>(operation)) {
      auto offset = operation->getAttrOfType<IntegerAttr>("bangc.offset");
      auto bytes = operation->getAttrOfType<IntegerAttr>("bangc.allocation_bytes");
      auto space = allocation.getType().getMemorySpaceAsInt();
      int64_t banks = space == dsa::matrixSpace ? 16 : 1;
      int64_t capacity = space == dsa::matrixSpace ? wram.getInt()
          : space == dsa::sharedSpace ? sram.getInt() : nram.getInt();
      if (!offset || !bytes || offset.getInt() < 0 || bytes.getInt() < 0 ||
          bytes.getInt() > capacity || offset.getInt() > (capacity - bytes.getInt()) / banks) {
        operation->emitError("BANG C allocation is missing or exceeds its bound storage interval");
        return WalkResult::interrupt();
      }
      if (space == dsa::matrixSpace) {
        auto layout = operation->getAttrOfType<StringAttr>("bangc.layout");
        if (!layout || layout.getValue() != "matrix_filter_interleaved64") {
          operation->emitError("BANG C matrix storage requires its selected filter layout");
          return WalkResult::interrupt();
        }
      }
    }
    if (isa<dsa::MatrixTileOp>(operation)) {
      auto implementation = operation->getAttrOfType<StringAttr>("bangc.implementation");
      if (!implementation || implementation.getValue() != "matmul_local_f32_accumulator") {
        operation->emitError("BANG C matrix tile has no selected accumulator implementation");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}


LogicalResult verifySurfaceOperations(func::FuncOp function) {
  auto walk = function.walk([&](Operation *op) {
    auto supportedType = [](Type type) {
      if (auto memory = dyn_cast<MemRefType>(type))
        type = memory.getElementType();
      return scalarType(type).has_value();
    };
    for (Type type : llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes()))
      if (!supportedType(type)) {
        op->emitError("type has no BANG C representation: ") << type;
        return WalkResult::interrupt();
      }
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (!supportedType(argument.getType())) {
            op->emitError("block argument has no BANG C representation: ")
                << argument.getType();
            return WalkResult::interrupt();
          }
    if (isStandardScalarOperation(op))
      return failed(verifyScalar(op)) ? WalkResult::interrupt() : WalkResult::advance();
    if (!isa<dsa::SynchronizeOp, dsa::GroupSynchronizeOp, dsa::GroupIdOp, dsa::GroupCountOp, dsa::LocalIdOp,
             dsa::IsMemoryCoreOp, dsa::StageTileOp, dsa::TaskIdOp, dsa::TaskCountOp, dsa::StrideOp, dsa::LoadScalarOp, dsa::StoreScalarOp,
             dsa::LoadTileOp, dsa::GatherPlanOp, dsa::GatherRowsOp, dsa::GroupGatherRowsOp, dsa::StoreTileOp, dsa::FillOp, dsa::IotaOp,
             dsa::IndexLayoutOp, dsa::IndexBinaryOp, dsa::BroadcastRowsOp, dsa::TransposeOp, dsa::SelectOp, dsa::MaskedFillOp, dsa::UnaryOp, dsa::BinaryOp,
             dsa::CastOp, dsa::CompareOp, dsa::CompareRangeOp, dsa::CompareRampOp, dsa::DivideCastOp, dsa::DivideRNOp, dsa::ReduceOp,
             dsa::PrepareMatrixOp, dsa::PrepareMatrixViewOp, dsa::MatrixTileOp,
             memref::DimOp, memref::AllocaOp, memref::ReinterpretCastOp, memref::LoadOp, memref::StoreOp, memref::CopyOp,
             scf::ForOp, scf::WhileOp, scf::IfOp, scf::ConditionOp, scf::YieldOp, func::FuncOp, func::ReturnOp>(op)) {
      op->emitError("operation is outside the bound BANG C surface"); return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}
} // namespace intent::bangc
