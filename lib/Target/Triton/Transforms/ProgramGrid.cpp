#include "Intent/Target/Triton/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"

using namespace mlir;

namespace intent::triton {

LogicalResult legalizeProgramGrid(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = gpu::getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<gpu::DelinearizeOp> mappings;
  kernel.walk(
      [&](gpu::DelinearizeOp mapping) { mappings.push_back(mapping); });
  if (mappings.size() != 1)
    return success();

  gpu::DelinearizeOp mapping = mappings.front();
  if (mapping.getCoordinates().empty() ||
      mapping.getCoordinates().size() > 3)
    return success();
  auto linearProgram = mapping.getLinear().getDefiningOp<gpu::ProgramIdOp>();
  if (!linearProgram || linearProgram.getAxis() != 0)
    return success();

  kernel->setAttr(gpu::programSpaceAttr, mapping.getLaunchExtents());
  kernel->setAttr(
      gpu::gridRankAttr,
      IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                       mapping.getCoordinates().size()));
  OpBuilder builder(mapping);
  for (auto [axis, coordinate] : llvm::enumerate(mapping.getCoordinates())) {
    Value program = builder.create<gpu::ProgramIdOp>(
        mapping.getLoc(), builder.getIndexType(), static_cast<int64_t>(axis));
    coordinate.replaceAllUsesWith(program);
  }
  mapping.erase();
  if (linearProgram->use_empty())
    linearProgram.erase();
  return success();
}

} // namespace intent::triton
