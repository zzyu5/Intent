#include "PassDetail.h"

#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace intent::tilelang {
namespace {

gpu::ParameterOp getOrCreateStages(func::FuncOp kernel) {
  gpu::ParameterOp existing;
  bool duplicate = false;
  kernel.walk([&](gpu::ParameterOp parameter) {
    if (parameter.getParameter().getName().getValue() != "NUM_STAGES")
      return;
    if (existing)
      duplicate = true;
    else
      existing = parameter;
  });
  auto candidates = DenseI64ArrayAttr::get(kernel.getContext(), {2, 3, 4});
  if (duplicate) {
    kernel.emitError("duplicates the TileLang NUM_STAGES parameter");
    return {};
  }
  if (existing) {
    gpu::ParameterAttr schema = existing.getParameter();
    if (schema.getRole() !=
            static_cast<uint32_t>(gpu::ParameterRole::ProviderStages) ||
        schema.getCandidates() != candidates) {
      existing.emitOpError(
          "NUM_STAGES is reused with a different role or candidate domain");
      return {};
    }
    return existing;
  }
  OpBuilder builder(&kernel.getBody().front(), kernel.getBody().front().begin());
  auto schema = gpu::ParameterAttr::get(
      kernel.getContext(), builder.getStringAttr("NUM_STAGES"),
      static_cast<uint32_t>(gpu::ParameterRole::ProviderStages), candidates);
  return builder.create<gpu::ParameterOp>(kernel.getLoc(), builder.getIndexType(),
                                          schema);
}

bool hasNativeContract(scf::ForOp loop) {
  return llvm::any_of(loop.getBody()->without_terminator(),
                      [](Operation &operation) { return isa<GemmOp>(operation); });
}

} // namespace

LogicalResult formPipelines(func::FuncOp kernel) {
  SmallVector<scf::ForOp> loops;
  kernel.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
    if (loop.getNumResults() == 0 && hasNativeContract(loop) &&
        !loop->getParentOfType<PipelineOp>())
      loops.push_back(loop);
  });
  if (loops.empty())
    return success();
  gpu::ParameterOp stages = getOrCreateStages(kernel);
  if (!stages)
    return failure();
  for (scf::ForOp loop : loops) {
    OpBuilder builder(loop);
    OperationState state(loop.getLoc(), PipelineOp::getOperationName());
    state.addOperands(stages.getResult());
    state.addRegion();
    auto pipeline = cast<PipelineOp>(builder.create(state));
    auto *body = new Block();
    pipeline.getBody().push_back(body);
    loop->moveBefore(body, body->end());
    OpBuilder::atBlockEnd(body).create<YieldOp>(loop.getLoc());
  }
  return success();
}

} // namespace intent::tilelang
