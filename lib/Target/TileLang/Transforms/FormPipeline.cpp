#include "PassDetail.h"

#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"

#include <limits>

using namespace mlir;

namespace intent::tilelang {
namespace {

gpu::ParameterOp getOrCreateStages(func::FuncOp kernel,
                                  const gpu::TuningProfiles &profiles) {
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
  auto rows = profiles.get("tilelang", "stages", kernel.getLoc());
  if (failed(rows))
    return {};
  SmallVector<int64_t> values;
  for (const auto &row : *rows)
    if (row[0] <= std::numeric_limits<int32_t>::max())
      values.push_back(row[0]);
  if (values.empty()) {
    kernel.emitError("TileLang tuning profile has no legal pipeline stage counts");
    return {};
  }
  auto candidates = DenseI64ArrayAttr::get(kernel.getContext(), values);
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
      static_cast<uint32_t>(gpu::ParameterRole::ProviderStages),
      static_cast<uint32_t>(gpu::ParameterCategory::Provider),
      /*elementBitWidth=*/0, candidates);
  return builder.create<gpu::ParameterOp>(kernel.getLoc(), builder.getIndexType(),
                                          schema);
}

bool dependsOnSharedBuffer(Value value, llvm::DenseSet<Value> &active) {
  if (!active.insert(value).second)
    return false;
  if (auto load = value.getDefiningOp<BufferLoadOp>()) {
    active.erase(value);
    return load.getBuffer().getType().getSpace().getValue() ==
           BufferSpace::Shared;
  }
  Operation *producer = value.getDefiningOp();
  bool dependent =
      producer && llvm::any_of(producer->getOperands(), [&](Value operand) {
        return dependsOnSharedBuffer(operand, active);
      });
  active.erase(value);
  return dependent;
}

bool hasPipelineableContract(scf::ForOp loop) {
  SmallVector<Operation *> contracts;
  bool transfer = false;
  llvm::DenseSet<Value> contractOperands;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (auto gemm = dyn_cast<GemmOp>(operation)) {
      contracts.push_back(gemm.getOperation());
      contractOperands.insert(gemm.getLhs());
      contractOperands.insert(gemm.getRhs());
    } else if (auto gemm = dyn_cast<SparseGemmOp>(operation)) {
      contracts.push_back(gemm.getOperation());
      contractOperands.insert(gemm.getCompressed());
      contractOperands.insert(gemm.getMetadata());
      contractOperands.insert(gemm.getRhs());
    }
    transfer |= isa<CopyInOp>(operation);
  }
  if (contracts.empty())
    return false;
  llvm::DenseSet<Value> sharedRematerializedBuffers;
  loop.walk([&](BufferStoreOp store) {
    if (!contractOperands.contains(store.getBuffer()))
      return;
    llvm::DenseSet<Value> active;
    if (dependsOnSharedBuffer(store.getValue(), active)) {
      sharedRematerializedBuffers.insert(store.getBuffer());
    } else {
      transfer = true;
    }
  });
  for (Operation *operation : contracts) {
    SmallVector<Value> operands;
    if (auto gemm = dyn_cast<GemmOp>(operation)) {
      operands.append({gemm.getLhs(), gemm.getRhs()});
    } else {
      auto sparse = cast<SparseGemmOp>(operation);
      operands.append(
          {sparse.getCompressed(), sparse.getMetadata(), sparse.getRhs()});
    }
    if (llvm::any_of(operands, [&](Value operand) {
          return sharedRematerializedBuffers.contains(operand);
        }))
      return false;
  }
  return transfer;
}

} // namespace

LogicalResult formPipelines(func::FuncOp kernel,
                           const gpu::TuningProfiles &profiles) {
  SmallVector<scf::ForOp> candidates;
  kernel.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
    if (loop.getNumResults() == 0 && hasPipelineableContract(loop) &&
        !loop->getParentOfType<PipelineOp>())
      candidates.push_back(loop);
  });
  llvm::DenseSet<Operation *> candidateOperations;
  for (scf::ForOp loop : candidates)
    candidateOperations.insert(loop.getOperation());
  SmallVector<scf::ForOp> loops;
  for (scf::ForOp loop : candidates) {
    bool nestedCandidate = false;
    for (Operation *parent = loop->getParentOp(); parent;
         parent = parent->getParentOp())
      if (candidateOperations.contains(parent)) {
        nestedCandidate = true;
        break;
      }
    if (!nestedCandidate)
      loops.push_back(loop);
  }
  if (loops.empty())
    return success();
  gpu::ParameterOp stages = getOrCreateStages(kernel, profiles);
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
