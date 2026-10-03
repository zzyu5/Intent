#include "Views.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::weft_provider {
namespace {

bool isViewDefinition(Operation *operation) {
  return operation && (isAxisView(operation) ||
      isa<memref::SubViewOp, memref::CastOp, memref::ExtractStridedMetadataOp>(operation));
}

class CaptureReifier {
public:
  CaptureReifier(cpu::TasksOp task) : task(task), builder(&task.getBody().front(), task.getBody().front().begin()) {}

  LogicalResult run() {
    SmallVector<Value> captures(task.getCaptures());
    auto &body = task.getBody().front();
    for (auto [number, capture] : llvm::enumerate(captures)) {
      auto argument = body.getArgument(number + 1);
      if (argument.use_empty() || !isa<MemRefType>(capture.getType())) continue;
      auto replacement = materialize(capture);
      if (failed(replacement)) return failure();
      if (argument != *replacement) argument.replaceAllUsesWith(*replacement);
    }
    return success();
  }

private:
  Value capture(Value value) {
    for (auto [number, existing] : llvm::enumerate(task.getCaptures()))
      if (existing == value) {
        Value argument = task.getBody().front().getArgument(number + 1);
        mapping.map(value, argument);
        return argument;
      }
    task.getCapturesMutable().append(value);
    auto argument = task.getBody().front().addArgument(value.getType(), value.getLoc());
    mapping.map(value, argument);
    return argument;
  }

  FailureOr<Value> materialize(Value value) {
    if (auto found = mapping.lookupOrNull(value)) return found;
    Operation *definition = value.getDefiningOp();
    // These values belong to the host descriptor being reified. Capturing an
    // already computed size/stride keeps metadata-only storage out of the task
    // ABI and does not replay its scalar dependencies inside each task.
    if (!isa<MemRefType>(value.getType()) &&
        !isa_and_nonnull<arith::ConstantOp>(definition)) return capture(value);
    bool clone = isViewDefinition(definition) ||
        isa_and_nonnull<arith::ConstantOp>(definition);
    if (!clone) {
      if (isa<MemRefType>(value.getType())) {
        cpu::StorageAnalysis storage(task->getParentOfType<func::FuncOp>());
        auto origins = storage.origins(value);
        if (!origins.complete || origins.values.empty())
          return emitError(value.getLoc(), "Weft task capture has unresolved storage origins"), failure();
        if (!cpu::isContiguousDescriptor(value))
          return emitError(value.getLoc(), "Weft task capture requires a proved contiguous descriptor; its native view ABI does not carry strides"), failure();
      }
      return capture(value);
    }
    for (Value operand : definition->getOperands())
      if (failed(materialize(operand))) return failure();
    builder.clone(*definition, mapping);
    return mapping.lookup(value);
  }

  cpu::TasksOp task;
  OpBuilder builder;
  IRMapping mapping;
};

} // namespace

bool isAxisView(Operation *operation) {
  return isa_and_nonnull<memref::ReinterpretCastOp, memref::ExpandShapeOp,
                        memref::CollapseShapeOp, memref::TransposeOp>(operation);
}

FailureOr<cpu::ViewAxisProjection> queryAxisView(Value value) {
  auto projection = cpu::queryViewAxisProjection(value);
  if (failed(projection))
    emitError(value.getLoc(), "Weft view requires an offset-preserving axis permutation or unit-axis insertion/removal; general strided reinterpretation and axis flattening are unsupported");
  return projection;
}

LogicalResult reifyTaskViewCaptures(func::FuncOp function) {
  SmallVector<cpu::TasksOp> tasks;
  function.walk([&](cpu::TasksOp task) { tasks.push_back(task); });
  for (auto task : tasks) if (failed(CaptureReifier(task).run())) return failure();
  return success();
}

} // namespace intent::weft_provider
