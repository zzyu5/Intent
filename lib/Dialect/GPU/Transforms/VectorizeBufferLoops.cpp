#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

void vectorizeIndependentUpdates(BufferOp buffer, func::FuncOp kernel,
                                 uint64_t &source, int64_t &dimension) {
  BufferType type = buffer.getResult().getType();
  if (buffer->getBlock() != &kernel.front() ||
      type.getScope().getValue() != BufferScope::ProgramPrivate ||
      type.getLifetime().getValue() != BufferLifetime::Program ||
      !isa<IntegerType, FloatType>(type.getElementType()))
    return;
  SmallVector<scf::ForOp> loops;
  for (Operation *user : buffer.getResult().getUsers())
    if (auto store = dyn_cast<StoreOp>(user))
      if (auto loop = dyn_cast<scf::ForOp>(store->getParentOp()))
        if (!llvm::is_contained(loops, loop))
          loops.push_back(loop);
  for (scf::ForOp loop : loops) {
    auto step = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (loop.getNumResults() || !step || step.value() != 1 ||
        !canPredicateScalarBlock(*loop.getBody()))
      continue;
    SmallVector<StoreOp> stores;
    for (Operation &operation : loop.getBody()->without_terminator())
      if (auto store = dyn_cast<StoreOp>(operation))
        stores.push_back(store);
    if (stores.size() != 1 || stores.front().getResource() != buffer.getResult())
      continue;
    StoreOp store = stores.front();
    std::optional<unsigned> independentAxis;
    for (auto [axis, coordinate] :
         llvm::zip(store.getSourceAxes(), store.getCoordinates()))
      if (coordinate == loop.getInductionVar()) {
        independentAxis = axis;
        break;
      }
    if (!independentAxis)
      continue;
    if (llvm::any_of(llvm::zip(store.getSourceAxes(), store.getCoordinates()),
                    [&](const auto &entry) {
          return std::get<0>(entry) != *independentAxis &&
                 !loop.isDefinedOutsideOfLoop(std::get<1>(entry));
        }))
      continue;
    // A fresh logical buffer cannot alias other resources. Writes are injective
    // in this axis, and reads of that allocation stay in the same iteration's
    // slice. Only these effect-independent iterations may execute together.
    bool independent = llvm::all_of(
        loop.getBody()->without_terminator(), [&](Operation &operation) {
          auto load = dyn_cast<LoadOp>(operation);
          if (!load || load.getResource() != buffer.getResult())
            return true;
          for (auto [axis, coordinate] :
               llvm::zip(load.getSourceAxes(), load.getCoordinates()))
            if (axis == *independentAxis)
              return coordinate == loop.getInductionVar();
          return false;
        });
    if (!independent)
      continue;
    auto logicalExtent = cast<PhysicalExprAttr>(type.getShape()[*independentAxis]);
    if (logicalExtent.getKind() !=
            static_cast<uint32_t>(PhysicalExprKind::Constant) ||
        logicalExtent.getValue() <= 1)
      continue;
    uint64_t padded = llvm::PowerOf2Ceil(
        static_cast<uint64_t>(logicalExtent.getValue()));
    int64_t bytes = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
    if (!padded || padded > static_cast<uint64_t>(
            capabilities.getMaxDynamicSharedMemoryPerBlock() / bytes))
      continue;

    OpBuilder builder(loop);
    Location location = loop.getLoc();
    auto extent = PhysicalExprAttr::get(
        kernel.getContext(), static_cast<uint32_t>(PhysicalExprKind::Constant),
        padded, builder.getStringAttr(""), builder.getArrayAttr({}));
    auto ordinal = AxisMapAttr::get(kernel.getContext(), source++,
                                   *independentAxis, dimension++, 0, false);
    auto shape = FragmentType::get(
        kernel.getContext(), builder.getIndexType(), builder.getArrayAttr({extent}),
        builder.getArrayAttr({ordinal}), 1, type.getOwner());
    auto boolean = FragmentType::get(
        kernel.getContext(), builder.getI1Type(), shape.getShape(),
        shape.getAxisMaps(), shape.getValidity(), shape.getOwner());
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value end = builder.create<arith::ConstantIndexOp>(
        location, logicalExtent.getValue());
    Value width = builder.create<arith::ConstantIndexOp>(location, padded);
    Value range = builder.create<MakeRangeOp>(
        location, shape, zero, width, loop.getStep(), zero, end,
        ordinal.getSourceId(), ordinal.getSourceAxis(), ordinal.getDerived());
    Value lower = builder.create<BroadcastOp>(location, shape, loop.getLowerBound());
    Value upper = builder.create<BroadcastOp>(location, shape, loop.getUpperBound());
    Value limit = builder.create<BroadcastOp>(location, shape, end);
    Value active = builder.create<CompareOp>(
        location, boolean, range, lower, ComparePredicate::Ge);
    Value below = builder.create<CompareOp>(
        location, boolean, range, upper, ComparePredicate::Lt);
    active = builder.create<BinaryOp>(
        location, boolean, active, below, BinaryOperator::LogicalAnd);
    Value inBuffer = builder.create<CompareOp>(
        location, boolean, range, limit, ComparePredicate::Lt);
    active = builder.create<BinaryOp>(
        location, boolean, active, inBuffer, BinaryOperator::LogicalAnd);
    IRMapping values;
    values.map(loop.getInductionVar(), range);
    for (Operation &operation : loop.getBody()->without_terminator())
      clonePredicatedScalarOperation(builder, &operation, values, active, shape);
    loop.erase();
  }
}

} // namespace

LogicalResult vectorizeBufferLoops(ModuleOp module) {
  auto physical = getPhysicalKernel(module);
  if (failed(physical))
    return failure();
  func::FuncOp kernel = *physical;
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!llvm::all_of(space, [](Attribute attribute) {
        auto extent = cast<PhysicalExprAttr>(attribute);
        return extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
               extent.getValue() == 1;
      }))
    return success();
  auto [source, dimension] = nextPhysicalAxisIdentities(kernel);
  SmallVector<BufferOp> buffers;
  kernel.walk([&](BufferOp buffer) { buffers.push_back(buffer); });
  for (BufferOp buffer : buffers)
    vectorizeIndependentUpdates(buffer, kernel, source, dimension);
  eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace intent::gpu
