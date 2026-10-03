#include "../PassSupport.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Bufferization.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_DEF_CPUBUFFERIZEVALUES
#define GEN_PASS_DEF_CPUREUSEPRIVATESTORAGE
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

class BufferizeValuesPass
    : public impl::CPUBufferizeValuesBase<BufferizeValuesPass> {
public:
  using CPUBufferizeValuesBase::CPUBufferizeValuesBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(detail::verifyPassInput(module, getArgument(),
                                       CPUProgramStage::Values)) ||
        failed(detail::finishTransform(module, getArgument(),
                                        bufferizeValues(module))))
      signalPassFailure();
  }
};

class ReusePrivateStoragePass
    : public impl::CPUReusePrivateStorageBase<ReusePrivateStoragePass> {
public:
  using CPUReusePrivateStorageBase::CPUReusePrivateStorageBase;
  void runOnOperation() final {
    if (failed(detail::transformFunctions(getOperation(), getArgument(),
                                          reusePrivateStorage)))
      signalPassFailure();
  }
};

} // namespace
} // namespace intent::cpu
