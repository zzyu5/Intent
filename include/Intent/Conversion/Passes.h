#ifndef INTENT_CONVERSION_PASSES_H
#define INTENT_CONVERSION_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}

namespace intent {

std::unique_ptr<mlir::Pass> createConvertIntentToSCFPass();
void registerIntentConversionPasses();

} // namespace intent

#endif
