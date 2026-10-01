#ifndef INTENT_GPU_CONFIGURATIONPARAMETEROPINTERFACE_H
#define INTENT_GPU_CONFIGURATIONPARAMETEROPINTERFACE_H

#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"

namespace intent::gpu {
enum class ConfigurationBindingPhase { Shared, Provider, Deferred };
} // namespace intent::gpu

#include "Intent/Dialect/GPU/IR/ConfigurationParameterOpInterface.h.inc"

#endif
