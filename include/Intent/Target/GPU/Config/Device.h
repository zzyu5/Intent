#ifndef INTENT_TARGET_GPU_CONFIG_DEVICE_H
#define INTENT_TARGET_GPU_CONFIG_DEVICE_H

#include <cstdint>

namespace intent::gpu {

struct DeviceCapabilities {
  int64_t device;
  int64_t computeUnits;
  int64_t sharedMemoryPerUnit;
  int64_t registersPerUnit;
  bool matrixUnits;
  bool dynamicVectorWidth;
};

} // namespace intent::gpu

#endif
