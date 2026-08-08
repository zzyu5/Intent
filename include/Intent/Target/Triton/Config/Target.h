#ifndef INTENT_TARGET_TRITON_CONFIG_TARGET_H
#define INTENT_TARGET_TRITON_CONFIG_TARGET_H

#include <cstdint>
#include <string>

namespace intent::triton {

struct TargetOptions {
  std::string architecture;
  int64_t device;
  int64_t warpSize;
};

} // namespace intent::triton

#endif
