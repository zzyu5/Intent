#ifndef INTENT_TARGET_CUTILE_CONFIG_TARGET_H
#define INTENT_TARGET_CUTILE_CONFIG_TARGET_H

#include <cstdint>
#include <string>

namespace intent::cutile {

struct TargetOptions {
  std::string architecture;
  int64_t device;
};

} // namespace intent::cutile

#endif
