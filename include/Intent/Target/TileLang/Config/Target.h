#ifndef INTENT_TARGET_TILELANG_CONFIG_TARGET_H
#define INTENT_TARGET_TILELANG_CONFIG_TARGET_H

#include <cstdint>
#include <string>

namespace intent::tilelang {

struct TargetOptions {
  std::string architecture;
  int64_t device;
};

} // namespace intent::tilelang

#endif
