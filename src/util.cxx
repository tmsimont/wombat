#include "util.h"

namespace wombat {
namespace util {
  static uint64_t next_random = (uint64_t) 1271240;
  uint64_t random() {
    next_random = next_random * (uint64_t) 25214903917 + 11;
    return next_random;
  }
}
}
