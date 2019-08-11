#include "neuralnet/vector.h"

namespace wombat {
namespace neuralnet {
  int32_t Vector::getIndex() {
    return _index;
  }

  float Vector::get(int32_t atIndex) const {
    // TODO: validate the range
    return _data[atIndex];
  }
}
}
