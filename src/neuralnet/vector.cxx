#include "neuralnet/vector.h"

namespace wombat {
namespace neuralnet {
  int32_t Vector::getIndex() {
    return _index;
  }

  const float * Vector::getData() const {
    return _data;
  }
}
}
