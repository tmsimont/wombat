#include "neuralnet/vector.h"

namespace wombat {
namespace neuralnet {
  const int32_t Vector::getIndex() const {
    return _index;
  }

  float Vector::get(const int32_t& atIndex) const {
    // TODO: validate the range
    return _data[atIndex];
  }

  void Vector::update(const int32_t& atIndex, const float& value) {
    // TODO: validate the range
    _data[atIndex] = value;
  }
}
}
