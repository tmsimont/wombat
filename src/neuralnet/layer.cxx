#include "neuralnet/layer.h"
#include "util.h"

#include <stdlib.h>

#include <cstring>
#include <new>

namespace wombat {
namespace neuralnet {
  Layer::Layer(int64_t numVectors, int32_t vectorLength) : _data(),
  _numVectors(numVectors),
  _vectorLength(vectorLength) {
    posix_memalign((void **)&_data, 128, (int64_t)numVectors * vectorLength * sizeof(float));
    if (_data == NULL) {
      throw std::bad_alloc();
    }
    for (int64_t i = 0; i < _numVectors; i++) {
      memset((void *) (_data + i * _vectorLength), 0, _vectorLength * sizeof(float));
    }
  }

  Layer::~Layer() {
    free((void *) _data);
  }

  Vector Layer::vectorAt(int64_t index) const {
    return Vector(index, _data + index * _vectorLength, _vectorLength);
  }

  void Layer::randomize() {
    for (int i = 0; i < _numVectors * _vectorLength; i++) {
      uint64_t random = util::random();
      _data[i] = (((random & 0xFFFF) / 65536.f) - 0.5f) / _vectorLength;
    }
  }
}
}
