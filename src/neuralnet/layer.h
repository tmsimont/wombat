#ifndef NEURAL_NET_LAYER_H_
#define NEURAL_NET_LAYER_H_

#include <stdlib.h>

#include <cstdint>

/**
 * Represents the weight matrix of a single neural net layer.
 */
namespace wombat {
namespace neuralnet {
  class Layer {
    public:
      Layer(int64_t numVectors, int32_t vectorLength);
      ~Layer();
      void randomize();
      const float * vectorAt(int64_t index);
    private:
      float * const _data;
      const int64_t _numVectors;
      const int32_t _vectorLength;
  };
}
}

#endif

