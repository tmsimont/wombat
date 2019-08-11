#ifndef NEURALNET_VECTOR_H_
#define NEURALNET_VECTOR_H_

#include <cstdint>

namespace wombat {
namespace neuralnet {

  /**
   * Although "Vector" may be a bad name given the use of std::vector throughout this
   * code... this is the "word vector" that is the result of the training algorithm.
   * I am keeping it "Vector" in case some day this is an "ngram vector" (as in the case of 
   * FastText).
   *
   * This class is basically a thin view into the neural network that is trained by
   * the word2vec algorithms.
   */
  class Vector {
    public:
      Vector(const int32_t index, const float * data, int32_t length) : 
        _index(index), _data(data), _length(length) {}

      /**
       * Get the index in the neural net this vector relates to. This index 
       * is also the word index in the word bag used to create the network.
       */
      int32_t getIndex();

      float get(const int32_t atIndex);

    private:
      const int32_t _index;
      const float * const _data;
      const int32_t _length;
  };
}
}

#endif
