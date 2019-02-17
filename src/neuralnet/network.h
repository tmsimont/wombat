#ifndef NEURAL_NET_NETWORK_H_
#define NEURAL_NET_NETWORK_H_

#include "neuralnet/layer.h"
#include "vocabulary/wordbag/wordbag.h"

#include <memory>

/**
 * Encapsulates network layer data structures: Wi and Wo.
 * This is used for a simple shallow network that is central to the
 * word2vec/fasttext algorithm.
 */
namespace wombat {
namespace neuralnet {
  class Network {
    public:
      Network(std::shared_ptr<WordBag> wordbag, int32_t vectorLength);

    private:
      Layer _inputLayer;
      Layer _outputLayer;
  };
}
}

#endif
