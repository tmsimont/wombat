#ifndef NEURAL_NET_NETWORK_H_
#define NEURAL_NET_NETWORK_H_

#include "neuralnet/weights.h"

/**
 * Encapsulates network layer data structures: Wi and Wo.
 * This is used for a simple shallow network that is central to the
 * word2vec/fasttext algorithm.
 */
namespace wombat {
  class Network {
    public:
      Network();
      Weights getInputLayer();
      Weights getOutputLayer();
  };
}

#endif
