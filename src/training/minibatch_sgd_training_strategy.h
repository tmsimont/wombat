#ifndef TRAINING_MINIBATCH_SGD_TRAINING_STRATEGY_H_
#define TRAINING_MINIBATCH_SGD_TRAINING_STRATEGY_H_

#include "training/batching/minibatch_indices.h"
#include "training/sgd/context.h"
#include "neuralnet/network.h"

#include <memory>
#include <vector>

using wombat::batching::MinibatchIndices;
using wombat::sgd::Context;

namespace wombat {
  /**
   * This defines an interface for a class that will train a neural network with 
   * Stochastic Gradient Descent with the given minibatch indices of network vectors.
   */
  // Threadsafe
  class MinibatchSgdTrainingStrategy {
    public:
      MinibatchSgdTrainingStrategy() {}
      ~MinibatchSgdTrainingStrategy() {}

      virtual void train(
          const neuralnet::Network& network,
          std::vector<std::unique_ptr<MinibatchIndices>> minibatchIndicesList,
          std::shared_ptr<Context> trainingContext) = 0;
  };
}
#endif
