#ifndef TRAINING_WORDS_TO_MINIBATCHED_SGD_TRAINER_H_
#define TRAINING_WORDS_TO_MINIBATCHED_SGD_TRAINER_H_

#include "neuralnet/network.h"
#include "training/batching/minibatch_provider.h"
#include "training/data/source/sentence_source.h"
#include "training/sgd/context.h"
#include "training/minibatch_sgd_training_strategy.h"

#include <memory>
#include <vector>

using wombat::batching::MinibatchProvider;
using wombat::sgd::Context;

namespace wombat {

  /**
   * Given a word source, parse out sentences and use batching strategy to build
   * batch indices, and update the neural network with an MinibatchSgdTrainingStrategy.
   */
  // not threadsafe
  class WordsToMinibatchedSgdTrainer {
    public:
      WordsToMinibatchedSgdTrainer(
          const neuralnet::Network& network,
          std::unique_ptr<SentenceSource> sentenceSource) 
        : _network(network),
          _sentenceSource(std::move(sentenceSource)) {
        }

      /**
       * Iterate over the word source for numEpochs iterations.
       * This will generate a large number of minibatch indices that will be passed
       * to the sgd trainer in sentence-based sets.
       */
      void multiEpochTrainingOnWordSource(int32_t numEpochs, std::shared_ptr<Context> context);

    private:
      // this should probably be unique to a single thread
      std::unique_ptr<SentenceSource> _sentenceSource;
      // likely shared across threads.
      const neuralnet::Network& _network;
      // likely shared across threads.
      std::shared_ptr<MinibatchProvider> _minibatchProvider;
      // likely shared across threads.
      std::shared_ptr<MinibatchSgdTrainingStrategy> _trainingStrategy;
  };

}

#endif
