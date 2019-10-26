#ifndef TRAINING_BATCHING_MINIBATCH_PROVIDER_H_
#define TRAINING_BATCHING_MINIBATCH_PROVIDER_H_

#include "training/batching/minibatch_indices.h"
#include "training/batching/minibatching_strategy.h"
#include "training/batching/negative_sampling.h"
#include "training/data/parsing/sentence_parser.h"
#include "training/data/structure/sentence.h"

#include <memory>
#include <vector>

namespace wombat {
namespace batching {

  /**
   * Minibatching Provider maps a Sentence to a collection of MinibatchIndices instances using a 
   * specific minibatching strategy.
   */
  // Threadsafe
  class MinibatchProvider {
    public:
      MinibatchProvider(std::unique_ptr<MinibatchingStrategy> minibatchingStrategy) 
        : _minibatchingStrategy(std::move(minibatchingStrategy)) {};

      /**
       * Given a setence, create a vector of MinibatchIndices instances.
       */
      std::vector<std::unique_ptr<MinibatchIndices>> provideMinibatchIndices(std::unique_ptr<Sentence> sentence);

    private:
      // TODO: pass in window size through args or something abstract
      const int32_t _windowSize = 8;

      std::unique_ptr<MinibatchingStrategy> _minibatchingStrategy;
  };
}
}

#endif
