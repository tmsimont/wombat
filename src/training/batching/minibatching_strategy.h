#ifndef TRAINING_BATCHING_MINIBATCHING_STRATEGY_H_
#define TRAINING_BATCHING_MINIBATCHING_STRATEGY_H_

#include "training/data/structure/word_with_context.h"
#include "training/batching/minibatch_indices.h"
#include "neuralnet/vector.h"

#include <memory>
#include <vector>

namespace wombat {
namespace batching {

  /**
   * Strategy used for pulling Vectors out of the neural net for a given word with context, and then
   * labeling the vectors. The subset of vectors and their labels are called a "mini batch"
   *
   * TODO: feed in a SentenceSource and SentenceParser to read and parse sentences into WordWithContext.
   * TODO: use ContiguousWordWithContextBuffer? maybe feed that in instead of source/parser
   * TODO: how do u split sources and parsers across multiple threads :(
   */
  // Threadsafe
  class MinibatchingStrategy {
    public:
      MinibatchingStrategy() {}

      virtual ~MinibatchingStrategy() {}

      virtual std::unique_ptr<MinibatchIndices> getMinibatch(const WordWithContext& wordWithContext) = 0;
  };
}
}

#endif
