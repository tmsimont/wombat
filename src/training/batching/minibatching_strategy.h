#ifndef TRAINING_BATCHING_MINIBATCHING_STRATEGY_H_
#define TRAINING_BATCHING_MINIBATCHING_STRATEGY_H_

#include "training/data/structure/word_with_context.h"
#include "training/batching/minibatch.h"
#include "neuralnet/network.h"
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
  class MinibatchingStrategy {
    public:
      MinibatchingStrategy(const neuralnet::Network& network)
        : _network(network) {}

      virtual ~MinibatchingStrategy() {}

      virtual int32_t maximumInputVectorsPerBatch() = 0;

      virtual int32_t maximumOutputVectorsPerBatch() = 0;

      virtual int32_t getVectorSize() = 0;

      virtual std::unique_ptr<Minibatch> getMinibatch(const WordWithContext& wordWithContext) = 0;

      virtual neuralnet::Vector getParentInputVector(const neuralnet::Vector& minibatchVector) {
        // The strategy should maintain the index when copying into the minibatch.
        int32_t parentIndex = minibatchVector.getIndex();
        return _network.getInputVector(parentIndex);
      }

      virtual neuralnet::Vector getParentOutputVector(const neuralnet::Vector& minibatchVector) {
        // The strategy should maintain the index when copying into the minibatch.
        int32_t parentIndex = minibatchVector.getIndex();
        return _network.getOutputVector(parentIndex);
      }

    protected:
      const neuralnet::Network& _network;
  };
}
}

#endif
