#ifndef TRAINING_SGD_MINIBATCHING_STRATEGY_H_
#define TRAINING_SGD_MINIBATCHING_STRATEGY_H_

#include "training/data/structure/word_with_context.h"
#include "neuralnet/network.h"
#include "neuralnet/vector.h"

#include <memory>
#include <vector>

namespace wombat {
namespace sgd {

  /**
   * Strategy used for pulling Vectors out of the neural net for a given word with context, and then
   * labeling the vectors. The subset of vectors and their labels are called a "mini batch"
   */
  class MinibatchingStrategy {
    public:
      MinibatchingStrategy(const neuralnet::Network& network) {}

      virtual ~MinibatchingStrategy() {}

      virtual int32_t maximumInputVectorsPerWordWithContext() = 0;

      virtual int32_t maximumOutputVectorsPerWordWithContext() = 0;

      virtual int32_t getVectorSize() = 0;

      virtual std::unique_ptr<MiniBatch> getMiniBatch() = 0;
  };
}
}

#endif
