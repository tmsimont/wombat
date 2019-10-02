#ifndef TRAINING_BATCHING_NEGATIVE_SAMPLING_H_
#define TRAINING_BATCHING_NEGATIVE_SAMPLING_H_

#include "neuralnet/network.h"
#include "neuralnet/vector.h"
#include "training/batching/minibatching_strategy.h"
#include "training/data/source/word_with_context.h"

using wombat::data::WordWithContext;

namespace wombat {
namespace batching {

  class NegativeSamplingStrategy : public MinibatchingStrategy {
    public:
      NegativeSamplingStrategy(const neuralnet::Network& network) :
        MinibatchingStrategy(network) {
      }

      virtual ~NegativeSamplingStrategy() {}

      virtual int32_t maximumInputVectorsPerBatch();

      virtual int32_t maximumOutputVectorsPerBatch();

      virtual int32_t getVectorSize();

      virtual std::unique_ptr<Minibatch> getMinibatch(const WordWithContext& wordWithContext);
  };

}
}

#endif
