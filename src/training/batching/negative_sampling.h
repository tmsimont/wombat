#ifndef TRAINING_BATCHING_NEGATIVE_SAMPLING_H_
#define TRAINING_BATCHING_NEGATIVE_SAMPLING_H_

#include "neuralnet/vector.h"
#include "training/batching/minibatching_strategy.h"
#include "training/data/structure/word_with_context.h"

namespace wombat {
namespace batching {

  class NegativeSamplingStrategy : public MinibatchingStrategy {
    public:
      NegativeSamplingStrategy() {}

      virtual ~NegativeSamplingStrategy() {}

      virtual std::unique_ptr<MinibatchIndices> getMinibatch(const WordWithContext& wordWithContext);

    private:
      const int32_t _negativeSamples = 5; // TODO: pass this in
  };

}
}

#endif
