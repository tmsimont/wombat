#ifndef TRAINING_SGD_VECTOR_SELECTION_STRATEGY_H_
#define TRAINING_SGD_VECTOR_SELECTION_STRATEGY_H_

#include "training/data/structure/word_with_context.h"
#include "neuralnet/network.h"
#include "neuralnet/vector.h"

#include <memory>
#include <vector>

namespace wombat {
namespace sgd {

  /**
   * Strategy used for pulling Vectors out of the neural net for a given word with context.
   */
  class VectorSelectionStrategy {
    public:
      VectorSelectionStrategy(const neuralnet::Network& network) {}

      virtual ~VectorSelectionStrategy() {}

      virtual int32_t maximumInputVectorsPerWordWithContext() = 0;

      virtual int32_t maximumOutputVectorsPerWordWithContext() = 0;

      virtual int32_t getVectorSize() = 0;

      virtual std::vector<neuralnet::Vector> getInputVectors(
          std::shared_ptr<WordWithContext> wordWithContext) = 0;

      virtual std::vector<int32_t> getLabels(
          std::shared_ptr<WordWithContext> wordWithContext) = 0;

      virtual std::vector<neuralnet::Vector> getOutputVectors(
          std::shared_ptr<WordWithContext> wordWithContext) = 0;
  };
}
}

#endif
