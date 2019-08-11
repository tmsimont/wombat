#ifndef TRAINING_SGD_MINIBATCH_H_
#define TRAINING_SGD_MINIBATCH_H_

#include "neuralnet/vector.h"

#include <vector>

namespace wombat {
namespace sgd {

  /**
   * Minibatch is a copy of vectors from a larger neural net into separate structures
   * that can be modified by a single thread without risking false sharing.
   */
  class MiniBatch {
    public:
      /**
       * Input vectors are copies from the main network.
       */
      MiniBatch(std::vector<neuralnet::Vector> inputLayerVectors,
                std::vector<int32_t> labels,
                std::vector<neuralnet::Vector> outputLayerVectors)
        : _inputLayerVectors(inputLayerVectors),
          _labels(labels),
          _outputLayerVectors(outputLayerVectors) {}

      const std::vector<neuralnet::Vector>& getInputLayerVectors() {
        return _inputLayerVectors;
      }

      const std::vector<int32_t>& getLabels() {
        return _labels;
      }

      const std::vector<neuralnet::Vector>& getOutputLayerVectors() {
        return _outputLayerVectors;
      }

      const int32_t numInputRows() {
        return _inputLayerVectors.size();
      }

      const int32_t numOutputCols() {
        return _outputLayerVectors.size();
      }

    private:
      const std::vector<neuralnet::Vector> _inputLayerVectors;
      const std::vector<int32_t> _labels;
      const std::vector<neuralnet::Vector> _outputLayerVectors;
  };
}
}

#endif
