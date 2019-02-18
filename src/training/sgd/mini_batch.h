#ifndef TRAINING_SGD_MINIBATCH_H_
#define TRAINING_SGD_MINIBATCH_H_

#include "neuralnet/vector.h"

#include <vector>

namespace wombat {
namespace sgd {
  class MiniBatch {
    public:
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

    private:
      const std::vector<neuralnet::Vector> _inputLayerVectors;
      const std::vector<int32_t> _labels;
      const std::vector<neuralnet::Vector> _outputLayerVectors;
  };
}
}

#endif
