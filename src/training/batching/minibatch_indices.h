#ifndef TRAINING_BATCHING_MINIBATCH_INDICES_H_
#define TRAINING_BATCHING_MINIBATCH_INDICES_H_

#include "neuralnet/vector.h"

#include <vector>

namespace wombat {
namespace batching {

  /**
   * MinibatchIndices represent a set of selected input and output vector indices we want
   * to use to train a minibatch. This includes the expected labels for the input/output 
   * activation.
   */
  class MinibatchIndices {
    public:
      /**
       * Input vectors are copies from the main network.
       */
      MinibatchIndices(std::vector<int32_t> inputLayerVectorIndices,
                       std::vector<int32_t> labels,
                       std::vector<int32_t> outputLayerVectorIndices)
        : _inputLayerVectors(inputLayerVectorIndices),
          _labels(labels),
          _outputLayerVectors(outputLayerVectorIndices) {}

      const std::vector<int32_t>& getInputLayerVectors() const {
        return _inputLayerVectors;
      }

      // TODO: shouldn't this be 2-dimensional?
      const std::vector<int32_t>& getLabels() const {
        return _labels;
      }

      const std::vector<int32_t>& getOutputLayerVectors() const {
        return _outputLayerVectors;
      }

      const int32_t numInputRows() const {
        return _inputLayerVectors.size();
      }

      const int32_t numOutputCols() const {
        return _outputLayerVectors.size();
      }

    private:
      const std::vector<int32_t> _inputLayerVectors;
      const std::vector<int32_t> _labels;
      const std::vector<int32_t> _outputLayerVectors;
  };
}
}

#endif
