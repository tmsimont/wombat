#ifndef TRAINING_BATCHING_MINIBATCH_H_
#define TRAINING_BATCHING_MINIBATCH_H_

#include "neuralnet/network.h"
#include "neuralnet/vector.h"

#include <vector>

namespace wombat {
namespace batching {

  /**
   * Minibatch is a copy of vectors from a larger neural net into separate structures
   * that can be modified by a single thread without risking false sharing.
   */
  class Minibatch {
    public:
      /**
       * Input vectors are copies from the main network.
       */
      Minibatch(const neuralnet::Network& network,
                std::vector<neuralnet::Vector> inputLayerVectors,
                std::vector<int32_t> labels,
                std::vector<neuralnet::Vector> outputLayerVectors)
        : _network(network),
          _inputLayerVectors(inputLayerVectors),
          _labels(labels),
          _outputLayerVectors(outputLayerVectors) {}

      const std::vector<neuralnet::Vector>& getInputLayerVectors() {
        return _inputLayerVectors;
      }

      // TODO: shouldn't this be 2-dimensional?
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

    private:
      const neuralnet::Network& _network;
      const std::vector<neuralnet::Vector> _inputLayerVectors;
      const std::vector<int32_t> _labels;
      const std::vector<neuralnet::Vector> _outputLayerVectors;
  };
}
}

#endif
