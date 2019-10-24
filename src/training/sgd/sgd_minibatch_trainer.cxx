#include "training/sgd/sgd_minibatch_trainer.h"

namespace wombat {
namespace sgd {

  /**
   * This implementation uses Vectors from a Minibatch instance to calculate
   * local minibatch network activation, loss and update matrices that can then be
   * applied back to the parent network from which the Minibatch was formed.
   * Each batch might be a slightly different size, but we need to allocate a fixed chunk
   * of memory, so ask the minibatcher what the maximum sizes of each batch will be.
   */
  SGDMinibatchTrainer::SGDMinibatchTrainer(int32_t hiddenSize,
                                           int32_t maximumNumberOfInputVectors,
                                           int32_t maximumNumberOfOutputVectors)
    : _hiddenSize(hiddenSize),
      _maximumNumberOfInputVectors(maximumNumberOfInputVectors),
      _maximumNumberOfOutputVectors(maximumNumberOfOutputVectors),
      _matrixManager(hiddenSize) {

    // Space for thread-local input layer update vector
    posix_memalign(
        reinterpret_cast<void **>(&_localInputLayerUpdateMatrix),
        64,
        maximumNumberOfInputVectors * _hiddenSize * sizeof(float));

    // Space for thread-local output layer update vector
    posix_memalign(
        reinterpret_cast<void **>(&_localOutputLayerUpdateMatrix),
        64,
        maximumNumberOfOutputVectors * _hiddenSize * sizeof(float));

    // Space for our thread-local correction matrix.
    posix_memalign(
        reinterpret_cast<void **>(&_correctionMatrix),
        64,
        maximumNumberOfInputVectors * maximumNumberOfOutputVectors * sizeof(float));
  }

  SGDMinibatchTrainer::~SGDMinibatchTrainer() {
    free(_localInputLayerUpdateMatrix);
    free(_localOutputLayerUpdateMatrix);
    free(_correctionMatrix);
  }

  void SGDMinibatchTrainer::train(
      std::unique_ptr<Minibatch> minibatch,
      std::shared_ptr<Context> trainingContext) {
    // TODO: validate minibatch size against local memory size params
    _matrixManager.activate(
        minibatch,
        _trainingContext,
        _correctionMatrix);
    _matrixManager.calculateError(
        minibatch,
        _trainingContext,
        _correctionMatrix);
    _matrixManager.calculateOutputLayerUpdate(
        minibatch,
        _trainingContext,
        _correctionMatrix,
        _localOutputLayerUpdateMatrix);
    _matrixManager.calculateInputLayerUpdate(
        minibatch,
        _trainingContext,
        _correctionMatrix,
        _localInputLayerUpdateMatrix);
    applyOutputLayersUpdate(minibatch);
    applyInputLayersUpdate(minibatch);
  }

  /**
   * This is where the false-sharing starts. We're going to take our minibatch results
   * and write back to the neural network shared with other threads.
   */
  void SGDMinibatchTrainer::applyOutputLayersUpdate(const std::unique_ptr<Minibatch>& minibatch) {
    int32_t colIndex = 0;
    for (auto const& col: minibatch->getOutputLayerVectors()) {
      neuralnet::Vector parentVector = minibatch->getParentOutputVector(col);
      for (int k = 0; k < _hiddenSize; k++) {
        float currentValue = parentVector.get(k);
        parentVector.update(
            k,
            currentValue + _localOutputLayerUpdateMatrix[colIndex * _hiddenSize + k]);
      }
      colIndex++;
    }
  }

  /**
   * This is where the false-sharing starts. We're going to take our minibatch results
   * and write back to the neural network shared with other threads.
   */
  void SGDMinibatchTrainer::applyInputLayersUpdate(const std::unique_ptr<Minibatch>& minibatch) {
    int32_t rowIndex = 0;
    for (auto const& row: minibatch->getInputLayerVectors()) {
      neuralnet::Vector parentVector = minibatch->getParentInputVector(row);
      for (int k = 0; k < _hiddenSize; k++) {
        float currentValue = parentVector.get(k);
        parentVector.update(
            k,
            currentValue + _localInputLayerUpdateMatrix[rowIndex * _hiddenSize + k]);
      }
      rowIndex++;
    }
  }
}
}
