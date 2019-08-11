#include "training/sgd/sgd_mini_batch_trainer.h"

namespace wombat {
namespace sgd {

  /**
   * This implementation uses Vectors from a Minibatch instance to calculate
   * local minibatch network activation, loss and update matrices that can then be
   * applied back to the parent network from which the Minibatch was formed.
   */
  SGDMiniBatchTrainer::SGDMiniBatchTrainer(std::unique_ptr<MinibatchingStrategy> miniBatcher)
    : _miniBatcher(std::move(miniBatcher)),
    _hiddenSize(miniBatcher->getVectorSize()),
    _matrixManager(_hiddenSize) {
    // Each batch might be a slightly different size, but we need to allocate a fixed chunk
    // of memory, so ask the minibatcher what the maximum sizes of each batch will be.
    int32_t maximumNumberOfInputVectors = miniBatcher->maximumInputVectorsPerBatch();
    int32_t maximumNumberOfOutputVectors = miniBatcher->maximumOutputVectorsPerBatch();

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

  SGDMiniBatchTrainer::~SGDMiniBatchTrainer() {
    free(_localInputLayerUpdateMatrix);
    free(_localOutputLayerUpdateMatrix);
    free(_correctionMatrix);
  }

  void SGDMiniBatchTrainer::train(std::shared_ptr<Context> trainingContext) {
    // TODO: validate miniBatch size?
    _miniBatch = _miniBatcher->getMiniBatch();
    _matrixManager.activate(
        _miniBatch,
        _trainingContext,
        _correctionMatrix);
    _matrixManager.calculateError(
        _miniBatch,
        _trainingContext,
        _correctionMatrix);
    _matrixManager.calculateOutputLayerUpdate(
        _miniBatch,
        _trainingContext,
        _correctionMatrix,
        _localOutputLayerUpdateMatrix);
    _matrixManager.calculateInputLayerUpdate(
        _miniBatch,
        _trainingContext,
        _correctionMatrix,
        _localInputLayerUpdateMatrix);
    applyOutputLayersUpdate();
    applyInputLayersUpdate();
  }

  /**
   * This is where the false-sharing starts. We're going to take our minibatch results
   * and write back to the neural network shared with other threads.
   */
  void SGDMiniBatchTrainer::applyOutputLayersUpdate() {
    int32_t colIndex = 0;
    for (auto const& col: _miniBatch->getOutputLayerVectors()) {
      neuralnet::Vector parentVector = _miniBatcher->getParentOutputVector(col);
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
  void SGDMiniBatchTrainer::applyInputLayersUpdate() {
    int32_t rowIndex = 0;
    for (auto const& row: _miniBatch->getInputLayerVectors()) {
      neuralnet::Vector parentVector = _miniBatcher->getParentInputVector(row);
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
