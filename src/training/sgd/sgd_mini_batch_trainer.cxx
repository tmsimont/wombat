#include "training/sgd/sgd_mini_batch_trainer.h"

namespace wombat {
namespace sgd {

  /**
   * This implementation uses Vectors from a Minibatch instance to calculate
   * local minibatch network activation, loss and update matrices that can then be
   * applied back to the parent network from which the Minibatch was formed.
   */
  SGDMiniBatchTrainer::SGDMiniBatchTrainer(std::unique_ptr<MinibatchingStrategy> miniBatcher)
  : _hiddenSize(miniBatcher->getVectorSize()) {
    // Each batch might be a slightly different size, but we need to allocate a fixed chunk
    // of memory, so ask the minibatcher what the maximum sizes of each batch will be.
    int32_t maximumNumberOfInputVectors = miniBatcher->maximumInputVectorsPerWordWithContext();
    int32_t maximumNumberOfOutputVectors = miniBatcher->maximumOutputVectorsPerWordWithContext();

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
    _miniBatch = miniBatcher->getMiniBatch();
    // TODO: validate miniBatch size? 
    activateHiddenLayer();
    calculateError();
    calculateOutputLayerUpdate();
    calculateInputLayerUpdate();
    applyOutputLayersUpdate();
    applyInputLayersUpdate();
  }

  /**
   * Basic un-optimized n^3 matrix multiplication of:
   * InputRows * OutputColumns^Transpose
   */
  void SGDMiniBatchTrainer::activateHiddenLayer() {
    int32_t rowIndex = 0, colIndex = 0;
    for (auto const& row: _miniBatch->getInputLayerVectors()) {
      for (auto const& col: _miniBatch->getOutputLayerVectors()) {
        float f = 0.f;
        // TODO: validate row/col length either here or before here.
        // We're assuming the row/col are both _hiddenSize in length.
        for (int k = 0; k < _hiddenSize; k++) {
          f += row->get(k) * col->get(k);
        }
        int32_t correctionMatrixIndex = rowIndex * _miniBatch->numOutputCols() + colIndex;
        // TODO: think about using something other than "training context?" also is "loss" correct?
        _correctionMatrix[correctionMatrixIndex] = f;
        colIndex++;
      }
      rowIndex++;
    }
  }

  /**
   * TODO: previously this was done during the "activateHiddenLayer" call to avoid a 
   * m x n interation over the _correctionMatrix... I reverted that for readability and consistency
   * with the MKL version... maybe worth investigation into if readability has a big impact on performance.
   */
  void SGDMiniBatchTrainer::calculateError() {
    for (int32_t rowIndex = 0; rowIndex < _miniBatch->numInputRows(); rowIndex++) {
      for (int32_t colIndex = 0; colIndex < _miniBatch->numOutputCols(); colIndex++) {
        int32_t correctionMatrixIndex = rowIndex * _miniBatch->numOutputCols() + colIndex;
        // TODO: think about using something other than "training context?" also is "loss" correct?
        _correctionMatrix[correctionMatrixIndex] = _trainingContext
          ->loss(_correctionMatrix[correctionMatrixIndex], _miniBatch->getLabels()->at(rowIndex));
      }
    }
  }

  /**
   * Basic unoptimized matrix multiply.
   */
  void SGDMiniBatchTrainer::calculateOutputLayerUpdate() {
    for (int32_t colIndex = 0; colIndex < _miniBatch->getOutputLayerVectors().size(); colIndex++) {
      for (int j = 0; j < _hiddenSize; j++) {
        float f = 0.f;
        int32_t rowIndex = 0;
        for (auto const& row: _miniBatch->getInputLayerVectors()) {
          f += _correctionMatrix[rowIndex * _miniBatch->numOutputCols() + colIndex] * row.get(j);
          rowIndex++;
        }
        _localOutputLayerUpdateMatrix[colIndex * _hiddenSize + j] = f;
      }
    }
  }

  /**
   * Basic unoptimized matrix multiply.
   */
  void SGDMiniBatchTrainer::calculateInputLayerUpdate() {
    for (int32_t rowIndex = 0; rowIndex < _miniBatch->getInputLayerVectors().size(); rowIndex++) {
      for (int32_t j = 0; j < _hiddenSize; j++) {
        float f = 0.f;
        int32_t colIndex = 0;
        for (auto const& col: _miniBatch->getOutputLayerVectors()) {
          f += _correctionMatrix[rowIndex * _miniBatch->numOutputCols() + colIndex] * col.get(j);
          colIndex++;
        }
        _localInputLayerUpdateMatrix[rowIndex * _hiddenSize + j] = f;
      }
    }
  }

  /**
   * This is where the false-sharing starts. We're going to take our minibatch results
   * and write back to the neural network shared with other threads.
   * TODO: move this out from the trainer and instead return the update matrix itself?
   */
  void SGDMiniBatchTrainer::applyOutputLayersUpdate() {
    int32_t colIndex = 0;
    for (auto const& col: _miniBatch->getOutputLayerVectors()) {
      for (int k = 0; k < _hiddenSize; k++) {
        // TODO: obtain some reference back to the original neural net instance
        // col->getIndex() here is the index of the Vector in the "global" neural network
        // Wih[col->getIndex() * _hiddenSize + k] += _localOutputLayerUpdateMatrix[colIndex * _hiddenSize + k];
      }
      colIndex++;
    }
  }

  void SGDMiniBatchTrainer::applyInputLayersUpdate() {
    int32_t rowIndex = 0;
    for (auto const& row: _miniBatch->getInputLayerVectors()) {
      for (int k = 0; k < _hiddenSize; k++) {
        // TODO: obtain some reference back to the original neural net instance
        // col->getIndex() here is the index of the Vector in the "global" neural network
        // Woh[col->getIndex() * _hiddenSize + k] += _localInputLayerUpdateMatrix[rowIndex * _hiddenSize + k];
      }
      rowIndex++;
    }
  }
}
}
