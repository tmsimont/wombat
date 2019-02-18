#include "training/sgd/sgd_mini_batch_trainer.h"

namespace wombat {
namespace sgd {
  SGDMiniBatchTrainer::SGDMiniBatchTrainer(
      std::unique_ptr<VectorSelectionStrategy> vectorSelectionStrategy)
  : _hiddenSize(vectorSelectionStrategy->getVectorSize()) {
    int32_t maximumNumberOfInputVectors =
      vectorSelectionStrategy->maximumInputVectorsPerWordWithContext();
    int32_t maximumNumberOfOutputVectors =
      vectorSelectionStrategy->maximumOutputVectorsPerWordWithContext();
    int32_t localInputMatrixSize = maximumNumberOfOutputVectors * _hiddenSize;
    int32_t localOutputMatrixSize = maximumNumberOfInputVectors * _hiddenSize;

    // TODO: ues weights class?
    posix_memalign(
        reinterpret_cast<void **>(&_localInputLayerVectorData),
        64,
        localInputMatrixSize * sizeof(float));

    posix_memalign(
        reinterpret_cast<void **>(&_localInputLayerUpdateVector),
        64,
        localInputMatrixSize * sizeof(float));

    posix_memalign(
        reinterpret_cast<void **>(&_localOutputLayerVectorData),
        64,
        localOutputMatrixSize * sizeof(float));

    posix_memalign(
        reinterpret_cast<void **>(&_localOutputLayerUpdateVector),
        64,
        localOutputMatrixSize * sizeof(float));

    posix_memalign(
        reinterpret_cast<void **>(&_correctionMatrix),
        64,
        maximumNumberOfInputVectors * maximumNumberOfOutputVectors * sizeof(float));
  }

  SGDMiniBatchTrainer::~SGDMiniBatchTrainer() {
    free(_localOutputLayerVectorData);
    free(_localInputLayerVectorData);
    free(_localInputLayerUpdateVector);
    free(_localOutputLayerUpdateVector);
    free(_correctionMatrix);
  }

  void SGDMiniBatchTrainer::train(std::unique_ptr<MiniBatch> miniBatch) {
    // TODO: validate minibatch size
    // copy the minibatch data into the local memory arrays
    _miniBatch.reset(miniBatch.release());

    activateHiddenLayer();
    calculateError();
    calculateCWordsUpdate();
    calculateTWordsUpdate();
    applyCWordsUpdate();
    applyTWordsUpdate();
  }

  void SGDMiniBatchTrainer::activateHiddenLayer() {
    for (int i = 0; i < _miniBatch->getInputLayerVectors().size(); i++) {
      for (int j = 0; j < _miniBatch->getOutputLayerVectors().size(); j++) {
        float f = 0.f, g;
        #pragma simd
        for (int k = 0; k < _hiddenSize; k++) {
          f += _localInputLayerVectorData[i * _hiddenSize + k]
            * _localOutputLayerVectorData[j * _hiddenSize + k];
        }
        int label = _miniBatch->getLabels()[i];
        /*
         * TODO: all this global jazz...
         if (f > MAX_EXP) {
         g = (label - 1) * alpha;
         } else if (f < -MAX_EXP) {
         g = label * alpha;
         } else {
         g = (label - expTable[static_cast<int>(
         (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
         }
         */
        _correctionMatrix[i * _miniBatch->getOutputLayerVectors().size() + j] = g;
      }
    }
  }

  void SGDMiniBatchTrainer::calculateError() {
  }

  void SGDMiniBatchTrainer::calculateCWordsUpdate() {
  }

  void SGDMiniBatchTrainer::calculateTWordsUpdate() {
  }

  void SGDMiniBatchTrainer::applyCWordsUpdate() {
  }

  void SGDMiniBatchTrainer::applyTWordsUpdate() {
  }
}
}
