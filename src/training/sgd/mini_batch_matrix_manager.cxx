#include "training/sgd/mini_batch_matrix_manager.h"

namespace wombat {
namespace sgd {

  /**
   * Basic un-optimized n^3 matrix multiplication.
   */
  void MiniBatchMatrixManager::activate(
          const std::unique_ptr<MiniBatch>& _miniBatch,
          const std::shared_ptr<Context>& _trainingContext,
          float * _correctionMatrix) {
    int32_t rowIndex = 0, colIndex = 0;
    for (auto const& row: _miniBatch->getInputLayerVectors()) {
      for (auto const& col: _miniBatch->getOutputLayerVectors()) {
        float f = 0.f;
        // TODO: validate row/col length either here or before here.
        // We're assuming the row/col are both _hiddenSize in length.
        for (int k = 0; k < _hiddenSize; k++) {
          f += row.get(k) * col.get(k);
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
  void MiniBatchMatrixManager::calculateError(
          const std::unique_ptr<MiniBatch>& _miniBatch,
          const std::shared_ptr<Context>& _trainingContext,
          float * _correctionMatrix) {
    for (int32_t rowIndex = 0; rowIndex < _miniBatch->numInputRows(); rowIndex++) {
      for (int32_t colIndex = 0; colIndex < _miniBatch->numOutputCols(); colIndex++) {
        int32_t correctionMatrixIndex = rowIndex * _miniBatch->numOutputCols() + colIndex;
        // TODO: think about using something other than "training context?" also is "loss" correct?
        _correctionMatrix[correctionMatrixIndex] = _trainingContext
          ->loss(_correctionMatrix[correctionMatrixIndex], _miniBatch->getLabels().at(rowIndex));
      }
    }
  }

  /**
   * Basic unoptimized matrix multiply.
   */
  void MiniBatchMatrixManager::calculateOutputLayerUpdate(
      const std::unique_ptr<MiniBatch>& _miniBatch,
      const std::shared_ptr<Context>& _trainingContext,
      float * _correctionMatrix,
      float * _localOutputLayerUpdateMatrix) {
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
  void MiniBatchMatrixManager::calculateInputLayerUpdate(
      const std::unique_ptr<MiniBatch>& _miniBatch,
      const std::shared_ptr<Context>& _trainingContext,
      float * _correctionMatrix,
      float * _localInputLayerUpdateMatrix) {
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
}
}
