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

    // Space for thread-local input layer vectors
    posix_memalign(
        reinterpret_cast<void **>(&_localInputLayerBatchMemory),
        64,
        maximumNumberOfInputVectors * _hiddenSize * sizeof(float));

    // Space for thread-local output layer vectors
    posix_memalign(
        reinterpret_cast<void **>(&_localOutputLayerBatchMemory),
        64,
        maximumNumberOfOutputVectors * _hiddenSize * sizeof(float));

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
    free(_localInputLayerBatchMemory);
    free(_localOutputLayerBatchMemory);
    free(_localInputLayerUpdateMatrix);
    free(_localOutputLayerUpdateMatrix);
    free(_correctionMatrix);
  }

  void SGDMinibatchTrainer::train(
      const neuralnet::Network& network,
      std::vector<std::unique_ptr<MinibatchIndices>> minibatchIndicesList,
      std::shared_ptr<Context> trainingContext) {
    // Iterate over all minibatch indices sequentially.
    // TODO: does it make sense to do a bunch of minibatches like this for CPU?
    // (maybe the matrix manager could be some multi-threaded task pool or something...)
    while (!minibatchIndicesList.empty()) {
      // TODO: reference last and pop in one move?
      auto indices = std::move(minibatchIndicesList.at(minibatchIndicesList.size() - 1));
      minibatchIndicesList.pop_back();

      // TODO: validate minibatch size against local memory size params

      // Build a thread-local minibatch
      // This is where we will probably hit false sharing.
      Minibatch minibatch = Minibatch::fromNetworkIndices(
          network,
          indices,
          _localInputLayerBatchMemory,
          _localOutputLayerBatchMemory);

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

      // Write back to global memory.
      // This is where we will probably hit false sharing.
      applyOutputLayersUpdate(minibatch);
      applyInputLayersUpdate(minibatch);
    }
  }

  /**
   * This is where the false-sharing starts. We're going to take our minibatch results
   * and write back to the neural network shared with other threads.
   */
  void SGDMinibatchTrainer::applyOutputLayersUpdate(const Minibatch& minibatch) {
    int32_t colIndex = 0;
    for (auto const& col: minibatch.getOutputLayerVectors()) {
      neuralnet::Vector parentVector = minibatch.getParentOutputVector(col);
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
  void SGDMinibatchTrainer::applyInputLayersUpdate(const Minibatch& minibatch) {
    int32_t rowIndex = 0;
    for (auto const& row: minibatch.getInputLayerVectors()) {
      neuralnet::Vector parentVector = minibatch.getParentInputVector(row);
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
