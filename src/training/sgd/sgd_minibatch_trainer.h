#ifndef TRAINING_SGD_SGDMINIBATCHTRAINER_H_
#define TRAINING_SGD_SGDMINIBATCHTRAINER_H_

#include "training/batching/minibatch.h"
#include "training/batching/minibatch_indices.h"
#include "training/batching/minibatching_strategy.h"
#include "training/sgd/context.h"
#include "training/sgd/minibatch_matrix_manager.h"
#include "training/minibatch_sgd_training_strategy.h"

#include <memory>

using wombat::batching::Minibatch;
using wombat::batching::MinibatchingStrategy;

namespace wombat {
namespace sgd {
  class SGDMinibatchTrainer : public ::wombat::MinibatchSgdTrainingStrategy {
    public:
      SGDMinibatchTrainer(int32_t hiddenSize,
                          int32_t maximumNumberOfInputVectors,
                          int32_t maximumNumberOfOutputVectors);

      ~SGDMinibatchTrainer();

      virtual void train(
          const neuralnet::Network& network,
          std::vector<std::unique_ptr<MinibatchIndices>> minibatchIndicesList,
          std::shared_ptr<Context> trainingContext);

      /**
       * Update the "global" output layer (i.e. not the minibatch but the actual neural net).
       */
      virtual void applyOutputLayersUpdate(const Minibatch& minibatch);

      /**
       * Update the "global" input layer (i.e. not the minibatch but the actual neural net).
       */
      virtual void applyInputLayersUpdate(const Minibatch& minibatch);

    private:
      const int32_t _hiddenSize;
      const int32_t _maximumNumberOfInputVectors;
      const int32_t _maximumNumberOfOutputVectors;
      std::unique_ptr<MinibatchingStrategy> _minibatcher;
      std::shared_ptr<Context> _trainingContext;
      MinibatchMatrixManager _matrixManager;
      float * _localInputLayerBatchMemory;
      float * _localOutputLayerBatchMemory;
      float * _localInputLayerUpdateMatrix;
      float * _localOutputLayerUpdateMatrix;
      float * _correctionMatrix;
  };
}
}

#endif
