#ifndef TRAINING_SGD_SGDMINIBATCHTRAINER_H_
#define TRAINING_SGD_SGDMINIBATCHTRAINER_H_

#include "training/sgd/context.h"
#include "training/sgd/minibatching_strategy.h"
#include "training/sgd/mini_batch_matrix_manager.h"
#include "training/sgd/mini_batch.h"

#include <memory>

namespace wombat {
namespace sgd {
  class SGDMiniBatchTrainer {
    public:
      SGDMiniBatchTrainer(std::unique_ptr<MinibatchingStrategy> miniBatcher);

      ~SGDMiniBatchTrainer();

      void train(std::shared_ptr<Context> trainingContext);

      /**
       * Update the "global" output layer (i.e. not the minibatch but the actual neural net).
       */
      virtual void applyOutputLayersUpdate();

      /**
       * Update the "global" input layer (i.e. not the minibatch but the actual neural net).
       */
      virtual void applyInputLayersUpdate();

    private:
      const int32_t _hiddenSize;
      std::unique_ptr<MinibatchingStrategy> _miniBatcher;
      std::unique_ptr<MiniBatch> _miniBatch;
      std::shared_ptr<Context> _trainingContext;
      MiniBatchMatrixManager _matrixManager;
      float * _localInputLayerUpdateMatrix;
      float * _localOutputLayerUpdateMatrix;
      float * _correctionMatrix;
  };
}
}

#endif
