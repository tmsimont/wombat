#ifndef TRAINING_SGD_SGDMINIBATCHTRAINER_H_
#define TRAINING_SGD_SGDMINIBATCHTRAINER_H_

#include "training/batching/minibatch.h"
#include "training/batching/minibatching_strategy.h"
#include "training/sgd/context.h"
#include "training/sgd/minibatch_matrix_manager.h"

#include <memory>

using wombat::batching::Minibatch;
using wombat::batching::MinibatchingStrategy;

namespace wombat {
namespace sgd {
  class SGDMinibatchTrainer {
    public:
      SGDMinibatchTrainer(std::unique_ptr<MinibatchingStrategy> minibatcher);

      ~SGDMinibatchTrainer();

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
      std::unique_ptr<MinibatchingStrategy> _minibatcher;
      std::unique_ptr<Minibatch> _minibatch;
      std::shared_ptr<Context> _trainingContext;
      MinibatchMatrixManager _matrixManager;
      float * _localInputLayerUpdateMatrix;
      float * _localOutputLayerUpdateMatrix;
      float * _correctionMatrix;
  };
}
}

#endif
