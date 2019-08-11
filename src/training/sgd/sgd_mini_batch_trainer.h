#ifndef TRAINING_SGD_SGDMINIBATCHTRAINER_H_
#define TRAINING_SGD_SGDMINIBATCHTRAINER_H_

#include "training/sgd/minibatching_strategy.h"
#include "training/sgd/mini_batch.h"
#include "training/context.h"

#include <memory>

namespace wombat {
namespace sgd {
  class SGDMiniBatchTrainer {
    public:
      SGDMiniBatchTrainer(std::unique_ptr<MinibatchingStrategy> miniBatcher);

      ~SGDMiniBatchTrainer();

      void train(std::shared_ptr<Context> trainingContext);

      /**
       * InputRows * OutputColumns^Transpose
       *
       * This performs Wih * Woh for set of input/output pairs to activate 
       * input/output interaction in a minibatch. (Forward propagation)
       */
      virtual void activateHiddenLayer();

      /**
       * error(softmax(above result))
       * This yields a float between -1 and 1 for the
       * adjustment required for each target/context vector
       * TODO: previously this was done during the "activateHiddenLayer" call to avoid a 
       * m x n interation over the _correctionMatrix... I reverted that for readability and consistency
       * with the MKL version... maybe worth investigation into if readability has a big impact on performance.
       */
      virtual void calculateError();

      /**
       * CorrectionMatrix^Transpose * InputRowMatrix
       * The result is numOutputColumns * hiddenSize update matrix that can be used
       * to update "output" values in the neural network.
       */
      virtual void calculateOutputLayerUpdate();

      /**
       * CorrectionMatrix * OutputColMatrix^Transpose
       * The result is numInputRows * hiddenSize update matrix that can be used
       * to update "input" values in the neural network.
       */
      virtual void calculateInputLayerUpdate();

      /**
       * This is where the false-sharing starts. We're going to take our minibatch results
       * and write back to the neural network shared with other threads.
       * TODO: move this out from the trainer and instead return the update matrix itself?
       */
      virtual void applyOutputLayersUpdate();

      /**
       * This is where the false-sharing starts. We're going to take our minibatch results
       * and write back to the neural network shared with other threads.
       * TODO: move this out from the trainer and instead return the update matrix itself?
       */
      virtual void applyInputLayersUpdate();

    private:
      const int32_t _hiddenSize;
      std::unique_ptr<MiniBatch> _miniBatch;
      std::shared_ptr<Context> _trainingContext;
      float * _localInputLayerUpdateMatrix;
      float * _localOutputLayerUpdateMatrix;
      float * _correctionMatrix;
  };
}
}

#endif
