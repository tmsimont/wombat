#ifndef TRAINING_BATCHING_MINIBATCH_MATRIX_MANAGER_H_
#define TRAINING_BATCHING_MINIBATCH_MATRIX_MANAGER_H_

#include "training/sgd/context.h"
#include "training/batching/minibatching_strategy.h"
#include "training/batching/minibatch.h"

#include <memory>

using wombat::batching::Minibatch;
using wombat::batching::MinibatchingStrategy;

namespace wombat{
namespace sgd {

  /**
   * This class encapsulates the matrix operations required for feed forward and back
   * propagation in the minibatch network operations.
   */
  class MinibatchMatrixManager {
    public:
      MinibatchMatrixManager(int32_t hiddenSize)
        : _hiddenSize(hiddenSize) { }

      /**
       * InputRows * OutputColumns^Transpose
       *
       * This performs Wih * Woh for set of input/output pairs to activate
       * input/output interaction in a minibatch. (Forward propagation)
       */
      virtual void activate(
          const std::unique_ptr<Minibatch>& _minibatch,
          const std::shared_ptr<Context>& _trainingContext,
          float * _correctionMatrix);

      /**
       * error(softmax(above result))
       * This yields a float between -1 and 1 for the
       * adjustment required for each target/context vector
       * TODO: previously this was done during the "activateHiddenLayer" call to avoid a
       * m x n interation over the _correctionMatrix... I reverted that for readability and consistency
       * with the MKL version... maybe worth investigation into if readability has a big impact on performance.
       */
      virtual void calculateError(
          const std::unique_ptr<Minibatch>& _minibatch,
          const std::shared_ptr<Context>& _trainingContext,
          float * _correctionMatrix);

      /**
       * CorrectionMatrix^Transpose * InputRowMatrix
       * The result is numOutputColumns * hiddenSize update matrix that can be used
       * to update "output" values in the neural network.
       */
      virtual void calculateOutputLayerUpdate(
          const std::unique_ptr<Minibatch>& _minibatch,
          const std::shared_ptr<Context>& _trainingContext,
          float * _correctionMatrix,
          float * _localOutputLayerUpdateMatrix);

      /**
       * CorrectionMatrix * OutputColMatrix^Transpose
       * The result is numInputRows * hiddenSize update matrix that can be used
       * to update "input" values in the neural network.
       */
      virtual void calculateInputLayerUpdate(
          const std::unique_ptr<Minibatch>& _minibatch,
          const std::shared_ptr<Context>& _trainingContext,
          float * _correctionMatrix,
          float * _localInputLayerUpdateMatrix);

    private:
      const int32_t _hiddenSize;
  };
}
}

#endif
