#ifndef TRAINING_SGD_SGDMINIBATCHTRAINER_H_
#define TRAINING_SGD_SGDMINIBATCHTRAINER_H_

#include "training/sgd/vector_selection_strategy.h"
#include "training/sgd/mini_batch.h"

#include <memory>

namespace wombat {
namespace sgd {
  class SGDMiniBatchTrainer {
    public:
      SGDMiniBatchTrainer(std::unique_ptr<VectorSelectionStrategy> vectorSelectionStrategy);

      ~SGDMiniBatchTrainer();

      void train(std::unique_ptr<MiniBatch> miniBatch);

      virtual void activateHiddenLayer();
      virtual void calculateError();
      virtual void calculateCWordsUpdate();
      virtual void calculateTWordsUpdate();
      virtual void applyCWordsUpdate();
      virtual void applyTWordsUpdate();

    private:
      const int32_t _hiddenSize;
      std::unique_ptr<MiniBatch> _miniBatch;
      float * _localInputLayerVectorData;
      float * _localInputLayerUpdateVector;
      float * _localOutputLayerVectorData;
      float * _localOutputLayerUpdateVector;
      float * _correctionMatrix;
  };
}
}

#endif
