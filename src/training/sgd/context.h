#ifndef TRAINING_SGD_CONTEXT_H_
#define TRAINING_SGD_CONTEXT_H_

#include <cmath>
#include <cstdint>
#include "stdlib.h"

// TODO: move exp stuff somewhere else??
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6


namespace wombat {
namespace sgd {
  /**
   * Training context encapsulates the current state of the training, such as
   * remaining epochs, alpha, etc.
   */
  class Context {
    private:
      float * expTable;

    public:
      Context() {
        expTable = reinterpret_cast<float *>(malloc((EXP_TABLE_SIZE + 1) * sizeof(float)));
        for (int i = 0; i < EXP_TABLE_SIZE + 1; i++) {
          expTable[i] = exp((i / (float) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
          expTable[i] = expTable[i] / (expTable[i] + 1);
        }
      }

      float getAlpha() {
        return 0.25f;
      }

      /**
       * Calculate the loss for a labeled inference? 
       * TODO: update what all this means, probably move it out from context into something else?
       */
      float loss(const float& f, int32_t label) {
        float alpha = getAlpha();
        if (f > MAX_EXP) {
          return (label - 1) * alpha;
        } else if (f < -MAX_EXP) {
          return label * alpha;
        } else {
          return (label - expTable[static_cast<int32_t>(
                (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        }
      }
  };
}
}

#endif
