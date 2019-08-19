#include "training/batching/negative_sampling.h"

namespace wombat {
namespace batching {

    int32_t NegativeSamplingStrategy::maximumInputVectorsPerBatch() {
      return 1;
    }

    int32_t NegativeSamplingStrategy::maximumOutputVectorsPerBatch() {
      return 1;
    }

    int32_t NegativeSamplingStrategy::getVectorSize() {
      return 4;
    }

    std::unique_ptr<Minibatch> NegativeSamplingStrategy::getMinibatch() {
      std::vector<neuralnet::Vector> inputs;
      std::vector<neuralnet::Vector> outputs;
      std::vector<int32_t> labels;

      // For the number of negative samples desired, randomly grab random vectors
      // from the input vector set in the network. To do this, we can use a 
      // random sampling strategy.

      // for (int k = 0; k < negative; k++) {
        // int sample;
        // old way of doing word-frequency-based sampling:
        // if (randomness) {
        //   next_random = next_random * (unsigned long long) 25214903917 + 11;
        //   sample = table[(next_random >> 16) % table_size];
        //   if (!sample) {
        //     sample = next_random % (vocab_size - 1) + 1;
        //   }
        
        // old way of doing random sampling without frequency
        // } else {
        //   next_random = (next_random + 20) % vocab_size;
        //   sample = next_random;
        // }
        // outputs.push_back (vector at sample)
        // labels.push_back(0);
      // }


      inputs.push_back(_network.getInputVector(0));
      outputs.push_back(_network.getOutputVector(0));
      for (int i = 0; i < getVectorSize(); ++i) {
        labels.push_back(i);
      }
      return std::make_unique<Minibatch>(inputs, labels, outputs);;
    }

}
}
