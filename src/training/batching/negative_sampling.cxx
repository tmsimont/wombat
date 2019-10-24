#include "training/batching/negative_sampling.h"

namespace wombat {
namespace batching {

    /**
     * For Skip-gram, we'll grab a set of vectors from the output layer to train.
     * TODO: what about CBOW?
     *
     *   SentenceSource gives us sentences:
     *     * Source is high level buil off network word bag and training source
     *   SentenceParser(Sentence) gives us WordWithContexts
     *     * Parser would have to be created per sentence
     *     * Parser produces limited count of WordWithContext (same as Sentence size)
     *  ------
     *   Minibatch is 1 WordWithContext plus negative samples...
     *   So **something** needs to pass in a WordWithContext that is safe to use?
     * TODO: where do we get the WordWithContext :(
     *
     */
    std::unique_ptr<MinibatchIndices> NegativeSamplingStrategy::getMinibatch(
        const WordWithContext& wordWithContext) {
      std::vector<int32_t> inputs;
      std::vector<int32_t> outputs;
      std::vector<int32_t> labels;

      // Start by adding the current target word to the ouput sample set.
      auto targetWordIndex = wordWithContext.getTargetWord();
      // TODO: some kind of validation between word index and the network wordbag?
      outputs.push_back(targetWordIndex);

      // For the number of negative samples desired, randomly grab random vectors
      // from the output vector set in the network. To do this, we can use a 
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
      
      
      // Next select input vectors based on context words.
      // for ( each context index in WordWithContext )
      //   inputs.push_back( index of word with context from _networkInputVectors )


      // TODO: how many labels?
      for (int i = 0; i < 1; ++i) {
        labels.push_back(i);
      }
      return std::make_unique<MinibatchIndices>(inputs, labels, outputs);
    }

}
}
