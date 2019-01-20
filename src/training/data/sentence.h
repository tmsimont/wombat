#ifndef TRAINING_DATA_SENTENCE_H_
#define TRAINING_DATA_SENTENCE_H_

#include "training/data/sentence.visitor.h"

#include <stdint.h>

namespace wombat {

  /**
   * A sentence is a fixed-length sequence of word indices.
   * Each word index references a word in the training vocabulary.
   * Each sentence in the training set gives us a set of in-order word usages 
   * to train from. Words found in sentences are randomly sampled, so the
   * Sentence structure also lets us keep a count of how many words we did not
   * include in our sampling.
   *
   * The sentence accepts visitors to see the indices of words in the order
   * they were found in training data.
   */
  class Sentence {
    public:
      /**
       * Get the number of words that can be used for training.
       * This is equal to the number of words iterated in #acceptWordVisitor().
       */
      virtual int32_t getNumberOfTrainingWords() = 0;

      /**
       * Get the number of words of the original input sentence. Some of the
       * words might be discared (not included in random sampling).
       */
      virtual int32_t getNumberOfWordsInput() = 0;

      // TODO: would probably be better to use an iterator. 
      // visitor pattern is less code than iterator
      virtual void acceptWordVisitor(SentenceVisitor& visitor) const = 0;
  };
}

#endif
