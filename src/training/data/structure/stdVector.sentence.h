#ifndef TRAINING_DATA_STRUCTURE_STD_VECTOR_SENTENCE_H_
#define TRAINING_DATA_STRUCTURE_STD_VECTOR_SENTENCE_H_

#include "training/data/structure/sentence.h"

#include <vector>

namespace wombat {

  /**
   * A simple implementation of Sentence that uses a std::vector to store the
   * in-order word indices.
   */
  class StdVectorSentence : public Sentence {
    public:
      int32_t getNumberOfTrainingWords();
      int32_t getNumberOfWordsInput();
      void acceptWordVisitor(SentenceVisitor& visitor) const;

      // TODO: use a builder or class friend to prevent mutability at training time.
      void addWord(int wordIndex);
      // TODO: use a builder or class friend to prevent mutability at training time.
      void countDiscardedWord();
    private:
      std::vector<int32_t> _sentence;
      int32_t _discarded = 0;
  };
}

#endif
