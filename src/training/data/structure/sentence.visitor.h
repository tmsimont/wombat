#ifndef TRAINING_DATA_STRUCTURE_SENTENCE_VISITOR_H_
#define TRAINING_DATA_STRUCTURE_SENTENCE_VISITOR_H_

#include <stdint.h>

namespace wombat {
  class SentenceVisitor {
    public:
      /**
       * Will be called in order for each word in the sentence.
       */
      virtual void visitWord(const int32_t& wordIndex) = 0;
  };
}

#endif
