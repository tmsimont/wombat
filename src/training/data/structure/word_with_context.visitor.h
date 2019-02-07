#ifndef TRAINING_DATA_STRUCTURE_WORD_WITH_CONTEXT_VISITOR_H_
#define TRAINING_DATA_STRUCTURE_WORD_WITH_CONTEXT_VISITOR_H_

#include <stdint.h>

namespace wombat {
  class WordWithContextVisitor {
    public:
      virtual ~WordWithContextVisitor() {}

      /**
       * Will be called in order for each word in the set of context words.
       */
      virtual void visitContextWord(const int32_t& wordIndex) = 0;
  };
}

#endif

