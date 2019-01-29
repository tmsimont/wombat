#ifndef TRAINING_DATA_STRUCTURE_WORD_WITH_CONTEXT_H_
#define TRAINING_DATA_STRUCTURE_WORD_WITH_CONTEXT_H_

#include "training/data/structure/word_with_context.visitor.h"

#include <stdint.h>

namespace wombat {

  /**
   * This is a word pulled from our training data with the surrounding context
   * words we found in a sentence.
   */
  class WordWithContext {
    public:
      virtual int32_t getTargetWord() = 0;
      // TODO: would probably be better to use an iterator. 
      // visitor pattern is less code than iterator
      virtual void acceptContextWordVisitor(WordWithContextVisitor& visitor) const = 0;
  };

}

#endif
