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
      virtual ~WordWithContext() {}

      /**
       * Return the target word that has context words around it.
       */
      virtual int32_t getTargetWord() const = 0;

      /**
       * Return the number of context words have we sampled for this training target word.
       */
      virtual int32_t getNumberOfContextWords() const = 0;

      // TODO: would probably be better to use an iterator. 
      // visitor pattern is less code than iterator, but makes for bad patterns
      virtual void acceptContextWordVisitor(WordWithContextVisitor& visitor) const = 0;
  };

}

#endif
