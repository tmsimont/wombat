#ifndef TRAINING_DATA_SOURCE_SENTENCE_PARSER_H_
#define TRAINING_DATA_SOURCE_SENTENCE_PARSER_H_

#include "training/data/structure/contiguous_buffer_backed.word_with_context.h"
#include "training/data/structure/sentence.h"
#include "training/data/structure/sentence.visitor.h"
#include "training/data/structure/word_with_context.h"

#include <vector>
#include <memory>

namespace wombat {

  /**
   * Sentence parser takes a single sentence and generates instances of WordWithContext.
   */
  class SentenceParser : public SentenceVisitor {
    public:
      /**
       * Parser will start by visiting the sentence and copying out all of the
       * sentence's word indices in order to a local vector.
       * TODO: maybe don't init with specific sentence, but take that as an arg to parse() function?
       */
      SentenceParser(
          const Sentence& sentence,
          int32_t windowSize)
      : _maxNumberOfContextWords(windowSize * 2),
        _windowSize(windowSize) {
        _currentPosition = 0;

        // TODO: this is stupid :( if the sentence were an iterator we could use
        // it directly... this is only here to maintain some abstraction of Sentence.
        // N steps to copy a vector to a vector :|
        sentence.acceptWordVisitor(*this);
      }

      /**
       * Implement virtual destructor.
       */
      ~SentenceParser() {}

      /**
       * Implement SentenceVisitor.
       */
      void visitWord(const int32_t& wordIndex);

      /**
       * Get the next WordWithContext instance from the sentence.
       *
       * @return pointer to a new WordWithContext instance, or nullptr if
       *   there's no more data to parse from this sentence.
       */
      std::unique_ptr<WordWithContext> nextWordWithContext();

    private:
      const int32_t _maxNumberOfContextWords;
      const int32_t _windowSize;
      int32_t _currentPosition;
      std::vector<int32_t> _wordIndices;
  };

}

#endif
