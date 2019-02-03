#ifndef TRAINING_DATA_STRUCTURE_CONTIGUOUS_BUFFER_BACKED_WORD_WITH_CONTEXT_H_
#define TRAINING_DATA_STRUCTURE_CONTIGUOUS_BUFFER_BACKED_WORD_WITH_CONTEXT_H_

#include "training/data/structure/word_with_context.h"
#include "training/data/structure/word_with_context.visitor.h"

#include <stdint.h>
#include <vector>
#include <memory>

namespace wombat {
  // Forward declaration so we can be friends.
  class ContiguousWordWithContextBuffer;
  // Forward declaration so we can be friends, and return builders from a static helper.
  class ContiguousBufferBackedWordWithContextBuilder;

  /**
   * This is a WordWithContext instance that is backed by a contiguous buffer 
   * of WordWithContext instances.
   */
  class ContiguousBufferBackedWordWithContext : public WordWithContext {
    // For sharing access to the private data vector. 
    friend class ContiguousWordWithContextBuffer;

    // For access to internal static data indices.
    friend class ContiguousBufferBackedWordWithContextBuilder;

    public:
      /**
       * The constructor expects a unique pointer to the input data source.
       * This structure expects to own the reference to its input.
       * Internally, it will convert the unique pointer to shared pointer, so 
       * it can be shared with the contiguous ring buffer during push attempts.
       * This structure may be deleted but the data that was shared with the ring buffer
       * could have been copied out to the contiguous memory space.
       */
      ContiguousBufferBackedWordWithContext(std::unique_ptr<std::vector<int32_t>> input) 
        : data(std::move(input)) {
      }

      /**
       * Return the target word that has context words around it.
       */
      int32_t getTargetWord() const;

      /**
       * During collection of training data we might drop context words around a target.
       * We keep track of how many times we drop context words. This tells us how many
       * context words were dropped from around the target word when collecting this 
       * WordWithContext.
       */
      int32_t getNumberOfDroppedContextWordSamples() const;

      /**
       * Return the number of context words have we sampled for this training target word.
       */
      int32_t getNumberOfContextWords() const;

      /**
       * Accept a visitor that will visit all of the context words in order.
       * TODO: an iterator would probably be more appropriate here.
       */
      void acceptContextWordVisitor(WordWithContextVisitor& visitor) const;

      /**
       * Builder for construction of instances of this WordWithContext type.
       *
       * @param maxNumberOfContextWords is the maximum number of context words supported.
       */
      static ContiguousBufferBackedWordWithContextBuilder builder(int32_t maxNumberOfContextWords);

    private:
      /**
       * This is the number of integers we store in addition to #maxNumberOfContextWords.
       */
      static const int32_t DATA_SIZE = 3;

      /**
       * Index in backing integer array structure for our target word index.
       */
      static const int32_t TARGET_WORD_INDEX = 0;

      /**
       * Index in backing integer array structure for our number of dropped words..
       */
      static const int32_t DROPPED_WORDS_INDEX = 1;

      /**
       * Index in backing integer array structure for our number of context words.
       */
      static const int32_t NUMBER_OF_CONTEXT_WORDS_INDEX = 2;

      /**
       * Index in backing integer array structure for the index at which context words start.
       */
      static const int32_t CONTEXT_WORDS_START_INDEX = 3;

      /**
       * This is shared with the ContiguousWordWithContextBuffer.
       */
      const std::shared_ptr<std::vector<int32_t>> data;
  };

  /**
   * Builder used for constructing a ContiguousBufferBackedWordWithContext.
   */
  class ContiguousBufferBackedWordWithContextBuilder {
    public:
      /**
       * Constructor for a builder that can build a buffer entry with the given 
       * maximum number of context words.
       *
       * @param maxNumberOfContextWords is the maximum number of context words supported.
       */
      ContiguousBufferBackedWordWithContextBuilder(int32_t maxNumberOfContextWords) :
        _entrySize(maxNumberOfContextWords + ContiguousBufferBackedWordWithContext::DATA_SIZE),
        _data(std::make_unique<std::vector<int32_t>>(_entrySize)) {
      }

      ContiguousBufferBackedWordWithContextBuilder& withTargetWord(int32_t target);

      ContiguousBufferBackedWordWithContextBuilder& withContextWord(int32_t wordIndex);

      ContiguousBufferBackedWordWithContextBuilder& withDroppedWordCount(int32_t count);

      /**
       * Return a unique pointer to a new instance of the word with context.
       */
      std::unique_ptr<ContiguousBufferBackedWordWithContext> build();

    private:
      const int32_t _entrySize;
      std::unique_ptr<std::vector<int32_t>> _data;
      int32_t _target;
      int32_t _droppedCount;
      int32_t _numberOfContextWords = 0;
  };
}

#endif
