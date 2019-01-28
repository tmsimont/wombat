#ifndef TRAINING_DATA_CONTIGUOUS_BUFFER_BACKED_WORD_WITH_CONTEXT_H_
#define TRAINING_DATA_CONTIGUOUS_BUFFER_BACKED_WORD_WITH_CONTEXT_H_

#include "training/data/word_with_context.h"
#include "training/data/word_with_context.visitor.h"

#include <stdint.h>

#include <array>
#include <memory>
#include <stdexcept>

namespace wombat {
  template <int ENTRY_SIZE> class ContiguousWordWithContextBuffer;
  template <int ENTRY_SIZE> class ContiguousBufferBackedWordWithContextBuilder;

  /**
   * This is a WordWithContext instance that is backed by a contiguous buffer 
   * of WordWithContext instances.
   */
  template <int ENTRY_SIZE>
  class ContiguousBufferBackedWordWithContext {

    // For sharing access to the private data array. 
    friend class ContiguousWordWithContextBuffer<ENTRY_SIZE>;

    // For access to internal static data indices.
    friend class ContiguousBufferBackedWordWithContextBuilder<ENTRY_SIZE>;

    public:
      /**
       * The constructor expects a unique pointer to the input data source.
       * This structure expects to own the reference to its input.
       * Internally, it will convert the unique pointer to shared pointer, so 
       * it can be shared with the contiguous ring buffer during push attempts.
       * This structure may be deleted but the data that was shared with the ring buffer
       * could have been copied out to the contiguous memory space.
       */
      ContiguousBufferBackedWordWithContext(
          std::unique_ptr<std::array<int, ENTRY_SIZE>> input) : data(std::move(input)) {
        if (ENTRY_SIZE < 4) {
          throw std::invalid_argument("Invalid ContiguourBufferBackedWordWithContext size.");
        }
      }

      int32_t getTargetWord() {
        return (*data)[TARGET_WORD_INDEX];
      }

      int32_t getNumberOfDroppedContextWordSamples() {
        return (*data)[DROPPED_WORDS_INDEX];
      }

      int32_t getNumberOfContextWords() {
        return (*data)[NUMBER_OF_CONTEXT_WORDS_INDEX];
      }

      void acceptContextWordVisitor(WordWithContextVisitor& visitor) {
        for (int i = 0; i < getNumberOfContextWords(); ++i) {
          visitor.visitContextWord((*data)[CONTEXT_WORDS_START_INDEX + i]);
        }
      }

      static ContiguousBufferBackedWordWithContextBuilder<ENTRY_SIZE> builder() {
        return ContiguousBufferBackedWordWithContextBuilder<ENTRY_SIZE>();
      }

    private:
      static const int TARGET_WORD_INDEX = 0;
      static const int DROPPED_WORDS_INDEX = 1;
      static const int NUMBER_OF_CONTEXT_WORDS_INDEX = 2;
      static const int CONTEXT_WORDS_START_INDEX = 3;

      /**
       * Private but intentionally shared with the ContiguousWordWithContextBuffer.
       */
      const std::shared_ptr<std::array<int, ENTRY_SIZE>> data;
  };

  /**
   * Builder used for constructing a ContiguousBufferBackedWordWithContext.
   */
  template <int ENTRY_SIZE>
  class ContiguousBufferBackedWordWithContextBuilder {
    public:
      ContiguousBufferBackedWordWithContextBuilder& withTargetWord(int32_t target) {
        _target = target;
        return *this;
      }

      ContiguousBufferBackedWordWithContextBuilder& withContextWord(int32_t wordIndex) {
        (*_data)[ContiguousBufferBackedWordWithContext<ENTRY_SIZE>
          ::CONTEXT_WORDS_START_INDEX + _numberOfContextWords++] = wordIndex;
        return *this;
      }

      ContiguousBufferBackedWordWithContextBuilder& withDroppedWordCount(int32_t count) {
        _droppedCount = count;
        return *this;
      }

      /**
       * Return a unique pointer to a new instance of the word with context.
       */
      std::unique_ptr<ContiguousBufferBackedWordWithContext<ENTRY_SIZE>> build() {
        // Assign values to the array.
        (*_data)[ContiguousBufferBackedWordWithContext<ENTRY_SIZE>
          ::TARGET_WORD_INDEX] = _target;
        (*_data)[ContiguousBufferBackedWordWithContext<ENTRY_SIZE>
          ::DROPPED_WORDS_INDEX] = _droppedCount;
        (*_data)[ContiguousBufferBackedWordWithContext<ENTRY_SIZE>
          ::NUMBER_OF_CONTEXT_WORDS_INDEX] = _numberOfContextWords;

        // Return a new instance of the word with context.
        return std::make_unique<ContiguousBufferBackedWordWithContext<ENTRY_SIZE>>(
            std::move(_data));
      }

    private:
        int32_t _target;
        int32_t _droppedCount;
        int32_t _numberOfContextWords = 0;
        std::unique_ptr<std::array<int, ENTRY_SIZE>> _data = 
          std::make_unique<std::array<int, ENTRY_SIZE>>();
  };
}

#endif
