#ifndef TRAINING_DATA_STRUCTURE_CONTIGUOUS_WORD_WITH_CONTEXT_BUFFER_H_
#define TRAINING_DATA_STRUCTURE_CONTIGUOUS_WORD_WITH_CONTEXT_BUFFER_H_

#include "training/data/structure/contiguous_buffer_backed.word_with_context.h"
#include "contiguous_buffer/int32_ring_buffer.hpp"

#include <stdint.h>

#include <array>
#include <memory>
#include <stdexcept>

namespace wombat {

  /**
   * A buffer of WordsWithContext. The memory that backs this buffer will be contiguous, 
   * which makes it easy/possible to ship to GPU.
   */
  class ContiguousWordWithContextBuffer {
    public:
      /**
       * Constructor will create a buffer for the given number of items.
       *
       * @param numberOfWordsWithContext 
       *  This buffer will hold this many WordsWithContext instances.
       * @param maxNumberOfContextWords
       *  At maximum, each WordsWithContext instance can have only this many context words.
       */
      ContiguousWordWithContextBuffer(
          int32_t numberOfWordsWithContext,
          int32_t maxNumberOfContextWords)
        : _buffer(
            numberOfWordsWithContext,
            // use the max number of context words + the overhead of the wrapper
            maxNumberOfContextWords + ContiguousBufferBackedWordWithContext::DATA_SIZE) {
      }

      /**
       * Implement virtual destructor.
       */
      ~ContiguousWordWithContextBuffer() {}

      /**
       * User passes in shared pointer to a WordWithContext instance.
       * Its data will be de-referenced and copied into the buffer, or ignored if the buffer
       * is full.
       * @return 0 on full buffer or 1 on successful insertion.
       */
      int32_t push(std::shared_ptr<WordWithContext> item);

      /**
       * Pop an item out of the buffer.
       * A unique pointer is returned to a new instance that holds a copy of the memory that
       * was in the contiguous buffer.
       * nullptr is returned if the buffer is empty.
       */
      std::unique_ptr<WordWithContext> pop();

    private:
      Int32RingBuffer _buffer;
  };
}

#endif
