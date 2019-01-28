#ifndef TRAINING_DATA_CONTIGUOUS_WORD_WITH_CONTEXT_BUFFER_H_
#define TRAINING_DATA_CONTIGUOUS_WORD_WITH_CONTEXT_BUFFER_H_

#include "training/data/contiguous_buffer_backed.word_with_context.hpp"
#include "contiguous_buffer/int32_ring_buffer.hpp"

#include <stdint.h>

#include <array>
#include <memory>
#include <stdexcept>

namespace wombat {

  template <int ENTRY_SIZE>
  class ContiguousWordWithContextBuffer {
    public:
      ContiguousWordWithContextBuffer(int32_t num_items):_buffer(num_items) {
      }

      /**
       * User passes in shared pointer to a WordWithContext instance.
       * Its data will be de-referenced and copied into the buffer, or ignored if the buffer
       * is full.
       * @return 0 on full buffer or 1 on successful insertion.
       */
      int32_t push(std::shared_ptr<ContiguousBufferBackedWordWithContext<ENTRY_SIZE>> item) {
        return _buffer.push(item->data);
      }

      std::unique_ptr<ContiguousBufferBackedWordWithContext<ENTRY_SIZE>> pop() {
        std::unique_ptr<ContiguousBufferBackedWordWithContext<ENTRY_SIZE>> item = 
          std::make_unique<ContiguousBufferBackedWordWithContext<ENTRY_SIZE>>(_buffer.pop());
        if (item->data == nullptr) {
          return nullptr;
        }
        return item;
      }

    private:
      Int32RingBuffer<ENTRY_SIZE> _buffer;
  };
}

#endif
