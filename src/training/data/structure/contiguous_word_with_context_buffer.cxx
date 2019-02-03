#include "training/data/structure/contiguous_word_with_context_buffer.h"

namespace wombat {
  int32_t ContiguousWordWithContextBuffer::push(
      std::shared_ptr<ContiguousBufferBackedWordWithContext> item) {
    // Delegate to backing buffer.
    return _buffer.push(item->data);
  }

  std::unique_ptr<ContiguousBufferBackedWordWithContext> ContiguousWordWithContextBuffer::pop() {
    // Pull the data from the backing buffer.
    auto data = _buffer.pop();
    if (data == nullptr) {
      // data from the backing buffer will be nullptr if the buffer is empty.
      return nullptr;
    }

    // Return a new wrapper instance around data container.
    // Pass ownership of data to this new instance.
    return std::make_unique<ContiguousBufferBackedWordWithContext>(std::move(data));
  }

}

