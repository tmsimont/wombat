#include "training/data/structure/contiguous_word_with_context_buffer.h"

namespace wombat {
  int32_t ContiguousWordWithContextBuffer::push(std::shared_ptr<WordWithContext> item) {
    // Ugly way to avoid unnecessary work if the derived clss of the item is
    // already ContiguousBufferBackedWordWithContext...
    auto derived = dynamic_cast<ContiguousBufferBackedWordWithContext*>(item.get());
    if (derived != nullptr) {
      return _buffer.push(derived->data);
    }

    // If the given item is not ContiguousBufferBackedWordWithContext, then
    // convert it, and push the backing data structure.
    return _buffer.push(
        ContiguousBufferBackedWordWithContext::builder(
          _buffer.itemSize() - ContiguousBufferBackedWordWithContext::DATA_SIZE)
        .fromWordWithContext(item)
        .build()->data);
  }

  std::unique_ptr<WordWithContext> ContiguousWordWithContextBuffer::pop() {
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

