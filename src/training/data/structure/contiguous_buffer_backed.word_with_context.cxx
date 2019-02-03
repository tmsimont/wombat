#include "contiguous_buffer_backed.word_with_context.h"

namespace wombat {
  int32_t ContiguousBufferBackedWordWithContext::getTargetWord() const {
    return (*data)[TARGET_WORD_INDEX];
  }

  int32_t ContiguousBufferBackedWordWithContext::getNumberOfDroppedContextWordSamples() const {
    return (*data)[DROPPED_WORDS_INDEX];
  }

  int32_t ContiguousBufferBackedWordWithContext::getNumberOfContextWords() const {
    return (*data)[NUMBER_OF_CONTEXT_WORDS_INDEX];
  }

  void ContiguousBufferBackedWordWithContext::acceptContextWordVisitor(
      WordWithContextVisitor& visitor) const {
    for (int32_t i = 0; i < getNumberOfContextWords(); ++i) {
      visitor.visitContextWord((*data)[CONTEXT_WORDS_START_INDEX + i]);
    }
  }

  ContiguousBufferBackedWordWithContextBuilder
    ContiguousBufferBackedWordWithContext::builder(int32_t entry_size) {
      return ContiguousBufferBackedWordWithContextBuilder(entry_size);
    }

  ContiguousBufferBackedWordWithContextBuilder& 
    ContiguousBufferBackedWordWithContextBuilder::withTargetWord(int32_t target) {
      _target = target;
      return *this;
    }

  ContiguousBufferBackedWordWithContextBuilder& 
    ContiguousBufferBackedWordWithContextBuilder::withContextWord(int32_t wordIndex) {
      if (_numberOfContextWords + 3 < _entrySize) {
        (*_data)[ContiguousBufferBackedWordWithContext
          ::CONTEXT_WORDS_START_INDEX + _numberOfContextWords++] = wordIndex;
      } else {
        // TODO: provide feedback to indicate extra words are ignored.
        // maybe increment _droppedCount?
      }
      return *this;
    }

  ContiguousBufferBackedWordWithContextBuilder& 
    ContiguousBufferBackedWordWithContextBuilder::withDroppedWordCount(int32_t count) {
      _droppedCount = count;
      return *this;
    }

  std::unique_ptr<ContiguousBufferBackedWordWithContext> 
    ContiguousBufferBackedWordWithContextBuilder::build() {
    // Assign values to the vector.
    (*_data)[ContiguousBufferBackedWordWithContext
      ::TARGET_WORD_INDEX] = _target;
    (*_data)[ContiguousBufferBackedWordWithContext
      ::DROPPED_WORDS_INDEX] = _droppedCount;
    (*_data)[ContiguousBufferBackedWordWithContext
      ::NUMBER_OF_CONTEXT_WORDS_INDEX] = _numberOfContextWords;

    // Return a new instance of the word with context.
    return std::make_unique<ContiguousBufferBackedWordWithContext>(std::move(_data));
  }
}

