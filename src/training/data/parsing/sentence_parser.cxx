#include "training/data/parsing/sentence_parser.h"

namespace wombat {
  std::unique_ptr<WordWithContext> SentenceParser::nextWordWithContext() {
    if (_currentPosition >= _wordIndices.size()) {
      return nullptr;
    }

    // TODO: pass in a factory for WordWithContext building
    auto builder = ContiguousBufferBackedWordWithContext::builder(_maxNumberOfContextWords);
    builder.withTargetWord(_wordIndices[_currentPosition]);

    // TODO: get random window size within constraint of configuration
    //int b = next_random % window;
    //for (int i = b; i < 2 * window + 1 - b; i++)
    for (int i = 0; i < 2 * _windowSize + 1; i++) {
      if (i != _windowSize) {
        int c = _currentPosition - _windowSize + i;
        if (c < 0)
          continue;
        if (c >= _wordIndices.size())
          break;
        builder.withContextWord(_wordIndices[c]);
      }
    }

    _currentPosition++;
    return builder.build();
  }

  void SentenceParser::visitWord(const int32_t& wordIndex) {
    _wordIndices.push_back(wordIndex);
  }

}
