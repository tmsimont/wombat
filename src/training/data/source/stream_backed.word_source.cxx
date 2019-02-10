#include "training/data/source/stream_backed.word_source.h"

namespace wombat {

  std::string StreamBackedWordSource::nextWord() {
    ReadWord(_word);
    return std::string(_word);
  }

  bool StreamBackedWordSource::rewind() {
    // seek back to the start of the stream
    _stream->clear();
    _stream->seekg(0, std::ios::beg);
    return false;
  }

  bool StreamBackedWordSource::hasNext() {
    _stream->get(_currentCharacter);
    if (_stream->eof()) {
      return false;
    };
    // TODO: rather than putback can this be used on next ReadWord?
    _stream->putback(_currentCharacter);
    return true;
  }

  /**
   * Original word2vec read word implementation (modified to work with stream).
   * TODO: clean this up for a more modern stream-y std lib approach
   */
  void StreamBackedWordSource::ReadWord(char *word) {
    int32_t a = 0;
    while (_stream->get(_currentCharacter)) {
      if (_currentCharacter == 13)
        continue;
      if ((_currentCharacter == ' ') 
          || (_currentCharacter == '\t') 
          || (_currentCharacter == '\n')) {
        if (a > 0) {
          if (_currentCharacter == '\n')
            _stream->putback(_currentCharacter);
          break;
        }
        if (_currentCharacter == '\n') {
          strcpy(word, (char *) "</s>");
          return;
        } else
          continue;
      }
      word[a] = _currentCharacter;
      a++;
      if (a >= MAX_STRING - 1)
        a--;   // Truncate too long words
    }
    word[a] = 0;
  }

}
