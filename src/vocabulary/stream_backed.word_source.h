#ifndef VOCABULARY_STREAM_BACKED_WORD_SOURCE_H_
#define VOCABULARY_STREAM_BACKED_WORD_SOURCE_H_

#include "vocabulary/word_source.h"

#include <istream>
#include <memory>
#include <string>

namespace wombat {

  class StreamBackedWordSource : public WordSource {
    public:
      /**
       * Construct a word source that will own a given stream pointer and
       * use this for getting in-order training words.
       */
      StreamBackedWordSource(std::unique_ptr<std::istream> inputStream)
      : _stream(std::move(inputStream)) {}

      /**
       * Implement virtual destructor.
       */
      ~StreamBackedWordSource() {}

      /**
       * Pull the next word from the stream. hasNext() should be checked first.
       */
      std::string nextWord();

      /**
       * Returns true if the stream has more words to parse.
       */
      bool hasNext();

      /**
       * Resets the internal stream pointer to the start of the training data.
       */
      bool rewind();

    private:
      static const int32_t MAX_STRING = 64;
      const std::unique_ptr<std::istream> _stream;
      char _currentCharacter;
      char _word[MAX_STRING];
      void ReadWord(char *word);
  };

}

#endif
