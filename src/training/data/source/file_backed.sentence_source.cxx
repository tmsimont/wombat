#include "training/data/source/file_backed.sentence_source.h"
#include "training/data/structure/stdVector.sentence.h"

#include <iostream>
#include <stdio.h>
#include <string.h>

namespace wombat {

  std::unique_ptr<Sentence> FileBackedSentenceSource::nextSentence() {
    std::unique_ptr<Sentence> sentence = std::make_unique<StdVectorSentence>();
    int32_t success, wordIndex;
    char word[MAX_STRING];

    while (true) {
      ReadWord(word);
      if (word[0] == 0) break;
      wordIndex = _wordBag->getWordIndex(word);
      if (wordIndex == -1) continue;
      if (wordIndex == 0) break;
      sentence->addWord(wordIndex);
      if (_fileStream.eof()) break;
    }

    // nullptr returned if parsing didn't yield anything
    if (sentence->getNumberOfWordsInput() == 0) {
      return nullptr;
    }

    return sentence;
  }

  bool FileBackedSentenceSource::rewind() {
    // seek back to the start of the stream
    _fileStream.seekg(0);
    return false;
  }

  int32_t FileBackedSentenceSource::setFile(const std::string& fileName) {
    if (_fileStream.is_open()) {
      _fileStream.close();
    }
    _fileStream.open(fileName, std::ios::out);
    if (_fileStream.is_open()) {
      return 1;
    }
    return 0;
  }

  bool FileBackedSentenceSource::hasNext() {
    _fileStream.get(ch);
    if (_fileStream.eof()) {
      return false;
    };
    _fileStream.putback(ch);
    return true;
  }

  /**
   * Original word2vec read word implementation (modified to work with ifstream).
   * TODO: clean this up for a more modern stream-y std lib approach
   */
  void FileBackedSentenceSource::ReadWord(char *word) {
    int32_t a = 0;
    while (_fileStream.get(ch)) {
      if (ch == 13)
        continue;
      if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
        if (a > 0) {
          if (ch == '\n')
            _fileStream.putback(ch);
          break;
        }
        if (ch == '\n') {
          strcpy(word, (char *) "</s>");
          return;
        } else
          continue;
      }
      word[a] = ch;
      a++;
      if (a >= MAX_STRING - 1)
        a--;   // Truncate too long words
    }
    word[a] = 0;
  }

}
