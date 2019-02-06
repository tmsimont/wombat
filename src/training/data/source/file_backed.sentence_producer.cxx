#include "training/data/source/file_backed.sentence_producer.h"
#include "training/data/structure/stdVector.sentence.h"

#include <iostream>
#include <stdio.h>
#include <string.h>

namespace wombat {

  std::unique_ptr<Sentence> FileBackedSentenceProducer::nextSentence() {
    std::unique_ptr<Sentence> sentence = std::make_unique<StdVectorSentence>();
    int32_t success, wordIndex;
    char word[MAX_STRING];

    while (true) {
      ReadWord(word);
      wordIndex = _wordBag->getWordIndex(word);
      if (wordIndex == -1) continue;
      if (wordIndex == 0) break;
      sentence->addWord(wordIndex);
      if (_fileStream.eof()) break;
    }

    return sentence;
  }

  bool FileBackedSentenceProducer::rewind() {
    // TODO: seek back to the start of the stream
    return false;
  }

  int32_t FileBackedSentenceProducer::setFile(const std::string& fileName) {
    _fileStream.open(fileName, std::ios::out);
    if (_fileStream.is_open()) {
      return 1;
    }
    return 0;
  }

  bool FileBackedSentenceProducer::hasNext() {
    return _fileStream.eof();
    // TODO: support starting mid-file
    // || (ftell(fi) - start) > chunkSize);
  }

  /**
   * Original word2vec read word implementation (modified to work with ifstream).
   */
  void FileBackedSentenceProducer::ReadWord(char *word) {
    int32_t a = 0;
    char ch;
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
