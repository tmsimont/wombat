#include "training/data/source/file_backed.sentence_source.h"
#include "training/data/structure/stdVector.sentence.h"
#include "util.h"

#include <cmath>
#include <exception>
#include <iostream>

namespace wombat {

  std::unique_ptr<Sentence> FileBackedSentenceSource::nextSentence() {
    std::unique_ptr<Sentence> sentence = std::make_unique<StdVectorSentence>();
    int32_t wordIndex;

    while (true) {
      ReadWord(_word);
      if (_word[0] == 0) break;
      wordIndex = _wordBag->getWordIndex(_word);
      if (wordIndex == -1) continue;
      if (wordIndex == 0) break;
      if (shouldDiscardWord(wordIndex)) {
        sentence->countDiscardedWord();
      } else {
        sentence->addWord(wordIndex);
      }
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
    _fileStream.clear();
    _fileStream.seekg(0, std::ios::beg);
    return false;
  }

  void FileBackedSentenceSource::setFile(const std::string& fileName) {
    if (_fileStream.is_open()) {
      _fileStream.close();
    }
    _fileStream.open(fileName, std::ios::out);
    if (!_fileStream.is_open()) {
      throw std::invalid_argument("Unable to open file: " + fileName);
    }
  }

  bool FileBackedSentenceSource::hasNext() {
    _fileStream.get(_currentCharacter);
    if (_fileStream.eof()) {
      return false;
    };
    // TODO: rather than putback can this be used on next ReadWord?
    _fileStream.putback(_currentCharacter);
    return true;
  }

  /**
   * Original word2vec read word implementation (modified to work with ifstream).
   * TODO: clean this up for a more modern stream-y std lib approach
   */
  void FileBackedSentenceSource::ReadWord(char *word) {
    int32_t a = 0;
    while (_fileStream.get(_currentCharacter)) {
      if (_currentCharacter == 13)
        continue;
      if ((_currentCharacter == ' ') 
          || (_currentCharacter == '\t') 
          || (_currentCharacter == '\n')) {
        if (a > 0) {
          if (_currentCharacter == '\n')
            _fileStream.putback(_currentCharacter);
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

  bool FileBackedSentenceSource::shouldDiscardWord(const int32_t& wordIndex) {
    if (_sample > 0) {
      float p = (std::sqrt(_wordBag->getWordFrequency(wordIndex) 
            / (_sample * _wordBag->getCardinality())) + 1)
        * (_sample * _wordBag->getCardinality())
        / _wordBag->getWordFrequency(wordIndex);
      if (p < (wombat::util::random() & 0xFFFF) / (float)65536) {
        return true;
      }
    }
    return false;
  }

}
