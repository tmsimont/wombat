#include "training/data/source/word_sampling.sentence_source.h"
#include "training/data/structure/stdVector.sentence.h"
#include "util.h"

#include <cmath>
#include <exception>
#include <iostream>

namespace wombat {

  std::unique_ptr<Sentence> WordSamplingSentenceSource::nextSentence() {
    std::unique_ptr<Sentence> sentence = std::make_unique<StdVectorSentence>();
    int32_t wordIndex;

    while (true) {
      std::string word = _wordSource->nextWord();
      if (word.size() == 0) break;
      wordIndex = _wordBag->getWordIndex(word);
      if (wordIndex == -1) continue;
      if (wordIndex == 0) break;
      if (shouldDiscardWord(wordIndex)) {
        sentence->countDiscardedWord();
      } else {
        sentence->addWord(wordIndex);
      }
      if (!_wordSource->hasNext()) break;
    }

    // nullptr returned if parsing didn't yield anything
    if (sentence->getNumberOfWordsInput() == 0) {
      return nullptr;
    }

    return sentence;
  }

  bool WordSamplingSentenceSource::rewind() {
    // seek back to the start of the stream
    _wordSource->rewind();
    return false;
  }

  bool WordSamplingSentenceSource::hasNext() {
    return _wordSource->hasNext();
  }

  bool WordSamplingSentenceSource::shouldDiscardWord(const int32_t& wordIndex) {
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
