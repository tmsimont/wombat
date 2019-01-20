#include "vocabulary/wordbag.h"
#include "vocabulary/word2vec.wordbag.h"
#include "vocabulary/word2vec.wordbag.builder.h"

#include <memory>

namespace wombat {
  Word2VecWordBagBuilder::Word2VecWordBagBuilder() {
    _wordbag = std::make_unique<Word2VecWordBag>();
    _threshold = 0;
  }

  WordBagBuilder& Word2VecWordBagBuilder::add(const std::string& word) {
    _wordbag->add(word);
    return *this;
  }

  WordBagBuilder& Word2VecWordBagBuilder::withFrequencyThreshold(uint32_t threshold) {
    _threshold = threshold;
    return *this;
  }

  std::unique_ptr<WordBag> Word2VecWordBagBuilder::build() {
    _wordbag->sortAndSumFrequency(_threshold);
    return std::move(_wordbag);
  }
}

