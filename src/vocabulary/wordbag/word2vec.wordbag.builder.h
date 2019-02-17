#ifndef VOCABULARY_WORDBAG_WORD2VEC_WORDBAG_BUILDER_H_
#define VOCABULARY_WORDBAG_WORD2VEC_WORDBAG_BUILDER_H_

#include "vocabulary/wordbag/wordbag.builder.h"
#include "vocabulary/wordbag/word2vec.wordbag.h"

#include <memory>

/**
 * Used to build the WordBag
 */
namespace wombat {
  class Word2VecWordBagBuilder : public WordBagBuilder {
    public:
      Word2VecWordBagBuilder();
      WordBagBuilder& add(const std::string& word);
      WordBagBuilder& withFrequencyThreshold(uint32_t threshold);
      std::unique_ptr<WordBag> build();
    private:
      std::unique_ptr<Word2VecWordBag> _wordbag;
      uint32_t _threshold;
  };
}

#endif


