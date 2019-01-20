#ifndef VOCABULARY_WORDBAG_BUILDER_H_
#define VOCABULARY_WORDBAG_BUILDER_H_

#include "vocabulary/wordbag.h"

#include <memory>

/**
 * Used to build the WordBag.
 */
namespace wombat {
  class WordBagBuilder {
    public:
      virtual WordBagBuilder& add(const std::string& word) = 0;
      virtual WordBagBuilder& withFrequencyThreshold(uint32_t threshold) = 0;
      virtual std::unique_ptr<WordBag> build() = 0;
  };
}

#endif

