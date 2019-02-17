#ifndef VOCABULARY_WORDBAG_PRODUCER_H_
#define VOCABULARY_WORDBAG_PRODUCER_H_

#include "vocabulary/wordbag/wordbag.h"
#include "vocabulary/word_source.h"

#include <istream>
#include <memory>

namespace wombat {
  class WordBagProducer {
    public: 
      static std::unique_ptr<WordBag> fromWordSource(
          const std::shared_ptr<WordSource> wordSource);

      static std::unique_ptr<WordBag> fromSavedBag(
          std::shared_ptr<std::istream> dataStream);
  };
}
#endif
