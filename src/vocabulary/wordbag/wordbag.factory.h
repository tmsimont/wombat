#ifndef VOCABULARY_WORDBAG_WORDBAG_FACTORY_H_
#define VOCABULARY_WORDBAG_WORDBAG_FACTORY_H_

#include "training/data/source/word_source.h"
#include "vocabulary/wordbag/wordbag.h"

#include <memory>

namespace wombat {
  class WordBagFactory {
    public:
      WordBagFactory(std::shared_ptr<WordSource> wordSource) 
        : _wordSource(wordSource) {
        }

      /**
       * Get a fully functioning wordbag.
       */
      std::unique_ptr<WordBag> makeWordBag();

    private:
      const std::shared_ptr<WordSource> _wordSource;
  };
}

#endif
