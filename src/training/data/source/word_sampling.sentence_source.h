#ifndef TRAINING_DATA_SOURCE_WORD_SAMPLING_SENTENCE_SOURCE_H_
#define TRAINING_DATA_SOURCE_WORD_SAMPLING_SENTENCE_SOURCE_H_

#include "training/data/source/sentence_source.h"
#include "training/data/source/word_source.h"
#include "vocabulary/wordbag.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <stdexcept>

namespace wombat {

  class WordSamplingSentenceSource : public SentenceSource {
    public:
      /**
       * Construct a sentence source that can create sentences made from the words
       * in the given word bag.
       *
       * @param WordBag that was built from the vocabulary of test data.
       * @param WordSource a source of in-order training words.
       */
      WordSamplingSentenceSource(
          const std::shared_ptr<WordBag> wordBag,
          const std::shared_ptr<WordSource> wordSource)
        : WordSamplingSentenceSource(wordBag, wordSource, 0) {}

      /**
       * Construct a sentence source that can create sentences made from the words
       * in the given word bag, and randomly discard words that appear in the word
       * bag with a high frequency.
       *
       * @param WordBag that built from the vocabulary of test data.
       * @param WordSource a source of in-order training words.
       * @param sample rate TODO: understand/describe this better (its from original word2vec)
       */
      WordSamplingSentenceSource(
          const std::shared_ptr<WordBag> wordBag,
          const std::shared_ptr<WordSource> wordSource,
          const float& sample) 
        : _wordBag(wordBag), _wordSource(wordSource), _sample(sample) {
          if (_wordBag->getSize() == 1) {
            throw std::invalid_argument("Cannot use an empty bag");
          }
        }

      /**
       * Implement virtual destructor.
       */
      ~WordSamplingSentenceSource() {}

      /**
       * Pull the next sentence from the word source. hasNext() should be checked first.
       * @return a unique_ptr to a Sentence instance that contains word indices of
       *   the training words in the order they were pulled from training data.
       *   nullptr is returned if there's no more training data. Call hasNext() first to
       *   avoid getting nullptr.
       */
      std::unique_ptr<Sentence> nextSentence();

      /**
       * Returns true if the file has more sentences to parse.
       */
      bool hasNext();

      /**
       * Resets the internal word source to the start of the training data.
       */
      bool rewind();

    protected:
      /**
       * The subsampling randomly discards frequent words while keeping the ranking same.
       * This uses the old word2vec method for randomly sampling based on word frequency
       * and the cardinality of the vocab.
       */
      bool shouldDiscardWord(const int32_t& wordIndex);

    private:
      static const int32_t MAX_STRING = 64;
      const std::shared_ptr<WordBag> _wordBag;
      const std::shared_ptr<WordSource> _wordSource;
      const float _sample;
  };

}

#endif
