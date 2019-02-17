#ifndef VOCABULARY_WORDBAG_WORDBAG_H_
#define VOCABULARY_WORDBAG_WORDBAG_H_

#include <stdint.h>
#include <string>

/**
 * Used for storing and looking up a vocabulary of words.
 */
namespace wombat {
  class WordBagBuilder;
  class WordBag {
    friend class WordBagBuilder;

    public:
      virtual ~WordBag() {};

      /**
       * What is the numeric index of the word? This can be used for mapping this
       * word to a word vector in a neural network. It should not change after
       * the neural net is created.
       */
      virtual int32_t getWordIndex(const std::string& word) = 0;

      /**
       * How often does the given word appear in the bag? (What is its multiplicity?)
       */
      virtual int32_t getWordFrequency(const std::string& word) = 0;

      /**
       * How often does the given word appear in the bag? (What is its multiplicity?)
       */
      virtual int32_t getWordFrequency(const int32_t& wordIndex) = 0;

      /**
       * Give the number of unique words in the bag.
       */
      virtual int32_t getSize() = 0;

      /**
       * Give the sum of the multiplicities of all words in the bag.
       */
      virtual uint64_t getCardinality() = 0;

    protected:
      virtual void add(const std::string& word) = 0;
      virtual uint64_t sortAndSumFrequency(int32_t infrequentThreshold) = 0;
  };
}

#endif
