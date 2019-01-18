#ifndef VOCABULARY_DICTIONARY_H_
#define VOCABULARY_DICTIONARY_H_

#include <string>

/**
 * Used for storing and looking up a vocabulary of words.
 */
namespace wombat {
  class Dictionary {
    public:
      Dictionary();
      virtual void add(const std::string& word) = 0;
      virtual int32_t getWordIndex(const std::string& word) = 0;
      virtual int32_t getWordFrequency(const std::string& word) = 0;
      virtual uint64_t sortAndSumFrequency(int32_t infrequentThreshold) = 0;
      virtual int32_t getSize() = 0;
  };
}

#endif


