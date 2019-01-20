#ifndef VOCABULARY_WORD2VEC_WORDBAG_H_
#define VOCABULARY_WORD2VEC_WORDBAG_H_

#include "vocabulary/wordbag.h"


namespace wombat {

  /** 
   * word2vec helper struct from original word2vec code.
   */
  struct vocab_word {
    // Number of times this word appears in training input.
    unsigned long long cn;

    // Huffman coding node.
    int *point;
    // Huffman coding code stuff.
    char *code, codelen;

    // Actual word characters.
    char *word;
  };

  /**
   * This implementation of WordBag uses the original C word2vec functions
   * from Google to maintain a lookup table of vocabulary words.
   */
  class Word2VecWordBag : public WordBag {
    public:
      Word2VecWordBag();
      ~Word2VecWordBag();
      void add(const std::string& word);
      int32_t getWordIndex(const std::string& word);
      int32_t getWordFrequency(const std::string& word);
      uint64_t sortAndSumFrequency(int32_t infrequentThreshold);
      int32_t getSize();
    private:
      // Maximum 30 * 0.7 = 21M words in the vocabulary.
      static const int VOCAB_HASH_SIZE = 30000000;

      // Words greater than 100 characters not supported.
      static const int MAX_STRING = 100;

      // This is for the huffman encoding code.
      static const int MAX_CODE_LENGTH = 40;

      // Custom hash table maintenance variables.
      int *vocab_hash;
      int vocab_max_size;
      int vocab_size;
      struct vocab_word *vocab = NULL;

      // Words occuring less than min_count times will be discarded from the vocab
      unsigned long long min_count;

      // This seems to also result in discarded infrequent words...
      // TODO: figure out why both this and min_count are used in original implementation.
      unsigned long long min_reduce;

      int GetWordHash(char *word);
      int SearchVocab(char *word);
      int AddWordToVocab(char *word);
      static int VocabCompare(const void *a, const void *b);
      void ReduceVocab();
  };
}

#endif
