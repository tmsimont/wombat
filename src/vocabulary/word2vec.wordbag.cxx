#include "vocabulary/wordbag.h"
#include "vocabulary/word2vec.wordbag.h"

namespace wombat {

  /**
   * Constructor will initialize memory for the vocabulary array
   * and for the hash table. The vocab array will grow as necessary, but
   * the hash to word index int array will start at maximum size.
   *
   * The first entry in the vocab will be "</s>"
   */
  Word2VecWordBag::Word2VecWordBag() {
    vocab_max_size = 1000;
    vocab_size = 0;
    vocab = static_cast<vocab_word*>(calloc(vocab_max_size, sizeof(struct vocab_word)));
    vocab_hash = static_cast<int*>(calloc(VOCAB_HASH_SIZE, sizeof(int)));
    memset(vocab_hash, -1, VOCAB_HASH_SIZE * sizeof(int));

    // Start with a special character in the vocab.
    AddWordToVocab((char *) "</s>");

    _cardinality = 0;
  }

  Word2VecWordBag::~Word2VecWordBag() {
    for (int i = 0; i < vocab_size; i++) {
      free(vocab[i].word);
      free(vocab[i].point);
      free(vocab[i].code);
    }
    free(vocab);
    free(vocab_hash);
  }

  /**
   * Adds a word to the wordbag. If it is already present,
   * it will increment a counter of how many times add() has been called
   * for this particular word.
   */
  void Word2VecWordBag::add(const std::string& word) {
    // TODO: avoid extra memory use here
    // Copy the word to a c string that we can use in the old Google C word2vec API.
    char c_word[MAX_STRING];
    strcpy(c_word, word.c_str());

    // Determine if the word is already in the wordbag.
    int i = SearchVocab(c_word);

    // Add the word if it's missing, else increment its count.
    if (i == -1) {
      int a = AddWordToVocab(c_word);
      vocab[a].cn = 1;
    } else {
      vocab[i].cn++;
    }

    // TODO: figure out how to manage this in API and tests
    if (vocab_size > VOCAB_HASH_SIZE * 0.7)
      ReduceVocab();

    return;
  }

  /**
   * Get the index of a word in the wordbag.
   *
   * @return the position of a word in the vocabulary or -1 if the word is not found
   */
  int32_t Word2VecWordBag::getWordIndex(const std::string& word) {
    // TODO: avoid extra memory use here
    // Copy the word to a c string that we can use in the old Google C word2vec API.
    char c_word[MAX_STRING];
    strcpy(c_word, word.c_str());

    return SearchVocab(c_word);
  }

  /**
   * Get the frequency of a word in the wordbag.
   *
   * @return the number of times the given word appears in the wordbag.
   */
  int32_t Word2VecWordBag::getWordFrequency(const std::string& word) {
    // TODO: avoid extra memory use here
    // Copy the word to a c string that we can use in the old Google C word2vec API.
    char c_word[MAX_STRING];
    strcpy(c_word, word.c_str());

    int i = SearchVocab(c_word);
    if (i == -1) {
      return 0;
    }

    // TODO: int and int32_t compatibility check?
    return vocab[i].cn;
  }

  /**
   * Sorts the vocabulary by frequency using word counts and calculates sum of 
   * all word frequencies. This also optimizes memory footprint of vocab.
   *
   * @param infrequentThreshold - If greater than 0, remove words in the vocabulary that
   *  appear less often than the threshold.
   * @return sum of all individual word frequencies.
   */
  uint64_t Word2VecWordBag::sortAndSumFrequency(int32_t infrequentThreshold) {
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);

    // Recompute hash after sort.
    memset(vocab_hash, -1, VOCAB_HASH_SIZE * sizeof(int));
    int size = vocab_size;
    for (int i = 0; i < size; i++) {
      // If given threshold is greater than 0, discard words occuring less than the given value.
      if (infrequentThreshold > 0 && (vocab[i].cn < infrequentThreshold) && (i != 0)) {
        vocab_size--;
        free(vocab[i].word);
      } else {
        int hash = GetWordHash(vocab[i].word);
        while (vocab_hash[hash] != -1)
          hash = (hash + 1) % VOCAB_HASH_SIZE;
        vocab_hash[hash] = i;
        _cardinality += vocab[i].cn;
      }
    }
    vocab = static_cast<vocab_word*>(realloc(vocab, vocab_size * sizeof(struct vocab_word)));

    // Allocate memory for the binary tree construction
    for (int a = 0; a < vocab_size; a++) {
      vocab[a].code = static_cast<char*>(calloc(MAX_CODE_LENGTH, sizeof(char)));
      vocab[a].point = static_cast<int*>(calloc(MAX_CODE_LENGTH, sizeof(int)));
    }

    return _cardinality;
  }

  int32_t Word2VecWordBag::getSize() {
    return vocab_size;
  }

  uint64_t Word2VecWordBag::getCardinality() {
    return _cardinality;
  }

  /**
   * Returns hash value of a word.
   */
  int Word2VecWordBag::GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % VOCAB_HASH_SIZE;
    return hash;
  }

  /**
   * Search the vocabulary for a word.
   *
   * @return the position of a word in the vocabulary or -1 if the word is not found
   */
  int Word2VecWordBag::SearchVocab(char *word) {
    int hash = GetWordHash(word);
    while (1) {
      if (vocab_hash[hash] == -1)
        return -1;
      if (!strcmp(word, vocab[vocab_hash[hash]].word))
        return vocab_hash[hash];
      hash = (hash + 1) % VOCAB_HASH_SIZE;
    }
    return -1;
  }
  
  /**
   * Add a word to the vocabulary with the assumption it is not currently
   * in the vocabulary. This will allocate memory for the word that the vocab_struct
   * points to.
   *
   * This also might increase the amount of memory that the vocab uses, and it will
   * update our vocab hash table.
   * 
   * @return The updated size of the vocabulary.
   */
  int Word2VecWordBag::AddWordToVocab(char *word) {
    int hash, length = strlen(word) + 1;
    if (length > MAX_STRING)
      length = MAX_STRING;
    vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    // NOTE: seems that we always need a -1 in our array to avoid an infinite loop 
    // when searching over hash collisions, which is likely why this is `+ 2`
    if (vocab_size + 2 >= vocab_max_size) {
      vocab_max_size += 1000;
      vocab = static_cast<vocab_word*>(realloc(vocab, vocab_max_size * sizeof(struct vocab_word)));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1)
      hash = (hash + 1) % VOCAB_HASH_SIZE;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
  }

  /**
   * Used for sorting by word counts.
   */
  int Word2VecWordBag::VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
  }

  /**
   * Reduces the vocabulary by removing infrequent tokens.
   */
  void Word2VecWordBag::ReduceVocab() {
    int count = 0;
    for (int i = 0; i < vocab_size; i++) {
      if (vocab[i].cn > min_reduce) {
        vocab[count].cn = vocab[i].cn;
        vocab[count].word = vocab[i].word;
        count++;
      } else {
        free(vocab[i].word);
      }
    }
    vocab_size = count;
    memset(vocab_hash, -1, VOCAB_HASH_SIZE * sizeof(int));

    for (int i = 0; i < vocab_size; i++) {
      // Hash will be re-computed, as it is not actual
      int hash = GetWordHash(vocab[i].word);
      while (vocab_hash[hash] != -1)
        hash = (hash + 1) % VOCAB_HASH_SIZE;
      vocab_hash[hash] = i;
    }
    min_reduce++;
  }
}

