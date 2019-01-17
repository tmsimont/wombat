#include "vocabulary/dictionary.h"
#include "vocabulary/word2vec.dictionary.h"

namespace wombat {
  Word2VecDictionary::Word2VecDictionary() {
    vocab_max_size = 1000;
    vocab_size = 0;
    vocab = reinterpret_cast<struct vocab_word *>(
        calloc(vocab_max_size, sizeof(struct vocab_word)));
    vocab_hash = reinterpret_cast<int *>(
        calloc(VOCAB_HASH_SIZE, sizeof(int)));
    memset(vocab_hash, -1, VOCAB_HASH_SIZE * sizeof(int));
    AddWordToVocab((char *) "</s>");
  }

  /**
   * Adds a word to the dictionary. If it is already present,
   * it will increment a counter of how many times add() has been called
   * for this particular word.
   */
  void Word2VecDictionary::add(const std::string& word) {
    // TODO: avoid extra memory use here
    char c_word[MAX_STRING];
    strcpy(c_word, word.c_str());

    int i = SearchVocab(c_word);
    if (i == -1) {
      int a = AddWordToVocab(c_word);
      vocab[a].cn = 1;
    } else
      vocab[i].cn++;

    // TODO: figure out how to manage this in API and tests
    if (vocab_size > VOCAB_HASH_SIZE * 0.7)
      ReduceVocab();
    return;
  }

  int32_t Word2VecDictionary::get(const std::string& word) {
    // TODO: avoid extra memory use here
    char c_word[MAX_STRING];
    strcpy(c_word, word.c_str());

    int i = SearchVocab(c_word);
    return i;
  }

  // TODO: figure out an API for getting count of word occurrences

  // Returns hash value of a word
  int Word2VecDictionary::GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % VOCAB_HASH_SIZE;
    return hash;
  }

  // Returns position of a word in the vocabulary; if the word is not found, returns -1
  int Word2VecDictionary::SearchVocab(char *word) {
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
  
  // Adds a word to the vocabulary
  int Word2VecDictionary::AddWordToVocab(char *word) {
    int hash, length = strlen(word) + 1;
    if (length > MAX_STRING)
      length = MAX_STRING;
    vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
      vocab_max_size += 1000;
      vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1)
      hash = (hash + 1) % VOCAB_HASH_SIZE;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
  }

  // Used later for sorting by word counts
  int Word2VecDictionary::VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
  }

  // Sorts the vocabulary by frequency using word counts
  void Word2VecDictionary::SortVocab() {
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    memset(vocab_hash, -1, VOCAB_HASH_SIZE * sizeof(int));

    int size = vocab_size;
    train_words = 0;
    for (int i = 0; i < size; i++) {
      // Words occuring less than min_count times will be discarded from the vocab
      if ((vocab[i].cn < min_count) && (i != 0)) {
        vocab_size--;
        free(vocab[i].word);
      } else {
        // Hash will be re-computed, as after the sorting it is not actual
        int hash = GetWordHash(vocab[i].word);
        while (vocab_hash[hash] != -1)
          hash = (hash + 1) % VOCAB_HASH_SIZE;
        vocab_hash[hash] = i;
        train_words += vocab[i].cn;
      }
    }
    vocab = (struct vocab_word *) realloc(vocab, vocab_size * sizeof(struct vocab_word));
    // Allocate memory for the binary tree construction
    for (int a = 0; a < vocab_size; a++) {
      vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
      vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }
  }

  // Reduces the vocabulary by removing infrequent tokens
  void Word2VecDictionary::ReduceVocab() {
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

