#include "vocabulary/dictionary.h"
#include "vocabulary/word2vec.dictionary.h"

namespace wombat {
  Word2VecDictionary::Word2VecDictionary() {
    vocab_max_size = 1000;
    vocab_size = 0;
  }

  void Word2VecDictionary::add(const std::string& word) {
    return;
  }

  int32_t Word2VecDictionary::get(const std::string& word) {
    return 0;
  }

  // Reads a single word from a file, assuming space + tab + EOL to be word boundaries
  void Word2VecDictionary::ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
      ch = fgetc(fin);
      if (ch == 13)
        continue;
      if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
        if (a > 0) {
          if (ch == '\n')
            ungetc(ch, fin);
          break;
        }
        if (ch == '\n') {
          strcpy(word, (char *) "</s>");
          return;
        } else
          continue;
      }
      word[a] = ch;
      a++;
      if (a >= MAX_STRING - 1)
        a--;   // Truncate too long words
    }
    word[a] = 0;
  }

  // Returns hash value of a word
  int Word2VecDictionary::GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
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
      hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
  }
  
  // Reads a word and returns its index in the vocabulary
  int Word2VecDictionary::ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin))
      return -1;
    return SearchVocab(word);
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
      hash = (hash + 1) % vocab_hash_size;
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
    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

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
          hash = (hash + 1) % vocab_hash_size;
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
    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    for (int i = 0; i < vocab_size; i++) {
      // Hash will be re-computed, as it is not actual
      int hash = GetWordHash(vocab[i].word);
      while (vocab_hash[hash] != -1)
        hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = i;
    }
    min_reduce++;
  }
}

