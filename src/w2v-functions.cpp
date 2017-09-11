// Copyright 2017 Trevor Simonton

#include "common.h"
#include "w2v-functions.h"

// The majority of the following code is taken directly from
// the original word2vec implementation, with some modifications taken 
// from Intel's conversion to c++
// Additional modifications have also been made by me, Trevor Simonton


// Original Google copyright

//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

// Intel copyright
//
/*
 * Copyright 2016 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * The code is developed based on the original word2vec implementation from Google:
 * https://code.google.com/archive/p/word2vec/
 */

int binary = 0, debug_mode = 2;
int hs = 0, negative = 5, num_threads = 12, iter = 5, window = 5;
int vocab_max_size = 1000, vocab_size = 0, hidden_size = 100;
unsigned long long min_count = 5, min_reduce = 1, train_words = 0, file_size = 0;
unsigned long long word_count_actual = 0;
real alpha = 0.025f, sample = 1e-3f;
real starting_alpha = alpha;
const real EXP_RESOLUTION = EXP_TABLE_SIZE / (MAX_EXP * 2.0f);
char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int table_size = 1e8;
struct vocab_word *vocab = NULL;
int *vocab_hash = NULL;
int *table = NULL;
real *Wih = NULL, *Woh = NULL, *expTable = NULL;

void InitUnigramTable() {
  table = (int *)malloc(table_size * sizeof(int));

  const real power = 0.75f;
  double train_words_pow = 0.;
#pragma omp parallel for num_threads(num_threads) reduction(+: train_words_pow)
  for (int i = 0; i < vocab_size; i++) {
    train_words_pow += pow(vocab[i].cn, power);
  }

  int i = 0;
  real d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (int a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real) table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size)
      i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
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
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
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
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin))
    return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
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
int VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
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
void ReduceVocab() {
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

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];

  memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

  FILE *fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }

  train_words = 0;
  vocab_size = 0;
  AddWordToVocab((char *) "</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin))
      break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    int i = SearchVocab(word);
    if (i == -1) {
      int a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else
      vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7)
      ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %d\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

  char c;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin))
      break;
    int i = AddWordToVocab(word);
    fscanf(fin, "%llu%c", &vocab[i].cn, &c);
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %d\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fclose(fin);

  // get file size
  FILE *fin2 = fopen(train_file, "rb");
  if (fin2 == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin2, 0, SEEK_END);
  file_size = ftell(fin2);
  fclose(fin2);
}

void InitNet() {
  unsigned long long next_random = 1;

  posix_memalign((void **)&Wih, 128, (long long)vocab_size * hidden_size * sizeof(real));
  if (Wih == NULL) {printf("Memory allocation failed\n"); exit(1);}
  posix_memalign((void **)&Woh, 128, (long long)vocab_size * hidden_size * sizeof(real));
  if (Woh == NULL) {printf("Memory allocation failed\n"); exit(1);}

#pragma omp parallel for num_threads(num_threads) schedule(static, 1)
  for (int i = 0; i < vocab_size; i++) {
    memset(Wih + i * hidden_size, 0.f, hidden_size * sizeof(real));
    memset(Woh + i * hidden_size, 0.f, hidden_size * sizeof(real));
  }

  // initialization
  for (int i = 0; i < vocab_size * hidden_size; i++) {
    next_random = next_random * (unsigned long long) 25214903917 + 11;
    Wih[i] = (((next_random & 0xFFFF) / 65536.f) - 0.5f) / hidden_size;
  }
  CreateBinaryTree();
}

void saveModel() {
  // save the model
  FILE *fo = fopen(output_file, "wb");
  // Save the word vectors
  fprintf(fo, "%d %d\n", vocab_size, hidden_size);
  for (int a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    if (binary)
      for (int b = 0; b < hidden_size; b++)
        fwrite(&Wih[a * hidden_size + b], sizeof(real), 1, fo);
    else
      for (int b = 0; b < hidden_size; b++)
        fprintf(fo, "%f ", Wih[a * hidden_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}
