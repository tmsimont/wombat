// Copyright 2017 Trevor Simonton

#ifndef W2V_FUNCTIONS_H_
#define W2V_FUNCTIONS_H_

#include "src/common.h"

typedef float real;

struct vocab_word {
  unsigned long long cn;
  int *point;
  char *word, *code, codelen;
};

extern int binary, debug_mode;
extern int hs, negative, num_threads, iter, window;
extern int vocab_max_size, vocab_size, hidden_size;
extern unsigned long long min_count, min_reduce, train_words, file_size;
extern unsigned long long word_count_actual;
extern real alpha, sample;
extern real starting_alpha;
extern const real EXP_RESOLUTION;
extern char train_file[MAX_STRING], output_file[MAX_STRING];
extern char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
extern const int vocab_hash_size;
extern const int table_size;
extern struct vocab_word *vocab;
extern int *vocab_hash;
extern int *table;
extern real *Wih, *Woh, *expTable;

void InitUnigramTable();
void ReadWord(char *word, FILE *fin);
int GetWordHash(char *word);
int SearchVocab(char *word);
int ReadWordIndex(FILE *fin);
int AddWordToVocab(char *word);
int VocabCompare(const void *a, const void *b);
void SortVocab();
void ReduceVocab();
void LearnVocabFromTrainFile();
void SaveVocab();
void ReadVocab();
void InitNet();
void saveModel();
void CreateBinaryTree();

#endif
