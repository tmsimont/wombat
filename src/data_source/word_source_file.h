// Copyright 2017 Trevor Simonton

#ifndef WORD_SOURCE_FILE_
#define WORD_SOURCE_FILE_

#include <omp.h>

#include "src/w2v-functions.h"
#include "src/data_source/word_source.h"

class WordSourceFile : public WordSource {
 public:
  virtual int getWord();
  virtual int iterationsRemaining();
  virtual bool rewind();
  WordSourceFile(int id,
      int iters,
      unsigned long long chunkSize,
      char *train_file);
  unsigned long long start,
                     chunkSize;
  FILE *fi;
};

#endif
