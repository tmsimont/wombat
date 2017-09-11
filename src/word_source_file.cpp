// Copyright 2017 Trevor Simonton

#include "word_source_file.h"

WordSourceFile::WordSourceFile(int id, int iters, unsigned long long chunkSize, char *train_file) {
  this->id = id;
  this->iters = iters;
  this->chunkSize = chunkSize;
  start = chunkSize * (unsigned long long)id;
  fi = fopen(train_file, "rb");
  fseek(fi, start, SEEK_SET);
}

int WordSourceFile::getWord() {
  int word = ReadWordIndex(fi);
  if (feof(fi)) return 0;
  return word;
}

int WordSourceFile::iterationsRemaining() {
  return iters;
}

bool WordSourceFile::rewind() {
  if (feof(fi) || (ftell(fi) - start) > chunkSize) {
    iters--;
    fseek(fi, start, SEEK_SET);
    return true;
  }
  return false;
}
