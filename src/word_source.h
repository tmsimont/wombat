// Copyright 2017 Trevor Simonton

#ifndef WORD_SOURCE_H_
#define WORD_SOURCE_H_

class WordSource {
public:
  int id, 
      iters;
  virtual int getWord() = 0;
  virtual int iterationsRemaining() = 0;
  virtual bool rewind() = 0;
};

#endif
