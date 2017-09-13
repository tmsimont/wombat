// Copyright 2017 Trevor Simonton

#ifndef SENTENCE_PRODUCER_H_
#define SENTENCE_PRODUCER_H_

#include "src/sen_buffer.h"
#include "src/w2v-functions.h"
#include "src/word_source.h"

class SentenceProducer {
 public:
  int buildSentence(SenBufferReader *sen_reader);
  void setSource(WordSource *s) {
    source = s;
  }
 protected:
  WordSource *source;
  bool shouldDiscardWord(int word);
  unsigned long long next_random = 1;
};


#endif
