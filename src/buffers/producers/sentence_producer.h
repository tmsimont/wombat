// Copyright 2017 Trevor Simonton

#ifndef SENTENCE_PRODUCER_H_
#define SENTENCE_PRODUCER_H_

#include "src/buffers/readers/sen_buffer.h"
#include "src/w2v-functions.h"
#include "src/data_source/word_source.h"

/**
 * Helper to read from a WordSource
 * and pull out work tokens from the sentence
 * source and put them into a Sentence buffer.
 */
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
