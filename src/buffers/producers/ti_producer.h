// Copyright 2017 Trevor Simonton

#ifndef TI_PRODUCER_H_
#define TI_PRODUCER_H_

#include "src/buffers/readers/tc_buffer.h"
#include "src/buffers/sen_buffer.h"
#include "src/buffers/readers/sen_buffer.h"
#include "src/w2v-functions.h"

/**
 * Target/index producer used to produce
 * sets of training/context words to a TCBuffer
 * from a Sentence Buffer
 */
class TIProducer {
 public:
  void produce(TCBufferReader *tc_reader);
  bool hasSentence();
  bool loadSentence(SenBuffer *sen_buffer);
 protected:
  bool sentenceLoaded = false;
  SenBufferReader sen_reader;
  unsigned long long next_random = 1;
};

#endif
