// Copyright 2017 Trevor Simonton

#ifndef TI_PRODUCER_H_
#define TI_PRODUCER_H_

#include "src/sen_buffer.h"
#include "src/tc_buffer.h"
#include "src/w2v-functions.h"

class TIProducer {
public:
  void buildTI(TCBufferReader *tc_reader);
  bool hasSentence();
  bool loadSentence(SenBuffer *sen_buffer);
protected:
  bool sentenceLoaded = false;
  SenBufferReader sen_reader;
  unsigned long long next_random = 1;
};

#endif
