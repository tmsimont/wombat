#ifndef TI_PRODUCER_H_
#define TI_PRODUCER_H_

#include "sen_buffer.h"
#include "tc_buffer.h"
#include "w2v-functions.h"

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
