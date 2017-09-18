// Copyright 2017 Trevor Simonton

#ifndef BATCH_CONSUMER_H_
#define BATCH_CONSUMER_H_

#include <vector>

#include "src/common.h"
#include "src/w2v-functions.h"
#include "src/sgd_trainers/sgd_batch_trainer.h"
#include "src/sgd_trainers/sgd_trainer.h"
#include "src/buffers/tc_buffer.h"
#include "src/consumer.h"

class BatchConsumer {
 public:
  explicit BatchConsumer(SGDBatchTrainer *trainer);
  ~BatchConsumer();
  int consume();
  int acquire();
  void setTCBuffer(TCBuffer *tcb);
  TCBuffer*  getTCBuffer();
 protected:
  long acquired = 0;
  unsigned long long word_count = 0, last_word_count = 0;
  SGDBatchTrainer *trainer;
  TCBuffer *tc_buffer;
};


#endif
