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

/**
 * This consumes a "batch" of minibatches...
 * This is part of the "batch_model" code, which reads from file input, and builds
 * batches of minibatches. The only implementation of this that I have is the CUDA-based one.
 * It's full of mess from several old ideas I was playing with while working on this the first time around.
 */
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
