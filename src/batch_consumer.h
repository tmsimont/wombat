#ifndef BATCH_CONSUMER_H_
#define BATCH_CONSUMER_H_

#include "common.h"
#include "w2v-functions.h"
#include "sgd_batch_trainer.h"
#include "sgd_trainer.h"
#include "tc_buffer.h"
#include <vector>
#include "timer.h"
#include "consumer.h"

class BatchConsumer {
public:
  BatchConsumer(SGDBatchTrainer *trainer);
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
