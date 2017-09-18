// Copyright 2017 Trevor Simonton

#ifdef USE_CUDA
#ifndef CUDA_CONSUMER_H_
#define CUDA_CONSUMER_H_

#include <vector>

#include "src/common.h"
#include "src/w2v-functions.h"
#include "src/sgd_cuda_trainer.h"
#include "src/sgd_cuda_htrainer.h"
#include "src/sgd_trainer.h"
#include "src/buffers/tc_buffer.h"
#include "src/consumer.h"

class CUDAConsumer {
 public:
  CUDAConsumer(int num_batches, int batch_size);
  ~CUDAConsumer();
  int consume();
  int acquire();
  void setTCBuffer(TCBuffer *tcb);
  TCBuffer*  getTCBuffer();
 protected:
  int batch_size;
  int num_batches;
  long acquired = 0;
  unsigned long long word_count = 0, last_word_count = 0;
  SGDCUDATrainer *trainer;
  TCBuffer *tc_buffer;
};


#endif
#endif
