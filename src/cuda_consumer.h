// Copyright 2017 Trevor Simonton

#ifdef USE_CUDA
#ifndef CUDA_CONSUMER_H_
#define CUDA_CONSUMER_H_

#include "common.h"
#include "w2v-functions.h"
#include "sgd_cuda_trainer.h"
#include "sgd_cuda_htrainer.h"
#include "sgd_trainer.h"
#include "tc_buffer.h"
#include <vector>
#include "timer.h"
#include "consumer.h"

class CUDAConsumer {
public:
  CUDAConsumer(int num_batches,int batch_size);
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
