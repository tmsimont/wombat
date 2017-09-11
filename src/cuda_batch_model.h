// Copyright 2017 Trevor Simonton

#ifndef CUDA_BATCH_MODEL_H_
#define CUDA_BATCH_MODEL_H_

#include "src/worker_model.h"
#include "src/w2v-functions.h"
#include "src/common.h"
#include "src/consumer.h"
#include <vector>
#include "src/tc_buffer.h"
#include "src/sen_buffer.h"
#include "src/console.h"
#include "src/timer.h"
#include "omp.h"
#include "src/sgd_batch_trainer.h"
#include "src/sgd_cuda_trainer.h"
#include "src/batch_consumer.h"
#include "src/batch_model.h"

class CUDABatchModel : public BatchModel {
public:
  void initWombat();
  SGDBatchTrainer* getTrainer();
  void train();
};

#endif
