#ifndef CUDA_BATCH_MODEL_H_
#define CUDA_BATCH_MODEL_H_

#include "worker_model.h"
#include "w2v-functions.h"
#include "common.h"
#include "consumer.h"
#include <vector>
#include "tc_buffer.h"
#include "sen_buffer.h"
#include "console.h"
#include "timer.h"
#include "omp.h"
#include "sgd_batch_trainer.h"
#include "sgd_cuda_trainer.h"
#include "batch_consumer.h"
#include "batch_model.h"

class CUDABatchModel : public BatchModel {
public:
  void initWombat();
  SGDBatchTrainer* getTrainer();
  void train();
};

#endif
