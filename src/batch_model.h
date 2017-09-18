// Copyright 2017 Trevor Simonton

#ifndef BATCH_MODEL_H_
#define BATCH_MODEL_H_

#include <omp.h>

#include <vector>

#include "src/worker_model.h"
#include "src/w2v-functions.h"
#include "src/common.h"
#include "src/consumer.h"
#include "src/buffers/tc_buffer.h"
#include "src/buffers/sen_buffer.h"
#include "src/console.h"
#include "src/sgd_batch_trainer.h"
#include "src/batch_consumer.h"

class BatchModel : public WorkerModel {
 public:
  void initWombat();
  void train();
  virtual SGDBatchTrainer* getTrainer();
};


class BatchWorker : public Worker {
 public:
  BatchWorker(int id, WorkerModel *m) : Worker(id, m) {}

  int work();
  bool tryConsumeTCBuffer(TCBuffer *tc_buffer);

 private:
  vector<SenBuffer *> local_sbs;
  vector<TIProducer> tipros;
  vector<TCBuffer *> local_tcbs;
  BatchConsumer *c;
  SenBuffer *sen_buffer;
};

extern int batch_size;
extern int batches_per_thread;

#endif
