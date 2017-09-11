#ifndef BATCH_MODEL_H_
#define BATCH_MODEL_H_

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
#include "batch_consumer.h"

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
