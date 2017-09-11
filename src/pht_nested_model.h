// Copyright 2017 Trevor Simonton

#ifndef PHT_NESTED_MODEL_H_
#define PHT_NESTED_MODEL_H_

#include "src/worker_model.h"
#include "src/w2v-functions.h"
#include "src/common.h"
#include "src/shared_consumer.h"
#include <vector>
#include "src/tc_buffer.h"
#include "src/sen_buffer.h"
#include "src/console.h"
#include "src/timer.h"
#include "omp.h"

class PHTNestedModel : public WorkerModel {
public:
  vector<SharedConsumer *> shared_consumers;
  int num_groups;

  void initWombat();
  void train();
};


class PHTNestedWorker : public Worker {
public:

  PHTNestedWorker(int id, WorkerModel *m) : Worker(id, m) {}

  int work();

private:
  SharedConsumer *c;
};

extern int num_phys;

#endif
