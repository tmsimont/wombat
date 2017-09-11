// Copyright 2017 Trevor Simonton

#ifndef PHT_NESTED_MODEL_H_
#define PHT_NESTED_MODEL_H_

#include "worker_model.h"
#include "w2v-functions.h"
#include "common.h"
#include "shared_consumer.h"
#include <vector>
#include "tc_buffer.h"
#include "sen_buffer.h"
#include "console.h"
#include "timer.h"
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
