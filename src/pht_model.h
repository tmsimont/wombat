// Copyright 2017 Trevor Simonton

#ifndef PHT_MODEL_H_
#define PHT_MODEL_H_

#include "worker_model.h"
#include "consumer.h"
#include "console.h"
#include "omp.h"

class PHTModel : public WorkerModel {
public:
  void initWombat();
  void train();
};


class PHTWorker : public Worker {
public:
  PHTWorker(int id, WorkerModel *m) : Worker(id, m) {}

  int work();

private:
  Consumer c;
};

extern int num_phys;

#endif
