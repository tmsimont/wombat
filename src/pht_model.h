// Copyright 2017 Trevor Simonton

#ifndef PHT_MODEL_H_
#define PHT_MODEL_H_

#include <omp.h>

#include "src/worker_model.h"
#include "src/consumer.h"
#include "src/console.h"

/**
 * The Paired-Hyperthread model extends the
 * WorkerModel to train a word2vec network.
 * The goal is to put threads that we know will
 * be working on the same physical core similar
 * vectors to train, to maximize the potential for
 * both threads hitting the same cache lines.
 */
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
