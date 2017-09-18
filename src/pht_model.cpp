// Copyright 2017 Trevor Simonton

#include "src/pht_model.h"

#ifdef USE_MKL
#include <mkl.h>
#endif

/**
 * After word2vec memory structures are initialized,
 * this worker model sets up a TCBuffer for each
 * physical core, and a Sentence buffer for each
 * thread.
 */
void PHTModel::initWombat() {
  // data buffers
  for (int i = 0; i < num_phys; ++i) {
    tcbs.push_back(new TCBuffer(items_in_tcb));
  }
  for (int i = 0; i < num_threads; ++i) {
    sbs.push_back(new SenBuffer(sentences_in_buffer));
  }

  // file  i/o
  sources = new WordSourceFileGroup(num_phys);
  sources->useLocks(true);
  sources->init();
}

/**
 * Each thread uses a PHTWorker to accomplish
 * its part of the training process.
 */
void PHTModel::train() {
  #ifdef USE_MKL
  mkl_set_num_threads(1);
  #endif

  #pragma omp parallel num_threads(num_threads)
  {
    int id = omp_get_thread_num();
    PHTWorker worker(id, this);

    #pragma omp barrier
    if (id == 0) {
      start = omp_get_wtime();
    }
    #pragma omp barrier

    worker.work();
  }
}

/**
 *
 */
int PHTWorker::work() {
  int i = id % num_phys;
  while (1) {
    trySourceToSenBuffer(i, model->sbs[id]);
    trySenBufferToTCBuffer(&tipro, model->sbs[id], model->tcbs[i]);

    int s = 1;
    c.setTCBuffer(model->tcbs[i]);
    while (s) {
      model->tcbs[i]->setLock();
      s = c.acquire();
      model->tcbs[i]->unsetLock();
      if (s) s = c.consume();
    }

    if (!tipro.hasSentence()
        && !model->sources->isActive(i)
        && model->sbs[id]->isEmpty()) {
      finished = 1;
      return 1;
    }
  }
  return 0;
}

