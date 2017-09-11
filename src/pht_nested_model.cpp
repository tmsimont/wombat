// Copyright 2017 Trevor Simonton

#include "src/pht_nested_model.h"
#include <sched.h>

#ifdef USE_MKL
#include "mkl.h"
#endif


void PHTNestedModel::initWombat() {
  num_groups = num_phys;

  // data buffers
  for (int i = 0; i < num_groups; ++i) {
    tcbs.push_back(new TCBuffer(items_in_tcb));
    sbs.push_back(new SenBuffer(sentences_in_buffer));
  }

  // file  i/o
  sources = new WordSourceFileGroup(num_groups);
  sources->useLocks(true);
  sources->init();

  for (int i = 0; i < num_groups; ++i) {
    shared_consumers.push_back(new SharedConsumer(tcbs_per_thread));
    shared_consumers[i]->id = i;
  }
}

void PHTNestedModel::train() {
  #ifdef USE_MKL
  mkl_set_num_threads(1);
  #endif

  #pragma omp parallel
  {
    int id = omp_get_thread_num();
    //#pragma omp parallel num_threads(num_threads/num_groups) 
    {
      PHTNestedWorker worker(id, this);

      #pragma omp barrier
      if (id == 0) {
        start = omp_get_wtime();
      }
      #pragma omp barrier

      worker.work();
    }
  }
}


// split producer consumer
int PHTNestedWorker::work() {
  c = ((PHTNestedModel *) model)->shared_consumers[id];

  // nested thread split
  #pragma omp parallel 
  {
    while (1) {
      // let one of the splits handle sentence generation
      #pragma omp single nowait
      {
        // id is shared by local nested threads
        trySourceToSenBuffer(id, model->sbs[id]);
        trySenBufferToTCBuffer(&tipro, model->sbs[id], model->tcbs[id]);
      }

      int s = 1;
      c->setTCBuffer(model->tcbs[id]);
      if (!tipro.hasSentence() && !model->sources->isActive(id) && model->sbs[id]->isEmpty()) {
        c->release();
      }
      while (s) {
        s = c->consume();
      }

      if (!c->working) {
        finish();
        c->consume();
        break;
      }
    }
  }

  return 0;
}

