// Copyright 2017 Trevor Simonton

#include "batch_model.h"

void BatchModel::initWombat() {

  // file  i/o
  sources = new WordSourceFileGroup(num_threads);
  sources->useLocks(true);
  sources->init();

  for (int i = 0; i < batch_size; ++i) {
    tcbs.push_back(new TCBuffer(items_in_tcb));
  }

}

void BatchModel::train() {
  #pragma omp parallel num_threads(num_threads)
  {
    int id = omp_get_thread_num();
    BatchWorker worker(id, this);

    #pragma omp barrier
    if (id == 0) {
      start = omp_get_wtime();
    }
    #pragma omp barrier

    worker.work();
  }
}

SGDBatchTrainer* BatchModel::getTrainer() {
  return new SGDBatchTrainer(batches_per_thread, batch_size);
}

int BatchWorker::work() {
  c = new BatchConsumer(((BatchModel *)model)->getTrainer());
  for (int i = 0; i < batch_size; ++i) {
    local_sbs.push_back(new SenBuffer(1));
    TIProducer t;
    tipros.push_back(t);
    local_tcbs.push_back(new TCBuffer(items_in_tcb));
  }

  int i = 0;
  int si = id;
  while (1) {

    // fill sen_buffers with different sentences
    if (model->sources->hasActiveSource()) {
      for (i = 0; i < batch_size; ++i) {
        if (!tipros[i].hasSentence()) {
          while (local_sbs[i]->isEmpty() && model->sources->hasActiveSource()) {
            trySourceToSenBuffer(si, local_sbs[i]);
            si = (si + 1) % model->sources->numSources();
          }
        }
      }
    }

    // load tipros and tcb's with sentences
    for (i = 0; i < batch_size; ++i) {
      trySenBufferToTCBuffer(&tipros[i], local_sbs[i], local_tcbs[i]);
      //trySenBufferToTCBuffer(&tipros[i], local_sbs[i], model->tcbs[i]);
    }

    i = 0;
    int s = 0;
    int empty_tcbs = 0;
    int dead_tcbs = 0;
    while (empty_tcbs == 0) {
      c->setTCBuffer(local_tcbs[i]);
      //c->setTCBuffer(model->tcbs[i]);
      c->getTCBuffer()->setLock();
      s = c->acquire();
      c->getTCBuffer()->unsetLock();
      switch (s) {
        case 1:
          // loaded an item
          i++;
          break;
        case -1:
          // consumer is full
          c->consume();
          break;
        case 0:
          // tcb is empty
          empty_tcbs++;
          if (local_sbs[i]->isEmpty() && !tipros[i].hasSentence() && !model->sources->hasActiveSource())
            dead_tcbs++;
          i++;
          break;
      }
      i = i % batch_size;
    }
    // try to consume what we have (might do nothing if not full)
    c->consume();


    // quit if unable to get more sentences and we have at least 1 empty tcb
    if (dead_tcbs > 0) {
      finished = 1;
      return 1;
    }

  }

  return 0;
}

