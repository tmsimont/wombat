// Copyright 2017 Trevor Simonton

#include "src/batch_model.h"

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
  // "batch_size" here is references to the number of minibatches we'll have in a batch of operations
  //  that the GPU can crunch all at once. "Minibatch" is not a concept in this version 1 code, but what 
  //  i mean by this is a set of target words (and either negative samples or hierarchical softmax node
  //  vectors) along with "context words" (i.e. vectors from the "output layer")
  //  "batches_per_thread" I THINK is something I gave up on... should probably be 1 always??
  return new SGDBatchTrainer(batches_per_thread, batch_size);
}

/**
 * The idea here is basically that we have multiple CPU threads reading from multiple different
 * points in a file, and then each one goes to put down a chunk of training data into a big "batch"
 * of word vector indices and training labels that get shipped over to GPU.
 * The GPU stuff is hidden in the "trainer"
 * I wrote this thinking I'd use it for both CPU and GPU but only ever implemented a CUDA based trainer.
 * Most of this junk is remnants of old and abandoned experimentation.
 */
int BatchWorker::work() {
  c = new BatchConsumer(reinterpret_cast<BatchModel *>(model)->getTrainer());
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
    }

    i = 0;
    int s = 0;
    int empty_tcbs = 0;
    int dead_tcbs = 0;
    while (empty_tcbs == 0) {
      c->setTCBuffer(local_tcbs[i]);
      c->getTCBuffer()->setLock();
      // here we're filling up that batch of minibatches, but not actually doing anything with it 
      // other than putting down a bunch of word vector indices, and generating negative samples or 
      // hierarchical softmax nodes and labels. So we're parsing the "target/context" word buffers
      // and building pointers to training data and labels for training steps.
      s = c->acquire();
      c->getTCBuffer()->unsetLock();
      switch (s) {
        case 1:
          // loaded an item
          i++;
          break;
        case -1:
          // consumer is full, meaning we have batch_size minibatches ready to crunch.
          // consume will send to GPU and i beleive once memory is on the way we're unblocked to 
          // keep on parsing more training data.
          c->consume();
          break;
        case 0:
          // tcb is empty
          empty_tcbs++;
          if (local_sbs[i]->isEmpty()
              && !tipros[i].hasSentence()
              && !model->sources->hasActiveSource())
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

