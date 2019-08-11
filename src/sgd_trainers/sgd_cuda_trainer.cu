// Copyright 2017 Trevor Simonton

#include "src/sgd_trainers/sgd_cuda_trainer.h"

float *d_Wih, *d_Woh, *d_expTable;

SGDCUDATrainer::SGDCUDATrainer(int num_batches, int batch_size) {
  this->batch_size = batch_size;
  this->num_batches = num_batches;

  if (hs) {
    twords_batch_size = MAX_CODE_LENGTH;
  } else {
    twords_batch_size = negative + 1;
  }
  labels_batch_size = twords_batch_size;

  cwords_batch_size = (2 * window + 1);

  int data_batch_size = cwords_batch_size * twords_batch_size;

  const long bytes_needed =
    (long)num_batches
    * (long)batch_size
    * (long)data_batch_size
    * sizeof(int);

  twords_bytes = bytes_needed;
  labels_bytes = bytes_needed;
  cwords_bytes = bytes_needed;

  checkCuda(cudaMallocHost(reinterpret_cast<void **>(&cwords), cwords_bytes));
  checkCuda(cudaMallocHost(reinterpret_cast<void **>(&twords), twords_bytes));
  checkCuda(cudaMallocHost(reinterpret_cast<void **>(&labels), labels_bytes));

  cudaMalloc(reinterpret_cast<void **>(&d_cwords), cwords_bytes);
  cudaMalloc(reinterpret_cast<void **>(&d_twords), twords_bytes);
  cudaMalloc(reinterpret_cast<void **>(&d_labels), labels_bytes);

  // I think this idea is dead and we always use stream[0]...
  num_streams = num_batches;

  streams = reinterpret_cast<cudaStream_t *>(
      malloc(num_streams * sizeof(cudaStream_t)));
  memoutEvents = reinterpret_cast<cudaEvent_t *>(
      malloc(num_streams * sizeof(cudaEvent_t)));

  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreate(&(streams[i]));
    cudaEventCreate(&(memoutEvents[i]));
  }
}

SGDCUDATrainer::~SGDCUDATrainer() {
  cudaFreeHost(cwords);
  cudaFreeHost(twords);
  cudaFreeHost(labels);

  cudaFree(d_cwords);
  cudaFree(d_twords);
  cudaFree(d_labels);
}

void SGDCUDATrainer::memtoCUDA() {
  int twi = 0;
  int cwi = 0;
  for (int i = 0; i < wombat.size(); i++) {
    wombat[i]->copyTWords(twords + twi);
    wombat[i]->copyLabels(labels + twi);
    wombat[i]->copyCWords(cwords + cwi);
    twi += wombat[i]->numTWords();
    cwi += wombat[i]->numCWords();
  }

  for (int i = 0; i < wovbat.size(); i++) {
    twords[twi] = wovbat[i].tword;
    labels[twi] = wovbat[i].label;
    cwords[cwi] = wovbat[i].cword;
    twi++;
    cwi++;
  }

  checkCuda(cudaMemcpyAsync(
        d_cwords,
        cwords,
        cwords_bytes,
        cudaMemcpyHostToDevice,
        streams[0]));
  checkCuda(cudaMemcpyAsync(
        d_twords,
        twords,
        twords_bytes,
        cudaMemcpyHostToDevice,
        streams[0]));
  checkCuda(cudaMemcpyAsync(
        d_labels,
        labels,
        labels_bytes,
        cudaMemcpyHostToDevice,
        streams[0]));
  checkCuda(cudaEventRecord(memoutEvents[0], streams[0]));
}

void SGDCUDATrainer::train() {
  if (loaded_sets < (batch_size * num_batches)) {
    return;
  }

  memtoCUDA();

  CallKernels(hs, wombat.size(), wovbat.size(), streams + 0, // here's the always use 0 part..
    d_Wih,
    d_Woh,
    d_cwords,
    0,
    d_twords,
    0,
    d_labels,
    labels_batch_size,
    hidden_size,
    alpha,
    MAX_EXP);
}

void SGDCUDATrainer::clear() {
  SGDBatchTrainer::clear();
  for (auto &p : wombat) {
    delete p;
  }
  wombat.clear();
  wovbat.clear();
  used.clear();
}

/**
 * I started calling things "sets" because the batch of batch shit is confusing AF.
 * I never got around to cleaning this up until now... The "set" is a bunch of "minibatches".
 * Check out the commit history on this note. I'm adding a bunch of notes in one big commit and
 * otherwise I'm not working on the master branch anymore. I'm starting to refactor things in the v2 branch.
 */
void SGDCUDATrainer::loadSet(TCBufferReader *tc_reader) {
  if (loaded_sets == batch_size * num_batches) return;
  if (loaded_sets == 0)
    cudaEventSynchronize(memoutEvents[0]);


  int target_indices[twords_batch_size];
  int labels[twords_batch_size];
  int targets_to_load = 0;
  if (hs) {
    int target = tc_reader->targetWord();
    for (int k = 0; k < vocab[target].codelen; k++) {
          target_indices[targets_to_load] = vocab[target].point[k];
          labels[targets_to_load] = vocab[target].code[k];
          targets_to_load++;
    }
  } else {
    for (int i = 0; i < twords_batch_size; ++i) {
      if (i == 0) {
        target_indices[i] = tc_reader->targetWord();
        labels[i] = 1;
      } else {
        int sample = 0;
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        sample = table[(next_random >> 16) % table_size];
        if (!sample)
          sample = next_random % (vocab_size - 1) + 1;
        target_indices[i] = sample;
        labels[i] = 0;
      }
      targets_to_load++;
    }
  }

  /**
   * This craziness maps all of the "minibatches" in our big batch of batches to a bunch of sub-operations.
   * We look for "minibatches" that have 4 input rows (e.g. target words / h-softmax nodes) and 8 context words
   * aka "output layers" in the minibatch.
   * We batch up all the 4x8 minibatches into MOP structures, which can then be crunched with a highly optimized
   * kernel.
   * The rest of our training minibatches get split up into VOP structures, where we use an optimized dot-product-based
   * kernel to take care of the non-4x8-minibatches.
   * We could take even more shapes here if we wanted to (e.g. 32x1, 8x4, 2x16) and write optimized kernels for them,
   * too, but i ran out of time while working on the thesis :P
   * The numbers are centered around 32 because of the way warp shuffling works in cuda...
   */
  int twi = 0;
  while (targets_to_load-twi > 0) {
    int numcw = tc_reader->numCWords();
    int cwi = 0;
    while (numcw > 0) {
      if (numcw >= 8 && (targets_to_load - twi) >= 4) {
        MOP4x8 *m =  new MOP4x8();
        for (int i = 0; i < 8; i++) {
          m->addCWord(*(tc_reader->cwords() + cwi++));
          numcw--;
        }
        for (int i = 0; twi+i < targets_to_load && i < 4; i++) {
          m->addTWord(target_indices[twi+i], labels[twi+i]);
        }
        wombat.push_back(m);
      } else {
        for (int i = 0; twi+i < targets_to_load && i < 4; i++) {
          VOP v(
              target_indices[twi+i],
              *(tc_reader->cwords() + cwi),
              labels[twi+i]);
          wovbat.push_back(v);
        }
        cwi++;
        numcw--;
      }
    }
    twi += 4;
  }

  loaded_sets++;
}

void InitNetCUDA(real **Wih, real **Woh) {
  checkCuda(cudaMalloc((void **)&d_Woh,
        (long long)vocab_size * hidden_size * sizeof(float)));
  checkCuda(cudaMemcpy(d_Woh, *Woh,
        (long long)vocab_size * hidden_size * sizeof(float),
        cudaMemcpyHostToDevice));
  checkCuda(cudaMalloc((void **)&d_Wih,
        (long long)vocab_size * hidden_size * sizeof(float)));
  checkCuda(cudaMemcpy(d_Wih, *Wih,
        (long long)vocab_size * hidden_size * sizeof(float),
        cudaMemcpyHostToDevice));
}

void InitExpCUDA() {
  checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_expTable),
        (EXP_TABLE_SIZE + 1) * sizeof(float)));
  checkCuda(cudaMemcpy(d_expTable, expTable,
        (EXP_TABLE_SIZE + 1) * sizeof(float),
        cudaMemcpyHostToDevice));
}

void WiToHost(real **Wih) {
  cudaDeviceSynchronize();
  checkCuda(cudaMemcpy(*Wih, d_Wih,
        (long long)vocab_size * hidden_size * sizeof(float),
        cudaMemcpyDeviceToHost));
}

void WoToHost(real **Woh) {
  cudaDeviceSynchronize();
  checkCuda(cudaMemcpy(*Woh, d_Woh,
        (long long)vocab_size * hidden_size * sizeof(float),
        cudaMemcpyDeviceToHost));
}
