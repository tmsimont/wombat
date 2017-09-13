// Copyright 2017 Trevor Simonton

#include "src/sgd_batch_trainer.h"

SGDBatchTrainer::SGDBatchTrainer(int num_batches, int batch_size) {
  this->batch_size = batch_size;
  this->num_batches = num_batches;

  if (hs) {
    twords_batch_size = MAX_CODE_LENGTH;
  } else {
    twords_batch_size = negative + 1;
  }

  cwords_batch_size = (2 * window + 1);
  labels_batch_size = twords_batch_size * cwords_batch_size;

  const long size = (long)num_batches * (long)batch_size;

  cwords_bytes = size
    * (long)cwords_batch_size
    * sizeof(int);

  num_cwords_bytes = size
    * sizeof(int);

  twords_bytes = size
    * (long)twords_batch_size
    * sizeof(int);

  num_twords_bytes = size
    * sizeof(int);

  labels_bytes = size
    * (long)labels_batch_size
    * sizeof(int);

  corrWo_bytes = size
    * (long)twords_batch_size
    * hidden_size
    * sizeof(float);

  gradients_bytes = size
    * (long)labels_batch_size
    * sizeof(float);

  posix_memalign(reinterpret_cast<void**>(&cwords),
      64, cwords_bytes);
  posix_memalign(reinterpret_cast<void**>(&num_cwords),
      64, num_cwords_bytes);
  posix_memalign(reinterpret_cast<void**>(&twords),
      64, twords_bytes);
  posix_memalign(reinterpret_cast<void**>(&num_twords),
      64, num_twords_bytes);
  posix_memalign(reinterpret_cast<void**>(&labels),
      64, labels_bytes);
  posix_memalign(reinterpret_cast<void**>(&corrWo),
      64, corrWo_bytes);
  posix_memalign(reinterpret_cast<void**>(&gradients),
      64, gradients_bytes);
}

SGDBatchTrainer::~SGDBatchTrainer() {
  free(cwords);
  free(num_cwords);
  free(twords);
  free(num_twords);
  free(labels);
  free(corrWo);
  free(gradients);
}

void SGDBatchTrainer::train() {
  if (loaded_sets < (batch_size * num_batches)) {
    return;
  }
}

void SGDBatchTrainer::clear() {
  loaded_sets = 0;
}

void SGDBatchTrainer::loadSet(TCBufferReader *tc_reader) {
  if (loaded_sets == batch_size * num_batches) return;

  int loaded_twords = 0;
  int loaded_cwords = 0;

  // Load in the target words
  if (hs) {
    int target = tc_reader->targetWord();
    for (int k = 0; k < vocab[target].codelen; k++) {
        twords[loaded_sets*twords_batch_size + loaded_twords] =
          vocab[target].point[k];
        for (int i = 0; i < cwords_batch_size; ++i) {
          if (i < tc_reader->numCWords())
            labels[loaded_sets*labels_batch_size
              + loaded_twords*cwords_batch_size + i] =
              vocab[target].code[k];
          else
            labels[loaded_sets*labels_batch_size
              + loaded_twords*cwords_batch_size + i] = 0;
        }
        loaded_twords++;
    }
  } else {
    int target = tc_reader->targetWord();
    twords[loaded_sets*twords_batch_size + loaded_twords] = target;
    for (int i = 0; i < cwords_batch_size; ++i) {
      if (i < tc_reader->numCWords())
        labels[loaded_sets*labels_batch_size
          + loaded_twords*cwords_batch_size + i] = 1;
      else
        labels[loaded_sets*labels_batch_size
          + loaded_twords*cwords_batch_size + i] = 0;
    }
    loaded_twords++;

    // generate negative samples for output layer
    for (int k = 0; k < negative; k++) {
      int sample = 0;
      if (randomness)  {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        sample = table[(next_random >> 16) % table_size];
        if (!sample)
          sample = next_random % (vocab_size - 1) + 1;
      } else {
        next_random = (next_random + 20) % vocab_size;
        sample = next_random;
      }
      twords[loaded_sets*twords_batch_size + loaded_twords] = sample;
      for (int i = 0; i < cwords_batch_size; ++i) {
        labels[loaded_sets*labels_batch_size
          + loaded_twords*cwords_batch_size + i] = 0;
      }
      loaded_twords++;
    }
  }

  // Load in the context words
  for (int i = 0; i < tc_reader->numCWords(); i++) {
    cwords[loaded_sets*cwords_batch_size + loaded_cwords] =
      *(tc_reader->cwords() + i);
    loaded_cwords++;
  }

  num_twords[loaded_sets] = loaded_twords;
  num_cwords[loaded_sets] = loaded_cwords;

  // pad 0's to make gpu data size consistent
  while (loaded_twords < twords_batch_size) {
    twords[loaded_sets*twords_batch_size + loaded_twords] = 0;
    for (int i = 0; i < cwords_batch_size; ++i) {
      labels[loaded_sets*labels_batch_size
        + loaded_twords*cwords_batch_size + i] = 0;
    }
    loaded_twords++;
  }
  while (loaded_cwords < cwords_batch_size) {
    cwords[loaded_sets*cwords_batch_size + loaded_cwords++] = 0;
  }

  loaded_sets++;
}

void SGDBatchTrainer::calcErrorGradients(int batch_index) {
}

void SGDBatchTrainer::updateWeights(int batch_index) {
}

