// Copyright 2017 Trevor Simonton

#include "src/sgd_trainer.h"


SGDTrainer::SGDTrainer() {
  if (hs)
      matrix_size = MAX_CODE_LENGTH;
  else
      matrix_size = negative + 1;
  int max_cwords = (2 * window + 1);

  posix_memalign((void **)&cwordsM, 64, max_cwords * hidden_size * sizeof(real));
  posix_memalign((void **)&twordsM, 64, matrix_size * hidden_size * sizeof(real));
  posix_memalign((void **)&twordsUpdate, 64, matrix_size * hidden_size * sizeof(real));
  posix_memalign((void **)&cwordsUpdate, 64, max_cwords * hidden_size * sizeof(real));
  posix_memalign((void **)&corrM, 64, matrix_size * max_cwords * sizeof(real));
  batch_indices = new SGDTargets(matrix_size, max_cwords);

  indices_loaded = 0;
}

SGDTrainer::~SGDTrainer() {
  free(cwordsM);
  free(twordsM);
  free(twordsUpdate);
  free(cwordsUpdate);
  free(corrM);
}

void SGDTrainer::train() {

  if (num_twords == 0 || num_cwords == 0) return;

  activateHiddenLayer();
  calculateError();
  calculateCWordsUpdate();
  calculateTWordsUpdate();
  applyCWordsUpdate();
  applyTWordsUpdate();

}

void SGDTrainer::loadIndices(TCBufferReader *tc_reader) {

  int offset = 0;
  if (hs) {
    int target = tc_reader->targetWord();
    for (int k = 0; k < vocab[target].codelen; k++) {
        batch_indices->twords[offset] = vocab[target].point[k];
        batch_indices->labels[offset] = vocab[target].code[k];
        offset++;
    }
  }
  else {
    int target = tc_reader->targetWord();
    //int tidx = offset;
    batch_indices->twords[offset] = target;
    batch_indices->labels[offset] = 1;
    //batch_indices->meta[offset] = 1;
    offset++;

    // generate negative samples for output layer
    for (int k = 0; k < negative; k++) {
      int sample;
      if (randomness)  {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        sample = table[(next_random >> 16) % table_size];
        if (!sample)
          sample = next_random % (vocab_size - 1) + 1;
      }
      else {
        next_random = (next_random + 20) % vocab_size;
        sample = next_random;
      }
      //int* p = find(batch_indices->twords, batch_indices->twords + offset, sample);
      //if (p == batch_indices->twords + offset) {
        batch_indices->twords[offset] = sample;
        batch_indices->labels[offset] = 0;
        //batch_indices->meta[offset] = 1;
        offset++;
      //} else {
        //int idx = p - batch_indices->twords;
        //batch_indices->meta[idx]++;
      //}
    }
    //batch_indices->meta[tidx] = 1;
  }
  num_twords = batch_indices->length = offset;


  for (int j = 0; j < tc_reader->numCWords(); ++j) {
    batch_indices->cwords[j] = *(tc_reader->cwords() + j);
  }
  num_cwords = batch_indices->numcwords = tc_reader->numCWords();

  indices_loaded = 1;
}

void SGDTrainer::loadCWords() {
  for (int i = 0; i < num_cwords; i++) {
    memcpy(cwordsM + i * hidden_size, Wih + batch_indices->cwords[i] * hidden_size, hidden_size * sizeof(real));
  }
}
void SGDTrainer::loadTWords() {
  for (int i = 0; i < num_twords; i++) {
    memcpy(twordsM + i * hidden_size, Woh + batch_indices->twords[i] * hidden_size, hidden_size * sizeof(real));
  }
}

void SGDTrainer::clear() {
  num_cwords = 0;
  num_twords = 0;
  indices_loaded = 0;
  cwordsM_loaded = 0;
  twordsM_loaded = 0;
  activated = 0;
  corrM_calculated = 0;
  cwordsU_calculated = 0;
  twordsU_calculated = 0;
  cwordsU_applied = 0;
  twordsU_applied = 0;
}

void SGDTrainer::activateHiddenLayer() {
  if (hs) {
    for (int i = 0; i < num_twords; i++) {
      for (int j = 0; j < num_cwords; j++) {
        real f = 0.f, g;
        #pragma simd
        for (int k = 0; k < hidden_size; k++) {
          f += twordsM[i * hidden_size + k] * cwordsM[j * hidden_size + k];
        }
        if (f >= MAX_EXP)
          g = 0;
        else if (f <= -MAX_EXP)
          g = 0;
        else {
          f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          g = (1 - batch_indices->labels[i] - f) * alpha;
        }
        corrM[i * num_cwords + j] = g;
      }
    }
  }
  else {
    for (int i = 0; i < num_twords; i++) {
      //int c = batch_indices->meta[i];
      for (int j = 0; j < num_cwords; j++) {
        real f = 0.f, g;
        #pragma simd
        for (int k = 0; k < hidden_size; k++) {
            f += twordsM[i * hidden_size + k] * cwordsM[j * hidden_size + k];
        }
        int label = (i ? 0 : 1);
        if (f > MAX_EXP)
          g = (label - 1) * alpha;
        else if (f < -MAX_EXP)
          g = label * alpha;
        else 
          g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        corrM[i * num_cwords + j] = g;
      }
    }
  }
}

void SGDTrainer::calculateError() {
  // calculated in activateHiddenLayer()
}

void SGDTrainer::calculateCWordsUpdate() {
  for (int i = 0; i < num_cwords; i++) {
    for (int j = 0; j < hidden_size; j++) {
      real f = 0.f;
      #pragma simd
      for (int k = 0; k < num_twords; k++) {
        f += corrM[k * num_cwords + i] * twordsM[k * hidden_size + j];
      }
      cwordsUpdate[i * hidden_size + j] = f;
      //cwordsM[i * hidden_size + j] = f;
    }
  }
}

void SGDTrainer::calculateTWordsUpdate() {
  // Apply error-based adjustment to context words to yield update for target words in Woh
  for (int i = 0; i < num_twords; i++) {
    for (int j = 0; j < hidden_size; j++) {
      real f = 0.f;
      #pragma simd
      for (int k = 0; k < num_cwords; k++) {
        f += corrM[i * num_cwords + k] * cwordsM[k * hidden_size + j];
      }
      twordsUpdate[i * hidden_size + j] = f;
    }
  }
}

void SGDTrainer::applyCWordsUpdate() {
  // update Wih
  for (int j = 0; j < batch_indices->numcwords; ++j) {
    int src = j * hidden_size;
    int des = batch_indices->cwords[j] * hidden_size;
    #pragma simd
    for (int k = 0; k < hidden_size; k++) {
      Wih[des + k] += cwordsUpdate[src + k];
      //Wih[des + k] += .01;
    }
  }
}

void SGDTrainer::applyTWordsUpdate() {
  // update Woh
  for (int i = 0; i < num_twords; i++) {
    int src = i * hidden_size;
    int des = batch_indices->twords[i] * hidden_size;
    #pragma simd
    for (int j = 0; j < hidden_size; j++) {
      Woh[des + j] += twordsUpdate[src + j];
      //Woh[des + j] += .01;
    }
  }
}
