// Copyright 2017 Trevor Simonton

#include "src/sgd_trainers/sgd_mkl_trainer.h"

void SGDMKLTrainer::activateHiddenLayer() {
  // cwordsM x (twordsM)T
  // Wih x Woh for set of target/context
  // pairs to activate target/context interaction
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      num_twords, num_cwords, hidden_size, 1.0f, twordsM,
      hidden_size, cwordsM, hidden_size, 0.0f, corrM, num_cwords);
}

void SGDMKLTrainer::calculateError() {
  // error(softmax(above result))
  // This yields a float between -1 and 1 for the
  // adjustment required for each target/context vector
  if (hs) {
    for (int i = 0; i < num_twords; i++) {
      int offset = i * num_cwords;
      #pragma simd
      for (int j = 0; j < num_cwords; j++) {
        real f = corrM[offset + j], g;
        int label = batch_indices->labels[i];
        if (f >= MAX_EXP) {
          g = 0;
        } else if (f <= -MAX_EXP) {
          g = 0;
        } else {
          f = expTable[static_cast<int>(
              (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          g = (1 - label - f) * alpha;
        }
        corrM[offset + j] = g;
      }
    }
  } else {
    for (int i = 0; i < num_twords; i++) {
      int c = 1;
      int offset = i * num_cwords;
      #pragma simd
      for (int j = 0; j < num_cwords; j++) {
        real f = corrM[offset + j];
        int label = batch_indices->labels[i];
        if (f > MAX_EXP) {
          f = (label - 1) * alpha;
        } else if (f < -MAX_EXP) {
          f = label * alpha;
        } else {
          f = (label - expTable[static_cast<unsigned int>(
                ((f + MAX_EXP) * EXP_RESOLUTION)]) * alpha;
        }
        corrM[offset + j] = f * c;
      }
    }
  }
}

void SGDMKLTrainer::calculateCWordsUpdate() {
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      num_cwords, hidden_size, num_twords, 1.0f, corrM,
      num_cwords, twordsM, hidden_size, 0.0f,
      cwordsUpdate, hidden_size);
}

void SGDMKLTrainer::calculateTWordsUpdate() {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      num_twords, hidden_size, num_cwords, 1.0f, corrM,
      num_cwords, cwordsM, hidden_size, 0.0f,
      twordsUpdate, hidden_size);
}
