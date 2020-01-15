// Copyright 2017 Trevor Simonton

#include <vector>
#include "src/sgd_trainers/cuda_kernel.wombat.cu"

void CallKernels(int hs, int wombat_size, int wovbat_size, cudaStream_t* stream,
    float *d_Wih,
    float *d_Woh,
    int *d_cwords,
    int bwords_start_idx,
    int *d_twords,
    int awords_start_idx,
    int *d_labels,
    int labels_batch_size,
    int hidden_size,
    float alpha,
    int max_exp) {

  dim3 wombatBlock(8, 4);
  if (wombat_size > 0) {
    Wombat4x8<<<wombat_size, wombatBlock,
      (hidden_size * 32) * sizeof(float), *stream>>>(
        d_Wih,
        d_Woh,
        d_cwords,
        0,
        d_twords,
        0,
        d_labels,
        hidden_size,
        alpha,
        max_exp,
        hs);
  }

  dim3 vectorBlock(32, 1);
  if (wovbat_size > 0) {
    VectorTrain<<<wovbat_size, vectorBlock,
      (hidden_size * 2) * sizeof(float), *stream>>>(
        d_Wih,
        d_Woh,
        d_cwords,
        wombat_size * 8,
        d_twords,
        wombat_size * 4,
        d_labels,
        hidden_size,
        alpha,
        max_exp,
        hidden_size,
        hs);
  }
}
