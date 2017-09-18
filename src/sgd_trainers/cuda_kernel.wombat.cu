// Copyright 2017 Trevor Simonton

#include <cuda_fp16.h>

__global__ void Wombat4x8(
    float *Wb,
    float *Wa,
    int *bwords,
    int bwords_start_idx,
    int *awords,
    int awords_start_idx,
    int *labels,
    int hidden_size,
    float alpha,
    int max_exp,
    int hs) {

  int row = threadIdx.y;
  int col = threadIdx.x;
  int batch_index = blockIdx.x;
  int a = 4;
  int b = 8;
  int awords_index = awords_start_idx + batch_index*a;
  int labels_index = awords_start_idx + batch_index*a;
  int bwords_index = bwords_start_idx + batch_index*b;

  extern __shared__ float sw[];
  float *As = &sw[0];
  float *Bs  = &sw[4 * hidden_size];

  // load in local sets of word vectors into As and Bs
  for (int i = 0; i < hidden_size; i += b) {
    if ((i+col) < hidden_size)
      As[(hidden_size*row) + (i+col)] =
        Wa[(hidden_size * awords[awords_index + row]) + (i+col)];
  }
  for (int i = 0; i < hidden_size; i += a) {
    if ((i+row) < hidden_size)
      Bs[(hidden_size*col) + (i+row)] =
        Wb[(hidden_size * bwords[bwords_index + col]) + (i+row)];
  }

  __syncthreads();

  // activate loaded vectors into Cs
  float f = 0;
  for (int i = 0; i < hidden_size; ++i) {
    f += As[(hidden_size*row) + i] * Bs[col*hidden_size + i];
  }
  if (hs == 1) {
    if (f >= max_exp) {
      f = 0;
    } else if (f <= -max_exp) {
      f = 0;
    } else {
      f = exp(f);
      f = f / (1.0f + f);
      f = (1.0f - labels[labels_index + row] - f) * alpha;
    }
  } else {
    if (f > max_exp) {
      f = (labels[labels_index + row] - 1) * alpha;
    } else if (f < -max_exp) {
      f = labels[labels_index + row] * alpha;
    } else {
      f = exp(f);
      f = f / (1.0f + f);
      f = (labels[labels_index + row] - f) * alpha;
    }
  }

  for (int i = 0; i < hidden_size; i++) {
    // calculate local update for this thread
    float uA = f * Bs[col*hidden_size + i];
    float uB = f * As[row*hidden_size + i];

    // update column of B
    uB += __shfl_down(uB, 16);
    uB += __shfl_down(uB, 8);
    if (row == 0) {
      atomicAdd(
          Wb + (hidden_size * bwords[bwords_index + col]) + i,
          uB);
    }

    // update column of A
    uA += __shfl_down(uA, 4, 8);
    uA += __shfl_down(uA, 2, 8);
    uA += __shfl_down(uA, 1, 8);
    if (col == 0) {
      atomicAdd(
          Wa + (hidden_size * awords[awords_index + row]) + i,
          uA);
    }
  }
}

__global__ void VectorTrain(
    float *Wb,
    float *Wa,
    int *bwords,
    int bwords_start_idx,
    int *awords,
    int awords_start_idx,
    int *labels,
    int hidden_size,
    float alpha,
    int max_exp,
    int B_start,
    int hs) {

  int batch_index = blockIdx.x;
  int awords_index = awords_start_idx + batch_index;
  int labels_index = awords_start_idx + batch_index;
  int bwords_index = bwords_start_idx + batch_index;

  extern __shared__ float sv[];
  float *A1s = &sv[0];
  float *Bs  = &sv[B_start];

  float f = 0;
  for (int i = 0; i < hidden_size / 32; i++) {
    A1s[i+threadIdx.x*hidden_size/32] =
      Wa[(hidden_size * awords[awords_index]) + i + threadIdx.x*hidden_size/32];
    Bs[i+threadIdx.x*hidden_size/32] =
      Wb[(hidden_size * bwords[bwords_index]) + i + threadIdx.x*hidden_size/32];
  }

  __syncthreads();

  for (int i = 0; i < hidden_size / 32; i++) {
    f += A1s[i + threadIdx.x*hidden_size/32]
      * Bs[i + threadIdx.x*hidden_size/32];
  }
  #pragma unroll
  for (int i = 16; i > 0; i /= 2) {
    f += __shfl_down(f, i);
  }
  if (threadIdx.x == 0) {
    if (hs == 1) {
      if (f >= max_exp) {
        f = 0;
      } else if (f <= -max_exp) {
        f = 0;
      } else {
        f = exp(f);
        f = f / (1.0f + f);
        f = (1.0f - labels[labels_index] - f) * alpha;
      }
    } else {
      if (f > max_exp) {
        f = (labels[labels_index] - 1) * alpha;
      } else if (f < -max_exp) {
        f = labels[labels_index] * alpha;
      } else {
        f = exp(f);
        f = f / (1.0f + f);
        f = (labels[labels_index] - f) * alpha;
      }
    }
  }

  f = __shfl(f, 0);


  // Calculate and apply updates
  for (int i = 0; i < hidden_size/32; i++) {
    atomicAdd(
        Wa + (hidden_size * awords[awords_index])
          + i+threadIdx.x*hidden_size/32,
        f * Bs[i+threadIdx.x*hidden_size/32]);
    atomicAdd(
        Wb + (hidden_size * bwords[bwords_index])
          + i+threadIdx.x*hidden_size/32,
        f * A1s[i+threadIdx.x*hidden_size/32]);
  }
}
