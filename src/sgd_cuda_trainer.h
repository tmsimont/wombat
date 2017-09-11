// Copyright 2017 Trevor Simonton

#ifndef SGD_CUDA_TRAINER_H_
#define SGD_CUDA_TRAINER_H_


#include "common.h"
#include "w2v-functions.h"
#include "tc_buffer.h"
#include "sgd_batch_trainer.h"
#include "timer.h"
#include <vector>
#include <cuda_runtime.h>
#include <unordered_map>


#define checkCuda(err) {\
  cudaError_t cet = err;\
  if (cudaSuccess != cet) {\
    printf("%s %d : %s\n", __FILE__, __LINE__, cudaGetErrorString(cet));\
    exit(0);\
  }\
}

class MOP {
public:
  void addTWord(int idx, int label) {
    if (ntw == max_tw) return;
    twords[ntw] = idx;
    labels[ntw] = label;
    ntw++;
  }
  void addCWord(int idx) {
    if (ncw == max_cw) return;
    cwords[ncw] = idx;
    ncw++;
  }
  virtual int numTWords() {
    return max_tw;
  }
  virtual int numCWords() {
    return max_cw;
  }
  virtual void copyTWords(int *dest) {
    for (int i = 0; i < max_tw; ++i)
      dest[i] = twords[i];
  }
  virtual void copyLabels(int *dest) {
    for (int i = 0; i < max_tw; ++i)
      dest[i] = labels[i];
  }
  virtual void copyCWords(int *dest) {
    for (int i = 0; i < max_cw; ++i)
      dest[i] = cwords[i];
  }
protected:
  int ntw = 0, ncw = 0;
  int max_tw, max_cw;
  int *twords, *cwords, *labels;
};

class MOP4x8 : public MOP {
public:
  MOP4x8() {
    max_tw = 4;
    max_cw = 8;
    twords = (int *)malloc(max_tw * sizeof(int));
    labels = (int *)malloc(max_tw * sizeof(int));
    cwords = (int *)malloc(max_cw * sizeof(int));
  }
  ~MOP4x8() {
    free(twords);
    free(labels);
    free(cwords);
  }
};

class VOP {
public:
  int tword, cword, label;
  VOP(int tw, int cw, int lbl) : tword(tw), cword(cw), label(lbl) {}
};

class SGDCUDATrainer : public SGDBatchTrainer {
public:
  SGDCUDATrainer(int num_batches, int batch_size);
  ~SGDCUDATrainer();
  void loadSet(TCBufferReader *buffer_item);
  void train();
  void clear();
  void memtoCUDA();
  std::unordered_map<int, int> used;

  vector<MOP*> wombat;
  vector<VOP> wovbat;

  cudaStream_t *streams;
  cudaEvent_t *memoutEvents;

  int   *d_cwords,
        *d_num_cwords,
        *d_twords,
        *d_num_twords,
        *d_labels;
  float *d_corrWo,
        *d_gradients;

};

void InitNetCUDA(real **Wih, real **Woh);
void InitExpCUDA();
void WiToHost(real **Wih);
void WoToHost(real **Woh);

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
    int max_exp);

#endif
