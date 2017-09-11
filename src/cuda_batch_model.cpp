// Copyright 2017 Trevor Simonton

#include "src/cuda_batch_model.h"

void CUDABatchModel::initWombat() {
  BatchModel::initWombat();
  cudaSetDevice(0);
  InitNetCUDA(&Wih, &Woh);
  InitExpCUDA();
}
void CUDABatchModel::train() {
  #pragma omp parallel num_threads(num_threads)
  {
    int id = omp_get_thread_num();
    BatchWorker worker(id, this);

    cudaSetDevice(0);

    #pragma omp barrier
    if (id == 0) {
      start = omp_get_wtime();
    }
    #pragma omp barrier

    worker.work();
  }
  cudaDeviceSynchronize();
  WiToHost(&Wih);
}

SGDBatchTrainer* CUDABatchModel::getTrainer() {
  return new SGDCUDATrainer(batches_per_thread, batch_size);
}

