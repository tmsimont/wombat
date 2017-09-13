// Copyright 2017 Trevor Simonton

#include "src/console.h"
#include "src/pht_model.h"


int main(int argc, char **argv) {
  if (readConsoleArgs(argc, argv)) {
    PHTModel t;
    t.init();
    t.train();

    double now = omp_get_wtime();
    printf("\nFinal: Alpha: %f  Progress: %.2f%%  Words/sec: %.2fk\n",  alpha,
            word_count_actual / (real) (iter * train_words + 1) * 100,
            word_count_actual / ((now - start) * 1000));
    saveModel();
  }
  return 0;
}
