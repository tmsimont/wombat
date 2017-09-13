// Copyright 2017 Trevor Simonton

#include "src/batch_consumer.h"

BatchConsumer::BatchConsumer(SGDBatchTrainer *trainer) {
  this->trainer = trainer;
  trainer->clear();
}
BatchConsumer::~BatchConsumer() {
  delete trainer;
}

int BatchConsumer::acquire() {
  if (acquired == trainer->numBatches() * trainer->batchSize()) {
    return -1;
  }
  TCBufferReader tc_reader;
  int got_item = tc_buffer->getReadyItem(&tc_reader);
  if (got_item) {
    trainer->loadSet(&tc_reader);
    word_count += 1 + tc_reader.droppedWords();
    acquired++;
    return 1;
  }
  return 0;
}

int BatchConsumer::consume() {
  if (acquired == trainer->numBatches() * trainer->batchSize()) {
    trainer->train();
    trainer->clear();
    acquired = 0;
    if (word_count - last_word_count > 10000) {
      #pragma omp atomic
      word_count_actual += word_count - last_word_count;

      last_word_count = word_count;
      if (debug_mode > 1) {
        double now = omp_get_wtime();
        printf("\rAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk",  alpha,
                word_count_actual / (real) (iter * train_words + 1) * 100,
                word_count_actual / ((now - start) * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha
        * (1 - word_count_actual / (real) (iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001f) {
        alpha = starting_alpha * 0.0001f;
      }
    }
    return 1;
  }
  return 0;
}


void BatchConsumer::setTCBuffer(TCBuffer *tcb) {
  tc_buffer = tcb;
}
TCBuffer* BatchConsumer::getTCBuffer() {
  return tc_buffer;
}
