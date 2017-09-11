#include "consumer.h"

#ifdef USE_MKL
#include "sgd_mkl_trainer.h"
Consumer::Consumer() {
  trainer = new SGDMKLTrainer();
  local_item = (int *) calloc(tc_buffer_item_size, sizeof(int));
}
#else
Consumer::Consumer() {
  trainer = new SGDTrainer();
  local_item = (int *) calloc(tc_buffer_item_size, sizeof(int));
}
#endif

Consumer::~Consumer() {
  delete trainer;
  free(local_item);
}

int Consumer::acquire() {
  TCBufferReader tc_reader;
  int got_item = tc_buffer->getReadyItem(&tc_reader);
  if (got_item) {
    memcpy(local_item, tc_reader.getData(), tc_buffer_item_size * sizeof(int));
    has_item = 1;
    return 1;
  }
  return 0;
}

int Consumer::consume() {
  if (has_item) {
    

    trainer->clear();
    TCBufferReader tc_reader(local_item);
    trainer->loadIndices(&tc_reader);
    trainer->loadTWords();
    trainer->loadCWords();
    trainer->train();
    word_count += 1 + tc_reader.droppedWords();
    if (word_count - last_word_count > 10000) {
      #pragma omp atomic
      word_count_actual += word_count - last_word_count;

      last_word_count = word_count;
      if (debug_mode > 1) {
        if (time_events) {
          double now = omp_get_wtime();
          printf("Alpha: %f  Progress: %.2f%%  Words/sec: %.2fk\n",  alpha,
              word_count_actual / (real) (iter * train_words + 1) * 100,
              word_count_actual / ((now - start) * 1000));
          printTimers();
          //fflush(stdout);
        }
        else {
          double now = omp_get_wtime();
          printf("\rAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk",  alpha,
              word_count_actual / (real) (iter * train_words + 1) * 100,
              word_count_actual / ((now - start) * 1000));
          fflush(stdout);
        }
      }
      alpha = starting_alpha * (1 - word_count_actual / (real) (iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001f)
        alpha = starting_alpha * 0.0001f;
    }

    return 1;
  }
  return 0;
}

void Consumer::setTCBuffer(TCBuffer *tcb) {
  tc_buffer = tcb;
}
TCBuffer* Consumer::getTCBuffer() {
  return tc_buffer;
}
