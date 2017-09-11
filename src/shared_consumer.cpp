// Copyright 2017 Trevor Simonton

#include "src/shared_consumer.h"

#ifdef USE_MKL
#include "src/sgd_mkl_trainer.h"
#endif

SharedConsumer::SharedConsumer(int numt) {
  lword_count = (omp_lock_t *) malloc(sizeof(omp_lock_t));
  omp_init_lock(lword_count);
  for (int i = 0; i < numt; ++i) {
    Trainer t;
#ifdef USE_MKL
    t.trainer = new SGDMKLTrainer();
#else
    t.trainer = new SGDTrainer();
#endif
    t.indexLoad = (omp_lock_t *) malloc(sizeof(omp_lock_t));
    t.twordsM = (omp_lock_t *) malloc(sizeof(omp_lock_t));
    t.cwordsM = (omp_lock_t *) malloc(sizeof(omp_lock_t));
    t.corrM = (omp_lock_t *) malloc(sizeof(omp_lock_t));
    t.twordsU = (omp_lock_t *) malloc(sizeof(omp_lock_t));
    t.cwordsU = (omp_lock_t *) malloc(sizeof(omp_lock_t));
    omp_init_lock(t.indexLoad);
    omp_init_lock(t.twordsM);
    omp_init_lock(t.cwordsM);
    omp_init_lock(t.corrM);
    omp_init_lock(t.twordsU);
    omp_init_lock(t.cwordsU);
    t.state = 0;
    trainers.push_back(t);
  }
}

void SharedConsumer::release() {
  //if (is_producer) omp_unset_lock(lock1);
  working = 0;
}

int SharedConsumer::acquire() {
  return 0;
}

int SharedConsumer::prep() {
  /*
  int s = 1;
  for (auto &t : trainers) {
    if (omp_test_lock(t.lockt)) {
      if (t.state == 1) {
        t.trainer->train();
        t.state = 2;
        s = 1;
      }
      if (t.state == 0) {
        TCBufferReader tc_reader;
        int got_item = tc_buffer->getReadyItem(&tc_reader);
        if (got_item) {
          t.trainer->clear();
          t.trainer->loadIndices(&tc_reader);
          t.state = 1;
          s = 1;
        }
        else {
          s = 0;
        }
      }
      if (t.state == 2) {
        //t.trainer->apply();
        t.state = 0;
        s = 1;
      }
      omp_unset_lock(t.lockt);
    }
  }
  return s;
  */
  return 1;
}

int SharedConsumer::train() {
  /*
  for (auto &t : trainers) {
    if (omp_test_lock(t.lockt)) {
      if (t.state == 1) {
        t.trainer->train();
        t.state = 2;
      }
      omp_unset_lock(t.lockt);
    }
  }
  return working;
  */
  return 1;
}

//  TASK
int SharedConsumer::consume() {
  int s = working;
  #pragma omp single nowait
  for (auto &t : trainers) {
    omp_set_lock(t.indexLoad);
    if (!t.trainer->indices_loaded) {
      // now get a fresh tcb
      TCBufferReader tc_reader;
      tc_buffer->setLock();
      int got_item = tc_buffer->getReadyItem(&tc_reader);
      if (got_item) {
        t.trainer->loadIndices(&tc_reader);
        omp_set_lock(lword_count);
        word_count += 1 + tc_reader.droppedWords();
        omp_unset_lock(lword_count);
        s = 1;
      }
      else {
        s = 0;
      }
      tc_buffer->unsetLock();
    }
    omp_unset_lock(t.indexLoad);
  }

  #pragma omp single
  for (unsigned int i = 0; i < trainers.size(); ++i) {

    #pragma omp task 
    {
      omp_set_lock(trainers[i].indexLoad);
      if (trainers[i].trainer->indices_loaded) {
      //#pragma omp task 
      trainers[i].trainer->loadCWords();
      //#pragma omp task 
      trainers[i].trainer->loadTWords();

      //#pragma omp taskwait

      // only one
      trainers[i].trainer->activateHiddenLayer();
      trainers[i].trainer->calculateError();


      // split
      //#pragma omp task 
      {
        trainers[i].trainer->calculateCWordsUpdate();
        trainers[i].trainer->applyCWordsUpdate();
      }
      //#pragma omp task 
      {
        trainers[i].trainer->calculateTWordsUpdate();
        trainers[i].trainer->applyTWordsUpdate();
      }

      //#pragma omp taskwait

      // only one
      trainers[i].trainer->clear();
      omp_set_lock(lword_count);
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
      omp_unset_lock(lword_count);
      }
      omp_unset_lock(trainers[i].indexLoad);
    }
  }
  if (!working)
    return 0;
  return s;
}

/* sections
int SharedConsumer::consume() {
  int s = working;
  auto &t = trainers[0];
  #pragma omp barrier
  #pragma omp single
  { 
    if (!t.trainer->indices_loaded) {
      // now get a fresh tcb
      TCBufferReader tc_reader;
      int got_item = tc_buffer->getReadyItem(&tc_reader);
      if (got_item) {
        t.trainer->loadIndices(&tc_reader);
        word_count += 1 + tc_reader.droppedWords();
        s = 1;
      }
      else {
        s = 0;
      }
    }
  }

  if (t.trainer->indices_loaded) {

  #pragma omp sections
  {
    #pragma omp section 
    t.trainer->loadCWords();
    #pragma omp section 
    t.trainer->loadTWords();
  }

  #pragma omp single
  {
    t.trainer->activateHiddenLayer();
    t.trainer->calculateError();
  }

  #pragma omp sections
  {
      #pragma omp section 
      t.trainer->calculateCWordsUpdate();
      #pragma omp section 
      t.trainer->calculateTWordsUpdate();
  }
      
  #pragma omp sections
  {
      #pragma omp section 
      t.trainer->applyCWordsUpdate();
      #pragma omp section 
      t.trainer->applyTWordsUpdate();
  }
  }

  #pragma omp single
  {
      t.trainer->clear();
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
  }

  if (!working)
    return 0;
  return s;
}
*/

void SharedConsumer::setTCBuffer(TCBuffer *tcb) {
  tc_buffer = tcb;
}
TCBuffer* SharedConsumer::getTCBuffer() {
  return tc_buffer;
}
