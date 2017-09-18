// Copyright 2017 Trevor Simonton

#ifndef WORKER_MODEL_H_
#define WORKER_MODEL_H_

#include <vector>

#include "src/w2v-functions.h"
#include "src/common.h"
#include "src/buffers/tc_buffer.h"
#include "src/buffers/sen_buffer.h"
#include "src/buffers/producers/ti_producer.h"
#include "src/buffers/producers/sentence_producer.h"
#include "src/data_source/word_source.h"
#include "src/data_source/word_source_group.h"
#include "src/data_source/word_source_file.h"
#include "src/data_source/word_source_file_group.h"

/**
 * Basic approach to working through the word2vec
 * training process is initialization
 * followed by training. A worker model
 * uses a vector of sentence buffers,
 * TCBuffers and a set of training data to train
 * the word2vec network.
 */
class WorkerModel {
 public:
  vector<SenBuffer *> sbs;
  vector<TCBuffer *> tcbs;
  WordSourceGroup *sources;

  WorkerModel();
  ~WorkerModel();
  void init() {
    initW2V();

    // TODO: better way of setting these...
    // currently externs set from externs....
    sen_buffer_item_size = MAX_SENTENCE_LENGTH + 3;
    tc_buffer_item_size = 4 + (2 * window + 1);

    initWombat();
  }
  void initW2V();
  virtual void initWombat() = 0;
  virtual void train() = 0;
};

/**
 * A worker is a helper to split the worker
 * model tasks into thread-specific
 * encapsulations of data.
 * Each worker will acquire a sentence
 * buffer and a tc buffer, along with a source
 * of training data and load the buffers
 * with data from the training data source.
 */
class Worker {
 public:
  int id, finished = 0;
  SentenceProducer sentenceProducer;
  TIProducer tipro;

  Worker(int id, WorkerModel *m) : id(id), model(m) {}
  WorkerModel *model;

  bool acquireSource(int idx);
  virtual bool trySourceToSenBuffer(int sourceIdx, SenBuffer *sb);
  virtual bool trySenBufferToTCBuffer(TIProducer *tipro,
      SenBuffer *sb,
      TCBuffer *tcb);
  virtual void finish();
  virtual int work() = 0;
};

#endif
