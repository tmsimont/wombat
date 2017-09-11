// Copyright 2017 Trevor Simonton

#ifndef WORKER_MODEL_H_
#define WORKER_MODEL_H_

#include "w2v-functions.h"
#include "common.h"
#include "tc_buffer.h"
#include "sen_buffer.h"
#include "ti_producer.h"
#include "sentence_producer.h"
#include "word_source.h"
#include "word_source_group.h"
#include "word_source_file.h"
#include "word_source_file_group.h"
#include <vector>
#include "timer.h"

class WorkerModel {
public:
  vector<SenBuffer *> sbs;
  vector<TCBuffer *> tcbs;
  WordSourceGroup *sources;

  WorkerModel();
  ~WorkerModel();
  void init() {
    initW2V();

    // TODO: better way of setting these... currently externs set from externs....
    sen_buffer_item_size = MAX_SENTENCE_LENGTH + 3;
    tc_buffer_item_size = 4 + (2 * window + 1);

    initWombat();
  }
  void initW2V();
  virtual void initWombat() = 0 ;
  virtual void train() = 0;
};

class Worker {
public:
  int  id, finished = 0;

  SentenceProducer senpro;
  TIProducer tipro;

  Worker(int id, WorkerModel *m) : id(id), model(m) {}
  WorkerModel *model;

  bool acquireSource(int idx);
  virtual bool trySourceToSenBuffer(int sourceIdx, SenBuffer *sb);
  virtual bool trySenBufferToTCBuffer(TIProducer *tipro, SenBuffer *sb, TCBuffer *tcb);
  virtual void finish();

  virtual int work() = 0;
};


void Train();

#endif
