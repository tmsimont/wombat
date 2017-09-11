// Copyright 2017 Trevor Simonton

#ifndef SGD_MKL_TRAINER_H_
#define SGD_MKL_TRAINER_H_

#include "src/common.h"
#include "src/w2v-functions.h"
#include "src/tc_buffer.h"
#include "src/timer.h"
#include "src/mkl.h"
#include "src/sgd_trainer.h"

class SGDMKLTrainer : public SGDTrainer {
public:
  virtual void activateHiddenLayer();
  virtual void calculateError();
  virtual void calculateCWordsUpdate();
  virtual void calculateTWordsUpdate();
};

#endif
