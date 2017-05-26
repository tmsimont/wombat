#ifndef SGD_MKL_TRAINER_H_
#define SGD_MKL_TRAINER_H_

#include "common.h"
#include "w2v-functions.h"
#include "tc_buffer.h"
#include "timer.h"
#include "mkl.h"
#include "sgd_trainer.h"

class SGDMKLTrainer : public SGDTrainer {
public:
	virtual void activateHiddenLayer();
	virtual void calculateError();
	virtual void calculateCWordsUpdate();
	virtual void calculateTWordsUpdate();
};

#endif
