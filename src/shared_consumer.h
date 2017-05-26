#ifndef SHARED_CONSUMER_H_
#define SHARED_CONSUMER_H_

#include "common.h"
#include "w2v-functions.h"
#include "sgd_trainer.h"
#include "tc_buffer.h"
#include <vector>
#include "timer.h"

class Trainer {
public:
	SGDTrainer *trainer;
	int state;
	omp_lock_t *indexLoad;
	omp_lock_t *twordsM;
	omp_lock_t *cwordsM;
	omp_lock_t *corrM;
	omp_lock_t *twordsU;
	omp_lock_t *cwordsU;
};

class SharedConsumer {
public:
	int id;
	SharedConsumer(int numt);
	~SharedConsumer();
	int consume();
	int acquire();
	void setTCBuffer(TCBuffer *tcb);
	TCBuffer*  getTCBuffer();

	int prep();
	int train();
	void release();
	int working = 1;
	omp_lock_t *lword_count;
protected:
	TCBuffer *tc_buffer;
	SGDTrainer *trainer;
	vector<Trainer> trainers;
	unsigned long long word_count = 0, last_word_count = 0;
};


#endif
