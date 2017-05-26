#ifndef WORKERS_CONSUMER_H_
#define WORKERS_CONSUMER_H_

#include "common.h"
#include "w2v-functions.h"
#include "sgd_trainer.h"
#include "tc_buffer.h"
#include <vector>
#include "timer.h"

class Consumer {
public:
	int id;
	Consumer();
	~Consumer();
	virtual int consume();
	virtual int acquire();
	void setTCBuffer(TCBuffer *tcb);
	TCBuffer*  getTCBuffer();
protected:
	int has_item = 0;
	int *local_item;
	TCBuffer *tc_buffer;
	SGDTrainer *trainer;
	unsigned long long word_count = 0, last_word_count = 0;
};


#endif
