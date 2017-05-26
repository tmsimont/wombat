#ifndef SGD_TRAINER_H_
#define SGD_TRAINER_H_

#include "common.h"
#include "w2v-functions.h"
#include "tc_buffer.h"
#include "timer.h"

class SGDTargets {
public:
    int *twords;
    int *meta;
		int *labels;
		int *cwords;
    int length;
    int numcwords;

    SGDTargets(int len, int numcw) {
        length = len;
        numcwords = numcw;
        twords = (int *) malloc(length * sizeof(int));
        //meta = (int *) malloc(length * sizeof(int));
        labels = (int *) malloc(length * sizeof(int));
        cwords = (int *) malloc(numcw * sizeof(int));
    }

    ~SGDTargets() {
        free(twords);
        //free(meta);
        free(labels);
        free(cwords);
    }
};

class SGDTrainer {
public:
	int randomness = 1;

	int indices_loaded = 0;
	int cwordsM_loaded = 0;
	int twordsM_loaded = 0;
	int activated = 0;
	int corrM_calculated = 0;
	int cwordsU_calculated = 0;
	int twordsU_calculated = 0;
	int cwordsU_applied = 0;
	int twordsU_applied = 0;

	SGDTrainer();
	~SGDTrainer();
	void loadIndices(TCBufferReader *buffer_item);
	void loadTWords();
	void loadCWords();
	void train();
	void clear();

//protected:
	int *local_tcb_item;

	real *cwordsM;
	real *twordsM;
	real *twordsUpdate;
	real *cwordsUpdate;
	real *corrM;

	virtual void activateHiddenLayer();
	virtual void calculateError();
	virtual void calculateCWordsUpdate();
	virtual void calculateTWordsUpdate();
	virtual void applyCWordsUpdate();
	virtual void applyTWordsUpdate();
	SGDTargets *batch_indices = nullptr;

	int matrix_size;
	int num_twords, num_cwords;
	unsigned long long word_count = 0, last_word_count = 0, next_random = 1;

};

#endif
