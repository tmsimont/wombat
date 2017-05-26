#ifndef SGD_BATCH_TRAINER_H_
#define SGD_BATCH_TRAINER_H_

#include "common.h"
#include "w2v-functions.h"
#include "tc_buffer.h"
#include "timer.h"
#include <vector>

class SGDBatchTrainer {
public:
	int randomness = 1;
	SGDBatchTrainer() = default;
	SGDBatchTrainer(int num_batches, int batch_size);
	~SGDBatchTrainer();
	virtual void loadSet(TCBufferReader *buffer_item);
	virtual void train();
	virtual void clear();
	int numBatches() {return num_batches;}
	int batchSize() {return batch_size;}
//private:

	int		num_batches,
				batch_size,
				num_streams,
				loaded_sets;

	long	cwords_bytes,
				num_cwords_bytes,
				twords_bytes,
				num_twords_bytes,
				labels_bytes,
				corrWo_bytes,
				gradients_bytes;

	int		cwords_batch_size,
				twords_batch_size,
				labels_batch_size;

	int   *cwords,
			  *num_cwords,
			  *twords,
			  *num_twords,
			  *labels;
	float	*corrWo,
				*gradients;

	unsigned long long word_count = 0, last_word_count = 0, next_random = 1;

	virtual void calcErrorGradients(int batch_index);
	virtual void updateWeights(int batch_index);
};


#endif
