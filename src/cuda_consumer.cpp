#ifdef USE_CUDA
#include "cuda_consumer.h"

CUDAConsumer::CUDAConsumer(int num_batches, int batch_size) {
	this->num_batches = num_batches;
	this->batch_size = batch_size;
	trainer = new SGDCUDATrainer(num_batches, batch_size);
	trainer->clear();
}
CUDAConsumer::~CUDAConsumer() {
	delete trainer;
}

int CUDAConsumer::acquire() {
	if (acquired == num_batches * batch_size) {
		return -1;
	}
	TCBufferReader tc_reader;
	int got_item = tc_buffer->getReadyItem(&tc_reader);
	if (got_item) {
		trainer->loadSet(&tc_reader);
		word_count += 1 + tc_reader.droppedWords();
		acquired++;
		return 1;
	}
	return 0;
}

int CUDAConsumer::consume() {
	if (acquired == num_batches * batch_size) {
		trainer->train();
		trainer->clear();
		acquired = 0;
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


void CUDAConsumer::setTCBuffer(TCBuffer *tcb) {
	tc_buffer = tcb;
}
TCBuffer* CUDAConsumer::getTCBuffer() {
	return tc_buffer;
}

#endif
