#include "worker_model.h"


WorkerModel::WorkerModel() {
}

WorkerModel::~WorkerModel() {
	// TODO: free memory
}

void WorkerModel::initW2V() {
	// init w2v structs...
	vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (int i = 0; i < EXP_TABLE_SIZE + 1; i++) {
		expTable[i] = exp((i / (real) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                    // Precompute f(x) = x / (x + 1)
	}

	// w2v-style input
	if (read_vocab_file[0] != 0) {
		ReadVocab();
	}
	else {
		LearnVocabFromTrainFile();
	}
	if (save_vocab_file[0] != 0) SaveVocab();

	// w2v-style initializations for nets and sample table
	InitNet();
	InitUnigramTable();
}

bool Worker::acquireSource(int idx) {
	if (!model->sources->isActive(idx)) return false;
	WordSource *source = model->sources->at(idx);
	if (source == nullptr) {
		// source is locked
		return false;
	}

	else if (source->iterationsRemaining() <= 0) {
		model->sources->setInactive(idx);
		model->sources->release(idx);
		return false;
	}

	else {
		senpro.setSource(source);
		return true;
	}
}

bool Worker::trySourceToSenBuffer(int sourceIdx, SenBuffer *sen_buffer) {
	bool success = false;
	if (acquireSource(sourceIdx)) {
		if (!sen_buffer->isFull()) {
			SenBufferReader sen_reader;
			if (sen_buffer->getEmptyItem(&sen_reader)) {
				if (senpro.buildSentence(&sen_reader)) {
					success = true;
				}
				else {
					// word source iterations exhausted
					model->sources->setInactive(sourceIdx);

					// TODO: deal with the fact that sen_reader is a 
					// reserved slot in sen_buffer and now isn't going to 
					// populate its data...
					sen_reader.markEmpty();
				}
			}
		}
		model->sources->release(sourceIdx);
	}
	return success;
}

bool Worker::trySenBufferToTCBuffer(TIProducer *tipro, SenBuffer *sen_buffer, TCBuffer *tc_buffer) {
	bool success = false;

	// get sentence from sen buffer
	if (!tipro->hasSentence()) {
		if (!tipro->loadSentence(sen_buffer)) {
			// unable to get sentence from sen_buffer
			return false;
		}
	}

	// try to generate tc items to tcbuffer
	tc_buffer->setLock();
	if (!tc_buffer->isFull()) {
		while (tipro->hasSentence() && !tc_buffer->isFull()) {
			TCBufferReader tc_reader;
			if (tc_buffer->getEmptyItem(&tc_reader)) {
				tipro->buildTI(&tc_reader);
				success = true;
			}
			else {
				// ran out of room in tcbuffer
				success = false;
			}
		}
	}
	tc_buffer->unsetLock();

	return success;
}

void Worker::finish() {
	finished = 1;
}
