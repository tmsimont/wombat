#include "ti_producer.h"

bool TIProducer::loadSentence(SenBuffer *sen_buffer) {
	if (!sen_buffer->isEmpty()) {
		if (sen_buffer->getReadyItem(&sen_reader)) {
			sentenceLoaded = true;
			return true;
		}
	}
	return false;
}

bool TIProducer::hasSentence() {
	if (sentenceLoaded)
	 	return sen_reader.length() > 0;
	return false;
}

void TIProducer::buildTI(TCBufferReader *tc_reader) {
	tc_reader->setNumCWords(0);

	// randomly get sample of context words around the target
	next_random = next_random * (unsigned long long)25214903917 + 11;
	// TODO: get away from w2v externs
	int b = next_random % window;
	for (int i = b; i < 2 * window + 1 - b; i++) {
		if (i != window) {
			int c = sen_reader.position() - window + i;
			if (c < 0)
				continue;
			if (c >= sen_reader.length())
				break;
			tc_reader->cwords()[tc_reader->numCWords()] = sen_reader.sen()[c];
			tc_reader->incNumCWords();
		}
	}

	tc_reader->setTargetWord(sen_reader.sen()[sen_reader.position()]);

	// pass dropped word count onto tc buffer item
	tc_reader->setDroppedWords(sen_reader.droppedWords());
	sen_reader.setDroppedWords(0);

	// increment position in sentence
	sen_reader.incPosition();
	if (sen_reader.position() >= sen_reader.length()) {
		sen_reader.setPosition(0);
		sen_reader.setLength(0);
		sentenceLoaded = false;
	}
}

