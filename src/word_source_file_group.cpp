#include "word_source_file_group.h"

void WordSourceFileGroup::init() {
	// TODO eliminate global references to vars in w2v-functions.h
	unsigned long long chunkSize = file_size / (long long)num_sources;
	for (int i = 0; i < num_sources; ++i) {
		sources.push_back(new WordSourceFile(i, iter, chunkSize, train_file));
	}

}
