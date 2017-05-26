#ifndef WORD_SOURCE_FILE_GROUP_H_
#define WORD_SOURCE_FILE_GROUP_H_

#include "w2v-functions.h"
#include "word_source_group.h"
#include "word_source_file.h"

class WordSourceFileGroup : public WordSourceGroup {
public:
	WordSourceFileGroup(int num_sources) : WordSourceGroup(num_sources) {}
	virtual void init();
};

#endif
