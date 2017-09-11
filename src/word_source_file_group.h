// Copyright 2017 Trevor Simonton

#ifndef WORD_SOURCE_FILE_GROUP_H_
#define WORD_SOURCE_FILE_GROUP_H_

#include "src/w2v-functions.h"
#include "src/word_source_group.h"
#include "src/word_source_file.h"

class WordSourceFileGroup : public WordSourceGroup {
public:
  WordSourceFileGroup(int num_sources) : WordSourceGroup(num_sources) {}
  virtual void init();
};

#endif
