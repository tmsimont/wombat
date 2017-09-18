// Copyright 2017 Trevor Simonton

#ifndef WORD_SOURCE_GROUP_H_
#define WORD_SOURCE_GROUP_H_

#include <omp.h>

#include <vector>

#include "src/common.h"
#include "src/data_source/word_source.h"

class WordSourceGroup {
 public:
  explicit WordSourceGroup(int num_sources);
  void useLocks(bool use);
  WordSource* at(int idx);
  void release(int idx);
  bool isActive(int idx);
  void setInactive(int idx);
  virtual void init() = 0;
  int numSources() { return num_sources; }
  bool hasActiveSource();
 protected:
  bool use_locks = false;
  int num_sources;
  int num_active;
  vector<WordSource *> sources;
  vector<int> activeList;
  vector<omp_lock_t *> source_locks;
  omp_lock_t *activeCountLock;
};

#endif
