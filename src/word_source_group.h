#ifndef WORD_SOURCE_GROUP_H_
#define WORD_SOURCE_GROUP_H_

#include "common.h"
#include "word_source.h"
#include <vector>
#include "omp.h"

class WordSourceGroup {
public:
  WordSourceGroup(int num_sources);
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
