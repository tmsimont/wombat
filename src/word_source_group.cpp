// Copyright 2017 Trevor Simonton

#include "word_source_group.h"

WordSourceGroup::WordSourceGroup(int num_sources) {
  this->num_sources = num_sources;
  for (int i = 0; i < num_sources; ++i) {
    omp_lock_t *l = (omp_lock_t *) malloc(sizeof(omp_lock_t));
    omp_init_lock(l);
    source_locks.push_back(l);
    activeList.push_back(1);
  }
  activeCountLock = (omp_lock_t *) malloc(sizeof(omp_lock_t));
  omp_init_lock(activeCountLock);
  num_active = num_sources;
}

void WordSourceGroup::useLocks(bool u) {
  use_locks = u;
}

WordSource* WordSourceGroup::at(int idx) {
  if (!use_locks || omp_test_lock(source_locks[idx])) {
    return sources[idx];
  }
  else {
    return nullptr;
  }
}

void WordSourceGroup::release(int idx) {
  if (use_locks) omp_unset_lock(source_locks[idx]);
}

void WordSourceGroup::setInactive(int idx) {
  if (use_locks) omp_set_lock(activeCountLock);
  if (isActive(idx)) {
    activeList[idx] = 0;
    num_active--;
  }
  if (use_locks) omp_unset_lock(activeCountLock);
}

bool WordSourceGroup::isActive(int idx) {
  return activeList[idx] == 1;
}

bool WordSourceGroup::hasActiveSource() {
  bool hasActive = false;
  if (use_locks) omp_set_lock(activeCountLock);
  hasActive = num_active > 0;
  if (use_locks) omp_unset_lock(activeCountLock);
  return hasActive;
}
