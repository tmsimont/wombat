// Copyright 2017 Trevor Simonton

#ifndef TC_BUFFER_H_
#define TC_BUFFER_H_

#include <omp.h>

#include <vector>

#include "src/common.h"
#include "src/buffers/buffer.h"
#include "src/buffers/readers/buffer.h"
#include "src/buffers/readers/tc_buffer.h"
#include "src/buffers/unlocked_buffer.h"

/**
 * The TCBuffer is a Target/Context word set buffer
 * that holds sets of training word integer tokens 
 * in a contiguous buffer of memory
 */
class TCBuffer : public Buffer {
 private:
  UnlockedBuffer *buffer;
  omp_lock_t *lock;
  std::vector<omp_lock_t *> itemLocks;
 public:
  explicit TCBuffer(int num_items);
  ~TCBuffer();
  int getEmptyItem(BufferReader *reader);
  int getReadyItem(BufferReader *reader);
  bool isFull();
  bool isEmpty();
  int itemSize();
  int numItems();
  void release(int idx);
  void setLock();
  bool testLock();
  void unsetLock();
};

extern int tc_buffer_item_size;
extern int tcbs_per_thread;
extern int items_in_tcb;
extern std::vector<TCBuffer *> tc_buffers;

void InitTCBuffers(int window, int num_buffers, int num_items);

#endif
