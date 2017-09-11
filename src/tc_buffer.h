// Copyright 2017 Trevor Simonton

#ifndef TC_BUFFER_H_
#define TC_BUFFER_H_

#include "src/common.h"
#include "src/buffer.cpp"
#include "src/unlocked_buffer.cpp"
#include <vector>
#include "omp.h"

class TCBuffer;

class TCBufferReader : public BufferReader {
public:
  TCBuffer *buffer;
  int idx;

  TCBufferReader();
  TCBufferReader (int *data);
  ~TCBufferReader();
  int status();
  int targetWord();
  int numCWords();
  int* cwords();
  int droppedWords();
  void setStatus(int value);
  void setTargetWord(int value);
  void setDroppedWords(int value);
  void setNumCWords(int value);
  void incNumCWords();
  void decNumCWords();
};

class TCBuffer : public Buffer {
private:
  UnlockedBuffer *buffer;
  omp_lock_t *lock;
  std::vector<omp_lock_t *> itemLocks;
public:
  TCBuffer(int num_items);
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
void printTCBuffers();


#endif
