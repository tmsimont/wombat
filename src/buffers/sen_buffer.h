// Copyright 2017 Trevor Simonton

#ifndef SEN_BUFFER_H_
#define SEN_BUFFER_H_

#include <vector>

#include "src/common.h"
#include "src/buffers/buffer.h"
#include "src/buffers/readers/sen_buffer.h"
#include "src/buffers/unlocked_buffer.h"

class SenBuffer : public Buffer {
 private:
  UnlockedBuffer *buffer;
 public:
  explicit SenBuffer(int num_items);
  ~SenBuffer();
  int getEmptyItem(BufferReader *reader);
  int getReadyItem(BufferReader *reader);
  bool isFull();
  bool isEmpty();
  int itemSize();
  int numItems();
};

extern int sen_buffer_item_size;
extern std::vector<SenBuffer *> sen_buffers;
extern int sentences_in_buffer;

void InitSenBuffers(int num, int sentences_in_buffer);

#endif
