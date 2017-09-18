// Copyright 2017 Trevor Simonton

#ifndef BUFFER_H_
#define BUFFER_H_

#include "src/common.h"

class BufferReader;

class Buffer {
 public:
  virtual ~Buffer() {}
  virtual bool isFull() = 0;
  virtual bool isEmpty() = 0;
  virtual int itemSize() = 0;
  virtual int numItems() = 0;
  virtual int getEmptyItem(BufferReader *reader) = 0;
  virtual int getReadyItem(BufferReader *reader) = 0;
};

#endif
