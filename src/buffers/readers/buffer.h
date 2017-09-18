// Copyright 2017 Trevor Simonton

#ifndef BUFFER_READER_H_
#define BUFFER_READER_H_

#include "src/common.h"

class BufferReader {
 protected:
  int *data;
 public:
  int idx;
  void setData(int *data) { this->data = data; }
  int* getData() { return data; }
};

#endif
