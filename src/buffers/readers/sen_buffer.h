// Copyright 2017 Trevor Simonton

#ifndef SEN_BUFFER_READER_H_
#define SEN_BUFFER_READER_H_

#include <vector>

#include "src/common.h"
#include "src/buffers/readers/buffer.h"

class SenBuffer;

class SenBufferReader : public BufferReader {
 public:
  SenBuffer *buffer;
  int idx;

  ~SenBufferReader();

  int length();
  int droppedWords();
  int position();
  int* sen();
  void setLength(int value);
  void incLength();
  void setDroppedWords(int value);
  void incDroppedWords();
  void setPosition(int value);
  void incPosition();
  void setStatus(int value);
  void markEmpty();
};

#endif
