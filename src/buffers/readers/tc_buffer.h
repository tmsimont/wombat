// Copyright 2017 Trevor Simonton

#ifndef TC_BUFFER_READER_H_
#define TC_BUFFER_READER_H_

#include <omp.h>

#include <vector>

#include "src/common.h"
#include "src/buffers/readers/buffer.h"
#include "src/buffers/tc_buffer.h"

class TCBuffer;

/**
 * The TCBufferReader is used to read and write
 * entries to TCBuffer instances
 */
class TCBufferReader : public BufferReader {
 public:
  TCBuffer *buffer;
  int idx;

  TCBufferReader();
  explicit TCBufferReader(int *data);
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

#endif
