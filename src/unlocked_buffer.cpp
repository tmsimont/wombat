// Copyright 2017 Trevor Simonton

#ifndef UNLOCKED_BUFFER_READER_
#define UNLOCKED_BUFFER_READER_

#include "src/buffer.cpp"

class UnlockedBuffer : public Buffer {
protected:
  int *data;
  int num_items;
  int item_size;

private:
  int i_empty = 0, i_empty_wrap = 0;
  int i_ready = 0, i_ready_wrap = 0;

public:
  UnlockedBuffer(int item_size, int num_items) {
    this->item_size = item_size;
    this->num_items = num_items;
    posix_memalign((void **)&data, 64, item_size * num_items * sizeof(int));
  }

  ~UnlockedBuffer() {
    free(data);
  }

  bool isFull() {
    return i_empty == i_ready && i_empty_wrap != i_ready_wrap;
  }

  bool isEmpty() {
    return i_empty == i_ready && i_empty_wrap == i_ready_wrap;
  }

  int itemSize() {
    return item_size;
  }

  int numItems() {
    return num_items;
  }

  int getEmptyItem(BufferReader *reader) {
    if (isFull()) return 0;
    reader->idx = i_empty;
    reader->setData(data + i_empty++ * item_size);
    if (i_empty == num_items) {
      i_empty = 0;
      ++i_empty_wrap;
      i_ready_wrap = i_ready_wrap - i_empty_wrap;
      i_empty_wrap = 0;
    }
    return 1;
  }

  int getReadyItem(BufferReader *reader) {
    if (isEmpty()) return 0;
    reader->idx = i_ready;
    reader->setData(data + i_ready++ * item_size);
    if (i_ready == num_items) {
      i_ready = 0;
      ++i_ready_wrap;
      i_ready_wrap = i_ready_wrap - i_empty_wrap;
      i_empty_wrap = 0;
    }
    return 1;
  }

};

#endif
