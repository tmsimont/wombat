// Copyright 2017 Trevor Simonton

#include "src/buffers/tc_buffer.h"


std::vector<TCBuffer *> tc_buffers;

TCBuffer::TCBuffer(int num_items) {
  buffer = new UnlockedBuffer(tc_buffer_item_size, num_items);
  lock = reinterpret_cast<omp_lock_t *>(malloc(sizeof(omp_lock_t)));
  omp_init_lock(lock);
}

TCBuffer::~TCBuffer() {
  delete buffer;
}

bool TCBuffer::testLock() {
  return omp_test_lock(lock);
}

void TCBuffer::setLock() {
  omp_set_lock(lock);
}

void TCBuffer::unsetLock() {
  omp_unset_lock(lock);
}

int TCBuffer::getEmptyItem(BufferReader *tc_reader) {
  BufferReader r;
  if (buffer->isFull()) {
    return 0;
  }
  buffer->getEmptyItem(&r);
  TCBufferReader *tcr = static_cast<TCBufferReader *>(tc_reader);
  tcr->setData(r.getData());
  tcr->buffer = this;
  tcr->idx = r.idx;
  tcr->setNumCWords(0);
  return 1;
}

int TCBuffer::getReadyItem(BufferReader *tc_reader) {
  BufferReader r;
  if (buffer->isEmpty()) {
    return 0;
  }
  buffer->getReadyItem(&r);
  TCBufferReader *tcr = static_cast<TCBufferReader *>(tc_reader);
  tcr->setData(r.getData());
  tcr->buffer = this;
  tcr->idx = r.idx;
  return 1;
}

void TCBuffer::release(int idx) {
}

bool TCBuffer::isFull() {
  return buffer->isFull();
}

bool TCBuffer::isEmpty() {
  return buffer->isEmpty();
}

int TCBuffer::itemSize() {
  return buffer->itemSize();
}

int TCBuffer::numItems() {
  return buffer->numItems();
}

void InitTCBuffers(int window, int num_buffers, int num_items) {
  tc_buffer_item_size = 4 + (2 * window + 1);
  for (int i = 0; i < num_buffers; ++i) {
    tc_buffers.push_back(new TCBuffer(num_items));
  }
}
