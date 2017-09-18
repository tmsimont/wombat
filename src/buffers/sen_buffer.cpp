// Copyright 2017 Trevor Simonton

#include "src/buffers/sen_buffer.h"

std::vector<SenBuffer *> sen_buffers;

SenBuffer::SenBuffer(int num_items) {
  buffer = new UnlockedBuffer(sen_buffer_item_size, num_items);
}
SenBuffer::~SenBuffer() {
  delete buffer;
}
int SenBuffer::getEmptyItem(BufferReader *sen_reader) {
  BufferReader r;
  int got_item = buffer->getEmptyItem(&r);
  if (got_item) {
    SenBufferReader *sr = static_cast<SenBufferReader*>(sen_reader);
    sr->setData(r.getData());
    sr->buffer = this;
    sr->idx = r.idx;
    sr->setLength(0);
    sr->setPosition(0);
    sr->setDroppedWords(0);
    return 1;
  }
  return 0;
}
int SenBuffer::getReadyItem(BufferReader *sen_reader) {
  BufferReader r;
  int got_item = buffer->getReadyItem(&r);
  if (got_item) {
    SenBufferReader *sr = static_cast<SenBufferReader*>(sen_reader);
    sr->setData(r.getData());
    sr->buffer = this;
    sr->idx = r.idx;
    return 1;
  }
  return 0;
}
bool SenBuffer::isFull() {
  return buffer->isFull();
}
bool SenBuffer::isEmpty() {
  return buffer->isEmpty();
}
int SenBuffer::itemSize() {
  return buffer->itemSize();
}
int SenBuffer::numItems() {
  return buffer->numItems();
}

void InitSenBuffers(int num, int sentences_in_buffer) {
  sen_buffer_item_size = MAX_SENTENCE_LENGTH + 3;
  for (int i = 0; i < num; ++i) {
    sen_buffers.push_back(new SenBuffer(sentences_in_buffer));
  }
}
