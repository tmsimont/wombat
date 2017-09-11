#ifndef SEN_BUFFER_H_
#define SEN_BUFFER_H_


#include "common.h"
#include "buffer.cpp"
#include "unlocked_buffer.cpp"
#include <vector>

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

class SenBuffer : public Buffer {
private:
  UnlockedBuffer *buffer;
public:
  SenBuffer(int num_items);
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
void printSenBuffers();

#endif
