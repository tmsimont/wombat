// Copyright 2017 Trevor Simonton

#include "src/buffers/readers/tc_buffer.h"


TCBufferReader::TCBufferReader() {
}

TCBufferReader::TCBufferReader(int *data) {
  setData(data);
}

TCBufferReader::~TCBufferReader() {
  buffer->release(idx);
}

int TCBufferReader::status() {
  return data[0];
}

int TCBufferReader::targetWord() {
  return data[1];
}

int TCBufferReader::numCWords() {
  return data[2];
}

int TCBufferReader::droppedWords() {
  return data[3];
}

int* TCBufferReader::cwords() {
  return data + 4;
}

void TCBufferReader::setStatus(int value) {
  data[0] = value;
}

void TCBufferReader::setTargetWord(int value) {
  data[1] = value;
}

void TCBufferReader::setNumCWords(int value) {
  data[2] = value;
}

void TCBufferReader::incNumCWords() {
  data[2]++;
}

void TCBufferReader::decNumCWords() {
  data[2]--;
}

void TCBufferReader::setDroppedWords(int value) {
  data[3] = value;
}
