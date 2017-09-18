// Copyright 2017 Trevor Simonton

#include "src/buffers/readers/sen_buffer.h"

SenBufferReader::~SenBufferReader() {
}

void SenBufferReader::setStatus(int s) {
}

void SenBufferReader::markEmpty() {
  setLength(0);
}

int SenBufferReader::length() {
  return data[0];
}

int SenBufferReader::droppedWords() {
  return data[1];
}

int SenBufferReader::position() {
  return data[2];
}

int* SenBufferReader::sen() {
  return data + 3;
}

void SenBufferReader::setLength(int value) {
  data[0] = value;
}

void SenBufferReader::incLength() {
  data[0]++;
}

void SenBufferReader::setDroppedWords(int value) {
  data[1] = value;
}

void SenBufferReader::incDroppedWords() {
  data[1]++;
}

void SenBufferReader::setPosition(int value) {
  data[2] = value;
}

void SenBufferReader::incPosition() {
  data[2]++;
}
