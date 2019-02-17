#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "vocabulary/stream_backed.word_source.h" 
#include <iostream>
#include <exception>
#include <fstream>

using wombat::StreamBackedWordSource;

const std::string TEST_FILE_NAME("tests/resources/simple_sentence.txt");
const std::string TEST_FILE_NAME_MULTILINE("tests/resources/multiline_sentences.txt");

std::unique_ptr<StreamBackedWordSource> getWordSourceFromFile(const std::string& fileName) {
  auto inputStream = std::make_unique<std::ifstream>();
  inputStream->open(fileName, std::ios::out);
  if (!inputStream->is_open()) {
    throw std::invalid_argument("Unable to open test file.");
  }
  return std::make_unique<StreamBackedWordSource>(std::move(inputStream));
}

TEST(StreamBackedWordSource, WordProductionFromStream) {
  auto source = getWordSourceFromFile(TEST_FILE_NAME);
  ASSERT_TRUE(source->hasNext());
  int i = 0;
  int escape = 0;
  while (source->hasNext() && escape++ < 10) {
    EXPECT_GT(source->nextWord().size(), 0);
    i++;
  }
  ASSERT_LT(escape, 10);
  // 8 words, plus "</s>"
  EXPECT_EQ(i, 9);
}

TEST(StreamBackedWordSource, MultipleLinesInInput) {
  auto source = getWordSourceFromFile(TEST_FILE_NAME_MULTILINE);
  ASSERT_TRUE(source->hasNext());
  int i = 0;
  int escape = 0;
  while (source->hasNext() && escape++ < 28) {
    EXPECT_GT(source->nextWord().size(), 0);
    i++;
  }
  ASSERT_LT(escape, 28);
  // 3 * (8 words, plus "</s>")
  EXPECT_EQ(i, 27);
}

TEST(StreamBackedWordSource, NoMoreWords) {
  auto source = getWordSourceFromFile(TEST_FILE_NAME);
  int escape = 0;
  while (source->hasNext() && escape++ < 10) {
    source->nextWord();
  }
  ASSERT_LT(escape, 10);
  std::string over = source->nextWord();

  // expect 0-length string when nextWord run and hasNext() is false.
  EXPECT_EQ(over.size(), 0);
}

