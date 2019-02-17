#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "training/data/source/stream_backed.word_source.h" 
#include "vocabulary/wordbag/wordbag.factory.h" 

#include <fstream>

using wombat::StreamBackedWordSource;
using wombat::WordBagFactory;
using wombat::StreamBackedWordSource;

const std::string TEST_FILE_NAME("tests/resources/wordbag.txt");

std::unique_ptr<WordBagFactory> getFactory(const std::string& fileName) {
  auto inputStream = std::make_unique<std::ifstream>();
  inputStream->open(fileName, std::ios::out);
  if (!inputStream->is_open()) {
    throw std::invalid_argument("Unable to open test file.");
  }
  return std::make_unique<WordBagFactory>(
      std::make_unique<StreamBackedWordSource>(std::move(inputStream)));
}

TEST(WordBagFactory, BuildExpectedBag) {
  auto wordBagFactory = getFactory(TEST_FILE_NAME);
  auto bag = wordBagFactory->makeWordBag();
  EXPECT_EQ(bag->getWordFrequency("one"), 1);
  EXPECT_EQ(bag->getWordFrequency("two"), 2);
  EXPECT_EQ(bag->getWordFrequency("three"), 3);
  EXPECT_EQ(bag->getWordFrequency("four"), 4);
  EXPECT_EQ(bag->getWordFrequency("five"), 5);
  EXPECT_EQ(bag->getWordFrequency("six"), 6);
  EXPECT_EQ(bag->getWordFrequency("seven"), 7);
  EXPECT_EQ(bag->getWordFrequency("eight"), 8);
  EXPECT_EQ(bag->getWordFrequency("nine"), 9);
  EXPECT_EQ(bag->getWordFrequency("ten"), 10);
}
