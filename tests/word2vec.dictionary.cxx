#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <string>

#include "vocabulary/word2vec.dictionary.h"

using wombat::Word2VecDictionary;

TEST(Word2VecDictionaryTest, AddWord) {
  std::string word("hi");
  Word2VecDictionary dict;
  dict.add(word);
  EXPECT_EQ(dict.get(word), 1);
}

TEST(Word2VecDictionaryTest, WordNotFound) {
  std::string word("notInDict");
  Word2VecDictionary dict;
  EXPECT_EQ(dict.get(word), -1);
}

TEST(Word2VecDictionaryTest, SpecialZero) {
  std::string special("</s>");
  Word2VecDictionary dict;
  EXPECT_EQ(dict.get(special), 0);
}

//TODO: Sort test
//TODO: Reduce test
