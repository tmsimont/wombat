#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <string>

#include "vocabulary/word2vec.dictionary.h"

using wombat::Word2VecDictionary;

TEST(Word2VecDictionaryTest, AddWord) {
  std::string word("hi");
  Word2VecDictionary dict;
  //dict.add(word);
  //EXPECT_EQ(dict.get(word), 0);
}
