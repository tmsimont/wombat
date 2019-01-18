#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <string>

#include "vocabulary/word2vec.dictionary.h"

using wombat::Word2VecDictionary;

TEST(Word2VecDictionaryTest, AddWord) {
  std::string word("hi");
  Word2VecDictionary dict;
  dict.add(word);
  EXPECT_EQ(dict.getWordIndex(word), 1);
}

TEST(Word2VecDictionaryTest, WordFrequency) {
  std::string word("hi");
  Word2VecDictionary dict;
  dict.add(word);
  dict.add(word);
  dict.add(word);
  EXPECT_EQ(dict.getWordFrequency(word), 3);
}

TEST(Word2VecDictionaryTest, WordNotFound) {
  std::string word("notInDict");
  Word2VecDictionary dict;
  EXPECT_EQ(dict.getWordIndex(word), -1);
  EXPECT_EQ(dict.getWordFrequency(word), 0);
}

TEST(Word2VecDictionaryTest, SpecialZero) {
  std::string special("</s>");
  Word2VecDictionary dict;
  EXPECT_EQ(dict.getWordIndex(special), 0);
}

TEST(Word2VecDictionaryTest, Size) {
  std::string hi("hi");
  std::string bye("bye");
  Word2VecDictionary dict;
  dict.add(hi);
  dict.add(bye);
  dict.add(bye);
  EXPECT_EQ(dict.getSize(), 3);
}

TEST(Word2VecDictionaryTest, Sort) {
  std::string hi("hi");
  std::string bye("bye");
  Word2VecDictionary dict;

  dict.add(hi);

  // when first added, hi should be word 1
  EXPECT_EQ(dict.getWordIndex(hi), 1);

  // bye is added 2 times
  dict.add(bye);
  dict.add(bye);

  dict.sortAndSumFrequency(0);

  // bye should be sorted to position 1, hi bumped to 2
  EXPECT_EQ(dict.getWordIndex(bye), 1);
  EXPECT_EQ(dict.getWordIndex(hi), 2);
}

TEST(Word2VecDictionaryTest, SumFrequency) {
  std::string infrequent("infrequent");
  std::string hi("hi");
  std::string bye("bye");
  Word2VecDictionary dict;

  dict.add(hi);
  dict.add(hi);
  dict.add(hi);
  dict.add(infrequent);
  dict.add(bye);
  dict.add(bye);
  dict.add(bye);
  dict.add(bye);

  // frequency should be 3 hi's + 4 bye's (and infrequent omitted)
  uint64_t frequency = dict.sortAndSumFrequency(2);
  EXPECT_EQ(frequency, 7);
}

//TODO: Reduce test
