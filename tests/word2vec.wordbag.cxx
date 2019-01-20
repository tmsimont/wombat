#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <string>

#include "vocabulary/word2vec.wordbag.h"

using wombat::Word2VecWordBag;

TEST(Word2VecWordBagTest, AddWord) {
  std::string word("hi");
  Word2VecWordBag bag;
  bag.add(word);
  EXPECT_EQ(bag.getWordIndex(word), 1);
}

TEST(Word2VecWordBagTest, WordFrequency) {
  std::string word("hi");
  Word2VecWordBag bag;
  bag.add(word);
  bag.add(word);
  bag.add(word);
  EXPECT_EQ(bag.getWordFrequency(word), 3);
}

TEST(Word2VecWordBagTest, WordNotFound) {
  std::string word("notInDict");
  Word2VecWordBag bag;
  EXPECT_EQ(bag.getWordIndex(word), -1);
  EXPECT_EQ(bag.getWordFrequency(word), 0);
}

TEST(Word2VecWordBagTest, SpecialZero) {
  std::string special("</s>");
  Word2VecWordBag bag;
  EXPECT_EQ(bag.getWordIndex(special), 0);
}

TEST(Word2VecWordBagTest, Size) {
  std::string hi("hi");
  std::string bye("bye");
  Word2VecWordBag bag;
  bag.add(hi);
  bag.add(bye);
  bag.add(bye);
  EXPECT_EQ(bag.getSize(), 3);
}

TEST(Word2VecWordBagTest, Sort) {
  std::string hi("hi");
  std::string bye("bye");
  Word2VecWordBag bag;

  bag.add(hi);

  // when first added, hi should be word 1
  EXPECT_EQ(bag.getWordIndex(hi), 1);

  // bye is added 2 times
  bag.add(bye);
  bag.add(bye);

  bag.sortAndSumFrequency(0);

  // bye should be sorted to position 1, hi bumped to 2
  EXPECT_EQ(bag.getWordIndex(bye), 1);
  EXPECT_EQ(bag.getWordIndex(hi), 2);
}

TEST(Word2VecWordBagTest, SumFrequency) {
  std::string infrequent("infrequent");
  std::string hi("hi");
  std::string bye("bye");
  Word2VecWordBag bag;

  bag.add(hi);
  bag.add(hi);
  bag.add(hi);
  bag.add(infrequent);
  bag.add(bye);
  bag.add(bye);
  bag.add(bye);
  bag.add(bye);

  // frequency should be 3 hi's + 4 bye's (and infrequent omitted)
  uint64_t frequency = bag.sortAndSumFrequency(2);
  EXPECT_EQ(frequency, 7);
}

//TODO: Reduce test
