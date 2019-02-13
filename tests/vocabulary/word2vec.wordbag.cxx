#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <string>

#include "vocabulary/word2vec.wordbag.h"
#include "vocabulary/word2vec.wordbag.builder.h"

#include <memory>

using wombat::WordBag;
using wombat::Word2VecWordBag;
using wombat::Word2VecWordBagBuilder;

TEST(Word2VecWordBagTest, AddWord) {
  std::string word("hi");
  Word2VecWordBagBuilder builder;
  builder.add(word);
  std::unique_ptr<WordBag> bag = builder.build();
  EXPECT_EQ(bag->getWordIndex(word), 1);
}

TEST(Word2VecWordBagTest, WordFrequency) {
  std::string word("hi");
  Word2VecWordBagBuilder builder;
  builder.add(word);
  builder.add(word);
  builder.add(word);
  std::unique_ptr<WordBag> bag = builder.build();
  EXPECT_EQ(bag->getWordFrequency(word), 3);
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
  Word2VecWordBagBuilder builder;
  builder.add(hi);
  builder.add(bye);
  builder.add(bye);
  std::unique_ptr<WordBag> bag = builder.build();
  EXPECT_EQ(bag->getSize(), 3);
}

TEST(Word2VecWordBagTest, SortA) {
  std::string hi("hi");
  std::string bye("bye");
  Word2VecWordBagBuilder builder;

  // hi is added 2 times
  builder.add(hi);
  builder.add(hi);

  // bye is added 1 time
  builder.add(bye);

  std::unique_ptr<WordBag> bag = builder.build();

  // hi should be first
  EXPECT_EQ(bag->getWordIndex(hi), 1);
  EXPECT_EQ(bag->getWordIndex(bye), 2);
}

TEST(Word2VecWordBagTest, SortB) {
  std::string hi("hi");
  std::string bye("bye");
  Word2VecWordBagBuilder builder;

  builder.add(hi);

  // bye is added 2 times
  builder.add(bye);
  builder.add(bye);

  std::unique_ptr<WordBag> bag = builder.build();

  // bye should be sorted to position 1, hi bumped to 2
  EXPECT_EQ(bag->getWordIndex(bye), 1);
  EXPECT_EQ(bag->getWordIndex(hi), 2);
}

TEST(Word2VecWordBagTest, Cardinality) {
  std::string infrequent("infrequent");
  std::string hi("hi");
  std::string bye("bye");
  Word2VecWordBagBuilder builder;
  std::unique_ptr<WordBag> bag = builder
    .add(hi)
    .add(hi)
    .add(hi)
    .add(infrequent)
    .add(bye)
    .add(bye)
    .add(bye)
    .add(bye)
    .withFrequencyThreshold(2)
    .build();

  // frequency should be 3 hi's + 4 bye's (and infrequent omitted)
  EXPECT_EQ(bag->getCardinality(), 7);
}

//TODO: Reduce test
