#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "training/data/stdVector.sentence.h"
#include "training/data/sentence.visitor.h"

#include <vector>

using wombat::StdVectorSentence;
using wombat::SentenceVisitor;

class TestVisitor : public SentenceVisitor {
  public:
    std::vector<int32_t> v;

    void visitWord(const int32_t& wordIndex) {
      v.push_back(wordIndex);
    }
};

TEST(StdVectorSenteceTest, VisitInOrder) {
  StdVectorSentence sentence;
  sentence.addWord(42);
  sentence.addWord(2);
  sentence.addWord(3);

  TestVisitor visitor;
  sentence.acceptWordVisitor(visitor);

  EXPECT_EQ(visitor.v[0], 42);
  EXPECT_EQ(visitor.v[1], 2);
  EXPECT_EQ(visitor.v[2], 3);
}

TEST(StdVectorSenteceTest, Size) {
  StdVectorSentence sentence;
  sentence.addWord(3);
  sentence.addWord(2);
  sentence.addWord(2);
  sentence.addWord(3);

  EXPECT_EQ(sentence.getNumberOfTrainingWords(), 4);
  EXPECT_EQ(sentence.getNumberOfWordsInput(), 4);
}

TEST(StdVectorSenteceTest, SizeWithDiscarded) {
  StdVectorSentence sentence;
  sentence.addWord(3);
  sentence.countDiscardedWord();
  sentence.addWord(2);
  sentence.addWord(2);
  sentence.addWord(3);
  sentence.countDiscardedWord();

  EXPECT_EQ(sentence.getNumberOfTrainingWords(), 4);
  EXPECT_EQ(sentence.getNumberOfWordsInput(), 6);
}
