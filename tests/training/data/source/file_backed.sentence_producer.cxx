#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "training/data/source/file_backed.sentence_producer.h"
#include "training/data/structure/sentence.visitor.h"
#include "vocabulary/word2vec.wordbag.builder.h"
#include "vocabulary/wordbag.h"

using wombat::Sentence;
using wombat::SentenceVisitor;
using wombat::FileBackedSentenceProducer;
using wombat::Word2VecWordBagBuilder;
using wombat::WordBag;

const std::string TEST_FILE_NAME("tests/resources/sentence_producer_input.txt");

/**
 * Helper Sentence visitor that builds an std::vector out of word indices in Sentence.
 */
class SentenceTestVisitor : public SentenceVisitor {
  public:
    std::vector<int32_t> v;
    void visitWord(const int32_t& wordIndex) {
      v.push_back(wordIndex);
    }
};

TEST(FileBackedSentenceProducer, FileNotFound) {
  Word2VecWordBagBuilder bagBuilder;
  FileBackedSentenceProducer producer(
      bagBuilder.build(),
      1024);
  EXPECT_EQ(producer.setFile("nonFile.txt"), 0);
}

TEST(FileBackedSentenceProducer, FileFound) {
  Word2VecWordBagBuilder bagBuilder;
  FileBackedSentenceProducer producer(
      bagBuilder.build(),
      1024);
  EXPECT_EQ(producer.setFile(TEST_FILE_NAME), 1);
}

TEST(FileBackedSentenceProducer, SentenceProductionFromFile) {
  // Put expected words into a word bag.
  Word2VecWordBagBuilder bagBuilder;
  bagBuilder.add("this");
  bagBuilder.add("is");
  bagBuilder.add("sentence");
  // some words will not be recognized in the bag
  std::shared_ptr<WordBag> bag = bagBuilder.build();

  // Create a file-backed sentence producer.
  FileBackedSentenceProducer producer(
      bag,
      1024);

  // Make sure we can read the test file.
  ASSERT_THAT(producer.setFile(TEST_FILE_NAME), 1);

  // Produce a sentence.
  auto sentence = producer.nextSentence();

  // Visit the sentence, copy indices to a std::vector
  SentenceTestVisitor visitor;
  sentence->acceptWordVisitor(visitor);
  ASSERT_THAT(visitor.v.size(), 3);
  EXPECT_EQ(visitor.v[0], bag->getWordIndex("this"));
  EXPECT_EQ(visitor.v[1], bag->getWordIndex("is"));
  EXPECT_EQ(visitor.v[2], bag->getWordIndex("sentence"));
}
