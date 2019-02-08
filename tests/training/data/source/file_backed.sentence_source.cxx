#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "training/data/source/file_backed.sentence_source.h"
#include "training/data/structure/sentence.visitor.h"
#include "vocabulary/word2vec.wordbag.builder.h"
#include "vocabulary/wordbag.h"

using wombat::Sentence;
using wombat::SentenceVisitor;
using wombat::FileBackedSentenceSource;
using wombat::Word2VecWordBagBuilder;
using wombat::WordBag;

const std::string TEST_FILE_NAME("tests/resources/sentence_source_input.txt");
const std::string TEST_FILE_NAME_MULTILINE("tests/resources/sentence_source_multiline_input.txt");

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

TEST(FileBackedSentenceSource, FileNotFound) {
  Word2VecWordBagBuilder bagBuilder;
  bagBuilder.add("word");
  FileBackedSentenceSource source(bagBuilder.build());
  EXPECT_EQ(source.setFile("nonFile.txt"), 0);
}

TEST(FileBackedSentenceSource, FileFound) {
  Word2VecWordBagBuilder bagBuilder;
  bagBuilder.add("word");
  FileBackedSentenceSource source(bagBuilder.build());
  EXPECT_EQ(source.setFile(TEST_FILE_NAME), 1);
}

TEST(FileBackedSentenceSource, SentenceProductionFromFile) {
  // Put expected words into a word bag.
  Word2VecWordBagBuilder bagBuilder;
  // some words will not be recognized in the bag
  bagBuilder.add("this");
  bagBuilder.add("is");
  bagBuilder.add("sentence");
  std::shared_ptr<WordBag> bag = bagBuilder.build();

  // Create a file-backed sentence source.
  FileBackedSentenceSource source(bag);

  // Make sure we can read the test file.
  ASSERT_THAT(source.setFile(TEST_FILE_NAME), 1);

  // Produce a sentence.
  auto sentence = source.nextSentence();

  // Visit the sentence, copy indices to a std::vector
  SentenceTestVisitor visitor;
  sentence->acceptWordVisitor(visitor);
  ASSERT_THAT(visitor.v.size(), 3);
  EXPECT_EQ(visitor.v[0], bag->getWordIndex("this"));
  EXPECT_EQ(visitor.v[1], bag->getWordIndex("is"));
  EXPECT_EQ(visitor.v[2], bag->getWordIndex("sentence"));
}

TEST(FileBackedSentenceSource, EmptyBag) {
  // Create an empty bag.
  Word2VecWordBagBuilder bagBuilder;
  std::shared_ptr<WordBag> bag = bagBuilder.build();

  // Shouldn't be able to pass empty bag to source.
  EXPECT_THROW({
    FileBackedSentenceSource source(bag);
  }, std::exception);
}

TEST(FileBackedSentenceSource, OverPollingInput) {
  // Put expected words into a word bag.
  Word2VecWordBagBuilder bagBuilder;
  bagBuilder.add("this");
  bagBuilder.add("is");
  bagBuilder.add("sentence");
  std::shared_ptr<WordBag> bag = bagBuilder.build();

  // Create a file-backed sentence source.
  FileBackedSentenceSource source(bag);

  // Make sure we can read the test file.
  ASSERT_THAT(source.setFile(TEST_FILE_NAME), 1);

  // Produce a sentence.
  auto sentence = source.nextSentence();

  ASSERT_THAT(source.hasNext(), false);
  sentence = source.nextSentence();

  // Make sure we're at the end of the input
  EXPECT_EQ(source.hasNext(), false);
  EXPECT_EQ(sentence, nullptr);
}

TEST(FileBackedSentenceSource, MultipleLinesInInput) {
  // Put expected words into a word bag.
  Word2VecWordBagBuilder bagBuilder;
  bagBuilder.add("this");
  bagBuilder.add("is");
  bagBuilder.add("sentence");
  std::shared_ptr<WordBag> bag = bagBuilder.build();

  // Create a file-backed sentence source.
  FileBackedSentenceSource source(bag);

  // Make sure we can read the test file.
  ASSERT_THAT(source.setFile(TEST_FILE_NAME_MULTILINE), 1);

  // Iterate over sentences
  int32_t numSentences = 0;
  while (source.hasNext()) {
    auto sentence = source.nextSentence();
    numSentences++;
  }
  EXPECT_EQ(numSentences, 3);
}
