#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "training/data/source/word_sampling.sentence_source.h"
#include "training/data/source/word_source.h"
#include "training/data/source/stream_backed.word_source.h"
#include "training/data/structure/sentence.visitor.h"
#include "vocabulary/wordbag/word2vec.wordbag.builder.h"
#include "vocabulary/wordbag/wordbag.h"

#include <iostream>

using wombat::Sentence;
using wombat::SentenceVisitor;
using wombat::StreamBackedWordSource;
using wombat::WordSamplingSentenceSource;
using wombat::Word2VecWordBagBuilder;
using wombat::WordBag;
using wombat::WordSource;

const std::string TEST_FILE_NAME("tests/resources/simple_sentence.txt");
const std::string TEST_FILE_NAME_MULTILINE("tests/resources/multiline_sentences.txt");

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

/**
 * Helper to build a file-stream-based word source.
 */
std::unique_ptr<WordSource> makeWordSource(const std::string& fileName) {
  auto inputStream = std::make_unique<std::ifstream>();
  inputStream->open(fileName, std::ios::out);
  if (!inputStream->is_open()) {
    throw std::invalid_argument("Unable to open test file.");
  }
  return std::make_unique<StreamBackedWordSource>(std::move(inputStream));
}

TEST(WordSamplingSentenceSource, SentenceProductionFromFile) {
  // Put expected words into a word bag.
  Word2VecWordBagBuilder bagBuilder;
  // some words will not be recognized in the bag
  bagBuilder.add("this");
  bagBuilder.add("is");
  bagBuilder.add("sentence");
  std::shared_ptr<WordBag> bag = bagBuilder.build();

  // Create a file-backed sentence source.
  WordSamplingSentenceSource source(bag, makeWordSource(TEST_FILE_NAME));

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

TEST(WordSamplingSentenceSource, EmptyBag) {
  // Create an empty bag.
  Word2VecWordBagBuilder bagBuilder;
  std::shared_ptr<WordBag> bag = bagBuilder.build();

  // Shouldn't be able to pass empty bag to source.
  EXPECT_THROW({
    WordSamplingSentenceSource source(bag, makeWordSource(TEST_FILE_NAME));
  }, std::exception);
}

TEST(WordSamplingSentenceSource, OverPollingInput) {
  // Put expected words into a word bag.
  Word2VecWordBagBuilder bagBuilder;
  bagBuilder.add("this");
  bagBuilder.add("is");
  bagBuilder.add("sentence");
  std::shared_ptr<WordBag> bag = bagBuilder.build();

  // Create a file-backed sentence source.
  WordSamplingSentenceSource source(bag, makeWordSource(TEST_FILE_NAME));

  // Produce a sentence.
  auto sentence = source.nextSentence();

  ASSERT_THAT(source.hasNext(), false);
  sentence = source.nextSentence();

  // Make sure we're at the end of the input
  EXPECT_EQ(source.hasNext(), false);
  EXPECT_EQ(sentence, nullptr);
}

TEST(WordSamplingSentenceSource, MultipleLinesInInput) {
  // Put expected words into a word bag.
  Word2VecWordBagBuilder bagBuilder;
  bagBuilder.add("this");
  bagBuilder.add("is");
  bagBuilder.add("sentence");
  std::shared_ptr<WordBag> bag = bagBuilder.build();

  // Create a file-backed sentence source.
  WordSamplingSentenceSource source(bag, makeWordSource(TEST_FILE_NAME_MULTILINE));

  // Iterate over sentences
  int32_t numSentences = 0;
  while (source.hasNext()) {
    auto sentence = source.nextSentence();
    numSentences++;
  }
  EXPECT_EQ(numSentences, 3);
}

TEST(WordSamplingSentenceSource, DownSampling) {
  // Put expected words into a word bag.
  Word2VecWordBagBuilder bagBuilder;
  bagBuilder.add("this");
  bagBuilder.add("is");
  bagBuilder.add("sentence");
  std::shared_ptr<WordBag> bag = bagBuilder.build();

  // Create a file-backed sentence source.
  WordSamplingSentenceSource source(
      bag,
      makeWordSource(TEST_FILE_NAME_MULTILINE),
      1e-4);

  // Iterate over the same input 1000 times, and count how many
  // words are discarded and sampled.
  int32_t numWordsDiscarded = 0;
  int32_t numWordsSampled = 0;
  int32_t numWordsInput = 0;
  for (int32_t i = 0; i < 1000; ++i) {
    while (source.hasNext()) {
      auto sentence = source.nextSentence();
      numWordsInput += sentence->getNumberOfWordsInput();
      numWordsSampled += sentence->getNumberOfTrainingWords();
      numWordsDiscarded += sentence->getNumberOfWordsInput() 
        - sentence->getNumberOfTrainingWords();
    }
    source.rewind();
  }

  // At least some words should be discarded.
  EXPECT_GT(numWordsDiscarded, 0);
  // At least some words should be input.
  EXPECT_GT(numWordsInput, 0);
  // At least some words should be sampled. Not too many, since
  // there's only 3 words repeating over and over...
  EXPECT_GT(numWordsSampled, 100);
}
