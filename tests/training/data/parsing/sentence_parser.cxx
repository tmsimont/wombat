#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "training/data/parsing/sentence_parser.h"
#include "training/data/structure/stdVector.sentence.h"
#include "training/data/structure/word_with_context.h"
#include "training/data/structure/word_with_context.visitor.h"

#include <algorithm>
#include <vector>

using wombat::SentenceParser;
using wombat::Sentence;
using wombat::StdVectorSentence;
using wombat::WordWithContext;
using wombat::WordWithContextVisitor;

class ContextWordsAsVectorForTest : public WordWithContextVisitor {
  public:
    std::vector<int32_t> stdVector;
    void visitContextWord(const int32_t& wordIndex) {
      stdVector.push_back(wordIndex);
    }
};


TEST(SentenceParserTest, TargetWord) {
  StdVectorSentence sentence;
  sentence.addWord(42);
  sentence.addWord(43);
  sentence.addWord(45);

  SentenceParser parser(sentence, 5 /* windowSize */);

  auto wordWithContext = parser.nextWordWithContext();

  EXPECT_EQ(wordWithContext->getTargetWord(), 42);
}

TEST(SentenceParserTest, ForwardContextWords) {
  StdVectorSentence sentence;
  ContextWordsAsVectorForTest contextWordsVector;
  int32_t windowSize = 5;

  // first "word with context" will have 42 as target
  sentence.addWord(42);

  // window starts here.
  sentence.addWord(1);
  sentence.addWord(2);
  sentence.addWord(3);
  sentence.addWord(4);
  sentence.addWord(5);

  // window should not include this word.
  sentence.addWord(6);

  SentenceParser parser(sentence, windowSize);

  auto wordWithContext = parser.nextWordWithContext();
  ASSERT_TRUE(wordWithContext->getTargetWord() == 42);

  // pass the context words to contextWordsVector's stdVector
  wordWithContext->acceptContextWordVisitor(contextWordsVector);

  // check for the expected context words
  for (int i = 0; i < windowSize; i++) { 
    EXPECT_NE(
        std::find(contextWordsVector.stdVector.begin(),
          contextWordsVector.stdVector.end(),
          i + 1),
        contextWordsVector.stdVector.end());
  }

  // expect word 6 to be excluded.
  EXPECT_EQ(
      std::find(contextWordsVector.stdVector.begin(),
                contextWordsVector.stdVector.end(),
                6),
      contextWordsVector.stdVector.end());

}

TEST(SentenceParserTest, BeforeAndAfterContext) {
  StdVectorSentence sentence;
  ContextWordsAsVectorForTest contextWordsVector;
  int32_t windowSize = 2;

  sentence.addWord(0);

  // window starts here.
  sentence.addWord(1);
  sentence.addWord(2);

  // thrid  "word with context" will have 42 as target
  sentence.addWord(42);

  sentence.addWord(3);
  sentence.addWord(4);

  // window should not include these words.
  sentence.addWord(5);

  SentenceParser parser(sentence, windowSize);

  // target: 0
  auto wordWithContext = parser.nextWordWithContext();
  // target: 1
  wordWithContext = parser.nextWordWithContext();
  // target: 2
  wordWithContext = parser.nextWordWithContext();
  // target: 42
  wordWithContext = parser.nextWordWithContext();
  ASSERT_TRUE(wordWithContext->getTargetWord() == 42);

  // pass the context words to contextWordsVector's stdVector
  wordWithContext->acceptContextWordVisitor(contextWordsVector);

  // check for the expected context words
  for (int i = 0; i < windowSize * 2; i++) { 
    EXPECT_NE(
        std::find(contextWordsVector.stdVector.begin(),
          contextWordsVector.stdVector.end(),
          i + 1),
        contextWordsVector.stdVector.end());
  }

  // expect word 0 to be excluded.
  EXPECT_EQ(
      std::find(contextWordsVector.stdVector.begin(),
                contextWordsVector.stdVector.end(),
                0),
      contextWordsVector.stdVector.end());

  // expect word 5 to be excluded.
  EXPECT_EQ(
      std::find(contextWordsVector.stdVector.begin(),
                contextWordsVector.stdVector.end(),
                5),
      contextWordsVector.stdVector.end());
}

TEST(SentenceParserTest, FullSentenceConsumption) {
  StdVectorSentence sentence;
  ContextWordsAsVectorForTest contextWordsVector;
  int32_t windowSize = 2;
  int32_t sentenceSize = 100;

  // Build a sentence with words 0,1,2,3...,sentenceSize
  for (int i = 0; i < sentenceSize; i++) {
    sentence.addWord(i);
  }

  // Initiate a parser for this sentence.
  SentenceParser parser(sentence, windowSize);

  // Expect that we have 0,1,2,3...,sentenceSize words parsed.
  int currentPosition = 0;
  for (; currentPosition < sentenceSize; currentPosition++) {
    auto wordWithContext = parser.nextWordWithContext();
    ASSERT_FALSE(wordWithContext == nullptr);
    EXPECT_EQ(wordWithContext->getTargetWord(), currentPosition);
  }

  // Expect any steps passed the end of the sentence to yield nullptr.
  for (int i = currentPosition; i < 5; i++) {
    auto wordWithContext = parser.nextWordWithContext();
    EXPECT_EQ(wordWithContext, nullptr);
  }
}
