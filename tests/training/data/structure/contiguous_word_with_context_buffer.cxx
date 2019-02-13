#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <stdint.h>
#include <iostream>

#include <memory>
#include <stdexcept>
#include <iostream>

#include "training/data/structure/word_with_context.h"
#include "training/data/structure/contiguous_buffer_backed.word_with_context.h"
#include "training/data/structure/contiguous_word_with_context_buffer.h"

using wombat::WordWithContext;
using wombat::WordWithContextVisitor;
using wombat::ContiguousWordWithContextBuffer;
using wombat::ContiguousBufferBackedWordWithContext;

const int S = 8;
const int targetIndex = 1;
const int32_t NonBufferBackedWordWithContext_targetWord = 1;
const int32_t NonBufferBackedWordWithContext_contextWord1 = 2;
const int32_t NonBufferBackedWordWithContext_contextWord2 = 3;

/**
 * Helper WordWithContext visitor that just builds a std::vector out of the context words.
 */
class ContiguousBufferTestVisitor : public WordWithContextVisitor {
  public:
    std::vector<int32_t> v;

    void visitContextWord(const int32_t& wordIndex) {
      v.push_back(wordIndex);
    }
};

/**
 * Dumb instance of WordWithContext that is not an actual
 * NonBufferBackedWordWithContext to test conversion.
 */
class NonBufferBackedWordWithContext : public WordWithContext {
  public:
    ~NonBufferBackedWordWithContext() {}
    int32_t getTargetWord() const { 
      return NonBufferBackedWordWithContext_targetWord;
    }

    int32_t getNumberOfContextWords() const { 
      return 2; 
    }

    void acceptContextWordVisitor(WordWithContextVisitor& visitor) const {
      visitor.visitContextWord(
          NonBufferBackedWordWithContext_contextWord1);
      visitor.visitContextWord(
          NonBufferBackedWordWithContext_contextWord2);
    }
};

/**
 * Build a WordWithContext.
 * Push it into the buffer.
 * Pop it out and make sure the popped target word matches what was input.
 */
TEST(ContiguousWordWithContextBuffer, TargetWord) {
  ContiguousWordWithContextBuffer buffer(1, S);

  buffer.push(ContiguousBufferBackedWordWithContext::builder(S)
    .withTargetWord(targetIndex)
    .build());

  auto popped = buffer.pop();
  EXPECT_EQ(popped->getTargetWord(), targetIndex);
}

/**
 * Build a WordWithContext.
 * Push it into the buffer.
 * Pop it out and make sure the popped context word count matches what was input.
 */
TEST(ContiguousWordWithContextBuffer, ContextWordsCount) {
  ContiguousWordWithContextBuffer buffer(1, S);

  buffer.push(ContiguousBufferBackedWordWithContext::builder(S)
    .withContextWord(5)
    .withContextWord(6)
    .withContextWord(7)
    .build());

  auto popped = buffer.pop();
  EXPECT_EQ(popped->getNumberOfContextWords(), 3);
}

/**
 * Build a WordWithContext.
 * Push it into the buffer.
 * Pop it out and make sure the popped context word set matches what was input.
 */
TEST(ContiguousWordWithContextBuffer, ContextWordsVisitor) {
  ContiguousWordWithContextBuffer buffer(1, S);

  buffer.push(ContiguousBufferBackedWordWithContext::builder(S)
    .withContextWord(5)
    .withContextWord(6)
    .withContextWord(7)
    .build());

  auto popped = buffer.pop();

  ContiguousBufferTestVisitor visitor;
  popped->acceptContextWordVisitor(visitor);
  EXPECT_EQ(visitor.v[0], 5);
  EXPECT_EQ(visitor.v[1], 6);
  EXPECT_EQ(visitor.v[2], 7);
}

/**
 * Simple test of empty buffer behavior.
 */
TEST(ContiguousWordWithContextBuffer, EmptyBuffer) {
  ContiguousWordWithContextBuffer buffer(1, S);
  auto popped = buffer.pop();
  EXPECT_EQ(popped, nullptr);
}

/**
 * Simple test of full buffer behavior.
 */
TEST(ContiguousWordWithContextBuffer, FullBuffer) {
  const int S = 8;
  const int targetIndex = 1;
  ContiguousWordWithContextBuffer buffer(1, S);

  int r1 = buffer.push(ContiguousBufferBackedWordWithContext::builder(S)
    .withTargetWord(targetIndex)
    .build());

  int r2 = buffer.push(ContiguousBufferBackedWordWithContext::builder(S)
    .withTargetWord(targetIndex)
    .build());

  EXPECT_EQ(r1, 1);
  EXPECT_EQ(r2, 0);
}

/**
 * Test pushing a WordWithContext that is not ContiguousBufferBackedWordWithContext,
 * pop it back out and make sure the values are correct.
 */
TEST(ContiguousWordWithContextBuffer, WordWithContextTypeConvert) {
  ContiguousWordWithContextBuffer buffer(1, S);

  // Put a non ContiguousBufferBackedWordWithContext WordWithContext into the buffer.
  buffer.push(std::make_unique<NonBufferBackedWordWithContext>());

  // Pop out a WordWithContext instance
  auto popped = buffer.pop();

  // Make sure the instance 
  ContiguousBufferTestVisitor visitor;
  popped->acceptContextWordVisitor(visitor);
  EXPECT_EQ(popped->getTargetWord(), 1);
  EXPECT_EQ(visitor.v[0], 2);
  EXPECT_EQ(visitor.v[1], 3);
}

// TODO: more complicated usage... multiple push/pop's, order checking, bad size..
