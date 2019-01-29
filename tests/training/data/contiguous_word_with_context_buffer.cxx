#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <stdint.h>
#include <iostream>

#include <array>
#include <memory>
#include <stdexcept>

#include "training/data/contiguous_buffer_backed.word_with_context.hpp"
#include "training/data/contiguous_word_with_context_buffer.hpp"

using wombat::WordWithContextVisitor;
using wombat::ContiguousWordWithContextBuffer;
using wombat::ContiguousBufferBackedWordWithContext;

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

const int S = 8;
const int targetIndex = 1;
const int droppedCount = 2;

/**
 * Build a WordWithContext.
 * Push it into the buffer.
 * Pop it out and make sure the popped target word matches what was input.
 */
TEST(ContiguousWordWithContextBuffer, TargetWord) {
  ContiguousWordWithContextBuffer<S> buffer(1);

  buffer.push(ContiguousBufferBackedWordWithContext<S>::builder()
    .withTargetWord(targetIndex)
    .build());

  std::unique_ptr<ContiguousBufferBackedWordWithContext<S>> popped = 
    buffer.pop();
  EXPECT_EQ(popped->getTargetWord(), targetIndex);
}

/**
 * Build a WordWithContext.
 * Push it into the buffer.
 * Pop it out and make sure the popped dropped count matches what was input.
 */
TEST(ContiguousWordWithContextBuffer, DroppedCount) {
  ContiguousWordWithContextBuffer<S> buffer(1);

  buffer.push(ContiguousBufferBackedWordWithContext<S>::builder()
    .withDroppedWordCount(droppedCount)
    .build());

  std::unique_ptr<ContiguousBufferBackedWordWithContext<S>> popped = 
    buffer.pop();
  EXPECT_EQ(popped->getNumberOfDroppedContextWordSamples(), droppedCount);
}

/**
 * Build a WordWithContext.
 * Push it into the buffer.
 * Pop it out and make sure the popped context word count matches what was input.
 */
TEST(ContiguousWordWithContextBuffer, ContextWordsCount) {
  ContiguousWordWithContextBuffer<S> buffer(1);

  buffer.push(ContiguousBufferBackedWordWithContext<S>::builder()
    .withContextWord(5)
    .withContextWord(6)
    .withContextWord(7)
    .build());

  std::unique_ptr<ContiguousBufferBackedWordWithContext<S>> popped = 
    buffer.pop();
  EXPECT_EQ(popped->getNumberOfContextWords(), 3);
}

/**
 * Build a WordWithContext.
 * Push it into the buffer.
 * Pop it out and make sure the popped context word set matches what was input.
 */
TEST(ContiguousWordWithContextBuffer, ContextWordsVisitor) {
  ContiguousWordWithContextBuffer<S> buffer(1);

  buffer.push(ContiguousBufferBackedWordWithContext<S>::builder()
    .withContextWord(5)
    .withContextWord(6)
    .withContextWord(7)
    .build());

  std::unique_ptr<ContiguousBufferBackedWordWithContext<S>> popped = 
    buffer.pop();

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
  const int S = 8;
  ContiguousWordWithContextBuffer<S> buffer(1);
  std::unique_ptr<ContiguousBufferBackedWordWithContext<S>> popped = 
    buffer.pop();
  EXPECT_EQ(popped, nullptr);
}

/**
 * Simple test of full buffer behavior.
 */
TEST(ContiguousWordWithContextBuffer, FullBuffer) {
  const int S = 8;
  const int targetIndex = 1;
  const int droppedCount = 2;
  ContiguousWordWithContextBuffer<S> buffer(1);

  int r1 = buffer.push(ContiguousBufferBackedWordWithContext<S>::builder()
    .withTargetWord(targetIndex)
    .withDroppedWordCount(droppedCount)
    .build());

  int r2 = buffer.push(ContiguousBufferBackedWordWithContext<S>::builder()
    .withTargetWord(targetIndex)
    .withDroppedWordCount(droppedCount)
    .build());

  EXPECT_EQ(r1, 1);
  EXPECT_EQ(r2, 0);
}

// TODO: more complicated usage... multiple push/pop's, order checking?
