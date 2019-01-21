#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "contiguous_buffer/int32_ring_buffer.hpp"

using wombat::Int32RingBuffer;

TEST(Int32RingBufferTest, ValidHappyCase) {
  static const int S = 2;
  Int32RingBuffer<S> buffer(2);
  std::unique_ptr<std::array<int, S>> data =
    std::make_unique<std::array<int, S>>(std::array<int, S>());
  (*data)[0] = 8;
  (*data)[1] = 9;
  buffer.push(std::move(data));
  std::unique_ptr<std::array<int, S>> item = buffer.pop();

  // Original data should be destroyed.
  EXPECT_EQ(data, nullptr);
  
  // Return value should be non-null.
  EXPECT_NE(item, nullptr);

  // We should see the original input in the returned item.
  EXPECT_EQ((*item)[0], 8);
  EXPECT_EQ((*item)[1], 9);
}

TEST(Int32RingBufferTest, EmptyBuffer) {
  static const int size = 2;
  Int32RingBuffer<size> buffer(2);
  std::unique_ptr<std::array<int, size>> item = buffer.pop();
  EXPECT_EQ(item, nullptr);
}
