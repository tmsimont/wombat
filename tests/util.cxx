#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "util.h"

TEST(UtilTest, Random) {
  uint64_t random0 = wombat::util::random();
  uint64_t random1 = wombat::util::random();
  uint64_t random2 = wombat::util::random();
  for (int i = 0; i < 100; ++i) {
    EXPECT_NE(wombat::util::random(), random0);
    EXPECT_NE(wombat::util::random(), random1);
    EXPECT_NE(wombat::util::random(), random2);
  }
}

