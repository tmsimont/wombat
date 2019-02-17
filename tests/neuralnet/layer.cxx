#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "neuralnet/layer.h"

using wombat::neuralnet::Layer;

TEST(NeuralNetLayer, Initialize) {
  Layer sixtyFourWords(64, 300);
  EXPECT_EQ(sixtyFourWords.vectorAt(0)[0], 0);
}

TEST(NeuralNetLayer, Randomize) {
  Layer sixtyFourWords(64, 300);
  sixtyFourWords.randomize();
  float sum = 0.0f;
  for (int i = 0; i < 20; ++i) {
    sum += sixtyFourWords.vectorAt(1)[i];
  }
  EXPECT_NE(sum, 0);
}
