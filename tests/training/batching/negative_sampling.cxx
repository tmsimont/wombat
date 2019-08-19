#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "../tests/test_utils.h"
#include "training/batching/negative_sampling.h"

using wombat::neuralnet::Network;
using wombat::neuralnet::Vector;
using wombat::batching::Minibatch;
using wombat::batching::NegativeSamplingStrategy;


TEST(NegativeSamplingTest, Simple) {
  Network net(testutils::getWordBag(), 100);
  NegativeSamplingStrategy strategy(net);
  auto batch = strategy.getMinibatch();
}
