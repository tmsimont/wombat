#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "../tests/test_utils.h"

#include "neuralnet/network.h"

using wombat::neuralnet::Network;

TEST(NeuralNetNetwork, Initialize) {
  Network net(testutils::getWordBag(), 100);
}

