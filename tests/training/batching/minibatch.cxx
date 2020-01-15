#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "training/batching/minibatch.h"
#include "training/sgd/context.h"
#include "training/sgd/minibatch_matrix_manager.h"
#include "../tests/test_utils.h"

#include <memory>
#include <vector>

using wombat::batching::Minibatch;
using wombat::neuralnet::Vector;
using wombat::neuralnet::Network;
using wombat::sgd::MinibatchMatrixManager;

class MockNetwork : public Network {
  MOCK_METHOD(Vector, getInputVector, (int64_t index), (const, override));
}

Minibatch getMinibatch() {
  Vector iv1(0, i1, 4);
  Vector iv2(1, i2, 4);
  Vector ov1(0, o1, 4);
  Vector ov2(1, o2, 4);
  Vector ov3(2, o3, 4);

  std::vector<Vector> inputLayerVectors;
  std::vector<int32_t> labels;
  std::vector<Vector> outputLayerVectors;

  inputLayerVectors.push_back(iv1);
  inputLayerVectors.push_back(iv2);

  labels.push_back(l1[0]);
  labels.push_back(l1[1]);
  labels.push_back(l1[2]);
  labels.push_back(l2[0]);
  labels.push_back(l2[1]);
  labels.push_back(l2[2]);

  outputLayerVectors.push_back(ov1);
  outputLayerVectors.push_back(ov2);
  outputLayerVectors.push_back(ov3);

  return Minibatch(
      Network(testutils::getWordBag(), 100),
      inputLayerVectors,
      labels,
      outputLayerVectors);
}

TEST(MinibatchTest, NonTest) {
}
