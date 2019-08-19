#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "training/batching/minibatch.h"
#include "training/sgd/context.h"
#include "training/sgd/minibatch_matrix_manager.h"

#include <memory>
#include <vector>

using wombat::batching::Minibatch;
using wombat::neuralnet::Vector;
using wombat::sgd::MinibatchMatrixManager;

class ContextForTest : public wombat::sgd::Context {
  public:
    ~ContextForTest() {
    }

    virtual float getAlpha() {
      return 1;
    }

    // By always returning 2 * label we can verify this test class.
    virtual float loss(const float& f, int32_t label) {
      return 2 * label;
    }
};


/**
 * Get a minibatch build from a network that looks like this:
 *
 * input    output  labels
 *
 *           0 1 2
 * 0 1 1 1   0 1 2   0 3 6
 * 0 0 1 1   0 1 2   0 2 4
 *           0 1 2
 *
 */
static float i1[4] = { 0, 1, 1, 1 };
static float i2[4] = { 0, 0, 1, 1 };
static float o1[4] = { 0, 0, 0, 0 };
static float o2[4] = { 1, 1, 1, 1 };
static float o3[4] = { 2, 2, 2, 2 };
static float l1[3] = { 0, 3, 6 };
static float l2[3] = { 0, 2, 4 };
std::unique_ptr<Minibatch> getMinibatch() {
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

  return std::move(std::make_unique<Minibatch>(inputLayerVectors, labels, outputLayerVectors));
}

TEST(MinibatchMatrixManagerTest, Activate) {
  ContextForTest testContext;
  auto batch = getMinibatch();
  auto context = std::make_shared<ContextForTest>();
  MinibatchMatrixManager manager(4);
  float correctionMatrix[6] = {};
  manager.activate(batch, context, correctionMatrix);

  /*
   *             0 1 2
   *
   * 0 1 1 1     0 1 2      0 3 6
   *          x          =
   * 0 0 1 1     0 1 2      0 2 4
   *
   *             0 1 2
   */
  EXPECT_FLOAT_EQ(correctionMatrix[0], 0);
  EXPECT_FLOAT_EQ(correctionMatrix[1], 3);
  EXPECT_FLOAT_EQ(correctionMatrix[2], 6);
  EXPECT_FLOAT_EQ(correctionMatrix[3], 0);
  EXPECT_FLOAT_EQ(correctionMatrix[4], 2);
  EXPECT_FLOAT_EQ(correctionMatrix[5], 4);
}

TEST(MinibatchMatrixManagerTest, Correction) {
  ContextForTest testContext;
  auto batch = getMinibatch();
  auto context = std::make_shared<ContextForTest>();
  MinibatchMatrixManager manager(4);
  float correctionMatrix[6] = {};
  manager.activate(batch, context, correctionMatrix);
  manager.calculateError(batch, context, correctionMatrix);

  // Test loss function always multiplies by 2 in correction matrix.
  EXPECT_FLOAT_EQ(correctionMatrix[0],  0);
  EXPECT_FLOAT_EQ(correctionMatrix[1],  6);
  EXPECT_FLOAT_EQ(correctionMatrix[2], 12);
  EXPECT_FLOAT_EQ(correctionMatrix[3],  0);
  EXPECT_FLOAT_EQ(correctionMatrix[4],  4);
  EXPECT_FLOAT_EQ(correctionMatrix[5],  8);
}

TEST(MinibatchMatrixManagerTest, InputUpdate) {
  ContextForTest testContext;
  auto batch = getMinibatch();
  auto context = std::make_shared<ContextForTest>();
  MinibatchMatrixManager manager(4);
  float correctionMatrix[6] = {};
  float inputUpdate[8] = {};
  manager.activate(batch, context, correctionMatrix);
  manager.calculateError(batch, context, correctionMatrix);
  manager.calculateInputLayerUpdate(batch, context, correctionMatrix, inputUpdate);

  /*
   *             0 0 0 0
   *  0 6 12                    30 30 30 30
   *          x  1 1 1 1    =
   *  0 4 8                     20 20 20 20
   *             2 2 2 2
   */
  EXPECT_FLOAT_EQ(inputUpdate[0], 30);
  EXPECT_FLOAT_EQ(inputUpdate[1], 30);
  EXPECT_FLOAT_EQ(inputUpdate[2], 30);
  EXPECT_FLOAT_EQ(inputUpdate[3], 30);
  EXPECT_FLOAT_EQ(inputUpdate[4], 20);
  EXPECT_FLOAT_EQ(inputUpdate[5], 20);
  EXPECT_FLOAT_EQ(inputUpdate[6], 20);
  EXPECT_FLOAT_EQ(inputUpdate[7], 20);
}

TEST(MinibatchMatrixManagerTest, OutputUpdate) {
  ContextForTest testContext;
  auto batch = getMinibatch();
  auto context = std::make_shared<ContextForTest>();
  MinibatchMatrixManager manager(4);
  float correctionMatrix[6] = {};
  float outputUpdate[12] = {};
  manager.activate(batch, context, correctionMatrix);
  manager.calculateError(batch, context, correctionMatrix);
  manager.calculateOutputLayerUpdate(batch, context, correctionMatrix, outputUpdate);

  /*
   *  0  0      0 1 1 1       0  0  0  0
   *  6  4   x             =  0  6  10 10
   *  12 8      0 0 1 1       0  12 20 20
   */
  EXPECT_FLOAT_EQ(outputUpdate[0], 0);
  EXPECT_FLOAT_EQ(outputUpdate[1], 0);
  EXPECT_FLOAT_EQ(outputUpdate[2], 0);
  EXPECT_FLOAT_EQ(outputUpdate[3], 0);
  EXPECT_FLOAT_EQ(outputUpdate[4], 0);
  EXPECT_FLOAT_EQ(outputUpdate[5], 6);
  EXPECT_FLOAT_EQ(outputUpdate[6], 10);
  EXPECT_FLOAT_EQ(outputUpdate[7], 10);
  EXPECT_FLOAT_EQ(outputUpdate[8], 0);
  EXPECT_FLOAT_EQ(outputUpdate[9], 12);
  EXPECT_FLOAT_EQ(outputUpdate[10], 20);
  EXPECT_FLOAT_EQ(outputUpdate[11], 20);
}
