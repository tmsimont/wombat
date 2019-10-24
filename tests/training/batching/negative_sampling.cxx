#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "../tests/test_utils.h"
#include "training/batching/negative_sampling.h"
#include "neuralnet/network.h"
#include "neuralnet/vector.h"
#include "training/data/structure/word_with_context.visitor.h"

using wombat::neuralnet::Network;
using wombat::neuralnet::Vector;
using wombat::batching::NegativeSamplingStrategy;
using wombat::WordWithContext;
using wombat::WordWithContextVisitor;


// TODO: figure out GMOCK
class MockWordWithContext : public WordWithContext {
  public:
    virtual ~MockWordWithContext() {}
    virtual int32_t getTargetWord() const {
      return 1;
    }
    virtual int32_t getNumberOfContextWords() const {
      return 1;
    }
    virtual void acceptContextWordVisitor(WordWithContextVisitor& visitor) const {
      visitor.visitContextWord(1);
    }
};

TEST(NegativeSamplingTest, Simple) {
  NegativeSamplingStrategy strategy;
  auto batch = strategy.getMinibatch(MockWordWithContext());
}
