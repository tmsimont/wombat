#include "neuralnet/network.h"

namespace wombat {
namespace neuralnet {
  Network::Network(std::shared_ptr<WordBag> wordbag, int32_t vectorLength) : 
  _inputLayer(wordbag->getSize(), vectorLength),
  _outputLayer(wordbag->getSize(), vectorLength) {
    _inputLayer.randomize();
  }
}
}
