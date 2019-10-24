#include "training/batching/minibatch_provider.h"

namespace wombat {
namespace batching {
  std::vector<std::unique_ptr<MinibatchIndices>> MinibatchProvider::provideMinibatchIndices(
      std::unique_ptr<Sentence> sentence) {
    std::vector<std::unique_ptr<MinibatchIndices>> minibatchIndices;

    // Parsers are created per-sentence.
    // TODO: make parser reusable
    SentenceParser parser(*sentence.get(), _windowSize);

    // Use the parser to pull out WordWithContext instances.
    auto wordWithContext = parser.nextWordWithContext();

    // For each WordWithContext, build a minibatch.
    while (wordWithContext != nullptr) {
      minibatchIndices.push_back(std::move(_minibatchingStrategy->getMinibatch(*wordWithContext.get())));
      wordWithContext = parser.nextWordWithContext();
    }

    return minibatchIndices;
  }
}
}
