#include "training/words_to_minibatched_sgd_trainer.h"

namespace wombat {
  void WordsToMinibatchedSgdTrainer::multiEpochTrainingOnWordSource(
      int32_t numEpochs,
      std::shared_ptr<Context> context) {
    bool training = true;
    while(training) {
      if(_sentenceSource->hasNext()) {
        _trainingStrategy->train(
            _network,
            _minibatchProvider->provideMinibatchIndices(_sentenceSource->nextSentence()),
            context);
      } else {
        // TODO: check epochs and possibly call _sentenceSource->rewind()
        training = false;
      }
    }
  }
}
