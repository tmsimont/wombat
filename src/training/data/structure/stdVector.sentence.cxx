#include "training/data/structure/stdVector.sentence.h"
#include "training/data/structure/sentence.visitor.h"

#include <vector>

namespace wombat {
  /**
   * Delegate to backing vector size method.
   */
  int32_t StdVectorSentence::getNumberOfTrainingWords() {
    return _sentence.size();
  }

  /**
   * Total number of input words is those sampled plus those discarded.
   */
  int32_t StdVectorSentence::getNumberOfWordsInput() {
    return _sentence.size() + _discarded;
  }

  /**
   * Adds a word to vector.
   */
  void StdVectorSentence::addWord(int wordIndex) {
    _sentence.push_back(wordIndex);
  }

  void StdVectorSentence::countDiscardedWord() {
    _discarded++;
  }

  /**
   * Accepts the visitor and iterates internal vector.
   */
  void StdVectorSentence::acceptWordVisitor(SentenceVisitor& visitor) const {
    for (auto wordIndex : _sentence) {
      visitor.visitWord(wordIndex);
    }
  }
}
