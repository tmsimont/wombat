#ifndef TRAINING_DATA_SOURCE_SENTENCE_SOURCE_H_
#define TRAINING_DATA_SOURCE_SENTENCE_SOURCE_H_

#include "training/data/structure/sentence.h"

#include <memory>

namespace wombat {

  /**
   * This generates instances of Sentence that are found in training data.
   */
  class SentenceSource {
    public:
      virtual ~SentenceSource() {}

      /**
       * Get a Sentence instance to parse for training.
       */
      virtual std::unique_ptr<Sentence> nextSentence() = 0;

      /**
       * Returns true if the file has more sentences to parse.
       */
      virtual bool hasNext() = 0;

      /**
       * Restart from the beginning of the data source.
       */
      virtual bool rewind() = 0;
  };

}

#endif
