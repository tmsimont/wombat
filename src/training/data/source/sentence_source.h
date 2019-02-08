#ifndef TRAINING_DATA_SOURCE_SENTENCE_SOURCE_H_
#define TRAINING_DATA_SOURCE_SENTENCE_SOURCE_H_

#include "training/data/structure/sentence.h"

#include <memory>

namespace wombat {

  /**
   * This generates instances of Sentence, that can be parsed to generate 
   * training data.
   */
  class SentenceSource {
    public:
      ~SentenceSource() {}

      /**
       * Get a Sentence instance to use for training.
       */
      virtual std::unique_ptr<Sentence> nextSentence() = 0;

      /**
       * Get a Sentence instance to use for training.
       */
      virtual bool hasNext() = 0;

      /**
       * Restart from the beginning of the data source.
       */
      virtual bool rewind() = 0;

    protected:
      /**
       * Determine if the word sample should be ignored.
       * TODO: decouple this from sentence source?
       */
      bool shouldDiscardWord(const int32_t& word);
  };

}

#endif
