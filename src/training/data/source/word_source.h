#ifndef TRAINING_DATA_SOURCE_WORD_SOURCE_H_
#define TRAINING_DATA_SOURCE_WORD_SOURCE_H_

#include <string>

namespace wombat {

  /**
   * This generates instances of Word, that can be parsed to generate 
   * training data.
   */
  class WordSource {
    public:
      virtual ~WordSource() {}

      /**
       * Get a Word instance to use for training.
       */
      virtual std::string nextWord() = 0;

      /**
       * Get a Word instance to use for training.
       */
      virtual bool hasNext() = 0;

      /**
       * Restart from the beginning of the data source.
       */
      virtual bool rewind() = 0;
  };

}

#endif

