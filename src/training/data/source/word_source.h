#ifndef TRAINING_DATA_SOURCE_WORD_SOURCE_H_
#define TRAINING_DATA_SOURCE_WORD_SOURCE_H_

#include <string>

namespace wombat {

  /**
   * A word source provides words found in natural language training data.
   * Successive calls for "next word" provide word use in natural order.
   */
  class WordSource {
    public:
      virtual ~WordSource() {}

      /**
       * Get a word instance as string to use for training.
       */
      virtual std::string nextWord() = 0;

      /**
       * Returns true if this source has another word to use for training.
       */
      virtual bool hasNext() = 0;

      /**
       * Restart from the beginning of the data source.
       */
      virtual bool rewind() = 0;
  };

}

#endif

