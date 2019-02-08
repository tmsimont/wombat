#ifndef TRAINING_DATA_SOURCE_FILE_BACKED_SENTENCE_SOURCE_H_
#define TRAINING_DATA_SOURCE_FILE_BACKED_SENTENCE_SOURCE_H_

#include "training/data/source/sentence_source.h"
#include "vocabulary/wordbag.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <stdexcept>

namespace wombat {

  class FileBackedSentenceSource : public SentenceSource {
    public:
      /**
       * Construct a sentence source that can create sentences made from the words
       * in the given word bag.
       *
       * @param WordBag that built from the vocabulary of test data.
       */
      FileBackedSentenceSource(const std::shared_ptr<WordBag> wordBag)
        : _wordBag(wordBag) {
            if (_wordBag->getSize() == 1) {
              throw std::invalid_argument("Cannot use an empty bag");
            }
      }

      /**
       * Implement virtual destructor.
       */
      ~FileBackedSentenceSource() {}

      /**
       * Pull the next sentence from the file. hasNext() should be checked first.
       */
      std::unique_ptr<Sentence> nextSentence();

      /**
       * Returns true if the file has more sentences to parse.
       */
      bool hasNext();

      /**
       * Resets the internal file pointer to the start of the training data.
       */
      bool rewind();

      /**
       * Set the file by name to parse for Sentences.
       * TODO: put filename into constructor
       */
      void setFile(const std::string& fileName);

    private:
      static const int MAX_STRING = 64;
      const std::shared_ptr<WordBag> _wordBag;
      // TODO: use istream? pass in unique_ptr to stream so its more generic
      std::ifstream _fileStream;
      char ch;

      /**
       * Read a single word from the file into a char array.
       */
      void ReadWord(char *word);
  };

}

#endif
