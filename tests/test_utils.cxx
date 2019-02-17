#include "../tests/test_utils.h"

#include "vocabulary/stream_backed.word_source.h" 
#include "vocabulary/wordbag_producer.h" 

using wombat::StreamBackedWordSource;
using wombat::WordBagProducer;

namespace testutils {
  std::shared_ptr<WordSource> getWordSource(const std::string& fileName) {
    auto inputStream = std::make_unique<std::ifstream>();
    inputStream->open(fileName, std::ios::out);
    if (!inputStream->is_open()) {
      throw std::invalid_argument("Unable to open test file.");
    }
    return std::make_unique<StreamBackedWordSource>(std::move(inputStream));
  }

  std::shared_ptr<WordBag> getWordBag() {
    auto wordSource = getWordSource(TEST_FILE_NAME);
    return WordBagProducer::fromWordSource(wordSource);
  }
}
