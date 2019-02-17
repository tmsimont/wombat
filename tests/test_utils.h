#ifndef TESTS_TEST_UTILS_H_
#define TESTS_TEST_UTILS_H_

#include "vocabulary/wordbag/wordbag.h"
#include "vocabulary/word_source.h"

#include <fstream>

using wombat::WordBag;
using wombat::WordSource;

namespace testutils {
  const std::string TEST_FILE_NAME("tests/resources/wordbag.txt");
  std::shared_ptr<WordSource> getWordSource(const std::string& fileName);
  std::shared_ptr<WordBag> getWordBag();
}

#endif
