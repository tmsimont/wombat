#include "vocabulary/wordbag/word2vec.wordbag.builder.h"
#include "vocabulary/wordbag/wordbag.factory.h"

namespace wombat {
  std::unique_ptr<WordBag> WordBagFactory::makeWordBag() {
    // original word2vec approach is all I have so far :p
    Word2VecWordBagBuilder builder;
    while (_wordSource->hasNext()) {
      builder.add(_wordSource->nextWord());
    }
    return builder.build();
  }
}
