#include "vocabulary/wordbag_producer.h"

#include "vocabulary/wordbag/word2vec.wordbag.builder.h"

namespace wombat {
  std::unique_ptr<WordBag> WordBagProducer::fromWordSource(
      const std::shared_ptr<WordSource> wordSource) {
    // original word2vec approach is all I have so far :p
    Word2VecWordBagBuilder builder;
    while (wordSource->hasNext()) {
      builder.add(wordSource->nextWord());
    }
    return builder.build();
  }

  std::unique_ptr<WordBag> WordBagProducer::fromSavedBag(
      std::shared_ptr<std::istream> dataStream) {
    // TODO: implement this
    return nullptr;
  }
}
