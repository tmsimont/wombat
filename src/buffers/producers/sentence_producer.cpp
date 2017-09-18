// Copyright 2017 Trevor Simonton

#include "src/buffers/producers/sentence_producer.h"

/**
 * Use the producer's data source to build a tokenized
 * sentence to the sentence buffer held by this
 * sentence buffer reader.
 */
int SentenceProducer::buildSentence(SenBufferReader *sen_reader) {
  while (1) {
    if (source->iterationsRemaining() == 0) {
      return 0;
    }
    while (1) {
      int word = source->getWord();
      if (word == -1) continue;
      if (word == 0) break;
      if (shouldDiscardWord(word)) {
        sen_reader->incDroppedWords();
        continue;
      }
      sen_reader->sen()[sen_reader->length()] = word;
      sen_reader->incLength();
      if (sen_reader->length() >= MAX_SENTENCE_LENGTH) break;
    }
    sen_reader->setPosition(0);

    // we might have hit the end of the word source
    if (source->rewind()) {
      int dropped_at_end = sen_reader->droppedWords() + sen_reader->length();

      // TODO: get away from w2v-functions externals
      #pragma omp atomic
      word_count_actual += dropped_at_end;

      sen_reader->setDroppedWords(0);
      sen_reader->setLength(0);
    } else {
      return 1;
    }

    return 1;
  }
}

bool SentenceProducer::shouldDiscardWord(int word) {
  // TODO: get away from w2v-functions externals
  // The subsampling randomly discards frequent
  // words while keeping the ranking same
  if (sample > 0) {
    real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1)
      * (sample * train_words) / vocab[word].cn;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    if (ran < (next_random & 0xFFFF) / (real)65536) {
      return true;
    }
  }
  return false;
}
