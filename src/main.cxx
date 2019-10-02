#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "arguments.h"

#include "vocabulary/word_source.h"
#include "vocabulary/stream_backed.word_source.h"
#include "vocabulary/wordbag_producer.h"

using namespace wombat;

void printUsage() {
  std::cout << "Use it right" << std::endl;
}

std::shared_ptr<WordSource> getWordSourceFromFile(const std::string& fileName) {
  auto inputStream = std::make_unique<std::ifstream>();
  inputStream->open(fileName, std::ios::out);
  if (!inputStream->is_open()) {
    throw std::invalid_argument("Unable to open test file.");
  }
  return std::make_shared<StreamBackedWordSource>(std::move(inputStream));
}

int main(int argc, char *argv[]) {
  std::vector<std::string> args(argv, argv + argc);
  if (args.size() < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  Arguments arguments = Arguments(args);

  // Load pre-trained vocab or learn from source. (learning from source now)
  // TODO: Implement switching between pre-trained and learn. For now using learn from file.
  auto vocabSource = getWordSourceFromFile(arguments.getVocabSourceFile());
  auto wordBag = WordBagProducer::fromWordSource(vocabSource);

  // Initialize vectors or load vectors.
  // Network(wordbag, vectorlength);

  // Get the word source for training (re-use vocabSource?)
  //  WordSamplingSentenceSource(
  //      const std::shared_ptr<WordBag> wordBag,
  //      const std::shared_ptr<WordSource> wordSource,
  //      const float& sample) 
 
  // With sentence source, split up sentences into multiple threads somehow.
  // Each sentence will be parsed with a SentenceParser into WordWithContext instances.
  // WordWithContexts are stashed in contiguous buffer? 
  // The contiguous buffer is consumed by minibatch trainer or gpu trainer

  // Update vectors with word source for configurable epochs.
  // SgdMinibatchTrainer? Strategy builder? (negative sampling, etc)

  // Get the trained word vectors (possibly from GPU)

  // Collect stats?

  // Save the trained word vectors.

  return 0;
}

