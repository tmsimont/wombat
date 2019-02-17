#include <iostream>
#include <vector>
#include <string>

#include "arguments.h"

using namespace wombat;

void printUsage() {
  std::cout << "Use it right" << std::endl;
}

int main(int argc, char *argv[]) {
  std::vector<std::string> args(argv, argv + argc);
  if (args.size() < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  Arguments arguments = Arguments(args);

  // Load pre-trained vocab or learn from source.
  // Initialize vectors or load vectors.
  // Get the word source for training.
  // Update vectors with word source for configurable epochs.
  // Get the trained word vectors (possibly from GPU)
  // Save the trained word vectors.

  return 0;
}

