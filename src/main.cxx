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
  return 0;
}
