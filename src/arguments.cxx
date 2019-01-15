#include <iostream>
#include "arguments.h"

namespace wombat {
  Arguments::Arguments(const std::vector<std::string>& args) {
    parse(args);
  }

  void Arguments::parse(const std::vector<std::string>& args) {
    std::cout << "parsing" << std::endl;
    std::cout << args[1] << std::endl;
  }

  void Arguments::printHelp() {
    std::cout << "help is on the way" << std::endl;
  }
}
