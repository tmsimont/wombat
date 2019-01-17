#ifndef ARGUMENTS_H_
#define ARGUMENTS_H_

#include <vector>
#include <string>

/**
 * Used for keeping track of user input.
 */
namespace wombat {
  class Arguments {
    public:
      Arguments(const std::vector<std::string>& args);
    private:
      void parse(const std::vector<std::string>& args);
      void printHelp();
  };
}

#endif