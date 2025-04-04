#pragma once

#include <sstream>
#include <string>
#include <vector>


struct ArgvBuilder {
    std::vector<std::string> args_str;
    std::vector<char *> argv;
  
    ArgvBuilder(const std::string &cmdline) {
      std::istringstream iss(cmdline);
      std::string token;
  
      argv.push_back(const_cast<char *>("test_program"));
  
      while (iss >> token) {
        args_str.push_back(token);
      }
  
      // Store .data() after args_str is fully built to avoid invalidation
      for (auto &arg : args_str) {
        argv.push_back(const_cast<char *>(arg.data())); // safe + linter-friendly
      }
    }
  
    int argc() const { return static_cast<int>(argv.size()); }
    char **argv_data() { return argv.data(); }
  };