// ----------------------------------------------------------------------------
// arg_parser.hpp
//
// Parses command-line arguments and returns a ProgramArgs instance.
// ----------------------------------------------------------------------------

#pragma once


#include "common/program_args.hpp"

class ArgParser {
public:
  static ProgramArgs Parse(int argc, char *const argv[]);
};
