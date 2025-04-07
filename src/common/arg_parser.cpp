// ----------------------------------------------------------------------------
// arg_parser.cpp
//
// Implements argument parsing logic using CLI11.
// ----------------------------------------------------------------------------

#include "common/arg_parser.hpp"
#include <CLI/CLI.hpp>
#include <utility>

ProgramArgs ArgParser::Parse(int argc, char *const argv[]) {
  CLI::App app{"Distributed prefix sum runner"};

  int seed = 1234;
  std::string backend = "mpi";
  bool verbose = false;
  std::vector<int> full_matrix_dim = {4, 4};
  std::vector<int> grid_dim = {2, 2};  // still parsed, might be useful later
  std::vector<int> tile_dim = {2, 2};

  app.add_option("-s, --seed", seed, "Random seed")->default_val("1234");
  app.add_option("-b, --backend", backend, "Backend to use (mpi or cuda)")
      ->check(CLI::IsMember({"mpi", "cuda"}))
      ->default_val("mpi");
  app.add_flag("-v,--verbose", verbose, "Enable verbose output");

  app.add_option(
         "-f, --full-matrix-dim",
         full_matrix_dim,
         "Full matrix dimensions (rows cols)"
  )
      ->expected(2)
      ->default_val(std::vector<std::string>{"4", "4"});

  app.add_option("-g, --grid-size", grid_dim, "Grid dimensions (rows cols)")
      ->expected(2)
      ->default_val(std::vector<std::string>{"2", "2"});

  app.add_option("-t, --tile-dim", tile_dim, "Tile dimensions (rows cols)")
      ->expected(2)
      ->default_val(std::vector<std::string>{"2", "2"});

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(app.exit(e));
  }

  return ProgramArgs(
      seed,
      backend,
      verbose,
      full_matrix_dim,
      tile_dim,
      argc,
      const_cast<char **>(argv)
  );
}
