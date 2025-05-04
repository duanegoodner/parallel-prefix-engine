// ----------------------------------------------------------------------------
// arg_parser.cpp
//
// Implements argument parsing logic using CLI11.
// ----------------------------------------------------------------------------

#include "common/arg_parser.hpp"

#include <CLI/CLI.hpp>
#include <utility>

#include "common/logger.hpp"

ProgramArgs ArgParser::Parse(int argc, char *const argv[]) {
  CLI::App app{"Distributed prefix sum runner"};

  int seed = 1234;
  std::string backend = "mpi";
  std::string cuda_kernel = "tiled";
  std::string log_level = "off";
  std::vector<int> full_matrix_dim = {4, 4};
  std::vector<int> tile_dim = {2, 2};

  app.add_option("-s, --seed", seed, "Random seed")->default_val("1234");
  app.add_option("-b, --backend", backend, "Backend to use (mpi or cuda)")
      ->check(CLI::IsMember({"mpi", "cuda"}))
      ->default_val("mpi");
  app.add_option("-k, --kernel", cuda_kernel, "CUDA kernel type")
      ->check(CLI::IsMember({"tiled", "single_element", "warp"}))
      ->default_val("tiled");

  app.add_option(
         "-L, --log-level",
         log_level,
         "Logging level (off, info, debug or error)"
  )
      ->check(CLI::IsMember({"off", "info", "debug", "error"}))
      ->default_val("off");

  app.add_option(
         "-f, --full-matrix-dim",
         full_matrix_dim,
         "Full matrix dimensions (rows cols)"
  )
      ->expected(2)
      ->default_val(std::vector<std::string>{"4", "4"});

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
      LogLevelUtils::FromString(log_level),
      full_matrix_dim,
      tile_dim,
      cuda_kernel,
      argc,
      const_cast<char **>(argv)
  );
}
