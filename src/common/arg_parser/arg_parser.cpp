// ----------------------------------------------------------------------------
// arg_parser.cpp
//
// Implements argument parsing logic using CLI11.
// ----------------------------------------------------------------------------

#include "common/arg_parser.hpp"

#include <CLI/CLI.hpp>
#include <optional>
#include <utility>

#include "common/logger.hpp"

ProgramArgs ArgParser::Parse(int argc, char *const argv[]) {
  CLI::App app{"Distributed prefix sum runner"};

  int seed = 1234;
  std::string backend = "mpi";
  std::string log_level = "warning";
  std::vector<int> full_matrix_dim = {4, 4};
  std::vector<int> tile_dim = {2, 2};

  // Optional CUDA-specific args
  std::optional<std::vector<int>> sub_tile_dim;
  std::optional<std::string> cuda_kernel;

  // Registered CLI options
  app.add_option("-r, --seed", seed, "Random seed")->default_val("1234");
  app.add_option("-b, --backend", backend, "Backend to use (mpi or cuda)")
      ->check(CLI::IsMember({"mpi", "cuda"}))
      ->default_val("mpi");

  app.add_option(
      "-L, --log-level",
      log_level,
      "Logging level"
  )
      ->check(CLI::IsMember({"off", "info", "warning", "error"}))
      ->default_val("warning");

  app.add_option(
      "-f, --full-matrix-dim",
      full_matrix_dim,
      "Full matrix dimensions (rows cols)"
  )
      ->expected(2)
      ->default_val(std::vector<std::string>{"4", "4"});

  app.add_option(
      "-t, --tile-dim",
      tile_dim,
      "Tile dimensions (rows cols)"
  )
      ->expected(2)
      ->default_val(std::vector<std::string>{"4", "4"});

  // CUDA-only options (no defaults here)
  auto cuda_kernel_option = app.add_option(
      "-k, --kernel",
      cuda_kernel,
      "CUDA kernel type (single_tile, multi_tile)"
  )->check(CLI::IsMember({"single_tile", "multi_tile"}));

  auto subtile_option = app.add_option(
      "-s, --sub-tile-dim",
      sub_tile_dim,
      "Sub-tile dimensions (rows cols, CUDA only)"
  )->expected(2);

  // Parse CLI args
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(app.exit(e));
  }

  // === Post-parse logic: handle CUDA-only args ===
  if (backend != "cuda") {
    if (cuda_kernel_option->count() > 0) {
      std::cerr << "[Warning] Ignoring --kernel because backend is not 'cuda'.\n";
    }
    if (subtile_option->count() > 0) {
      std::cerr << "[Warning] Ignoring --sub-tile-dim because backend is not 'cuda'.\n";
    }

    cuda_kernel = std::nullopt;
    sub_tile_dim = std::nullopt;
  } else {
    if (!cuda_kernel_option->count()) {
      cuda_kernel = "single_tile";  // Default kernel if not set
    }

    if (!subtile_option->count()) {
      sub_tile_dim = std::vector<int>{2, 2};  // Default subtile size
    }
  }

  return ProgramArgs(
      seed,
      backend,
      LogLevelUtils::FromString(log_level),
      full_matrix_dim,
      tile_dim,
      sub_tile_dim,
      cuda_kernel,
      argc,
      const_cast<char **>(argv)
  );
}
