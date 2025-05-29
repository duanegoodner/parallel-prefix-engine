#include "common/solver_dispatch.hpp"
#include "zscaffold_prefix_sum/zscaffold_prefix_sum_solver.hpp"
#include "zscaffold_prefix_sum/zscaffold_solver_registration.hpp"

struct ZScaffoldSolverRegistrar {
    ZScaffoldSolverRegistrar() {
        PrefixSumSolverFactory::RegisterBackend("zscaffold",
            [](const ProgramArgs& args) {
                return std::make_unique<ZScaffoldPrefixSumSolver>(args);
            });
    }
};

void ForceZScaffoldSolverRegistration() {
    static ZScaffoldSolverRegistrar instance;
}