#include <iostream>
#include "common/solver_dispatch.hpp"
#include "mpi_prefix_sum/mpi_prefix_sum_solver.hpp"
#include "mpi_prefix_sum/mpi_solver_registration.hpp"

struct MpiSolverRegistrar {
    MpiSolverRegistrar() {
        std::cout << "[mpi] Registering MPI backend...\n";
        PrefixSumSolverFactory::RegisterBackend("mpi",
            [](const ProgramArgs& args) {
                return std::make_unique<MpiPrefixSumSolver>(args);
            });
    }
};

void ForceMpiSolverRegistration() {
    static MpiSolverRegistrar instance;
}
