#pragma once

#include <vector>
#include <string>
#include <chrono>
#include "photon_transport.h"
#include "bragg_solver.h"
#include "crystal_geometry.h"

namespace synchrotron {

struct SimulationConfig {
    double energy_start_eV;
    double energy_end_eV;
    int num_energy_points;
    int num_photons_per_point;
    double offset_mm;
    double crystal_thickness_mm;
    int miller_h;
    int miller_k;
    int miller_l;
    double lattice_constant_A;
    double heartbeat_timeout_sec;
};

struct SimulationResult {
    std::vector<SpectrumPoint> spectrum;
    std::vector<BraggSolution> bragg_solutions;
    std::vector<DCMConfiguration> dcm_configs;
};

class MPIScheduler {
public:
    MPIScheduler();
    ~MPIScheduler();

    SimulationResult run_simulation(const SimulationConfig& config);

    int rank() const { return rank_; }
    int size() const { return size_; }

    static bool is_initialized();

private:
    int rank_;
    int size_;
    bool owns_mpi_;

    std::vector<SpectrumPoint> gather_spectrum(
        const std::vector<SpectrumPoint>& local_spectrum,
        double heartbeat_timeout_sec);

    void send_heartbeat(int tag = 9999);
    bool check_heartbeat(int source, double timeout_sec, int tag = 9999);
};

}
