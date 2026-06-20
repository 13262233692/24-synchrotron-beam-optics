#include "mpi_scheduler.h"
#include "hdf5_writer.h"
#include <mpi.h>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace synchrotron {

MPIScheduler::MPIScheduler() : owns_mpi_(false) {
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);

    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag) {
        MPI_Init(nullptr, nullptr);
        owns_mpi_ = true;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
}

MPIScheduler::~MPIScheduler() {
    if (owns_mpi_) {
        MPI_Finalize();
    }
}

bool MPIScheduler::is_initialized() {
    int flag = 0;
    MPI_Initialized(&flag);
    return flag != 0;
}

SimulationResult MPIScheduler::run_simulation(const SimulationConfig& config) {
    SimulationResult result;

    BraggSolver solver(config.miller_h, config.miller_k, config.miller_l,
                       config.lattice_constant_A);

    result.bragg_solutions = solver.solve_energy_scan(
        config.energy_start_eV, config.energy_end_eV, config.num_energy_points);

    CrystalGeometry geometry(config.offset_mm, config.crystal_thickness_mm);
    result.dcm_configs = geometry.compute_dcm_scan(
        config.energy_start_eV, config.energy_end_eV,
        config.num_energy_points, solver);

    std::vector<double> bragg_angles, darwin_widths;
    bragg_angles.reserve(result.bragg_solutions.size());
    darwin_widths.reserve(result.bragg_solutions.size());
    for (const auto& sol : result.bragg_solutions) {
        bragg_angles.push_back(sol.bragg_angle_rad);
        darwin_widths.push_back(solver.darwin_width_rad(sol.energy_eV));
    }

    int total_points = static_cast<int>(result.bragg_solutions.size());
    int points_per_rank = total_points / size_;
    int remainder = total_points % size_;
    int local_start = rank_ * points_per_rank + std::min(rank_, remainder);
    int local_count = points_per_rank + (rank_ < remainder ? 1 : 0);

    PhotonTransport transport(static_cast<uint64_t>(42 + rank_));

    double step = (config.energy_end_eV - config.energy_start_eV) /
                  (config.num_energy_points - 1);

    std::vector<SpectrumPoint> local_spectrum;
    local_spectrum.reserve(local_count);

    for (int i = local_start; i < local_start + local_count; ++i) {
        double e = config.energy_start_eV + i * step;
        if (i >= total_points) break;

        auto res = transport.simulate_single_energy(
            e, bragg_angles[i], config.offset_mm,
            config.crystal_thickness_mm, darwin_widths[i],
            config.num_photons_per_point);

        SpectrumPoint pt;
        pt.energy_eV = e;
        pt.transmitted_intensity = res.transmitted_weight;
        pt.absorption_fraction = res.absorbed_weight;
        pt.path_length_mm = res.total_path_length_mm;
        pt.crystal1_pitch_deg = bragg_angles[i] * RAD_TO_DEG;
        pt.crystal2_pitch_deg = bragg_angles[i] * RAD_TO_DEG;

        double sin_theta = std::sin(bragg_angles[i]);
        pt.crystal2_x_mm = (config.offset_mm / 2.0) / sin_theta;
        pt.crystal2_y_mm = config.offset_mm;

        local_spectrum.push_back(pt);
    }

    result.spectrum = gather_spectrum(local_spectrum);
    return result;
}

std::vector<SpectrumPoint> MPIScheduler::gather_spectrum(
    const std::vector<SpectrumPoint>& local_spectrum) {

    int local_size = static_cast<int>(local_spectrum.size());

    int num_fields = 8;
    std::vector<double> local_flat(local_size * num_fields);
    for (int i = 0; i < local_size; ++i) {
        local_flat[i * num_fields + 0] = local_spectrum[i].energy_eV;
        local_flat[i * num_fields + 1] = local_spectrum[i].transmitted_intensity;
        local_flat[i * num_fields + 2] = local_spectrum[i].absorption_fraction;
        local_flat[i * num_fields + 3] = local_spectrum[i].path_length_mm;
        local_flat[i * num_fields + 4] = local_spectrum[i].crystal1_pitch_deg;
        local_flat[i * num_fields + 5] = local_spectrum[i].crystal2_pitch_deg;
        local_flat[i * num_fields + 6] = local_spectrum[i].crystal2_x_mm;
        local_flat[i * num_fields + 7] = local_spectrum[i].crystal2_y_mm;
    }

    std::vector<int> recv_counts(size_);
    std::vector<int> displs(size_);

    MPI_Gather(&local_size, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    int total_size = 0;
    if (rank_ == 0) {
        displs[0] = 0;
        for (int r = 0; r < size_; ++r) {
            recv_counts[r] *= num_fields;
            if (r > 0) displs[r] = displs[r - 1] + recv_counts[r - 1];
            total_size += recv_counts[r];
        }
    }

    std::vector<double> global_flat;
    if (rank_ == 0) global_flat.resize(total_size);

    MPI_Gatherv(local_flat.data(), local_size * num_fields, MPI_DOUBLE,
                global_flat.data(), recv_counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<SpectrumPoint> global_spectrum;
    if (rank_ == 0) {
        int num_points = total_size / num_fields;
        global_spectrum.reserve(num_points);
        for (int i = 0; i < num_points; ++i) {
            SpectrumPoint pt;
            pt.energy_eV = global_flat[i * num_fields + 0];
            pt.transmitted_intensity = global_flat[i * num_fields + 1];
            pt.absorption_fraction = global_flat[i * num_fields + 2];
            pt.path_length_mm = global_flat[i * num_fields + 3];
            pt.crystal1_pitch_deg = global_flat[i * num_fields + 4];
            pt.crystal2_pitch_deg = global_flat[i * num_fields + 5];
            pt.crystal2_x_mm = global_flat[i * num_fields + 6];
            pt.crystal2_y_mm = global_flat[i * num_fields + 7];
            global_spectrum.push_back(pt);
        }
    }

    MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank_ != 0) {
        global_flat.resize(total_size);
    }

    MPI_Bcast(global_flat.data(), total_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank_ != 0) {
        int num_points = total_size / num_fields;
        global_spectrum.reserve(num_points);
        for (int i = 0; i < num_points; ++i) {
            SpectrumPoint pt;
            pt.energy_eV = global_flat[i * num_fields + 0];
            pt.transmitted_intensity = global_flat[i * num_fields + 1];
            pt.absorption_fraction = global_flat[i * num_fields + 2];
            pt.path_length_mm = global_flat[i * num_fields + 3];
            pt.crystal1_pitch_deg = global_flat[i * num_fields + 4];
            pt.crystal2_pitch_deg = global_flat[i * num_fields + 5];
            pt.crystal2_x_mm = global_flat[i * num_fields + 6];
            pt.crystal2_y_mm = global_flat[i * num_fields + 7];
            global_spectrum.push_back(pt);
        }
    }

    return global_spectrum;
}

}
