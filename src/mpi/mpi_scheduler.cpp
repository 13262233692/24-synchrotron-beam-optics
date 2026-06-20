#include "mpi_scheduler.h"
#include "hdf5_writer.h"
#include "fp_control.h"
#include <mpi.h>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <thread>

namespace synchrotron {

MPIScheduler::MPIScheduler() : owns_mpi_(false), rank_(0), size_(1) {
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag) {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);
        if (!flag) {
            MPI_Initialized(&flag);
            if (flag) owns_mpi_ = true;
        }
    }

    MPI_Initialized(&flag);
    if (flag) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }
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

void MPIScheduler::send_heartbeat(int tag) {
    char pulse = 1;
    MPI_Send(&pulse, 1, MPI_BYTE, 0, tag, MPI_COMM_WORLD);
}

bool MPIScheduler::check_heartbeat(int source, double timeout_sec, int tag) {
    MPI_Status status;
    int flag = 0;

    auto deadline = std::chrono::steady_clock::now() +
        std::chrono::duration<double>(timeout_sec);

    while (std::chrono::steady_clock::now() < deadline) {
        MPI_Iprobe(source, tag, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            char buf;
            MPI_Recv(&buf, 1, MPI_BYTE, source, tag, MPI_COMM_WORLD, &status);
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return false;
}

SimulationResult MPIScheduler::run_simulation(const SimulationConfig& config) {
    ScopedFTZ ftz_guard;

    SimulationResult result;
    double timeout = config.heartbeat_timeout_sec > 0
        ? config.heartbeat_timeout_sec : 300.0;

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

    bool local_error = false;
    for (int i = local_start; i < local_start + local_count; ++i) {
        double e = config.energy_start_eV + i * step;
        if (i >= total_points) break;

        try {
            auto res = transport.simulate_single_energy(
                e, bragg_angles[i], config.offset_mm,
                config.crystal_thickness_mm, darwin_widths[i],
                config.num_photons_per_point);

            SpectrumPoint pt;
            pt.energy_eV = e;
            pt.transmitted_intensity = flush_subnormal(res.transmitted_weight);
            pt.absorption_fraction = flush_subnormal(res.absorbed_weight);
            pt.path_length_mm = res.total_path_length_mm;
            pt.crystal1_pitch_deg = bragg_angles[i] * RAD_TO_DEG;
            pt.crystal2_pitch_deg = bragg_angles[i] * RAD_TO_DEG;

            double sin_theta = std::sin(bragg_angles[i]);
            pt.crystal2_x_mm = (config.offset_mm / 2.0) / sin_theta;
            pt.crystal2_y_mm = config.offset_mm;

            local_spectrum.push_back(pt);
        } catch (...) {
            local_error = true;
            SpectrumPoint pt{};
            pt.energy_eV = e;
            pt.transmitted_intensity = 0.0;
            pt.absorption_fraction = 1.0;
            local_spectrum.push_back(pt);
        }
    }

    int error_flag = local_error ? 1 : 0;
    int global_error = 0;
    MPI_Allreduce(&error_flag, &global_error, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    result.spectrum = gather_spectrum(local_spectrum, timeout);
    return result;
}

std::vector<SpectrumPoint> MPIScheduler::gather_spectrum(
    const std::vector<SpectrumPoint>& local_spectrum,
    double heartbeat_timeout_sec) {

    int local_size = static_cast<int>(local_spectrum.size());

    int num_fields = 8;
    std::vector<double> local_flat(local_size * num_fields);
    for (int i = 0; i < local_size; ++i) {
        local_flat[i * num_fields + 0] = local_spectrum[i].energy_eV;
        local_flat[i * num_fields + 1] = flush_subnormal(local_spectrum[i].transmitted_intensity);
        local_flat[i * num_fields + 2] = flush_subnormal(local_spectrum[i].absorption_fraction);
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
            pt.transmitted_intensity = flush_subnormal(global_flat[i * num_fields + 1]);
            pt.absorption_fraction = flush_subnormal(global_flat[i * num_fields + 2]);
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
            pt.transmitted_intensity = flush_subnormal(global_flat[i * num_fields + 1]);
            pt.absorption_fraction = flush_subnormal(global_flat[i * num_fields + 2]);
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
