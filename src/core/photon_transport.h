#pragma once

#include <vector>
#include <random>
#include <cstddef>
#include <atomic>
#include "fp_control.h"
#include "thermal_solver.h"

namespace synchrotron {

struct PhotonState {
    double x_mm;
    double y_mm;
    double z_mm;
    double direction_x;
    double direction_y;
    double direction_z;
    double energy_eV;
    double weight;
    double path_length_mm;
    bool alive;
};

struct PhotonTransportResult {
    double transmitted_weight;
    double absorbed_weight;
    double total_path_length_mm;
    double crystal1_absorption;
    double crystal2_absorption;
    double crystal1_path_mm;
    double crystal2_path_mm;
    double reflectivity_crystal1;
    double reflectivity_crystal2;
    double transmission_fraction;
    double pzt_pitch_bias_rad;
    double pzt_pitch_bias_urad;
    double beam_height_error_um;
    double compensated_height_error_um;
    double max_temp_k;
    double delta_t_max_k;
    double max_displacement_um;
    double max_slope_rad;
    bool thermal_converged;
    int thermal_iterations;
};

struct ThermalSpectrumPoint {
    double energy_eV;
    double max_temp_k;
    double delta_t_max_k;
    double max_displacement_um;
    double max_slope_rad;
    double mean_slope_rad;
    double pzt_pitch_bias_rad;
    double pzt_pitch_bias_urad;
    double beam_height_error_um;
    double compensated_height_error_um;
    bool thermal_converged;
    int thermal_iterations;
};

struct SpectrumPoint {
    double energy_eV;
    double transmitted_intensity;
    double absorption_fraction;
    double path_length_mm;
    double crystal1_pitch_deg;
    double crystal2_pitch_deg;
    double crystal2_pitch_bias_deg;
    double crystal2_x_mm;
    double crystal2_y_mm;
    ThermalSpectrumPoint thermal;
};

class PhotonTransport {
public:
    PhotonTransport(uint64_t seed = 42);

    PhotonTransportResult simulate_single_energy(
        double energy_eV,
        double bragg_angle_rad,
        double offset_mm,
        double crystal_thickness_mm,
        double darwin_width_rad,
        int num_photons);

    std::vector<SpectrumPoint> simulate_energy_scan(
        double energy_start_eV,
        double energy_end_eV,
        int num_energy_points,
        const std::vector<double>& bragg_angles_rad,
        double offset_mm,
        double crystal_thickness_mm,
        const std::vector<double>& darwin_widths_rad,
        int num_photons_per_point);

    void enable_thermal(bool enable) { thermal_enabled_ = enable; }
    bool thermal_enabled() const { return thermal_enabled_; }
    void set_thermal_config(const ThermalConfig& cfg) { thermal_cfg_ = cfg; }
    const ThermalConfig& thermal_config() const { return thermal_cfg_; }
    ThermalFDMSolver& thermal_solver() { return thermal_solver_; }
    const ThermalFDMSolver& thermal_solver() const { return thermal_solver_; }

private:
    uint64_t base_seed_;
    bool thermal_enabled_;
    ThermalConfig thermal_cfg_;
    ThermalFDMSolver thermal_solver_;

    double compute_crystal_absorption(double energy_eV, double path_length_mm) const;
    double sample_angular_deviation(std::mt19937_64& rng, double darwin_width_rad);
    bool reflect_from_crystal(std::mt19937_64& rng, double deviation, double darwin_width_rad);
    double flush_weight(double weight) const;
    ThermalState run_thermal_solve(double energy_eV, double bragg_angle_rad, double offset_mm);
};

}
