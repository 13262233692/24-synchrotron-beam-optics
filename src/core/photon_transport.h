#pragma once

#include <vector>
#include <random>
#include <cstddef>

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
};

struct SpectrumPoint {
    double energy_eV;
    double transmitted_intensity;
    double absorption_fraction;
    double path_length_mm;
    double crystal1_pitch_deg;
    double crystal2_pitch_deg;
    double crystal2_x_mm;
    double crystal2_y_mm;
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

private:
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_dist_;

    double compute_crystal_absorption(double energy_eV, double path_length_mm) const;
    double sample_angular_deviation(double darwin_width_rad);
    bool reflect_from_crystal(double deviation, double darwin_width_rad);
};

}
