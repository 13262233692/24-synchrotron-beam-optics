#include "photon_transport.h"
#include "physics_constants.h"
#include <cmath>
#include <algorithm>

namespace synchrotron {

PhotonTransport::PhotonTransport(uint64_t seed)
    : rng_(seed), uniform_dist_(0.0, 1.0) {}

double PhotonTransport::compute_crystal_absorption(double energy_eV,
                                                    double path_length_mm) const {
    double mu = si_absorption_coefficient(energy_eV);
    double path_cm = path_length_mm * 0.1;
    return 1.0 - std::exp(-mu * path_cm);
}

double PhotonTransport::sample_angular_deviation(double darwin_width_rad) {
    double u = uniform_dist_(rng_);
    return darwin_width_rad * (2.0 * u - 1.0);
}

bool PhotonTransport::reflect_from_crystal(double deviation, double darwin_width_rad) {
    double normalized = deviation / (darwin_width_rad / 2.0);
    if (std::abs(normalized) <= 1.0) return true;
    double x = std::abs(normalized) - std::sqrt(normalized * normalized - 1.0);
    double reflectivity = x * x;
    return uniform_dist_(rng_) < reflectivity;
}

PhotonTransportResult PhotonTransport::simulate_single_energy(
    double energy_eV,
    double bragg_angle_rad,
    double offset_mm,
    double crystal_thickness_mm,
    double darwin_width_rad,
    int num_photons) {

    PhotonTransportResult result{};
    result.total_path_length_mm = 0.0;
    result.crystal1_absorption = 0.0;
    result.crystal2_absorption = 0.0;
    result.crystal1_path_mm = 0.0;
    result.crystal2_path_mm = 0.0;
    result.reflectivity_crystal1 = 0.0;
    result.reflectivity_crystal2 = 0.0;

    double sin_theta = std::sin(bragg_angle_rad);
    double cos_theta = std::cos(bragg_angle_rad);
    double path_in_crystal = crystal_thickness_mm / sin_theta;

    int reflected1 = 0;
    int reflected2 = 0;

    for (int i = 0; i < num_photons; ++i) {
        double weight = 1.0;
        double total_path = 0.0;

        double dev1 = sample_angular_deviation(darwin_width_rad);
        if (!reflect_from_crystal(dev1, darwin_width_rad)) {
            result.absorbed_weight += weight;
            continue;
        }
        reflected1++;

        double abs_frac1 = compute_crystal_absorption(energy_eV, path_in_crystal);
        weight *= (1.0 - abs_frac1);
        result.crystal1_absorption += abs_frac1;
        total_path += path_in_crystal;

        double dev2 = sample_angular_deviation(darwin_width_rad);
        if (!reflect_from_crystal(dev2, darwin_width_rad)) {
            result.absorbed_weight += weight;
            continue;
        }
        reflected2++;

        double abs_frac2 = compute_crystal_absorption(energy_eV, path_in_crystal);
        weight *= (1.0 - abs_frac2);
        result.crystal2_absorption += abs_frac2;
        total_path += path_in_crystal;

        double free_path = offset_mm / cos_theta;
        total_path += free_path;

        result.transmitted_weight += weight;
        result.total_path_length_mm += total_path;
        result.crystal1_path_mm += path_in_crystal;
        result.crystal2_path_mm += path_in_crystal;
    }

    if (num_photons > 0) {
        result.reflectivity_crystal1 = static_cast<double>(reflected1) / num_photons;
        result.reflectivity_crystal2 = static_cast<double>(reflected2) / num_photons;
        int transmitted_count = reflected1 > 0 ? reflected2 : 0;
        result.transmission_fraction = static_cast<double>(transmitted_count) / num_photons;

        if (result.transmitted_weight > 0) {
            result.total_path_length_mm /= transmitted_count;
            result.crystal1_path_mm /= transmitted_count;
            result.crystal2_path_mm /= transmitted_count;
        }
        result.absorbed_weight /= num_photons;
        result.transmitted_weight /= num_photons;
    }

    return result;
}

std::vector<SpectrumPoint> PhotonTransport::simulate_energy_scan(
    double energy_start_eV,
    double energy_end_eV,
    int num_energy_points,
    const std::vector<double>& bragg_angles_rad,
    double offset_mm,
    double crystal_thickness_mm,
    const std::vector<double>& darwin_widths_rad,
    int num_photons_per_point) {

    std::vector<SpectrumPoint> spectrum;
    spectrum.reserve(num_energy_points);
    double step = (energy_end_eV - energy_start_eV) / (num_energy_points - 1);

    for (int i = 0; i < num_energy_points; ++i) {
        double e = energy_start_eV + i * step;
        if (i >= static_cast<int>(bragg_angles_rad.size())) break;
        if (i >= static_cast<int>(darwin_widths_rad.size())) break;

        auto res = simulate_single_energy(e, bragg_angles_rad[i],
                                          offset_mm, crystal_thickness_mm,
                                          darwin_widths_rad[i],
                                          num_photons_per_point);

        SpectrumPoint pt;
        pt.energy_eV = e;
        pt.transmitted_intensity = res.transmitted_weight;
        pt.absorption_fraction = res.absorbed_weight;
        pt.path_length_mm = res.total_path_length_mm;
        pt.crystal1_pitch_deg = bragg_angles_rad[i] * RAD_TO_DEG;
        pt.crystal2_pitch_deg = bragg_angles_rad[i] * RAD_TO_DEG;

        double sin_theta = std::sin(bragg_angles_rad[i]);
        double cos_theta = std::cos(bragg_angles_rad[i]);
        pt.crystal2_x_mm = (offset_mm / 2.0) / sin_theta;
        pt.crystal2_y_mm = offset_mm;

        spectrum.push_back(pt);
    }

    return spectrum;
}

}
