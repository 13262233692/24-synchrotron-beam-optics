#include "photon_transport.h"
#include "physics_constants.h"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace synchrotron {

PhotonTransport::PhotonTransport(uint64_t seed)
    : base_seed_(seed) {}

double PhotonTransport::flush_weight(double weight) const {
    return synchrotron::flush_subnormal(weight);
}

double PhotonTransport::compute_crystal_absorption(double energy_eV,
                                                    double path_length_mm) const {
    double mu = si_absorption_coefficient(energy_eV);
    double path_cm = path_length_mm * 0.1;
    double exponent = -mu * path_cm;
    double transmission = safe_exp(exponent);
    double absorption = 1.0 - transmission;
    return flush_subnormal(absorption);
}

double PhotonTransport::sample_angular_deviation(std::mt19937_64& rng,
                                                  double darwin_width_rad) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(rng);
    return darwin_width_rad * (2.0 * u - 1.0);
}

bool PhotonTransport::reflect_from_crystal(std::mt19937_64& rng,
                                            double deviation,
                                            double darwin_width_rad) {
    double half_dw = darwin_width_rad / 2.0;
    if (half_dw == 0.0) return false;

    double normalized = deviation / half_dw;
    if (std::abs(normalized) <= 1.0) return true;

    double sq = normalized * normalized - 1.0;
    if (sq < 0.0) return false;

    double x = std::abs(normalized) - std::sqrt(sq);
    double reflectivity = flush_subnormal(x * x);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng) < reflectivity;
}

PhotonTransportResult PhotonTransport::simulate_single_energy(
    double energy_eV,
    double bragg_angle_rad,
    double offset_mm,
    double crystal_thickness_mm,
    double darwin_width_rad,
    int num_photons) {

    ScopedFTZ ftz_guard;

    double sin_theta = std::sin(bragg_angle_rad);
    double cos_theta = std::cos(bragg_angle_rad);
    double path_in_crystal = crystal_thickness_mm / sin_theta;
    double free_path = offset_mm / cos_theta;

    double mu = si_absorption_coefficient(energy_eV);
    double path_cm = path_in_crystal * 0.1;
    double abs_fraction = flush_subnormal(1.0 - safe_exp(-mu * path_cm));
    double surv_fraction = flush_subnormal(1.0 - abs_fraction);

    int n_reflected1 = 0;
    int n_reflected2 = 0;
    double sum_transmitted_weight = 0.0;
    double sum_absorbed_weight = 0.0;
    double sum_c1_abs = 0.0;
    double sum_c2_abs = 0.0;
    double sum_path = 0.0;

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
#else
    int n_threads = 1;
#endif

    std::vector<int> thread_ref1(n_threads, 0);
    std::vector<int> thread_ref2(n_threads, 0);
    std::vector<double> thread_tw(n_threads, 0.0);
    std::vector<double> thread_aw(n_threads, 0.0);
    std::vector<double> thread_c1a(n_threads, 0.0);
    std::vector<double> thread_c2a(n_threads, 0.0);
    std::vector<double> thread_path(n_threads, 0.0);

    bool global_fault = false;

#pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif

        ScopedFTZ thread_ftz;

        std::mt19937_64 thread_rng(base_seed_ + tid * 7919ULL + 1ULL);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        int local_ref1 = 0;
        int local_ref2 = 0;
        double local_tw = 0.0;
        double local_aw = 0.0;
        double local_c1a = 0.0;
        double local_c2a = 0.0;
        double local_path = 0.0;

        bool thread_fault = false;

#pragma omp for schedule(static) nowait
        for (int i = 0; i < num_photons; ++i) {
            if (thread_fault) continue;

            double weight = 1.0;
            double photon_path = 0.0;

            double dev1 = darwin_width_rad * (2.0 * dist(thread_rng) - 1.0);
            double half_dw = darwin_width_rad / 2.0;

            bool ref1 = false;
            if (half_dw != 0.0) {
                double norm1 = dev1 / half_dw;
                if (std::abs(norm1) <= 1.0) {
                    ref1 = true;
                } else {
                    double sq1 = norm1 * norm1 - 1.0;
                    if (sq1 >= 0.0) {
                        double x1 = std::abs(norm1) - std::sqrt(sq1);
                        double r1 = flush_subnormal(x1 * x1);
                        ref1 = dist(thread_rng) < r1;
                    }
                }
            }

            if (!ref1) {
                local_aw += weight;
                continue;
            }
            local_ref1++;

            weight = flush_weight(weight * surv_fraction);
            local_c1a += abs_fraction;
            photon_path += path_in_crystal;

            double dev2 = darwin_width_rad * (2.0 * dist(thread_rng) - 1.0);

            bool ref2 = false;
            if (half_dw != 0.0) {
                double norm2 = dev2 / half_dw;
                if (std::abs(norm2) <= 1.0) {
                    ref2 = true;
                } else {
                    double sq2 = norm2 * norm2 - 1.0;
                    if (sq2 >= 0.0) {
                        double x2 = std::abs(norm2) - std::sqrt(sq2);
                        double r2 = flush_subnormal(x2 * x2);
                        ref2 = dist(thread_rng) < r2;
                    }
                }
            }

            if (!ref2) {
                local_aw += weight;
                continue;
            }
            local_ref2++;

            weight = flush_weight(weight * surv_fraction);
            local_c2a += abs_fraction;
            photon_path += path_in_crystal + free_path;

            local_tw += weight;
            local_path += photon_path;
        }

#pragma omp critical(reduce_results)
        {
            thread_ref1[tid] = local_ref1;
            thread_ref2[tid] = local_ref2;
            thread_tw[tid] = local_tw;
            thread_aw[tid] = local_aw;
            thread_c1a[tid] = local_c1a;
            thread_c2a[tid] = local_c2a;
            thread_path[tid] = local_path;
        }

#pragma omp barrier

#pragma omp single
        {
            for (int t = 0; t < n_threads; ++t) {
                n_reflected1 += thread_ref1[t];
                n_reflected2 += thread_ref2[t];
                sum_transmitted_weight += thread_tw[t];
                sum_absorbed_weight += thread_aw[t];
                sum_c1_abs += thread_c1a[t];
                sum_c2_abs += thread_c2a[t];
                sum_path += thread_path[t];
            }
        }
    }

    PhotonTransportResult result{};
    if (num_photons > 0) {
        result.reflectivity_crystal1 = static_cast<double>(n_reflected1) / num_photons;
        result.reflectivity_crystal2 = (n_reflected1 > 0)
            ? static_cast<double>(n_reflected2) / num_photons : 0.0;
        result.transmission_fraction = static_cast<double>(n_reflected2) / num_photons;
        result.absorbed_weight = sum_absorbed_weight / num_photons;
        result.transmitted_weight = sum_transmitted_weight / num_photons;
        if (n_reflected2 > 0) {
            result.total_path_length_mm = sum_path / n_reflected2;
            result.crystal1_path_mm = path_in_crystal;
            result.crystal2_path_mm = path_in_crystal;
        }
        result.crystal1_absorption = sum_c1_abs / num_photons;
        result.crystal2_absorption = sum_c2_abs / num_photons;
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
    double step = (num_energy_points > 1)
        ? (energy_end_eV - energy_start_eV) / (num_energy_points - 1)
        : 0.0;

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
        pt.crystal2_x_mm = (offset_mm / 2.0) / sin_theta;
        pt.crystal2_y_mm = offset_mm;

        spectrum.push_back(pt);
    }

    return spectrum;
}

}
