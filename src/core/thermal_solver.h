#pragma once

#include <vector>
#include <cstddef>
#include <string>
#include "fp_control.h"

namespace synchrotron {

struct ThermalMaterialProps {
    double k_w_per_mk;
    double alpha_per_k;
    double cp_j_per_kgk;
    double rho_kg_per_m3;
    double youngs_modulus_pa;
    double poisson_ratio;
};

struct ThermalConfig {
    double crystal_width_mm;
    double crystal_height_mm;
    double crystal_thickness_mm;
    int grid_nx;
    int grid_ny;
    double beam_sigma_mm;
    double beam_power_w;
    double heat_sink_temp_k;
    double ambient_temp_k;
    double dt_sec;
    int max_iterations;
    double convergence_tolerance_k;
    double dcm_offset_mm;
};

struct ThermalDeformation {
    std::vector<double> surface_displacement_um;
    std::vector<double> surface_slope_rad;
    double max_displacement_um;
    double max_slope_rad;
    double mean_slope_rad;
};

struct PZTCompensation {
    double pitch_bias_rad;
    double pitch_bias_deg;
    double pitch_bias_urad;
    double estimated_beam_height_error_um;
    double compensated_height_error_um;
    bool active;
};

struct ThermalState {
    std::vector<double> temperature_k;
    ThermalDeformation deformation;
    PZTCompensation compensation;
    double max_temp_k;
    double min_temp_k;
    double avg_temp_k;
    double delta_t_max_k;
    bool converged;
    int iterations;
};

inline ThermalMaterialProps si_thermal_props_300k() {
    return {156.0, 2.62e-6, 702.0, 2329.0, 1.3011e11, 0.279};
}

inline ThermalConfig default_thermal_config() {
    return {50.0, 50.0, 1.0, 128, 128, 0.5, 500.0, 298.0, 298.0, 0.0, 20000, 1e-3, 15.0};
}

class ThermalFDMSolver {
public:
    ThermalFDMSolver(const ThermalConfig& config = default_thermal_config(),
                     const ThermalMaterialProps& props = si_thermal_props_300k());

    void set_beam_position(double x_center_mm, double y_center_mm);
    void set_beam_power(double power_w);
    void set_bragg_angle(double bragg_angle_rad);

    ThermalState solve_steady_state();

    ThermalDeformation compute_deformation(const std::vector<double>& temperature_k) const;
    PZTCompensation compute_pzt_compensation(const ThermalDeformation& def,
                                             double offset_mm,
                                             double bragg_angle_rad) const;

    const ThermalConfig& config() const { return cfg_; }
    const ThermalMaterialProps& props() const { return props_; }

    int nx() const { return cfg_.grid_nx; }
    int ny() const { return cfg_.grid_ny; }
    double dx() const { return dx_; }
    double dy() const { return dy_; }

    std::vector<double> build_heat_source() const;
    double temp_at(int i, int j) const { return T_[j * nx_ + i]; }
    double x_coord(int i) const { return (i - cx_idx_) * dx_; }
    double y_coord(int j) const { return (j - cy_idx_) * dy_; }

private:
    ThermalConfig cfg_;
    ThermalMaterialProps props_;
    std::vector<double> T_;
    std::vector<double> T_prev_;
    std::vector<double> Q_;
    double dx_;
    double dy_;
    double beam_x_mm_;
    double beam_y_mm_;
    double bragg_angle_rad_;
    int cx_idx_;
    int cy_idx_;

    void initialize_grid();
    void apply_boundary_conditions();
    double step_explicit(double dt);
    double thermal_diffusivity() const;
    double max_stable_dt() const;
};

}
