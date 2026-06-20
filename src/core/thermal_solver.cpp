#include "thermal_solver.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace synchrotron {

ThermalFDMSolver::ThermalFDMSolver(const ThermalConfig& config,
                                   const ThermalMaterialProps& props)
    : cfg_(config), props_(props),
      beam_x_mm_(0.0), beam_y_mm_(0.0), bragg_angle_rad_(0.0) {
    initialize_grid();
}

void ThermalFDMSolver::initialize_grid() {
    int n = cfg_.grid_nx * cfg_.grid_ny;
    T_.assign(n, cfg_.ambient_temp_k);
    T_prev_.assign(n, cfg_.ambient_temp_k);
    dx_ = cfg_.crystal_width_mm / static_cast<double>(cfg_.grid_nx - 1);
    dy_ = cfg_.crystal_height_mm / static_cast<double>(cfg_.grid_ny - 1);
    cx_idx_ = cfg_.grid_nx / 2;
    cy_idx_ = cfg_.grid_ny / 2;
    Q_ = build_heat_source();
}

void ThermalFDMSolver::set_beam_position(double x_mm, double y_mm) {
    beam_x_mm_ = x_mm;
    beam_y_mm_ = y_mm;
    Q_ = build_heat_source();
}

void ThermalFDMSolver::set_beam_power(double power_w) {
    cfg_.beam_power_w = power_w;
    Q_ = build_heat_source();
}

void ThermalFDMSolver::set_bragg_angle(double bragg_rad) {
    bragg_angle_rad_ = bragg_rad;
}

double ThermalFDMSolver::thermal_diffusivity() const {
    double alpha_m2_per_s = props_.k_w_per_mk / (props_.rho_kg_per_m3 * props_.cp_j_per_kgk);
    return alpha_m2_per_s * 1e6;
}

double ThermalFDMSolver::max_stable_dt() const {
    double alpha = thermal_diffusivity();
    double dx2 = dx_ * dx_;
    double dy2 = dy_ * dy_;
    if (alpha <= 0.0 || dx2 <= 0.0 || dy2 <= 0.0) return 1e-3;
    double r = 1.0 / dx2 + 1.0 / dy2;
    if (r <= 0.0) return 1e-3;
    double dt_max = 0.4 / (alpha * r);
    return std::max(dt_max, 1e-8);
}

std::vector<double> ThermalFDMSolver::build_heat_source() const {
    int n = cfg_.grid_nx * cfg_.grid_ny;
    std::vector<double> Q(n, 0.0);
    double sigma = cfg_.beam_sigma_mm;
    if (sigma <= 0.0 || cfg_.beam_power_w <= 0.0) return Q;

    double two_sigma2 = 2.0 * sigma * sigma;
    double total_integral = 0.0;
    double dA_mm2 = dx_ * dy_;

    for (int j = 0; j < cfg_.grid_ny; ++j) {
        double y = (j - cy_idx_) * dy_ - beam_y_mm_;
        for (int i = 0; i < cfg_.grid_nx; ++i) {
            double x = (i - cx_idx_) * dx_ - beam_x_mm_;
            double r2 = x * x + y * y;
            double val = std::exp(-r2 / two_sigma2);
            Q[j * cfg_.grid_nx + i] = val;
            total_integral += val * dA_mm2;
        }
    }

    double thickness_mm = cfg_.crystal_thickness_mm;
    double rho_kg_mm3 = props_.rho_kg_per_m3 * 1e-9;
    double cp_j_kgk = props_.cp_j_per_kgk;
    double denom = rho_kg_mm3 * cp_j_kgk * thickness_mm;

    if (total_integral > 0.0 && denom > 0.0) {
        for (auto& q : Q) {
            double power_density_w_per_mm2 = cfg_.beam_power_w * q / total_integral;
            q = power_density_w_per_mm2 / denom;
            q = flush_subnormal(q);
        }
    }
    return Q;
}

void ThermalFDMSolver::apply_boundary_conditions() {
    double T_sink = cfg_.heat_sink_temp_k;
    for (int i = 0; i < cfg_.grid_nx; ++i) {
        T_[i] = T_sink;
        T_[(cfg_.grid_ny - 1) * cfg_.grid_nx + i] = T_sink;
    }
    for (int j = 0; j < cfg_.grid_ny; ++j) {
        T_[j * cfg_.grid_nx] = T_sink;
        T_[j * cfg_.grid_nx + cfg_.grid_nx - 1] = T_sink;
    }
}

double ThermalFDMSolver::step_explicit(double dt) {
    ScopedFTZ ftz;
    double alpha = thermal_diffusivity();
    double dx2 = dx_ * dx_;
    double dy2 = dy_ * dy_;

    T_.swap(T_prev_);

    double max_delta = 0.0;
    for (int j = 1; j < cfg_.grid_ny - 1; ++j) {
        for (int i = 1; i < cfg_.grid_nx - 1; ++i) {
            int idx = j * cfg_.grid_nx + i;
            int ip1 = idx + 1;
            int im1 = idx - 1;
            int jp1 = (j + 1) * cfg_.grid_nx + i;
            int jm1 = (j - 1) * cfg_.grid_nx + i;

            double lap_x = (T_prev_[ip1] - 2.0 * T_prev_[idx] + T_prev_[im1]) / dx2;
            double lap_y = (T_prev_[jp1] - 2.0 * T_prev_[idx] + T_prev_[jm1]) / dy2;
            double update = dt * (alpha * (lap_x + lap_y) + Q_[idx]);
            update = flush_subnormal(update);

            double new_T = T_prev_[idx] + update;
            double delta = std::fabs(update);
            if (delta > max_delta) max_delta = delta;
            T_[idx] = new_T;
        }
    }
    apply_boundary_conditions();
    return max_delta;
}

ThermalState ThermalFDMSolver::solve_steady_state() {
    ScopedFTZ ftz;
    Q_ = build_heat_source();
    std::fill(T_.begin(), T_.end(), cfg_.ambient_temp_k);
    std::fill(T_prev_.begin(), T_prev_.end(), cfg_.ambient_temp_k);
    apply_boundary_conditions();

    double tol = cfg_.convergence_tolerance_k;
    int max_iter = std::max(cfg_.max_iterations, 50000);
    double dt_opt = (cfg_.dt_sec > 0.0) ? std::min(max_stable_dt(), cfg_.dt_sec) : max_stable_dt();
    int iter = 0;
    double delta = 1e9;
    int stationary_count = 0;
    double prev_max_T = cfg_.ambient_temp_k;

    while (iter < max_iter && delta > tol) {
        delta = step_explicit(dt_opt);
        ++iter;

        double cur_max = 0.0;
        for (double t : T_) cur_max = std::max(cur_max, t);
        if (iter > 100 && std::fabs(cur_max - prev_max_T) < tol * 0.1) {
            ++stationary_count;
            if (stationary_count > 500) break;
        } else {
            stationary_count = 0;
        }
        prev_max_T = cur_max;
    }

    ThermalState state;
    state.temperature_k = T_;
    state.converged = (delta <= tol) || (stationary_count > 500);
    state.iterations = iter;

    double tmax = T_[0], tmin = T_[0], tsum = 0.0;
    for (auto t : T_) {
        tmax = std::max(tmax, t);
        tmin = std::min(tmin, t);
        tsum += t;
    }
    state.max_temp_k = tmax;
    state.min_temp_k = tmin;
    state.avg_temp_k = tsum / static_cast<double>(T_.size());
    state.delta_t_max_k = tmax - cfg_.heat_sink_temp_k;

    state.deformation = compute_deformation(T_);
    state.compensation = compute_pzt_compensation(
        state.deformation, cfg_.dcm_offset_mm, bragg_angle_rad_);
    return state;
}

ThermalDeformation ThermalFDMSolver::compute_deformation(
    const std::vector<double>& temperature_k) const {
    ScopedFTZ ftz;
    int n = static_cast<int>(temperature_k.size());
    (void)n;
    int surface_row = cfg_.grid_ny - 2;

    ThermalDeformation def;
    def.surface_displacement_um.resize(cfg_.grid_nx, 0.0);
    def.surface_slope_rad.resize(cfg_.grid_nx, 0.0);
    def.max_displacement_um = 0.0;
    def.max_slope_rad = 0.0;
    def.mean_slope_rad = 0.0;

    double alpha = props_.alpha_per_k;
    double thickness_m = cfg_.crystal_thickness_mm * 1e-3;

    double max_disp = 0.0;
    double max_slope = 0.0;
    double sum_slope = 0.0;

    for (int i = 0; i < cfg_.grid_nx; ++i) {
        int idx_surface = surface_row * cfg_.grid_nx + i;
        int idx_bottom = 1 * cfg_.grid_nx + i;
        double T_surf = temperature_k[idx_surface];
        double T_bot = temperature_k[idx_bottom];
        double dT = T_surf - T_bot;

        double curvature = alpha * dT / thickness_m;
        double x_m = (i - cx_idx_) * dx_ * 1e-3;
        double disp_m = 0.5 * curvature * x_m * x_m;

        double dT_avg = 0.0;
        for (int j = 1; j < cfg_.grid_ny - 1; ++j) {
            dT_avg += (temperature_k[j * cfg_.grid_nx + i] - cfg_.ambient_temp_k);
        }
        dT_avg /= static_cast<double>(cfg_.grid_ny - 2);
        disp_m += alpha * dT_avg * cfg_.crystal_thickness_mm * 1e-3;

        double disp_um = disp_m * 1e6;
        def.surface_displacement_um[i] = disp_um;
        if (std::fabs(disp_um) > max_disp) max_disp = std::fabs(disp_um);
    }

    for (int i = 1; i < cfg_.grid_nx - 1; ++i) {
        double ddisp = def.surface_displacement_um[i + 1] - def.surface_displacement_um[i - 1];
        double ddx = 2.0 * dx_ * 1e3;
        double slope = ddisp / ddx;
        def.surface_slope_rad[i] = slope;
        if (std::fabs(slope) > max_slope) max_slope = std::fabs(slope);
        sum_slope += slope;
    }

    if (cfg_.grid_nx > 2) {
        def.mean_slope_rad = sum_slope / static_cast<double>(cfg_.grid_nx - 2);
    }
    def.max_displacement_um = max_disp;
    def.max_slope_rad = max_slope;
    return def;
}

PZTCompensation ThermalFDMSolver::compute_pzt_compensation(
    const ThermalDeformation& def,
    double offset_mm,
    double bragg_angle_rad) const {
    ScopedFTZ ftz;
    PZTCompensation pzt;
    pzt.active = true;

    double beam_center_slope = 0.0;
    if (cx_idx_ >= 1 && cx_idx_ < static_cast<int>(def.surface_slope_rad.size()) - 1) {
        beam_center_slope = def.surface_slope_rad[cx_idx_];
    }

    double c1_slope_rad = beam_center_slope;
    double cos_theta = std::cos(bragg_angle_rad);
    if (std::fabs(cos_theta) < 1e-9) cos_theta = 1e-9;
    double c2_pitch_bias_rad = -c1_slope_rad / cos_theta;

    pzt.pitch_bias_rad = c2_pitch_bias_rad;
    pzt.pitch_bias_deg = c2_pitch_bias_rad * 180.0 / M_PI;
    pzt.pitch_bias_urad = c2_pitch_bias_rad * 1e6;

    double L_mm = offset_mm;
    double uncomp_height_error_um = L_mm * 1e3 * std::tan(2.0 * c1_slope_rad);
    double comp_error_um = L_mm * 1e3 * std::tan(2.0 * c1_slope_rad + 2.0 * c2_pitch_bias_rad);

    pzt.estimated_beam_height_error_um = uncomp_height_error_um;
    pzt.compensated_height_error_um = flush_subnormal(std::fabs(comp_error_um));
    return pzt;
}

}
