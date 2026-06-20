#include "bragg_solver.h"
#include "physics_constants.h"
#include <stdexcept>
#include <algorithm>

namespace synchrotron {

BraggSolver::BraggSolver(int h, int k, int l, double lattice_const_A)
    : h_(h), k_(k), l_(l), lattice_const_(lattice_const_A) {
    d_spacing_ = compute_d_spacing();
}

double BraggSolver::compute_d_spacing() const {
    double sum_sq = static_cast<double>(h_ * h_ + k_ * k_ + l_ * l_);
    if (sum_sq == 0.0) {
        throw std::invalid_argument("Miller indices cannot all be zero");
    }
    return lattice_const_ / std::sqrt(sum_sq);
}

double BraggSolver::bragg_angle(double energy_eV, int order) const {
    double lambda_A = energy_to_wavelength_A(energy_eV);
    double sin_theta = (order * lambda_A) / (2.0 * d_spacing_);
    if (std::abs(sin_theta) > 1.0) {
        throw std::runtime_error("Energy below accessible range for given reflection and order");
    }
    return std::asin(sin_theta);
}

BraggSolution BraggSolver::solve(double energy_eV, int order) const {
    BraggSolution sol;
    sol.energy_eV = energy_eV;
    sol.wavelength_A = energy_to_wavelength_A(energy_eV);
    sol.d_spacing_A = d_spacing_;
    sol.miller_h = h_;
    sol.miller_k = k_;
    sol.miller_l = l_;
    sol.order = order;

    double sin_theta = (order * sol.wavelength_A) / (2.0 * d_spacing_);
    if (std::abs(sin_theta) > 1.0) {
        throw std::runtime_error("Energy " + std::to_string(energy_eV) +
                                 " eV is below the accessible range for Si(" +
                                 std::to_string(h_) + std::to_string(k_) + std::to_string(l_) +
                                 ") reflection at order " + std::to_string(order));
    }
    sol.bragg_angle_rad = std::asin(sin_theta);
    sol.bragg_angle_deg = sol.bragg_angle_rad * RAD_TO_DEG;
    return sol;
}

std::vector<BraggSolution> BraggSolver::solve_energy_scan(double energy_start_eV,
                                                           double energy_end_eV,
                                                           int num_points,
                                                           int order) const {
    std::vector<BraggSolution> results;
    results.reserve(num_points);
    double step = (energy_end_eV - energy_start_eV) / (num_points - 1);
    for (int i = 0; i < num_points; ++i) {
        double e = energy_start_eV + i * step;
        try {
            results.push_back(solve(e, order));
        } catch (const std::runtime_error&) {
            break;
        }
    }
    return results;
}

double BraggSolver::darwin_width_rad(double energy_eV) const {
    double theta = bragg_angle(energy_eV);
    double F_h = compute_structure_factor_si();
    double r_e = 2.8179403227e-5;
    double lambda_A = energy_to_wavelength_A(energy_eV);
    double V_cell = std::pow(lattice_const_, 3);
    double dw = (2.0 * r_e * lambda_A * lambda_A * F_h) /
                (M_PI * V_cell * std::sin(2.0 * theta));
    return dw;
}

double BraggSolver::reflectivity(double energy_eV, double angle_offset_rad) const {
    double theta_B = bragg_angle(energy_eV);
    double dw = darwin_width_rad(energy_eV);
    double delta = angle_offset_rad / (dw / 2.0);
    if (std::abs(delta) <= 1.0) {
        return 1.0;
    }
    double x = (std::abs(delta) - std::sqrt(delta * delta - 1.0));
    return x * x;
}

double BraggSolver::compute_structure_factor_si() const {
    double f_si = 14.0;
    double multiplier = 4.0;
    bool all_odd = (h_ % 2 != 0) && (k_ % 2 != 0) && (l_ % 2 != 0);
    bool all_even = (h_ % 2 == 0) && (k_ % 2 == 0) && (l_ % 2 == 0);
    bool sum_even = (h_ + k_ + l_) % 2 == 0;

    if (all_odd) {
        return multiplier * f_si * std::sqrt(2.0);
    } else if (all_even && sum_even) {
        return multiplier * f_si * 2.0;
    } else {
        return 0.0;
    }
}

}
